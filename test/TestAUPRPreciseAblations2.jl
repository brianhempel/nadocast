import Dates
import Random
import Printf
import Serialization
import Statistics

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import Metrics
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations2")
import HREFPredictionAblations2

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
# import PlotMap
# import StormEvents
import ForecastCombinators

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

GRID = Conus.href_cropped_5km_grid();

function unzip3(triples)
  ( map(t -> t[1], triples)
  , map(t -> t[2], triples)
  , map(t -> t[3], triples)
  )
end

const nbootstraps = parse(Int64, get(ENV, "NBOOTSTRAPS", "0")) # 10,000 takes ~10hr

println("$nbootstraps bootstraps")

function two_sided_bootstrap_p_value_paired(bootstraps_1, bootstraps_2)
  nbootstraps = length(bootstraps_1)
  Float32(min(
    count(bootstraps_1 .<= bootstraps_2) / (nbootstraps / 2),
    count(bootstraps_1 .>= bootstraps_2) / (nbootstraps / 2),
    1.0
  ))
end

function do_it(forecasts, model_names; suffix = "")

  model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1")

  event_names = model_name_to_event_name.(model_names)

  println("************ ablations$(suffix) ************")

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  # We don't have storm events past this time.
  cutoff = Dates.DateTime(2022, 6, 1, 12)

  println("$(length(test_forecasts)) unfiltered test forecasts")
  test_forecasts_0z  = filter(forecast -> forecast.run_hour == 0  && Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
  test_forecasts_12z = filter(forecast -> forecast.run_hour == 12 && Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
  test_forecasts = []
  println("$(length(test_forecasts_0z)) 0z test forecasts before the event data cutoff date") #
  println("$(length(test_forecasts_12z)) 12z test forecasts before the event data cutoff date") #

  event_name_to_day_labeler = Dict(map(unique(event_names)) do event_name
    event_name => TrainingShared.event_name_to_day_labeler[event_name]
  end)

  # rm("day_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z"; recursive = true)
  # rm("day_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z"; recursive = true)

  X_0z, Ys_0z, weights_0z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_0z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablations2_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times_0z = Serialization.deserialize("day_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z/run_times.serialized")
  @assert length(unique(run_times_0z)) == length(test_forecasts_0z)
  @assert sort(unique(run_times_0z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_0z))

  X_12z, Ys_12z, weights_12z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_12z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablations2_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times_12z = Serialization.deserialize("day_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z/run_times.serialized")
  @assert length(unique(run_times_12z)) == length(test_forecasts_12z)
  @assert sort(unique(run_times_12z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_12z))

  @assert run_times_0z == (run_times_12z .- Dates.Hour(12))

  println(sort(unique(run_times_0z)))

  nforecasts = length(unique(run_times_0z))

  @assert size(X_0z) == size(X_12z)
  @assert size(y_0z) == size(y_12z)
  @assert weights_0z == weights_12z

  @assert length(y_0z) == size(X_0z,1)
  @assert length(y_0z) / nforecasts == round(length(y_0z) / nforecasts)
  ndata = length(y_0z)
  ndata_per_forecast = ndata ÷ nforecasts
  @assert run_times_0z[ndata_per_forecast*10] == run_times_0z[ndata_per_forecast*10 - 1]
  @assert run_times_0z[ndata_per_forecast*10] != run_times_0z[ndata_per_forecast*10 + 1]

  # Use same bootstraps across all predictors
  rng = Random.MersenneTwister(12345)
  bootstrap_forecast_iss = map(_ -> rand(rng, 1:nforecasts, nforecasts), 1:nbootstraps)

  nmodels = size(X_0z,2)
  model_au_pr_bootstraps_0z   = map(_ -> Float32[], 1:nmodels)
  model_au_pr_bootstraps_12z  = map(_ -> Float32[], 1:nmodels)
  model_au_pr_bootstraps_mean = map(_ -> Float32[], 1:nmodels)

  data_is = Vector{Int64}(undef, ndata)
  for bootstrap_i in 1:nbootstraps
    bootstrap_forecast_is = bootstrap_forecast_iss[bootstrap_i]
    Threads.@threads :static for fcst_i in 1:nforecasts
      bs_fcst_i = bootstrap_forecast_is[fcst_i]
      data_is[ndata_per_forecast*(fcst_i-1)+1 : ndata_per_forecast*fcst_i] = ndata_per_forecast*(bs_fcst_i-1)+1 : ndata_per_forecast*bs_fcst_i
    end

    for prediction_i in 1:nmodels
      event_name = event_names[prediction_i]
      y_0z, y_12z = Ys_0z[event_name], Ys_12z[event_name]

      au_pr_0z  = Metrics.area_under_pr_curve_fast(view(X_0z,  data_is, prediction_i), view(y_0z,  data_is), view(weights_0z,  data_is); bin_count = 1000)
      au_pr_12z = Metrics.area_under_pr_curve_fast(view(X_12z, data_is, prediction_i), view(y_12z, data_is), view(weights_12z, data_is); bin_count = 1000)
      au_pr_mean = (au_pr_0z + au_pr_12z) / 2

      push!(model_au_pr_bootstraps_0z[prediction_i],   au_pr_0z)
      push!(model_au_pr_bootstraps_12z[prediction_i],  au_pr_12z)
      push!(model_au_pr_bootstraps_mean[prediction_i], au_pr_mean)
    end
    print(".")
    flush(stdout)
  end
  println()

  row = Any[
    "model_name",
    "au_pr_0z",
    "au_pr_12z",
    "au_pr_mean",
    "logloss_0z",
    "logloss_12z",
    "logloss_mean",
  ]
  println(join(row, ","))
  total_0z_weight  = sum(weights_0z)
  total_12z_weight = sum(weights_12z)
  for prediction_i in 1:nmodels
    model_name = model_names[prediction_i]
    event_name = event_names[prediction_i]
    y_0z, y_12z = Ys_0z[event_name], Ys_12z[event_name]
    au_pr_0z   = Metrics.area_under_pr_curve(view(X_0z,  :, prediction_i), y_0z,  weights_0z)
    au_pr_12z  = Metrics.area_under_pr_curve(view(X_12z, :, prediction_i), y_12z, weights_12z)
    au_pr_mean = (au_pr_0z + au_pr_12z) / 2
    logloss_0z   = sum(MemoryConstrainedTreeBoosting.logloss.(y_0z,  view(X_0z,  :, prediction_i)) .* weights_0z)  / total_0z_weight
    logloss_12z  = sum(MemoryConstrainedTreeBoosting.logloss.(y_12z, view(X_12z, :, prediction_i)) .* weights_12z) / total_12z_weight
    logloss_mean = (logloss_0z + logloss_12z) / 2
    row = Any[
      model_name,
      au_pr_0z,
      au_pr_12z,
      au_pr_mean,
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[1]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[2]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[3]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      # two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[1]),
      # two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[2]),
      # two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[3]),
      logloss_0z,
      logloss_12z,
      logloss_mean,
    ]
    println(join(row, ","))
  end
end


# Going to use all the HREF forecasts, 2018-7-1 to 2022-5-31.

TASKS = eval(Meta.parse(get(ENV, "TASKS", "")))
if isnothing(TASKS)
  TASKS = typemin(Int64):typemax(Int64)
end

model_names = first.(HREFPredictionAblations2.models)

1 in TASKS && do_it(HREFPredictionAblations2.forecasts_day_spc_calibrated(), model_names)

# Absolutely calibrated should produce the same result for AU-PR, not not logloss
2 in TASKS && do_it(HREFPredictionAblations2.forecasts_day(), model_names; suffix = "_absolutely_calibrated")
