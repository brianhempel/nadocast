import Dates
import Random
import Printf
import Serialization
import Statistics

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import Metrics
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations")
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations2")
import HREFPredictionAblations2

push!(LOAD_PATH, (@__DIR__) * "/../models/href_day_experiment")
import HREFDayExperiment

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

const nbootstraps = parse(Int64, get(ENV, "NBOOTSTRAPS", "10")) # 10,000 takes ~10hr

println("$nbootstraps bootstraps")

function two_sided_bootstrap_p_value_paired(bootstraps_1, bootstraps_2)
  nbootstraps = length(bootstraps_1)
  Float32(min(
    count(bootstraps_1 .<= bootstraps_2) / (nbootstraps / 2),
    count(bootstraps_1 .>= bootstraps_2) / (nbootstraps / 2),
    1.0
  ))
end

function mean(xs)
  sum(xs) / length(xs)
end

model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1", r"\A((sig_)?(tornado|wind|hail))_day\z" => s"\1")

function do_it(forecasts, model_names; reference_model_is = map(_ -> nothing, model_names), suffix = "", use_5km_grid = false)

  event_names = model_name_to_event_name.(model_names)

  println("************ ablations$(suffix) ************")

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      use_5km_grid ? ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID) : forecasts;
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
  run_times_0z = Serialization.deserialize("day_ablations2_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z/run_times.serialized")
  @assert length(unique(run_times_0z)) == length(test_forecasts_0z)
  @assert sort(unique(run_times_0z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_0z))

  X_12z, Ys_12z, weights_12z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_12z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablations2_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times_12z = Serialization.deserialize("day_ablations2_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z/run_times.serialized")
  @assert length(unique(run_times_12z)) == length(test_forecasts_12z)
  @assert sort(unique(run_times_12z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_12z))

  @assert run_times_0z == (run_times_12z .- Dates.Hour(12))

  println(sort(unique(run_times_0z)))

  nforecasts = length(unique(run_times_0z))

  println("$nforecasts forecasts")

  @assert size(X_0z) == size(X_12z)
  @assert weights_0z == weights_12z

  @assert length(weights_0z) == size(X_0z,1)
  @assert length(weights_0z) / nforecasts == round(length(weights_0z) / nforecasts)
  ndata = length(weights_0z)
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
    "p_value_vs_reference",
    "logloss_0z",
    "logloss_12z",
    "logloss_mean",
  ]
  println(join(row, ","))
  model_rows = []
  total_0z_weight  = sum(weights_0z)
  total_12z_weight = sum(weights_12z)
  for prediction_i in 1:nmodels
    model_name = model_names[prediction_i]
    event_name = event_names[prediction_i]
    y_0z, y_12z = Ys_0z[event_name], Ys_12z[event_name]
    @assert size(y_0z) == size(y_12z)
    au_pr_0z   = Metrics.area_under_pr_curve(view(X_0z,  :, prediction_i), y_0z,  weights_0z)
    au_pr_12z  = Metrics.area_under_pr_curve(view(X_12z, :, prediction_i), y_12z, weights_12z)
    au_pr_mean = (au_pr_0z + au_pr_12z) / 2
    logloss_0z   = sum(MemoryConstrainedTreeBoosting.logloss.(y_0z,  view(X_0z,  :, prediction_i)) .* weights_0z)  / total_0z_weight
    logloss_12z  = sum(MemoryConstrainedTreeBoosting.logloss.(y_12z, view(X_12z, :, prediction_i)) .* weights_12z) / total_12z_weight
    logloss_mean = (logloss_0z + logloss_12z) / 2
    reference_model_i = reference_model_is[prediction_i]
    row = Any[
      model_name,
      au_pr_0z,
      au_pr_12z,
      au_pr_mean,
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[1]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[2]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      # Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[3]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      isnothing(reference_model_i) ? "" : two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[reference_model_i]),
      logloss_0z,
      logloss_12z,
      logloss_mean,
    ]
    push!(model_rows, row)
    println(join(row, ","))
  end

  for prediction_i in 1:nmodels
    model_name = model_names[prediction_i]
    event_name = event_names[prediction_i]
    reference_model_i = reference_model_is[prediction_i]

    if event_name == "hail" && !isnothing(reference_model_i)
      @assert event_names[prediction_i - 2] == "tornado"
      @assert event_names[prediction_i - 1] == "wind"

      @assert event_names[reference_model_i - 2] == "tornado"
      @assert event_names[reference_model_i - 1] == "wind"
      @assert event_names[reference_model_i - 0] == "hail"

      tornado_row = model_rows[prediction_i - 2]
      wind_row    = model_rows[prediction_i - 1]
      hail_row    = model_rows[prediction_i - 0]

      tornado_ref_row = model_rows[reference_model_i - 2]
      wind_ref_row    = model_rows[reference_model_i - 1]
      hail_ref_row    = model_rows[reference_model_i - 0]

      au_pr_0z_vs_ref   = mean([tornado_row[2] / tornado_ref_row[2], wind_row[2] / wind_ref_row[2], hail_row[2] / hail_ref_row[2]])
      au_pr_12z_vs_ref  = mean([tornado_row[3] / tornado_ref_row[3], wind_row[3] / wind_ref_row[3], hail_row[3] / hail_ref_row[3]])
      au_pr_mean_vs_ref = mean([
        tornado_row[4] / tornado_ref_row[4],
        wind_row[4]    / wind_ref_row[4],
        hail_row[4]    / hail_ref_row[4],
      ]) # this is consistent with how we do the booststraps below, which itself is consistent with the bootstraps for the individual models

      model_au_pr_bootstraps_mean_vs_ref_mean = map(1:nbootstraps) do bootstrap_i
        tornado_vs_ref = model_au_pr_bootstraps_mean[prediction_i - 2][bootstrap_i] / model_au_pr_bootstraps_mean[reference_model_i - 2][bootstrap_i]
        wind_vs_ref    = model_au_pr_bootstraps_mean[prediction_i - 1][bootstrap_i] / model_au_pr_bootstraps_mean[reference_model_i - 1][bootstrap_i]
        hail_vs_ref    = model_au_pr_bootstraps_mean[prediction_i - 0][bootstrap_i] / model_au_pr_bootstraps_mean[reference_model_i - 0][bootstrap_i]
        @assert isfinite(tornado_vs_ref)
        @assert isfinite(wind_vs_ref)
        @assert isfinite(hail_vs_ref)
        mean([tornado_vs_ref, wind_vs_ref, hail_vs_ref])
      end

      logloss_0z_vs_ref   = mean([tornado_row[6] / tornado_ref_row[6], wind_row[6] / wind_ref_row[6], hail_row[6] / hail_ref_row[6]])
      logloss_12z_vs_ref  = mean([tornado_row[7] / tornado_ref_row[7], wind_row[7] / wind_ref_row[7], hail_row[7] / hail_ref_row[7]])
      logloss_mean_vs_ref = mean([
        tornado_row[8] / tornado_ref_row[8],
        wind_row[8]    / wind_ref_row[8],
        hail_row[8]    / hail_ref_row[8],
      ]) # this is consistent with how we do the booststraps below, which itself is consistent with the bootstraps for the individual models

      row = Any[
        replace(model_name, event => "tornado_wind_hail_mean_relative_to_reference"),
        au_pr_0z_vs_ref,
        au_pr_12z_vs_ref,
        au_pr_mean_vs_ref,
        two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean_vs_ref_mean, ones(size(model_au_pr_bootstraps_mean_vs_ref_mean))),
        logloss_0z_vs_ref,
        logloss_12z_vs_ref,
        logloss_mean_vs_ref,
      ]
      println(join(row, ","))
    end
  end
end


# Going to use all the HREF forecasts, 2018-7-1 to 2022-5-31.

TASKS = eval(Meta.parse(get(ENV, "TASKS", "")))
if isnothing(TASKS)
  TASKS = typemin(Int64):typemax(Int64)
end

model_names = first.(HREFPredictionAblations2.models)

# 183 0z test forecasts before the event data cutoff date
# 183 12z test forecasts before the event data cutoff date

1 in TASKS && do_it(HREFPredictionAblations2.forecasts_day_spc_calibrated(), model_names)
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,logloss_0z,logloss_12z,logloss_mean
# tornado_mean_prob_computed_climatology_blurs_910,0.19201612,0.24624602,0.21913108,0.008547409,0.008061981,0.008304695
# wind_mean_prob_computed_climatology_blurs_910,0.39439,0.42877078,0.41158038,0.039662477,0.037288636,0.03847556
# hail_mean_prob_computed_climatology_blurs_910,0.28069717,0.3085839,0.29464054,0.02225437,0.020791082,0.021522727
# tornado_mean_prob_computed_climatology_blurs_910_before_20200523,0.18599437,0.2298384,0.20791638,0.008638885,0.008213083,0.0084259845
# wind_mean_prob_computed_climatology_blurs_910_before_20200523,0.3816257,0.4161498,0.39888775,0.040180884,0.037831593,0.03900624
# hail_mean_prob_computed_climatology_blurs_910_before_20200523,0.27238187,0.2980081,0.285195,0.022558123,0.020990126,0.021774124
# tornado_full_13831,0.1832289,0.22881232,0.20602061,0.008578844,0.008126701,0.008352773
# wind_full_13831,0.40504867,0.43850836,0.4217785,0.039378766,0.037123807,0.03825129
# hail_full_13831,0.2808212,0.31219268,0.29650694,0.022180142,0.02069553,0.021437835

# Absolutely calibrated should produce the same result for AU-PR, not not logloss
2 in TASKS && do_it(HREFPredictionAblations2.forecasts_day(), model_names; suffix = "_absolutely_calibrated")
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,logloss_0z,logloss_12z,logloss_mean
# tornado_mean_prob_computed_climatology_blurs_910,0.19201612,0.24624604,0.21913108,0.008285367,0.007743316,0.008014342
# wind_mean_prob_computed_climatology_blurs_910,0.39439,0.42877078,0.41158038,0.038538743,0.036201518,0.03737013
# hail_mean_prob_computed_climatology_blurs_910,0.28069717,0.3085839,0.29464054,0.021895804,0.02047679,0.021186296
# tornado_mean_prob_computed_climatology_blurs_910_before_20200523,0.18599437,0.2298384,0.20791638,0.008382405,0.007907587,0.008144996
# wind_mean_prob_computed_climatology_blurs_910_before_20200523,0.3816257,0.4161498,0.39888775,0.039138388,0.036856156,0.037997272
# hail_mean_prob_computed_climatology_blurs_910_before_20200523,0.27238187,0.2980081,0.285195,0.022142077,0.02072011,0.021431092
# tornado_full_13831,0.18322891,0.22881232,0.20602062,0.008286422,0.0077760066,0.008031215
# wind_full_13831,0.40504867,0.43850836,0.4217785,0.03818646,0.0358155,0.03700098
# hail_full_13831,0.2808212,0.31219268,0.29650694,0.021770103,0.020417279,0.02109369


day_experiment_model_names = first.(HREFDayExperiment.models)

# 183 0z test forecasts before the event data cutoff date
# 183 12z test forecasts before the event data cutoff date

3 in TASKS && do_it(HREFDayExperiment.blurred_spc_calibrated_day_prediction_forecasts(),  day_experiment_model_names; suffix = "_href_day_experiment")
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,logloss_0z,logloss_12z,logloss_mean
# tornado,0.16121893,0.19874004,0.17997947,0.008726703,0.008231592,0.008479148
# wind,0.3878628,0.42487022,0.40636653,0.039855875,0.03749772,0.0386768
# hail,0.27455902,0.29973814,0.2871486,0.022266427,0.020918034,0.02159223
# sig_tornado,0.09940558,0.09686993,0.09813775,0.0016321819,0.001517131,0.0015746565

# Absolutely calibrated should produce the same result for AU-PR, not not logloss
4 in TASKS && do_it(HREFDayExperiment.blurred_calibrated_day_prediction_forecasts(),      day_experiment_model_names; suffix = "_href_day_experiment_absolutely_calibrated")
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,logloss_0z,logloss_12z,logloss_mean
# tornado,0.16121893,0.19874005,0.17997949,0.008495962,0.008038434,0.008267198
# wind,0.3878628,0.42487022,0.40636653,0.038846515,0.036485177,0.037665844
# hail,0.27455905,0.29973814,0.2871486,0.021912074,0.020601101,0.021256588
# sig_tornado,0.09940558,0.09686993,0.09813775,0.0015366459,0.0014734392,0.0015050425


function make_time_title_to_forecast(forecasts)
  out = Dict{String,Forecasts.Forecast}()
  for forecast in forecasts
    time_title = Forecasts.time_title(forecast)
    @assert !haskey(out, time_title)
    out[time_title] = forecast
  end
  out
end

function associate(forecastss...)
  dicts = map(make_time_title_to_forecast, forecastss)

  out = []
  for time_title in keys(dicts[1])
    if all(dict -> haskey(dict, time_title), dicts)
      push!(out, map(dict -> dict[time_title], dicts))
    end
  end
  sort(out; alg = MergeSort, by = forecasts -> (Forecasts.run_utc_datetime(forecasts[1]), forecasts[1].forecast_hour))
end


experimental_forecasts = ForecastCombinators.concat_forecasts(associate(
  HREFPredictionAblations.forecasts_day(),
  HREFPredictionAblations2.forecasts_day(),
  HREFDayExperiment.blurred_calibrated_day_prediction_forecasts(),
))

experiment_model_names = vcat(
  first.(HREFPredictionAblations.models),
  first.(HREFPredictionAblations2.models),
  map(str -> str * "_day", first.(HREFDayExperiment.models)),
)

reference_model_is = map(experiment_model_names) do model_name
  findfirst(experiment_model_names) do ref_name
    event_name = model_name_to_event_name(model_name)
    startswith(ref_name, event_name * "_full")
  end
end

println(experiment_model_names)
println(reference_model_is)

5 in TASKS && do_it(experimental_forecasts,  experiment_model_names; reference_model_is = reference_model_is, suffix = "_all_experiments")

