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

function do_it(forecasts, model_names; reference_model_is = map(_ -> nothing, model_names), suffix = "", use_5km_grid = false, cutoff = Dates.DateTime(2022, 6, 1, 12), use_all_days_of_week_after = cutoff)

  event_names = model_name_to_event_name.(model_names)

  println("************ ablations$(suffix) ************")

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      use_5km_grid ? ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID) : forecasts;
      just_hours_near_storm_events = false
    );

  println("$(length(train_forecasts)) unfiltered train forecasts")
  println("$(length(validation_forecasts)) unfiltered validation forecasts")
  println("$(length(test_forecasts)) unfiltered test forecasts")
  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts)
  more_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff && Forecasts.valid_utc_datetime(forecast) >= use_all_days_of_week_after, vcat(train_forecasts, validation_forecasts))
  test_forecasts = vcat(test_forecasts, more_forecasts)
  sort!(test_forecasts, by = Forecasts.run_utc_datetime, alg = MergeSort)

  test_forecasts_0z  = filter(forecast -> forecast.run_hour == 0,  test_forecasts);
  test_forecasts_12z = filter(forecast -> forecast.run_hour == 12, test_forecasts);
  test_forecasts = []

  run_timeset_0z  = Set(Forecasts.run_utc_datetime.(test_forecasts_0z))
  run_timeset_12z = Set(Forecasts.run_utc_datetime.(test_forecasts_12z) .- Dates.Hour(12))
  times_to_use = intersect(run_timeset_0z, run_timeset_12z)

  filter!(forecast -> Forecasts.run_utc_datetime(forecast)                  in times_to_use, test_forecasts_0z);
  filter!(forecast -> Forecasts.run_utc_datetime(forecast) - Dates.Hour(12) in times_to_use, test_forecasts_12z);

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
        replace(model_name, event_name => "tornado_wind_hail_mean_relative_to_reference"),
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
  event_name = model_name_to_event_name(model_name)
  findfirst(experiment_model_names) do ref_name
    # half data and day experiments should use their ablated baseline as the reference
    if occursin("_before_", model_name)
      ref_name == event_name * "_mean_prob_computed_climatology_blurs_910"
    elseif endswith(model_name, "_day")
      ref_name == event_name * "_mean_prob_computed_climatology_blurs_910"
    else
      startswith(ref_name, event_name * "_full")
    end
  end
end

println(experiment_model_names)
# 1 ["tornado_mean_58",
# 2 "tornado_prob_80",
# 3 "tornado_mean_prob_138",
# 4 "tornado_mean_prob_computed_no_sv_219",
# 5 "tornado_mean_prob_computed_220",
# 6 "tornado_mean_prob_computed_partial_climatology_227",
# 7 "tornado_mean_prob_computed_climatology_253",
# 8 "tornado_mean_prob_computed_climatology_blurs_910",
# 9 "tornado_mean_prob_computed_climatology_grads_1348",
# 10 "tornado_mean_prob_computed_climatology_blurs_grads_2005",
# 11 "tornado_mean_prob_computed_climatology_prior_next_hrs_691",
# 12 "tornado_mean_prob_computed_climatology_3hr_1567",
# 13 "tornado_full_13831",
# 14 "tornado_mean_prob_computed_climatology_blurs_910",
# 15 "wind_mean_prob_computed_climatology_blurs_910",
# 16 "hail_mean_prob_computed_climatology_blurs_910",
# 17 "tornado_mean_prob_computed_climatology_blurs_910_before_20200523",
# 18 "wind_mean_prob_computed_climatology_blurs_910_before_20200523",
# 19 "hail_mean_prob_computed_climatology_blurs_910_before_20200523",
# 20 "tornado_full_13831",
# 21 "wind_full_13831",
# 22 "hail_full_13831",
# 23 "tornado_day",
# 24 "wind_day",
# 25 "hail_day",
# 26 "sig_tornado_day"]
println(reference_model_is)
# Union{Nothing, Int64}[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 21, 22, 8, 15, 16, 13, 21, 22, 8, 15, 16, nothing]


5 in TASKS && do_it(experimental_forecasts,  experiment_model_names; reference_model_is = reference_model_is, suffix = "_all_experiments")


# Use Sundays 2018-7-1 to 2022-5-31 and then all days through 2023-9-30

# FORECAST_DISK_PREFETCH=false NBOOTSTRAPS=2 TASKS=[6] JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. TestAUPRPreciseAblations2.jl
# FORECAST_DISK_PREFETCH=false NBOOTSTRAPS=10000 TASKS=[6] JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. TestAUPRPreciseAblations2.jl
6 in TASKS && do_it(experimental_forecasts,  experiment_model_names; reference_model_is = reference_model_is, suffix = "_all_experiments_all_nontraining_days", cutoff = Dates.DateTime(2023, 10, 1, 12), use_all_days_of_week_after = Dates.DateTime(2022, 6, 1, 12))
# 4777 unfiltered train forecasts
# 950 unfiltered validation forecasts
# 963 unfiltered test forecasts
# 598 0z test forecasts before the event data cutoff date
# 598 12z test forecasts before the event data cutoff date
# [Dates.DateTime("2018-07-01T00:00:00"), Dates.DateTime("2018-07-08T00:00:00"), Dates.DateTime("2018-07-15T00:00:00"), Dates.DateTime("2018-07-22T00:00:00"), Dates.DateTime("2018-07-29T00:00:00"), Dates.DateTime("2018-08-05T00:00:00"), Dates.DateTime("2018-08-12T00:00:00"), Dates.DateTime("2018-08-19T00:00:00"), Dates.DateTime("2018-08-26T
# 00:00:00"), Dates.DateTime("2018-09-02T00:00:00"), Dates.DateTime("2018-09-09T00:00:00"), Dates.DateTime("2018-09-16T00:00:00"), Dates.DateTime("2018-09-23T00:00:00"), Dates.DateTime("2018-09-30T00:00:00"), Dates.DateTime("2018-10-07T00:00:00"), Dates.DateTime("2018-10-14T00:00:00"), Dates.DateTime("2018-10-21T00:00:00"), Dates.DateTime("
# 2018-10-28T00:00:00"), Dates.DateTime("2018-11-04T00:00:00"), Dates.DateTime("2018-11-11T00:00:00"), Dates.DateTime("2018-11-18T00:00:00"), Dates.DateTime("2018-11-25T00:00:00"), Dates.DateTime("2018-12-02T00:00:00"), Dates.DateTime("2018-12-09T00:00:00"), Dates.DateTime("2018-12-16T00:00:00"), Dates.DateTime("2018-12-30T00:00:00"), Dates
# .DateTime("2019-01-06T00:00:00"), Dates.DateTime("2019-01-13T00:00:00"), Dates.DateTime("2019-01-20T00:00:00"), Dates.DateTime("2019-01-27T00:00:00"), Dates.DateTime("2019-02-03T00:00:00"), Dates.DateTime("2019-02-10T00:00:00"), Dates.DateTime("2019-02-17T00:00:00"), Dates.DateTime("2019-02-24T00:00:00"), Dates.DateTime("2019-03-03T00:00:
# 00"), Dates.DateTime("2019-03-10T00:00:00"), Dates.DateTime("2019-03-17T00:00:00"), Dates.DateTime("2019-03-24T00:00:00"), Dates.DateTime("2019-03-31T00:00:00"), Dates.DateTime("2019-04-07T00:00:00"), Dates.DateTime("2019-04-14T00:00:00"), Dates.DateTime("2019-04-21T00:00:00"), Dates.DateTime("2019-04-28T00:00:00"), Dates.DateTime("2019-0
# 5-05T00:00:00"), Dates.DateTime("2019-05-12T00:00:00"), Dates.DateTime("2019-05-19T00:00:00"), Dates.DateTime("2019-05-26T00:00:00"), Dates.DateTime("2019-06-02T00:00:00"), Dates.DateTime("2019-06-09T00:00:00"), Dates.DateTime("2019-06-16T00:00:00"), Dates.DateTime("2019-06-23T00:00:00"), Dates.DateTime("2019-06-30T00:00:00"), Dates.DateT
# ime("2019-07-07T00:00:00"), Dates.DateTime("2019-07-14T00:00:00"), Dates.DateTime("2019-07-21T00:00:00"), Dates.DateTime("2019-07-28T00:00:00"), Dates.DateTime("2019-08-04T00:00:00"), Dates.DateTime("2019-08-11T00:00:00"), Dates.DateTime("2019-08-18T00:00:00"), Dates.DateTime("2019-08-25T00:00:00"), Dates.DateTime("2019-09-01T00:00:00"),
# Dates.DateTime("2019-09-08T00:00:00"), Dates.DateTime("2019-09-15T00:00:00"), Dates.DateTime("2019-09-22T00:00:00"), Dates.DateTime("2019-09-29T00:00:00"), Dates.DateTime("2019-10-06T00:00:00"), Dates.DateTime("2019-10-13T00:00:00"), Dates.DateTime("2019-10-20T00:00:00"), Dates.DateTime("2019-10-27T00:00:00"), Dates.DateTime("2019-11-03T0
# 0:00:00"), Dates.DateTime("2019-11-10T00:00:00"), Dates.DateTime("2019-11-17T00:00:00"), Dates.DateTime("2019-11-24T00:00:00"), Dates.DateTime("2019-12-01T00:00:00"), Dates.DateTime("2019-12-08T00:00:00"), Dates.DateTime("2019-12-15T00:00:00"), Dates.DateTime("2019-12-22T00:00:00"), Dates.DateTime("2019-12-29T00:00:00"), Dates.DateTime("2
# 020-01-05T00:00:00"), Dates.DateTime("2020-01-12T00:00:00"), Dates.DateTime("2020-01-19T00:00:00"), Dates.DateTime("2020-01-26T00:00:00"), Dates.DateTime("2020-02-02T00:00:00"), Dates.DateTime("2020-02-09T00:00:00"), Dates.DateTime("2020-02-16T00:00:00"), Dates.DateTime("2020-02-23T00:00:00"), Dates.DateTime("2020-03-01T00:00:00"), Dates.
# DateTime("2020-03-08T00:00:00"), Dates.DateTime("2020-03-15T00:00:00"), Dates.DateTime("2020-03-22T00:00:00"), Dates.DateTime("2020-03-29T00:00:00"), Dates.DateTime("2020-04-05T00:00:00"), Dates.DateTime("2020-04-12T00:00:00"), Dates.DateTime("2020-04-19T00:00:00"), Dates.DateTime("2020-04-26T00:00:00"), Dates.DateTime("2020-05-03T00:00:0
# 0"), Dates.DateTime("2020-05-10T00:00:00"), Dates.DateTime("2020-05-17T00:00:00"), Dates.DateTime("2020-05-24T00:00:00"), Dates.DateTime("2020-05-31T00:00:00"), Dates.DateTime("2020-06-07T00:00:00"), Dates.DateTime("2020-06-14T00:00:00"), Dates.DateTime("2020-06-21T00:00:00"), Dates.DateTime("2020-06-28T00:00:00"), Dates.DateTime("2020-07
# -05T00:00:00"), Dates.DateTime("2020-07-12T00:00:00"), Dates.DateTime("2020-07-19T00:00:00"), Dates.DateTime("2020-07-26T00:00:00"), Dates.DateTime("2020-08-02T00:00:00"), Dates.DateTime("2020-08-09T00:00:00"), Dates.DateTime("2020-08-16T00:00:00"), Dates.DateTime("2020-08-23T00:00:00"), Dates.DateTime("2020-08-30T00:00:00"), Dates.DateTi
# me("2020-09-06T00:00:00"), Dates.DateTime("2020-09-13T00:00:00"), Dates.DateTime("2020-09-20T00:00:00"), Dates.DateTime("2020-09-27T00:00:00"), Dates.DateTime("2020-10-04T00:00:00"), Dates.DateTime("2020-10-11T00:00:00"), Dates.DateTime("2020-10-18T00:00:00"), Dates.DateTime("2021-03-21T00:00:00"), Dates.DateTime("2021-03-28T00:00:00"), D
# ates.DateTime("2021-04-04T00:00:00"), Dates.DateTime("2021-04-11T00:00:00"), Dates.DateTime("2021-04-18T00:00:00"), Dates.DateTime("2021-04-25T00:00:00"), Dates.DateTime("2021-05-02T00:00:00"), Dates.DateTime("2021-05-09T00:00:00"), Dates.DateTime("2021-05-16T00:00:00"), Dates.DateTime("2021-05-23T00:00:00"), Dates.DateTime("2021-05-30T00
# :00:00"), Dates.DateTime("2021-06-06T00:00:00"), Dates.DateTime("2021-06-13T00:00:00"), Dates.DateTime("2021-06-20T00:00:00"), Dates.DateTime("2021-06-27T00:00:00"), Dates.DateTime("2021-07-04T00:00:00"), Dates.DateTime("2021-07-11T00:00:00"), Dates.DateTime("2021-07-18T00:00:00"), Dates.DateTime("2021-07-25T00:00:00"), Dates.DateTime("20
# 21-08-01T00:00:00"), Dates.DateTime("2021-08-08T00:00:00"), Dates.DateTime("2021-08-15T00:00:00"), Dates.DateTime("2021-08-22T00:00:00"), Dates.DateTime("2021-08-29T00:00:00"), Dates.DateTime("2021-09-05T00:00:00"), Dates.DateTime("2021-09-12T00:00:00"), Dates.DateTime("2021-09-19T00:00:00"), Dates.DateTime("2021-09-26T00:00:00"), Dates.D
# ateTime("2021-10-03T00:00:00"), Dates.DateTime("2021-10-10T00:00:00"), Dates.DateTime("2021-10-17T00:00:00"), Dates.DateTime("2021-10-24T00:00:00"), Dates.DateTime("2021-10-31T00:00:00"), Dates.DateTime("2021-11-07T00:00:00"), Dates.DateTime("2021-11-14T00:00:00"), Dates.DateTime("2021-11-21T00:00:00"), Dates.DateTime("2021-11-28T00:00:00
# "), Dates.DateTime("2021-12-05T00:00:00"), Dates.DateTime("2021-12-12T00:00:00"), Dates.DateTime("2021-12-19T00:00:00"), Dates.DateTime("2021-12-26T00:00:00"), Dates.DateTime("2022-01-02T00:00:00"), Dates.DateTime("2022-01-09T00:00:00"), Dates.DateTime("2022-01-16T00:00:00"), Dates.DateTime("2022-01-23T00:00:00"), Dates.DateTime("2022-01-
# 30T00:00:00"), Dates.DateTime("2022-02-06T00:00:00"), Dates.DateTime("2022-02-13T00:00:00"), Dates.DateTime("2022-02-20T00:00:00"), Dates.DateTime("2022-02-27T00:00:00"), Dates.DateTime("2022-03-06T00:00:00"), Dates.DateTime("2022-03-13T00:00:00"), Dates.DateTime("2022-03-20T00:00:00"), Dates.DateTime("2022-03-27T00:00:00"), Dates.DateTim
# e("2022-04-03T00:00:00"), Dates.DateTime("2022-04-10T00:00:00"), Dates.DateTime("2022-04-17T00:00:00"), Dates.DateTime("2022-04-24T00:00:00"), Dates.DateTime("2022-05-01T00:00:00"), Dates.DateTime("2022-05-08T00:00:00"), Dates.DateTime("2022-05-15T00:00:00"), Dates.DateTime("2022-05-22T00:00:00"), Dates.DateTime("2022-05-29T00:00:00"), Da
# tes.DateTime("2022-06-01T00:00:00"), Dates.DateTime("2022-06-02T00:00:00"), Dates.DateTime("2022-06-03T00:00:00"), Dates.DateTime("2022-06-04T00:00:00"), Dates.DateTime("2022-06-05T00:00:00"), Dates.DateTime("2022-06-06T00:00:00"), Dates.DateTime("2022-06-07T00:00:00"), Dates.DateTime("2022-06-08T00:00:00"), Dates.DateTime("2022-06-09T00:
# 00:00"), Dates.DateTime("2022-06-10T00:00:00"), Dates.DateTime("2022-06-11T00:00:00"), Dates.DateTime("2022-06-12T00:00:00"), Dates.DateTime("2022-06-13T00:00:00"), Dates.DateTime("2022-06-14T00:00:00"), Dates.DateTime("2022-06-15T00:00:00"), Dates.DateTime("2022-06-16T00:00:00"), Dates.DateTime("2022-06-17T00:00:00"), Dates.DateTime("202
# 2-06-18T00:00:00"), Dates.DateTime("2022-06-19T00:00:00"), Dates.DateTime("2022-06-20T00:00:00"), Dates.DateTime("2022-06-21T00:00:00"), Dates.DateTime("2022-06-22T00:00:00"), Dates.DateTime("2022-06-23T00:00:00"), Dates.DateTime("2022-06-24T00:00:00"), Dates.DateTime("2022-06-25T00:00:00"), Dates.DateTime("2022-06-26T00:00:00"), Dates.Da
# teTime("2022-06-27T00:00:00"), Dates.DateTime("2022-06-28T00:00:00"), Dates.DateTime("2022-06-29T00:00:00"), Dates.DateTime("2022-06-30T00:00:00"), Dates.DateTime("2022-07-01T00:00:00"), Dates.DateTime("2022-07-02T00:00:00"), Dates.DateTime("2022-07-03T00:00:00"), Dates.DateTime("2022-07-04T00:00:00"), Dates.DateTime("2022-07-05T00:00:00"
# ), Dates.DateTime("2022-07-06T00:00:00"), Dates.DateTime("2022-07-07T00:00:00"), Dates.DateTime("2022-07-08T00:00:00"), Dates.DateTime("2022-07-09T00:00:00"), Dates.DateTime("2022-07-10T00:00:00"), Dates.DateTime("2022-07-11T00:00:00"), Dates.DateTime("2022-07-12T00:00:00"), Dates.DateTime("2022-07-13T00:00:00"), Dates.DateTime("2022-07-1
# 4T00:00:00"), Dates.DateTime("2022-07-15T00:00:00"), Dates.DateTime("2022-07-16T00:00:00"), Dates.DateTime("2022-07-17T00:00:00"), Dates.DateTime("2022-07-18T00:00:00"), Dates.DateTime("2022-07-19T00:00:00"), Dates.DateTime("2022-07-20T00:00:00"), Dates.DateTime("2022-07-21T00:00:00"), Dates.DateTime("2022-07-22T00:00:00"), Dates.DateTime
# ("2022-07-23T00:00:00"), Dates.DateTime("2022-07-24T00:00:00"), Dates.DateTime("2022-07-25T00:00:00"), Dates.DateTime("2022-07-26T00:00:00"), Dates.DateTime("2022-07-27T00:00:00"), Dates.DateTime("2022-07-28T00:00:00"), Dates.DateTime("2022-07-29T00:00:00"), Dates.DateTime("2022-07-30T00:00:00"), Dates.DateTime("2022-07-31T00:00:00"), Dat
# es.DateTime("2022-08-01T00:00:00"), Dates.DateTime("2022-08-02T00:00:00"), Dates.DateTime("2022-08-03T00:00:00"), Dates.DateTime("2022-08-04T00:00:00"), Dates.DateTime("2022-08-05T00:00:00"), Dates.DateTime("2022-08-06T00:00:00"), Dates.DateTime("2022-08-07T00:00:00"), Dates.DateTime("2022-08-08T00:00:00"), Dates.DateTime("2022-08-09T00:0
# 0:00"), Dates.DateTime("2022-08-10T00:00:00"), Dates.DateTime("2022-08-11T00:00:00"), Dates.DateTime("2022-08-12T00:00:00"), Dates.DateTime("2022-08-13T00:00:00"), Dates.DateTime("2022-08-14T00:00:00"), Dates.DateTime("2022-08-15T00:00:00"), Dates.DateTime("2022-08-16T00:00:00"), Dates.DateTime("2022-08-17T00:00:00"), Dates.DateTime("2022
# -08-18T00:00:00"), Dates.DateTime("2022-08-19T00:00:00"), Dates.DateTime("2022-08-20T00:00:00"), Dates.DateTime("2022-08-21T00:00:00"), Dates.DateTime("2022-08-22T00:00:00"), Dates.DateTime("2022-08-23T00:00:00"), Dates.DateTime("2022-08-24T00:00:00"), Dates.DateTime("2022-08-25T00:00:00"), Dates.DateTime("2022-08-26T00:00:00"), Dates.Dat
# eTime("2022-08-27T00:00:00"), Dates.DateTime("2022-08-28T00:00:00"), Dates.DateTime("2022-08-29T00:00:00"), Dates.DateTime("2022-08-30T00:00:00"), Dates.DateTime("2022-08-31T00:00:00"), Dates.DateTime("2022-09-01T00:00:00"), Dates.DateTime("2022-09-02T00:00:00"), Dates.DateTime("2022-09-03T00:00:00"), Dates.DateTime("2022-09-04T00:00:00")
# , Dates.DateTime("2022-09-05T00:00:00"), Dates.DateTime("2022-09-06T00:00:00"), Dates.DateTime("2022-09-07T00:00:00"), Dates.DateTime("2022-09-08T00:00:00"), Dates.DateTime("2022-09-09T00:00:00"), Dates.DateTime("2022-09-10T00:00:00"), Dates.DateTime("2022-09-11T00:00:00"), Dates.DateTime("2022-09-12T00:00:00"), Dates.DateTime("2022-09-13
# T00:00:00"), Dates.DateTime("2022-09-14T00:00:00"), Dates.DateTime("2022-09-15T00:00:00"), Dates.DateTime("2022-09-16T00:00:00"), Dates.DateTime("2022-09-17T00:00:00"), Dates.DateTime("2022-09-18T00:00:00"), Dates.DateTime("2022-09-19T00:00:00"), Dates.DateTime("2022-09-20T00:00:00"), Dates.DateTime("2022-09-21T00:00:00"), Dates.DateTime(
# "2022-09-22T00:00:00"), Dates.DateTime("2022-09-23T00:00:00"), Dates.DateTime("2022-09-24T00:00:00"), Dates.DateTime("2022-09-25T00:00:00"), Dates.DateTime("2022-09-26T00:00:00"), Dates.DateTime("2022-09-27T00:00:00"), Dates.DateTime("2022-09-28T00:00:00"), Dates.DateTime("2022-09-29T00:00:00"), Dates.DateTime("2022-09-30T00:00:00"), Date
# s.DateTime("2022-10-01T00:00:00"), Dates.DateTime("2022-10-02T00:00:00"), Dates.DateTime("2022-10-03T00:00:00"), Dates.DateTime("2022-10-04T00:00:00"), Dates.DateTime("2022-10-05T00:00:00"), Dates.DateTime("2022-10-06T00:00:00"), Dates.DateTime("2022-10-07T00:00:00"), Dates.DateTime("2022-10-09T00:00:00"), Dates.DateTime("2022-10-10T00:00
# :00"), Dates.DateTime("2022-10-11T00:00:00"), Dates.DateTime("2022-10-12T00:00:00"), Dates.DateTime("2022-10-13T00:00:00"), Dates.DateTime("2022-10-14T00:00:00"), Dates.DateTime("2022-10-15T00:00:00"), Dates.DateTime("2022-10-16T00:00:00"), Dates.DateTime("2022-10-17T00:00:00"), Dates.DateTime("2022-10-18T00:00:00"), Dates.DateTime("2022-
# 10-19T00:00:00"), Dates.DateTime("2022-10-20T00:00:00"), Dates.DateTime("2022-10-21T00:00:00"), Dates.DateTime("2022-10-22T00:00:00"), Dates.DateTime("2022-10-23T00:00:00"), Dates.DateTime("2022-10-24T00:00:00"), Dates.DateTime("2022-10-25T00:00:00"), Dates.DateTime("2022-10-26T00:00:00"), Dates.DateTime("2022-10-27T00:00:00"), Dates.Date
# Time("2022-10-28T00:00:00"), Dates.DateTime("2022-10-29T00:00:00"), Dates.DateTime("2022-10-30T00:00:00"), Dates.DateTime("2022-10-31T00:00:00"), Dates.DateTime("2022-11-01T00:00:00"), Dates.DateTime("2022-11-02T00:00:00"), Dates.DateTime("2022-11-03T00:00:00"), Dates.DateTime("2022-11-04T00:00:00"), Dates.DateTime("2022-11-05T00:00:00"),
#  Dates.DateTime("2022-11-06T00:00:00"), Dates.DateTime("2022-11-07T00:00:00"), Dates.DateTime("2022-11-08T00:00:00"), Dates.DateTime("2022-11-09T00:00:00"), Dates.DateTime("2022-11-10T00:00:00"), Dates.DateTime("2022-11-11T00:00:00"), Dates.DateTime("2022-11-12T00:00:00"), Dates.DateTime("2022-11-13T00:00:00"), Dates.DateTime("2022-11-14T
# 00:00:00"), Dates.DateTime("2022-11-15T00:00:00"), Dates.DateTime("2022-11-16T00:00:00"), Dates.DateTime("2022-11-17T00:00:00"), Dates.DateTime("2022-11-18T00:00:00"), Dates.DateTime("2022-11-19T00:00:00"), Dates.DateTime("2022-11-20T00:00:00"), Dates.DateTime("2022-11-21T00:00:00"), Dates.DateTime("2022-11-22T00:00:00"), Dates.DateTime("
# 2022-11-23T00:00:00"), Dates.DateTime("2022-11-24T00:00:00"), Dates.DateTime("2022-11-25T00:00:00"), Dates.DateTime("2022-11-26T00:00:00"), Dates.DateTime("2022-11-27T00:00:00"), Dates.DateTime("2022-11-28T00:00:00"), Dates.DateTime("2022-11-29T00:00:00"), Dates.DateTime("2022-11-30T00:00:00"), Dates.DateTime("2022-12-01T00:00:00"), Dates
# .DateTime("2022-12-02T00:00:00"), Dates.DateTime("2022-12-03T00:00:00"), Dates.DateTime("2022-12-04T00:00:00"), Dates.DateTime("2022-12-05T00:00:00"), Dates.DateTime("2022-12-06T00:00:00"), Dates.DateTime("2022-12-07T00:00:00"), Dates.DateTime("2022-12-08T00:00:00"), Dates.DateTime("2022-12-09T00:00:00"), Dates.DateTime("2022-12-10T00:00:00"), Dates.DateTime("2022-12-11T00:00:00"), Dates.DateTime("2022-12-12T00:00:00"), Dates.DateTime("2022-12-13T00:00:00"), Dates.DateTime("2022-12-14T00:00:00"), Dates.DateTime("2022-12-15T00:00:00"), Dates.DateTime("2022-12-16T00:00:00"), Dates.DateTime("2022-12-17T00:00:00"), Dates.DateTime("2022-12-18T00:00:00"), Dates.DateTime("2022-12-19T00:00:00"), Dates.DateTime("2022-12-20T00:00:00"), Dates.DateTime("2022-12-21T00:00:00"), Dates.DateTime("2022-12-22T00:00:00"), Dates.DateTime("2022-12-23T00:00:00"), Dates.DateTime("2022-12-24T00:00:00"), Dates.DateTime("2023-01-29T00:00:00"), Dates.DateTime("2023-01-30T00:00:00"), Dates.DateTime("2023-01-31T00:00:00"), Dates.DateTime("2023-02-01T00:00:00"), Dates.DateTime("2023-02-02T00:00:00"), Dates.DateTime("2023-02-03T00:00:00"), Dates.DateTime("2023-02-04T00:00:00"), Dates.DateTime("2023-02-05T00:00:00"), Dates.DateTime("2023-02-06T00:00:00"), Dates.DateTime("2023-02-07T00:00:00"), Dates.DateTime("2023-02-08T00:00:00"), Dates.DateTime("2023-02-09T00:00:00"), Dates.DateTime("2023-02-10T00:00:00"), Dates.DateTime("2023-02-11T00:00:00"), Dates.DateTime("2023-02-12T00:00:00"), Dates.DateTime("2023-02-13T00:00:00"), Dates.DateTime("2023-02-14T00:00:00"), Dates.DateTime("2023-02-15T00:00:00"), Dates.DateTime("2023-02-16T00:00:00"), Dates.DateTime("2023-02-17T00:00:00"), Dates.DateTime("2023-02-18T00:00:00"), Dates.DateTime("2023-02-19T00:00:00"), Dates.DateTime("2023-02-20T00:00:00"), Dates.DateTime("2023-02-21T00:00:00"), Dates.DateTime("2023-02-22T00:00:00"), Dates.DateTime("2023-02-23T00:00:00"), Dates.DateTime("2023-02-24T00:00:00"), Dates.DateTime("2023-02-25T00:00:00"), Dates.DateTime("2023-02-26T00:00:00"), Dates.DateTime("2023-02-27T00:00:00"), Dates.DateTime("2023-02-28T00:00:00"), Dates.DateTime("2023-03-01T00:00:00"), Dates.DateTime("2023-03-02T00:00:00"), Dates.DateTime("2023-03-03T00:00:00"), Dates.DateTime("2023-03-04T00:00:00"), Dates.DateTime("2023-03-05T00:00:00"), Dates.DateTime("2023-03-06T00:00:00"), Dates.DateTime("2023-03-07T00:00:00"), Dates.DateTime("2023-03-08T00:00:00"), Dates.DateTime("2023-03-09T00:00:00"), Dates.DateTime("2023-03-10T00:00:00"), Dates.DateTime("2023-03-11T00:00:00"), Dates.DateTime("2023-03-12T00:00:00"), Dates.DateTime("2023-03-13T00:00:00"), Dates.DateTime("2023-03-14T00:00:00"), Dates.DateTime("2023-03-15T00:00:00"), Dates.DateTime("2023-03-16T00:00:00"), Dates.DateTime("2023-03-17T00:00:00"), Dates.DateTime("2023-03-18T00:00:00"), Dates.DateTime("2023-03-19T00:00:00"), Dates.DateTime("2023-03-20T00:00:00"), Dates.DateTime("2023-03-21T00:00:00"), Dates.DateTime("2023-03-22T00:00:00"), Dates.DateTime("2023-03-23T00:00:00"), Dates.DateTime("2023-03-24T00:00:00"), Dates.DateTime("2023-03-25T00:00:00"), Dates.DateTime("2023-03-26T00:00:00"), Dates.DateTime("2023-03-27T00:00:00"), Dates.DateTime("2023-03-28T00:00:00"), Dates.DateTime("2023-03-29T00:00:00"), Dates.DateTime("2023-03-30T00:00:00"), Dates.DateTime("2023-03-31T00:00:00"), Dates.DateTime("2023-04-01T00:00:00"), Dates.DateTime("2023-04-02T00:00:00"), Dates.DateTime("2023-04-03T00:00:00"), Dates.DateTime("2023-04-04T00:00:00"), Dates.DateTime("2023-04-05T00:00:00"), Dates.DateTime("2023-04-06T00:00:00"), Dates.DateTime("2023-04-07T00:00:00"), Dates.DateTime("2023-04-08T00:00:00"), Dates.DateTime("2023-04-09T00:00:00"), Dates.DateTime("2023-04-10T00:00:00"), Dates.DateTime("2023-04-11T00:00:00"), Dates.DateTime("2023-04-12T00:00:00"), Dates.DateTime("2023-04-13T00:00:00"), Dates.DateTime("2023-04-14T00:00:00"), Dates.DateTime("2023-04-15T00:00:00"), Dates.DateTime("2023-04-16T00:00:00"), Dates.DateTime("2023-04-17T00:00:00"), Dates.DateTime("2023-04-18T00:00:00"), Dates.DateTime("2023-04-19T00:00:00"), Dates.DateTime("2023-04-20T00:00:00"), Dates.DateTime("2023-04-21T00:00:00"), Dates.DateTime("2023-04-28T00:00:00"), Dates.DateTime("2023-04-29T00:00:00"), Dates.DateTime("2023-04-30T00:00:00"), Dates.DateTime("2023-05-01T00:00:00"), Dates.DateTime("2023-05-02T00:00:00"), Dates.DateTime("2023-05-03T00:00:00"), Dates.DateTime("2023-05-04T00:00:00"), Dates.DateTime("2023-05-05T00:00:00"), Dates.DateTime("2023-05-06T00:00:00"), Dates.DateTime("2023-05-07T00:00:00"), Dates.DateTime("2023-05-08T00:00:00"), Dates.DateTime("2023-05-09T00:00:00"), Dates.DateTime("2023-05-10T00:00:00"), Dates.DateTime("2023-05-11T00:00:00"), Dates.DateTime("2023-05-12T00:00:00"), Dates.DateTime("2023-05-13T00:00:00"), Dates.DateTime("2023-05-14T00:00:00"), Dates.DateTime("2023-05-15T00:00:00"), Dates.DateTime("2023-05-16T00:00:00"), Dates.DateTime("2023-05-17T00:00:00"), Dates.DateTime("2023-05-18T00:00:00"), Dates.DateTime("2023-05-19T00:00:00"), Dates.DateTime("2023-05-20T00:00:00"), Dates.DateTime("2023-05-21T00:00:00"), Dates.DateTime("2023-05-22T00:00:00"), Dates.DateTime("2023-05-23T00:00:00"), Dates.DateTime("2023-05-24T00:00:00"), Dates.DateTime("2023-05-25T00:00:00"), Dates.DateTime("2023-05-26T00:00:00"), Dates.DateTime("2023-05-27T00:00:00"), Dates.DateTime("2023-05-28T00:00:00"), Dates.DateTime("2023-05-29T00:00:00"), Dates.DateTime("2023-06-05T00:00:00"), Dates.DateTime("2023-06-06T00:00:00"), Dates.DateTime("2023-06-07T00:00:00"), Dates.DateTime("2023-06-08T00:00:00"), Dates.DateTime("2023-06-09T00:00:00"), Dates.DateTime("2023-06-10T00:00:00"), Dates.DateTime("2023-06-11T00:00:00"), Dates.DateTime("2023-06-12T00:00:00"), Dates.DateTime("2023-06-13T00:00:00"), Dates.DateTime("2023-06-14T00:00:00"), Dates.DateTime("2023-06-15T00:00:00"), Dates.DateTime("2023-06-16T00:00:00"), Dates.DateTime("2023-06-17T00:00:00"), Dates.DateTime("2023-06-18T00:00:00"), Dates.DateTime("2023-06-19T00:00:00"), Dates.DateTime("2023-06-20T00:00:00"), Dates.DateTime("2023-06-21T00:00:00"), Dates.DateTime("2023-06-22T00:00:00"), Dates.DateTime("2023-06-23T00:00:00"), Dates.DateTime("2023-06-24T00:00:00"), Dates.DateTime("2023-06-25T00:00:00"), Dates.DateTime("2023-06-26T00:00:00"), Dates.DateTime("2023-06-27T00:00:00"), Dates.DateTime("2023-06-28T00:00:00"), Dates.DateTime("2023-06-29T00:00:00"), Dates.DateTime("2023-06-30T00:00:00"), Dates.DateTime("2023-07-01T00:00:00"), Dates.DateTime("2023-07-02T00:00:00"), Dates.DateTime("2023-07-03T00:00:00"), Dates.DateTime("2023-07-04T00:00:00"), Dates.DateTime("2023-07-05T00:00:00"), Dates.DateTime("2023-07-06T00:00:00"), Dates.DateTime("2023-07-07T00:00:00"), Dates.DateTime("2023-07-08T00:00:00"), Dates.DateTime("2023-07-09T00:00:00"), Dates.DateTime("2023-07-10T00:00:00"), Dates.DateTime("2023-07-11T00:00:00"), Dates.DateTime("2023-07-12T00:00:00"), Dates.DateTime("2023-07-13T00:00:00"), Dates.DateTime("2023-07-14T00:00:00"), Dates.DateTime("2023-07-15T00:00:00"), Dates.DateTime("2023-07-16T00:00:00"), Dates.DateTime("2023-07-17T00:00:00"), Dates.DateTime("2023-07-18T00:00:00"), Dates.DateTime("2023-07-19T00:00:00"), Dates.DateTime("2023-07-20T00:00:00"), Dates.DateTime("2023-07-21T00:00:00"), Dates.DateTime("2023-07-22T00:00:00"), Dates.DateTime("2023-07-23T00:00:00"), Dates.DateTime("2023-07-24T00:00:00"), Dates.DateTime("2023-07-25T00:00:00"), Dates.DateTime("2023-07-26T00:00:00"), Dates.DateTime("2023-07-27T00:00:00"), Dates.DateTime("2023-07-28T00:00:00"), Dates.DateTime("2023-07-29T00:00:00"), Dates.DateTime("2023-07-30T00:00:00"), Dates.DateTime("2023-07-31T00:00:00"), Dates.DateTime("2023-08-01T00:00:00"), Dates.DateTime("2023-08-02T00:00:00"), Dates.DateTime("2023-08-03T00:00:00"), Dates.DateTime("2023-08-28T00:00:00"), Dates.DateTime("2023-08-29T00:00:00"), Dates.DateTime("2023-08-30T00:00:00"), Dates.DateTime("2023-08-31T00:00:00"), Dates.DateTime("2023-09-01T00:00:00"), Dates.DateTime("2023-09-02T00:00:00"), Dates.DateTime("2023-09-03T00:00:00"), Dates.DateTime("2023-09-04T00:00:00"), Dates.DateTime("2023-09-05T00:00:00"), Dates.DateTime("2023-09-06T00:00:00"), Dates.DateTime("2023-09-07T00:00:00"), Dates.DateTime("2023-09-08T00:00:00"), Dates.DateTime("2023-09-09T00:00:00"), Dates.DateTime("2023-09-10T00:00:00"), Dates.DateTime("2023-09-11T00:00:00"), Dates.DateTime("2023-09-12T00:00:00"), Dates.DateTime("2023-09-13T00:00:00"), Dates.DateTime("2023-09-14T00:00:00"), Dates.DateTime("2023-09-15T00:00:00"), Dates.DateTime("2023-09-16T00:00:00"), Dates.DateTime("2023-09-17T00:00:00"), Dates.DateTime("2023-09-18T00:00:00"), Dates.DateTime("2023-09-19T00:00:00"), Dates.DateTime("2023-09-20T00:00:00"), Dates.DateTime("2023-09-21T00:00:00"), Dates.DateTime("2023-09-22T00:00:00"), Dates.DateTime("2023-09-23T00:00:00"), Dates.DateTime("2023-09-24T00:00:00"), Dates.DateTime("2023-09-25T00:00:00"), Dates.DateTime("2023-09-26T00:00:00"), Dates.DateTime("2023-09-27T00:00:00"), Dates.DateTime("2023-09-28T00:00:00"), Dates.DateTime("2023-09-29T00:00:00"), Dates.DateTime("2023-09-30T00:00:00")]
# 598 forecasts

# model_name,au_pr_0z,au_pr_12z,au_pr_mean,p_value_vs_reference,logloss_0z,logloss_12z,logloss_mean
# tornado_mean_58,0.1380769,0.1698487,0.15396279,0.0242,0.0082720155,0.0076935166,0.007982766
# tornado_prob_80,0.16501614,0.18206348,0.17353982,0.2474,0.008077488,0.007586314,0.007831901
# tornado_mean_prob_138,0.159204,0.18900454,0.17410427,0.327,0.008034322,0.007504595,0.007769459
# tornado_mean_prob_computed_no_sv_219,0.16171832,0.20553061,0.18362448,0.7364,0.008005749,0.007426741,0.007716245
# tornado_mean_prob_computed_220,0.16545503,0.20286459,0.18415982,0.7302,0.007989318,0.007419265,0.0077042915
# tornado_mean_prob_computed_partial_climatology_227,0.17536521,0.20901904,0.19219212,0.6688,0.007898663,0.007324978,0.0076118205
# tornado_mean_prob_computed_climatology_253,0.17452854,0.20430718,0.18941787,0.8276,0.00786058,0.007276659,0.0075686197
# tornado_mean_prob_computed_climatology_blurs_910,0.1757724,0.20646888,0.19112064,0.4892,0.007814151,0.007253217,0.0075336844
# tornado_mean_prob_computed_climatology_grads_1348,0.17149976,0.21032669,0.19091323,0.2038,0.0078037414,0.007248117,0.0075259292
# tornado_mean_prob_computed_climatology_blurs_grads_2005,0.16878782,0.21904363,0.19391572,0.0156,0.007761531,0.007221429,0.00749148
# tornado_mean_prob_computed_climatology_prior_next_hrs_691,0.17914766,0.22008976,0.19961871,0.1554,0.007812941,0.007235485,0.0075242133
# tornado_mean_prob_computed_climatology_3hr_1567,0.17648162,0.21322876,0.19485518,0.398,0.00779421,0.007212169,0.0075031896
# tornado_full_13831,0.17057654,0.20169412,0.18613532,1.0,0.007804586,0.0072749276,0.0075397566
# tornado_mean_prob_computed_climatology_blurs_910,0.1757724,0.20646888,0.19112064,0.4892,0.007814151,0.007253217,0.0075336844
# wind_mean_prob_computed_climatology_blurs_910,0.37542477,0.4078922,0.39165848,0.0,0.046093278,0.04336693,0.044730105
# hail_mean_prob_computed_climatology_blurs_910,0.2862824,0.32148322,0.3038828,0.088,0.02576595,0.024120446,0.024943199
# tornado_mean_prob_computed_climatology_blurs_910_before_20200523,0.15995778,0.18939424,0.174676,0.0326,0.007999139,0.0074716797,0.0077354093
# wind_mean_prob_computed_climatology_blurs_910_before_20200523,0.3590528,0.39136642,0.37520963,0.0,0.04710717,0.044442534,0.045774855
# hail_mean_prob_computed_climatology_blurs_910_before_20200523,0.26908073,0.30371615,0.28639844,0.0,0.026199095,0.024574956,0.025387026
# tornado_full_13831,0.17057654,0.20169412,0.18613532,1.0,0.007804586,0.0072749276,0.0075397566
# wind_full_13831,0.38753828,0.42426723,0.40590274,1.0,0.045696996,0.04282092,0.04425896
# hail_full_13831,0.28860843,0.32727647,0.30794245,1.0,0.025630055,0.02400869,0.024819372
# tornado_day,0.15328296,0.17654984,0.1649164,0.0002,0.008017649,0.0074880845,0.0077528665
# wind_day,0.37739006,0.41358942,0.39548975,0.3484,0.046214674,0.04356108,0.044887878
# hail_day,0.2791898,0.3038601,0.29152495,0.006,0.026050145,0.024470204,0.025260175
# sig_tornado_day,0.09477509,0.10064449,0.09770979,,0.0016689355,0.0015669786,0.0016179571
# tornado_wind_hail_mean_relative_to_reference_mean_prob_computed_climatology_blurs_910,0.99704784,0.98912525,0.9928358,0.444,1.0050665,1.0048072,1.004943
# tornado_wind_hail_mean_relative_to_reference_mean_prob_computed_climatology_blurs_910_before_20200523,0.93544406,0.9405069,0.9381407,0.0,1.0208269,1.0245882,1.0226423
# tornado_wind_hail_mean_relative_to_reference_full_13831,1.0,1.0,1.0,1.0,1.0,1.0,1.0
# tornado_wind_hail_mean_relative_to_reference_day,0.95083785,0.9380804,0.9440024,0.0,1.0132352,1.0171195,1.0151097
