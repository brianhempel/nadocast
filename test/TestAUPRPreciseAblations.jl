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

const nbootstraps = parse(Int64, get(ENV, "NBOOTSTRAPS", "10000")) # 10,000 takes ~10hr

function two_sided_bootstrap_p_value_paired(bootstraps_1, bootstraps_2)
  nbootstraps = length(bootstraps_1)
  Float32(min(
    count(bootstraps_1 .<= bootstraps_2) / (nbootstraps / 2),
    count(bootstraps_1 .>= bootstraps_2) / (nbootstraps / 2),
    1.0
  ))
end

function do_it(forecasts; suffix = "")

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

  event_name_to_day_labeler = Dict(
    "tornado" => TrainingShared.event_name_to_day_labeler["tornado"]
  )

  # rm("day_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z"; recursive = true)
  # rm("day_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z"; recursive = true)

  X_0z, Ys_0z, weights_0z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_0z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times_0z = Serialization.deserialize("day_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z/run_times.serialized")
  @assert length(unique(run_times_0z)) == length(test_forecasts_0z)
  @assert sort(unique(run_times_0z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_0z))

  X_12z, Ys_12z, weights_12z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_12z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times_12z = Serialization.deserialize("day_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z/run_times.serialized")
  @assert length(unique(run_times_12z)) == length(test_forecasts_12z)
  @assert sort(unique(run_times_12z)) == sort(map(Forecasts.run_utc_datetime, test_forecasts_12z))

  @assert run_times_0z == (run_times_12z .- Dates.Hour(12))

  println(sort(unique(run_times_0z)))

  y_0z, y_12z = Ys_0z["tornado"], Ys_12z["tornado"]

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
    "p au_pr_mean > au_pr_model_1 (one-sided)",
    "p au_pr_mean > au_pr_model_2 (one-sided)",
    "p au_pr_mean > au_pr_model_3 (one-sided)",
    "p au_pr_mean ≠ au_pr_model_1 (two-sided)",
    "p au_pr_mean ≠ au_pr_model_2 (two-sided)",
    "p au_pr_mean ≠ au_pr_model_3 (two-sided)",
    "logloss_0z",
    "logloss_12z",
    "logloss_mean",
  ]
  println(join(row, ","))
  total_0z_weight  = sum(weights_0z)
  total_12z_weight = sum(weights_12z)
  for prediction_i in 1:nmodels
    model_name = HREFPredictionAblations.models[prediction_i][1]
    au_pr_0z   = Metrics.area_under_pr_curve_fast(view(X_0z,  :, prediction_i), y_0z,  weights_0z;  bin_count = 1000)
    au_pr_12z  = Metrics.area_under_pr_curve_fast(view(X_12z, :, prediction_i), y_12z, weights_12z; bin_count = 1000)
    au_pr_mean = (au_pr_0z + au_pr_12z) / 2
    logloss_0z   = sum(MemoryConstrainedTreeBoosting.logloss.(y_0z,  view(X_0z,  :, prediction_i)) .* weights_0z)  / total_0z_weight
    logloss_12z  = sum(MemoryConstrainedTreeBoosting.logloss.(y_12z, view(X_12z, :, prediction_i)) .* weights_12z) / total_12z_weight
    logloss_mean = (logloss_0z + logloss_12z) / 2
    row = Any[
      model_name,
      au_pr_0z,
      au_pr_12z,
      au_pr_mean,
      Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[1]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[2]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      Float32(count(model_au_pr_bootstraps_mean[prediction_i] .<= model_au_pr_bootstraps_mean[3]) / nbootstraps), # One-sided test https://blogs.sas.com/content/iml/2011/11/02/how-to-compute-p-values-for-a-bootstrap-distribution.html
      two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[1]),
      two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[2]),
      two_sided_bootstrap_p_value_paired(model_au_pr_bootstraps_mean[prediction_i], model_au_pr_bootstraps_mean[3]),
      logloss_0z,
      logloss_12z,
      logloss_mean,
    ]
    println(join(row, ","))
  end
end

function do_it_hourly(forecasts; suffix = "")

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

  event_name_to_labeler = Dict(
    "tornado" => TrainingShared.event_name_to_labeler["tornado"]
  )

  # rm("hourly_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z"; recursive = true)
  # rm("hourly_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z"; recursive = true)

  X_0z, Ys_0z, weights_0z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_0z;
      event_name_to_labeler = event_name_to_labeler,
      save_dir = "hourly_ablation_$(length(test_forecasts_0z))_test_forecasts$(suffix)_0z",
    );

  X_12z, Ys_12z, weights_12z =
    TrainingShared.get_data_labels_weights(
      test_forecasts_12z;
      event_name_to_labeler = event_name_to_labeler,
      save_dir = "hourly_ablation_$(length(test_forecasts_12z))_test_forecasts$(suffix)_12z",
    );

  y_0z, y_12z = Ys_0z["tornado"], Ys_12z["tornado"]

  nmodels = size(X_0z,2)

  row = Any[
    "model_name",
    "au_pr_0z",
    "au_pr_12z",
    "au_pr_mean",
  ]
  println(join(row, ","))
  for prediction_i in 1:nmodels
    model_name = HREFPredictionAblations.models[prediction_i][1]
    au_pr_0z   = Metrics.area_under_pr_curve_fast(view(X_0z,  :, prediction_i), y_0z,  weights_0z;  bin_count = 1000)
    au_pr_12z  = Metrics.area_under_pr_curve_fast(view(X_12z, :, prediction_i), y_12z, weights_12z; bin_count = 1000)
    au_pr_mean = (au_pr_0z + au_pr_12z) / 2
    row = Any[
      model_name,
      au_pr_0z,
      au_pr_12z,
      au_pr_mean,
    ]
    println(join(row, ","))
  end
end

# function only_forecasts_with_runtimes(reference_forecasts, forecasts_to_filter)
#   reference_runtimes = Set(Forecasts.run_utc_datetime.(reference_forecasts))

#   filter(fcst -> Forecasts.run_utc_datetime(fcst) in reference_runtimes, forecasts_to_filter)
# end


# Going to use all the HREF forecasts, 2018-7-1 to 2022-5-31.

TASKS = eval(Meta.parse(get(ENV, "TASKS", "")))
if isnothing(TASKS)
  TASKS = typemin(Int64):typemax(Int64)
end

1 in TASKS && do_it(HREFPredictionAblations.forecasts_day_spc_calibrated())
# 807 unfiltered test forecasts
# 183 0z test forecasts before the event data cutoff date
# 183 12z test forecasts before the event data cutoff date
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,p au_pr_mean > au_pr_model_1 (one-sided),p au_pr_mean > au_pr_model_2 (one-sided),p au_pr_mean > au_pr_model_3 (one-sided),p au_pr_mean ≠ au_pr_model_1 (two-sided),p au_pr_mean ≠ au_pr_model_2 (two-sided),p au_pr_mean ≠ au_pr_model_3 (two-sided),logloss_0z,logloss_12z,logloss_mean
# tornado_mean_58,0.16520426,0.20508213,0.1851432,1.0,0.8191,0.9854,1.0,0.3618,0.0292,0.009001203,0.008452371,0.008726787
# tornado_prob_80,0.17958304,0.21147676,0.19552991,0.1809,1.0,0.9775,0.3618,1.0,0.045,0.008867842,0.00835025,0.008609046
# tornado_mean_prob_138,0.18377462,0.2309311,0.20735286,0.0146,0.0225,1.0,0.0292,0.045,1.0,0.008777854,0.008275257,0.008526556
# tornado_mean_prob_computed_no_sv_219,0.16955514,0.24469897,0.20712706,0.0466,0.1555,0.401,0.0932,0.311,0.802,0.008786575,0.008215624,0.008501099
# tornado_mean_prob_computed_220,0.17677765,0.23604892,0.20641328,0.0382,0.1315,0.3781,0.0764,0.263,0.7562,0.008772889,0.008199918,0.008486403
# tornado_mean_prob_computed_partial_climatology_227,0.18325676,0.24207765,0.2126672,0.0051,0.0328,0.1895,0.0102,0.0656,0.379,0.008699691,0.008169423,0.008434556
# tornado_mean_prob_computed_climatology_253,0.19213827,0.24444655,0.21829242,0.0052,0.0054,0.0592,0.0104,0.0108,0.1184,0.008616757,0.008100594,0.008358676
# tornado_mean_prob_computed_climatology_blurs_910,0.19199078,0.24575341,0.2188721,0.0096,0.0197,0.0994,0.0192,0.0394,0.1988,0.008549996,0.008064838,0.008307417
# tornado_mean_prob_computed_climatology_grads_1348,0.1817587,0.24398394,0.21287131,0.0137,0.0596,0.2135,0.0274,0.1192,0.427,0.008591862,0.008092828,0.008342345
# tornado_mean_prob_computed_climatology_blurs_grads_2005,0.18374325,0.24477243,0.21425784,0.0154,0.0475,0.1797,0.0308,0.095,0.3594,0.008557143,0.008061813,0.008309478
# tornado_mean_prob_computed_climatology_prior_next_hrs_691,0.18686068,0.24971777,0.21828923,0.003,0.0163,0.0672,0.006,0.0326,0.1344,0.008646648,0.008083872,0.00836526
# tornado_mean_prob_computed_climatology_3hr_1567,0.18016264,0.23578782,0.20797524,0.0162,0.1075,0.35,0.0324,0.215,0.7,0.008616469,0.008043831,0.00833015
# tornado_full_13831,0.1830765,0.22834672,0.2057116,0.0376,0.1671,0.4131,0.0752,0.3342,0.8262,0.008582215,0.008129388,0.008355802

# Absolutely calibrated should produce the same result for AU-PR, not not logloss
2 in TASKS && do_it(HREFPredictionAblations.forecasts_day(); suffix = "_absolutely_calibrated")
# model_name,au_pr_0z,au_pr_12z,au_pr_mean,p au_pr_mean > au_pr_model_1 (one-sided),p au_pr_mean > au_pr_model_2 (one-sided),p au_pr_mean > au_pr_model_3 (one-sided),p au_pr_mean ≠ au_pr_model_1 (two-sided),p au_pr_mean ≠ au_pr_model_2 (two-sided),p au_pr_mean ≠ au_pr_model_3 (two-sided),logloss_0z,logloss_12z,logloss_mean
# tornado_mean_58,0.16521966,0.20509183,0.18515575,1.0,0.819,0.9854,1.0,0.362,0.0292,0.008801795,0.008223626,0.00851271
# tornado_prob_80,0.1795941,0.21147126,0.19553268,0.181,1.0,0.9774,0.362,1.0,0.0452,0.008681542,0.008151181,0.008416362
# tornado_mean_prob_138,0.18379883,0.23090757,0.2073532,0.0146,0.0226,1.0,0.0292,0.0452,1.0,0.008555891,0.008006548,0.00828122
# tornado_mean_prob_computed_no_sv_219,0.16956905,0.24465644,0.20711274,0.0468,0.1566,0.4016,0.0936,0.3132,0.8032,0.008558372,0.007927335,0.008242853
# tornado_mean_prob_computed_220,0.17679158,0.2360288,0.2064102,0.0383,0.1318,0.3785,0.0766,0.2636,0.757,0.008541047,0.007929334,0.00823519
# tornado_mean_prob_computed_partial_climatology_227,0.18324287,0.24204578,0.21264432,0.0051,0.033,0.1912,0.0102,0.066,0.3824,0.008431493,0.007835488,0.008133491
# tornado_mean_prob_computed_climatology_253,0.192143,0.2444256,0.2182843,0.0052,0.0054,0.0593,0.0104,0.0108,0.1186,0.008350832,0.0077634696,0.008057151
# tornado_mean_prob_computed_climatology_blurs_910,0.19199552,0.24578702,0.21889126,0.0096,0.0198,0.0993,0.0192,0.0396,0.1986,0.008288683,0.0077467924,0.008017737
# tornado_mean_prob_computed_climatology_grads_1348,0.18172297,0.24395801,0.2128405,0.0138,0.0598,0.2142,0.0276,0.1196,0.4284,0.008309526,0.0077439854,0.008026756
# tornado_mean_prob_computed_climatology_blurs_grads_2005,0.18374169,0.2447659,0.21425378,0.0156,0.0478,0.1797,0.0312,0.0956,0.3594,0.00825848,0.007730019,0.007994249
# tornado_mean_prob_computed_climatology_prior_next_hrs_691,0.18684013,0.24972838,0.21828425,0.0029,0.0163,0.0673,0.0058,0.0326,0.1346,0.008362837,0.007746257,0.008054547
# tornado_mean_prob_computed_climatology_3hr_1567,0.18013741,0.23575017,0.2079438,0.0162,0.1079,0.3505,0.0324,0.2158,0.701,0.008326078,0.0076826084,0.008004343
# tornado_full_13831,0.18307142,0.2283096,0.2056905,0.038,0.1681,0.4141,0.076,0.3362,0.8282,0.00829113,0.0077795587,0.008035344

3 in TASKS && do_it_hourly(HREFPredictionAblations.forecasts_calibrated())
# ************ ablations ************
# 27300 unfiltered test forecasts
# 6172 0z test forecasts before the event data cutoff date
# 6174 12z test forecasts before the event data cutoff date
# model_name,au_pr_0z,au_pr_12z,au_pr_mean
# tornado_mean_58,0.056471076,0.041277263,0.04887417
# tornado_prob_80,0.05819818,0.048271976,0.053235076
# tornado_mean_prob_138,0.0724223,0.054892294,0.0636573
# tornado_mean_prob_computed_no_sv_219,0.06786085,0.054882046,0.061371446
# tornado_mean_prob_computed_220,0.07049692,0.05771204,0.06410448
# tornado_mean_prob_computed_partial_climatology_227,0.067347035,0.05210603,0.059726533
# tornado_mean_prob_computed_climatology_253,0.076340534,0.055012826,0.06567668
# tornado_mean_prob_computed_climatology_blurs_910,0.08468938,0.05921395,0.071951665
# tornado_mean_prob_computed_climatology_grads_1348,0.070857614,0.066277474,0.068567544
# tornado_mean_prob_computed_climatology_blurs_grads_2005,0.07242646,0.06648052,0.06945349
# tornado_mean_prob_computed_climatology_prior_next_hrs_691,0.07835097,0.06561739,0.07198418
# tornado_mean_prob_computed_climatology_3hr_1567,0.08371321,0.058747128,0.07123017
# tornado_full_13831,0.07942711,0.060344998,0.06988605


# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[1] NBOOTSTRAPS=10 julia -t 16 --project=.. TestAUPRPreciseAblations.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[2] NBOOTSTRAPS=10 julia -t 16 --project=.. TestAUPRPreciseAblations.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[3] julia -t 16 --project=.. TestAUPRPreciseAblations.jl
