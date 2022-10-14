import Dates
import Random
import Printf
import Statistics

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
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


function do_it(forecasts; suffix = "", run_hour)

  println("************ $(run_hour)z$(suffix) ************")

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  println("$(length(test_forecasts)) unfiltered test forecasts")
  test_forecasts = filter(forecast -> forecast.run_hour == run_hour, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts")

  # We don't have storm events past this time.
  cutoff = Dates.DateTime(2022, 6, 1, 12)

  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts before the event data cutoff date") #

  event_name_to_day_labeler = Dict(
    "tornado" => TrainingShared.event_name_to_day_labeler["tornado"]
  )

  # rm("day_ablation_$(length(test_forecasts))_test_forecasts$(suffix)_$(run_hour)z"; recursive = true)

  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      test_forecasts;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "day_ablation_$(length(test_forecasts))_test_forecasts$(suffix)_$(run_hour)z",
    );
  # We're just looking at Day 1 forecasts, so run times will be unique between forecasts.
  run_times = Serialization.deserialize("day_ablation_test_forecasts_$(run_hour)z/run_times.serialized")
  @assert length(unique(run_times)) == length(test_forecasts)
  @assert sort(unique(run_times)) == sort(map(Forecast.run_utc_datetime, test_forecasts))

  println(sort(unique(run_times)))

  function do_bootstraps(X, y, run_times, nbootstraps)
    nforecasts = unique(length(run_times))

    @assert length(y) == size(X,2)
    @assert length(y) / nforecasts == round(length(y) / nforecasts)
    ndata_per_forecast = length(y) ÷ nforecasts
    @assert run_times[ndata_per_forecast*10] == run_times[ndata_per_forecast*10 - 1]
    @assert run_times[ndata_per_forecast*10] != run_times[ndata_per_forecast*10 + 1]

    # Use same bootstraps across all predictors
    # Unnecessary for large nbootstraps, but increases the validity of comparisons for smaller nbootstraps
    rng = Random.MersenneTwister(12345)
    bootstrap_forecast_iss = map(_ -> rand(rng, 1:nforecasts, nforecasts), 1:nbootstraps)

    data_is = Vector{Int64}(undef, length(y))
    for prediction_i in 1:size(X,2)
      model_name = HREFPredictionAblations.models[prediction_i]

      au_pr_bootstraps = map(1:nbootstraps) do bootstrap_i
        bootstrap_forecast_is = bootstrap_forecast_iss[bootstrap_i]
        Threads.@threads for fcst_i in 1:nforecasts
          bs_fcst_i = bootstrap_forecast_iss[fcst_i]
          data_is[fcst_i*(nforecasts-1)+1 : fcst_i*nforecasts] = bs_fcst_i*(nforecasts-1)+1 : bs_fcst_i*nforecasts
        end

        Metrics.area_under_pr_curve_fast(view(X, prediction_i, data_is), view(y, data_is), view(weights, data_is); bin_count = 1000)
      end

      au_pr = Metrics.area_under_pr_curve_fast(view(X, prediction_i, :), y, weights; bin_count = 1000)
      println("$model_name,$au_pr,$(Statistics.quantile(au_pr_bootstraps, 0.025)),$(Statistics.quantile(au_pr_bootstraps, 0.975))")
    end
  end

  do_bootstraps(X, Ys["tornado"], run_times, 10)
end

# function only_forecasts_with_runtimes(reference_forecasts, forecasts_to_filter)
#   reference_runtimes = Set(Forecasts.run_utc_datetime.(reference_forecasts))

#   filter(fcst -> Forecasts.run_utc_datetime(fcst) in reference_runtimes, forecasts_to_filter)
# end


# Going to use all the HREF forecasts

ablation_model_names = map(m -> m[1], HREFPredictionAblations.models)

1 in TASKS && do_it(HREFPredictionAblations.forecasts_day_spc_calibrated(); run_hour = 0)
2 in TASKS && do_it(HREFPredictionAblations.forecasts_day_spc_calibrated(); run_hour = 12)

# Absolutely calibrated should produce the same result
3 in TASKS && do_it(HREFPredictionAblations.forecasts_day(); run_hour = 0,  suffix = "_absolutely_calibrated")
4 in TASKS && do_it(HREFPredictionAblations.forecasts_day(); run_hour = 12, suffix = "_absolutely_calibrated")


# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[13] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[15] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[14] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[16] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[17] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[18] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[19] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[20] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
