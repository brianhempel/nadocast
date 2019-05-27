module StackedHREFSREF

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import ConcatForecasts
import PredictionForecasts
import ResampleForecasts

push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF

push!(LOAD_PATH, (@__DIR__) * "/../sref_mid_2018_forward")
import SREF


HREF_MODEL_PATH = (@__DIR__) * "/../href_mid_2018_forward/gbdt_f1-36_2019-03-28T13.34.42.186/99_trees_annealing_round_1_loss_0.0012652115.model"
SREF_MODEL_PATH = (@__DIR__) * "/../sref_mid_2018_forward/gbdt_f1-39_2019-03-26T00.59.57.772/78_trees_loss_0.001402743.model"

# The SREF run is always either 3 hours newer or 3 hours older than the HREF run.

# See the forecast_scheduling spreadsheet.

# Nadocast run hours 0-1z, 6-7z,  12-13z, 18-19z use SREF newer than HREF but older than the Nadocast run hour (HRRR).
# Nadocast run hours 2-5z, 8-11z, 14-17z, 20-23z use HREF newer than SREF but older than the Nadocast run hour (HRRR).

# To avoid unnecessary duplication, we therefore only need Nadocast run hours 0,2,6,8,12,14,18,20.

NADOCAST_RUN_HOURS_WHERE_NEW_HREF_OR_SREF_RUN_AVAILABLE = [0,2,6,8,12,14,18,20]
NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN                    = [0, 6, 12, 18]

MINUTE = 60
HOUR   = 60*MINUTE

_forecasts = []
_get_feature_engineered_data = nothing

function forecasts_with_href_newer_than_sref()
  filter(forecasts()) do forecast
    !(forecast.run_hour in NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN)
  end
end

function forecasts_with_sref_newer_than_href()
  filter(forecasts()) do forecast
    forecast.run_hour in NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN
  end
end

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
  else
    _forecasts
  end
end

function example_forecast()
  forecasts()[1]
end

function grid()
  Forecasts.grid(example_forecast())
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

function get_feature_engineered_data(forecast, data)
  _get_feature_engineered_data(forecast, data)
end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts
  global _get_feature_engineered_data

  _forecasts = []

  all_href_forecasts = HREF.forecasts()
  href_bin_splits, href_trees = MemoryConstrainedTreeBoosting.load(HREF_MODEL_PATH)
  href_prediction_forecasts, _, _, href_get_feature_engineered_prediction_data =
    PredictionForecasts.forecasts_example_forecast_grid_get_feature_engineered_data(
      all_href_forecasts,
      HREF.vector_wind_layers,
      HREF.get_feature_engineered_data,
      (href_data -> MemoryConstrainedTreeBoosting.predict(href_data, href_bin_splits, href_trees))
    )

  all_sref_forecasts = SREF.forecasts()
  sref_bin_splits, sref_trees = MemoryConstrainedTreeBoosting.load(SREF_MODEL_PATH)
  sref_prediction_forecasts, _, _, sref_get_feature_engineered_prediction_data =
    PredictionForecasts.forecasts_example_forecast_grid_get_feature_engineered_data(
      all_sref_forecasts,
      SREF.vector_wind_layers,
      SREF.get_feature_engineered_data,
      (sref_data -> MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees))
    )

  sref_upsampled_prediction_forecasts, _, _, sref_get_upsampled_feature_engineered_prediction_data =
    ResampleForecasts.forecasts_example_forecast_grid_get_feature_engineered_data(
      sref_prediction_forecasts,
      sref_get_feature_engineered_prediction_data,
      Grids.get_interpolating_upsampler(SREF.grid(), HREF.grid()),
      HREF.grid()
    )


  # Index to avoid O(n^2)

  run_time_seconds_to_href_forecasts = Dict{Int64,Vector{Forecasts.Forecast}}()

  for href_forecast in href_prediction_forecasts
    run_time = Forecasts.run_time_in_seconds_since_epoch_utc(href_forecast)
    href_forecasts_at_run_time = get(run_time_seconds_to_href_forecasts, run_time, Forecasts.Forecast[])
    push!(href_forecasts_at_run_time, href_forecast)
    run_time_seconds_to_href_forecasts[run_time] = href_forecasts_at_run_time
  end

  run_time_seconds_to_sref_forecasts = Dict{Int64,Vector{Forecasts.Forecast}}()

  for sref_forecast in sref_upsampled_prediction_forecasts
    run_time = Forecasts.run_time_in_seconds_since_epoch_utc(sref_forecast)
    sref_forecasts_at_run_time = get(run_time_seconds_to_sref_forecasts, run_time, Forecasts.Forecast[])
    push!(sref_forecasts_at_run_time, sref_forecast)
    run_time_seconds_to_sref_forecasts[run_time] = sref_forecasts_at_run_time
  end

  paired_forecasts                = []
  nadocast_run_and_forecast_times = []

  run_date = Dates.Date(2018, 6, 1)
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in NADOCAST_RUN_HOURS_WHERE_NEW_HREF_OR_SREF_RUN_AVAILABLE
      if run_hour in NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN
        href_delay_hours = 6 # Nadocast run hour 0,6,12,18z uses 18,0,6,12z HREF respectively
        sref_delay_hours = 3 # Nadocast run hour 0,6,12,18z uses 21,3,9,15z SREF respectively
      else
        href_delay_hours = 2 # Nadocast run hour 2,8,14,20z uses 0,6,12,18z HREF respectively
        sref_delay_hours = 5 # Nadocast run hour 2,8,14,20z uses 21,3,9,15z SREF respectively
      end

      run_time_in_seconds_since_epoch_utc = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      href_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - href_delay_hours*HOUR
      sref_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - sref_delay_hours*HOUR

      hrefs_for_run_time = get(run_time_seconds_to_href_forecasts, href_run_time_in_seconds_since_epoch_utc, Forecasts.Forecast[])
      srefs_for_run_time = get(run_time_seconds_to_sref_forecasts, sref_run_time_in_seconds_since_epoch_utc, Forecasts.Forecast[])

      for forecast_hour in 0:36 # Actually the furthest we can go out is 34 hours, shortest is 30. We have 3 time lagged HRRRs up through +16, so +17 is the first we would use.
        valid_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc + forecast_hour*HOUR

        perhaps_href_forecast = filter(href_forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast), hrefs_for_run_time)
        perhaps_sref_forecast = filter(sref_forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast), srefs_for_run_time)

        if length(perhaps_href_forecast) >= 2
          error("shouldn't have two matching href forecasts!")
        elseif length(perhaps_sref_forecast) >= 2
          error("shouldn't have two matching sref forecasts!")
        elseif length(perhaps_href_forecast) == 1 && length(perhaps_sref_forecast) == 1
          push!(paired_forecasts, (perhaps_href_forecast[1], perhaps_sref_forecast[1]))
          push!(nadocast_run_and_forecast_times, (run_year, run_month, run_day, run_hour, forecast_hour))
        end
      end
    end

    run_date += Dates.Day(1)
  end

  stacked_href_sref_prediction_forecasts, _, _, get_stacked_feature_engineered_data =
    ConcatForecasts.forecasts_example_forecast_grid_get_feature_engineered_data(
      paired_forecasts,
      (href_get_feature_engineered_prediction_data, sref_get_upsampled_feature_engineered_prediction_data)
    )

  _get_feature_engineered_data = get_stacked_feature_engineered_data

  for (stacked_href_sref_prediction_forecast, (run_year, run_month, run_day, run_hour, forecast_hour)) in Iterators.zip(stacked_href_sref_prediction_forecasts, nadocast_run_and_forecast_times)
    stacked_href_sref_prediction_forecast.run_year      = run_year
    stacked_href_sref_prediction_forecast.run_month     = run_month
    stacked_href_sref_prediction_forecast.run_day       = run_day
    stacked_href_sref_prediction_forecast.run_hour      = run_hour
    stacked_href_sref_prediction_forecast.forecast_hour = forecast_hour
  end

  _forecasts = stacked_href_sref_prediction_forecasts

  _forecasts
end

end # module StackedHREFSREF