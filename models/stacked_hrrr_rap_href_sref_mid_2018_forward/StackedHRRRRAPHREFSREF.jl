module StackedHRRRRAPHREFSREF

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids
import StormEvents

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts

push!(LOAD_PATH, (@__DIR__) * "/../hrrr_mid_july_2016_forward")
import HRRR

push!(LOAD_PATH, (@__DIR__) * "/../rap_march_2014_forward")
import RAP

push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF

push!(LOAD_PATH, (@__DIR__) * "/../sref_mid_2018_forward")
import SREF

RAP_MODEL_PATH  = (@__DIR__) * "/../rap_march_2014_forward/gbdt_f12_2019-04-17T19.27.16.893/568_trees_loss_0.0012037802.model"
HRRR_MODEL_PATH = (@__DIR__) * "/../hrrr_mid_july_2016_forward/gbdt_f12_2019-05-04T13.05.05.929/157_trees_loss_0.0011697214.model"
HREF_MODEL_PATH = (@__DIR__) * "/../href_mid_2018_forward/gbdt_f1-36_2019-03-28T13.34.42.186/99_trees_annealing_round_1_loss_0.0012652115.model"
SREF_MODEL_PATH = (@__DIR__) * "/../sref_mid_2018_forward/gbdt_f1-39_2019-03-26T00.59.57.772/78_trees_loss_0.001402743.model"


# Nadocast run hours 0-1z, 6-7z,  12-13z, 18-19z use SREF newer than HREF but older than the Nadocast run hour (HRRR).
# Nadocast run hours 2-5z, 8-11z, 14-17z, 20-23z use HREF newer than SREF but older than the Nadocast run hour (HRRR).

# Copy of the forecast_scheduling spreadsheet.
# Run hours for nadocast, hrrr, rap, href, sref
# nadocast run hour == HRRR run hour. RAP is usually the same too, except the 0Z and 12Z raps are late.
FORECAST_SCHEDULE =
  [ (0,  0,  23, 18, 21)
  , (1,  1,  1,  18, 21)
  , (2,  2,  2,  0,  21)
  , (3,  3,  3,  0,  21)
  , (4,  4,  4,  0,  21)
  , (5,  5,  5,  0,  21)
  , (6,  6,  6,  0,  3)
  , (7,  7,  7,  0,  3)
  , (8,  8,  8,  6,  3)
  , (9,  9,  9,  6,  3)
  , (10, 10, 10, 6,  3)
  , (11, 11, 11, 6,  3)
  , (12, 12, 11, 6,  9)
  , (13, 13, 13, 6,  9)
  , (14, 14, 14, 12, 9)
  , (15, 15, 15, 12, 9)
  , (16, 16, 16, 12, 9)
  , (17, 17, 17, 12, 9)
  , (18, 18, 18, 12, 15)
  , (19, 19, 19, 12, 15)
  , (20, 20, 20, 18, 15)
  , (21, 21, 21, 18, 15)
  , (22, 22, 22, 18, 15)
  , (23, 23, 23, 18, 15)
  ]


MINUTE = 60
HOUR   = 60*MINUTE

_forecasts = []

# function forecasts_with_href_newer_than_sref()
#   filter(forecasts()) do forecast
#     !(forecast.run_hour in NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN)
#   end
# end
#
# function forecasts_with_sref_newer_than_href()
#   filter(forecasts()) do forecast
#     forecast.run_hour in NADOCAST_RUN_HOURS_WITH_NEW_SREF_RUN
#   end
# end

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
  HREF.grid()
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

# function get_feature_engineered_data(forecast, base_data)
#   global _get_stacked_feature_engineered_data
#
#   stacked_predictions_data = _get_stacked_feature_engineered_data(forecast, base_data)
#
#   data_count = size(stacked_predictions_data, 1)
#
#   run_hour, hrrr_run_hour, rap_run_hour, href_run_hour, sref_run_hour =
#     filter(associated_run_hours -> associated_run_hours[1] == forecast.run_hour, FORECAST_SCHEDULE)[1]
#
#   href_age_hours = forecast.forecast_hour + (href_run_hour > run_hour ? (run_hour + 24) - href_run_hour : run_hour - href_run_hour)
#   sref_age_hours = forecast.forecast_hour + (sref_run_hour > run_hour ? (run_hour + 24) - sref_run_hour : run_hour - sref_run_hour)
#
#   hcat(stacked_predictions_data, fill(Float32(href_age_hours), data_count), fill(Float32(sref_age_hours), data_count))
# end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts

  _forecasts = []

  storm_event_hours_set = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_events(), 30*MINUTE)


  hrrr_bin_splits, hrrr_trees = MemoryConstrainedTreeBoosting.load(HRRR_MODEL_PATH)
  hrrr_prediction_forecasts =
    PredictionForecasts.feature_engineered_prediction_forecasts(
      HRRR.feature_engineered_forecasts(),
      (hrrr_data -> MemoryConstrainedTreeBoosting.predict(hrrr_data, hrrr_bin_splits, hrrr_trees));
      base_forecasts_no_feature_engineering = HRRR.forecasts(),
      vector_wind_layers = HRRR.vector_wind_layers
    )

  hrrr_upsampled_prediction_forecasts =
    ForecastCombinators.resample_forecasts(
      hrrr_prediction_forecasts,
      Grids.get_upsampler,
      HREF.grid()
    )


  rap_bin_splits, rap_trees = MemoryConstrainedTreeBoosting.load(RAP_MODEL_PATH)
  rap_prediction_forecasts =
    PredictionForecasts.feature_engineered_prediction_forecasts(
      RAP.feature_engineered_forecasts(),
      (rap_data -> MemoryConstrainedTreeBoosting.predict(rap_data, rap_bin_splits, rap_trees));
      base_forecasts_no_feature_engineering = RAP.forecasts(),
      vector_wind_layers = RAP.vector_wind_layers
    )

  rap_upsampled_prediction_forecasts =
    ForecastCombinators.resample_forecasts(
      rap_prediction_forecasts,
      Grids.get_upsampler,
      HREF.grid()
    )


  href_bin_splits, href_trees = MemoryConstrainedTreeBoosting.load(HREF_MODEL_PATH)
  href_prediction_forecasts =
    PredictionForecasts.feature_engineered_prediction_forecasts(
      HREF.feature_engineered_forecasts(),
      (href_data -> MemoryConstrainedTreeBoosting.predict(href_data, href_bin_splits, href_trees));
      base_forecasts_no_feature_engineering = HREF.forecasts(),
      vector_wind_layers = HREF.vector_wind_layers
    )


  sref_bin_splits, sref_trees = MemoryConstrainedTreeBoosting.load(SREF_MODEL_PATH)
  sref_prediction_forecasts =
    PredictionForecasts.feature_engineered_prediction_forecasts(
      SREF.feature_engineered_forecasts(),
      (sref_data -> MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees));
      base_forecasts_no_feature_engineering = SREF.forecasts(),
      vector_wind_layers = SREF.vector_wind_layers
    )

  sref_upsampled_prediction_forecasts =
    ForecastCombinators.resample_forecasts(
      sref_prediction_forecasts,
      Grids.get_interpolating_upsampler,
      HREF.grid()
    )


  # Index to avoid O(n^2)

  run_time_seconds_to_hrrr_forecasts = Forecasts.run_time_seconds_to_forecasts(hrrr_upsampled_prediction_forecasts)
  run_time_seconds_to_rap_forecasts  = Forecasts.run_time_seconds_to_forecasts(rap_upsampled_prediction_forecasts)
  run_time_seconds_to_href_forecasts = Forecasts.run_time_seconds_to_forecasts(href_prediction_forecasts)
  run_time_seconds_to_sref_forecasts = Forecasts.run_time_seconds_to_forecasts(sref_upsampled_prediction_forecasts)

  associated_forecasts            = []
  nadocast_run_and_forecast_times = []

  run_date = Dates.Date(2018, 6, 1)
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for (run_hour, hrrr_run_hour, rap_run_hour, href_run_hour, sref_run_hour) in FORECAST_SCHEDULE
      run_time_in_seconds_since_epoch_utc = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      hrrr_delay_hours = hrrr_run_hour > run_hour ? (run_hour + 24) - hrrr_run_hour : run_hour - hrrr_run_hour # HRRR run hour == nadocast run hour so this line is superfulous.
      rap_delay_hours  = rap_run_hour  > run_hour ? (run_hour + 24) - rap_run_hour  : run_hour - rap_run_hour
      href_delay_hours = href_run_hour > run_hour ? (run_hour + 24) - href_run_hour : run_hour - href_run_hour
      sref_delay_hours = sref_run_hour > run_hour ? (run_hour + 24) - sref_run_hour : run_hour - sref_run_hour

      hrrr_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - hrrr_delay_hours*HOUR
      rap_run_time_in_seconds_since_epoch_utc  = run_time_in_seconds_since_epoch_utc - rap_delay_hours*HOUR
      href_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - href_delay_hours*HOUR
      sref_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - sref_delay_hours*HOUR

      hrrrs_for_run_time                 = get(run_time_seconds_to_hrrr_forecasts, hrrr_run_time_in_seconds_since_epoch_utc - 0*HOUR, Forecasts.Forecast[])
      hrrrs_for_run_time_minus_one_hour  = get(run_time_seconds_to_hrrr_forecasts, hrrr_run_time_in_seconds_since_epoch_utc - 1*HOUR, Forecasts.Forecast[])
      hrrrs_for_run_time_minus_two_hours = get(run_time_seconds_to_hrrr_forecasts, hrrr_run_time_in_seconds_since_epoch_utc - 2*HOUR, Forecasts.Forecast[])
      raps_for_run_time                  = get(run_time_seconds_to_rap_forecasts,  rap_run_time_in_seconds_since_epoch_utc  - 0*HOUR, Forecasts.Forecast[])
      raps_for_run_time_minus_one_hour   = get(run_time_seconds_to_rap_forecasts,  rap_run_time_in_seconds_since_epoch_utc  - 1*HOUR, Forecasts.Forecast[])
      raps_for_run_time_minus_two_hours  = get(run_time_seconds_to_rap_forecasts,  rap_run_time_in_seconds_since_epoch_utc  - 2*HOUR, Forecasts.Forecast[])
      hrefs_for_run_time                 = get(run_time_seconds_to_href_forecasts, href_run_time_in_seconds_since_epoch_utc, Forecasts.Forecast[])
      srefs_for_run_time                 = get(run_time_seconds_to_sref_forecasts, sref_run_time_in_seconds_since_epoch_utc, Forecasts.Forecast[])

      for forecast_hour in 1:16
        valid_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc + forecast_hour*HOUR

        perhaps_hrrr_forecast                 = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), hrrrs_for_run_time)
        perhaps_hrrr_forecast_minus_one_hour  = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), hrrrs_for_run_time_minus_one_hour)
        perhaps_hrrr_forecast_minus_two_hours = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), hrrrs_for_run_time_minus_two_hours)
        perhaps_rap_forecast                  = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), raps_for_run_time)
        perhaps_rap_forecast_minus_one_hour   = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), raps_for_run_time_minus_one_hour)
        perhaps_rap_forecast_minus_two_hours  = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), raps_for_run_time_minus_two_hours)
        perhaps_href_forecast                 = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), hrefs_for_run_time)
        perhaps_sref_forecast                 = filter(forecast -> valid_time_in_seconds_since_epoch_utc == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), srefs_for_run_time)

        if length(perhaps_hrrr_forecast) >= 2
          error("shouldn't have two matching hrrr_forecast forecasts!")
        elseif length(perhaps_hrrr_forecast_minus_one_hour) >= 2
          error("shouldn't have two matching hrrr_forecast_minus_one_hour forecasts!")
        elseif length(perhaps_hrrr_forecast_minus_two_hours) >= 2
          error("shouldn't have two matching hrrr_forecast_minus_two_hours forecasts!")
        elseif length(perhaps_rap_forecast) >= 2
          error("shouldn't have two matching rap_forecast forecasts!")
        elseif length(perhaps_rap_forecast_minus_one_hour) >= 2
          error("shouldn't have two matching rap_forecast_minus_one_hour forecasts!")
        elseif length(perhaps_rap_forecast_minus_two_hours) >= 2
          error("shouldn't have two matching rap_forecast_minus_two_hours forecasts!")
        elseif length(perhaps_href_forecast) >= 2
          error("shouldn't have two matching href forecasts!")
        elseif length(perhaps_sref_forecast) >= 2
          error("shouldn't have two matching sref forecasts!")
        elseif length.([perhaps_hrrr_forecast, perhaps_hrrr_forecast_minus_one_hour, perhaps_hrrr_forecast_minus_two_hours, perhaps_rap_forecast, perhaps_rap_forecast_minus_one_hour, perhaps_rap_forecast_minus_two_hours, perhaps_href_forecast, perhaps_sref_forecast]) == [1,1,1, 1,1,1, 1, 1]
          push!(
            associated_forecasts,
            ( perhaps_hrrr_forecast[1]
            , perhaps_hrrr_forecast_minus_one_hour[1]
            , perhaps_hrrr_forecast_minus_two_hours[1]
            , perhaps_rap_forecast[1]
            , perhaps_rap_forecast_minus_one_hour[1]
            , perhaps_rap_forecast_minus_two_hours[1]
            , perhaps_href_forecast[1]
            , perhaps_sref_forecast[1]
            )
          )
          push!(nadocast_run_and_forecast_times, (run_year, run_month, run_day, run_hour, forecast_hour))
        elseif forecast_hour == 11 && valid_time_in_seconds_since_epoch_utc in storm_event_hours_set
          println(((run_year, run_month, run_day, run_hour, forecast_hour), length.([perhaps_hrrr_forecast, perhaps_hrrr_forecast_minus_one_hour, perhaps_hrrr_forecast_minus_two_hours, perhaps_rap_forecast, perhaps_rap_forecast_minus_one_hour, perhaps_rap_forecast_minus_two_hours, perhaps_href_forecast, perhaps_sref_forecast])))
        end
      end
    end

    run_date += Dates.Day(1)
  end

  stacked_hrrr_rap_href_sref_prediction_forecasts = ForecastCombinators.concat_forecasts(associated_forecasts)

  # Add columns with the HREF and SREF age.
  # _get_stacked_feature_engineered_data = get_stacked_feature_engineered_data

  for (stacked_hrrr_rap_href_sref_prediction_forecast, (run_year, run_month, run_day, run_hour, forecast_hour)) in Iterators.zip(stacked_hrrr_rap_href_sref_prediction_forecasts, nadocast_run_and_forecast_times)
    stacked_hrrr_rap_href_sref_prediction_forecast.run_year      = run_year
    stacked_hrrr_rap_href_sref_prediction_forecast.run_month     = run_month
    stacked_hrrr_rap_href_sref_prediction_forecast.run_day       = run_day
    stacked_hrrr_rap_href_sref_prediction_forecast.run_hour      = run_hour
    stacked_hrrr_rap_href_sref_prediction_forecast.forecast_hour = forecast_hour
  end

  _forecasts = stacked_hrrr_rap_href_sref_prediction_forecasts

  _forecasts
end

end # module StackedHRRRRAPHREFSREF