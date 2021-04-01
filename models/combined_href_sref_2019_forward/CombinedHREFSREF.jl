# Output is a 2-feature forecast: layer 1 is the HREF-based prediction, layer 2 is the SREF-based prediction
module CombinedHREFSREF

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts
import Climatology

push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF

push!(LOAD_PATH, (@__DIR__) * "/../sref_mid_2018_forward")
import SREF


MINUTE = 60
HOUR   = 60*MINUTE

# Forecast run time is always the newer forecast.

_forecasts_href_newer = []
_forecasts_sref_newer = []

# SREF 3 hours behind HREF
function forecasts_href_newer()
  if isempty(_forecasts_href_newer)
    reload_forecasts()
    _forecasts_href_newer
  else
    _forecasts_href_newer
  end
end

# HREF 3 hours behind SREF
function forecasts_sref_newer()
  if isempty(_forecasts_sref_newer)
    reload_forecasts()
    _forecasts_sref_newer
  else
    _forecasts_sref_newer
  end
end

function example_forecast()
  forecasts()[1]
end

function grid()
  HREF.grid()
end

function forecasts_href_newer_with_blurs_and_hour_climatology()
  if isempty(_forecasts_href_newer_with_blurs_and_hour_climatology)
    reload_forecasts()
    _forecasts_href_newer_with_blurs_and_hour_climatology
  else
    _forecasts_href_newer_with_blurs_and_hour_climatology
  end
end

function forecasts_sref_newer_with_blurs_and_hour_climatology()
  if isempty(_forecasts_sref_newer_with_blurs_and_hour_climatology)
    reload_forecasts()
    _forecasts_sref_newer_with_blurs_and_hour_climatology
  else
    _forecasts_sref_newer_with_blurs_and_hour_climatology
  end

end

blur_radii = [10, 15, 25, 35, 50, 70, 100]

function with_blurs_and_hour_climatology(href_sref_prediction_forecasts)
  grid = href_sref_prediction_forecasts[1].grid

  blur_radii_grid_is = map(miles -> Grids.radius_grid_is(grid, miles), Float64.(blur_radii))

  inventory_transformer(base_forecast, base_inventory) = begin
    href_line = base_inventory[1]
    sref_line = base_inventory[2]

    new_inventory = [
      href_line,
      sref_line
    ]

    for miles in blur_radii
      push!(new_inventory, revise_with_feature_engineering(href_line, "$(miles)mi mean"))
      push!(new_inventory, revise_with_feature_engineering(sref_line, "$(miles)mi mean"))
    end

    push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, "forecast_hour",                                                                        "calculated", "hour fcst", "", ""))
    push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_tornado_probability_feature(grid)[1],                         "calculated", "hour fcst", "", ""))
    push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_severe_probability_feature(grid)[1],                          "calculated", "hour fcst", "", ""))
    push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_tornado_given_severe_probability_feature(grid)[1],            "calculated", "hour fcst", "", ""))
    push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)[1], "calculated", "hour fcst", "", ""))

    new_inventory
  end

  data_transformer(base_forecast, base_data) = begin
    point_count               = size(base_data, 1)

    out = Array{Float32}(undef, (point_count, 2 + 2*length(blur_radii) + 5))

    href_data = @view base_data[:, 1]
    sref_data = @view base_data[:, 2]

    feature_i = 1
    out[:, feature_i] = href_data
    feature_i += 1
    out[:, feature_i] = sref_data

    for blur_i in 1:length(blur_radii)
      feature_i += 1
      out[:, feature_i] = FeatureEngineeringShared.meanify_threaded(href_data, blur_radii_grid_is[blur_i])
      feature_i += 1
      out[:, feature_i] = FeatureEngineeringShared.meanify_threaded(sref_data, blur_radii_grid_is[blur_i])
    end

    feature_i += 1
    out[:, feature_i] .= Float32(base_forecast.forecast_hour)
    feature_i += 1
    out[:, feature_i] = Climatology.forecast_hour_tornado_probability_feature(grid)[2](base_forecast)
    feature_i += 1
    out[:, feature_i] = Climatology.forecast_hour_severe_probability_feature(grid)[2](base_forecast)
    feature_i += 1
    out[:, feature_i] = Climatology.forecast_hour_tornado_given_severe_probability_feature(grid)[2](base_forecast)
    feature_i += 1
    out[:, feature_i] = Climatology.forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)[2](base_forecast)

    out
  end

  ForecastCombinators.map_forecasts(href_sref_prediction_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_href_newer
  global _forecasts_sref_newer
  global _forecasts_href_newer_with_blurs_and_hour_climatology
  global _forecasts_sref_newer_with_blurs_and_hour_climatology

  _forecasts_href_newer = []
  _forecasts_sref_newer = []

  href_forecasts = HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  href_predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2020-08-31T01.09.34.597/165_trees_loss_0.0010708775.model")
  href_predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2020-09-05T14.17.00.494/205_trees_loss_0.0011394698.model")
  href_predict_f24_to_f35 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2020-09-11T02.05.26.327/192_trees_loss_0.0011718384.model")

  href_predict(forecast, data) =
    if forecast.forecast_hour in 25:35
      href_predict_f24_to_f35(data)
    elseif forecast.forecast_hour in 24:24
      0.5 .* (href_predict_f24_to_f35(data) .+ href_predict_f13_to_f24(data))
    elseif forecast.forecast_hour in 14:23
      href_predict_f13_to_f24(data)
    elseif forecast.forecast_hour in 13:13
      0.5 .* (href_predict_f13_to_f24(data) .+ href_predict_f2_to_f13(data))
    elseif forecast.forecast_hour in 2:12
      href_predict_f2_to_f13(data)
    else
      error("HREF forecast hour $(forecast.forecast_hour) not in 2:35")
    end

  href_prediction_forecasts = PredictionForecasts.simple_prediction_forecasts(href_forecasts, href_predict)

  sref_forecasts = SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  sref_predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2020-08-01T06.24.27.615/177_trees_loss_0.0012903158.model")
  sref_predict_f12_to_f23 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2020-07-26T07.54.57.491/173_trees_loss_0.0013711528.model")
  sref_predict_f21_to_f38 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2020-08-16T18.14.40.282/147_trees_loss_0.00141136.model")

  sref_predict(forecast, data) = begin
    if forecast.forecast_hour in 24:38
      sref_predict_f21_to_f38(data)
    elseif forecast.forecast_hour in 21:23
      0.5 .* (sref_predict_f21_to_f38(data) .+ sref_predict_f12_to_f23(data))
    elseif forecast.forecast_hour in 14:20
      sref_predict_f12_to_f23(data)
    elseif forecast.forecast_hour in 12:13
      0.5 .* (sref_predict_f12_to_f23(data) .+ sref_predict_f2_to_f13(data))
    elseif forecast.forecast_hour in 2:11
      sref_predict_f2_to_f13(data)
    else
      error("SREF forecast hour $(forecast.forecast_hour) not in 2:38")
    end
  end

  sref_prediction_forecasts = PredictionForecasts.simple_prediction_forecasts(sref_forecasts, sref_predict)

  sref_upsampled_prediction_forecasts =
    ForecastCombinators.resample_forecasts(
      sref_prediction_forecasts,
      Grids.get_interpolating_upsampler,
      HREF.grid()
    )

  # Index to avoid O(n^2)

  run_time_seconds_to_href_forecasts = Forecasts.run_time_seconds_to_forecasts(href_prediction_forecasts)
  run_time_seconds_to_sref_forecasts = Forecasts.run_time_seconds_to_forecasts(sref_upsampled_prediction_forecasts)

  paired_href_newer = []
  paired_sref_newer = []

  run_date = Dates.Date(2019, 1, 9)
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in 0:3:21
      run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      hrefs_for_run_time = get(run_time_seconds_to_href_forecasts, run_time_seconds, Forecasts.Forecast[])
      srefs_for_run_time = get(run_time_seconds_to_sref_forecasts, run_time_seconds, Forecasts.Forecast[])
      hrefs_3hrs_earlier = get(run_time_seconds_to_href_forecasts, run_time_seconds - 3*HOUR, Forecasts.Forecast[])
      srefs_3hrs_earlier = get(run_time_seconds_to_sref_forecasts, run_time_seconds - 3*HOUR, Forecasts.Forecast[])

      for forecast_hour in 0:39 # 2:35 or 2:32 in practice
        valid_time_seconds = run_time_seconds + forecast_hour*HOUR

        perhaps_href_forecast              = filter(href_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast), hrefs_for_run_time)
        perhaps_sref_forecast              = filter(sref_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast), srefs_for_run_time)
        perhaps_href_forecast_3hrs_earlier = filter(href_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast), hrefs_3hrs_earlier)
        perhaps_sref_forecast_3hrs_earlier = filter(sref_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast), srefs_3hrs_earlier)

        if length(perhaps_href_forecast) >= 2
          error("shouldn't have two matching href forecasts!")
        elseif length(perhaps_sref_forecast) >= 2
          error("shouldn't have two matching sref forecasts!")
        elseif length(perhaps_href_forecast) == 1 && length(perhaps_sref_forecast_3hrs_earlier) == 1
          push!(paired_href_newer, (perhaps_href_forecast[1], perhaps_sref_forecast_3hrs_earlier[1]))
        elseif length(perhaps_sref_forecast) == 1 && length(perhaps_href_forecast_3hrs_earlier) == 1
          push!(paired_sref_newer, (perhaps_href_forecast_3hrs_earlier[1], perhaps_sref_forecast[1]))
        end
      end
    end

    run_date += Dates.Day(1)
  end

  _forecasts_href_newer = ForecastCombinators.concat_forecasts(paired_href_newer)
  _forecasts_sref_newer = ForecastCombinators.concat_forecasts(paired_sref_newer)

  _forecasts_href_newer_with_blurs_and_hour_climatology = with_blurs_and_hour_climatology(_forecasts_href_newer)
  _forecasts_sref_newer_with_blurs_and_hour_climatology = with_blurs_and_hour_climatology(_forecasts_sref_newer)

  ()
end

end # module StackedHREFSREF