module SREFPrediction

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids
import Inventories

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts
import Climatology
import FeatureEngineeringShared


push!(LOAD_PATH, (@__DIR__) * "/../sref_mid_2018_forward")
import SREF


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred_and_forecast_hour = [] # For downstream combination with other forecasts

blur_radii = [35, 50, 70, 100]

# Determined in Train.jl
blur_radius_f2  = 50
blur_radius_f38 = 0

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
    _forecasts
  else
    _forecasts
  end
end

function example_forecast()
  forecasts()[1]
end

function grid()
  SREF.grid()
end

function forecasts_with_blurs_and_forecast_hour()
  if isempty(_forecasts_with_blurs_and_forecast_hour)
    reload_forecasts()
    _forecasts_with_blurs_and_forecast_hour
  else
    _forecasts_with_blurs_and_forecast_hour
  end
end

function forecasts_blurred_and_forecast_hour()
  if isempty(_forecasts_blurred_and_forecast_hour)
    reload_forecasts()
    _forecasts_blurred_and_forecast_hour
  else
    _forecasts_blurred_and_forecast_hour
  end
end


function blurred_and_forecast_hour(prediction_forecasts)
  grid = prediction_forecasts[1].grid

  blur_lo_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f2))
  # Blur hi is no blur. That's what Train.jl says!

  inventory_transformer(base_forecast, base_inventory) = begin
    no_blur_line = base_inventory[1]

    # push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_tornado_probability_feature(grid)[1],                         "calculated", "hour fcst", "", ""))
    # push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_severe_probability_feature(grid)[1],                          "calculated", "hour fcst", "", ""))
    # push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_tornado_given_severe_probability_feature(grid)[1],            "calculated", "hour fcst", "", ""))
    # push!(new_inventory, Inventories.InventoryLine("", "", href_line.date_str, Climatology.forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)[1], "calculated", "hour fcst", "", ""))

    [
      Inventories.revise_with_feature_engineering(no_blur_line, "blurred"),
      Inventories.InventoryLine("", "", no_blur_line.date_str, "forecast_hour", "calculated", "hour fcst", "", "")
    ]
  end

  data_transformer(base_forecast, base_data) = begin
    point_count = size(base_data, 1)

    out = Array{Float32}(undef, (point_count, 2))

    no_blur_data = @view base_data[:, 1]
    blur_lo_data = FeatureEngineeringShared.meanify_threaded(no_blur_data, blur_lo_grid_is)

    forecast_hour = Float32(base_forecast.forecast_hour)
    forecast_ratio = (forecast_hour - 2f0) * (1f0/(38f0-2f0))
    one_minus_forecast_ratio = 1f0 - forecast_ratio

    Threads.@threads for i in 1:point_count
      out[i, 1] = blur_lo_data[i] * one_minus_forecast_ratio + no_blur_data[i] * forecast_ratio
      out[i, 2] = forecast_hour
    end

    # feature_i += 1
    # out[:, feature_i] = Climatology.forecast_hour_tornado_probability_feature(grid)[2](base_forecast)
    # feature_i += 1
    # out[:, feature_i] = Climatology.forecast_hour_severe_probability_feature(grid)[2](base_forecast)
    # feature_i += 1
    # out[:, feature_i] = Climatology.forecast_hour_tornado_given_severe_probability_feature(grid)[2](base_forecast)
    # feature_i += 1
    # out[:, feature_i] = Climatology.forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)[2](base_forecast)

    out
  end

  ForecastCombinators.map_forecasts(prediction_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred_and_forecast_hour

  _forecasts = []
  _forecasts_with_blurs_and_forecast_hour = []
  _forecasts_blurred_and_forecast_hour =[]

  sref_forecasts = SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_Dates.DateTime(\"2021-04-20T04.17.36.114\")/173_trees_loss_0.0011276418.model")
  predict_f12_to_f23 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_Dates.DateTime(\"2021-04-22T01.22.58.76\")/175_trees_loss_0.0012076722.model")
  predict_f21_to_f38 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2021-04-25T00.37.19.274/198_trees_loss_0.001240725.model")

  predict(forecast, data) = begin
    if forecast.forecast_hour in 24:38
      predict_f21_to_f38(data)
    elseif forecast.forecast_hour in 21:23
      0.5f0 .* (predict_f21_to_f38(data) .+ predict_f12_to_f23(data))
    elseif forecast.forecast_hour in 14:20
      predict_f12_to_f23(data)
    elseif forecast.forecast_hour in 12:13
      0.5f0 .* (predict_f12_to_f23(data) .+ predict_f2_to_f13(data))
    elseif forecast.forecast_hour in 2:11
      predict_f2_to_f13(data)
    else
      error("SREF forecast hour $(forecast.forecast_hour) not in 2:38")
    end
  end

  _forecasts = PredictionForecasts.simple_prediction_forecasts(sref_forecasts, predict; inventory_misc = "calculated prob")

  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  _forecasts_blurred_and_forecast_hour = blurred_and_forecast_hour(_forecasts)

  ()
end

end # module StackedHREFSREF