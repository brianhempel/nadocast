module HREFPrediction

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


push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred = [] # For downstream combination with other forecasts

blur_radii = [15, 25, 35, 50, 70, 100]

# # Determined in Train.jl
blur_radius_f2  = 35
blur_radius_f35 = 35

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
  HREF.grid()
end

function forecasts_with_blurs_and_forecast_hour()
  if isempty(_forecasts_with_blurs_and_forecast_hour)
    reload_forecasts()
    _forecasts_with_blurs_and_forecast_hour
  else
    _forecasts_with_blurs_and_forecast_hour
  end
end

function forecasts_blurred()
  if isempty(_forecasts_blurred)
    reload_forecasts()
    _forecasts_blurred
  else
    _forecasts_blurred
  end
end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred

  _forecasts = []

  href_forecasts = HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2021-05-25T11.12.14.168/416_trees_loss_0.0009960941.model")
  predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2021-05-20T16.41.52.952/164_trees_loss_0.0010514505.model")
  predict_f24_to_f35 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2021-05-26T15.55.17.39/305_trees_loss_0.0010976213.model")

  predict(forecast, data) = begin
    if forecast.forecast_hour in 25:35
      predict_f24_to_f35(data)
    elseif forecast.forecast_hour == 24
      0.5f0 .* (predict_f24_to_f35(data) .+ predict_f13_to_f24(data))
    elseif forecast.forecast_hour in 14:23
      predict_f13_to_f24(data)
    elseif forecast.forecast_hour == 13
      0.5f0 .* (predict_f13_to_f24(data) .+ predict_f2_to_f13(data))
    elseif forecast.forecast_hour in 2:12
      predict_f2_to_f13(data)
    else
      error("HREF forecast hour $(forecast.forecast_hour) not in 2:35")
    end
  end

  _forecasts = PredictionForecasts.simple_prediction_forecasts(href_forecasts, predict)

  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  blur_lo_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f2))
  blur_hi_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f35))

  _forecasts_blurred = ForecastCombinators.circumvent_gc_forecasts(PredictionForecasts.blurred(_forecasts, 2:35, blur_lo_grid_is, blur_hi_grid_is))

  ()
end

end # module HREFPrediction