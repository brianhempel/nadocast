module HRRRPrediction

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


push!(LOAD_PATH, (@__DIR__) * "/../hrrr_late_aug_2016_forward")
import HRRR


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred = [] # For downstream combination with other forecasts

blur_radii = [10, 15, 25, 35, 50, 70]

# # Determined in Train.jl
blur_radius_f2  = 10
blur_radius_f17 = 35

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
  HRRR.grid()
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
  # hrrr_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HRRR_1/hrrr")

  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred

  _forecasts = []

  hrrr_forecasts = HRRR.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predict_f2  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2_2021-05-21T22.37.26.544/463_trees_loss_0.0008549077.model")
  predict_f6  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f6_2021-05-27T17.12.43.406/465_trees_loss_0.00092996453.model")
  predict_f12 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12_2021-05-10T20.00.43.881/471_trees_loss_0.0010017317.model")
  predict_f17 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f17_2021-04-30T10.49.47.495/341_trees_loss_0.0010138382.model")

  predict(forecast, data) = begin
    if forecast.forecast_hour <= 2
      predict_f2(data)
    elseif forecast.forecast_hour in 3:5
      PredictionForecasts.weighted_prediction_between_models_at_different_forecast_hours(forecast, data, 2, 6, predict_f2, predict_f6)
    elseif forecast.forecast_hour == 6
      predict_f6(data)
    elseif forecast.forecast_hour in 7:11
      PredictionForecasts.weighted_prediction_between_models_at_different_forecast_hours(forecast, data, 6, 12, predict_f6, predict_f12)
    elseif forecast.forecast_hour == 12
      predict_f12(data)
    elseif forecast.forecast_hour in 13:16
      PredictionForecasts.weighted_prediction_between_models_at_different_forecast_hours(forecast, data, 12, 17, predict_f12, predict_f17)
    elseif forecast.forecast_hour >= 17
      predict_f17(data)
    end
  end

  _forecasts = PredictionForecasts.simple_prediction_forecasts(hrrr_forecasts, predict; inventory_misc = "calculated prob")

  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  blur_lo_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f2))
  blur_hi_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f17))

  _forecasts_blurred = PredictionForecasts.blurred(_forecasts, 2:17, blur_lo_grid_is, blur_hi_grid_is)

  ()
end

end # module HRRRPrediction