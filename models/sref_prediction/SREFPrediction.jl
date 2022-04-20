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
_forecasts_blurred = [] # For downstream combination with other forecasts

blur_radii = [35, 50, 70, 100]

# Determined in Train.jl
blur_radius_f2  = 35
blur_radius_f38 = 50

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

function forecasts_blurred()
  if isempty(_forecasts_blurred)
    reload_forecasts()
    _forecasts_blurred
  else
    _forecasts_blurred
  end
end

# Yeah just dump them here
model_paths = "
	gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_hail/365_trees_loss_0.00337179.model	gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_sig_tornado/232_trees_loss_0.00019671764.model
	gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_hail/325_trees_loss_0.0036047401.model	gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-20T04.30.47.008_sig_tornado/231_trees_loss_0.00020749163.model
	…	…
"

# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f12_to_f23, gbdt_f21_to_f38)
models = [
  ("tornado", "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_tornado/389_trees_loss_0.001139937.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_tornado/271_trees_loss_0.0012234614.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-12T21.11.05.785_tornado/187_trees_loss_0.0012662631.model"
  ),
  ("wind", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_wind/527_trees_loss_0.0067868917.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_wind/464_trees_loss_0.007258802.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-13T01.36.58.700_wind/560_trees_loss_0.0074025104.model"
  ),
]

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred

  _forecasts = []
  _forecasts_with_blurs_and_forecast_hour = []
  _forecasts_blurred =[]

  sref_forecasts = SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  # (event_name, grib2_var_name, predict)
  predictors = map(models) do (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f12_to_f23, gbdt_f21_to_f38)
    predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/" * gbdt_f2_to_f13)
    predict_f12_to_f23 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/" * gbdt_f12_to_f23)
    predict_f21_to_f38 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../sref_mid_2018_forward/" * gbdt_f21_to_f38)

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

    (event_name, grib2_var_name, predict)
  end

  _forecasts = PredictionForecasts.simple_prediction_forecasts(sref_forecasts, predictors)

  # The forecast hour is needed for training purposes.
  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  blur_lo_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f2))
  blur_hi_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f38))

  _forecasts_blurred = ForecastCombinators.circumvent_gc_forecasts(PredictionForecasts.blurred(_forecasts, 2:38, blur_lo_grid_is, blur_hi_grid_is))

  ()
end

end # module StackedHREFSREF