module SREFPrediction

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts


push!(LOAD_PATH, (@__DIR__) * "/../sref_mid_2018_forward")
import SREF


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred = [] # For downstream combination with other forecasts

blur_radii = [35, 50, 70, 100]


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
  ("hail", "HAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_hail/365_trees_loss_0.00337179.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_hail/325_trees_loss_0.0036047401.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-20T09.45.13.302_hail/480_trees_loss_0.0037651379.model"
  ),
  ("sig_tornado", "STORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_sig_tornado/232_trees_loss_0.00019671764.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-20T04.30.47.008_sig_tornado/231_trees_loss_0.00020749163.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-20T09.45.13.302_sig_tornado/189_trees_loss_0.00021580876.model"
  ),
  ("sig_wind", "SWINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-22T08.00.21.554_sig_wind/211_trees_loss_0.0010130208.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-22T08.06.21.664_sig_wind/262_trees_loss_0.001065569.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-22T08.00.27.390_sig_wind/183_trees_loss_0.0010853795.model"
  ),
  ("sig_hail", "SHAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-22T08.00.21.554_sig_hail/263_trees_loss_0.00052539556.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-22T08.06.21.664_sig_hail/290_trees_loss_0.0005572099.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-22T08.00.27.390_sig_hail/319_trees_loss_0.0005862602.model"
  ),
]

function reload_forecasts()
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

  # Don't forget to clear the cache during development.
  # rm -r lib/computation_cache/cached_forecasts/sref_prediction_raw_2021_models_*
  _forecasts =
    ForecastCombinators.disk_cache_forecasts(
      PredictionForecasts.simple_prediction_forecasts(sref_forecasts, predictors),
      "sref_prediction_raw_2021_models_$(hash(models))"
    )

  # The forecast hour is needed for training purposes.
  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  # Determined in Train.jl
  # event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado     0                   50                   0.020957047
  # wind        35                  50                   0.09067886
  # hail        0                   35                   0.057110623
  # sig_tornado 35                  35                   0.013260133
  # sig_wind    35                  70                   0.012308959
  # sig_hail    0                   50                   0.013652643

  blur_0mi_grid_is  = Grids.radius_grid_is(grid, 0.0)
  blur_35mi_grid_is = Grids.radius_grid_is(grid, 35.0)
  blur_50mi_grid_is = Grids.radius_grid_is(grid, 50.0)
  blur_70mi_grid_is = Grids.radius_grid_is(grid, 70.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_0mi_grid_is,  blur_50mi_grid_is), # tornado
    (blur_35mi_grid_is, blur_50mi_grid_is), # wind
    (blur_0mi_grid_is,  blur_35mi_grid_is), # hail
    (blur_35mi_grid_is, blur_35mi_grid_is), # sig_tornado
    (blur_35mi_grid_is, blur_70mi_grid_is), # sig_wind
    (blur_0mi_grid_is,  blur_50mi_grid_is), # sig_hail
  ]

  _forecasts_blurred = PredictionForecasts.blurred(_forecasts, 2:38, blur_grid_is)

  ()
end

end # module StackedHREFSREF