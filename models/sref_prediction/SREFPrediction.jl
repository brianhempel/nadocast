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
  ("tornado", "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-16T21.10.32.236_tornado_climatology_all/1174_trees_loss_0.0012410119.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T20.17.27.417_tornado_climatology_all/514_trees_loss_0.001334934.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-22T15.45.20.679_tornado_climatology_all/549_trees_loss_0.0013913331.model"
  ),
  ("wind", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-16T21.10.32.236_wind_climatology_all/1251_trees_loss_0.007144468.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T20.17.27.417_wind_climatology_all/946_trees_loss_0.0076652328.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-22T15.45.20.679_wind_climatology_all/1102_trees_loss_0.0079156775.model"
  ),
  ("wind_adj", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T19.36.04.508_wind_adj_climatology_all/1109_trees_loss_0.0025972943.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T19.43.23.541_wind_adj_climatology_all/594_trees_loss_0.0027557628.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-21T19.43.25.532_wind_adj_climatology_all/1031_trees_loss_0.0028386333.model"
  ),
  ("hail", "HAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T19.36.04.508_hail_climatology_all/1249_trees_loss_0.0036505193.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T19.43.23.541_hail_climatology_all/1191_trees_loss_0.003924385.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-21T19.43.25.532_hail_climatology_all/705_trees_loss_0.0041216174.model"
  ),
  ("sig_tornado", "STORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-16T21.10.32.236_sig_tornado_climatology_all/650_trees_loss_0.00021289948.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T20.17.27.417_sig_tornado_climatology_all/794_trees_loss_0.00022598228.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-22T15.45.20.679_sig_tornado_climatology_all/446_trees_loss_0.00023354763.model"
  ),
  ("sig_wind", "SWINDPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-16T21.10.32.236_sig_wind_climatology_all/517_trees_loss_0.001052755.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T20.17.27.417_sig_wind_climatology_all/517_trees_loss_0.0011016179.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-22T15.45.20.679_sig_wind_climatology_all/482_trees_loss_0.0011263784.model"
  ),
  ("sig_wind_adj", "SWINDPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T19.36.04.508_sig_wind_adj_climatology_all/337_trees_loss_0.00038395263.model",
                               "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T19.43.23.541_sig_wind_adj_climatology_all/165_trees_loss_0.0004010354.model",
                               "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-21T19.43.25.532_sig_wind_adj_climatology_all/432_trees_loss_0.00041955602.model"
  ),
  ("sig_hail", "SHAILPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T19.36.04.508_sig_hail_climatology_all/1194_trees_loss_0.00055598136.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-09-21T19.43.23.541_sig_hail_climatology_all/455_trees_loss_0.0005870462.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-09-21T19.43.25.532_sig_hail_climatology_all/566_trees_loss_0.00062155735.model"
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
      "sref_prediction_raw_2022_models_$(hash(models))"
    )

  # The forecast hour is needed for training purposes.
  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  # Determined in Train.jl
  # event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado     0                   50                   0.02083797
  # wind        50                  50                   0.09102787
  # hail        0                   35                   0.057103045
  # sig_tornado 50                  35                   0.012623444 lol
  # sig_wind    35                  70                   0.012305792
  # sig_hail    0                   50                   0.013548006

  blur_0mi_grid_is  = Grids.radius_grid_is(grid, 0.0)
  # blur_35mi_grid_is = Grids.radius_grid_is(grid, 35.0)
  # blur_50mi_grid_is = Grids.radius_grid_is(grid, 50.0)
  # blur_70mi_grid_is = Grids.radius_grid_is(grid, 70.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_0mi_grid_is, blur_0mi_grid_is), # tornado
    (blur_0mi_grid_is, blur_0mi_grid_is), # wind
    (blur_0mi_grid_is, blur_0mi_grid_is), # wind_adj
    (blur_0mi_grid_is, blur_0mi_grid_is), # hail
    (blur_0mi_grid_is, blur_0mi_grid_is), # sig_tornado
    (blur_0mi_grid_is, blur_0mi_grid_is), # sig_wind
    (blur_0mi_grid_is, blur_0mi_grid_is), # sig_wind_adj
    (blur_0mi_grid_is, blur_0mi_grid_is), # sig_hail
  ]

  _forecasts_blurred = PredictionForecasts.blurred(_forecasts, 2:38, blur_grid_is)

  ()
end

end # module StackedHREFSREF