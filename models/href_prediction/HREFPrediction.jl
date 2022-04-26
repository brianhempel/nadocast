module HREFPrediction

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts



push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred = [] # For downstream combination with other forecasts

blur_radii = [15, 25, 35, 50, 70, 100]

# # Determined in Train.jl
blur_radius_f2  = 1
blur_radius_f35 = 1

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


# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
models = [
  ("tornado", "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_tornado/391_trees_loss_0.0010360148.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_tornado/317_trees_loss_0.001094988.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-16T14.36.46.241_tornado/308_trees_loss_0.0011393429.model"
  ),
  ("wind", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_wind/754_trees_loss_0.0062351814.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_wind/581_trees_loss_0.00660574.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_wind/414_trees_loss_0.006970079.model"
  ),
  ("hail", "HAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_hail/460_trees_loss_0.003063131.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_hail/560_trees_loss_0.003272809.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_hail/485_trees_loss_0.0034841662.model"
  ),
  ("sig_tornado", "STORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T17.13.12.938_sig_tornado/368_trees_loss_0.00015682736.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_sig_tornado/158_trees_loss_0.0001665851.model",
                              "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_sig_tornado/310_trees_loss_0.00017311782.model"
  ),
  ("sig_wind", "SWINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-21T23.06.01.396_sig_wind/269_trees_loss_0.00094322924.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-21T23.05.58.511_sig_wind/176_trees_loss_0.0009875718.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-22T04.03.41.799_sig_wind/184_trees_loss_0.0010262702.model"
  ),
  ("sig_hail", "SHAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-21T23.06.01.396_sig_hail/274_trees_loss_0.00049601856.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-21T23.05.58.511_sig_hail/358_trees_loss_0.00052869593.model",
                            "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-22T04.03.41.799_sig_hail/485_trees_loss_0.0005760971.model"
  ),
]

function reload_forecasts()
  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred

  _forecasts = []

  href_forecasts = HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predictors = map(models) do (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
    predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f2_to_f13)
    predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f13_to_f24)
    predict_f24_to_f35 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f24_to_f35)

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

    (event_name, grib2_var_name, predict)
  end

  # Don't forget to clear the cache during development.
  # rm -r lib/computation_cache/cached_forecasts/href_prediction_raw_2021_models
  _forecasts =
    ForecastCombinators.disk_cache_forecasts(
      PredictionForecasts.simple_prediction_forecasts(href_forecasts, predictors),
      "href_prediction_raw_2021_models_$(hash(models))"
    )

  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts, blur_radii)

  grid = _forecasts[1].grid

  blur_lo_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f2))
  blur_hi_grid_is = Grids.radius_grid_is(grid, Float64(blur_radius_f35))

  _forecasts_blurred = ForecastCombinators.circumvent_gc_forecasts(PredictionForecasts.blurred(_forecasts, 2:35, blur_lo_grid_is, blur_hi_grid_is))

  ()
end

end # module HREFPrediction