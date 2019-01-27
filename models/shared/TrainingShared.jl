module TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
# import NNTrain
import StormEvents


MINUTE = 60 # seconds

TORNADO_TIME_WINDOW_HALF_SIZE = 30*MINUTE
TORNADO_SPACIAL_RADIUS_MILES  = 25.0

function is_relevant_forecast(forecast)
  for tornado in StormEvents.tornadoes
    tornado_relevant_time_range =
      (tornado.start_seconds_from_epoch_utc - TORNADO_TIME_WINDOW_HALF_SIZE + 1):(tornado.end_seconds_from_epoch_utc + TORNADO_TIME_WINDOW_HALF_SIZE - 1)

    if Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in tornado_relevant_time_range
      return true
    end
  end
  false
end

# returns (grid, conus_on_grid, train_forecasts, validation_forecasts, test_forecasts)
function forecasts_grid_conus_grid_bitmask_train_validation_test(all_forecasts)
  forecasts = filter(is_relevant_forecast, all_forecasts)

  grid = Forecasts.grid(forecasts[1])

  train_forecasts      = filter(Forecasts.is_train, forecasts)
  validation_forecasts = filter(Forecasts.is_validation, forecasts)
  test_forecasts       = filter(Forecasts.is_test, forecasts)

  conus_on_grid      = map(latlon -> Conus.is_in_conus(latlon) ? 1.0f0 : 0.0f0, grid.latlons)
  conus_grid_bitmask = (conus_on_grid .== 1.0f0)

  (grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts)
end

function forecast_labels(grid, forecast) :: Array{Float32,1}
  StormEvents.grid_to_tornado_neighborhoods(grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), TORNADO_TIME_WINDOW_HALF_SIZE)
end

end # module TrainingShared