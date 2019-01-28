module TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
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


# get_feature_engineered_data should be a function that takes a forecast and the raw data and returns new data
# c.f. SREF.get_feature_engineered_data
function get_data_and_labels(grid, conus_grid_bitmask, get_feature_engineered_data, forecasts; X_transformer = identity)
  Xs = []
  Ys = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data = get_feature_engineered_data(forecast, data)

    data_in_conus = data[conus_grid_bitmask, :]
    labels        = forecast_labels(grid, forecast)[conus_grid_bitmask] :: Array{Float32,1}

    push!(Xs, X_transformer(data_in_conus))
    push!(Ys, labels)

    print(".")
  end

  (vcat(Xs...), vcat(Ys...))
end


end # module TrainingShared