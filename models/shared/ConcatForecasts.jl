module ConcatForecasts

# Create a bunch of Forecast structs whose data is the
# concatenation of the data from two underlying forecasts.
#
# Forecasts should already be on the same grid.
#
# get_feature_engineered_data is forwarded to the underlying
# forecasts separately.

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grids
import Inventories


# forecasts, example_forecast, grid, get_feature_engineered_data
function forecasts_example_forecast_grid_get_feature_engineered_data(paired_forecasts, (left_get_feature_engineered_data, right_get_feature_engineered_data))

  left_example_forecast  = paired_forecasts[1][1]
  # right_example_forecast = paired_forecasts[1][2]

  forecasts = map(paired_forecasts) do (left_forecast, right_forecast)
    get_inventory(forecast) = begin
      vcat(Forecasts.inventory(left_forecast), Forecasts.inventory(right_forecast))
    end

    get_data(forecast) = begin
      # feature_engineered_base_data = base_get_feature_engineered_data(base_forecast, Forecasts.get_data(base_forecast))
      #
      # predictions = model_predict(feature_engineered_base_data)
      #
      # reshape(predictions, (:,1)) # Make the predictions a 2D features array with 1 feature
      hcat(Forecasts.get_data(left_forecast), Forecasts.get_data(right_forecast))
    end

    later_forecast =
      if Forecasts.run_time_in_seconds_since_epoch_utc(left_forecast) >= Forecasts.run_time_in_seconds_since_epoch_utc(right_forecast)
        left_forecast
      else
        right_forecast
      end

    Forecasts.Forecast(later_forecast.run_year, later_forecast.run_month, later_forecast.run_day, later_forecast.run_hour, later_forecast.forecast_hour, [left_forecast, right_forecast], later_forecast._get_grid, get_inventory, get_data)
  end

  example_forecast = forecasts[1]
  grid             = Forecasts.grid(example_forecast)

  # twenty_five_mi_mean_is    = Grids.radius_grid_is(grid, 25.0)
  # unique_fifty_mi_mean_is   = Grids.radius_grid_is_less_other_is(grid, 50.0, _twenty_five_mi_mean_is)
  # unique_hundred_mi_mean_is = Grids.radius_grid_is_less_other_is(grid, 100.0, vcat(_twenty_five_mi_mean_is, _unique_fifty_mi_mean_is))

  get_feature_engineered_data(forecast, base_data) = begin
    left_inventory_size  = length(Forecasts.inventory(left_example_forecast))
    total_inventory_size = length(Forecasts.inventory(example_forecast))

    left_data  = base_data[:,1:left_inventory_size]
    right_data = base_data[:,(left_inventory_size+1):total_inventory_size]

    left_forecast, right_forecast = forecast.based_on

    hcat(left_get_feature_engineered_data(left_forecast, left_data), right_get_feature_engineered_data(right_forecast, right_data))
  end

  (forecasts, example_forecast, grid, get_feature_engineered_data)
end

# function forecasts_at_forecast_hour(forecast_hour)
#   filter(forecast -> forecast.forecast_hour == forecast_hour, forecasts())
# end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

end # module ConcatForecasts
