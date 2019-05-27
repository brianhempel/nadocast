module ConcatForecasts

# Create a bunch of Forecast structs whose data is the
# concatenation of the data of various underlying forecasts.
#
# Forecasts should already be on the same grid.
#
# get_feature_engineered_data is forwarded to the underlying
# forecasts separately.

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grids
import Inventories


# Hand a list of tuples of associated forecasts. Also provide a tuple of get_feature_engineered_data functions.
#
# forecasts, example_forecast, grid, get_feature_engineered_data
function forecasts_example_forecast_grid_get_feature_engineered_data(associated_forecasts, get_feature_engineered_data_functions)

  example_forecasts = collect(associated_forecasts[1])
  # left_example_forecast  = paired_forecasts[1][1]
  # right_example_forecast = paired_forecasts[1][2]

  forecasts = map(associated_forecasts) do forecasts_tuple
    forecasts_array = collect(forecasts_tuple)

    get_inventory(forecast) = begin
      vcat(Forecasts.inventory.(forecasts_tuple)...)
    end

    get_data(forecast) = begin
      hcat(Forecasts.get_data.(forecasts_tuple)...)
    end

    latest_forecast = last(sort(forecasts_array, by=Forecasts.run_time_in_seconds_since_epoch_utc))

    Forecasts.Forecast(latest_forecast.run_year, latest_forecast.run_month, latest_forecast.run_day, latest_forecast.run_hour, latest_forecast.forecast_hour, forecasts_array, latest_forecast._get_grid, get_inventory, get_data)
  end

  example_forecast = forecasts[1]
  grid             = Forecasts.grid(example_forecast)

  # twenty_five_mi_mean_is    = Grids.radius_grid_is(grid, 25.0)
  # unique_fifty_mi_mean_is   = Grids.radius_grid_is_less_other_is(grid, 50.0, _twenty_five_mi_mean_is)
  # unique_hundred_mi_mean_is = Grids.radius_grid_is_less_other_is(grid, 100.0, vcat(_twenty_five_mi_mean_is, _unique_fifty_mi_mean_is))

  get_feature_engineered_data(forecast, base_data) = begin
    inventory_sizes = length.(Forecasts.inventory.(example_forecasts))

    last_column = 0
    data_chunks = map(pairs(inventory_sizes)) do (inventory_i, inventory_size)
      base_data[:, (last_column + 1):(last_column + inventory_size)]
      last_column = last_column+inventory_size
    end

    # left_data  = base_data[:,1:left_inventory_size]
    # right_data = base_data[:,(left_inventory_size+1):total_inventory_size]

    # left_forecast, right_forecast = forecast.based_on

    hcat(left_get_feature_engineered_data.(forecast.based_on, data_chunks)...)
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
