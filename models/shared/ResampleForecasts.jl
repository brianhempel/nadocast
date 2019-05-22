module ResampleForecasts

# Create a bunch of Forecast structs whose data is the
# resampling of the data from some original Forecast structs.
#
# For efficiency, only get_feature_engineered_data actually loads data (get_data does not).

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grids
import Inventories


# forecasts, example_forecast, grid, get_feature_engineered_data
function forecasts_example_forecast_grid_get_feature_engineered_data(original_forecasts, original_get_feature_engineered_data, layer_resampler, output_grid)

  forecasts = map(original_forecasts) do original_forecast
    get_inventory(forecast) = begin
      Forecasts.inventory(original_forecast)
    end

    # Only get_feature_engineered_data operates. It'd be a waste to call this, resample, then call get_feature_engineered_data which cannot use the resampled data.
    get_data(forecast) = begin
      []
    end

    get_grid(forecast) = begin
      output_grid
    end

    Forecast(original_forecast.run_year, original_forecast.run_month, original_forecast.run_day, original_forecast.run_hour, original_forecast.forecast_hour, [original_forecast], get_grid, get_inventory, get_data)
  end

  example_forecast = forecasts[1]

  grid_point_count = output_grid.height * output_grid.width

  get_feature_engineered_data(forecast, junk_data) = begin
    original_forecast = forecast.based_on[1]

    feature_engineered_base_data = original_get_feature_engineered_data(original_forecast, Forecasts.get_data(original_forecast))

    feature_count = size(feature_engineered_base_data, 2)

    resampled = Array{Float32}(undef, (grid_point_count, feature_count))

    for feature_i in 1:feature_count
      resampled[:, feature_i] = layer_resampler(feature_engineered_base_data[:, feature_i])
    end

    resampled
  end

  (forecasts, example_forecast, output_grid, get_feature_engineered_data)
end

# function forecasts_at_forecast_hour(forecast_hour)
#   filter(forecast -> forecast.forecast_hour == forecast_hour, forecasts())
# end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

end # module ResampleForecasts
