module ForecastCombinators

push!(LOAD_PATH, (@__DIR__))

import Forecasts
import Grids
import Inventories


# Copy time info, but set based_on and grid/inventory/data
function revised_forecast(forecast, grid, get_inventory, get_data)
  Forecasts.Forecast(forecast.model_name, forecast.run_year, forecast.run_month, forecast.run_day, forecast.run_hour, forecast.forecast_hour, [forecast], grid, get_inventory, get_data)
end

# Create a bunch of Forecast structs whose inventory and data
# are transformed by the given functions (if provided).
#
# Each transformer function is given (old_forecast, old_thing).
function map_forecasts(old_forecasts; new_grid = nothing, inventory_transformer = nothing, data_transformer = nothing)

  get_inventory(forecast) = begin
    old_forecast = forecast.based_on[1]
    inventory_transformer(old_forecast, Forecasts.inventory(old_forecast))
  end

  get_data(forecast) = begin
    old_forecast = forecast.based_on[1]
    data_transformer(old_forecast, Forecasts.data(old_forecast))
  end

  map(old_forecasts) do old_forecast
    grid           = isnothing(new_grid)              ? old_forecast.grid           : new_grid
    _get_inventory = isnothing(inventory_transformer) ? old_forecast._get_inventory : get_inventory
    _get_data      = isnothing(data_transformer)      ? old_forecast._get_data      : get_data
    revised_forecast(old_forecast, grid, _get_inventory, _get_data)
  end
end

# Create a bunch of Forecast structs whose data is the
# concatenation of the data of various underlying forecasts.
#
# Forecasts should already be on the same grid.
#
# Hand a list of tuples of associated forecasts.
function concat_forecasts(associated_forecasts)
  get_inventory(forecast) = begin
    vcat(Forecasts.inventory.(forecast.based_on)...)
  end

  get_data(forecast) = begin
    hcat(Forecasts.data.(forecast.based_on)...)
  end

  map(associated_forecasts) do forecasts_tuple
    forecasts_array = collect(forecasts_tuple)

    latest_forecast = last(sort(forecasts_array, by=Forecasts.run_time_in_seconds_since_epoch_utc))

    model_names = map(forecast -> forecast.model_name, forecasts_array)

    Forecasts.Forecast(join(model_names, "|"), latest_forecast.run_year, latest_forecast.run_month, latest_forecast.run_day, latest_forecast.run_hour, latest_forecast.forecast_hour, forecasts_array, latest_forecast.grid, get_inventory, get_data)
  end
end


# Create a bunch of Forecast structs whose data is
# that of other forecasts, but with some features removed.
#
# The predicate is given an inventory line. Features matching
# the predicate are retained.
function filter_features_forecasts(old_forecasts, predicate)
  inventory_transformer(old_forecast, old_inventory) = begin
    filter(predicate, old_inventory)
  end

  data_transformer(old_forecast, old_data) = begin
    features_is = findall(predicate, Forecasts.inventory(old_forecast))
    old_data[:, features_is]
  end

  map_forecasts(old_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end


# Create a bunch of Forecast structs whose data is the data from
# some original Forecast structs, but resampled to a new grid.
#
#    get_layer_resampler = Grids.get_interpolating_upsampler
# or get_layer_resampler = Grids.get_upsampler
function resample_forecasts(old_forecasts, get_layer_resampler, new_grid)

  if isempty(old_forecasts)
    return old_forecasts
  end

  layer_resampler = get_layer_resampler(old_forecasts[1].grid, new_grid)

  data_transformer(old_forecast, old_data) = begin
    feature_count    = size(old_data, 2)
    grid_point_count = new_grid.height * new_grid.width
    resampled = Array{Float32}(undef, (grid_point_count, feature_count))

    for feature_i in 1:feature_count
      resampled[:, feature_i] = layer_resampler(old_data[:, feature_i])
    end

    resampled
  end

  map_forecasts(old_forecasts; new_grid = new_grid, data_transformer = data_transformer)
end


# Wrapper that caches the underlying inventory and data on disk.
#
# Grid not cached; probably wasteful to do so since most grids are pointed to a single place.
function cache_forecasts(old_forecasts)

  item_key_parts(forecast) =
    [ forecast.model_name
    , string(forecast.run_year) * string(forecast.run_month)
    , string(forecast.run_year) * string(forecast.run_month) * string(forecast.run_day)
    , string(forecast.run_year) * string(forecast.run_month) * string(forecast.run_day) * "_t" * string(forecast.run_hour) * "z_f" * string(forecast.forecast_hour)
    ]

  get_inventory(forecast) = begin
    old_forecast = forecast.based_on[1]
    Cache.cached(item_key_parts(old_forecast), "inventory") do
      Forecasts.inventory(old_forecast)
    end
  end

  get_data(forecast) = begin
    old_forecast = forecast.based_on[1]
    Cache.cached(item_key_parts(old_forecast), "data") do
      Forecasts.data(old_forecast)
    end
  end

  map(old_forecasts) do old_forecast
    revised_forecast(old_forecast, old_forecast.grid, get_inventory, get_data)
  end
end

end # module ForecastCombinators
