module ForecastCombinators

push!(LOAD_PATH, (@__DIR__))

import Forecasts
import Grids
import Inventories
import Cache
import Printf


# Copy time info, but set based_on and grid/inventory/data
function revised_forecast(forecast, grid, get_inventory, get_data; model_name = forecast.model_name)
  Forecasts.Forecast(model_name, forecast.run_year, forecast.run_month, forecast.run_day, forecast.run_hour, forecast.forecast_hour, [forecast], grid, get_inventory, get_data, forecast.preload_paths)
end

# Create a bunch of Forecast structs whose inventory and data
# are transformed by the given functions (if provided).
#
# Each transformer function is given (old_forecast, old_thing).
function map_forecasts(old_forecasts; new_grid = nothing, inventory_transformer = nothing, data_transformer = nothing, model_name = nothing)
  map(old_forecasts) do old_forecast
    get_inventory() = inventory_transformer(old_forecast, Forecasts.inventory(old_forecast))
    get_data()      = begin
      duration = @elapsed (out = data_transformer(old_forecast, Forecasts.data(old_forecast)))
      if get(ENV, "SHOW_TIMING", "false") == "true"
        name = isnothing(model_name) ? old_forecast.model_name : model_name
        println("$(Float32(duration))\t$name")
      end
      out
    end



    grid           = isnothing(new_grid)              ? old_forecast.grid           : new_grid
    _get_inventory = isnothing(inventory_transformer) ? old_forecast._get_inventory : get_inventory
    _get_data      = isnothing(data_transformer)      ? old_forecast._get_data      : get_data

    revised_forecast(old_forecast, grid, _get_inventory, _get_data; model_name = isnothing(model_name) ? old_forecast.model_name : model_name)
  end
end

# Create a bunch of Forecast structs whose data is the
# concatenation of the data of various underlying forecasts.
#
# Forecasts should already be on the same grid.
#
# Hand a list of tuples of associated forecasts, and possible a function that takes such a
# tuple and returns the forecast whose runtime + forecast time + grid should be used as the
# time and grid for the output forecast. If not provided, a forecast with the latest runtime is used.
function concat_forecasts(associated_forecasts; forecasts_tuple_to_canonical_forecast = nothing, model_name = nothing)
  map(associated_forecasts) do forecasts_tuple
    forecasts_array = collect(forecasts_tuple)

    get_inventory() = vcat(Forecasts.inventory.(forecasts_array)...)
    get_data()      = begin
      out_datas = map(Forecasts.data, forecasts_array) # Vector of 2D arrays

      # Threaded hcat

      sizes = map(size, out_datas)

      point_count = first(sizes[1])

      for out_data in out_datas
        @assert size(out_data, 1) == point_count
      end

      aggregate_sizes = cumsum(map(last, sizes))
      feature_count   = last(aggregate_sizes)

      # print("Concating... ")
      # 0.6s out of 15s loading time. trivial allocation count
      begin
        out = Array{Float32}(undef, (point_count, feature_count))

        Threads.@threads for feature_i in 1:feature_count
          out_data_i         = findfirst(n -> feature_i <= n, aggregate_sizes)
          out_data           = out_datas[out_data_i]
          out_data_feature_i = out_data_i == 1 ? feature_i : feature_i - aggregate_sizes[out_data_i - 1]

          out[:, feature_i] = @view out_data[:, out_data_feature_i]
        end

        out
      end
    end

    canonical_forecast =
      isnothing(forecasts_tuple_to_canonical_forecast) ? last(sort(forecasts_array, by=Forecasts.run_time_in_seconds_since_epoch_utc)) : forecasts_tuple_to_canonical_forecast(forecasts_tuple)

    model_name = isnothing(model_name) ? join(map(forecast -> forecast.model_name, forecasts_array), "|") : model_name

    preload_paths = vcat(map(forecast -> forecast.preload_paths, forecasts_array)...)

    Forecasts.Forecast(model_name, canonical_forecast.run_year, canonical_forecast.run_month, canonical_forecast.run_day, canonical_forecast.run_hour, canonical_forecast.forecast_hour, forecasts_array, canonical_forecast.grid, get_inventory, get_data, preload_paths)
  end
end


# Create a bunch of Forecast structs whose data is
# that of other forecasts, but with some features removed.
#
# The predicate is given an inventory line. Features matching
# the predicate are retained.
function filter_features_forecasts(old_forecasts, predicate; model_name = nothing)
  inventory_transformer(old_forecast, old_inventory) = filter(predicate, old_inventory)
  data_transformer(old_forecast, old_data)           = old_data[:, findall(predicate, Forecasts.inventory(old_forecast))]

  map_forecasts(old_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer, model_name = model_name)
end

# Concat the features from all the appropriate hourly forecasts in a day/fourhourly together.
# Note that for a runtime during the convective day there may be less that 24 forecasts, so the data width will vary.
# Returns:
# (
#   day1_hourlies_concated,
#   day2_hourlies_concated,
#   fourhourlies_concated
# )
function gather_daily_and_fourhourly(hourly_forecasts)
  run_time_seconds_to_hourly_forecasts = Forecasts.run_time_seconds_to_forecasts(hourly_forecasts)

  run_ymdhs = sort(unique(map(Forecasts.run_year_month_day_hour, hourly_forecasts)), alg=MergeSort)

  associated_fourhourly_forecasts = []
  associated_forecasts_in_day1  = []
  associated_forecasts_in_day2 = []
  for (run_year, run_month, run_day, run_hour) in run_ymdhs
    run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

    forecasts_for_run_time = get(run_time_seconds_to_hourly_forecasts, run_time_seconds, Forecasts.Forecast[])

    forecast_hours = map(f -> f.forecast_hour, forecasts_for_run_time)
    min_forecast_hour = minimum(forecast_hours)
    max_forecast_hour = maximum(forecast_hours)
    for forecast_hour in (min_forecast_hour+3):max_forecast_hour
      forecast_hours_in_fourhourly_period = forecast_hour-3 : forecast_hour
      forecasts_for_fourhourly_period = filter(forecast -> forecast.forecast_hour in forecast_hours_in_fourhourly_period, forecasts_for_run_time)
      if length(forecasts_for_fourhourly_period) == 4
        push!(associated_fourhourly_forecasts, forecasts_for_fourhourly_period)
      end
    end

    forecast_hours_in_convective_day1 = max(12-run_hour,2) : 35-run_hour
    forecast_hours_in_convective_day2 = forecast_hours_in_convective_day1.stop+1 : forecast_hours_in_convective_day1.stop+24

    forecasts_for_convective_day1 = filter(forecast -> forecast.forecast_hour in forecast_hours_in_convective_day1, forecasts_for_run_time)
    if length(forecast_hours_in_convective_day1) == length(forecasts_for_convective_day1)
      push!(associated_forecasts_in_day1, forecasts_for_convective_day1)
    end
    forecasts_for_convective_day2 = filter(forecast -> forecast.forecast_hour in forecast_hours_in_convective_day2, forecasts_for_run_time)
    if length(forecast_hours_in_convective_day2) == length(forecasts_for_convective_day2)
      push!(associated_forecasts_in_day2, forecasts_for_convective_day2)
    end
  end

  # Which run time and forecast hour to use for the set.
  # Namely: latest run time, then longest forecast hour
  choose_canonical_forecast(associated_hourlies) = begin
    canonical = associated_hourlies[1]
    for forecast in associated_hourlies
      if (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)) > (Forecasts.run_time_in_seconds_since_epoch_utc(canonical), Forecasts.valid_time_in_seconds_since_epoch_utc(canonical))
        canonical = forecast
      end
    end
    canonical
  end

  day1_hourlies_concated = concat_forecasts(associated_forecasts_in_day1,    forecasts_tuple_to_canonical_forecast = choose_canonical_forecast)
  day2_hourlies_concated = concat_forecasts(associated_forecasts_in_day2,    forecasts_tuple_to_canonical_forecast = choose_canonical_forecast)
  fourhourlies_concated  = concat_forecasts(associated_fourhourly_forecasts, forecasts_tuple_to_canonical_forecast = choose_canonical_forecast)

  ( day1_hourlies_concated, day2_hourlies_concated, fourhourlies_concated )
end


# Create a bunch of Forecast structs whose data is the data from
# some original Forecast structs, but resampled to a new grid.
#
#    get_layer_resampler = Grids.get_interpolating_upsampler
# or get_layer_resampler = Grids.get_upsampler
function resample_forecasts(old_forecasts, get_layer_resampler, new_grid; model_name = nothing)

  if isempty(old_forecasts)
    return old_forecasts
  end

  layer_resampler = get_layer_resampler(old_forecasts[1].grid, new_grid)

  data_transformer(old_forecast, old_data) = begin
    feature_count    = size(old_data, 2)
    grid_point_count = new_grid.height * new_grid.width
    resampled = Array{Float32}(undef, (grid_point_count, feature_count))

    Threads.@threads for feature_i in 1:feature_count
      resampled[:, feature_i] = layer_resampler(@view old_data[:, feature_i])
    end

    resampled
  end

  map_forecasts(old_forecasts; new_grid = new_grid, data_transformer = data_transformer, model_name = model_name)
end

_caching_on         = false
_cached_inventories = Tuple{Forecasts.Forecast,Vector{Inventories.InventoryLine}}[]
_cached_data        = Tuple{Forecasts.Forecast,Array{Float32,2}}[]

function turn_forecast_caching_on()
  global _caching_on
  _caching_on = true
end

function turn_forecast_caching_off()
  global _caching_on
  _caching_on = false
end

function clear_cached_forecasts()
  global _cached_inventories
  global _cached_data
  _cached_inventories = []
  _cached_data        = []
  GC.gc(true)
end

function cache_lookup(f, cache, max_cache_size_bytes, forecast)
  if !_caching_on
    f()
  else
    i = findfirst(entry -> entry[1] === forecast, cache)
    if isnothing(i)
      out = f()
      push!(cache, (forecast, out))
      cache_eviction_happened = false
      while sum(sizeof.(last.(cache))) > max_cache_size_bytes
        popfirst!(cache)
        cache_eviction_happened = true
      end
      if cache_eviction_happened
        println("$(length(cache)) cache entries after eviction")
      end
      out
    else
      # println("cache hit")
      cache[i][2]
    end
  end
end

# Wrapper that caches the underlying inventory and data in memory.
#
# Grid not cached; probably wasteful to do so since most grids are pointed to a single place.
function cache_forecasts(old_forecasts)
  map(old_forecasts) do old_forecast
    get_inventory() = begin
      cache_lookup(_cached_inventories, 1_000_000, old_forecast) do
        Forecasts.inventory(old_forecast)
      end :: Vector{Inventories.InventoryLine}
    end

    get_data() = begin
      cache_lookup(_cached_data, 50_000_000_000, old_forecast) do
        Forecasts.data(old_forecast)
      end :: Array{Float32,2}
    end

    revised_forecast(old_forecast, old_forecast.grid, get_inventory, get_data)
  end
end


# Wrapper that caches the underlying inventory and data on disk in the lib/computation_cache/cached_forecasts folder.
#
# It's up to you to clean this out to expire the cache (or choose a base_key that will change when the underlying computation changes).
#
# Right now we are using this in these places:
# - The hourly individual prediction forecasts just after running the GBDT but before bluring and combination and calibration
# - The SPC outlooks (rasterizing the probs is kinda slow)
#
# Set the environment variable FORECAST_DISK_PREFETCH=false when you know you are going to hit cache instead of the grib2s on disk. This
# setting will tell Forecasts.jl not to dump the next forecast's grib2s to /dev/null while the prior forecast is loading.
#
# Grid not cached; probably wasteful to do so since most grids are pointed to a single place.
function disk_cache_forecasts(old_forecasts, base_key)

  function item_key_parts(forecast)
    yyyymm   = Printf.@sprintf "%04d%02d"     forecast.run_year forecast.run_month
    yyyymmdd = Printf.@sprintf "%04d%02d%02d" forecast.run_year forecast.run_month forecast.run_day
    [ "cached_forecasts"
    , base_key
    , yyyymm
    , yyyymmdd
    , yyyymmdd * "_t" * string(forecast.run_hour) * "z_f" * string(forecast.forecast_hour)
    ]
  end

  map(old_forecasts) do old_forecast
    get_inventory() =
      Cache.cached(item_key_parts(old_forecast), "inventory") do
        Forecasts.inventory(old_forecast)
      end :: Vector{Inventories.InventoryLine}

    get_data() = begin
      Cache.cached(item_key_parts(old_forecast), "data") do
        Forecasts.data(old_forecast)
      end :: Array{Float32,2}
    end

    revised_forecast(old_forecast, old_forecast.grid, get_inventory, get_data)
  end
end


# Experiment with disabling GC. It was not helpful.

_gc_circumvention_on = false

function turn_forecast_gc_circumvention_on()
  global _gc_circumvention_on
  _gc_circumvention_on = true
end

function turn_forecast_gc_circumvention_off()
  global _gc_circumvention_on
  _gc_circumvention_on = false
end

function circumvent_gc_forecasts(forecasts)
  map(forecasts) do forecast
    get_data() = begin
      if _gc_circumvention_on
        prior_gc_state = GC.enable(false)
      end
      out = Forecasts.data(forecast)
      if _gc_circumvention_on
        GC.enable(prior_gc_state)
        if prior_gc_state
          GC.gc(true)
        end
      end
      out
    end

    revised_forecast(forecast, forecast.grid, forecast._get_inventory, get_data)
  end
end

end # module ForecastCombinators


