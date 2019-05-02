module Forecasts

import Dates
using Printf

push!(LOAD_PATH, @__DIR__)

import Grids
import Inventories

MINUTE = 60
HOUR   = 60*MINUTE
DAY    = 24*HOUR

# Mutable only for lazy loading of the actual data.
mutable struct Forecast
  run_year       :: Int64
  run_month      :: Int64
  run_day        :: Int64
  run_hour       :: Int64
  forecast_hour  :: Int64
  _grid          :: Union{Grids.Grid, Nothing}                               # For lazy loading. Use grid() below.
  _inventory     :: Union{Vector{Inventories.InventoryLine}, Nothing}        # For lazy loading. Use inventory() below.
  _data          :: Union{Array{Float32,2}, Nothing}                         # For lazy loading. Use data() below.
  _get_grid      # :: Function((Forecast,), Grids.Grid)                        # For lazy loading.
  _get_inventory # :: Function((Forecast,), Vector{Inventories.InventoryLine}) # For lazy loading.
  _get_data      # :: Function((Forecast,), Array{Float32,2})                  # For lazy loading.

  Forecast(run_year, run_month, run_day, run_hour, forecast_hour, get_grid, get_inventory, get_data) =
    new(run_year, run_month, run_day, run_hour, forecast_hour, nothing, nothing, nothing, get_grid, get_inventory, get_data)
end

function run_time_in_seconds_since_epoch_utc(run_year :: Int64, run_month :: Int64, run_day :: Int64, run_hour :: Int64) :: Int64
  Int64(Dates.datetime2unix(Dates.DateTime(run_year, run_month, run_day, run_hour)))
end

function run_time_in_seconds_since_epoch_utc(forecast :: Forecast) :: Int64
  run_time_in_seconds_since_epoch_utc(forecast.run_year, forecast.run_month, forecast.run_day, forecast.run_hour)
end

function valid_time_in_seconds_since_epoch_utc(forecast :: Forecast) :: Int64
  run_time_in_seconds_since_epoch_utc(forecast) + forecast.forecast_hour*HOUR
end

function valid_utc_datetime(forecast :: Forecast) :: Dates.DateTime
  Dates.unix2datetime(valid_time_in_seconds_since_epoch_utc(forecast))
end

function valid_time_in_convective_days_since_epoch_utc(forecast :: Forecast) :: Int64
  fld(valid_time_in_seconds_since_epoch_utc(forecast) - 12*HOUR, DAY)
end

function time_title(forecast :: Forecast) :: String
  @sprintf "%04d-%02d-%02d %02dZ +%d" forecast.run_year forecast.run_month forecast.run_day forecast.run_hour forecast.forecast_hour
end

function valid_yyyymmdd_hhz(forecast :: Forecast) :: String
  utc_datetime = valid_utc_datetime(forecast)
  @sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime)
end

function yyyymmdd_thhz_fhh(forecast :: Forecast) :: String
  @sprintf "%04d%02d%02d_t%02dz_f%02d" forecast.run_year forecast.run_month forecast.run_day forecast.run_hour forecast.forecast_hour
end

function yyyymmdd_thhz(forecast :: Forecast) :: String
  @sprintf "%04d%02d%02d_t%02dz" forecast.run_year forecast.run_month forecast.run_day forecast.run_hour
end

function yyyymmdd(forecast :: Forecast) :: String
  @sprintf "%04d%02d%02d" forecast.run_year forecast.run_month forecast.run_day
end


function grid(forecast :: Forecast) :: Grids.Grid
  if forecast._grid == nothing
    forecast._grid = forecast._get_grid(forecast)
  end
  forecast._grid
end

function inventory(forecast :: Forecast) :: Vector{Inventories.InventoryLine}
  if forecast._inventory == nothing
    forecast._inventory = forecast._get_inventory(forecast)
  end
  forecast._inventory
end

# If not cached, get but don't cache the data.
function get_data(forecast :: Forecast) :: Array{Float32,2}
  if forecast._data == nothing
    forecast._get_data(forecast)
  else
    forecast._data
  end
end

# If not cached, get and cache the data.
function data(forecast :: Forecast) :: Array{Float32,2}
  if forecast._data == nothing
    forecast._data = forecast._get_data(forecast)
  end
  forecast._data
end

# # Note raw data is W -> E, S -> N.
# #
# # Reinterpretation lets us index it as [x, y, layer]
# function data_in_x_y_layers_3d_array(forecast :: Forecast) :: Array{Float32,3}
#   width  = grid(forecast).width
#   height = grid(forecast).height
#   reshape(data(forecast), (width, height, :))
# end


### Iteration ###

struct UncorruptedForecastsDataIteratorNoCache
  forecasts :: Vector{Forecasts.Forecast}
end

function iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
  UncorruptedForecastsDataIteratorNoCache(collect(forecasts))
end

function Base.iterate(iterator::UncorruptedForecastsDataIteratorNoCache, state=1)
  i         = state
  forecasts = iterator.forecasts

  if i > length(forecasts)
    return nothing
  end

  forecast = forecasts[i]

  data =
    try
      Forecasts.get_data(forecast)
    catch exception
      if isa(exception, EOFError) || isa(exception, ErrorException)
        println("Bad forecast: $(Forecasts.time_title(forecast))")
        return Base.iterate(iterator, i+1)
      elseif isa(exception, Inventories.FieldMissing)
        println(exception)
        return Base.iterate(iterator, i+1)
      else
        rethrow(exception)
      end
    end

  ((forecast, data), i+1)
end

end # module Forecasts