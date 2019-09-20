module Forecasts

import Dates
using Printf

push!(LOAD_PATH, @__DIR__)

import Grids
import Inventories

MINUTE = 60
HOUR   = 60*MINUTE
DAY    = 24*HOUR

struct Forecast
  model_name     :: String
  run_year       :: Int64
  run_month      :: Int64
  run_day        :: Int64
  run_hour       :: Int64
  forecast_hour  :: Int64
  based_on       :: Vector{Forecast} # If createded by a forecast combinator, point to original(s).
  grid           :: Grids.Grid
  _get_inventory # :: Function((Forecast,), Vector{Inventories.InventoryLine}) # For lazy loading.
  _get_data      # :: Function((Forecast,), Array{Float32,2})                  # For lazy loading.
end

function time_in_seconds_since_epoch_utc(run_year :: Int64, run_month :: Int64, run_day :: Int64, run_hour :: Int64, forecast_hour = 0) :: Int64
  Int64(Dates.datetime2unix(Dates.DateTime(run_year, run_month, run_day, run_hour))) + forecast_hour*HOUR
end

function run_time_in_seconds_since_epoch_utc(forecast :: Forecast) :: Int64
  time_in_seconds_since_epoch_utc(forecast.run_year, forecast.run_month, forecast.run_day, forecast.run_hour)
end

function run_utc_datetime(forecast :: Forecast) :: Dates.DateTime
  Dates.unix2datetime(run_time_in_seconds_since_epoch_utc(forecast))
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
  time_title(forecast.run_year, forecast.run_month, forecast.run_day, forecast.run_hour, forecast.forecast_hour)
end

function time_title(run_year :: Int64, run_month :: Int64, run_day :: Int64, run_hour :: Int64, forecast_hour :: Int64) :: String
  @sprintf "%04d-%02d-%02d %02dZ +%d" run_year run_month run_day run_hour forecast_hour
end

function valid_hhz(forecast :: Forecast) :: String
  utc_datetime = valid_utc_datetime(forecast)
  @sprintf "%02dz" Dates.hour(utc_datetime)
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


function inventory(forecast :: Forecast) :: Vector{Inventories.InventoryLine}
  forecast._get_inventory()
end

function data(forecast :: Forecast) :: Array{Float32,2}
  forecast._get_data()
end


# # Note raw data is W -> E, S -> N.
# #
# # Reinterpretation lets us index it as [x, y, layer]
# function data_in_x_y_layers_3d_array(forecast :: Forecast) :: Array{Float32,3}
#   width  = forecast.grid.width
#   height = forecast.grid.height
#   reshape(data(forecast), (width, height, :))
# end


### Iteration ###

struct UncorruptedForecastsDataIteratorNoCache
  forecasts :: Vector{Forecasts.Forecast}
end

function iterate_data_of_uncorrupted_forecasts(forecasts)
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
      Forecasts.data(forecast)
    catch exception
      if isa(exception, Inventories.FieldMissing)
        println(exception)
        return Base.iterate(iterator, i+1)
      elseif isa(exception, EOFError) || isa(exception, ErrorException)
        println(exception)
        println("Bad forecast: $(forecast.model_name) $(Forecasts.time_title(forecast))")
        return Base.iterate(iterator, i+1)
      else
        rethrow(exception)
      end
    end

  ((forecast, data), i+1)
end


# Index a list of forecasts to avoid O(n^2) lookups/associations.

function run_time_seconds_to_forecasts(forecasts)
  run_time_seconds_to_forecasts = Dict{Int64,Vector{Forecast}}()

  for forecast in forecasts
    run_time = Forecasts.run_time_in_seconds_since_epoch_utc(forecast)
    forecasts_at_run_time = get(run_time_seconds_to_forecasts, run_time, Forecast[])
    push!(forecasts_at_run_time, forecast)
    run_time_seconds_to_forecasts[run_time] = forecasts_at_run_time
  end

  run_time_seconds_to_forecasts
end


end # module Forecasts