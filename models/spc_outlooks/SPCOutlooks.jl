module SPCOutlooks

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids
import Inventories
import SPC
import Conus

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts

MINUTE = 60
HOUR   = 60*MINUTE

# Forecast run time is always the newer forecast.

_forecasts_day_0600 = []
_forecasts_day_1300 = []
_forecasts_day_1630 = []  # For us, run_hour is truncated to 16

function forecasts_day_0600()
  if isempty(_forecasts_day_0600)
    reload_forecasts()
    _forecasts_day_0600
  else
    _forecasts_day_0600
  end
end

function forecasts_day_1300()
  if isempty(_forecasts_day_1300)
    reload_forecasts()
    _forecasts_day_1300
  else
    _forecasts_day_1300
  end
end

function forecasts_day_1630()
  if isempty(_forecasts_day_1630)
    reload_forecasts()
    _forecasts_day_1630
  else
    _forecasts_day_1630
  end
end


function example_forecast()
  forecasts_day_0600()[1]
end

function grid()
  Conus.href_cropped_5km_grid
end

thresholds = [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6]

function reload_forecasts()
  global _forecasts_day_0600
  global _forecasts_day_1300
  global _forecasts_day_1630

  _forecasts_day_0600 = []
  _forecasts_day_1300 = []
  _forecasts_day_1630 = []

  # threshold = 0.02
  # grid = Conus.href_cropped_5km_grid
  # # shapefile_path = "geo_regions/ln_us/ln_us.shp"
  # shapefile_path = "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp"
  #
  # @time mask = SPC.rasterize_prob_regions(grid, threshold, "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp")
  #
  # target_latlons = grid.latlons[mask]

  shapefile_paths = []

  for (dir_path, _, file_names) in walkdir((@__DIR__) * "/../../spc")
    println(dir_path)
    for file_name in file_names
      if endswith(file_name, "_torn.shp")
        push!(shapefile_paths, joinpath(dir_path, file_name)) # path to files
      end
    end
  end

  for shapefile_path in shapefile_paths
    year_str, month_str, day_str, time_str = match(r"/\w+_(\d\d\d\d)(\d\d)(\d\d)_(\d\d\d\d)_torn.shp", shapefile_path).captures

    run_year      = parse(Int64, year_str)
    run_month     = parse(Int64, month_str)
    run_day       = parse(Int64, day_str)
    run_hour      = time_str == "1200" ? 6 : parse(Int64, time_str) ÷ 100
    forecast_hour = 35 - run_hour # Valid end time should be 36 - run_hour, based on how the outlook valid times are labeled online, but that would cause the forecasts to be classified in the next day per our criteria

    get_inventory() = [ Inventories.InventoryLine("", "", "$year_str$month_str$day_str", "tornado probability", "", "day fcst", "", "") ]

    get_data() = begin
      # ArchGDAL is probably not threadsafe, so don't thread.
      data = zeros(Float32, (length(grid().latlons),1))
      for threshold in thresholds
        mask = SPC.rasterize_prob_regions(grid(), threshold, shapefile_path)
        data[mask,1] .= threshold
      end
      data
    end

    forecast = Forecasts.Forecast("SPC Day 1 Convective Outlook", run_year, run_month, run_day, run_hour, forecast_hour, [], grid(), get_inventory, get_data, [])

    if run_hour == 6
      push!(_forecasts_day_0600, forecast)
    elseif run_hour == 13
      push!(_forecasts_day_1300, forecast)
    elseif run_hour == 16
      push!(_forecasts_day_1630, forecast)
    end
  end

  ()
end

end # module SPCOutlooks
