module HRRR

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Inventories
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import FeatureEngineeringShared

# HRRR is on its own 3km grid:
# $ wgrib2 -grid -end /Volumes/HRRR_1/hrrr/201903/20190314/hrrr_conus_sfc_20190314_t04z_f12.grib2
# 1:0:grid_template=30:winds(grid):
#         Lambert Conformal: (1799 x 1059) input WE:SN output WE:SN res 8
#         Lat1 21.138123 Lon1 237.280472 LoV 262.500000
#         LatD 38.500000 Latin1 38.500000 Latin2 38.500000
#         LatSP 0.000000 LonSP 0.000000
#         North Pole (1799 x 1059) Dx 3000.000000 m Dy 3000.000000 m mode 8

# Uncommon layers:
# UGRD:300 mb:hour fcst:
# VGRD:300 mb:hour fcst:
# USWRF:top of atmosphere:hour fcst:

_forecasts = []
downsample = 3

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
  else
    _forecasts
  end
end

function example_forecast()
  forecasts()[1]
end

function grid()
  Forecasts.grid(example_forecast())
end

layer_blocks_to_make = FeatureEngineeringShared.all_layer_blocks

function feature_i_to_name(feature_i)
  inventory = Forecasts.inventory(example_forecast())
  FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
end

common_layers = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers.txt"), String), "\n"))

vector_wind_layers = [
  "GRD:250 mb:hour fcst:",
  "GRD:500 mb:hour fcst:",
  "GRD:700 mb:hour fcst:",
  "GRD:850 mb:hour fcst:",
  "GRD:925 mb:hour fcst:",
  "GRD:1000 mb:hour fcst:",
  "GRD:80 m above ground:hour fcst:",
  "GRD:10 m above ground:hour fcst:",
  "STM:6000-0 m above ground:hour fcst:",
  "VCSH:1000-0 m above ground:hour fcst:", # VUCSH:1000-0 m above ground:hour fcst: and VVCSH:1000-0 m above ground:hour fcst: (special handling in FeatureEngineeringShared)
  "VCSH:6000-0 m above ground:hour fcst:", # VUCSH:6000-0 m above ground:hour fcst: and VVCSH:6000-0 m above ground:hour fcst: (special handling in FeatureEngineeringShared)
]

_twenty_five_mi_mean_is    = Vector{Int64}[] # Grid point indicies within 25mi
_unique_fifty_mi_mean_is   = Vector{Int64}[] # Grid point indicies within 50mi but not within 25mi
_unique_hundred_mi_mean_is = Vector{Int64}[] # Grid point indicies within 100mi but not within 50mi

function get_feature_engineered_data(forecast, data)
  global _twenty_five_mi_mean_is
  global _unique_fifty_mi_mean_is
  global _unique_hundred_mi_mean_is

  _twenty_five_mi_mean_is    = isempty(_twenty_five_mi_mean_is)    ? Grids.radius_grid_is(grid(), 25.0)                                                                         : _twenty_five_mi_mean_is
  _unique_fifty_mi_mean_is   = isempty(_unique_fifty_mi_mean_is)   ? Grids.radius_grid_is_less_other_is(grid(), 50.0, _twenty_five_mi_mean_is)                                  : _unique_fifty_mi_mean_is
  _unique_hundred_mi_mean_is = isempty(_unique_hundred_mi_mean_is) ? Grids.radius_grid_is_less_other_is(grid(), 100.0, vcat(_twenty_five_mi_mean_is, _unique_fifty_mi_mean_is)) : _unique_hundred_mi_mean_is

  FeatureEngineeringShared.make_data(grid(), Forecasts.inventory(forecast), forecast.forecast_hour, data, vector_wind_layers, layer_blocks_to_make, _twenty_five_mi_mean_is, _unique_fifty_mi_mean_is, _unique_hundred_mi_mean_is)
end

function reload_forecasts()
  hrrr_paths = Grib2.all_grib2_file_paths_in("/Volumes/HRRR_1/hrrr")

  global _forecasts

  _forecasts = []

  for hrrr_path in hrrr_paths
    # "/Volumes/HRRR_1/hrrr/201607/20160715/hrrr_conus_sfc_20160715_t08z_f12.grib2"

    year_str, month_str, day_str, run_hour_str, forecast_hour_str = match(r"/hrrr_conus_sfc_(\d\d\d\d)(\d\d)(\d\d)_t(\d\d)z_f(\d\d)\.gri?b2", hrrr_path).captures

    run_year      = parse(Int64, year_str)
    run_month     = parse(Int64, month_str)
    run_day       = parse(Int64, day_str)
    run_hour      = parse(Int64, run_hour_str)
    forecast_hour = parse(Int64, forecast_hour_str)

    # This should speed up loading times and save some space in our disk cache.
    grid =
      if isempty(_forecasts)
        nothing
      else
        Forecasts.grid(_forecasts[1])
      end

    get_grid(forecast) = begin
      if grid == nothing
        Grib2.read_grid(hrrr_path, downsample = downsample)
      else
        grid
      end
    end

    get_inventory(forecast) = begin
      # Somewhat inefficient that each hour must trigger wgrib2 on the same file...prefer using Forecasts.inventory(example_forecast()) if you don't need the particular file's exact byte locations of the layers
      inventory = Grib2.read_inventory(hrrr_path)

      layer_key_to_inventory_line(key) = begin
        i = findfirst(line -> Inventories.inventory_line_key(line) == key, inventory)
        if i != nothing
          inventory[i]
        else
          exception = Inventories.FieldMissing("HRRR forecast $(Forecasts.time_title(forecast))", key, inventory)

          throw(exception)
        end
      end

      inventory_to_use = map(layer_key_to_inventory_line, common_layers)

      inventory_to_use
    end

    get_data(forecast) = begin
      downsample_grid =
        if downsample == 1
          nothing
        else
          Forecasts.grid(forecast)
        end

      data = Grib2.read_layers_data_raw(hrrr_path, Forecasts.inventory(forecast), downsample_grid = downsample_grid)

      data
    end

    forecast = Forecasts.Forecast(run_year, run_month, run_day, run_hour, forecast_hour, [], get_grid, get_inventory, get_data)

    push!(_forecasts, forecast)
  end

  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))

  _forecasts
end

end # module HRRR
