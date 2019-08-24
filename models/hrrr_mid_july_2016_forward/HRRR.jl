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

forecasts_root() = get(ENV, "FORECASTS_ROOT", "/Volumes")

layer_blocks_to_make = FeatureEngineeringShared.all_layer_blocks

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

_forecasts = []
downsample = 3

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
  else
    _forecasts
  end
end

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    feature_interaction_terms = []
  )
end

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end


common_layers = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers.txt"), String), "\n"))

function reload_forecasts()
  hrrr_paths = Grib2.all_grib2_file_paths_in("$(forecasts_root())/HRRR_1/hrrr")

  global _forecasts

  _forecasts = []

  grid = nothing

  for hrrr_path in hrrr_paths
    # println(hrrr_path)
    # "/Volumes/HRRR_1/hrrr/201607/20160715/hrrr_conus_sfc_20160715_t08z_f12.grib2"

    year_str, month_str, day_str, run_hour_str, forecast_hour_str = match(r"/hrrr_conus_sfc_(\d\d\d\d)(\d\d)(\d\d)_t(\d\d)z_f(\d\d)\.gri?b2", hrrr_path).captures

    run_year      = parse(Int64, year_str)
    run_month     = parse(Int64, month_str)
    run_day       = parse(Int64, day_str)
    run_hour      = parse(Int64, run_hour_str)
    forecast_hour = parse(Int64, forecast_hour_str)

    if isnothing(grid)
      grid = Grib2.read_grid(hrrr_path, downsample = downsample)
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
          forecast.grid
        end

      data = Grib2.read_layers_data_raw(hrrr_path, Forecasts.inventory(forecast), downsample_grid = downsample_grid)

      data
    end

    forecast = Forecasts.Forecast("HRRR", run_year, run_month, run_day, run_hour, forecast_hour, [], grid, get_inventory, get_data)

    push!(_forecasts, forecast)
  end

  print("sorting...")
  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))
  println("done")

  _forecasts
end

end # module HRRR
