module HRRR

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Inventories
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import FeatureEngineeringShared
import ThreeHourWindowForecasts

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

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

# get_layer = FeatureEngineeringShared.get_layer
#
# sbcape_key      = "CAPE:surface:hour fcst:wt ens mean"
# sbcin_key       = "CIN:surface:hour fcst:wt ens mean"
# helicity3km_key = "HLCY:3000-0 m above ground:hour fcst:wt ens mean"
#
# function compute_0_500mb_BWD(inventory, data)
#   diff_u = get_layer(data, inventory, "UGRD:500 mb:hour fcst:wt ens mean") .- get_layer(data, inventory, "UGRD:10 m above ground:hour fcst:wt ens mean")
#   diff_v = get_layer(data, inventory, "VGRD:500 mb:hour fcst:wt ens mean") .- get_layer(data, inventory, "VGRD:10 m above ground:hour fcst:wt ens mean")
#   sqrt.(diff_u.^2 .+ diff_v.^2)
# end
#
# interaction_terms = [
#   # 0-3km EHI, roughly
#   (    "SBCAPE*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* get_layer(data, inventory, helicity3km_key)),
#   ("sqrtSBCAPE*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* get_layer(data, inventory, helicity3km_key)),
#
#   # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
#   (    "SBCAPE*BWD0-500mb", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data)),
#   ("sqrtSBCAPE*BWD0-500mb", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data)),
#
#   # Pseudo-STP terms
#   (    "SBCAPE*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
#   ("sqrtSBCAPE*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
#
#   (    "SBCAPE*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
#   ("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
#
#   (    "SBCAPE*BWD0-500mb*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key)),
#   ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key)),
#
#   (    "SBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
#   ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
# ]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = []
  )
end

function three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts_with_climatology(feature_engineered_forecasts())
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end


common_layers = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers.txt"), String), "\n"))

function reload_forecasts()
  # HRRR_1 contains runs through 2018
  # HRRR_2 contains runs 2019+
  hrrr_paths = vcat(
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/HRRR_1/hrrr"),
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/HRRR_2/hrrr")
  )

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

    get_inventory() = begin
      # Somewhat inefficient that each hour must trigger wgrib2 on the same file...prefer using Forecasts.inventory(example_forecast()) if you don't need the particular file's exact byte locations of the layers
      inventory = Grib2.read_inventory(hrrr_path)

      layer_key_to_inventory_line(key) = begin
        i = findfirst(line -> Inventories.inventory_line_key(line) == key, inventory)
        if i != nothing
          inventory[i]
        else
          exception = Inventories.FieldMissing("HRRR forecast $(Forecasts.time_title(run_year, run_month, run_day, run_hour, forecast_hour))", key, inventory)

          throw(exception)
        end
      end

      inventory_to_use = map(layer_key_to_inventory_line, common_layers)

      inventory_to_use
    end

    get_data() = begin
      Grib2.read_layers_data_raw(hrrr_path, get_inventory(), crop_downsample_grid = grid)
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
