module RAP

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Inventories
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import FeatureEngineeringShared
import ThreeHourWindowForecasts

# RAP is on grid 130: http://www.nco.ncep.noaa.gov/pmb/docs/on388/grids/grid130.gif https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID130
#
# To match the HREF grid, we'll cut 14 off the W, 0 off the E, 26 off the S, 55 off the N
crop = ((1+14):(451-0), (1+26):(337-55))

# RAP gained low-level (180-0 mb agl, 90-0 mb agl) CAPE/CIN layers on the 2014-02-25 12z run. These are key fields so we'll use forecasts after that date.
# RAP is missing convective cloud top field from 2018-07-12 12z run through the 2018-08-10 13z run; in prior experiments, this was the most important feature so we'll skip these forecasts.


forecasts_root() = get(ENV, "FORECASTS_ROOT", "/Volumes")

layer_blocks_to_make = FeatureEngineeringShared.all_layer_blocks

# layer_blocks_to_make = [
#   FeatureEngineeringShared.raw_features_block,
#   # FeatureEngineeringShared.twenty_five_mi_mean_block,
#   # FeatureEngineeringShared.fifty_mi_mean_block,
#   # FeatureEngineeringShared.hundred_mi_mean_block,
#   # FeatureEngineeringShared.twenty_five_mi_forward_gradient_block,
#   # FeatureEngineeringShared.twenty_five_mi_leftward_gradient_block,
#   # FeatureEngineeringShared.twenty_five_mi_linestraddling_gradient_block,
#   # FeatureEngineeringShared.fifty_mi_forward_gradient_block,
#   # FeatureEngineeringShared.fifty_mi_leftward_gradient_block,
#   # FeatureEngineeringShared.fifty_mi_linestraddling_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_forward_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_leftward_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_linestraddling_gradient_block,
# ]

vector_wind_layers = [
  "GRD:1000 mb:hour fcst:",
  # "GRD:975 mb:hour fcst:",
  "GRD:950 mb:hour fcst:",
  # "GRD:925 mb:hour fcst:",
  "GRD:900 mb:hour fcst:",
  # "GRD:875 mb:hour fcst:",
  "GRD:850 mb:hour fcst:",
  # "GRD:825 mb:hour fcst:",
  "GRD:800 mb:hour fcst:",
  # "GRD:775 mb:hour fcst:",
  "GRD:750 mb:hour fcst:",
  # "GRD:725 mb:hour fcst:",
  "GRD:700 mb:hour fcst:",
  # "GRD:675 mb:hour fcst:",
  "GRD:650 mb:hour fcst:",
  # "GRD:625 mb:hour fcst:",
  "GRD:600 mb:hour fcst:",
  # "GRD:575 mb:hour fcst:",
  "GRD:550 mb:hour fcst:",
  # "GRD:525 mb:hour fcst:",
  "GRD:500 mb:hour fcst:",
  # "GRD:475 mb:hour fcst:",
  "GRD:450 mb:hour fcst:",
  # "GRD:425 mb:hour fcst:",
  "GRD:400 mb:hour fcst:",
  # "GRD:375 mb:hour fcst:",
  "GRD:350 mb:hour fcst:",
  # "GRD:325 mb:hour fcst:",
  "GRD:300 mb:hour fcst:",
  # "GRD:275 mb:hour fcst:",
  "GRD:250 mb:hour fcst:",
  # "GRD:225 mb:hour fcst:",
  "GRD:200 mb:hour fcst:",
  # "GRD:175 mb:hour fcst:",
  "GRD:150 mb:hour fcst:",
  # "GRD:125 mb:hour fcst:",
  "GRD:100 mb:hour fcst:",
  "GRD:10 m above ground:hour fcst:",
  "GRD:tropopause:hour fcst:",
  "GRD:max wind:hour fcst:",
  "GRD:30-0 mb above ground:hour fcst:",
  "GRD:60-30 mb above ground:hour fcst:",
  "GRD:90-60 mb above ground:hour fcst:",
  "GRD:120-90 mb above ground:hour fcst:",
  "GRD:150-120 mb above ground:hour fcst:",
  "GRD:180-150 mb above ground:hour fcst:",
  "STM:6000-0 m above ground:hour fcst:",
  "GRD:80 m above ground:hour fcst:",
  "VCSH:6000-0 m above ground:hour fcst:", # VUCSH:6000-0 m above ground:hour fcst: and VVCSH:6000-0 m above ground:hour fcst: (special handling in FeatureEngineeringShared)
]

_forecasts = []
downsample = 1 # Don't downsample.

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
  else
    _forecasts
  end
end

get_layer = FeatureEngineeringShared.get_layer

# I think https://rapidrefresh.noaa.gov/hrrr/GRIB2Table_hrrrncep_2d.txt explains the CAPE differences

# These term names end up being the same as the HRRR

lifted_index_key      = "LFTX:500-1000 mb:hour fcst:" # surface
# best_lifted_index_key = "4LFTX:180-0 mb above ground:hour fcst:"
sbcape_key            = "CAPE:surface:hour fcst:"
mlcape_key            = "CAPE:90-0 mb above ground:hour fcst:"
# mulayercape_key       = "CAPE:180-0 mb above ground:hour fcst:"
# mucape_key            = "CAPE:255-0 mb above ground:hour fcst:"
sbcin_key             = "CIN:surface:hour fcst:"
mlcin_key             = "CIN:90-0 mb above ground:hour fcst:"
# mulayercin_key        = "CIN:180-0 mb above ground:hour fcst:"
# mucin_key             = "CIN:255-0 mb above ground:hour fcst:"
helicity1km_key       = "HLCY:1000-0 m above ground:hour fcst:"
helicity3km_key       = "HLCY:3000-0 m above ground:hour fcst:"

function compute_0_6km_BWD(inventory, data)
  diff_u = get_layer(data, inventory, "VUCSH:6000-0 m above ground:hour fcst:")
  diff_v = get_layer(data, inventory, "VVCSH:6000-0 m above ground:hour fcst:")
  sqrt.(diff_u.^2 .+ diff_v.^2)
end

# Some early experiments showed a preference for lifted index over CAPE, so try that here too
interaction_terms = [
  # 0-1km EHI, roughly
  (    "SBCAPE*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* get_layer(data, inventory, helicity1km_key)),
  (    "MLCAPE*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* get_layer(data, inventory, helicity1km_key)),
  (      "LFTX*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* get_layer(data, inventory, helicity1km_key)),
  ("sqrtSBCAPE*HLCY1000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* get_layer(data, inventory, helicity1km_key)),
  ("sqrtMLCAPE*HLCY1000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* get_layer(data, inventory, helicity1km_key)),

  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* get_layer(data, inventory, helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* get_layer(data, inventory, helicity3km_key)),
  (      "LFTX*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* get_layer(data, inventory, helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* get_layer(data, inventory, helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* get_layer(data, inventory, helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-6km", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* compute_0_6km_BWD(inventory, data)),
  (    "MLCAPE*BWD0-6km", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* compute_0_6km_BWD(inventory, data)),
  (      "LFTX*BWD0-6km", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* compute_0_6km_BWD(inventory, data)),
  ("sqrtSBCAPE*BWD0-6km", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* compute_0_6km_BWD(inventory, data)),
  ("sqrtMLCAPE*BWD0-6km", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* compute_0_6km_BWD(inventory, data)),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  (    "MLCAPE*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  (      "LFTX*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtMLCAPE*(200+MLCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* (200f0 .+ get_layer(data, inventory, mlcin_key))),

  (    "SBCAPE*HLCY1000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  (    "MLCAPE*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  (      "LFTX*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  ("sqrtSBCAPE*HLCY1000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtMLCAPE*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key)),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key)),
  (      "LFTX*BWD0-6km*HLCY1000-0m", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key)),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key)),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key)),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)       .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, mlcape_key)       .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  (      "LFTX*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) ->       get_layer(data, inventory, lifted_index_key) .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key))      .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, mlcape_key))      .* compute_0_6km_BWD(inventory, data) .* get_layer(data, inventory, helicity1km_key) .* (200f0 .+ get_layer(data, inventory, mlcin_key))),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = interaction_terms
  )
end

# Don't have 3-hour windows of data from before mid-2018?
# Have to be content with 2-hour windows or get the old data.
function three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts_with_climatology(feature_engineered_forecasts())
end

# function forecasts_at_forecast_hour(forecast_hour)
#   filter(forecast -> forecast.forecast_hour == forecast_hour, forecasts())
# end

low_level_energy_fields_start_seconds = Int64(Dates.datetime2unix(Dates.DateTime(2014, 2, 25, 12)))

function forecast_has_low_level_energy_fields(forecast)
  Forecasts.run_time_in_seconds_since_epoch_utc(forecast) >= low_level_energy_fields_start_seconds
end

missing_convective_cloud_top_start_seconds = Int64(Dates.datetime2unix(Dates.DateTime(2018, 7, 12, 12)))
missing_convective_cloud_top_stop_seconds  = Int64(Dates.datetime2unix(Dates.DateTime(2018, 8, 10, 13)))
missing_convective_cloud_top_seconds_range = missing_convective_cloud_top_start_seconds:missing_convective_cloud_top_stop_seconds

function forecast_has_convective_cloud_top_field(forecast)
  !(Forecasts.run_time_in_seconds_since_epoch_utc(forecast) in missing_convective_cloud_top_seconds_range)
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
  rap_paths = vcat(
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/RAP_1/rap"),
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/RAP_3/rap")
  )

  global _forecasts

  _forecasts = []

  grid = nothing

  for rap_path in rap_paths
    # println(rap_path)
    # "/Volumes/RAP_1/rap/201402/20140225/rap_130_20140225_1300_012.grb2"

    if occursin("rap_130_", rap_path) # Skip old ruc files
      year_str, month_str, day_str, run_hour_str, forecast_hour_str = match(r"/rap_130_(\d\d\d\d)(\d\d)(\d\d)_(\d\d)00_0(\d\d)\.gri?b2", rap_path).captures

      run_year      = parse(Int64, year_str)
      run_month     = parse(Int64, month_str)
      run_day       = parse(Int64, day_str)
      run_hour      = parse(Int64, run_hour_str)
      forecast_hour = parse(Int64, forecast_hour_str)

      if isnothing(grid)
        grid = Grib2.read_grid(rap_path, crop = crop, downsample = downsample)
      end

      get_inventory() = begin
        # Somewhat inefficient that each hour must trigger wgrib2 on the same file...prefer using Forecasts.inventory(example_forecast()) if you don't need the particular file's exact byte locations of the layers
        inventory = Grib2.read_inventory(rap_path)

        layer_key_to_inventory_line(key) = begin
          i = findfirst(line -> Inventories.inventory_line_key(line) == key, inventory)
          if i != nothing
            inventory[i]
          else
            throw("RAP forecast $(Forecasts.time_title(run_year, run_month, run_day, run_hour, forecast_hour)) does not have $key: $inventory")
          end
        end

        inventory_to_use = map(layer_key_to_inventory_line, common_layers)

        inventory_to_use
      end

      get_data() = begin
        Grib2.read_layers_data_raw(rap_path, get_inventory(), crop_downsample_grid = grid)
      end

      preload_paths = [rap_path]

      forecast = Forecasts.Forecast("RAP", run_year, run_month, run_day, run_hour, forecast_hour, [], grid, get_inventory, get_data, preload_paths)

      push!(_forecasts, forecast)
    end
  end

  println("filtering")
  _forecasts = filter(forecast_has_low_level_energy_fields, _forecasts)
  _forecasts = filter(forecast_has_convective_cloud_top_field, _forecasts)
  print("sorting...")
  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))
  println("done")

  _forecasts
end

end # module RAP
