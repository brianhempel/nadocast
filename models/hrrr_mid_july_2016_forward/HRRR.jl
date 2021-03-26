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

_twenty_five_mi_mean_is = nothing

function get_twenty_five_mi_mean_is()
  global _twenty_five_mi_mean_is
  if isnothing(_twenty_five_mi_mean_is)
    _twenty_five_mi_mean_is, _, _ = FeatureEngineeringShared.compute_mean_is(grid())
  end
  _twenty_five_mi_mean_is
end

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

# I think https://rapidrefresh.noaa.gov/hrrr/GRIB2Table_hrrrncep_2d.txt explains the CAPE differences

lifted_index_key      = "LFTX:500-1000 mb:hour fcst:" # surface
# best_lifted_index_key = "4LFTX:180-0 mb above ground:hour fcst:"
sbcape_key            = "CAPE:surface:hour fcst:"
mlcape_key            = "CAPE:90-0 mb above ground:hour fcst:"
mulayercape_key       = "CAPE:180-0 mb above ground:hour fcst:"
# mucape_key            = "CAPE:255-0 mb above ground:hour fcst:"
sbcin_key             = "CIN:surface:hour fcst:"
mlcin_key             = "CIN:90-0 mb above ground:hour fcst:"
# mulayercin_key        = "CIN:180-0 mb above ground:hour fcst:"
# mucin_key             = "CIN:255-0 mb above ground:hour fcst:"
helicity1km_key       = "HLCY:1000-0 m above ground:hour fcst:"
helicity3km_key       = "HLCY:3000-0 m above ground:hour fcst:"

surface_u_key         = "UGRD:10 m above ground:hour fcst:"
surface_v_key         = "VGRD:10 m above ground:hour fcst:"
above_surface_u_key   = "UGRD:80 m above ground:hour fcst:"
above_surface_v_key   = "VGRD:80 m above ground:hour fcst:"

dpt_key               = "DPT:2 m above ground:hour fcst:"
rh_key                = "RH:2 m above ground:hour fcst:"
percip_rate_key       = "PRATE:surface:hour fcst:"


function compute_0_6km_BWD(get_layer)
  diff_u = get_layer("VUCSH:6000-0 m above ground:hour fcst:")
  diff_v = get_layer("VVCSH:6000-0 m above ground:hour fcst:")
  sqrt.(diff_u.^2 .+ diff_v.^2)
end

function meanify_25mi(feature_data)
  FeatureEngineeringShared.meanify_threaded(feature_data, get_twenty_five_mi_mean_is())
end

# Upstream feature gated by SCP > 0.1
# Computed output gated by SCP > 1
function storm_upstream_feature_gated_by_SCP(grid, get_layer, key; hours)
  FeatureEngineeringShared.compute_upstream_mean_threaded(
    grid         = grid,
    u_data       = get_layer("USTM:6000-0 m above ground:hour fcst:"),
    v_data       = get_layer("VSTM:6000-0 m above ground:hour fcst:"),
    feature_data = meanify_25mi(get_layer(key) .* Float32.(get_layer("SCPish(RM)") .> 0.1f0)),
    hours        = hours
  ) .* get_layer("SCPish(RM)>1")
end

function upstream_feature(grid, get_layer, u_key, v_key, feature_key; hours)
  FeatureEngineeringShared.compute_upstream_mean_threaded(
    grid         = grid,
    u_data       = get_layer(u_key),
    v_data       = get_layer(v_key),
    feature_data = get_layer(feature_key),
    hours        = hours
  )
end

# Some early experiments showed a preference for lifted index over CAPE, so try that here too
interaction_terms = [
  ("BWD0-6km", (_, get_layer) -> compute_0_6km_BWD(get_layer)),

  # 0-1km EHI, roughly
  (    "SBCAPE*HLCY1000-0m", (_, get_layer) ->       get_layer(sbcape_key)       .* get_layer(helicity1km_key)),
  (    "MLCAPE*HLCY1000-0m", (_, get_layer) ->       get_layer(mlcape_key)       .* get_layer(helicity1km_key)),
  (      "LFTX*HLCY1000-0m", (_, get_layer) ->       get_layer(lifted_index_key) .* get_layer(helicity1km_key)),
  ("sqrtSBCAPE*HLCY1000-0m", (_, get_layer) -> sqrt.(get_layer(sbcape_key))      .* get_layer(helicity1km_key)),
  ("sqrtMLCAPE*HLCY1000-0m", (_, get_layer) -> sqrt.(get_layer(mlcape_key))      .* get_layer(helicity1km_key)),

  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, get_layer) ->       get_layer(sbcape_key)       .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m", (_, get_layer) ->       get_layer(mlcape_key)       .* get_layer(helicity3km_key)),
  (      "LFTX*HLCY3000-0m", (_, get_layer) ->       get_layer(lifted_index_key) .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, get_layer) -> sqrt.(get_layer(sbcape_key))      .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m", (_, get_layer) -> sqrt.(get_layer(mlcape_key))      .* get_layer(helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-6km", (_, get_layer) ->       get_layer(sbcape_key)       .* get_layer("BWD0-6km")),
  (    "MLCAPE*BWD0-6km", (_, get_layer) ->       get_layer(mlcape_key)       .* get_layer("BWD0-6km")),
  (      "LFTX*BWD0-6km", (_, get_layer) ->       get_layer(lifted_index_key) .* get_layer("BWD0-6km")),
  ("sqrtSBCAPE*BWD0-6km", (_, get_layer) -> sqrt.(get_layer(sbcape_key))      .* get_layer("BWD0-6km")),
  ("sqrtMLCAPE*BWD0-6km", (_, get_layer) -> sqrt.(get_layer(mlcape_key))      .* get_layer("BWD0-6km")),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, get_layer) ->       get_layer(sbcape_key)       .* (200f0 .+ get_layer(sbcin_key))),
  (    "MLCAPE*(200+MLCIN)", (_, get_layer) ->       get_layer(mlcape_key)       .* (200f0 .+ get_layer(mlcin_key))),
  (      "LFTX*(200+MLCIN)", (_, get_layer) ->       get_layer(lifted_index_key) .* (200f0 .+ get_layer(mlcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, get_layer) -> sqrt.(get_layer(sbcape_key))      .* (200f0 .+ get_layer(sbcin_key))),
  ("sqrtMLCAPE*(200+MLCIN)", (_, get_layer) -> sqrt.(get_layer(mlcape_key))      .* (200f0 .+ get_layer(mlcin_key))),

  (    "SBCAPE*HLCY1000-0m*(200+SBCIN)", (_, get_layer) -> get_layer(    "SBCAPE*(200+SBCIN)") .* get_layer(helicity1km_key)),
  (    "MLCAPE*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(    "MLCAPE*(200+MLCIN)") .* get_layer(helicity1km_key)),
  (      "LFTX*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(      "LFTX*(200+MLCIN)") .* get_layer(helicity1km_key)),
  ("sqrtSBCAPE*HLCY1000-0m*(200+SBCIN)", (_, get_layer) -> get_layer("sqrtSBCAPE*(200+SBCIN)") .* get_layer(helicity1km_key)),
  ("sqrtMLCAPE*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer("sqrtMLCAPE*(200+MLCIN)") .* get_layer(helicity1km_key)),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer) -> get_layer(    "SBCAPE*BWD0-6km") .* get_layer(helicity1km_key)),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer) -> get_layer(    "MLCAPE*BWD0-6km") .* get_layer(helicity1km_key)),
  (      "LFTX*BWD0-6km*HLCY1000-0m", (_, get_layer) -> get_layer(      "LFTX*BWD0-6km") .* get_layer(helicity1km_key)),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer) -> get_layer("sqrtSBCAPE*BWD0-6km") .* get_layer(helicity1km_key)),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer) -> get_layer("sqrtMLCAPE*BWD0-6km") .* get_layer(helicity1km_key)),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, get_layer) -> get_layer(    "SBCAPE*HLCY1000-0m*(200+SBCIN)") .* get_layer("BWD0-6km")),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(    "MLCAPE*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),
  (      "LFTX*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(      "LFTX*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, get_layer) -> get_layer("sqrtSBCAPE*HLCY1000-0m*(200+SBCIN)") .* get_layer("BWD0-6km")),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer) -> get_layer("sqrtMLCAPE*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),

  ("SCPish(RM)", (_, get_layer) -> get_layer(mulayercape_key) .* get_layer(helicity3km_key) .* get_layer("BWD0-6km") .* (1f0 / (1000f0 * 50f0 * 20f0))),

  ("Divergence10m*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:10 m above ground:hour fcst:"), get_layer("VGRD:10 m above ground:hour fcst:"))),
  ("Divergence80m*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:80 m above ground:hour fcst:"), get_layer("VGRD:80 m above ground:hour fcst:"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:850 mb:hour fcst:"), get_layer("VGRD:850 mb:hour fcst:"))),
  ("Divergence250mb*10^5"                , (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:250 mb:hour fcst:"), get_layer("VGRD:250 mb:hour fcst:"))),
  ("DifferentialDivergence250-850mb*10^5", (grid, get_layer) -> get_layer("Divergence250mb*10^5") - get_layer("Divergence850mb*10^5")),

  ("ConvergenceOnly10m*10^5"  , (grid, get_layer) -> max.(0f0, 0f0 .- get_layer("Divergence10m*10^5"  ))),
  ("ConvergenceOnly80m*10^5"  , (grid, get_layer) -> max.(0f0, 0f0 .- get_layer("Divergence80m*10^5"  ))),
  ("ConvergenceOnly850mb*10^5", (grid, get_layer) -> max.(0f0, 0f0 .- get_layer("Divergence850mb*10^5"))),

  ("AbsVorticity10m*10^5"  , (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:10 m above ground:hour fcst:"), get_layer("VGRD:10 m above ground:hour fcst:"))),
  ("AbsVorticity80m*10^5"  , (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:80 m above ground:hour fcst:"), get_layer("VGRD:80 m above ground:hour fcst:"))),
  ("AbsVorticity850mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:850 mb:hour fcst:"           ), get_layer("VGRD:850 mb:hour fcst:"           ))),
  ("AbsVorticity500mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:500 mb:hour fcst:"           ), get_layer("VGRD:500 mb:hour fcst:"           ))),
  ("AbsVorticity250mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:250 mb:hour fcst:"           ), get_layer("VGRD:250 mb:hour fcst:"           ))),

  # Earlier experiments seemed to have trouble with conditions where supercells moved off fronts.
  #
  # Latent supercell indicator based on SCP and convergence upstream (following storm motion).
  #
  # Should follow goldilocks principle: too much and too little are both bad.

  ("SCPish(RM)>1", (grid, get_layer) -> Float32.(meanify_25mi(get_layer("SCPish(RM)")) .> 1f0)),

  ("StormUpstream10mConvergence3hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly10m*10^5"             , hours = 3)),
  ("StormUpstream10mConvergence6hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly10m*10^5"             , hours = 6)),
  ("StormUpstream10mConvergence9hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly10m*10^5"             , hours = 9)),

  ("StormUpstream80mConvergence3hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly80m*10^5"             , hours = 3)),
  ("StormUpstream80mConvergence6hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly80m*10^5"             , hours = 6)),
  ("StormUpstream80mConvergence9hrGatedBySCP"                 , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly80m*10^5"             , hours = 9)),

  ("StormUpstream850mbConvergence3hrGatedBySCP"               , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly850mb*10^5"           , hours = 3)),
  ("StormUpstream850mbConvergence6hrGatedBySCP"               , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly850mb*10^5"           , hours = 6)),
  ("StormUpstream850mbConvergence9hrGatedBySCP"               , (grid, get_layer) ->           storm_upstream_feature_gated_by_SCP(grid, get_layer, "ConvergenceOnly850mb*10^5"           , hours = 9)),

  ("StormUpstreamDifferentialDivergence250-850mb3hrGatedBySCP", (grid, get_layer) -> max.(0f0, storm_upstream_feature_gated_by_SCP(grid, get_layer, "DifferentialDivergence250-850mb*10^5", hours = 3))),
  ("StormUpstreamDifferentialDivergence250-850mb6hrGatedBySCP", (grid, get_layer) -> max.(0f0, storm_upstream_feature_gated_by_SCP(grid, get_layer, "DifferentialDivergence250-850mb*10^5", hours = 6))),
  ("StormUpstreamDifferentialDivergence250-850mb9hrGatedBySCP", (grid, get_layer) -> max.(0f0, storm_upstream_feature_gated_by_SCP(grid, get_layer, "DifferentialDivergence250-850mb*10^5", hours = 9))),

  # Low level upstream features. What kind of air is the storm ingesting?

  ("UpstreamSBCAPE1hr", (grid, get_layer) -> upstream_feature(grid, get_layer, surface_u_key      , surface_v_key      , sbcape_key      , hours = 1)),
  ("UpstreamSBCAPE2hr", (grid, get_layer) -> upstream_feature(grid, get_layer, surface_u_key      , surface_v_key      , sbcape_key      , hours = 2)),

  ("UpstreamMLCAPE1hr", (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, mlcape_key      , hours = 1)),
  ("UpstreamMLCAPE2hr", (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, mlcape_key      , hours = 2)),

  ("UpstreamLFTX1hr"  , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, lifted_index_key, hours = 1)),
  ("UpstreamLFTX2hr"  , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, lifted_index_key, hours = 2)),

  ("Upstream2mDPT1hr" , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, dpt_key         , hours = 1)),
  ("Upstream2mDPT2hr" , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, dpt_key         , hours = 2)),

  ("Upstream2mRH1hr"  , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, rh_key          , hours = 1)),
  ("Upstream2mRH2hr"  , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, rh_key          , hours = 2)),

  ("UpstreamPRATE1hr" , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, percip_rate_key , hours = 1)),
  ("UpstreamPRATE2hr" , (grid, get_layer) -> upstream_feature(grid, get_layer, above_surface_u_key, above_surface_v_key, percip_rate_key , hours = 2)),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = interaction_terms
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
  # HRRR_1 contains runs through 2018-5
  # HRRR_2 contains runs 2018-6 and forward
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

    preload_paths = [hrrr_path]

    forecast = Forecasts.Forecast("HRRR", run_year, run_month, run_day, run_hour, forecast_hour, [], grid, get_inventory, get_data, preload_paths)

    push!(_forecasts, forecast)
  end

  print("sorting...")
  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))
  println("done")

  _forecasts
end

end # module HRRR
