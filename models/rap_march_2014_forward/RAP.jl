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

_twenty_five_mi_mean_is2 = nothing

function get_twenty_five_mi_mean_is2()
  global _twenty_five_mi_mean_is2
  if isnothing(_twenty_five_mi_mean_is2)
    _twenty_five_mi_mean_is2, _, _ = FeatureEngineeringShared.compute_mean_is2(grid())
  end
  _twenty_five_mi_mean_is2
end


# I think https://rapidrefresh.noaa.gov/hrrr/GRIB2Table_hrrrncep_2d.txt explains the CAPE differences

# These term names end up being the same as the HRRR

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
surface_layer_u_key   = "UGRD:30-0 mb above ground:hour fcst:"
surface_layer_v_key   = "VGRD:30-0 mb above ground:hour fcst:"

dpt_key               = "DPT:2 m above ground:hour fcst:"
rh_key                = "RH:30-0 mb above ground:hour fcst:"
percip_rate_key       = "PRATE:surface:hour fcst:"

function compute_0_6km_BWD!(get_layer, out)
  diff_u = get_layer("VUCSH:6000-0 m above ground:hour fcst:")
  diff_v = get_layer("VVCSH:6000-0 m above ground:hour fcst:")
  out .= sqrt.(diff_u.^2 .+ diff_v.^2)

  ()
end

function meanify_25mi(grid, feature_data)
  FeatureEngineeringShared.meanify_threaded2(grid, feature_data, get_twenty_five_mi_mean_is2())
end

# Upstream feature gated by SCP > 0.1
# Computed output gated by SCP > 1
# also returns out
function storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, key; hours)
  out .=
    FeatureEngineeringShared.compute_upstream_mean_threaded(
      grid         = grid,
      u_data       = get_layer("USTM:6000-0 m above ground:hour fcst:"),
      v_data       = get_layer("VSTM:6000-0 m above ground:hour fcst:"),
      feature_data = meanify_25mi(grid, get_layer(key) .* Float32.(get_layer("SCPish(RM)") .> 0.1f0)),
      hours        = hours,
      out          = out
    ) .* get_layer("SCPish(RM)>1")
end

function upstream_feature!(grid, get_layer, out, u_key, v_key, feature_key; hours)
  FeatureEngineeringShared.compute_upstream_mean_threaded(
    grid         = grid,
    u_data       = get_layer(u_key),
    v_data       = get_layer(v_key),
    feature_data = get_layer(feature_key),
    hours        = hours,
    out          = out
  )
end

# Some early experiments showed a preference for lifted index over CAPE, so try that here too
interaction_terms = [
  ("BWD0-6km", (_, get_layer, out) -> compute_0_6km_BWD!(get_layer, out)),

  # 0-1km EHI, roughly
  (    "SBCAPE*HLCY1000-0m", (_, get_layer, out) -> out .=       get_layer(sbcape_key)       .* get_layer(helicity1km_key)),
  (    "MLCAPE*HLCY1000-0m", (_, get_layer, out) -> out .=       get_layer(mlcape_key)       .* get_layer(helicity1km_key)),
  (      "LFTX*HLCY1000-0m", (_, get_layer, out) -> out .=       get_layer(lifted_index_key) .* get_layer(helicity1km_key)),
  ("sqrtSBCAPE*HLCY1000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key))      .* get_layer(helicity1km_key)),
  ("sqrtMLCAPE*HLCY1000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key))      .* get_layer(helicity1km_key)),

  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(sbcape_key)       .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(mlcape_key)       .* get_layer(helicity3km_key)),
  (      "LFTX*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(lifted_index_key) .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key))      .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key))      .* get_layer(helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-6km", (_, get_layer, out) -> out .=       get_layer(sbcape_key)       .* get_layer("BWD0-6km")),
  (    "MLCAPE*BWD0-6km", (_, get_layer, out) -> out .=       get_layer(mlcape_key)       .* get_layer("BWD0-6km")),
  (      "LFTX*BWD0-6km", (_, get_layer, out) -> out .=       get_layer(lifted_index_key) .* get_layer("BWD0-6km")),
  ("sqrtSBCAPE*BWD0-6km", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key))      .* get_layer("BWD0-6km")),
  ("sqrtMLCAPE*BWD0-6km", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key))      .* get_layer("BWD0-6km")),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .=       get_layer(sbcape_key)       .* (200f0 .+ get_layer(sbcin_key))),
  (    "MLCAPE*(200+MLCIN)", (_, get_layer, out) -> out .=       get_layer(mlcape_key)       .* (200f0 .+ get_layer(mlcin_key))),
  (      "LFTX*(200+MLCIN)", (_, get_layer, out) -> out .=       get_layer(lifted_index_key) .* (200f0 .+ get_layer(mlcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key))      .* (200f0 .+ get_layer(sbcin_key))),
  ("sqrtMLCAPE*(200+MLCIN)", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key))      .* (200f0 .+ get_layer(mlcin_key))),

  (    "SBCAPE*HLCY1000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*(200+SBCIN)") .* get_layer(helicity1km_key)),
  (    "MLCAPE*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*(200+MLCIN)") .* get_layer(helicity1km_key)),
  (      "LFTX*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(      "LFTX*(200+MLCIN)") .* get_layer(helicity1km_key)),
  ("sqrtSBCAPE*HLCY1000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*(200+SBCIN)") .* get_layer(helicity1km_key)),
  ("sqrtMLCAPE*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*(200+MLCIN)") .* get_layer(helicity1km_key)),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*HLCY1000-0m") .* get_layer("BWD0-6km")),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*HLCY1000-0m") .* get_layer("BWD0-6km")),
  (      "LFTX*BWD0-6km*HLCY1000-0m", (_, get_layer, out) -> out .= get_layer(      "LFTX*HLCY1000-0m") .* get_layer("BWD0-6km")),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*HLCY1000-0m") .* get_layer("BWD0-6km")),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*HLCY1000-0m") .* get_layer("BWD0-6km")),

  (    "SBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*HLCY1000-0m*(200+SBCIN)") .* get_layer("BWD0-6km")),
  (    "MLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),
  (      "LFTX*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(      "LFTX*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),
  ("sqrtSBCAPE*BWD0-6km*HLCY1000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*HLCY1000-0m*(200+SBCIN)") .* get_layer("BWD0-6km")),
  ("sqrtMLCAPE*BWD0-6km*HLCY1000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*HLCY1000-0m*(200+MLCIN)") .* get_layer("BWD0-6km")),

  ("SCPish(RM)", (_, get_layer, out) -> out .= get_layer(mulayercape_key) .* get_layer(helicity3km_key) .* get_layer("BWD0-6km") .* (1f0 / (1000f0 * 50f0 * 20f0))),

  ("Divergence10m*10^5"   , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:"   ), get_layer("VGRD:10 m above ground:hour fcst:"   ))),
  ("Divergence30-0mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:30-0 mb above ground:hour fcst:"), get_layer("VGRD:30-0 mb above ground:hour fcst:"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:"), get_layer("VGRD:850 mb:hour fcst:"))),
  ("Divergence250mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:"), get_layer("VGRD:250 mb:hour fcst:"))),
  ("DifferentialDivergence250-850mb*10^5", (grid, get_layer, out) -> out .= get_layer("Divergence250mb*10^5") .- get_layer("Divergence850mb*10^5")),

  ("ConvergenceOnly10m*10^5"   , (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence10m*10^5"   ))),
  ("ConvergenceOnly30-0mb*10^5", (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence30-0mb*10^5"))),
  ("ConvergenceOnly850mb*10^5" , (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence850mb*10^5" ))),

  ("AbsVorticity10m*10^5"   , (grid, get_layer, out) -> FeatureEngineeringShared.compute_vorticity_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:"   ), get_layer("VGRD:10 m above ground:hour fcst:"   ))),
  ("AbsVorticity30-0mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_vorticity_threaded!(grid, out, get_layer("UGRD:30-0 mb above ground:hour fcst:"), get_layer("VGRD:30-0 mb above ground:hour fcst:"))),
  ("AbsVorticity850mb*10^5" , (grid, get_layer, out) -> FeatureEngineeringShared.compute_vorticity_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:"              ), get_layer("VGRD:850 mb:hour fcst:"              ))),
  ("AbsVorticity250mb*10^5" , (grid, get_layer, out) -> FeatureEngineeringShared.compute_vorticity_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:"              ), get_layer("VGRD:250 mb:hour fcst:"              ))),

  # Earlier experiments seemed to have trouble with conditions where supercells moved off fronts.
  #
  # Latent supercell indicator based on SCP and convergence upstream (following storm motion).
  #
  # Should follow goldilocks principle: too much and too little are both bad.

  ("SCPish(RM)>1", (grid, get_layer, out) -> out .= Float32.(meanify_25mi(grid, get_layer("SCPish(RM)")) .> 1f0)),

  ("StormUpstream10mConvergence3hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 3)),
  ("StormUpstream10mConvergence6hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 6)),
  ("StormUpstream10mConvergence9hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 9)),

  ("StormUpstream30-0mbConvergence3hrGatedBySCP"              , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly30-0mb*10^5"          , hours = 3)),
  ("StormUpstream30-0mbConvergence6hrGatedBySCP"              , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly30-0mb*10^5"          , hours = 6)),
  ("StormUpstream30-0mbConvergence9hrGatedBySCP"              , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly30-0mb*10^5"          , hours = 9)),

  ("StormUpstream850mbConvergence3hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 3)),
  ("StormUpstream850mbConvergence6hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 6)),
  ("StormUpstream850mbConvergence9hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 9)),

  ("StormUpstreamDifferentialDivergence250-850mb3hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 3))),
  ("StormUpstreamDifferentialDivergence250-850mb6hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 6))),
  ("StormUpstreamDifferentialDivergence250-850mb9hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 9))),

  # Low level upstream features. What kind of air is the storm ingesting?

  ("UpstreamSBCAPE1hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key      , surface_v_key      , sbcape_key      , hours = 1)),
  ("UpstreamSBCAPE2hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key      , surface_v_key      , sbcape_key      , hours = 2)),

  ("UpstreamMLCAPE1hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, mlcape_key      , hours = 1)),
  ("UpstreamMLCAPE2hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, mlcape_key      , hours = 2)),

  ("UpstreamLFTX1hr"  ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, lifted_index_key, hours = 1)),
  ("UpstreamLFTX2hr"  ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, lifted_index_key, hours = 2)),

  ("Upstream2mDPT1hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, dpt_key         , hours = 1)),
  ("Upstream2mDPT2hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, dpt_key         , hours = 2)),

  ("UpstreamSurfaceLayerRH1hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, rh_key          , hours = 1)),
  ("UpstreamSurfaceLayerRH2hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, rh_key          , hours = 2)),

  ("UpstreamPRATE1hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, percip_rate_key , hours = 1)),
  ("UpstreamPRATE2hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_layer_u_key, surface_layer_v_key, percip_rate_key , hours = 2)),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = interaction_terms,
    use_2020_models_buggy_100mi_calc = true
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

        inventory_line_keys = Inventories.inventory_line_key.(inventory) # avoid n^2 nasty allocs by precomputing this

        layer_key_to_inventory_line(key) = begin
          i = findfirst(isequal(key), inventory_line_keys)
          if !isnothing(i)
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
