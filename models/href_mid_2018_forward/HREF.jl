module HREF

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared
import FeatureEngineeringShared
import ThreeHourWindowForecasts

# Techincally, the HREF is on grid 227: http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID227
# Natively 1473x1025 (5km)
# BUT there's lots of missing data near the edges. The effective bounds of the grid appear to match the HRRR.
#
# See HREF_raw_usage.txt: it's not exactly square on the grid.
#
# We'll conservatively cut 214 off the W, 99 off the E, 119 off the S, 228 off the N
const crop = ((1+214):(1473 - 99), (1+119):(1025-228))


# Native:
#
# 1:0:grid_template=30:winds(grid):
# 	Lambert Conformal: (1473 x 1025) input WE:SN output WE:SN res 56
# 	Lat1 12.190000 Lon1 226.541000 LoV 265.000000
# 	LatD 25.000000 Latin1 25.000000 Latin2 25.000000
# 	LatSP 0.000000 LonSP 0.000000
# 	North Pole (1473 x 1025) Dx 5079.000000 m Dy 5079.000000 m mode 56
#
# Native: lambert:265.000000:25.000000:25.000000:25.000000 226.541000:1473:5079.000000 12.190000:1025:5079.000000
# Crop:   lambert:265.000000:25.000000:25.000000:25.000000 234.906000:387:15237.000000 19.858000:226:15237.000000
#
# $ wgrib2 href_one_field_for_grid.grib2 -new_grid_winds grid -new_grid lambert:265.000000:25.000000:25.000000:25.000000 234.906000:387:15237.000000 19.858000:226:15237.000000 href_one_field_for_grid_cropped_3x_downsampled.grib2
# $ wgrib2 href_one_field_for_grid_cropped_3x_downsampled.grib2 -gridout cropped_downsampled.csv


forecasts_root() = get(ENV, "FORECASTS_ROOT", "/Volumes")

const layer_blocks_to_make = FeatureEngineeringShared.fewer_grad_blocks

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

const vector_wind_layers = [
  "GRD:250 mb:hour fcst:wt ens mean",
  "GRD:500 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:925 mb:hour fcst:wt ens mean",
  "STM", # Our computed Bunkers storm motion.
  "STM½", # half as much deviation from the mean wind
  "½STM½500mb", # mean between bunkers and 500mb wind
  "SHEAR",
  "MEAN",
]

const downsample = 3 # 3x downsample, roughly 15km grid.

# lambert:$lov:$latin1:$latin2:$latd $lon1:$nx:$dx $lat1:$ny:$dy
# wgrib2 IN.grib -new_grid `grid_defn.pl output_grid.grb` OUT.grib

_forecasts = []

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


const sbcape_key     = "CAPE:surface:hour fcst:wt ens mean"
const mlcape_key     = "CAPE:90-0 mb above ground:hour fcst:wt ens mean"
const mulayercape_key = "CAPE:180-0 mb above ground:hour fcst:wt ens mean"
const sbcin_key      = "CIN:surface:hour fcst:wt ens mean"
const mlcin_key      = "CIN:90-0 mb above ground:hour fcst:wt ens mean"
# const mulayercin_key = "CIN:180-0 mb above ground:hour fcst:wt ens mean"
const helicity3km_key = "HLCY:3000-0 m above ground:hour fcst:wt ens mean"
const bwd_0_6km_key   = "VWSH:6000-0 m above ground:hour fcst:wt ens mean"

const tmp_500mb_key   = "TMP:500 mb:hour fcst:wt ens mean"

const low_level_u_key = "UGRD:925 mb:hour fcst:wt ens mean"
const low_level_v_key = "VGRD:925 mb:hour fcst:wt ens mean"

const dpt_key  = "DPT:2 m above ground:hour fcst:wt ens mean"
const rain_key = "CRAIN:surface:hour fcst:wt ens mean"

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
      u_data       = get_layer("USTM"),
      v_data       = get_layer("VSTM"),
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


const extra_features = [
  ("700-500mbLapseRate", (_, get_layer, out) -> FeatureEngineeringShared.lapse_rate_from_ensemble_mean!(get_layer, out, "700 mb", "500 mb")),
  ("925-700mbLapseRate", (_, get_layer, out) -> FeatureEngineeringShared.lapse_rate_from_ensemble_mean!(get_layer, out, "925 mb", "700 mb")),

  ("Wind700mb", (_, get_layer, out) -> out .= sqrt.(get_layer("UGRD:700 mb:hour fcst:wt ens mean").^2 .+ get_layer("VGRD:700 mb:hour fcst:wt ens mean").^2) ),
  ("Wind500mb", (_, get_layer, out) -> out .= sqrt.(get_layer("UGRD:500 mb:hour fcst:wt ens mean").^2 .+ get_layer("VGRD:500 mb:hour fcst:wt ens mean").^2) ),

  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(mlcape_key)  .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key)) .* get_layer(helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-6km", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* get_layer(bwd_0_6km_key)),
  (    "MLCAPE*BWD0-6km", (_, get_layer, out) -> out .=       get_layer(mlcape_key)  .* get_layer(bwd_0_6km_key)),
  ("sqrtSBCAPE*BWD0-6km", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* get_layer(bwd_0_6km_key)),
  ("sqrtMLCAPE*BWD0-6km", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key)) .* get_layer(bwd_0_6km_key)),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* (200f0 .+ get_layer(sbcin_key))),
  (    "MLCAPE*(200+MLCIN)", (_, get_layer, out) -> out .=       get_layer(mlcape_key)  .* (200f0 .+ get_layer(mlcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* (200f0 .+ get_layer(sbcin_key))),
  ("sqrtMLCAPE*(200+MLCIN)", (_, get_layer, out) -> out .= sqrt.(get_layer(mlcape_key)) .* (200f0 .+ get_layer(mlcin_key))),

  (    "SBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*(200+MLCIN)") .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*(200+MLCIN)") .* get_layer(helicity3km_key)),

  (    "SBCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  (    "MLCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*BWD0-6km") .* get_layer(helicity3km_key)),

  (    "SBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*HLCY3000-0m*(200+SBCIN)") .* get_layer(bwd_0_6km_key)),
  (    "MLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer(    "MLCAPE*HLCY3000-0m*(200+MLCIN)") .* get_layer(bwd_0_6km_key)),
  ("sqrtSBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)") .* get_layer(bwd_0_6km_key)),
  ("sqrtMLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN)", (_, get_layer, out) -> out .= get_layer("sqrtMLCAPE*HLCY3000-0m*(200+MLCIN)") .* get_layer(bwd_0_6km_key)),

  # SHIP terms
  # https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html
  # [(MUCAPE j/kg) * (Mixing Ratio of MU PARCEL g/kg) * (700-500mb LAPSE RATE c/km) * (-500mb TEMP C) * (0-6km Shear m/s) ] / 42,000,000
  # "0-6 km shear is confined to a range of 7-27 m s-1, mixing ratio is confined to a range of 11-13.6 g kg-1, and the 500 mb temperature is set to -5.5 C for any warmer values."
  # ignoring other adjustments

  ("700-500mbLapseRate*BWD0-6km",                      (_, get_layer, out) -> out .= clamp.(get_layer(bwd_0_6km_key), 7f0, 27f0) .* get_layer("700-500mbLapseRate")),
  ("700-500mbLapseRate*-Celcius500mb*BWD0-6km",        (_, get_layer, out) -> out .= get_layer("700-500mbLapseRate*BWD0-6km") .* (273.15f0 .- min.(get_layer(tmp_500mb_key), -5.5f0 + 273.15f0))),
  ("MUCAPE*BWD0-6km",                                  (_, get_layer, out) -> out .= get_layer(mulayercape_key) .* clamp.(get_layer(bwd_0_6km_key), 7f0, 27f0)),
  ("MUCAPE*700-500mbLapseRate*BWD0-6km",               (_, get_layer, out) -> out .= get_layer(mulayercape_key) .* get_layer("700-500mbLapseRate*BWD0-6km")),
  ("MUCAPE*700-500mbLapseRate*-Celcius500mb*BWD0-6km", (_, get_layer, out) -> out .= get_layer(mulayercape_key) .* get_layer("700-500mbLapseRate*-Celcius500mb*BWD0-6km")),

  ("MUCAPE*MixingRatio925mb*700-500mbLapseRate*-Celcius500mb*BWD0-6km", (_, get_layer, out) -> out .= get_layer("MUCAPE*700-500mbLapseRate*-Celcius500mb*BWD0-6km") .* clamp.(FeatureEngineeringShared.mixing_ratio.(get_layer("DPT:925 mb:hour fcst:wt ens mean"), 925f0), 11f0, 13.6f0)),
  ("MUCAPE*MixingRatio850mb*700-500mbLapseRate*-Celcius500mb*BWD0-6km", (_, get_layer, out) -> out .= get_layer("MUCAPE*700-500mbLapseRate*-Celcius500mb*BWD0-6km") .* clamp.(FeatureEngineeringShared.mixing_ratio.(get_layer("DPT:850 mb:hour fcst:wt ens mean"), 850f0), 11f0, 13.6f0)),

  # low level wind fields
  # Andy Wade's idea
  ("MaxWind<=850mb", (_, get_layer, out) -> out .= max.(get_layer("WIND:10 m above ground:hour fcst:wt ens mean"), get_layer("WIND:80 m above ground:hour fcst:wt ens mean"), get_layer("WIND:925 mb:hour fcst:wt ens mean"), get_layer("WIND:850 mb:hour fcst:wt ens mean"))),
  ("SumWind<=850mb", (_, get_layer, out) -> out .=   .+(get_layer("WIND:10 m above ground:hour fcst:wt ens mean"), get_layer("WIND:80 m above ground:hour fcst:wt ens mean"), get_layer("WIND:925 mb:hour fcst:wt ens mean"), get_layer("WIND:850 mb:hour fcst:wt ens mean"))),
  ("MaxWind<=700mb", (_, get_layer, out) -> out .= max.(get_layer("MaxWind<=850mb"), get_layer("Wind700mb"))),
  ("SumWind<=700mb", (_, get_layer, out) -> out .=   .+(get_layer("SumWind<=850mb"), get_layer("Wind700mb"))),

  ("SCPish(RM)", (_, get_layer, out) -> out .= get_layer(mulayercape_key) .* get_layer(helicity3km_key) .* get_layer(bwd_0_6km_key) .* (1f0 / (1000f0 * 50f0 * 20f0))),

  ("Divergence925mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:925 mb:hour fcst:wt ens mean"), get_layer("VGRD:925 mb:hour fcst:wt ens mean"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("Divergence250mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),
  ("DifferentialDivergence250-925mb*10^5", (grid, get_layer, out) -> out .= get_layer("Divergence250mb*10^5") - get_layer("Divergence925mb*10^5")),

  ("ConvergenceOnly925mb*10^5", (grid, get_layer, out) -> out .= max.(0f0, .-get_layer("Divergence925mb*10^5"))),
  ("ConvergenceOnly850mb*10^5", (grid, get_layer, out) -> out .= max.(0f0, .-get_layer("Divergence850mb*10^5"))),

  ("AbsVorticity925mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:925 mb:hour fcst:wt ens mean"), get_layer("VGRD:925 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity850mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity700mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:700 mb:hour fcst:wt ens mean"), get_layer("VGRD:700 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity500mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:500 mb:hour fcst:wt ens mean"), get_layer("VGRD:500 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity250mb*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),

  # Estimating Bunkers Storm motion
  #
  # The unusual weighting was chosen by lots of guess and check, trying to match RAP's storm motion. We don't have a lot of wind layers to work with.
  ("UMEAN", (_, get_layer, out) -> out .= (1/3.75f0) .* (get_layer("UGRD:925 mb:hour fcst:wt ens mean") .+ get_layer("UGRD:850 mb:hour fcst:wt ens mean") .+ 0.5f0 .* get_layer("UGRD:700 mb:hour fcst:wt ens mean") .+ get_layer("UGRD:500 mb:hour fcst:wt ens mean") .+ 0.05f0 .* get_layer("UGRD:250 mb:hour fcst:wt ens mean"))),
  ("VMEAN", (_, get_layer, out) -> out .= (1/3.75f0) .* (get_layer("VGRD:925 mb:hour fcst:wt ens mean") .+ get_layer("VGRD:850 mb:hour fcst:wt ens mean") .+ 0.5f0 .* get_layer("VGRD:700 mb:hour fcst:wt ens mean") .+ get_layer("VGRD:500 mb:hour fcst:wt ens mean") .+ 0.05f0 .* get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),

  # *Supposed* to be 250m - 5750m shear vector.
  ("USHEAR", (_, get_layer, out) -> out .= 0.05f0 .* get_layer("UGRD:250 mb:hour fcst:wt ens mean") .+ 0.95f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean") .- 0.93f0 .* get_layer("UGRD:925 mb:hour fcst:wt ens mean")),
  ("VSHEAR", (_, get_layer, out) -> out .= 0.05f0 .* get_layer("VGRD:250 mb:hour fcst:wt ens mean") .+ 0.95f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean") .- 0.93f0 .* get_layer("VGRD:925 mb:hour fcst:wt ens mean")),

  ("SHEAR", (_, get_layer, out) -> out .= sqrt.(get_layer("USHEAR").^2 .+ get_layer("VSHEAR").^2) .+ eps(1f0)),

  # mean + D(shear x k) / |shear|
  # D = 7.5 m/s

  ("USTM", (_, get_layer, out) -> out .= get_layer("UMEAN") .+ 7.5f0 .* get_layer("VSHEAR") ./ (get_layer("SHEAR") .+ 0.25f0)),
  ("VSTM", (_, get_layer, out) -> out .= get_layer("VMEAN") .- 7.5f0 .* get_layer("USHEAR") ./ (get_layer("SHEAR") .+ 0.25f0)),

  ("USTM½", (_, get_layer, out) -> out .= get_layer("UMEAN") .+ 3.25f0 .* get_layer("VSHEAR") ./ (get_layer("SHEAR") .+ 0.25f0)),
  ("VSTM½", (_, get_layer, out) -> out .= get_layer("VMEAN") .- 3.25f0 .* get_layer("USHEAR") ./ (get_layer("SHEAR") .+ 0.25f0)),

  ("U½STM½500mb", (_, get_layer, out) -> out .= 0.5f0 .* get_layer("USTM") .+ 0.5f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean")),
  ("V½STM½500mb", (_, get_layer, out) -> out .= 0.5f0 .* get_layer("VSTM") .+ 0.5f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean")),

  # Earlier experiments seemed to have trouble with conditions where supercells moved off fronts.
  #
  # Latent supercell indicator based on SCP and convergence upstream (following storm motion).
  #
  # Should follow goldilocks principle: too much and too little are both bad.

  ("SCPish(RM)>1", (grid, get_layer, out) -> out .= Float32.(meanify_25mi(grid, get_layer("SCPish(RM)")) .> 1f0)),

  ("StormUpstream925mbConvergence3hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly925mb*10^5"           , hours = 3)),
  ("StormUpstream925mbConvergence6hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly925mb*10^5"           , hours = 6)),
  ("StormUpstream925mbConvergence9hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly925mb*10^5"           , hours = 9)),

  ("StormUpstream850mbConvergence3hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 3)),
  ("StormUpstream850mbConvergence6hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 6)),
  ("StormUpstream850mbConvergence9hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 9)),

  ("StormUpstreamDifferentialDivergence250-925mb3hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-925mb*10^5", hours = 3))),
  ("StormUpstreamDifferentialDivergence250-925mb6hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-925mb*10^5", hours = 6))),
  ("StormUpstreamDifferentialDivergence250-925mb9hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-925mb*10^5", hours = 9))),

  # Low level upstream features. What kind of air is the storm ingesting?

  ("UpstreamSBCAPE2hr", (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, low_level_u_key, low_level_v_key, sbcape_key, hours = 2)),

  ("UpstreamMLCAPE2hr", (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, low_level_u_key, low_level_v_key, mlcape_key, hours = 2)),

  ("Upstream2mDPT1hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, low_level_u_key, low_level_v_key, dpt_key   , hours = 1)),
  ("Upstream2mDPT2hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, low_level_u_key, low_level_v_key, dpt_key   , hours = 2)),

  ("UpstreamCRAIN2hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, low_level_u_key, low_level_v_key, rain_key  , hours = 2)),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = extra_features
  )
end

function extra_features_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = String[],
    layer_blocks_to_make = [FeatureEngineeringShared.raw_features_block],
    new_features_pre = extra_features
  )
end


# $ wgrib2 href_conus_20210511_t06z_prob_f01.grib2 | grep UPHL
# 24:5624584:d=2021051106:UPHL:5000-2000 m above ground:1 hour fcst:prob >25:prob fcst 0/8:Neighborhood Probability
# 25:5820903:d=2021051106:UPHL:5000-2000 m above ground:1 hour fcst:prob >100:prob fcst 0/8:Neighborhood Probability
# 29:6707470:d=2021051106:MXUPHL:5000-2000 m above ground:1 hour fcst:prob >25:prob fcst 0/8:Neighborhood Probability
# 30:6912528:d=2021051106:MXUPHL:5000-2000 m above ground:1 hour fcst:prob >100:prob fcst 0/8:Neighborhood Probability
# $ wgrib2 href_conus_20210511_t12z_prob_f01.grib2 | grep UPHL
# 25:5484237:d=2021051112:MXUPHL:5000-2000 m above ground:1 hour fcst:prob >25:prob fcst 0/10:Neighborhood Probability
# 26:5687886:d=2021051112:MXUPHL:5000-2000 m above ground:1 hour fcst:prob >75:prob fcst 0/10:Neighborhood Probability
# 27:5881021:d=2021051112:MXUPHL:5000-2000 m above ground:1 hour fcst:prob >150:prob fcst 0/10:Neighborhood Probability

const HREFv3_implementation_datetime = Dates.DateTime(2021,5,11,12)

function run_datetime_to_simulation_version(datetime)
  if datetime >= HREFv3_implementation_datetime
    3
  else
    2 # I don't know when v2.1 started or what it is
  end
end

function three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts_with_climatology_etc(feature_engineered_forecasts(); run_datetime_to_simulation_version = run_datetime_to_simulation_version)
end

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

_original_grid_cropped = nothing

function original_grid_cropped()
  global _original_grid_cropped

  if isnothing(_original_grid_cropped)
    href_paths = Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/href")
    _original_grid_cropped = Grib2.read_grid(href_paths[1], crop = crop)
  end

  _original_grid_cropped
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

# Least probable threshold must be first.
const mean_layers_to_compute_from_prob = [
  [ # Because of HREF changes, not all these fields may be available. We use what is there. (HREFv2 had prob >100, HREFv3 replaced it with >75 and >150.)
    ("MXUPHL:5000-2000 m above ground:hour fcst:prob >150", 150f0),
    ("MXUPHL:5000-2000 m above ground:hour fcst:prob >100", 100f0),
    ("MXUPHL:5000-2000 m above ground:hour fcst:prob >75", 75f0),
    ("MXUPHL:5000-2000 m above ground:hour fcst:prob >25", 25f0)
  ],
  [
    ("REFD:1000 m above ground:hour fcst:prob >50", 50f0),
    ("REFD:1000 m above ground:hour fcst:prob >40", 40f0),
    ("REFD:1000 m above ground:hour fcst:prob >30", 30f0),
  ],
  [
    ("MAXREF:1000 m above ground:hour fcst:prob >50", 50f0),
    ("MAXREF:1000 m above ground:hour fcst:prob >40", 40f0),
  ],
  [
    ("REFC:entire atmosphere:hour fcst:prob >50", 50f0),
    ("REFC:entire atmosphere:hour fcst:prob >40", 40f0),
    ("REFC:entire atmosphere:hour fcst:prob >30", 30f0),
    ("REFC:entire atmosphere:hour fcst:prob >20", 20f0),
    ("REFC:entire atmosphere:hour fcst:prob >10", 10f0),
  ],
  [
    ("RETOP:entire atmosphere:hour fcst:prob >15240", 15240f0),
    ("RETOP:entire atmosphere:hour fcst:prob >12192", 12192f0),
    ("RETOP:entire atmosphere:hour fcst:prob >10668", 10668f0),
    ("RETOP:entire atmosphere:hour fcst:prob >9144", 9144f0),
    ("RETOP:entire atmosphere:hour fcst:prob >6096", 6096f0),
  ],
  [
    ("MAXUVV:400-1000 mb:hour fcst:prob >20", 20f0),
    ("MAXUVV:400-1000 mb:hour fcst:prob >10", 10f0),
    ("MAXUVV:400-1000 mb:hour fcst:prob >1", 1f0),
  ]
]


function reload_forecasts()
  common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
  common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

  href_paths =
    if get(ENV, "USE_ALT_DISK", "false") == "true"
      println("Using SREF_HREF_2 and SREF_HREF_4")
      vcat(
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_2/href"),
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_4/href")
      )
    else
      vcat(
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/href"),
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_3/href")
      )
    end

  global _forecasts

  _forecasts = []

  grid = nothing

  for href_path in href_paths
    # "/Volumes/SREF_HREF_1/href/201807/20180728/href_conus_20180728_t06z_mean_f15.grib2"

    if isnothing(grid)
      grid = Grib2.read_grid(href_path, crop = crop, downsample = downsample) # mean and prob better have the same grid!
    end

    if occursin("z_mean_f", href_path)
      mean_href_path = href_path
      prob_href_path = replace(mean_href_path, "z_mean_f" => "z_prob_f")

      forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("HREF", mean_href_path, prob_href_path, common_layers_mean, common_layers_prob, mean_layers_to_compute_from_prob, grid = grid)

      push!(_forecasts, forecast)
    end
  end

  _forecasts
end

end # module HREF