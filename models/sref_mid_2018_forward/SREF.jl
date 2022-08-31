module SREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
# import ForecastCombinators
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared
import FeatureEngineeringShared
import ThreeHourWindowForecasts

# SREF is on grid 212: http://www.nco.ncep.noaa.gov/pmb/docs/on388/grids/grid212.gif
#
# To match the HREF grid, we'll cut 26 off the W, 12 off the E, 14 off the S, 28 off the N
crop = ((1+26):(185-12), (1+14):(129-28))


# import Conus
# import Inventories
# import PlotMap
# function debug_plot(forecast)
#   raw_layer_count = length(Forecasts.inventory(example_forecast())) + length(extra_features)

#   conus_mask = Conus.is_in_conus.(grid().latlons)
#   data = Forecasts.data(forecast)
#   inventory = Forecasts.inventory(forecast)
#   for layer_i in [1:raw_layer_count; rand(1:length(inventory), 20); length(inventory)-34:length(inventory)]
#     layer_data = @view data[:, layer_i]
#     lo, hi = extrema(@view layer_data[conus_mask])

#     str = Inventories.inventory_line_description(inventory[layer_i])
#     base_path = "feature_$(layer_i)_$(replace(str, r"[: ]" => "_"))_$(lo)_$(hi)"
#     println(base_path)

#     range = hi - lo
#     if range > 0
#       PlotMap.plot_fast(base_path, grid(), clamp.((layer_data .- lo) ./ range, 0f0, 1f0))
#     else
#       PlotMap.plot_fast(base_path, grid(), clamp.(layer_data .- lo, 0f0, 1f0))
#     end
#   end
# end

# data = SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]._get_data()
# inv  = SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]._get_inventory()

# us = data[:,201]
# vs = data[:,202]

# import Grids

# function kts(grid, us, vs, latlon)
#   u = us[Grids.latlon_to_closest_grid_i(grid, latlon)]
#   v = vs[Grids.latlon_to_closest_grid_i(grid, latlon)]
#   (u * 1.94, v * 1.94, 1.94 * sqrt(u^2 + v^2))
# end

# push!(LOAD_PATH, (@__DIR__) * "/../rap_march_2014_forward")

# import RAP

# rap_forecast = RAP.forecasts()[1]

# rap_data = rap_forecast._get_data()

# findall(line -> line.abbrev == "USTM", rap_forecast._get_inventory()) # 183
# findall(line -> line.abbrev == "VSTM", rap_forecast._get_inventory()) # 184

# const rap_us = rap_data[:,183]
# const rap_vs = rap_data[:,184]

# function mad(u_i, v_i)
#   us = @view data[:,u_i]
#   vs = @view data[:,v_i]
#   abs_dev = 0.0
#   n       = 0
#   conus_mask = Conus.is_in_conus.(grid().latlons)

#   for latlon in grid().latlons[conus_mask]
#     du = us[Grids.latlon_to_closest_grid_i(grid(), latlon)] - rap_us[Grids.latlon_to_closest_grid_i(RAP.grid(), latlon)]
#     dv = vs[Grids.latlon_to_closest_grid_i(grid(), latlon)] - rap_vs[Grids.latlon_to_closest_grid_i(RAP.grid(), latlon)]
#     abs_dev += 1.94 * sqrt(du^2 + dv^2)
#     n += 1
#   end

#   abs_dev / n
# end


# FORECASTS_ROOT="../../test_grib2s"
forecasts_root() = get(ENV, "FORECASTS_ROOT", "/Volumes")

layer_blocks_to_make = FeatureEngineeringShared.fewer_grad_blocks

# Didn't realize the 1hrly file didn't have hours divisible by 3 and needed to grab the 3hrly files too.
# 2019-1-9 is the first day with all hours.


vector_wind_layers = [
  "GRD:10 m above ground:hour fcst:wt ens mean",
  "GRD:1000 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:600 mb:hour fcst:wt ens mean",
  "GRD:500 mb:hour fcst:wt ens mean",
  "GRD:300 mb:hour fcst:wt ens mean",
  "GRD:250 mb:hour fcst:wt ens mean",
  "MEAN", # lower atmosphere mean wind for Bunkers motion
  "SHEAR", # shear vector for Bunkers motion. *Supposed* to be 250m - 5750m shear vector.
  "STM", # Our computed Bunkers storm motion.
  "½STM½500mb", # mean between Vunkers and 500mb wind. HREF composite reflectivity movement near storm events is most correlated with this
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

sbcape_key      = "CAPE:surface:hour fcst:wt ens mean"
sbcin_key       = "CIN:surface:hour fcst:wt ens mean"
helicity3km_key = "HLCY:3000-0 m above ground:hour fcst:wt ens mean"

surface_u_key = "UGRD:10 m above ground:hour fcst:wt ens mean"
surface_v_key = "VGRD:10 m above ground:hour fcst:wt ens mean"

dpt_key       = "DPT:2 m above ground:hour fcst:wt ens mean"
rh_key        = "RH:2 m above ground:hour fcst:wt ens mean"
rain_key      = "CRAIN:surface:hour fcst:wt ens mean"
tmp_500mb_key = "TMP:500 mb:hour fcst:wt ens mean"

function compute_0_500mb_BWD!(get_layer, out)
  diff_u = get_layer("UGRD:500 mb:hour fcst:wt ens mean") .- get_layer("UGRD:10 m above ground:hour fcst:wt ens mean")
  diff_v = get_layer("VGRD:500 mb:hour fcst:wt ens mean") .- get_layer("VGRD:10 m above ground:hour fcst:wt ens mean")
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
      u_data       = get_layer("USTM"),
      v_data       = get_layer("VSTM"),
      feature_data = meanify_25mi(grid, get_layer(key) .* Float32.(get_layer("SBCAPE*BWD0-500mb*HLCY3000-0m*10^-6") .> 0.1f0)),
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


extra_features = [
  ("BWD0-500mb",         (_, get_layer, out) -> compute_0_500mb_BWD!(get_layer, out)),
  ("700-500mbLapseRate", (_, get_layer, out) -> FeatureEngineeringShared.lapse_rate_from_ensemble_mean!(get_layer, out, "700 mb", "500 mb")),
  ("850-700mbLapseRate", (_, get_layer, out) -> FeatureEngineeringShared.lapse_rate_from_ensemble_mean!(get_layer, out, "850 mb", "700 mb")),

  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* get_layer(helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-500mb", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* get_layer("BWD0-500mb")),
  ("sqrtSBCAPE*BWD0-500mb", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* get_layer("BWD0-500mb")),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .=       get_layer(sbcape_key)  .* (200f0 .+ get_layer(sbcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, get_layer, out) -> out .= sqrt.(get_layer(sbcape_key)) .* (200f0 .+ get_layer(sbcin_key))),

  (    "SBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),

  # Next one is right-moving SCP more or less.
  (    "SBCAPE*BWD0-500mb*HLCY3000-0m*10^-6", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*BWD0-500mb") .* get_layer(helicity3km_key) .* (1f0 / (1000f0 * 50f0 * 20f0))), # Add normalization term to make it the SCP, basically.
  ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m"      , (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*BWD0-500mb") .* get_layer(helicity3km_key)),

  (    "SBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer(    "SBCAPE*HLCY3000-0m*(200+SBCIN)") .* get_layer("BWD0-500mb")),
  ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, get_layer, out) -> out .= get_layer("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)") .* get_layer("BWD0-500mb")),

  # SHIP terms
  # SREF only has SBCAPE, no MUCAPE, so ignoring CAPE
  # https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html
  # [(MUCAPE j/kg) * (Mixing Ratio of MU PARCEL g/kg) * (700-500mb LAPSE RATE c/km) * (-500mb TEMP C) * (0-6km Shear m/s) ] / 42,000,000
  # "0-6 km shear is confined to a range of 7-27 m s-1, mixing ratio is confined to a range of 11-13.6 g kg-1, and the 500 mb temperature is set to -5.5 C for any warmer values."
  # ignoring other adjustments

  ("700-500mbLapseRate*BWD0-500mb",               (_, get_layer, out) -> out .= get_layer("BWD0-500mb") .* get_layer("700-500mbLapseRate")),
  ("700-500mbLapseRate*-Celcius500mb*BWD0-500mb", (_, get_layer, out) -> out .= get_layer("700-500mbLapseRate*BWD0-500mb") .* (273.15f0 .- min.(get_layer(tmp_500mb_key), -5.5f0 + 273.15f0))),

  ("MixingRatio850mb", (_, get_layer, out) -> out .= clamp.(FeatureEngineeringShared.mixing_ratio.(get_layer("DPT:850 mb:hour fcst:wt ens mean"), 850f0), 11f0, 13.6f0)),
  ("MixingRatio700mb", (_, get_layer, out) -> out .= clamp.(FeatureEngineeringShared.mixing_ratio.(get_layer("DPT:700 mb:hour fcst:wt ens mean"), 700f0), 11f0, 13.6f0)),
  ("MixingRatio500mb", (_, get_layer, out) -> out .= clamp.(FeatureEngineeringShared.mixing_ratio.(get_layer("DPT:500 mb:hour fcst:wt ens mean"), 500f0), 11f0, 13.6f0)),

  ("MixingRatio850mb*700-500mbLapseRate*-Celcius500mb*BWD0-500mb", (_, get_layer, out) -> out .= get_layer("700-500mbLapseRate*-Celcius500mb*BWD0-500mb") .* get_layer("MixingRatio850mb")),
  ("MixingRatio700mb*700-500mbLapseRate*-Celcius500mb*BWD0-500mb", (_, get_layer, out) -> out .= get_layer("700-500mbLapseRate*-Celcius500mb*BWD0-500mb") .* get_layer("MixingRatio700mb")),
  ("MixingRatio500mb*700-500mbLapseRate*-Celcius500mb*BWD0-500mb", (_, get_layer, out) -> out .= get_layer("700-500mbLapseRate*-Celcius500mb*BWD0-500mb") .* get_layer("MixingRatio500mb")),

  # low level wind fields
  # Andy Wade's idea
  ("MaxWind<=850mb", (_, get_layer, out) -> out .= max.(get_layer("WIND:10 m above ground:hour fcst:wt ens mean"), get_layer("WIND:1000 mb:hour fcst:wt ens mean"), get_layer("WIND:850 mb:hour fcst:wt ens mean"))),
  ("SumWind<=850mb", (_, get_layer, out) -> out .=   .+(get_layer("WIND:10 m above ground:hour fcst:wt ens mean"), get_layer("WIND:1000 mb:hour fcst:wt ens mean"), get_layer("WIND:850 mb:hour fcst:wt ens mean"))),
  ("MaxWind<=700mb", (_, get_layer, out) -> out .= max.(get_layer("MaxWind<=850mb"), get_layer("WIND:700 mb:hour fcst:wt ens mean"))),
  ("SumWind<=700mb", (_, get_layer, out) -> out .=   .+(get_layer("SumWind<=850mb"), get_layer("WIND:700 mb:hour fcst:wt ens mean"))),


  ("Divergence10m*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:wt ens mean"), get_layer("VGRD:10 m above ground:hour fcst:wt ens mean"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("Divergence250mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),
  ("DifferentialDivergence250-850mb*10^5", (grid, get_layer, out) -> out .= get_layer("Divergence250mb*10^5") - get_layer("Divergence850mb*10^5")),

  ("ConvergenceOnly10m*10^5"  , (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence10m*10^5"  ))),
  ("ConvergenceOnly850mb*10^5", (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence850mb*10^5"))),

  ("AbsVorticity10m*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_abs_vorticity_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:wt ens mean"), get_layer("VGRD:10 m above ground:hour fcst:wt ens mean"))),

  # Estimating Bunkers Storm motion
  #
  # The unusual weighting was chosen by lots of guess and check, trying to match RAP's storm motion. We don't have a lot of wind layers to work with.
  ("UMEAN", (_, get_layer, out) -> out .= (1/3.03f0) .* (1.5f0 .* get_layer("UGRD:10 m above ground:hour fcst:wt ens mean") .+ 0.5f0 .* get_layer("UGRD:850 mb:hour fcst:wt ens mean") .+ 0.45f0 .* get_layer("UGRD:700 mb:hour fcst:wt ens mean") .+ 0.1f0 .* get_layer("UGRD:600 mb:hour fcst:wt ens mean") .+ 0.75f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean") .+ 0.08f0 .* get_layer("UGRD:300 mb:hour fcst:wt ens mean"))),
  ("VMEAN", (_, get_layer, out) -> out .= (1/3.03f0) .* (1.5f0 .* get_layer("VGRD:10 m above ground:hour fcst:wt ens mean") .+ 0.5f0 .* get_layer("VGRD:850 mb:hour fcst:wt ens mean") .+ 0.45f0 .* get_layer("VGRD:700 mb:hour fcst:wt ens mean") .+ 0.1f0 .* get_layer("VGRD:600 mb:hour fcst:wt ens mean") .+ 0.75f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean") .+ 0.08f0 .* get_layer("VGRD:300 mb:hour fcst:wt ens mean"))),

  # *Supposed* to be 250m - 5750m shear vector.
  ("USHEAR", (_, get_layer, out) -> out .= 0.06f0 .* get_layer("UGRD:300 mb:hour fcst:wt ens mean") .+ 0.94f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean") .- 1.07f0 .* (0.6f0 .* get_layer("UGRD:10 m above ground:hour fcst:wt ens mean") .+ 0.4f0 .* get_layer("UGRD:850 mb:hour fcst:wt ens mean"))),
  ("VSHEAR", (_, get_layer, out) -> out .= 0.06f0 .* get_layer("VGRD:300 mb:hour fcst:wt ens mean") .+ 0.94f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean") .- 1.07f0 .* (0.6f0 .* get_layer("VGRD:10 m above ground:hour fcst:wt ens mean") .+ 0.4f0 .* get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),

  ("SHEAR", (_, get_layer, out) -> out .= sqrt.(get_layer("USHEAR").^2 .+ get_layer("VSHEAR").^2) .+ eps(1f0)),

  # mean + D(shear x k) / |shear|
  # D = 7.5 m/s

  ("USTM", (_, get_layer, out) -> out .= get_layer("UMEAN") .+ 7.5f0 .* get_layer("VSHEAR") ./ (get_layer("SHEAR") .+ 0.23f0)),
  ("VSTM", (_, get_layer, out) -> out .= get_layer("VMEAN") .- 7.5f0 .* get_layer("USHEAR") ./ (get_layer("SHEAR") .+ 0.23f0)),

  ("U½STM½500mb", (_, get_layer, out) -> out .= 0.5f0 .* (get_layer("USTM") .+ get_layer("UGRD:500 mb:hour fcst:wt ens mean"))),
  ("V½STM½500mb", (_, get_layer, out) -> out .= 0.5f0 .* (get_layer("VSTM") .+ get_layer("VGRD:500 mb:hour fcst:wt ens mean"))),

  # Earlier experiments seemed to have trouble with conditions where supercells moved off fronts.
  #
  # Latent supercell indicator based on SCP and convergence upstream (following storm motion).
  #
  # Should follow goldilocks principle: too much and too little are both bad.

  ("SCPish(RM)>1", (grid, get_layer, out) -> out .= Float32.(meanify_25mi(grid, get_layer("SBCAPE*BWD0-500mb*HLCY3000-0m*10^-6")) .> 1f0)),

  ("StormUpstream10mConvergence3hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 3)),
  ("StormUpstream10mConvergence6hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 6)),
  ("StormUpstream10mConvergence9hrGatedBySCP"                 , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly10m*10^5"             , hours = 9)),

  ("StormUpstream850mbConvergence3hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 3)),
  ("StormUpstream850mbConvergence6hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 6)),
  ("StormUpstream850mbConvergence9hrGatedBySCP"               , (grid, get_layer, out) ->                  storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "ConvergenceOnly850mb*10^5"           , hours = 9)),

  ("StormUpstreamDifferentialDivergence250-850mb3hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 3))),
  ("StormUpstreamDifferentialDivergence250-850mb6hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 6))),
  ("StormUpstreamDifferentialDivergence250-850mb9hrGatedBySCP", (grid, get_layer, out) -> out .= max.(0f0, storm_upstream_feature_gated_by_SCP!(grid, get_layer, out, "DifferentialDivergence250-850mb*10^5", hours = 9))),

  # Low level upstream features. What kind of air is the storm ingesting?

  ("Upstream2mDPT1hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, dpt_key   , hours = 1)),
  ("Upstream2mDPT2hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, dpt_key   , hours = 2)),

  ("UpstreamSurfaceLayerRH1hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, rh_key    , hours = 1)),
  ("UpstreamSurfaceLayerRH2hr" , (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, rh_key    , hours = 2)),

  ("UpstreamCRAIN1hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, rain_key  , hours = 1)),
  ("UpstreamCRAIN2hr" ,          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, rain_key  , hours = 2)),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = extra_features
  )
end

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

# function three_hour_window_feature_engineered_forecasts()
#   ThreeHourWindowForecasts.three_hour_window_forecasts(feature_engineered_forecasts())
# end

# # Debug
# function three_hour_window_feature_engineered_forecasts_middle_hour_only()
#   inventory_transformer(base_forecast, base_inventory) = begin
#     single_hour_feature_count = div(length(base_inventory),3)
#
#     forecast_hour_inventory = base_inventory[(single_hour_feature_count + 1):(2*single_hour_feature_count)]
#     forecast_hour_inventory
#   end
#
#   data_transformer(base_forecast, base_data) = begin
#     point_count               = size(base_data, 1)
#     single_hour_feature_count = div(size(base_data, 2),3)
#
#     forecast_hour_data = base_data[:, (1*single_hour_feature_count + 1):(2*single_hour_feature_count)]
#     forecast_hour_data
#   end
#
#   ForecastCombinators.map_forecasts(three_hour_window_feature_engineered_forecasts(); inventory_transformer = inventory_transformer, data_transformer = data_transformer)
# end

# Last upgrade to SREF I could find was 2015ish
function run_datetime_to_simulation_version(datetime)
  7
end

function three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts_with_climatology_etc(feature_engineered_forecasts(); run_datetime_to_simulation_version = run_datetime_to_simulation_version)
end

# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

mean_layers_to_compute_from_prob = []

function reload_forecasts()
  sref_paths =
    if get(ENV, "USE_ALT_DISK", "false") == "true"
      println("Using SREF_HREF_2 and SREF_HREF_4")
      vcat(
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_2/sref"),
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_4/sref")
      )
    else
      vcat(
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/sref"),
        Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_3/sref")
      )
    end


  global _forecasts

  _forecasts = []

  grid = nothing

  for sref_path in sref_paths
    # "/Volumes/SREF_HREF_1/sref/201807/20180728/sref_20180728_t03z_mean_1hrly.grib2"

    # This should speed up loading times and save some space in our disk cache.
    if isnothing(grid)
      grid = Grib2.read_grid(sref_path, crop = crop, downsample = downsample) # mean and prob better have the same grid!
    end

    if occursin("mean_1hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_1hrly" => "prob_1hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) != 0, 1:39)

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("SREF", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, mean_layers_to_compute_from_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end

    # Didn't start gathering the 3-hourlies until 2019-1-9
    if occursin("mean_3hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_3hrly" => "prob_3hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) == 0, 1:87)

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("SREF", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, mean_layers_to_compute_from_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end
  end

  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))

  _forecasts
end

end # module SREF