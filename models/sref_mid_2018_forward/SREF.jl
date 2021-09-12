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

# FORECASTS_ROOT="../../test_grib2s"
forecasts_root() = get(ENV, "FORECASTS_ROOT", "/Volumes")

layer_blocks_to_make = FeatureEngineeringShared.all_layer_blocks

# Didn't realize the 1hrly file didn't have hours divisible by 3 and needed to grab the 3hrly files too.
# 2019-1-9 is the first day with all hours.


# # For elasticnet models:
# layer_blocks_to_make = [
#   FeatureEngineeringShared.raw_features_block,
#   # FeatureEngineeringShared.twenty_five_mi_mean_block,
#   FeatureEngineeringShared.fifty_mi_mean_block,
#   # FeatureEngineeringShared.hundred_mi_mean_block,
#   # FeatureEngineeringShared.twenty_five_mi_forward_gradient_block,
#   # FeatureEngineeringShared.twenty_five_mi_leftward_gradient_block,
#   # FeatureEngineeringShared.twenty_five_mi_linestraddling_gradient_block,
#   FeatureEngineeringShared.fifty_mi_forward_gradient_block,
#   FeatureEngineeringShared.fifty_mi_leftward_gradient_block,
#   FeatureEngineeringShared.fifty_mi_linestraddling_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_forward_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_leftward_gradient_block,
#   # FeatureEngineeringShared.hundred_mi_linestraddling_gradient_block,
# ]

# # For reduced elasticnet models:
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
  "GRD:10 m above ground:hour fcst:wt ens mean",
  "GRD:1000 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:600 mb:hour fcst:wt ens mean",
  "GRD:500 mb:hour fcst:wt ens mean",
  "GRD:300 mb:hour fcst:wt ens mean",
  "GRD:250 mb:hour fcst:wt ens mean",
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
      u_data       = get_layer("UGRD:700 mb:hour fcst:wt ens mean"),
      v_data       = get_layer("VGRD:700 mb:hour fcst:wt ens mean"),
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


interaction_terms = [
  ("BWD0-500mb", (_, get_layer, out) -> compute_0_500mb_BWD!(get_layer, out)),

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

  ("Divergence10m*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:wt ens mean"), get_layer("VGRD:10 m above ground:hour fcst:wt ens mean"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("Divergence250mb*10^5"                , (grid, get_layer, out) -> FeatureEngineeringShared.compute_divergence_threaded!(grid, out, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),
  ("DifferentialDivergence250-850mb*10^5", (grid, get_layer, out) -> out .= get_layer("Divergence250mb*10^5") - get_layer("Divergence850mb*10^5")),

  ("ConvergenceOnly10m*10^5"  , (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence10m*10^5"  ))),
  ("ConvergenceOnly850mb*10^5", (grid, get_layer, out) -> out .= max.(0f0, 0f0 .- get_layer("Divergence850mb*10^5"))),

  ("AbsVorticity10m*10^5", (grid, get_layer, out) -> FeatureEngineeringShared.compute_vorticity_threaded!(grid, out, get_layer("UGRD:10 m above ground:hour fcst:wt ens mean"), get_layer("VGRD:10 m above ground:hour fcst:wt ens mean"))),

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

  ("UpstreamSBCAPE1hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, sbcape_key, hours = 1)),
  ("UpstreamSBCAPE2hr",          (grid, get_layer, out) -> upstream_feature!(grid, get_layer, out, surface_u_key, surface_v_key, sbcape_key, hours = 2)),

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
    new_features_pre = interaction_terms,
    use_2020_models_buggy_100mi_calc = true
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

function three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts_with_climatology(feature_engineered_forecasts())
end



# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function reload_forecasts()
  sref_paths = vcat(
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/sref"),
    Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_3/sref")
  )

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

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("SREF", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end

    # Didn't start gathering the 3-hourlies until 2019-1-9
    if occursin("mean_3hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_3hrly" => "prob_3hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) == 0, 1:87)

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("SREF", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end
  end

  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))

  _forecasts
end

end # module SREF