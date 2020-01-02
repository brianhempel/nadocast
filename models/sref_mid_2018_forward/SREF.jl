module SREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
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

get_layer = FeatureEngineeringShared.get_layer

sbcape_key      = "CAPE:surface:hour fcst:wt ens mean"
sbcin_key       = "CIN:surface:hour fcst:wt ens mean"
helicity3km_key = "HLCY:3000-0 m above ground:hour fcst:wt ens mean"

function compute_0_500mb_BWD(inventory, data)
  diff_u = get_layer(data, inventory, "UGRD:500 mb:hour fcst:wt ens mean") .- get_layer(data, inventory, "UGRD:10 m above ground:hour fcst:wt ens mean")
  diff_v = get_layer(data, inventory, "VGRD:500 mb:hour fcst:wt ens mean") .- get_layer(data, inventory, "VGRD:10 m above ground:hour fcst:wt ens mean")
  sqrt.(diff_u.^2 .+ diff_v.^2)
end

interaction_terms = [
  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* get_layer(data, inventory, helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* get_layer(data, inventory, helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-500mb", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data)),
  ("sqrtSBCAPE*BWD0-500mb", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data)),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),

  (    "SBCAPE*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),

  (    "SBCAPE*BWD0-500mb*HLCY3000-0m", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key)),
  ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key)),

  (    "SBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) ->       get_layer(data, inventory, sbcape_key)  .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
  ("sqrtSBCAPE*BWD0-500mb*HLCY3000-0m*(200+SBCIN)", (_, inventory, data) -> sqrt.(get_layer(data, inventory, sbcape_key)) .* compute_0_500mb_BWD(inventory, data) .* get_layer(data, inventory, helicity3km_key) .* (200f0 .+ get_layer(data, inventory, sbcin_key))),
]

function feature_engineered_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    forecasts();
    vector_wind_layers = vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = interaction_terms
  )
end

function example_forecast()
  forecasts()[1]
end

function grid()
  example_forecast().grid
end

function three_hour_window_feature_engineered_forecasts()
  ThreeHourWindowForecasts.three_hour_window_forecasts(feature_engineered_forecasts())
end

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
  ThreeHourWindowForecasts.three_hour_window_and_min_mean_max_delta_forecasts(feature_engineered_forecasts())
end



# function feature_i_to_name(feature_i)
#   inventory = Forecasts.inventory(example_forecast())
#   FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
# end

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function reload_forecasts()
  sref_paths = Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/sref")

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

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("sref", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end

    # Didn't start gathering the 3-hourlies until 2019-1-9
    if occursin("mean_3hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_3hrly" => "prob_3hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) == 0, 1:87)

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("sref", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end
  end

  _forecasts = sort(_forecasts, by = (forecast -> (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))))

  _forecasts
end

end # module SREF