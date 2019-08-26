module HREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared
import FeatureEngineeringShared

# Techincally, the HREF is on grid 227: http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID227
# Natively 1473x1025 (5km)
# BUT there's lots of missing data near the edges. The effective bounds of the grid appear to match the HRRR.
#
# See HREF_raw_usage.txt: it's not exactly square on the grid.
#
# We'll conservatively cut 214 off the W, 99 off the E, 119 off the S, 228 off the N
crop = ((1+214):(1473 - 99), (1+119):(1025-228))

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
  "GRD:250 mb:hour fcst:wt ens mean",
  "GRD:500 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:925 mb:hour fcst:wt ens mean",
]

downsample = 3 # 3x downsample, roughly 15km grid.

_forecasts = []

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

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function reload_forecasts()
  href_paths = Grib2.all_grib2_file_paths_in("$(forecasts_root())/SREF_HREF_1/href")

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

      forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("href", mean_href_path, prob_href_path, common_layers_mean, common_layers_prob, grid = grid)

      push!(_forecasts, forecast)
    end
  end

  _forecasts
end

end # module HREF