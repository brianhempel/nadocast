module HREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared
import FeatureEngineeringShared

# HREF is on grid 227: http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID227
# Natively 1473x1025

downsample = 3 # 3x downsample, roughly 15km grid.

_forecasts = []

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
  Forecasts.grid(example_forecast())
end

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

function feature_i_to_name(feature_i)
  inventory = Forecasts.inventory(example_forecast())
  FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
end

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

vector_wind_layers = [
  "GRD:250 mb:hour fcst:wt ens mean",
  "GRD:500 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:925 mb:hour fcst:wt ens mean",
]

_twenty_five_mi_mean_is    = Vector{Int64}[] # Grid point indicies within 25mi
_unique_fifty_mi_mean_is   = Vector{Int64}[] # Grid point indicies within 50mi but not within 25mi
_unique_hundred_mi_mean_is = Vector{Int64}[] # Grid point indicies within 100mi but not within 50mi

function get_feature_engineered_data(forecast, data)
  global _twenty_five_mi_mean_is
  global _unique_fifty_mi_mean_is
  global _unique_hundred_mi_mean_is

  _twenty_five_mi_mean_is    = isempty(_twenty_five_mi_mean_is)    && FeatureEngineeringShared.twenty_five_mi_mean_block in layer_blocks_to_make ? Grids.radius_grid_is(grid(), 25.0)                                        : _twenty_five_mi_mean_is
  _unique_fifty_mi_mean_is   = isempty(_unique_fifty_mi_mean_is)   && FeatureEngineeringShared.fifty_mi_mean_block in layer_blocks_to_make       ? Grids.radius_grid_is_less_other_is(grid(), 50.0, _twenty_five_mi_mean_is) : _unique_fifty_mi_mean_is
  _unique_hundred_mi_mean_is = isempty(_unique_hundred_mi_mean_is) && FeatureEngineeringShared.hundred_mi_mean_block in layer_blocks_to_make     ? Grids.radius_grid_is_less_other_is(grid(), 100.0, vcat(_twenty_five_mi_mean_is, _unique_fifty_mi_mean_is)) : _unique_hundred_mi_mean_is

  FeatureEngineeringShared.make_data(grid(), Forecasts.inventory(forecast), forecast.forecast_hour, data, vector_wind_layers, layer_blocks_to_make, _twenty_five_mi_mean_is, _unique_fifty_mi_mean_is, _unique_hundred_mi_mean_is)
end

function reload_forecasts()
  href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts

  _forecasts = []

  for href_path in href_paths
    # "/Volumes/SREF_HREF_1/href/201807/20180728/href_conus_20180728_t06z_mean_f15.grib2"

    if occursin("z_mean_f", href_path)
      mean_href_path = href_path
      prob_href_path = replace(mean_href_path, "z_mean_f" => "z_prob_f")

      # Downsampling requires loading (and downsampling) the grid, so this should speed up loading times.
      # And save some space in our disk cache.
      grid =
        if isempty(_forecasts)
          nothing
        else
          Forecasts.grid(_forecasts[1])
        end

      forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("href", mean_href_path, prob_href_path, common_layers_mean, common_layers_prob, grid = grid, downsample = downsample)

      push!(_forecasts, forecast)
    end
  end

  _forecasts
end

end # module HREF