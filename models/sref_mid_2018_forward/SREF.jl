module SREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared
import FeatureEngineeringShared

# SREF is on grid 212: http://www.nco.ncep.noaa.gov/pmb/docs/on388/grids/grid212.gif


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


function feature_i_to_name(feature_i)
  inventory = Forecasts.inventory(example_forecast())
  FeatureEngineeringShared.feature_i_to_name(inventory, layer_blocks_to_make, feature_i)
end

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

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

_twenty_five_mi_mean_is    = Vector{Int64}[] # Grid point indicies within 25mi
_unique_fifty_mi_mean_is   = Vector{Int64}[] # Grid point indicies within 50mi but not within 25mi
_unique_hundred_mi_mean_is = Vector{Int64}[] # Grid point indicies within 100mi but not within 50mi

function get_feature_engineered_data(forecast, data)
  global _twenty_five_mi_mean_is
  global _unique_fifty_mi_mean_is
  global _unique_hundred_mi_mean_is

  _twenty_five_mi_mean_is    = isempty(_twenty_five_mi_mean_is)    ? Grids.radius_grid_is(grid(), 25.0)                                                                         : _twenty_five_mi_mean_is
  _unique_fifty_mi_mean_is   = isempty(_unique_fifty_mi_mean_is)   ? Grids.radius_grid_is_less_other_is(grid(), 50.0, _twenty_five_mi_mean_is)                                  : _unique_fifty_mi_mean_is
  _unique_hundred_mi_mean_is = isempty(_unique_hundred_mi_mean_is) ? Grids.radius_grid_is_less_other_is(grid(), 100.0, vcat(_twenty_five_mi_mean_is, _unique_fifty_mi_mean_is)) : _unique_hundred_mi_mean_is

  FeatureEngineeringShared.make_data(grid(), Forecasts.inventory(forecast), forecast.forecast_hour, data, vector_wind_layers, layer_blocks_to_make, _twenty_five_mi_mean_is, _unique_fifty_mi_mean_is, _unique_hundred_mi_mean_is)
end

function reload_forecasts()
  sref_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/sref")
  # sref_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/sref")

  global _forecasts

  _forecasts = []

  for sref_path in sref_paths
    # "/Volumes/SREF_HREF_1/sref/201807/20180728/sref_20180728_t03z_mean_1hrly.grib2"

    # This should speed up loading times and save some space in our disk cache.
    grid =
      if isempty(_forecasts)
        nothing
      else
        Forecasts.grid(_forecasts[1])
      end

    if occursin("mean_1hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_1hrly" => "prob_1hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) != 0, 1:39)

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("sref", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, grid = grid, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end

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