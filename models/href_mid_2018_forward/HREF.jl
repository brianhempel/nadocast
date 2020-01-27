module HREF

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
crop = ((1+214):(1473 - 99), (1+119):(1025-228))

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

get_layer = FeatureEngineeringShared.get_layer

sbcape_key     = "CAPE:surface:hour fcst:wt ens mean"
mlcape_key     = "CAPE:90-0 mb above ground:hour fcst:wt ens mean"
# mulayercape_key = "CAPE:180-0 mb above ground:hour fcst:wt ens mean"
sbcin_key      = "CIN:surface:hour fcst:wt ens mean"
mlcin_key      = "CIN:90-0 mb above ground:hour fcst:wt ens mean"
# mulayercin_key = "CIN:180-0 mb above ground:hour fcst:wt ens mean"
helicity3km_key = "HLCY:3000-0 m above ground:hour fcst:wt ens mean"
bwd_0_6km_key   = "VWSH:6000-0 m above ground:hour fcst:wt ens mean"

interaction_terms = [
  # 0-3km EHI, roughly
  (    "SBCAPE*HLCY3000-0m", (_, get_layer) ->       get_layer(sbcape_key)  .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m", (_, get_layer) ->       get_layer(mlcape_key)  .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m", (_, get_layer) -> sqrt.(get_layer(sbcape_key)) .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m", (_, get_layer) -> sqrt.(get_layer(mlcape_key)) .* get_layer(helicity3km_key)),

  # Terms following Togstad et al 2011 "Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles"
  (    "SBCAPE*BWD0-6km", (_, get_layer) ->       get_layer(sbcape_key)  .* get_layer(bwd_0_6km_key)),
  (    "MLCAPE*BWD0-6km", (_, get_layer) ->       get_layer(mlcape_key)  .* get_layer(bwd_0_6km_key)),
  ("sqrtSBCAPE*BWD0-6km", (_, get_layer) -> sqrt.(get_layer(sbcape_key)) .* get_layer(bwd_0_6km_key)),
  ("sqrtMLCAPE*BWD0-6km", (_, get_layer) -> sqrt.(get_layer(mlcape_key)) .* get_layer(bwd_0_6km_key)),

  # Pseudo-STP terms
  (    "SBCAPE*(200+SBCIN)", (_, get_layer) ->       get_layer(sbcape_key)  .* (200f0 .+ get_layer(sbcin_key))),
  (    "MLCAPE*(200+MLCIN)", (_, get_layer) ->       get_layer(mlcape_key)  .* (200f0 .+ get_layer(mlcin_key))),
  ("sqrtSBCAPE*(200+SBCIN)", (_, get_layer) -> sqrt.(get_layer(sbcape_key)) .* (200f0 .+ get_layer(sbcin_key))),
  ("sqrtMLCAPE*(200+MLCIN)", (_, get_layer) -> sqrt.(get_layer(mlcape_key)) .* (200f0 .+ get_layer(mlcin_key))),

  (    "SBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer) -> get_layer(    "SBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),
  (    "MLCAPE*HLCY3000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(    "MLCAPE*(200+MLCIN)") .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)", (_, get_layer) -> get_layer("sqrtSBCAPE*(200+SBCIN)") .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*HLCY3000-0m*(200+MLCIN)", (_, get_layer) -> get_layer("sqrtMLCAPE*(200+MLCIN)") .* get_layer(helicity3km_key)),

  (    "SBCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer) -> get_layer(    "SBCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  (    "MLCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer) -> get_layer(    "MLCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  ("sqrtSBCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer) -> get_layer("sqrtSBCAPE*BWD0-6km") .* get_layer(helicity3km_key)),
  ("sqrtMLCAPE*BWD0-6km*HLCY3000-0m", (_, get_layer) -> get_layer("sqrtMLCAPE*BWD0-6km") .* get_layer(helicity3km_key)),

  (    "SBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN)", (_, get_layer) -> get_layer(    "SBCAPE*HLCY3000-0m*(200+SBCIN)")),
  (    "MLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN)", (_, get_layer) -> get_layer(    "MLCAPE*HLCY3000-0m*(200+MLCIN)")),
  ("sqrtSBCAPE*BWD0-6km*HLCY3000-0m*(200+SBCIN)", (_, get_layer) -> get_layer("sqrtSBCAPE*HLCY3000-0m*(200+SBCIN)")),
  ("sqrtMLCAPE*BWD0-6km*HLCY3000-0m*(200+MLCIN)", (_, get_layer) -> get_layer("sqrtMLCAPE*HLCY3000-0m*(200+MLCIN)")),

  ("Divergence925mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:925 mb:hour fcst:wt ens mean"), get_layer("VGRD:925 mb:hour fcst:wt ens mean"))),

  # Following SPC Mesoscale analysis page
  ("Divergence850mb*10^5"                , (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("Divergence250mb*10^5"                , (grid, get_layer) -> FeatureEngineeringShared.compute_divergence_threaded(grid, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),
  ("DifferentialDivergence250-850mb*10^5", (grid, get_layer) -> get_layer("Divergence250mb*10^5") - get_layer("Divergence850mb*10^5")),

  ("ConvergenceOnly925mb*10^5", (grid, get_layer) -> max.(0f0, 0f0 .- get_layer("Divergence925mb*10^5"))),
  ("ConvergenceOnly850mb*10^5", (grid, get_layer) -> max.(0f0, 0f0 .- get_layer("Divergence850mb*10^5"))),

  ("AbsVorticity925mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:925 mb:hour fcst:wt ens mean"), get_layer("VGRD:925 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity850mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:850 mb:hour fcst:wt ens mean"), get_layer("VGRD:850 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity500mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:500 mb:hour fcst:wt ens mean"), get_layer("VGRD:500 mb:hour fcst:wt ens mean"))),
  ("AbsVorticity250mb*10^5", (grid, get_layer) -> FeatureEngineeringShared.compute_vorticity_threaded(grid, get_layer("UGRD:250 mb:hour fcst:wt ens mean"), get_layer("VGRD:250 mb:hour fcst:wt ens mean"))),
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