module HREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Inventories
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared

push!(LOAD_PATH, @__DIR__)
# import FeatureEngineering

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

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function get_feature_engineered_data(forecast, data)
  # FeatureEngineering.make_data(grid(), forecast, data)
  data
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

      forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("href", mean_href_path, prob_href_path, common_layers_mean, common_layers_prob, downsample = 3)

      push!(_forecasts, forecast)
    end
  end

  _forecasts
end

end # module HREF