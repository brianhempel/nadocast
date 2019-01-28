module SREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Inventories
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import SREFHREFShared

push!(LOAD_PATH, @__DIR__)
import FeatureEngineering

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

common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function get_feature_engineered_data(forecast, data)
  FeatureEngineering.make_data(grid(), forecast, data)
end

function reload_forecasts()
  sref_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/sref")
  # sref_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/sref")

  global _forecasts

  _forecasts = []

  for sref_path in sref_paths
    # "/Volumes/SREF_HREF_1/sref/201807/20180728/sref_20180728_t03z_mean_1hrly.grib2"

    if occursin("mean_1hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_1hrly" => "prob_1hrly")

      for forecast_hour in filter(hr -> mod(hr, 3) != 0, 1:38) # Need to update to grab 3hrly forecasts as well.

        forecast = SREFHREFShared.mean_prob_grib2s_to_forecast("sref", mean_sref_path, prob_sref_path, common_layers_mean, common_layers_prob, forecast_hour = forecast_hour)

        push!(_forecasts, forecast)
      end
    end
  end

  _forecasts
end

end # module SREF