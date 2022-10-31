module PublishedForecasts2021Models

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../lib")

import Forecasts
import Grib2
import Grids
import Inventories
import HREF15KMGrid

# Deeply traverse root_path to find grib2 file paths
function all_matching_file_paths_in(predicate, root_path)
  paths = []
  if !isdir(root_path)
    println()
    println("$root_path does not exist! Can't read files!")
    println()
    return paths
  end
  for (dir_path, _, file_names) in walkdir(root_path)
    for file_name in file_names
      if predicate(file_name)
        push!(paths, joinpath(dir_path, file_name)) # path to files
      end
    end
  end
  paths
end

is_grib2_path(file_name)   = endswith(file_name, ".grb2") || endswith(file_name, ".grib2")
is_float16_path(file_name) = endswith(file_name, ".float16.bin")

const root = "/home/brian/nadocast_operational_2021/forecasts"
const yyyymmdd_regex = r"_conus_tornado_(\d\d\d\d)(\d\d)(\d\d)_"

function grid()
  HREF15KMGrid.HREF_CROPPED_15KM_GRID
end

function forecasts_12z()
  forecasts = Forecasts.Forecast[]
  for path in all_matching_file_paths_in(name -> contains(name, "nadocast_conus_tornado_") && contains(name, "_t12z_f02-23.") && (is_grib2_path(name) || is_float16_path(name)), root)
    run_year_str, run_month_str, run_day_str = match(yyyymmdd_regex, path).captures
    run_year, run_month, run_day             = parse.(Int64, (run_year_str, run_month_str, run_day_str))

    (run_year, run_month, run_day) >= (2022, 4, 29) || continue

    function get_inventory()
      if is_grib2_path(path)
        Grib2.read_inventory(path)
      elseif is_float16_path(path)
        Inventories.InventoryLine[
          Inventories.InventoryLine(
            "",
            "",
            "d=$(run_year_str)$(run_month_str)$(run_day_str)12",
            "TORPROB",
            "surface",
            "2-23 hour ave fcst",
            "prob >0",
            "prob fcst 1/1"
          )
        ]
      else
        error("bad path $path")
      end
    end

    function get_data()
      if is_grib2_path(path)
        data = 0.01f0 .* Grib2.read_layers_data_raw(path, get_inventory())
        @assert size(data,1) == length(grid().latlons)
        @assert maximum(data) <= 1
        @assert minimum(data) >= 0
      elseif is_float16_path(path)
        @assert filesize(path) == 2*length(grid().latlons)
        data = Float32.(read!(path, Vector{Float16}(undef, length(grid.latlons))))
        @assert maximum(data) <= 1
        @assert minimum(data) >= 0
      else
        error("bad path $path")
      end
      data
    end

    forecast = Forecasts.Forecast(
      "Published2021Models",
      run_year, run_month, run_day,
      12, # run_hour
      23, # forecast_hour
      [], # based_on,
      grid(),
      get_inventory,
      get_data,
      [path] # preload paths
    )

    push!(forecasts, forecast)
  end

  sort!(forecasts, alg=MergeSort, by=Forecasts.run_utc_datetime)

  run_times = Forecasts.run_utc_datetime.(forecasts)

  @assert run_times == unique(run_times)

  forecasts
end

end