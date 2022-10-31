module PublishedForecasts2020Models

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
    println("$root_path does not exist! Can't read Grib2s!!")
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

const root = "/home/brian/nadocast_operational_2020/forecasts"
const yyyymmdd_regex = r"_conus_tornado_(\d\d\d\d)(\d\d)(\d\d)_"

function grid()
  HREF15KMGrid.HREF_CROPPED_15KM_GRID
end

function forecasts_14z()
  forecasts = Forecasts.Forecast[]
  for path in all_matching_file_paths_in(name -> contains(name, "nadocast_2020_models_") && contains(name, "_t14z_f02-21.") && (is_grib2_path(name) || is_float16_path(name)), root)
    run_year_str, run_month_str, run_day_str = match(yyyymmdd_regex, path).captures
    run_year, run_month, run_day             = parse.(Int64, (run_year_str, run_month_str, run_day_str))

    if is_grib2_path(path)
      get_inventory() = Grib2.read_inventory(path)
      get_data()      = begin
        data = 0.01f0 .* Grib2.read_layers_data_raw(path, get_inventory())
        @assert size(data,1) == length(grid().latlons)
        @assert maximum(data) <= 1
        @assert minimum(data) >= 0
        data
      end
    elseif is_float16_path(path)
      get_inventory() = Inventories.InventoryLine[
        Inventories.InventoryLine(
          "",
          "",
          "d=$(run_year_str)$(run_month_str)$(run_day_str)14",
          "TORPROB",
          "surface",
          "2-21 hour ave fcst",
          "prob >0",
          "prob fcst 1/1"
        )
      ]
      get_data() = begin
        @assert filesize(float16_path) == 2*length(grid().latlons)
        data = Float32.(read!(float16_path, Vector{Float16}(undef, length(grid.latlons))))
        @assert maximum(data) <= 1
        @assert minimum(data) >= 0
        data
      end
    else
      error("bad path $path")
    end

    forecast = Forecasts.Forecast(
      "Published2020Models",
      run_year, run_month, run_day,
      14, # run_hour
      21, # forecast_hour
      [], # based_on,
      grid(),
      get_inventory,
      get_data,
      [path] # preload paths
    )

    push!(forecast, forecasts)
  end

  sort!(forecasts, alg=MergeSort, by=Forecasts.run_utc_datetime)

  run_times = Forecasts.run_utc_datetime.(forecasts)

  @assert run_times == unique(run_times)

  forecasts
end

end