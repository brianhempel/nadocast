module SREF

push!(LOAD_PATH, (@__DIR__) * "/../..")

import Forecasts
import Inventories
import Grib2
import Grids

_forecasts = []

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
  else
    _forecasts
  end
end

# The acc layers will be skipped below, so this list isn't final.
common_layers_mean = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_mean.txt"), String), "\n"))
common_layers_prob = filter(line -> line != "", split(read(open((@__DIR__) * "/common_layers_prob.txt"), String), "\n"))

function reload_forecasts()
  sref_paths = Grib2.all_grib2_file_paths_in("/Volumes/Tornadoes/sref")

  global _forecasts

  _forecasts = []

  for sref_path in sref_paths
    # "/Volumes/Tornadoes/sref/201807/20180728/sref.t03z.pgrb212.mean_1hrly.grib2"

    if occursin("mean_1hrly", sref_path)
      mean_sref_path = sref_path
      prob_sref_path = replace(mean_sref_path, "mean_1hrly" => "prob_1hrly")

      year_str, month_str, day_str, run_hour_str = match(r"/sref_(\d\d\d\d)(\d\d)(\d\d)_t(\d\d)z_mean_1hrly.gri?b2", mean_sref_path).captures

      for forecast_hour in filter(hr -> mod(hr, 3) != 0, 1:38) # Need to update to grab 3hrly forecasts as well.
        run_year  = parse(Int64, year_str)
        run_month = parse(Int64, month_str)
        run_day   = parse(Int64, day_str)
        run_hour  = parse(Int64, run_hour_str)

        get_grid(forecast) = Grib2.read_grid(mean_sref_path) # mean and prob better have the same grid!

        get_inventory(forecast) = begin
          forecast_hour_str = "$forecast_hour hour fcst" # Accumulation fields are already excluded by find_common_layers.rb, so it's okay that we don't look for "acc fcst"
          # Somewhat inefficient that each hour must trigger wgrib2 on the same file...could add another layer of caching here.
          mean_inventory = filter(line -> forecast_hour_str == line.forecast_hour_str, Grib2.read_inventory(mean_sref_path))
          prob_inventory = filter(line -> forecast_hour_str == line.forecast_hour_str, Grib2.read_inventory(prob_sref_path))

          mean_layer_key_to_inventory_line(key) = begin
            i = findfirst(line -> Inventories.inventory_line_key(line) == key, mean_inventory)
            if i != nothing
              mean_inventory[i]
            else
              throw("SREF forecast $(Forecasts.time_title(forecast)) does not have $key in mean layers: $mean_inventory")
            end
          end

          prob_layer_key_to_inventory_line(key) = begin
            i = findfirst(line -> Inventories.inventory_line_key(line) == key, prob_inventory)
            if i != nothing
              prob_inventory[i]
            else
              throw("SREF forecast $(Forecasts.time_title(forecast)) does not have $key in prob layers: $prob_inventory")
            end
          end

          mean_inventory_to_use = map(mean_layer_key_to_inventory_line, common_layers_mean)
          prob_inventory_to_use = map(prob_layer_key_to_inventory_line, common_layers_prob)

          vcat(mean_inventory_to_use, prob_inventory_to_use)
        end

        get_data(forecast) = begin
          mean_inventory = collect(Iterators.take(Forecasts.inventory(forecast), length(common_layers_mean)))
          prob_inventory = collect(Iterators.drop(Forecasts.inventory(forecast), length(common_layers_mean)))

          mean_data = Grib2.read_layers_data_raw(mean_sref_path, mean_inventory)
          prob_data = Grib2.read_layers_data_raw(prob_sref_path, prob_inventory)

          hcat(mean_data, prob_data)
        end

        forecast = Forecasts.Forecast(run_year, run_month, run_day, run_hour, forecast_hour, get_grid, get_inventory, get_data)
        push!(_forecasts, forecast)
      end
    end
  end

  _forecasts
end

end # module SREF