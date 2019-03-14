module SREFHREFShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2
import Inventories


function mean_prob_grib2s_to_forecast(
            href_or_sref_str,
            mean_grib2_path,
            prob_grib2_path,
            common_layers_mean, # List of layer keys
            common_layers_prob; # List of layer keys
            forecast_hour = nothing,
            grid          = nothing, # After downsampling. Prevents having to reload the grid.
            downsample    = 1
          ) :: Forecasts.Forecast

  year_str, month_str, day_str, run_hour_str = match(r"/\w+_(\d\d\d\d)(\d\d)(\d\d)_t(\d\d)z_mean_\w+.gri?b2", mean_grib2_path).captures

  run_year  = parse(Int64, year_str)
  run_month = parse(Int64, month_str)
  run_day   = parse(Int64, day_str)
  run_hour  = parse(Int64, run_hour_str)

  if forecast_hour == nothing
    forecast_hour_str, = match( r"_mean_f(\d+)\.grib2", mean_grib2_path).captures
    forecast_hour      = parse(Int64, forecast_hour_str)
  end

  get_grid(forecast) = begin
    if grid == nothing
      Grib2.read_grid(mean_grib2_path, downsample = downsample) # mean and prob better have the same grid!
    else
      grid
    end
  end

  get_inventory(forecast) = begin
    forecast_hour_str = "$forecast_hour hour fcst" # Accumulation fields are already excluded by find_common_layers.rb, so it's okay that we don't look for "acc fcst"
    # Somewhat inefficient that each hour must trigger wgrib2 on the same file...could add another layer of caching here.
    mean_inventory = filter(line -> forecast_hour_str == line.forecast_hour_str, Grib2.read_inventory(mean_grib2_path))
    prob_inventory = filter(line -> forecast_hour_str == line.forecast_hour_str, Grib2.read_inventory(prob_grib2_path))

    mean_layer_key_to_inventory_line(key) = begin
      i = findfirst(line -> Inventories.inventory_line_key(line) == key, mean_inventory)
      if i != nothing
        mean_inventory[i]
      else
        throw("$href_or_sref_str forecast $(Forecasts.time_title(forecast)) does not have $key in mean layers: $mean_inventory")
      end
    end

    prob_layer_key_to_inventory_line(key) = begin
      i = findfirst(line -> Inventories.inventory_line_key(line) == key, prob_inventory)
      if i != nothing
        prob_inventory[i]
      else
        throw("$href_or_sref_str forecast $(Forecasts.time_title(forecast)) does not have $key in prob layers: $prob_inventory")
      end
    end

    mean_inventory_to_use = map(mean_layer_key_to_inventory_line, common_layers_mean)
    prob_inventory_to_use = map(prob_layer_key_to_inventory_line, common_layers_prob)

    vcat(mean_inventory_to_use, prob_inventory_to_use)
  end

  get_data(forecast) = begin
    downsample_grid =
      if downsample == 1
        nothing
      else
        Forecasts.grid(forecast)
      end

    mean_inventory = collect(Iterators.take(Forecasts.inventory(forecast), length(common_layers_mean)))
    prob_inventory = collect(Iterators.drop(Forecasts.inventory(forecast), length(common_layers_mean)))

    mean_data = Grib2.read_layers_data_raw(mean_grib2_path, mean_inventory, downsample_grid = downsample_grid)
    prob_data = Grib2.read_layers_data_raw(prob_grib2_path, prob_inventory, downsample_grid = downsample_grid)

    hcat(mean_data, prob_data)
  end

  Forecasts.Forecast(run_year, run_month, run_day, run_hour, forecast_hour, get_grid, get_inventory, get_data)
end


end # module SREFHREFShared