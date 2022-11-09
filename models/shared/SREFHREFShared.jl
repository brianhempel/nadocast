module SREFHREFShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2
import Inventories

# prob layers must be in reverse order: least probable first (i.e. furthest from 0 first)
# prob layers are 0-100 (not 0-1)
# layers must be 100% prob for a val of 0
function convert_probs_to_mean(prob_layers)
  out = Vector{Float32}(undef, length(prob_layers[1][1]))

  # @inbounds Threads.@threads :static for i in 1:length(out)
  Threads.@threads :static for i in 1:length(out)
    out_x     = 0f0
    last_prob = 0f0
    for (layer_probs, x) in prob_layers
      weight = layer_probs[i] - last_prob
      out_x += weight * x
      last_prob = layer_probs[i]
    end
    # Implicit final prob of 100% at x = 0
    # weight = 100f0 - last_prob
    # out_x += weight * 0
    out[i] = out_x * 0.01f0 # probs were 0-100
  end

  out
end


function mean_prob_grib2s_to_forecast(
            href_or_sref_str,
            mean_grib2_path,
            prob_grib2_path,
            common_layers_mean, # List of layer keys
            common_layers_prob, # List of layer keys
            mean_layers_to_compute_from_prob; # List of lists of (layer key, threshold)
            forecast_hour = nothing,
            grid
          ) :: Forecasts.Forecast

  fname_match = match(r"/\w+_(\d\d\d\d)(\d\d)(\d\d)_t(\d\d)z_mean_\w+.gri?b2", mean_grib2_path)
  if isnothing(fname_match)
    # "href.t00z.conus.mean.f01.grib2"
    # Need to query via wgrib2
    date_str = read(`$(Grib2.wgrib2_path()) $mean_grib2_path -end -t -ncpu 1`, String) # "1:0:d=2022110700\n"
    year_str, month_str, day_str, run_hour_str = match(r"d=(\d\d\d\d)(\d\d)(\d\d)(\d\d)\n", date_str).captures
  else
    year_str, month_str, day_str, run_hour_str = fname_match.captures
  end

  run_year  = parse(Int64, year_str)
  run_month = parse(Int64, month_str)
  run_day   = parse(Int64, day_str)
  run_hour  = parse(Int64, run_hour_str)

  if isnothing(forecast_hour)
    forecast_hour_str, = match( r"f(\d+)\.grib2", mean_grib2_path).captures
    forecast_hour      = parse(Int64, forecast_hour_str)
  end

  layer_key_to_inventory_line(key, inventory, inventory_line_keys) = begin
    i = findfirst(isequal(key), inventory_line_keys)
    if !isnothing(i)
      inventory[i]
    else
      throw("$href_or_sref_str forecast $(Forecasts.time_title(run_year, run_month, run_day, run_hour, forecast_hour)) does not have $key in layers: $inventory")
    end
  end

  get_inventory() = begin
    # Somewhat inefficient that each hour must trigger wgrib2 on the same file...could add another layer of caching here.
    mean_inventory = filter(line -> forecast_hour == Inventories.forecast_hour(line), Grib2.read_inventory(mean_grib2_path))
    prob_inventory = filter(line -> forecast_hour == Inventories.forecast_hour(line), Grib2.read_inventory(prob_grib2_path))

    mean_inventory_line_keys = Inventories.inventory_line_key.(mean_inventory) # avoid n^2 nasty allocs by precomputing this
    prob_inventory_line_keys = Inventories.inventory_line_key.(prob_inventory) # avoid n^2 nasty allocs by precomputing this

    mean_inventory_to_use = map(key -> layer_key_to_inventory_line(key, mean_inventory, mean_inventory_line_keys), common_layers_mean)
    prob_inventory_to_use = map(key -> layer_key_to_inventory_line(key, prob_inventory, prob_inventory_line_keys), common_layers_prob)

    mean_layers_computed_from_prob_inventory = map(mean_layers_to_compute_from_prob) do (keys_and_thresholds)
      line_key, _ = keys_and_thresholds[1]
      abbrev, level = split(line_key, ":")
      Inventories.InventoryLine(
        "", # message_dot_submessage
        "", # position_str
        "", # date_str, who cares
        abbrev, # abbrev
        level, # level
        "$forecast_hour hour fcst", # forecast_hour_str
        "estimated from probs", # misc
        "" # feature_engineering
      )
    end

    vcat(mean_inventory_to_use, prob_inventory_to_use, mean_layers_computed_from_prob_inventory)
  end

  get_data() = begin
    # Somewhat inefficient that each hour must trigger wgrib2 on the same file...could add another layer of caching here.
    mean_inventory = filter(line -> forecast_hour == Inventories.forecast_hour(line), Grib2.read_inventory(mean_grib2_path))
    prob_inventory = filter(line -> forecast_hour == Inventories.forecast_hour(line), Grib2.read_inventory(prob_grib2_path))

    mean_inventory_line_keys = Inventories.inventory_line_key.(mean_inventory) # avoid n^2 nasty allocs by precomputing this
    prob_inventory_line_keys = Inventories.inventory_line_key.(prob_inventory) # avoid n^2 nasty allocs by precomputing this

    mean_inventory_to_use = map(key -> layer_key_to_inventory_line(key, mean_inventory, mean_inventory_line_keys), common_layers_mean)
    prob_inventory_to_use = map(key -> layer_key_to_inventory_line(key, prob_inventory, prob_inventory_line_keys), common_layers_prob)

    keys_needed_for_computing_means = unique(Iterators.flatten(map(keys_and_thresholds -> map(first, keys_and_thresholds), mean_layers_to_compute_from_prob)))

    extra_inventory_and_keys_needed_to_compute_means = filter(collect(zip(prob_inventory, prob_inventory_line_keys))) do (_inv_line, line_key)
      (line_key in keys_needed_for_computing_means) &&
      !(line_key in common_layers_prob)
    end

    extra_inventory_needed_to_compute_means = map(first, extra_inventory_and_keys_needed_to_compute_means)
    extra_keys_needed_to_compute_means      = map(last,  extra_inventory_and_keys_needed_to_compute_means)

    prob_inventory_needed           = vcat(prob_inventory_to_use, extra_inventory_needed_to_compute_means)
    prob_inventory_needed_line_keys = vcat(common_layers_prob,    extra_keys_needed_to_compute_means)

    mean_data = Grib2.read_layers_data_raw(mean_grib2_path, mean_inventory_to_use, crop_downsample_grid = grid)
    prob_data = Grib2.read_layers_data_raw(prob_grib2_path, prob_inventory_needed, crop_downsample_grid = grid)

    compact(arr) = filter(x -> !isnothing(x), arr)

    mean_data_from_prob = map(mean_layers_to_compute_from_prob) do keys_and_thresholds
      prob_layers = map(keys_and_thresholds) do (line_key, threshold)
        i = findfirst(isequal(line_key), prob_inventory_needed_line_keys)
        if !isnothing(i)
          ((@view prob_data[:, i]), threshold)
        else
          nothing
        end
      end
      convert_probs_to_mean(compact(prob_layers))
    end

    hcat(mean_data, (@view prob_data[:, 1:length(prob_inventory_to_use)]), mean_data_from_prob...)
  end

  preload_paths = [mean_grib2_path, prob_grib2_path]

  Forecasts.Forecast(href_or_sref_str, run_year, run_month, run_day, run_hour, forecast_hour, [], grid, get_inventory, get_data, preload_paths)
end


end # module SREFHREFShared