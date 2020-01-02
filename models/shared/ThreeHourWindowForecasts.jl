module ThreeHourWindowForecasts

# Concat e.g. +4, +5, +6 hour forecasts together to make a new +5 hour forecast.
#
# Hopefully help the models learn some time insensitivity.

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import ForecastCombinators
import Inventories

function add_misc_suffix_to_inventory_lines(inventory, suffix)
  map(inventory) do inventory_line
    Inventories.revise_with_misc(inventory_line, inventory_line.misc * suffix)
  end
end

function three_hour_window_forecasts(base_forecasts)
  base_forecasts_by_run_time = Forecasts.run_time_seconds_to_forecasts(base_forecasts)

  tag_inventory_lines(forecast, suffix) = begin
    inventory_transformer(forecast, inventory) = add_misc_suffix_to_inventory_lines(inventory, suffix)

    ForecastCombinators.map_forecasts([forecast], inventory_transformer = inventory_transformer)[1]
  end

  associated_forecasts = Tuple{Forecasts.Forecast,Forecasts.Forecast,Forecasts.Forecast}[]

  for base_forecast in base_forecasts
    base_forecast_hour = base_forecast.forecast_hour
    siblings = base_forecasts_by_run_time[Forecasts.run_time_in_seconds_since_epoch_utc(base_forecast)]
    prior_hour_i = findfirst(forecast -> forecast.forecast_hour == base_forecast_hour - 1, siblings)
    next_hour_i  = findfirst(forecast -> forecast.forecast_hour == base_forecast_hour + 1, siblings)
    if !isnothing(prior_hour_i) && !isnothing(next_hour_i)
      prior_forecast = tag_inventory_lines(siblings[prior_hour_i], " -1hr")
      next_forecast  = tag_inventory_lines(siblings[next_hour_i],  " +1hr")
      push!(associated_forecasts, (prior_forecast, base_forecast, next_forecast))
    end
  end

  # base_model_name = isempty(base_forecasts) ? "" : base_forecasts[1].model_name
  # model_name = "$(base_model_name)-1hr|$(base_model_name)|$(base_model_name)+1hr"

  # Use runtime/forecast hour of middle forecast.
  forecasts_tuple_to_canonical_forecast(forecasts_tuple) = forecasts_tuple[2]

  ForecastCombinators.concat_forecasts(
    associated_forecasts,
    forecasts_tuple_to_canonical_forecast = forecasts_tuple_to_canonical_forecast
  )
end

# Separate function for speed, ostensibly.
#
# Reads prior_hour_data, forecast_hour_data, next_hour_data
# Mutates window_min_data, window_mean_data, window_max_data, window_delta_data
function compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)
  Threads.@threads for i in 1:length(prior_hour_data)
    prior_hour_value    = prior_hour_data[i]
    forecast_hour_value = forecast_hour_data[i]
    next_hour_value     = next_hour_data[i]

    window_min_data[i]   = min(prior_hour_value, forecast_hour_value, next_hour_value)
    window_mean_data[i]  = (prior_hour_value + forecast_hour_value + next_hour_value) / 3f0
    window_max_data[i]   = max(prior_hour_value, forecast_hour_value, next_hour_value)
    window_delta_data[i] = next_hour_value - prior_hour_value
  end
end

# 3 hour forecasts plus 3hr min, mean, max, delta
# (7x the features of the base forecasts)
function three_hour_window_and_min_mean_max_delta_forecasts(base_forecasts)
  window_forecasts = three_hour_window_forecasts(base_forecasts)

  inventory_transformer(base_forecast, base_inventory) = begin
    single_hour_feature_count = div(length(base_inventory),3)

    forecast_hour_inventory = base_inventory[(single_hour_feature_count + 1):(2*single_hour_feature_count)]

    min_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr min")
    mean_inventory  = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr mean")
    max_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr max")
    delta_inventory = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr delta")

    vcat(base_inventory, min_inventory, mean_inventory, max_inventory, delta_inventory)
  end

  data_transformer(base_forecast, base_data) = begin
    point_count               = size(base_data, 1)
    single_hour_feature_count = div(size(base_data, 2),3)

    out = Array{Float32}(undef, (point_count, 7*single_hour_feature_count))

    out[:, 1:(3*single_hour_feature_count)] = base_data

    prior_hour_data    = @view out[:, (0*single_hour_feature_count + 1):(1*single_hour_feature_count)]
    forecast_hour_data = @view out[:, (1*single_hour_feature_count + 1):(2*single_hour_feature_count)]
    next_hour_data     = @view out[:, (2*single_hour_feature_count + 1):(3*single_hour_feature_count)]
    window_min_data    = @view out[:, (3*single_hour_feature_count + 1):(4*single_hour_feature_count)]
    window_mean_data   = @view out[:, (4*single_hour_feature_count + 1):(5*single_hour_feature_count)]
    window_max_data    = @view out[:, (5*single_hour_feature_count + 1):(6*single_hour_feature_count)]
    window_delta_data  = @view out[:, (6*single_hour_feature_count + 1):(7*single_hour_feature_count)]

    compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)

    out
  end

  ForecastCombinators.map_forecasts(window_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

end # module ThreeHourWindowForecasts