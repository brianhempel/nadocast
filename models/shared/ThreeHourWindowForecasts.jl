module ThreeHourWindowForecasts

# Concat e.g. +4, +5, +6 hour forecasts together to make a new +5 hour forecast.
#
# Hopefully help the models learn some time insensitivity.

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import ForecastCombinators
import Inventories

push!(LOAD_PATH, @__DIR__)
import Climatology
import FeatureEngineeringShared

function add_misc_suffix_to_inventory_lines(inventory, suffix)
  map(inventory) do inventory_line
    Inventories.revise_with_misc(inventory_line, inventory_line.misc * suffix)
  end
end

function three_hour_window_forecasts(base_forecasts)
  # base_forecasts = ForecastCombinators.cache_forecasts(base_forecasts)
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
#
# new_features_post should be a list of pairs of (feature_name, compute_feature_function(base_forecast))
function three_hour_window_and_min_mean_max_delta_forecasts(base_forecasts; new_features_post=[])
  window_forecasts = three_hour_window_forecasts(base_forecasts)

  inventory_transformer(base_forecast, base_inventory) = begin
    single_hour_feature_count = div(length(base_inventory),3)

    forecast_hour_inventory = base_inventory[(single_hour_feature_count + 1):(2*single_hour_feature_count)]

    min_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr min")
    mean_inventory  = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr mean")
    max_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr max")
    delta_inventory = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr delta")

    new_features_post_lines =
      map(new_features_post) do feature_name_and_compute_function
        Inventories.InventoryLine(
          "",                                   # message_dot_submessage
          "",                                   # position_str
          base_inventory[1].date_str,
          feature_name_and_compute_function[1], # abbrev
          "calculated",                         # level
          "hour fcst",                          # forecast_hour_str
          "",                                   # misc
          ""                                    # feature_engineering
        )
      end

    vcat(base_inventory, min_inventory, mean_inventory, max_inventory, delta_inventory, new_features_post_lines)
  end

  data_transformer(base_forecast, base_data) = begin
    point_count               = size(base_data, 1)
    single_hour_feature_count = div(size(base_data, 2),3)
    hours_feature_count       = 7*single_hour_feature_count

    out = Array{Float32}(undef, (point_count, hours_feature_count + length(new_features_post)))

    Threads.@threads for feature_i in 1:(3*single_hour_feature_count)
      out[:, feature_i] = @view base_data[:, feature_i]
    end

    prior_hour_data    = @view out[:, (0*single_hour_feature_count + 1):(1*single_hour_feature_count)]
    forecast_hour_data = @view out[:, (1*single_hour_feature_count + 1):(2*single_hour_feature_count)]
    next_hour_data     = @view out[:, (2*single_hour_feature_count + 1):(3*single_hour_feature_count)]
    window_min_data    = @view out[:, (3*single_hour_feature_count + 1):(4*single_hour_feature_count)]
    window_mean_data   = @view out[:, (4*single_hour_feature_count + 1):(5*single_hour_feature_count)]
    window_max_data    = @view out[:, (5*single_hour_feature_count + 1):(6*single_hour_feature_count)]
    window_delta_data  = @view out[:, (6*single_hour_feature_count + 1):(7*single_hour_feature_count)]

    compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)

    Threads.@threads for new_post_feature_i in 1:length(new_features_post)
      new_feature_name, compute_new_feature_post = new_features_post[new_post_feature_i]
      out[:, hours_feature_count + new_post_feature_i] = compute_new_feature_post(base_forecast)
    end

    out
  end

  ForecastCombinators.map_forecasts(window_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

function three_hour_window_and_min_mean_max_delta_forecasts_with_climatology(forecasts)
  grid = forecasts[1].grid

  new_features_post = [
    FeatureEngineeringShared.forecast_hour_feature_post(grid),
    Climatology.tornado_day_spacial_probability_feature(grid),
    Climatology.severe_day_spacial_probability_feature(grid),
    Climatology.tornado_day_given_severe_day_spacial_probability_feature(grid),
    Climatology.geomean_tornado_and_conditional_spacial_probability_feature(grid),
    Climatology.forecast_hour_tornado_probability_feature(grid),
    Climatology.forecast_hour_severe_probability_feature(grid),
    Climatology.forecast_hour_tornado_given_severe_probability_feature(grid),
    Climatology.forecast_hour_geomean_tornado_and_conditional_probability_feature(grid),
    Climatology.month_tornado_day_probability_feature(grid),
    Climatology.month_severe_day_probability_feature(grid),
    Climatology.month_tornado_day_given_severe_day_probability_feature(grid),
    Climatology.month_geomean_tornado_and_conditional_probability_feature(grid)
  ]

  three_hour_window_and_min_mean_max_delta_forecasts(
    forecasts;
    new_features_post = new_features_post
  )
end


end # module ThreeHourWindowForecasts