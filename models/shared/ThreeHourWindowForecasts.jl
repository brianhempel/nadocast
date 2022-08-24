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



function three_hour_window_associated_forecasts(base_forecasts)
  base_forecasts = ForecastCombinators.cache_forecasts(base_forecasts)
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

  associated_forecasts
end

function three_hour_window_forecasts(base_forecasts)
  # base_model_name = isempty(base_forecasts) ? "" : base_forecasts[1].model_name
  # model_name = "$(base_model_name)-1hr|$(base_model_name)|$(base_model_name)+1hr"

  # Use runtime/forecast hour of middle forecast.
  forecasts_tuple_to_canonical_forecast(forecasts_tuple) = forecasts_tuple[2]

  ForecastCombinators.concat_forecasts(
    three_hour_window_associated_forecasts(base_forecasts),
    forecasts_tuple_to_canonical_forecast = forecasts_tuple_to_canonical_forecast
  )
end

# Separate function for speed, ostensibly.
#
# Reads prior_hour_data, forecast_hour_data, next_hour_data
# Mutates out
# function compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)
function compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, out)
  single_hour_length = length(forecast_hour_data)
  Threads.@threads for i in 1:length(prior_hour_data)
    @inbounds begin
      prior_hour_value    = prior_hour_data[i]
      forecast_hour_value = forecast_hour_data[i]
      next_hour_value     = next_hour_data[i]

      out[i+0*single_hour_length] = prior_hour_value
      out[i+1*single_hour_length] = forecast_hour_value
      out[i+2*single_hour_length] = next_hour_value

      # min mean max delta
      out[i+3*single_hour_length] = min(prior_hour_value, forecast_hour_value, next_hour_value)
      out[i+4*single_hour_length] = (prior_hour_value + forecast_hour_value + next_hour_value) / 3f0
      out[i+5*single_hour_length] = max(prior_hour_value, forecast_hour_value, next_hour_value)
      out[i+6*single_hour_length] = next_hour_value - prior_hour_value
    end
  end

  ()
end

# 3 hour forecasts plus 3hr min, mean, max, delta
# (7x the features of the base forecasts)
#
# new_features_post should be a list of pairs of (feature_name, compute_feature_function(base_forecast))
#
# Duplicating a lot of ForecastCombinators.concat_forecasts to save LARGE allocations and copies
function three_hour_window_and_min_mean_max_delta_forecasts(base_forecasts; new_features_post=[])
  associated_forecasts = three_hour_window_associated_forecasts(base_forecasts)
  # window_forecasts = three_hour_window_forecasts(base_forecasts)

  map(associated_forecasts) do (prior_forecast, hour_forecast, next_forecast)
    get_inventory() = begin
      prior_inventory = Forecasts.inventory(prior_forecast)
      hour_inventory  = Forecasts.inventory(hour_forecast)
      next_inventory  = Forecasts.inventory(next_forecast)

      min_inventory   = add_misc_suffix_to_inventory_lines(hour_inventory, " 3hr min")
      mean_inventory  = add_misc_suffix_to_inventory_lines(hour_inventory, " 3hr mean")
      max_inventory   = add_misc_suffix_to_inventory_lines(hour_inventory, " 3hr max")
      delta_inventory = add_misc_suffix_to_inventory_lines(hour_inventory, " 3hr delta")

      new_features_post_lines =
        map(new_features_post) do feature_name_and_compute_function
          Inventories.InventoryLine(
            "",                                   # message_dot_submessage
            "",                                   # position_str
            hour_inventory[1].date_str,
            feature_name_and_compute_function[1], # abbrev
            "calculated",                         # level
            "hour fcst",                          # forecast_hour_str
            "",                                   # misc
            ""                                    # feature_engineering
          )
        end

      vcat(prior_inventory, hour_inventory, next_inventory, min_inventory, mean_inventory, max_inventory, delta_inventory, new_features_post_lines)
    end

    get_data() = begin
      # out_datas = map(Forecasts.data, forecasts_array) # Vector of 2D arrays

      # Threaded hcat

      # sizes = map(size, out_datas)

      # point_count = first(sizes[1])

      # for out_data in out_datas
      #   @assert size(out_data, 1) == point_count
      # end

      # aggregate_sizes = cumsum(map(last, sizes))
      # feature_count   = last(aggregate_sizes)

      # # print("Concating... ")
      # # 0.6s out of 15s loading time. trivial allocation count
      # begin
      #   out = Array{Float32}(undef, (point_count, feature_count))

      #   Threads.@threads for feature_i in 1:feature_count
      #     out_data_i         = findfirst(n -> feature_i <= n, aggregate_sizes)
      #     out_data           = out_datas[out_data_i]
      #     out_data_feature_i = out_data_i == 1 ? feature_i : feature_i - aggregate_sizes[out_data_i - 1]

      #     out[:, feature_i] = @view out_data[:, out_data_feature_i]
      #   end

      #   out
      # end

      prior_hour_data, forecast_hour_data, next_hour_data = Forecasts.data(prior_forecast), Forecasts.data(hour_forecast), Forecasts.data(next_forecast)

      point_count, single_hour_feature_count = size(forecast_hour_data)
      hours_feature_count                    = 7*single_hour_feature_count


      # print("Computing min mean max delta + climatology")
      # 1.6s out of 15s. minimal allocations, albeit LARGE

      out = Array{Float32}(undef, (point_count, hours_feature_count + length(new_features_post)))

      # prior_hour_data    = @view out[:, (0*single_hour_feature_count + 1):(1*single_hour_feature_count)]
      # forecast_hour_data = @view out[:, (1*single_hour_feature_count + 1):(2*single_hour_feature_count)]
      # next_hour_data     = @view out[:, (2*single_hour_feature_count + 1):(3*single_hour_feature_count)]

      # Threads.@threads for feature_i in 1:single_hour_feature_count
      #   prior_hour_data[:, feature_i] = @view prior_data[:, feature_i]
      # end
      # Threads.@threads for feature_i in 1:single_hour_feature_count
      #   forecast_hour_data[:, feature_i] = @view hour_data[:, feature_i]
      # end
      # Threads.@threads for feature_i in 1:single_hour_feature_count
      #   next_hour_data[:, feature_i] = @view next_data[:, feature_i]
      # end
      # prior_data, hour_data, next_data = nothing, nothing, nothing # free

      # window_min_data    = @view out[:, (3*single_hour_feature_count + 1):(4*single_hour_feature_count)]
      # window_mean_data   = @view out[:, (4*single_hour_feature_count + 1):(5*single_hour_feature_count)]
      # window_max_data    = @view out[:, (5*single_hour_feature_count + 1):(6*single_hour_feature_count)]
      # window_delta_data  = @view out[:, (6*single_hour_feature_count + 1):(7*single_hour_feature_count)]

      # compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)
      compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, out)

      Threads.@threads for new_post_feature_i in 1:length(new_features_post)
        new_feature_name, compute_new_feature_post = new_features_post[new_post_feature_i]
        out[:, hours_feature_count + new_post_feature_i] = compute_new_feature_post(hour_forecast)
      end

      out
    end

    forecasts_array = [prior_forecast, hour_forecast, next_forecast]

    canonical_forecast = hour_forecast

    model_names = map(forecast -> forecast.model_name, forecasts_array)

    preload_paths = vcat(map(forecast -> forecast.preload_paths, forecasts_array)...)

    Forecasts.Forecast(join(model_names, "|"), canonical_forecast.run_year, canonical_forecast.run_month, canonical_forecast.run_day, canonical_forecast.run_hour, canonical_forecast.forecast_hour, forecasts_array, canonical_forecast.grid, get_inventory, get_data, preload_paths)
  end


  # inventory_transformer(base_forecast, base_inventory) = begin
  #   single_hour_feature_count = div(length(base_inventory),3)

  #   forecast_hour_inventory = base_inventory[(single_hour_feature_count + 1):(2*single_hour_feature_count)]

  #   min_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr min")
  #   mean_inventory  = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr mean")
  #   max_inventory   = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr max")
  #   delta_inventory = add_misc_suffix_to_inventory_lines(forecast_hour_inventory, " 3hr delta")

  #   new_features_post_lines =
  #     map(new_features_post) do feature_name_and_compute_function
  #       Inventories.InventoryLine(
  #         "",                                   # message_dot_submessage
  #         "",                                   # position_str
  #         base_inventory[1].date_str,
  #         feature_name_and_compute_function[1], # abbrev
  #         "calculated",                         # level
  #         "hour fcst",                          # forecast_hour_str
  #         "",                                   # misc
  #         ""                                    # feature_engineering
  #       )
  #     end

  #   vcat(base_inventory, min_inventory, mean_inventory, max_inventory, delta_inventory, new_features_post_lines)
  # end

  # data_transformer(base_forecast, base_data) = begin
  #   point_count               = size(base_data, 1)
  #   single_hour_feature_count = div(size(base_data, 2),3)
  #   hours_feature_count       = 7*single_hour_feature_count

  #   # print("Computing min mean max delta + climatology")
  #   # 1.6s out of 15s. minimal allocations, albeit LARGE
  #   begin
  #     out = Array{Float32}(undef, (point_count, hours_feature_count + length(new_features_post)))

  #     Threads.@threads for feature_i in 1:(3*single_hour_feature_count)
  #       out[:, feature_i] = @view base_data[:, feature_i]
  #     end

  #     prior_hour_data    = @view out[:, (0*single_hour_feature_count + 1):(1*single_hour_feature_count)]
  #     forecast_hour_data = @view out[:, (1*single_hour_feature_count + 1):(2*single_hour_feature_count)]
  #     next_hour_data     = @view out[:, (2*single_hour_feature_count + 1):(3*single_hour_feature_count)]
  #     window_min_data    = @view out[:, (3*single_hour_feature_count + 1):(4*single_hour_feature_count)]
  #     window_mean_data   = @view out[:, (4*single_hour_feature_count + 1):(5*single_hour_feature_count)]
  #     window_max_data    = @view out[:, (5*single_hour_feature_count + 1):(6*single_hour_feature_count)]
  #     window_delta_data  = @view out[:, (6*single_hour_feature_count + 1):(7*single_hour_feature_count)]

  #     compute_min_mean_max_delta!(prior_hour_data, forecast_hour_data, next_hour_data, window_min_data, window_mean_data, window_max_data, window_delta_data)

  #     Threads.@threads for new_post_feature_i in 1:length(new_features_post)
  #       new_feature_name, compute_new_feature_post = new_features_post[new_post_feature_i]
  #       out[:, hours_feature_count + new_post_feature_i] = compute_new_feature_post(base_forecast)
  #     end

  #     out
  #   end
  # end

  # forecasts_tuple_to_canonical_forecast(forecasts_tuple) = forecasts_tuple[2]

  # ForecastCombinators.map_forecasts(window_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

function three_hour_window_and_min_mean_max_delta_forecasts_with_climatology_etc(forecasts; run_datetime_to_simulation_version)
  grid = forecasts[1].grid

  new_features_post = [
    Climatology.hail_day_spatial_probability_feature(grid),
    Climatology.hail_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.hail_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.severe_day_spatial_probability_feature(grid),
    Climatology.sig_hail_day_spatial_probability_feature(grid),
    Climatology.sig_hail_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.sig_hail_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.sig_severe_day_spatial_probability_feature(grid),
    Climatology.sig_severe_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.sig_severe_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.sig_tornado_day_spatial_probability_feature(grid),
    Climatology.sig_tornado_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.sig_tornado_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.sig_wind_day_spatial_probability_feature(grid),
    Climatology.sig_wind_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.sig_wind_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.tornado_day_spatial_probability_feature(grid),
    Climatology.tornado_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.tornado_day_given_severe_day_spatial_probability_feature(grid),
    Climatology.wind_day_spatial_probability_feature(grid),
    Climatology.wind_day_geomean_absolute_and_conditional_spatial_probability_feature(grid),
    Climatology.wind_day_given_severe_day_spatial_probability_feature(grid),

    Climatology.asos_gust_days_per_year_feature(grid),
    Climatology.asos_sig_gust_days_per_year_feature(grid),

    Climatology.hour_in_day_severe_probability_feature(grid),

    Climatology.month_tornado_day_probability_feature(grid),
    Climatology.month_wind_day_probability_feature(grid),
    Climatology.month_hail_day_probability_feature(grid),
    Climatology.month_severe_day_probability_feature(grid),
    Climatology.month_tornado_day_given_severe_day_probability_feature(grid),
    Climatology.month_wind_day_given_severe_day_probability_feature(grid),
    Climatology.month_hail_day_given_severe_day_probability_feature(grid),

    Climatology.population_density_feature(grid),

    ("simulation_version", forecast -> Climatology.fill_grid(run_datetime_to_simulation_version(Forecasts.run_utc_datetime(forecast)), grid))
  ]

  three_hour_window_and_min_mean_max_delta_forecasts(
    forecasts;
    new_features_post = new_features_post
  )
end


end # module ThreeHourWindowForecasts