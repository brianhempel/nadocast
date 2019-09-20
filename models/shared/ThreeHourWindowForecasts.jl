module ThreeHourWindowForecasts

# Concat e.g. +4, +5, +6 hour forecasts together to make a new +5 hour forecast.
#
# Hopefully help the models learn some time insensitivity.

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import ForecastCombinators
import Inventories


function three_hour_window_forecasts(base_forecasts)
  base_forecasts_by_run_time = Forecasts.run_time_seconds_to_forecasts(base_forecasts)

  tag_inventory_lines(forecast, suffix) = begin
    inventory_transformer(forecast, inventory) = begin
      map(inventory) do inventory_line
        Inventories.revise_with_misc(inventory_line, inventory_line.misc * suffix)
      end
    end

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

  base_model_name = isempty(base_forecasts) ? "" : base_forecasts[1].model_name
  model_name = "$(base_model_name)-1hr|$(base_model_name)|$(base_model_name)+1hr"

  map(ForecastCombinators.concat_forecasts(associated_forecasts)) do three_hour_forecast
    # concat_forecasts always chooses the latest forecast as the canonical: we want the middle.
    Forecasts.Forecast(
      model_name,
      three_hour_forecast.run_year,
      three_hour_forecast.run_month,
      three_hour_forecast.run_day,
      three_hour_forecast.run_hour,
      three_hour_forecast.forecast_hour - 1,
      three_hour_forecast.based_on,
      three_hour_forecast.grid,
      three_hour_forecast._get_inventory,
      three_hour_forecast._get_data
    )
  end
end

end # module ThreeHourWindowForecasts