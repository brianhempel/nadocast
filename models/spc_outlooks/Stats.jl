import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
import PlotMap
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

# Run below uses outlooks from 2019-01-7 through 2021-12-31

forecasts_day = vcat(SPCOutlooks.forecasts_day_0600(), SPCOutlooks.forecasts_day_1300(), SPCOutlooks.forecasts_day_1630());

(training_forecasts, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(forecasts_day; just_hours_near_storm_events = false);

training_and_validation_forecasts = vcat(training_forecasts, validation_forecasts);

length(training_and_validation_forecasts) #

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
training_and_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, training_and_validation_forecasts);

length(training_and_validation_forecasts) #

@time Forecasts.data(training_and_validation_forecasts[10]) # Check if a forecast loads

# Check that they look right
# test_data = Forecasts.data(training_and_validation_forecasts[10])
# PlotMap.plot_debug_map("day_1_outlook", training_and_validation_forecasts[10].grid, test_data[:,1]); # grid so fine it takes FOREVER

event_name_to_events = Dict(
  "tornado"     => StormEvents.conus_tornado_events(),
  "wind"        => StormEvents.conus_severe_wind_events(),
  "hail"        => StormEvents.conus_severe_hail_events(),
  "sig_tornado" => StormEvents.conus_sig_tornado_events(),
  "sig_wind"    => StormEvents.conus_sig_wind_events(),
  "sig_hail"    => StormEvents.conus_sig_hail_events(),
)

event_name_to_thresholds = Dict(
  "tornado"     => [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
  "wind"        => [0.05, 0.15, 0.3, 0.45, 0.6],
  "hail"        => [0.05, 0.15, 0.3, 0.45, 0.6],
  "sig_tornado" => [0.1],
  "sig_wind"    => [0.1],
  "sig_hail"    => [0.1],
)

compute_forecast_labels(event_name, forecast) = begin
  events = event_name_to_events[event_name]
  # Annoying that we have to recalculate this.
  start_seconds =
    if forecast.run_hour == 6
      Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 6*HOUR
    elseif forecast.run_hour == 13
      Forecasts.run_time_in_seconds_since_epoch_utc(forecast)
    elseif forecast.run_hour == 16
      Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
    end
  end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + HOUR
  # println(event_name)
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime = Dates.unix2datetime(start_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_names = collect(keys(event_name_to_events))
painted_areas        = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))
true_positive_areas  = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))
false_negative_areas = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))


# conus_area           = sum(SPCOutlooks.grid().point_areas_sq _miles[Conus.conus_mask_href_cropped_5km_grid()])

for forecast in training_and_validation_forecasts
  global painted_areas
  global true_positive_areas
  global false_negative_areas

  for model_i in 1:length(SPCOutlooks.models)
    event_name, _, _ = SPCOutlooks.models[model_i]

    forecast_labels = compute_forecast_labels(event_name, forecast) .> 0.5
    data            = Forecasts.data(forecast)

    thresholds = event_name_to_thresholds[event_name]
    for i in 1:length(thresholds)
      threshold = thresholds[i]
      painted   = ((@view data[:,model_i]) .>= threshold*0.999) .* Conus.conus_mask_href_cropped_5km_grid()
      unpainted = ((@view data[:,model_i]) .<  threshold*0.999) .* Conus.conus_mask_href_cropped_5km_grid()
      painted_area        = sum(forecast.grid.point_areas_sq_miles[painted])
      true_positive_area  = sum(forecast.grid.point_areas_sq_miles[painted   .* forecast_labels])
      false_negative_area = sum(forecast.grid.point_areas_sq_miles[unpainted .* forecast_labels])
      println("$event_name\t$(Forecasts.time_title(forecast))\t$threshold\t$painted_area\t$true_positive_area\t$(false_negative_area)")
      painted_areas[event_name][i]        += painted_area
      true_positive_areas[event_name][i]  += true_positive_area
      false_negative_areas[event_name][i] += false_negative_area
    end
  end
end


println()
println()
println("event_name\tthreshold\tpainted_sq_mi\ttrue_positive_sq_mi\tfalse_negative_sq_mi\tsuccess_ratio\tPOD")

for model_i in 1:length(SPCOutlooks.models)
  event_name, _, _ = SPCOutlooks.models[model_i]

  thresholds = event_name_to_thresholds[event_name]
  for i in 1:length(thresholds)
    threshold = thresholds[i]
    row = [
      event_name,
      threshold,
      painted_areas[event_name][i],
      true_positive_areas[event_name][i],
      false_negative_areas[event_name][i],
      true_positive_areas[event_name][i] / painted_areas[event_name][i], # success ratio
      true_positive_areas[event_name][i] / (true_positive_areas[event_name][i] + false_negative_areas[event_name][i]) # POD
    ]
    println(join(row,"\t"))
  end
end

# Mon-Sat:
# threshold       painted_sq_mi   true_positive_sq_mi     false_negative_sq_mi    success_ratio   POD


# target_success_ratios = [
#   (0.02, 0.0485357084712472),
#   (0.05, 0.11142399542325385),
#   (0.1, 0.22373785045573905),
#   (0.15, 0.33311809995812625),
#   (0.3, 0.42931275151328113)
# ]

# target_PODs = [
#   (0.02, 0.672673334503246),
#   (0.05, 0.4098841063979484),
#   (0.1, 0.14041993254469878),
#   (0.15, 0.029338579808637542),
#   (0.3, 0.00679180457194558)
# ]
