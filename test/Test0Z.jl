import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import TrainingShared
using Metrics

push!(LOAD_PATH, (@__DIR__) * "/../models/spc_outlooks")
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
import PlotMap
import StormEvents
import ForecastCombinators


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

GRID       = Conus.href_cropped_5km_grid();
CONUS_MASK = Conus.conus_mask_href_cropped_5km_grid();

# Run below is 2019-1-7 through 2021-12-31, but we are missing lots of HREFs between Nov 2020 and mid-March 2021

# conus_area = sum(GRID.point_areas_sq_miles[CONUS_MASK))

spc_forecasts = SPCOutlooks.forecasts_day_0600();

length(spc_forecasts) # 1090


(train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_train_validation_test(
    ForecastCombinators.resample_forecasts(CombinedHREFSREF.forecasts_day_spc_calibrated(), Grids.get_upsampler, GRID);
    just_hours_near_storm_events = false
  );

length(test_forecasts) # 623
test_forecasts = filter(forecast -> forecast.run_hour == 0, test_forecasts);
length(test_forecasts) # 156

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)

test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
length(test_forecasts) # 133

# If you want to augment the test set with all the days after training
#
# training_data_end = Dates.DateTime(2022, 1, 1, 0)
# other_test_forecasts = vcat(train_forecasts, validation_forecasts);
# length(other_test_forecasts) # 6358
# other_test_forecasts = filter(forecast -> forecast.run_hour == 0, other_test_forecasts);
# length(other_test_forecasts) # 796
# other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, other_test_forecasts);
# length(other_test_forecasts) # 694
# other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) > training_data_end, other_test_forecasts);
# length(other_test_forecasts) # 144
#
# test_forecasts = vcat(test_forecasts, other_test_forecasts);
# length(test_forecasts) # 260

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

# Want this sorted for niceness
event_names = map(first, SPCOutlooks.models)

compute_forecast_labels(event_name, spc_forecast) = begin
  events = event_name_to_events[event_name]
  # Annoying that we have to recalculate this.
  start_seconds =
    if spc_forecast.run_hour == 6
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
    elseif spc_forecast.run_hour == 13
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
    elseif spc_forecast.run_hour == 16
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
    end
  end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
  println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
  utc_datetime = Dates.unix2datetime(start_seconds)
  println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end


function forecast_stats(data, labels, threshold)
  painted   = ((@view data[:,1]) .>= threshold*0.9999) .* CONUS_MASK
  unpainted = ((@view data[:,1]) .<  threshold*0.9999) .* CONUS_MASK
  painted_area        = sum(GRID.point_areas_sq_miles[painted])
  true_positive_area  = sum(GRID.point_areas_sq_miles[painted   .* labels])
  false_negative_area = sum(GRID.point_areas_sq_miles[unpainted .* labels])
  (painted_area, true_positive_area, false_negative_area)
end


using ColorVectorSpace # for color math


open((@__DIR__) * "/test_0z.csv", "w") do csv

  headers = ["yymmdd", "spc", "nadocast"]

  for event_name in event_names
    for threshold in event_name_to_thresholds[event_name]
      headers = vcat(headers, ["$(event_name)_spc_painted_sq_mi_$threshold",  "$(event_name)_spc_true_positive_sq_mi_$threshold", "$(event_name)_spc_false_negative_sq_mi_$threshold"])
      headers = vcat(headers, ["$(event_name)_nadocast_painted_sq_mi_$threshold", "$(event_name)_nadocast_true_positive_sq_mi_$threshold", "$(event_name)_nadocast_false_negative_sq_mi_$threshold"])
    end
  end

  println(join(headers, ","))
  println(csv, join(headers, ","))

  for spc_forecast in spc_forecasts
    test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
    if isnothing(test_forecast_i)
      continue
    end
    test_forecast = test_forecasts[test_forecast_i]

    row = [Forecasts.yyyymmdd(spc_forecast), Forecasts.time_title(spc_forecast), Forecasts.time_title(test_forecast)]

    spc_data        = Forecasts.data(spc_forecast)
    ForecastCombinators.turn_forecast_caching_on()
    test_data       = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    for event_name in event_names
      spc_event_i  = findfirst(m -> m[1] == event_name, SPCOutlooks.models)
      test_event_i = findfirst(m -> m[1] == event_name, CombinedHREFSREF.models)

      spc_event_probs  = @view spc_data[:, spc_event_i]
      test_event_probs = @view test_data[:, test_event_i]

      forecast_labels = compute_forecast_labels(event_name, spc_forecast) .> 0.5

      map_root = ((@__DIR__) * "/maps/$(Forecasts.yyyymmdd(spc_forecast))")
      mkpath(map_root)

      post_process(img) = PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img))

      make_plot(file_name, data) = begin
        path = map_root * "/" * file_name
        PlotMap.plot_fast(path, GRID, data .* CONUS_MASK; val_to_color=PlotMap.event_name_to_colorer[event_name], post_process=post_process)
        PlotMap.optimize_png(path; wait = false)
      end

      make_plot("spc_day_1_$(event_name)_$(Forecasts.yyyymmdd_thhz(spc_forecast))", spc_event_probs)
      make_plot("nadocast_$(event_name)_$(Forecasts.yyyymmdd_thhz(test_forecast))", test_event_probs)

      for threshold in event_name_to_thresholds[event_name]
        (spc_painted_area,  spc_true_positive_area,  spc_false_negative_area)  = forecast_stats(spc_event_probs,  forecast_labels, threshold)
        (test_painted_area, test_true_positive_area, test_false_negative_area) = forecast_stats(test_event_probs, forecast_labels, threshold)

        row = vcat(row, [spc_painted_area,  spc_true_positive_area,  spc_false_negative_area])
        row = vcat(row, [test_painted_area, test_true_positive_area, test_false_negative_area])
      end
    end

    println(join(row, ","))
    println(csv, join(row, ","))
  end

end

# scp -r nadocaster2:/home/brian/nadocast_dev/test/ ./





# Do the same, but without the final SPC-like prob rescaling

import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import TrainingShared
using Metrics

push!(LOAD_PATH, (@__DIR__) * "/../models/spc_outlooks")
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
import PlotMap
import StormEvents
import ForecastCombinators


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

GRID       = Conus.href_cropped_5km_grid();
CONUS_MASK = Conus.conus_mask_href_cropped_5km_grid();

# Run below is 2019-1-7 through 2021-12-31, but we are missing lots of HREFs between Nov 2020 and mid-March 2021

# conus_area = sum(GRID.point_areas_sq_miles[CONUS_MASK))

spc_forecasts = SPCOutlooks.forecasts_day_0600();

length(spc_forecasts) # 1090


(train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_train_validation_test(
    ForecastCombinators.resample_forecasts(CombinedHREFSREF.forecasts_day(), Grids.get_upsampler, GRID);
    just_hours_near_storm_events = false
  );

length(test_forecasts) # 623
test_forecasts = filter(forecast -> forecast.run_hour == 0, test_forecasts);
length(test_forecasts) # 156

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)

test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
length(test_forecasts) # 133

# If you want to augment the test set with all the days after training
#
# training_data_end = Dates.DateTime(2022, 1, 1, 0)
# other_test_forecasts = vcat(train_forecasts, validation_forecasts);
# length(other_test_forecasts) # 6358
# other_test_forecasts = filter(forecast -> forecast.run_hour == 0, other_test_forecasts);
# length(other_test_forecasts) # 796
# other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, other_test_forecasts);
# length(other_test_forecasts) # 694
# other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) > training_data_end, other_test_forecasts);
# length(other_test_forecasts) # 144
#
# test_forecasts = vcat(test_forecasts, other_test_forecasts);
# length(test_forecasts) # 260

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

# Want this sorted for niceness
event_names = map(first, SPCOutlooks.models)

compute_forecast_labels(event_name, spc_forecast) = begin
  events = event_name_to_events[event_name]
  # Annoying that we have to recalculate this.
  start_seconds =
    if spc_forecast.run_hour == 6
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
    elseif spc_forecast.run_hour == 13
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
    elseif spc_forecast.run_hour == 16
      Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
    end
  end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
  println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
  utc_datetime = Dates.unix2datetime(start_seconds)
  println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end


function forecast_stats(data, labels, threshold)
  painted   = ((@view data[:,1]) .>= threshold*0.9999) .* CONUS_MASK
  unpainted = ((@view data[:,1]) .<  threshold*0.9999) .* CONUS_MASK
  painted_area        = sum(GRID.point_areas_sq_miles[painted])
  true_positive_area  = sum(GRID.point_areas_sq_miles[painted   .* labels])
  false_negative_area = sum(GRID.point_areas_sq_miles[unpainted .* labels])
  (painted_area, true_positive_area, false_negative_area)
end


using ColorVectorSpace # for color math


open((@__DIR__) * "/test_0z_absolutely_calibrated.csv", "w") do csv

  headers = ["yymmdd", "spc", "nadocast"]

  for event_name in event_names
    for threshold in event_name_to_thresholds[event_name]
      headers = vcat(headers, ["$(event_name)_spc_painted_sq_mi_$threshold",  "$(event_name)_spc_true_positive_sq_mi_$threshold", "$(event_name)_spc_false_negative_sq_mi_$threshold"])
      headers = vcat(headers, ["$(event_name)_nadocast_painted_sq_mi_$threshold", "$(event_name)_nadocast_true_positive_sq_mi_$threshold", "$(event_name)_nadocast_false_negative_sq_mi_$threshold"])
    end
  end

  println(join(headers, ","))
  println(csv, join(headers, ","))

  for spc_forecast in spc_forecasts
    test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
    if isnothing(test_forecast_i)
      continue
    end
    test_forecast = test_forecasts[test_forecast_i]

    row = [Forecasts.yyyymmdd(spc_forecast), Forecasts.time_title(spc_forecast), Forecasts.time_title(test_forecast)]

    spc_data        = Forecasts.data(spc_forecast)
    ForecastCombinators.turn_forecast_caching_on()
    test_data       = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    for event_name in event_names
      spc_event_i  = findfirst(m -> m[1] == event_name, SPCOutlooks.models)
      test_event_i = findfirst(m -> m[1] == event_name, CombinedHREFSREF.models)

      spc_event_probs  = @view spc_data[:, spc_event_i]
      test_event_probs = @view test_data[:, test_event_i]

      forecast_labels = compute_forecast_labels(event_name, spc_forecast) .> 0.5

      map_root = ((@__DIR__) * "/maps/$(Forecasts.yyyymmdd(spc_forecast))")
      mkpath(map_root)

      post_process(img) = PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img))

      make_plot(file_name, data) = begin
        path = map_root * "/" * file_name
        PlotMap.plot_fast(path, GRID, data .* CONUS_MASK; val_to_color=PlotMap.event_name_to_colorer[event_name], post_process=post_process)
        PlotMap.optimize_png(path; wait = false)
      end

      # make_plot("spc_day_1_$(event_name)_$(Forecasts.yyyymmdd_thhz(spc_forecast))", spc_event_probs)
      make_plot("nadocast_absolutely_calibrated_$(event_name)_$(Forecasts.yyyymmdd_thhz(test_forecast))", test_event_probs)

      for threshold in event_name_to_thresholds[event_name]
        (spc_painted_area,  spc_true_positive_area,  spc_false_negative_area)  = forecast_stats(spc_event_probs,  forecast_labels, threshold)
        (test_painted_area, test_true_positive_area, test_false_negative_area) = forecast_stats(test_event_probs, forecast_labels, threshold)

        row = vcat(row, [spc_painted_area,  spc_true_positive_area,  spc_false_negative_area])
        row = vcat(row, [test_painted_area, test_true_positive_area, test_false_negative_area])
      end
    end

    println(join(row, ","))
    println(csv, join(row, ","))
  end

end

# scp -r nadocaster2:/home/brian/nadocast_dev/test/ ./
