import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, (@__DIR__) * "/../models/spc_outlooks")
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_hrrr_rap_href_sref")
import CombinedHRRRRAPHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
import PlotMap
import StormEvents
import ForecastCombinators


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

GRID       = Conus.href_cropped_5km_grid;
CONUS_MASK = Conus.conus_mask_href_cropped_5km_grid;

# Run below is 2019-1-7 through 2021-5-31, but we are missing lots of HREFs between Nov 2020 and mid-March 2021

# conus_area = sum(GRID.point_areas_sq_miles[CONUS_MASK))

(_, _, spc_test_forecasts) = TrainingShared.forecasts_train_validation_test(SPCOutlooks.forecasts_day_1300(); just_hours_near_storm_events = false);

length(spc_test_forecasts) # 129

# We don't have storm events past this time.
cutoff = Dates.DateTime(2021, 7, 1, 0)
spc_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, spc_test_forecasts);

length(spc_test_forecasts) # 129


(_, _, test_forecasts) =
  TrainingShared.forecasts_train_validation_test(
    ForecastCombinators.resample_forecasts(CombinedHRRRRAPHREFSREF.forecasts_day_spc_calibrated(), Grids.get_upsampler, GRID);
    just_hours_near_storm_events = false
  );

length(test_forecasts) # 207
test_forecasts = filter(forecast -> forecast.run_hour == 10, test_forecasts);
length(test_forecasts) # 103
test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
length(test_forecasts) # 91


compute_forecast_labels(spc_forecast) = begin
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
  StormEvents.grid_to_conus_tornado_neighborhoods(spc_forecast.grid, TrainingShared.TORNADO_SPACIAL_RADIUS_MILES, window_mid_time, window_half_size)
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


open((@__DIR__) * "/test_10z.csv", "w") do csv

  headers = ["yymmdd", "spc", "nadocast"]

  for threshold in SPCOutlooks.thresholds
    headers = vcat(headers, ["spc_painted_sq_mi_$threshold",  "spc_true_positive_sq_mi_$threshold",  "spc_false_negative_sq_mi_$threshold"])
    headers = vcat(headers, ["nadocast_painted_sq_mi_$threshold", "nadocast_true_positive_sq_mi_$threshold", "nadocast_false_negative_sq_mi_$threshold"])
  end

  println(join(headers, ","))
  println(csv, join(headers, ","))

  for spc_forecast in spc_test_forecasts
    test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
    if isnothing(test_forecast_i)
      continue
    end
    test_forecast = test_forecasts[test_forecast_i]

    forecast_labels = compute_forecast_labels(spc_forecast) .> 0.5
    spc_data        = Forecasts.data(spc_forecast)
    ForecastCombinators.turn_forecast_caching_on()
    test_data       = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    map_root = ((@__DIR__) * "/maps/$(Forecasts.yyyymmdd(spc_forecast))")
    mkpath(map_root)

    post_process(img) = PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img))

    PlotMap.plot_fast(map_root * "/spc_day_1_$(Forecasts.yyyymmdd_thhz(spc_forecast))", GRID, spc_data  .* CONUS_MASK; val_to_color=PlotMap.prob_to_spc_color, post_process=post_process)
    PlotMap.plot_fast(map_root * "/nadocast_$(Forecasts.yyyymmdd_thhz(test_forecast))", GRID, test_data .* CONUS_MASK; val_to_color=PlotMap.prob_to_spc_color, post_process=post_process)

    row = [Forecasts.yyyymmdd(spc_forecast), Forecasts.time_title(spc_forecast), Forecasts.time_title(test_forecast)]

    for threshold in SPCOutlooks.thresholds
      (spc_painted_area,  spc_true_positive_area,  spc_false_negative_area)  = forecast_stats(spc_data,  forecast_labels, threshold)
      (test_painted_area, test_true_positive_area, test_false_negative_area) = forecast_stats(test_data, forecast_labels, threshold)

      row = vcat(row, [spc_painted_area,  spc_true_positive_area,  spc_false_negative_area])
      row = vcat(row, [test_painted_area, test_true_positive_area, test_false_negative_area])
    end

    println(join(row, ","))
    println(csv, join(row, ","))
  end

end

# scp -r nadocaster2:/home/brian/nadocast_dev/test/ ./
