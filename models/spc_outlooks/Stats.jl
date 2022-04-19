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

# Run below uses outlooks from 2019-01-7 through 2020-10-31

forecasts_day = vcat(SPCOutlooks.forecasts_day_0600(), SPCOutlooks.forecasts_day_1300(), SPCOutlooks.forecasts_day_1630());

(training_forecasts, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(forecasts_day; just_hours_near_storm_events = false);

training_and_validation_forecasts = vcat(training_forecasts, validation_forecasts);

length(training_and_validation_forecasts) #

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
training_and_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, training_and_validation_forecasts);

length(training_and_validation_forecasts) # 282

@time Forecasts.data(training_and_validation_forecasts[10]) # Check if a forecast loads

# Check that they look right
# test_data = Forecasts.data(training_and_validation_forecasts[10])
# PlotMap.plot_debug_map("day_1_outlook", training_and_validation_forecasts[10].grid, test_data[:,1]); # grid so fine it takes FOREVER

compute_forecast_labels(forecast) = begin
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
  println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  utc_datetime = Dates.unix2datetime(start_seconds)
  println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_conus_tornado_neighborhoods(forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

painted_areas        = map(_ -> 0.0, SPCOutlooks.thresholds)
true_positive_areas  = map(_ -> 0.0, SPCOutlooks.thresholds)
false_negative_areas = map(_ -> 0.0, SPCOutlooks.thresholds)
# conus_area           = sum(SPCOutlooks.grid().point_areas_sq _miles[Conus.conus_mask_href_cropped_5km_grid])

for forecast in training_and_validation_forecasts
  global painted_areas
  global true_positive_areas
  global false_negative_areas

  forecast_labels = compute_forecast_labels(forecast) .> 0.5
  data            = Forecasts.data(forecast)

  for i in 1:length(SPCOutlooks.thresholds)
    threshold = SPCOutlooks.thresholds[i]
    painted   = ((@view data[:,1]) .>= threshold*0.999) .* Conus.conus_mask_href_cropped_5km_grid()
    unpainted = ((@view data[:,1]) .<  threshold*0.999) .* Conus.conus_mask_href_cropped_5km_grid()
    painted_area        = sum(forecast.grid.point_areas_sq_miles[painted])
    true_positive_area  = sum(forecast.grid.point_areas_sq_miles[painted   .* forecast_labels])
    false_negative_area = sum(forecast.grid.point_areas_sq_miles[unpainted .* forecast_labels])
    println("$(Forecasts.time_title(forecast))\t$threshold\t$painted_area\t$true_positive_area\t$(false_negative_area)\t$(true_positive_area/painted_area)")
    painted_areas[i]        += painted_area
    true_positive_areas[i]  += true_positive_area
    false_negative_areas[i] += false_negative_area
  end
end

println()
println()
println("threshold\tpainted_sq_mi\ttrue_positive_sq_mi\tfalse_negative_sq_mi\tsuccess_ratio\tPOD")

for i in 1:length(SPCOutlooks.thresholds)
  threshold = SPCOutlooks.thresholds[i]
  println("$threshold\t$(painted_areas[i])\t$(true_positive_areas[i])\t$(false_negative_areas[i])\t$(true_positive_areas[i]/painted_areas[i])\t$(true_positive_areas[i]/(true_positive_areas[i] + false_negative_areas[i]))")
end

# Saturdays only:
# 0.02    2.1466801580716092e7    1.019805790261507e6     0.04750618234518979
# 0.05    5.62979933654829e6      632177.9355236491       0.11229137980453249
# 0.1     1.3459093366134688e6    252723.45228532178      0.18777152770276184
# 0.15    282470.3387442689       71442.54191712453       0.2529205092284191
# 0.3     0.0     0.0     NaN
# 0.45    0.0     0.0     NaN
# 0.6     0.0     0.0     NaN


# Mon-Sat:
# threshold       painted_sq_mi   true_positive_sq_mi     false_negative_sq_mi    success_ratio   POD
# 0.02    1.0804878406953493e8    5.244224284271685e6     2.5518693250348675e6    0.0485357084712472      0.672673334503246
# 0.05    2.8678695736111447e7    3.1954948624453717e6    4.600598746861178e6     0.11142399542325385     0.4098841063979484
# 0.1     4.892900045750411e6     1.0947269387309842e6    6.701366670575575e6     0.22373785045573905     0.14041993254469878
# 0.15    686622.2957593745       228726.31455224945      7.567367294754305e6     0.33311809995812625     0.029338579808637542
# 0.3     123335.59632776468      52949.54421900399       7.743144065087553e6     0.42931275151328113     0.00679180457194558
# 0.45    18883.740121199895      9320.584653272863       7.786773024653284e6     0.49357725712445477     0.0011955455027048995
# 0.6     0.0     0.0     7.796093609306557e6     NaN     0.0

target_success_ratios = [
  (0.02, 0.0485357084712472),
  (0.05, 0.11142399542325385),
  (0.1, 0.22373785045573905),
  (0.15, 0.33311809995812625),
  (0.3, 0.42931275151328113)
]

target_PODs = [
  (0.02, 0.672673334503246),
  (0.05, 0.4098841063979484),
  (0.1, 0.14041993254469878),
  (0.15, 0.029338579808637542),
  (0.3, 0.00679180457194558)
]
