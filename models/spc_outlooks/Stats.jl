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

# Run below uses outlooks from 2019-01-7 through 2021-12-30

forecasts_day = vcat(SPCOutlooks.forecasts_day_0600(), SPCOutlooks.forecasts_day_1300(), SPCOutlooks.forecasts_day_1630());

(training_forecasts, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(forecasts_day; just_hours_near_storm_events = false);

training_and_validation_forecasts = vcat(training_forecasts, validation_forecasts);

length(training_and_validation_forecasts) #

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
training_and_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, training_and_validation_forecasts);

length(training_and_validation_forecasts) # 2802

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
# event_name      threshold       painted_sq_mi   true_positive_sq_mi     false_negative_sq_mi    success_ratio           POD
# tornado         0.02    6.386448611681273e7     3.2397647153028967e6    1.6577719483148907e6    0.050728736928645046    0.6615090274606944
# tornado         0.05    1.6694967233203141e7    1.9175904378469985e6    2.979946225770788e6     0.11486038942521988     0.3915418238912139
# tornado         0.1     2.815212805805681e6     638781.5920943354       4.258755071523452e6     0.22690348338037045     0.13042915979366457
# tornado         0.15    446436.83216448         136019.61114461088      4.761517052473178e6     0.3046782911820712      0.027773066438696918
# tornado         0.3     129168.0566771936       40766.24942881179       4.856770414188976e6     0.31560627664076046     0.008323827309277956
# tornado         0.45    0.0                     0.0                     4.897536663617788e6     NaN                     0.0
# tornado         0.6     0.0                     0.0                     4.897536663617788e6     NaN                     0.0
# wind            0.05    1.8581204496095577e8    2.5013717401839804e7    1.007100741559508e7     0.13461838497658143     0.7129517911854782
# wind            0.15    5.3666155698113665e7    1.3677258033300247e7    2.140746678413463e7     0.25485816629457203     0.38983512353226496
# wind            0.3     8.618666714120239e6     3.7471356610639347e6    3.133758915637097e7     0.43476976026058434     0.10680248115276206
# wind            0.45    867222.0427319047       536433.4666554114       3.454829135077946e7     0.6185653041815565      0.015289658660478881
# wind            0.6     0.0                     0.0                     3.508472481743488e7     NaN                     0.0
# hail            0.05    1.4416599602902466e8    1.1563866616911879e7    3.941616126087076e6     0.08021216469509043     0.7457921051915138
# hail            0.15    3.953456035357932e7     6.37945383515651e6      9.126028907842448e6     0.16136397567347519     0.41143213280714935
# hail            0.3     4.0086940423934264e6    1.2071445539445844e6    1.4298338189054368e7    0.3011316257061734      0.0778527553093847
# hail            0.45    275790.3099877621       150146.84283561673      1.5355335900163336e7    0.5444239242570899      0.009683467798086529
# hail            0.6     0.0                     0.0                     1.5505482742998956e7    NaN                     0.0
# sig_tornado     0.1     1.5701683111527124e6    136945.44902796252      633938.3358334055       0.08721705058958064     0.17764733377105618
# sig_wind        0.1     2.806279055756146e6     326624.41913420113      3.4613344426788697e6    0.1163905700911105      0.08622702385365015
# sig_hail        0.1     6.787716545718989e6     568770.6103029839       1.7484464603093283e6    0.08379410166467652     0.2454541775633861

target_success_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.050728736928645046),
    (0.05, 0.11486038942521988),
    (0.1,  0.22690348338037045),
    (0.15, 0.3046782911820712),
    (0.3,  0.31560627664076046), # lol
  ],
  "wind" => [
    (0.05, 0.13461838497658143),
    (0.15, 0.25485816629457203),
    (0.3,  0.43476976026058434),
    (0.45, 0.6185653041815565),
  ],
  "hail" => [
    (0.05, 0.08021216469509043),
    (0.15, 0.16136397567347519),
    (0.3,  0.3011316257061734),
    (0.45, 0.5444239242570899),
  ],
  "sig_tornado" => [(0.1, 0.08721705058958064)],
  "sig_wind"    => [(0.1, 0.1163905700911105)],
  "sig_hail"    => [(0.1, 0.08379410166467652)],
)

target_PODs = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.6615090274606944),
    (0.05, 0.3915418238912139),
    (0.1,  0.13042915979366457),
    (0.15, 0.027773066438696918),
    (0.3,  0.008323827309277956),
  ],
  "wind" => [
    (0.05, 0.7129517911854782),
    (0.15, 0.38983512353226496),
    (0.3,  0.10680248115276206),
    (0.45, 0.015289658660478881),
  ],
  "hail" => [
    (0.05, 0.7457921051915138),
    (0.15, 0.41143213280714935),
    (0.3,  0.0778527553093847),
    (0.45, 0.009683467798086529),
  ],
  "sig_tornado" => [(0.1, 0.17764733377105618)],
  "sig_wind"    => [(0.1, 0.08622702385365015)],
  "sig_hail"    => [(0.1, 0.2454541775633861)],
)

# target_PODs = [
#   (0.02, 0.6615090274606944),
#   (0.05, 0.3915418238912139),
#   (0.1,  0.13042915979366457),
#   (0.15, 0.027773066438696918),
#   (0.3,  0.008323827309277956)
# ]
