import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
# import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
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
# import PlotMap
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

event_names          = collect(keys(event_name_to_events))

function stats_for_run_hours(run_hours)
  total_areas          = Dict(map(event_name -> event_name => 0.0, event_names)) # should end up as CONUS_area * n_days
  painted_areas        = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))
  true_positive_areas  = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))
  false_negative_areas = Dict(map(event_name -> event_name => map(_ -> 0.0, event_name_to_thresholds[event_name]), event_names))

  conus_area = sum(SPCOutlooks.grid().point_areas_sq_miles[Conus.conus_mask_href_cropped_5km_grid()])

  for forecast in filter(forecast -> forecast.run_hour in run_hours, training_and_validation_forecasts)
    # global total_areas
    # global painted_areas
    # global true_positive_areas
    # global false_negative_areas

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

      total_areas[event_name] += conus_area
    end
  end

  println()
  println()
  println("Outlook hours: $(run_hours)")
  println("event_name\tthreshold\ttotal_sq_mi\tpainted_sq_mi\ttrue_positive_sq_mi\tfalse_negative_sq_mi\tsuccess_ratio\tPOD\tpainting_ratio")

  for model_i in 1:length(SPCOutlooks.models)
    event_name, _, _ = SPCOutlooks.models[model_i]

    thresholds = event_name_to_thresholds[event_name]
    for i in 1:length(thresholds)
      threshold = thresholds[i]
      row = [
        event_name,
        threshold,
        Float32(total_areas[event_name]),
        Float32(painted_areas[event_name][i]),
        Float32(true_positive_areas[event_name][i]),
        Float32(false_negative_areas[event_name][i]),
        Float32(true_positive_areas[event_name][i] / painted_areas[event_name][i]), # success ratio
        Float32(true_positive_areas[event_name][i] / (true_positive_areas[event_name][i] + false_negative_areas[event_name][i])), # POD
        Float32(painted_areas[event_name][i] / total_areas[event_name]),
      ]
      println(join(row,"\t"))
    end
  end
end

stats_for_run_hours([6])
# Mon-Sat
# Outlook hours: [6]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD          painting_ratio
# tornado     0.02      2.7770545e9 5.407083e7    2.672531e6          1.4357058e6          0.049426477   0.6505299    0.019470569
# tornado     0.05      2.7770545e9 1.3563396e7   1.515951e6          2.5922855e6          0.111767806   0.36900285   0.0048840945
# tornado     0.1       2.7770545e9 2.37616e6     514057.2            3.5941795e6          0.21633947    0.12512843   0.0008556404
# tornado     0.15      2.7770545e9 382727.7      102061.42           4.0061752e6          0.2666685     0.024843121  0.00013781784
# tornado     0.3       2.7770545e9 84705.9       20985.922           4.0872508e6          0.24775042    0.0051082554 3.0502066e-5
# tornado     0.45      2.7770545e9 0.0           0.0                 4.1082365e6          NaN           0.0          0.0
# tornado     0.6       2.7770545e9 0.0           0.0                 4.1082365e6          NaN           0.0          0.0
# wind        0.05      2.7770545e9 1.5798806e8   2.1593148e7         9.102145e6           0.13667582    0.7034677    0.056890514
# wind        0.15      2.7770545e9 4.5019748e7   1.1600551e7         1.9094742e7          0.25767696    0.37792608   0.016211329
# wind        0.3       2.7770545e9 6.559841e6    2.877264e6          2.7818028e7          0.43861794    0.09373633   0.002362158
# wind        0.45      2.7770545e9 581307.94     363251.8            3.0332042e7          0.62488705    0.011834121  0.00020932537
# wind        0.6       2.7770545e9 0.0           0.0                 3.0695294e7          NaN           0.0          0.0
# hail        0.05      2.7770545e9 1.2271457e8   9.699059e6          3.3617152e6          0.07903755    0.7426098    0.04418875
# hail        0.15      2.7770545e9 3.3856444e7   5.181758e6          7.879016e6           0.15305087    0.39674205   0.012191495
# hail        0.3       2.7770545e9 3.1561588e6   875876.3            1.2184898e7          0.2775134     0.067061596  0.0011365131
# hail        0.45      2.7770545e9 143574.6      82895.34            1.2977879e7          0.5773678     0.006346894  5.1700317e-5
# hail        0.6       2.7770545e9 0.0           0.0                 1.3060774e7          NaN           0.0          0.0
# sig_tornado 0.1       2.7770545e9 1.4062496e6   124848.77           491995.6             0.08878137    0.20239913   0.0005063817
# sig_wind    0.1       2.7770545e9 2.6112152e6   311948.3            3.0869872e6          0.11946479    0.09177823   0.00094028236
# sig_hail    0.1       2.7770545e9 5.3213665e6   459081.44           1.51416e6            0.086271346   0.23265345   0.0019161908

stats_for_run_hours([13])
# Mon-Sat
# Outlook hours: [13]
# event_name  total_areas threshold   painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD         painting_ratio
# tornado     0.02        2.7770545e9 5.5372844e7   2.8299688e6         1.2164862e6          0.051107522   0.6993699   0.019939415
# tornado     0.05        2.7770545e9 1.466232e7    1.7587581e6         2.287697e6           0.119950876   0.43464172  0.0052798097
# tornado     0.1         2.7770545e9 2.7095908e6   625634.3            3.4208208e6          0.23089623    0.15461293  0.0009757067
# tornado     0.15        2.7770545e9 484289.03     141225.56           3.9052295e6          0.2916142     0.034901056 0.00017438947
# tornado     0.3         2.7770545e9 106975.09     35099.85            4.0113552e6          0.32811236    0.008674222 3.8521062e-5
# tornado     0.45        2.7770545e9 0.0           0.0                 4.046455e6           NaN           0.0         0.0
# tornado     0.6         2.7770545e9 0.0           0.0                 4.046455e6           NaN           0.0         0.0
# wind        0.05        2.7770545e9 1.6470022e8   2.2864836e7         7.544904e6           0.138827      0.75189185  0.059307523
# wind        0.15        2.7770545e9 4.7925348e7   1.283787e7          1.757187e7           0.2678722     0.4221631   0.01725762
# wind        0.3         2.7770545e9 7.7312715e6   3.4871905e6         2.692255e7           0.45105004    0.11467347  0.0027839828
# wind        0.45        2.7770545e9 810323.3      534556.2            2.9875184e7          0.65968263    0.017578453 0.00029179236
# wind        0.6         2.7770545e9 0.0           0.0                 3.040974e7           NaN           0.0         0.0
# hail        0.05        2.7770545e9 1.20645064e8  9.878793e6          3.0703745e6          0.08188311    0.7628902   0.043443535
# hail        0.15        2.7770545e9 3.463211e7    5.6297555e6         7.319412e6           0.16255884    0.4347581   0.012470807
# hail        0.3         2.7770545e9 3.5058935e6   1.0632659e6         1.1885902e7          0.30327955    0.08211075  0.0012624504
# hail        0.45        2.7770545e9 192925.61     107259.5            1.2841908e7          0.55596304    0.00828312  6.947131e-5
# hail        0.6         2.7770545e9 0.0           0.0                 1.2949167e7          NaN           0.0         0.0
# sig_tornado 0.1         2.7770545e9 1.7074171e6   172336.77           434094.88            0.10093419    0.28418168  0.00061483023
# sig_wind    0.1         2.7770545e9 3.2282902e6   418233.3            2.9375832e6          0.12955257    0.12462937  0.0011624872
# sig_hail    0.1         2.7770545e9 6.4169265e6   576934.8            1.3895915e6          0.08990827    0.2933776   0.0023106951

stats_for_run_hours([16])
# Mon-Sat
# Outlook hours: [16]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD          painting_ratio
# tornado     0.02      2.780028e9  5.4185816e7   2.7865958e6         1.0590946e6          0.051426664   0.7246022    0.019491106
# tornado     0.05      2.780028e9  1.5079861e7   1.8150101e6         2.0306801e6          0.120359875   0.47195953   0.005424356
# tornado     0.1       2.780028e9  3.0552068e6   723906.1            3.1217842e6          0.2369418     0.18823828   0.0010989842
# tornado     0.15      2.780028e9  537734.7      183038.3            3.662652e6           0.34038776    0.047595695  0.00019342781
# tornado     0.3       2.780028e9  121275.625    44173.5             3.8015168e6          0.36424053    0.011486494  4.3623888e-5
# tornado     0.45      2.780028e9  27254.31      13652.455           3.8320378e6          0.5009283     0.0035500661 9.803611e-6
# tornado     0.6       2.780028e9  0.0           0.0                 3.8456902e6          NaN           0.0          0.0
# wind        0.05      2.780028e9  1.6595149e8   2.2646964e7         6.6206185e6          0.13646737    0.77379006   0.059694186
# wind        0.15      2.780028e9  5.0347852e7   1.3672996e7         1.5594586e7          0.2715706     0.46717203   0.018110557
# wind        0.3       2.780028e9  8.3808605e6   3.8650595e6         2.5402522e7          0.46117693    0.13205941   0.003014668
# wind        0.45      2.780028e9  1.0545485e6   678112.7            2.858947e7           0.643036      0.023169413  0.00037933022
# wind        0.6       2.780028e9  0.0           0.0                 2.9267582e7          NaN           0.0          0.0
# hail        0.05      2.780028e9  1.1923929e8   9.864314e6          2.6036078e6          0.08272705    0.7911755    0.0428914
# hail        0.15      2.780028e9  3.5515284e7   5.8793495e6         6.5885725e6          0.16554421    0.4715581    0.012775154
# hail        0.3       2.780028e9  3.8029982e6   1.2690401e6         1.1198882e7          0.33369464    0.10178441   0.0013679713
# hail        0.45      2.780028e9  230919.22     130569.61           1.2337352e7          0.56543416    0.010472444  8.3063635e-5
# hail        0.6       2.780028e9  0.0           0.0                 1.2467922e7          NaN           0.0          0.0
# sig_tornado 0.1       2.780028e9  1.9121954e6   200446.45           378284.8             0.1048253     0.346355     0.0006878332
# sig_wind    0.1       2.780028e9  4.112904e6    575388.9            2.6308728e6          0.13989845    0.17945786   0.0014794471
# sig_hail    0.1       2.780028e9  6.8661865e6   626315.06           1.2811101e6          0.09121732    0.3283563    0.0024698265


stats_for_run_hours([6,13,16])
# Mon-Sat:
# Outlook hours: [6, 13, 16]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD         painting_ratio
# tornado     0.02      8.3311636e9 1.6343005e8   8.28591e6           3.7083288e6          0.050700042   0.69082415  0.019616714
# tornado     0.05      8.3311636e9 4.3239304e7   5.0897195e6         6.9045195e6          0.11771048    0.424347    0.005190068
# tornado     0.1       8.3311636e9 8.1409575e6   1.8635976e6         1.0130641e7          0.22891627    0.1553744   0.0009771693
# tornado     0.15      8.3311636e9 1.4047514e6   426325.28           1.1567913e7          0.30348805    0.035544172 0.00016861407
# tornado     0.3       8.3311636e9 312956.62     100259.27           1.1893979e7          0.32036155    0.008358953 3.7564576e-5
# tornado     0.45      8.3311636e9 27254.31      13652.455           1.1980586e7          0.5009283     0.001138251 3.271369e-6
# tornado     0.6       8.3311636e9 0.0           0.0                 1.1994239e7          NaN           0.0         0.0
# wind        0.05      8.3311636e9 4.88362e8     6.7104948e7         2.3267668e7          0.13740821    0.7425363   0.058618702
# wind        0.15      8.3311636e9 1.4321832e8   3.8111416e7         5.22612e7            0.26610714    0.42171422  0.017190674
# wind        0.3       8.3311636e9 2.2671974e7   1.0229514e7         8.01431e7            0.45119646    0.11319263  0.0027213453
# wind        0.45      8.3311636e9 2.4461798e6   1.5759208e6         8.8796696e7          0.64423746    0.017438034 0.000293618
# wind        0.6       8.3311636e9 0.0           0.0                 9.0372616e7          NaN           0.0         0.0
# hail        0.05      8.3311636e9 3.6233414e8   2.9442166e7         9.035698e6           0.08125695    0.7651715   0.04349142
# hail        0.15      8.3311636e9 1.0394975e8   1.6690863e7         2.1787e7             0.16056664    0.43377835  0.012477219
# hail        0.3       8.3311636e9 1.046505e7    3.2081822e6         3.526968e7           0.30656156    0.083377354 0.0012561331
# hail        0.45      8.3311636e9 567419.44     320724.47           3.815714e7           0.56523347    0.008335298 6.8108064e-5
# hail        0.6       8.3311636e9 0.0           0.0                 3.8477864e7          NaN           0.0         0.0
# sig_tornado 0.1       8.3311636e9 5.025862e6    497632.0            1.3043752e6          0.09901425    0.27615425  0.00060326053
# sig_wind    0.1       8.3311636e9 9.95241e6     1.3055705e6         8.655443e6           0.13118134    0.13106804  0.0011946002
# sig_hail    0.1       8.3311636e9 1.860448e7    1.6623312e6         4.1848615e6          0.08935113    0.28429562  0.002233119


target_success_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.050700042),
    (0.05, 0.11771048),
    (0.1,  0.22891627),
    (0.15, 0.30348805),
    (0.3,  0.32036155), # lol
    (0.45, 0.5009283),
  ],
  "wind" => [
    (0.05, 0.13740821),
    (0.15, 0.26610714),
    (0.3,  0.45119646),
    (0.45, 0.64423746),
  ],
  "hail" => [
    (0.05, 0.08125695),
    (0.15, 0.16056664),
    (0.3,  0.30656156),
    (0.45, 0.56523347),
  ],
  "sig_tornado" => [(0.1, 0.09901425)],
  "sig_wind"    => [(0.1, 0.13118134)],
  "sig_hail"    => [(0.1, 0.08935113)],
)

target_PODs = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.69082415),
    (0.05, 0.424347),
    (0.1,  0.1553744),
    (0.15, 0.035544172),
    (0.3,  0.008358953),
    (0.45, 0.001138251),
  ],
  "wind" => [
    (0.05, 0.7425363),
    (0.15, 0.42171422),
    (0.3,  0.11319263),
    (0.45, 0.017438034),
  ],
  "hail" => [
    (0.05, 0.7651715),
    (0.15, 0.43377835),
    (0.3,  0.083377354),
    (0.45, 0.008335298),
  ],
  "sig_tornado" => [(0.1, 0.27615425)],
  "sig_wind"    => [(0.1, 0.13106804)],
  "sig_hail"    => [(0.1, 0.28429562)],
)

target_warning_ratio = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.019616714),
    (0.05, 0.005190068),
    (0.1,  0.0009771693),
    (0.15, 0.00016861407),
    (0.3,  3.7564576e-5),
    (0.45, 3.271369e-6),
  ],
  "wind" => [
    (0.05, 0.058618702),
    (0.15, 0.017190674),
    (0.3,  0.0027213453),
    (0.45, 0.000293618),
  ],
  "hail" => [
    (0.05, 0.04349142),
    (0.15, 0.012477219),
    (0.3,  0.0012561331),
    (0.45, 6.8108064e-5),
  ],
  "sig_tornado" => [(0.1, 0.00060326053)],
  "sig_wind"    => [(0.1, 0.0011946002)],
  "sig_hail"    => [(0.1, 0.002233119)],
)
