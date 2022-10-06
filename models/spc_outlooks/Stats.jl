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

# Run below uses outlooks from 2018-6-29 through 2022-5-31

forecasts_day = vcat(SPCOutlooks.forecasts_day_0600(), SPCOutlooks.forecasts_day_1300(), SPCOutlooks.forecasts_day_1630());

(training_forecasts, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(forecasts_day; just_hours_near_storm_events = false);

training_and_validation_forecasts = vcat(training_forecasts, validation_forecasts);

length(training_and_validation_forecasts) # 3684

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
training_and_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, training_and_validation_forecasts);

length(training_and_validation_forecasts) # 3684

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

  verifiable_grid_bitmask = Conus.conus_mask_href_cropped_5km_grid() .&& TrainingShared.is_verifiable.(SPCOutlooks.grid().latlons) :: BitVector

  conus_verifiable_area = sum(SPCOutlooks.grid().point_areas_sq_miles[verifiable_grid_bitmask])

  for forecast in filter(forecast -> forecast.run_hour in run_hours, training_and_validation_forecasts)
    # global total_areas
    # global painted_areas
    # global true_positive_areas
    # global false_negative_areas
    data = Forecasts.data(forecast)

    Threads.@threads for model_i in 1:length(SPCOutlooks.models)
      event_name, _, _ = SPCOutlooks.models[model_i]

      forecast_labels = compute_forecast_labels(event_name, forecast) .> 0.5

      thresholds = event_name_to_thresholds[event_name]
      for i in 1:length(thresholds)
        threshold = thresholds[i]
        painted   = ((@view data[:,model_i]) .>= threshold*0.999) .* verifiable_grid_bitmask
        unpainted = ((@view data[:,model_i]) .<  threshold*0.999) .* verifiable_grid_bitmask
        painted_area        = sum(forecast.grid.point_areas_sq_miles[painted])
        true_positive_area  = sum(forecast.grid.point_areas_sq_miles[painted   .* forecast_labels])
        false_negative_area = sum(forecast.grid.point_areas_sq_miles[unpainted .* forecast_labels])
        println("$event_name\t$(Forecasts.time_title(forecast))\t$threshold\t$painted_area\t$true_positive_area\t$(false_negative_area)")
        painted_areas[event_name][i]        += painted_area
        true_positive_areas[event_name][i]  += true_positive_area
        false_negative_areas[event_name][i] += false_negative_area
      end

      total_areas[event_name] += conus_verifiable_area
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
# tornado     0.02      2.7857797e9 7.0551576e7   3.712704e6          1.7329516e6          0.05262397    0.68177354   0.025325613
# tornado     0.05      2.7857797e9 1.944113e7    2.2395662e6         3.2060895e6          0.11519733    0.41125742   0.006978703
# tornado     0.1       2.7857797e9 3.7065245e6   831750.2            4.6139055e6          0.22440165    0.15273647   0.0013305161
# tornado     0.15      2.7857797e9 577339.5      165682.36           5.2799735e6          0.28697562    0.030424684  0.0002072452
# tornado     0.3       2.7857797e9 84705.9       20985.922           5.4246695e6          0.24775042    0.0038537    3.0406534e-5
# tornado     0.45      2.7857797e9 0.0           0.0                 5.4456555e6          NaN           0.0          0.0
# tornado     0.6       2.7857797e9 0.0           0.0                 5.4456555e6          NaN           0.0          0.0
# wind        0.05      2.7857797e9 1.927985e8    2.6784788e7         1.0958834e7          0.13892634    0.7096507    0.06920809
# wind        0.15      2.7857797e9 5.772987e7    1.4647687e7         2.3095936e7          0.25372803    0.38808376   0.020723058
# wind        0.3       2.7857797e9 8.860872e6    3.8965508e6         3.384707e7           0.43974802    0.10323733   0.003180751
# wind        0.45      2.7857797e9 855613.3      526978.56           3.7216644e7          0.61590743    0.0139620565 0.00030713604
# wind        0.6       2.7857797e9 0.0           0.0                 3.7743624e7          NaN           0.0          0.0
# hail        0.05      2.7857797e9 1.5080246e8   1.2204424e7         4.020566e6           0.080929875   0.7521992    0.054132946
# hail        0.15      2.7857797e9 4.2428616e7   6.5944735e6         9.630516e6           0.15542515    0.4064393    0.0152304275
# hail        0.3       2.7857797e9 3.9136002e6   1.1310906e6         1.5093899e7          0.28901535    0.06971287   0.0014048492
# hail        0.45      2.7857797e9 171313.97     93316.85            1.6131673e7          0.5447125     0.0057514273 6.149588e-5
# hail        0.6       2.7857797e9 0.0           0.0                 1.622499e7           NaN           0.0          0.0
# sig_tornado 0.1       2.7857797e9 2.4261432e6   206186.47           648654.5             0.08498528    0.24119863   0.00087090285
# sig_wind    0.1       2.7857797e9 3.06824e6     359918.2            3.8815878e6          0.11730445    0.08485623   0.0011013936
# sig_hail    0.1       2.7857797e9 6.755856e6    577385.6            1.8802576e6          0.08546447    0.23493469   0.002425122

stats_for_run_hours([13])
# Mon-Sat
# Outlook hours: [13]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD          painting_ratio
# tornado     0.02      2.7857797e9 7.181403e7    3.9250882e6         1.4308548e6          0.054656286   0.7328473    0.025778791
# tornado     0.05      2.7857797e9 2.0791076e7   2.5409858e6         2.8149572e6          0.122215204   0.4744236    0.007463288
# tornado     0.1       2.7857797e9 4.1120465e6   987255.8            4.368687e6           0.24008869    0.18432903   0.0014760846
# tornado     0.15      2.7857797e9 714689.94     223072.45           5.1328705e6          0.3121248     0.04164952   0.00025654933
# tornado     0.3       2.7857797e9 106975.09     35099.85            5.320843e6           0.32811236    0.00655344   3.8400416e-5
# tornado     0.45      2.7857797e9 0.0           0.0                 5.355943e6           NaN           0.0          0.0
# tornado     0.6       2.7857797e9 0.0           0.0                 5.355943e6           NaN           0.0          0.0
# wind        0.05      2.7857797e9 1.9803082e8   2.8129798e7         9.267732e6           0.14204758    0.7521833    0.07108632
# wind        0.15      2.7857797e9 6.0517204e7   1.6045295e7         2.1352236e7          0.2651361     0.42904693   0.021723615
# wind        0.3       2.7857797e9 1.0394136e7   4.6899025e6         3.2707628e7          0.45120656    0.12540674   0.0037311409
# wind        0.45      2.7857797e9 1.1788485e6   773819.94           3.662371e7           0.6564202     0.02069174   0.00042316646
# wind        0.6       2.7857797e9 0.0           0.0                 3.7397532e7          NaN           0.0          0.0
# hail        0.05      2.7857797e9 1.4662266e8   1.2409142e7         3.667435e6           0.08463319    0.7718771    0.052632537
# hail        0.15      2.7857797e9 4.286728e7    7.098905e6          8.977672e6           0.16560194    0.4415682    0.015387894
# hail        0.3       2.7857797e9 4.350951e6    1.349659e6          1.4726918e7          0.3101986     0.08395189   0.0015618432
# hail        0.45      2.7857797e9 243860.84     131897.38           1.5944679e7          0.5408715     0.0082043195 8.753774e-5
# hail        0.6       2.7857797e9 0.0           0.0                 1.6076577e7          NaN           0.0          0.0
# sig_tornado 0.1       2.7857797e9 2.6702575e6   261720.22           578490.0             0.0980131     0.31149372   0.00095853145
# sig_wind    0.1       2.7857797e9 4.1214218e6   581609.56           3.6134498e6          0.14111866    0.13864155   0.00147945
# sig_hail    0.1       2.7857797e9 8.1116555e6   726729.44           1.7177492e6          0.089590766   0.29729423   0.002911808

stats_for_run_hours([16])
# Mon-Sat
# Outlook hours: [16]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD          painting_ratio
# tornado     0.02      2.7857797e9 6.947651e7    3.838573e6          1.2464348e6          0.05524994    0.7548804    0.024939701
# tornado     0.05      2.7857797e9 2.1085884e7   2.6071845e6         2.4778232e6          0.12364597    0.51271987   0.007569114
# tornado     0.1       2.7857797e9 4.571014e6    1.1190004e6         3.9660075e6          0.24480352    0.22005874   0.0016408383
# tornado     0.15      2.7857797e9 826018.1      290966.78           4.794041e6           0.3522523     0.05722052   0.0002965124
# tornado     0.3       2.7857797e9 121275.625    44173.5             5.0408345e6          0.36424053    0.008687007  4.353382e-5
# tornado     0.45      2.7857797e9 27254.31      13652.455           5.0713555e6          0.5009283     0.0026848444 9.783369e-6
# tornado     0.6       2.7857797e9 0.0           0.0                 5.085008e6           NaN           0.0          0.0
# wind        0.05      2.7857797e9 1.9750443e8   2.7907066e7         8.1105185e6          0.14129843    0.7748178    0.07089736
# wind        0.15      2.7857797e9 6.2550756e7   1.6969212e7         1.9048372e7          0.27128708    0.4711369    0.022453591
# wind        0.3       2.7857797e9 1.1080914e7   5.119525e6          3.0898058e7          0.46201286    0.1421396    0.003977671
# wind        0.45      2.7857797e9 1.4445998e6   942435.7            3.5075148e7          0.6523853     0.026165988  0.0005185621
# wind        0.6       2.7857797e9 0.0           0.0                 3.6017584e7          NaN           0.0          0.0
# hail        0.05      2.7857797e9 1.42448e8     1.2344135e7         3.1499702e6          0.08665713    0.7966988    0.05113398
# hail        0.15      2.7857797e9 4.3557656e7   7.4152715e6         8.0788335e6          0.17024037    0.4785866    0.015635714
# hail        0.3       2.7857797e9 4.731469e6    1.5921961e6         1.3901909e7          0.33651203    0.10276142   0.0016984362
# hail        0.45      2.7857797e9 332244.94     173335.33           1.532077e7           0.52170944    0.011187179  0.00011926461
# hail        0.6       2.7857797e9 0.0           0.0                 1.5494105e7          NaN           0.0          0.0
# sig_tornado 0.1       2.7857797e9 2.8664658e6   291060.22           508377.53            0.10153975    0.36408117   0.0010289636
# sig_wind    0.1       2.7857797e9 5.0843165e6   755512.0            3.2651992e6          0.14859657    0.18790506   0.0018250964
# sig_hail    0.1       2.7857797e9 8.485609e6    777051.6            1.5902326e6          0.09157287    0.32824603   0.0030460448


stats_for_run_hours([6,13,16])
# Mon-Sat:
# Outlook hours: [6, 13, 16]
# event_name  threshold total_sq_mi painted_sq_mi true_positive_sq_mi false_negative_sq_mi success_ratio POD          painting_ratio
# tornado     0.02      8.357339e9  2.1184211e8   1.1476365e7         4.410241e6           0.054174144   0.7223925    0.025348036
# tornado     0.05      8.357339e9  6.131809e7    7.3877365e6         8.49887e6            0.12048217    0.46502924   0.007337035
# tornado     0.1       8.357339e9  1.2389585e7   2.9380062e6         1.29486e7            0.23713517    0.18493606   0.0014824796
# tornado     0.15      8.357339e9  2.1180475e6   679721.56           1.5206885e7          0.32091895    0.042785827  0.00025343563
# tornado     0.3       8.357339e9  312956.62     100259.27           1.5786347e7          0.32036155    0.0063109305 3.7446924e-5
# tornado     0.45      8.357339e9  27254.31      13652.455           1.5872954e7          0.5009283     0.0008593689 3.261123e-6
# tornado     0.6       8.357339e9  0.0           0.0                 1.5886606e7          NaN           0.0          0.0
# wind        0.05      8.357339e9  5.8833376e8   8.2821656e7         2.8337084e7          0.14077325    0.7450755    0.07039726
# wind        0.15      8.357339e9  1.8079784e8   4.7662196e7         6.349654e7           0.2636215     0.42877597   0.021633422
# wind        0.3       8.357339e9  3.0335922e7   1.3705978e7         9.745276e7           0.45180687    0.123300955  0.0036298542
# wind        0.45      8.357339e9  3.4790618e6   2.2432342e6         1.089155e8           0.6447814     0.020180458  0.0004162882
# wind        0.6       8.357339e9  0.0           0.0                 1.1115874e8          NaN           0.0          0.0
# hail        0.05      8.357339e9  4.3987312e8   3.69577e7           1.0837971e7          0.084019005   0.77324367   0.052633155
# hail        0.15      8.357339e9  1.2885355e8   2.110865e7          2.6687022e7          0.16381893    0.44164357   0.015418012
# hail        0.3       8.357339e9  1.299602e7    4.0729458e6         4.372273e7           0.31339946    0.085215785  0.0015550428
# hail        0.45      8.357339e9  747419.75     398549.56           4.7397124e7          0.53323394    0.008338612  8.9432746e-5
# hail        0.6       8.357339e9  0.0           0.0                 4.779567e7           NaN           0.0          0.0
# sig_tornado 0.1       8.357339e9  7.9628665e6   758966.94           1.735522e6           0.09531328    0.30425748   0.0009527993
# sig_wind    0.1       8.357339e9  1.2273978e7   1.6970398e6         1.0760237e7          0.13826323    0.13622878   0.0014686467
# sig_hail    0.1       8.357339e9  2.335312e7    2.0811668e6         5.1882395e6          0.08911729    0.28629115   0.002794325


# Copy from the above table to the below

target_warning_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.025348036),
    (0.05, 0.007337035),
    (0.1,  0.0014824796),
    (0.15, 0.00025343563),
    (0.3,  3.7446924e-5),
    (0.45, 3.261123e-6),
  ],
  "wind" => [
    (0.05, 0.07039726),
    (0.15, 0.021633422),
    (0.3,  0.0036298542),
    (0.45, 0.0004162882),
  ],
  "hail" => [
    (0.05, 0.052633155),
    (0.15, 0.015418012),
    (0.3,  0.0015550428),
    (0.45, 8.9432746e-5),
  ],
  "sig_tornado" => [(0.1, 0.0009527993)],
  "sig_wind"    => [(0.1, 0.0014686467)],
  "sig_hail"    => [(0.1, 0.002794325)],
)
