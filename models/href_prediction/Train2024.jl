# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.


import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction2024.forecasts_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2024, 2, 28, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# X, Ys, weights =
#     TrainingShared.get_data_labels_weights(
#       validation_forecasts;
#       event_name_to_labeler = TrainingShared.event_name_to_labeler,
#       save_dir = "validation_forecasts_with_blurs_and_forecast_hour_2024"
#     );

TrainingShared.prepare_data_labels_weights(
  validation_forecasts;
  event_name_to_labeler = TrainingShared.event_name_to_labeler,
  save_dir = "validation_forecasts_with_blurs_and_forecast_hour_2024"
)


blur_radii = [0; HREFPrediction2024.blur_radii]

for prediction_i in 1:length(HREFPrediction2024.models)
# for prediction_i in [3,7] # wind_adj and sig_wind_adj
  (event_name, _, _) = HREFPrediction2024.models[prediction_i]

  prediction_i_base = (prediction_i - 1) * length(blur_radii) # 0-indexed

  feature_names = readlines("validation_forecasts_with_blurs_and_forecast_hour_2024/features.txt")

  # oof it's just a little too big, so need to work in chunks
  # one hazard at a time, plus the forecast hour
  only_features = [prediction_i_base+1:prediction_i_base+length(blur_radii); length(HREFPrediction2024.models)*length(blur_radii)+1]

  println(event_name)
  println(only_features)
  println(feature_names[only_features])

  X, Ys, weights =
    TrainingShared.read_data_labels_weights_from_disk("validation_forecasts_with_blurs_and_forecast_hour_2024"; only_features = only_features)

  forecast_hour_j = size(X, 2)

  function test_predictive_power(forecasts, X, Ys, weights, only_features)
    for prediction_i in 1:length(HREFPrediction2024.models)
      (event_name, _, _) = HREFPrediction2024.models[prediction_i]
      y = Ys[event_name]
      for j in 1:(size(X,2)-1)
        x = @view X[:,j]
        au_pr_curve = Float32(Metrics.area_under_pr_curve(x, y, weights))
        println("$event_name ($(round(sum(y)))) feature $(prediction_i_base+j)=$(only_features[j])\tAU-PR-curve: $au_pr_curve")
      end
    end
  end
  test_predictive_power(validation_forecasts, X, Ys, weights, only_features)
end

# tornado
# ["TORPROB:calculated:hour fcst:calculated_prob:", "TORPROB:calculated:hour fcst:calculated_prob:15mi mean", "TORPROB:calculated:hour fcst:calculated_prob:25mi mean", "TORPROB:calculated:hour fcst:calculated_prob:35mi mean", "TORPROB:calculated:hour fcst:calculated_prob:50mi mean", "TORPROB:calculated:hour fcst:calculated_prob:70mi mean", "TORPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04458257
# tornado (101642.0)         feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.044846594
# tornado (101642.0)         feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.044660322
# tornado (101642.0)         feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.043800037
# tornado (101642.0)         feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04156998
# tornado (101642.0)         feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.037759554
# tornado (101642.0)         feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03102177
# wind (874384.0)            feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.051420897
# wind (874384.0)            feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.051494535
# wind (874384.0)            feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.051235344
# wind (874384.0)            feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.050470702
# wind (874384.0)            feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04875791
# wind (874384.0)            feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.04557444
# wind (874384.0)            feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03978436
# wind_adj (278174.0)        feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012910408
# wind_adj (278174.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012873274
# wind_adj (278174.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012752202
# wind_adj (278174.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012476351
# wind_adj (278174.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011908886
# wind_adj (278174.0)        feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.010995837
# wind_adj (278174.0)        feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.009452826
# hail (405123.0)            feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02218699
# hail (405123.0)            feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022128839
# hail (405123.0)            feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021926094
# hail (405123.0)            feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.021447338
# hail (405123.0)            feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020441411
# hail (405123.0)            feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.018837493
# hail (405123.0)            feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016192595
# sig_tornado (13792.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027445951
# sig_tornado (13792.0)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.028389782
# sig_tornado (13792.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.028972166
# sig_tornado (13792.0)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.029353565
# sig_tornado (13792.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.028786186
# sig_tornado (13792.0)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.027141767
# sig_tornado (13792.0)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021901364
# sig_wind (84250.0)         feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0076630577
# sig_wind (84250.0)         feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.007648342
# sig_wind (84250.0)         feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00757643
# sig_wind (84250.0)         feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0074103763
# sig_wind (84250.0)         feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007064466
# sig_wind (84250.0)         feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0064972215
# sig_wind (84250.0)         feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005527759
# sig_wind_adj (31404.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0016301085
# sig_wind_adj (31404.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0016249514
# sig_wind_adj (31404.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0016100901
# sig_wind_adj (31404.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0015785949
# sig_wind_adj (31404.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0015126985
# sig_wind_adj (31404.0)     feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.001404068
# sig_wind_adj (31404.0)     feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0012119444
# sig_hail (51908.0)         feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0041780425
# sig_hail (51908.0)         feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0041632997
# sig_hail (51908.0)         feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004118933
# sig_hail (51908.0)         feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004012486
# sig_hail (51908.0)         feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0037959367
# sig_hail (51908.0)         feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0034512295
# sig_hail (51908.0)         feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0028937378
# tornado_life_risk (3093.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0064463015
# tornado_life_risk (3093.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0066459044
# tornado_life_risk (3093.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0067401244
# tornado_life_risk (3093.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0067231283
# tornado_life_risk (3093.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0062892623
# tornado_life_risk (3093.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0054824594
# tornado_life_risk (3093.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0039054176

# wind
# ["WINDPROB:calculated:hour fcst:calculated_prob:", "WINDPROB:calculated:hour fcst:calculated_prob:15mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:25mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:35mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:50mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:70mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010589085
# tornado (101642.0)         feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010687708
# tornado (101642.0)         feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010656607
# tornado (101642.0)         feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.010505419
# tornado (101642.0)         feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.010120345
# tornado (101642.0)         feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.009323221
# tornado (101642.0)         feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007895621
# wind (874384.0)            feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.12782907
# wind (874384.0)            feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.12906836
# wind (874384.0)            feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.12925549
# wind (874384.0)            feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.12847693
# wind (874384.0)            feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.12560302
# wind (874384.0)            feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.11934432
# wind (874384.0)            feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.10668006
# wind_adj (278174.0)        feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03774708
# wind_adj (278174.0)        feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.03803853
# wind_adj (278174.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.037904996
# wind_adj (278174.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03756368
# wind_adj (278174.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.036224145
# wind_adj (278174.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03371297
# wind_adj (278174.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.028613193
# hail (405123.0)            feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.022103406
# hail (405123.0)            feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022114059
# hail (405123.0)            feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021948108
# hail (405123.0)            feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.02155953
# hail (405123.0)            feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020670062
# hail (405123.0)            feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.019258074
# hail (405123.0)            feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016872102
# sig_tornado (13792.0)      feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004057158
# sig_tornado (13792.0)      feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004160798
# sig_tornado (13792.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0041581015
# sig_tornado (13792.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0040954268
# sig_tornado (13792.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004004421
# sig_tornado (13792.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0038498708
# sig_tornado (13792.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0034231232
# sig_wind (84250.0)         feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015860096
# sig_wind (84250.0)         feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016017107
# sig_wind (84250.0)         feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.015977671
# sig_wind (84250.0)         feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015863271
# sig_wind (84250.0)         feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015382411
# sig_wind (84250.0)         feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.014461649
# sig_wind (84250.0)         feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012575701
# sig_wind_adj (31404.0)     feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005745996
# sig_wind_adj (31404.0)     feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0058170967
# sig_wind_adj (31404.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0058026724
# sig_wind_adj (31404.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0058195647
# sig_wind_adj (31404.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005686624
# sig_wind_adj (31404.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0053757997
# sig_wind_adj (31404.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0045962296
# sig_hail (51908.0)         feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0030991847
# sig_hail (51908.0)         feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0030940569
# sig_hail (51908.0)         feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0030659959
# sig_hail (51908.0)         feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0030002496
# sig_hail (51908.0)         feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0028635242
# sig_hail (51908.0)         feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026494649
# sig_hail (51908.0)         feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0022984515
# tornado_life_risk (3093.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005027309
# tornado_life_risk (3093.0) feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005034586
# tornado_life_risk (3093.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00049928884
# tornado_life_risk (3093.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.00048760482
# tornado_life_risk (3093.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0004645387
# tornado_life_risk (3093.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00042451956
# tornado_life_risk (3093.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00036548005


# wind_adj
# [15, 16, 17, 18, 19, 20, 21, 64]
# ["WINDPROB:calculated:hour fcst:calculated_prob:", "WINDPROB:calculated:hour fcst:calculated_prob:15mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:25mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:35mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:50mi mea
# n", "WINDPROB:calculated:hour fcst:calculated_prob:70mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 15=15 AU-PR-curve: 0.006824035
# tornado (101642.0)         feature 16=16 AU-PR-curve: 0.006834726
# tornado (101642.0)         feature 17=17 AU-PR-curve: 0.00681208
# tornado (101642.0)         feature 18=18 AU-PR-curve: 0.006562505
# tornado (101642.0)         feature 19=19 AU-PR-curve: 0.0060667386
# tornado (101642.0)         feature 20=20 AU-PR-curve: 0.0054548425
# tornado (101642.0)         feature 21=21 AU-PR-curve: 0.004511799
# wind (874384.0)            feature 15=15 AU-PR-curve: 0.07665216
# wind (874384.0)            feature 16=16 AU-PR-curve: 0.07707598
# wind (874384.0)            feature 17=17 AU-PR-curve: 0.07689357
# wind (874384.0)            feature 18=18 AU-PR-curve: 0.07593012
# wind (874384.0)            feature 19=19 AU-PR-curve: 0.073478006
# wind (874384.0)            feature 20=20 AU-PR-curve: 0.06911435
# wind (874384.0)            feature 21=21 AU-PR-curve: 0.060987126
# wind_adj (278174.0)        feature 15=15 AU-PR-curve: 0.081498474
# wind_adj (278174.0)        feature 16=16 AU-PR-curve: 0.08251173
# wind_adj (278174.0)        feature 17=17 AU-PR-curve: 0.08271261
# wind_adj (278174.0)        feature 18=18 AU-PR-curve: 0.0821972
# wind_adj (278174.0)        feature 19=19 AU-PR-curve: 0.08009963
# wind_adj (278174.0)        feature 20=20 AU-PR-curve: 0.075880855
# wind_adj (278174.0)        feature 21=21 AU-PR-curve: 0.0671901
# hail (405123.0)            feature 15=15 AU-PR-curve: 0.025571592
# hail (405123.0)            feature 16=16 AU-PR-curve: 0.025581101
# hail (405123.0)            feature 17=17 AU-PR-curve: 0.02537839
# hail (405123.0)            feature 18=18 AU-PR-curve: 0.024867807
# hail (405123.0)            feature 19=19 AU-PR-curve: 0.023716865
# hail (405123.0)            feature 20=20 AU-PR-curve: 0.021948123
# hail (405123.0)            feature 21=21 AU-PR-curve: 0.018987754
# sig_tornado (13792.0)      feature 15=15 AU-PR-curve: 0.0021061425
# sig_tornado (13792.0)      feature 16=16 AU-PR-curve: 0.00203055
# sig_tornado (13792.0)      feature 17=17 AU-PR-curve: 0.0019877092
# sig_tornado (13792.0)      feature 18=18 AU-PR-curve: 0.0018446221
# sig_tornado (13792.0)      feature 19=19 AU-PR-curve: 0.0016068516
# sig_tornado (13792.0)      feature 20=20 AU-PR-curve: 0.0013555104
# sig_tornado (13792.0)      feature 21=21 AU-PR-curve: 0.0010289017
# sig_wind (84250.0)         feature 15=15 AU-PR-curve: 0.022231944
# sig_wind (84250.0)         feature 16=16 AU-PR-curve: 0.022602905
# sig_wind (84250.0)         feature 17=17 AU-PR-curve: 0.022733694
# sig_wind (84250.0)         feature 18=18 AU-PR-curve: 0.022699956
# sig_wind (84250.0)         feature 19=19 AU-PR-curve: 0.022352422
# sig_wind (84250.0)         feature 20=20 AU-PR-curve: 0.021490503
# sig_wind (84250.0)         feature 21=21 AU-PR-curve: 0.019334396
# sig_wind_adj (31404.0)     feature 15=15 AU-PR-curve: 0.016002113
# sig_wind_adj (31404.0)     feature 16=16 AU-PR-curve: 0.016343279
# sig_wind_adj (31404.0)     feature 17=17 AU-PR-curve: 0.01651068
# sig_wind_adj (31404.0)     feature 18=18 AU-PR-curve: 0.016673988
# sig_wind_adj (31404.0)     feature 19=19 AU-PR-curve: 0.016787147
# sig_wind_adj (31404.0)     feature 20=20 AU-PR-curve: 0.016610371
# sig_wind_adj (31404.0)     feature 21=21 AU-PR-curve: 0.015832776
# sig_hail (51908.0)         feature 15=15 AU-PR-curve: 0.005165366
# sig_hail (51908.0)         feature 16=16 AU-PR-curve: 0.0052758576
# sig_hail (51908.0)         feature 17=17 AU-PR-curve: 0.0053003957
# sig_hail (51908.0)         feature 18=18 AU-PR-curve: 0.005183506
# sig_hail (51908.0)         feature 19=19 AU-PR-curve: 0.0049264114
# sig_hail (51908.0)         feature 20=20 AU-PR-curve: 0.004508551
# sig_hail (51908.0)         feature 21=21 AU-PR-curve: 0.0036438094
# tornado_life_risk (3093.0) feature 15=15 AU-PR-curve: 0.00023260407
# tornado_life_risk (3093.0) feature 16=16 AU-PR-curve: 0.00023166538
# tornado_life_risk (3093.0) feature 17=17 AU-PR-curve: 0.0002287612
# tornado_life_risk (3093.0) feature 18=18 AU-PR-curve: 0.00022060363
# tornado_life_risk (3093.0) feature 19=19 AU-PR-curve: 0.00020560045
# tornado_life_risk (3093.0) feature 20=20 AU-PR-curve: 0.00018692142
# tornado_life_risk (3093.0) feature 21=21 AU-PR-curve: 0.0001613835

# hail
# ["HAILPROB:calculated:hour fcst:calculated_prob:", "HAILPROB:calculated:hour fcst:calculated_prob:15mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:25mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:35mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:50mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:70mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.007297086
# tornado (101642.0)         feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0073290663
# tornado (101642.0)         feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.007298768
# tornado (101642.0)         feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0071581146
# tornado (101642.0)         feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0067913895
# tornado (101642.0)         feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.006192091
# tornado (101642.0)         feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0051869852
# wind (874384.0)            feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.037065726
# wind (874384.0)            feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.037033882
# wind (874384.0)            feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.03675523
# wind (874384.0)            feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03605872
# wind (874384.0)            feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.034615297
# wind (874384.0)            feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.032338165
# wind (874384.0)            feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.028555334
# wind_adj (278174.0)        feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.020148933
# wind_adj (278174.0)        feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.020234473
# wind_adj (278174.0)        feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.020149669
# wind_adj (278174.0)        feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.019857861
# wind_adj (278174.0)        feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.019138655
# wind_adj (278174.0)        feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.017918186
# wind_adj (278174.0)        feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015787683
# hail (405123.0)            feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.09149645
# hail (405123.0)            feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.092647225
# hail (405123.0)            feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.09272799
# hail (405123.0)            feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.091651455
# hail (405123.0)            feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.08822686
# hail (405123.0)            feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0815599
# hail (405123.0)            feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.06952726
# sig_tornado (13792.0)      feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0025716498
# sig_tornado (13792.0)      feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0026195375
# sig_tornado (13792.0)      feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0026439372
# sig_tornado (13792.0)      feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0026329819
# sig_tornado (13792.0)      feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0025185533
# sig_tornado (13792.0)      feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022743498
# sig_tornado (13792.0)      feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0017834442
# sig_wind (84250.0)         feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.006867611
# sig_wind (84250.0)         feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.006885819
# sig_wind (84250.0)         feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0068374043
# sig_wind (84250.0)         feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.006700213
# sig_wind (84250.0)         feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00639286
# sig_wind (84250.0)         feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005876067
# sig_wind (84250.0)         feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0050469614
# sig_wind_adj (31404.0)     feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032326577
# sig_wind_adj (31404.0)     feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003250134
# sig_wind_adj (31404.0)     feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0032337126
# sig_wind_adj (31404.0)     feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.003179589
# sig_wind_adj (31404.0)     feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0030445037
# sig_wind_adj (31404.0)     feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0027965766
# sig_wind_adj (31404.0)     feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0023894587
# sig_hail (51908.0)         feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.022759821
# sig_hail (51908.0)         feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.023222687
# sig_hail (51908.0)         feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.023420911
# sig_hail (51908.0)         feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.023353716
# sig_hail (51908.0)         feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.022783604
# sig_hail (51908.0)         feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0215332
# sig_hail (51908.0)         feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.019015664
# tornado_life_risk (3093.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0006553588
# tornado_life_risk (3093.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0006675654
# tornado_life_risk (3093.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0006722062
# tornado_life_risk (3093.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.000663792
# tornado_life_risk (3093.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0006222865
# tornado_life_risk (3093.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0005455593
# tornado_life_risk (3093.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00041630407

# sig_tornado
# ["STORPROB:calculated:hour fcst:calculated_prob:", "STORPROB:calculated:hour fcst:calculated_prob:15mi mean", "STORPROB:calculated:hour fcst:calculated_prob:25mi mean", "STORPROB:calculated:hour fcst:calculated_prob:35mi mean", "STORPROB:calculated:hour fcst:calculated_prob:50mi mean", "STORPROB:calculated:hour fcst:calculated_prob:70mi m
# ean", "STORPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.032265045
# tornado (101642.0)         feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.032318044
# tornado (101642.0)         feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.032005627
# tornado (101642.0)         feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03114941
# tornado (101642.0)         feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.029329598
# tornado (101642.0)         feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02635234
# tornado (101642.0)         feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021628633
# wind (874384.0)            feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.046982918
# wind (874384.0)            feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.04694493
# wind (874384.0)            feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.046603665
# wind (874384.0)            feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.045765713
# wind (874384.0)            feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.044001147
# wind (874384.0)            feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.04083301
# wind (874384.0)            feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03541002
# wind_adj (278174.0)        feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.011943924
# wind_adj (278174.0)        feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.011905571
# wind_adj (278174.0)        feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.011794113
# wind_adj (278174.0)        feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01155057
# wind_adj (278174.0)        feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011068058
# wind_adj (278174.0)        feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.010264817
# wind_adj (278174.0)        feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00889574
# hail (405123.0)            feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.019043827
# hail (405123.0)            feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.018979628
# hail (405123.0)            feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0188036
# hail (405123.0)            feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.018415714
# hail (405123.0)            feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.017646324
# hail (405123.0)            feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01637637
# hail (405123.0)            feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014250935
# sig_tornado (13792.0)      feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.031367242
# sig_tornado (13792.0)      feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.031650614
# sig_tornado (13792.0)      feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.031269126
# sig_tornado (13792.0)      feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.030035706
# sig_tornado (13792.0)      feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.027383663
# sig_tornado (13792.0)      feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.023388445
# sig_tornado (13792.0)      feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.017623857
# sig_wind (84250.0)         feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.006835445
# sig_wind (84250.0)         feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0068151625
# sig_wind (84250.0)         feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.006752512
# sig_wind (84250.0)         feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0066145915
# sig_wind (84250.0)         feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.006334261
# sig_wind (84250.0)         feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005851701
# sig_wind (84250.0)         feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0050128847
# sig_wind_adj (31404.0)     feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017000715
# sig_wind_adj (31404.0)     feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0016952861
# sig_wind_adj (31404.0)     feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0016812349
# sig_wind_adj (31404.0)     feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.001651115
# sig_wind_adj (31404.0)     feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0015912488
# sig_wind_adj (31404.0)     feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0014923983
# sig_wind_adj (31404.0)     feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0013108619
# sig_hail (51908.0)         feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0035185488
# sig_hail (51908.0)         feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003505364
# sig_hail (51908.0)         feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0034689524
# sig_hail (51908.0)         feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0033883622
# sig_hail (51908.0)         feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0032329846
# sig_hail (51908.0)         feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0029784683
# sig_hail (51908.0)         feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0025565021
# tornado_life_risk (3093.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010123185
# tornado_life_risk (3093.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010063962
# tornado_life_risk (3093.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00975427
# tornado_life_risk (3093.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009048249
# tornado_life_risk (3093.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0077871797
# tornado_life_risk (3093.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0061529996
# tornado_life_risk (3093.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0042618965

# sig_wind
# ["SWINDPRO:calculated:hour fcst:calculated_prob:", "SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010221196
# tornado (101642.0)         feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010250007
# tornado (101642.0)         feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010174918
# tornado (101642.0)         feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009935485
# tornado (101642.0)         feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0094199
# tornado (101642.0)         feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008609302
# tornado (101642.0)         feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007291271
# wind (874384.0)            feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.08659278
# wind (874384.0)            feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.08715066
# wind (874384.0)            feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.087004766
# wind (874384.0)            feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0860994
# wind (874384.0)            feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0836549
# wind (874384.0)            feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.07904909
# wind (874384.0)            feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.070160866
# wind_adj (278174.0)        feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0546422
# wind_adj (278174.0)        feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.05526082
# wind_adj (278174.0)        feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05528199
# wind_adj (278174.0)        feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.054841246
# wind_adj (278174.0)        feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.053170532
# wind_adj (278174.0)        feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.050040647
# wind_adj (278174.0)        feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04364945
# hail (405123.0)            feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027099736
# hail (405123.0)            feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02710439
# hail (405123.0)            feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.02690581
# hail (405123.0)            feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.026428387
# hail (405123.0)            feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.025340807
# hail (405123.0)            feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.023568414
# hail (405123.0)            feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02054999
# sig_tornado (13792.0)      feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027873488
# sig_tornado (13792.0)      feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0027932958
# sig_tornado (13792.0)      feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0027619095
# sig_tornado (13792.0)      feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0026840423
# sig_tornado (13792.0)      feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0025364668
# sig_tornado (13792.0)      feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022947108
# sig_tornado (13792.0)      feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018972879
# sig_wind (84250.0)         feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.025028422
# sig_wind (84250.0)         feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02541912
# sig_wind (84250.0)         feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.025494266
# sig_wind (84250.0)         feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.025370913
# sig_wind (84250.0)         feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.024844557
# sig_wind (84250.0)         feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02380682
# sig_wind (84250.0)         feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021160657
# sig_wind_adj (31404.0)     feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012106531
# sig_wind_adj (31404.0)     feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012422113
# sig_wind_adj (31404.0)     feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012584977
# sig_wind_adj (31404.0)     feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012735926
# sig_wind_adj (31404.0)     feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.012779744
# sig_wind_adj (31404.0)     feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012806546
# sig_wind_adj (31404.0)     feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011711587
# sig_hail (51908.0)         feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0055913213
# sig_hail (51908.0)         feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0056044515
# sig_hail (51908.0)         feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0055645113
# sig_hail (51908.0)         feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.005452066
# sig_hail (51908.0)         feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005188876
# sig_hail (51908.0)         feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0047719576
# sig_hail (51908.0)         feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004059665
# tornado_life_risk (3093.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005923344
# tornado_life_risk (3093.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005936034
# tornado_life_risk (3093.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00058817817
# tornado_life_risk (3093.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0005728559
# tornado_life_risk (3093.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00054026407
# tornado_life_risk (3093.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00048483734
# tornado_life_risk (3093.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00039651856


# sig_wind_adj
# [43, 44, 45, 46, 47, 48, 49, 64]
# ["SWINDPRO:calculated:hour fcst:calculated_prob:", "SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:50mi mea
# n", "SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 43=43 AU-PR-curve: 0.004890062
# tornado (101642.0)         feature 44=44 AU-PR-curve: 0.004905254
# tornado (101642.0)         feature 45=45 AU-PR-curve: 0.0048695593
# tornado (101642.0)         feature 46=46 AU-PR-curve: 0.0047656
# tornado (101642.0)         feature 47=47 AU-PR-curve: 0.0045294412
# tornado (101642.0)         feature 48=48 AU-PR-curve: 0.004088605
# tornado (101642.0)         feature 49=49 AU-PR-curve: 0.0034726295
# wind (874384.0)            feature 43=43 AU-PR-curve: 0.059483252
# wind (874384.0)            feature 44=44 AU-PR-curve: 0.059689656
# wind (874384.0)            feature 45=45 AU-PR-curve: 0.0595112
# wind (874384.0)            feature 46=46 AU-PR-curve: 0.05882284
# wind (874384.0)            feature 47=47 AU-PR-curve: 0.057086162
# wind (874384.0)            feature 48=48 AU-PR-curve: 0.05390894
# wind (874384.0)            feature 49=49 AU-PR-curve: 0.047764417
# wind_adj (278174.0)        feature 43=43 AU-PR-curve: 0.07005078
# wind_adj (278174.0)        feature 44=44 AU-PR-curve: 0.07071517
# wind_adj (278174.0)        feature 45=45 AU-PR-curve: 0.07081734
# wind_adj (278174.0)        feature 46=46 AU-PR-curve: 0.070357695
# wind_adj (278174.0)        feature 47=47 AU-PR-curve: 0.06864917
# wind_adj (278174.0)        feature 48=48 AU-PR-curve: 0.06514225
# wind_adj (278174.0)        feature 49=49 AU-PR-curve: 0.0577076
# hail (405123.0)            feature 43=43 AU-PR-curve: 0.024391165
# hail (405123.0)            feature 44=44 AU-PR-curve: 0.02436891
# hail (405123.0)            feature 45=45 AU-PR-curve: 0.024188675
# hail (405123.0)            feature 46=46 AU-PR-curve: 0.023781592
# hail (405123.0)            feature 47=47 AU-PR-curve: 0.022851916
# hail (405123.0)            feature 48=48 AU-PR-curve: 0.021297984
# hail (405123.0)            feature 49=49 AU-PR-curve: 0.018580697
# sig_tornado (13792.0)      feature 43=43 AU-PR-curve: 0.00076751853
# sig_tornado (13792.0)      feature 44=44 AU-PR-curve: 0.0007639717
# sig_tornado (13792.0)      feature 45=45 AU-PR-curve: 0.00075525243
# sig_tornado (13792.0)      feature 46=46 AU-PR-curve: 0.0007367691
# sig_tornado (13792.0)      feature 47=47 AU-PR-curve: 0.00069981365
# sig_tornado (13792.0)      feature 48=48 AU-PR-curve: 0.0006408032
# sig_tornado (13792.0)      feature 49=49 AU-PR-curve: 0.00054220064
# sig_wind (84250.0)         feature 43=43 AU-PR-curve: 0.022323525
# sig_wind (84250.0)         feature 44=44 AU-PR-curve: 0.022616401
# sig_wind (84250.0)         feature 45=45 AU-PR-curve: 0.02270699
# sig_wind (84250.0)         feature 46=46 AU-PR-curve: 0.022625752
# sig_wind (84250.0)         feature 47=47 AU-PR-curve: 0.022147596
# sig_wind (84250.0)         feature 48=48 AU-PR-curve: 0.02096183
# sig_wind (84250.0)         feature 49=49 AU-PR-curve: 0.018651694
# sig_wind_adj (31404.0)     feature 43=43 AU-PR-curve: 0.017062819
# sig_wind_adj (31404.0)     feature 44=44 AU-PR-curve: 0.017356653
# sig_wind_adj (31404.0)     feature 45=45 AU-PR-curve: 0.017525448
# sig_wind_adj (31404.0)     feature 46=46 AU-PR-curve: 0.017610494
# sig_wind_adj (31404.0)     feature 47=47 AU-PR-curve: 0.017511101
# sig_wind_adj (31404.0)     feature 48=48 AU-PR-curve: 0.016932815
# sig_wind_adj (31404.0)     feature 49=49 AU-PR-curve: 0.0155548835
# sig_hail (51908.0)         feature 43=43 AU-PR-curve: 0.0065138973
# sig_hail (51908.0)         feature 44=44 AU-PR-curve: 0.0065952595
# sig_hail (51908.0)         feature 45=45 AU-PR-curve: 0.006569653
# sig_hail (51908.0)         feature 46=46 AU-PR-curve: 0.0064590024
# sig_hail (51908.0)         feature 47=47 AU-PR-curve: 0.0061560967
# sig_hail (51908.0)         feature 48=48 AU-PR-curve: 0.0055058855
# sig_hail (51908.0)         feature 49=49 AU-PR-curve: 0.004440739
# tornado_life_risk (3093.0) feature 43=43 AU-PR-curve: 0.00017459104
# tornado_life_risk (3093.0) feature 44=44 AU-PR-curve: 0.00017375844
# tornado_life_risk (3093.0) feature 45=45 AU-PR-curve: 0.0001715792
# tornado_life_risk (3093.0) feature 46=46 AU-PR-curve: 0.00016699058
# tornado_life_risk (3093.0) feature 47=47 AU-PR-curve: 0.0001578545
# tornado_life_risk (3093.0) feature 48=48 AU-PR-curve: 0.00014294375
# tornado_life_risk (3093.0) feature 49=49 AU-PR-curve: 0.00012005462

# sig_hail
# ["SHAILPRO:calculated:hour fcst:calculated_prob:", "SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.00554653
# tornado (101642.0)         feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.005542353
# tornado (101642.0)         feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0054915217
# tornado (101642.0)         feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0053568347
# tornado (101642.0)         feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0050675613
# tornado (101642.0)         feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00462997
# tornado (101642.0)         feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003946096
# wind (874384.0)            feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.025069771
# wind (874384.0)            feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0250241
# wind (874384.0)            feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.024806505
# wind (874384.0)            feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.024320452
# wind (874384.0)            feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.02336532
# wind (874384.0)            feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.021927029
# wind (874384.0)            feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.019628493
# wind_adj (278174.0)        feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01672821
# wind_adj (278174.0)        feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016845934
# wind_adj (278174.0)        feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.01680309
# wind_adj (278174.0)        feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01662577
# wind_adj (278174.0)        feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.016172696
# wind_adj (278174.0)        feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01544345
# wind_adj (278174.0)        feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014180284
# hail (405123.0)            feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.068348184
# hail (405123.0)            feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.06880169
# hail (405123.0)            feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.068399005
# hail (405123.0)            feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.06695164
# hail (405123.0)            feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0635662
# hail (405123.0)            feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.05795798
# hail (405123.0)            feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.048725642
# sig_tornado (13792.0)      feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0011347303
# sig_tornado (13792.0)      feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0011295904
# sig_tornado (13792.0)      feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0011159689
# sig_tornado (13792.0)      feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0010858984
# sig_tornado (13792.0)      feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0010286389
# sig_tornado (13792.0)      feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00094459153
# sig_tornado (13792.0)      feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00080697367
# sig_wind (84250.0)         feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0057923705
# sig_wind (84250.0)         feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0058095013
# sig_wind (84250.0)         feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0057672244
# sig_wind (84250.0)         feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.005654097
# sig_wind (84250.0)         feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005408866
# sig_wind (84250.0)         feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005008822
# sig_wind (84250.0)         feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0043693944
# sig_wind_adj (31404.0)     feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0030041132
# sig_wind_adj (31404.0)     feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003023236
# sig_wind_adj (31404.0)     feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0030065603
# sig_wind_adj (31404.0)     feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.002954833
# sig_wind_adj (31404.0)     feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0028288553
# sig_wind_adj (31404.0)     feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026104767
# sig_wind_adj (31404.0)     feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0022669483
# sig_hail (51908.0)         feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.023658833
# sig_hail (51908.0)         feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.023957629
# sig_hail (51908.0)         feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.023886377
# sig_hail (51908.0)         feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.023357052
# sig_hail (51908.0)         feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.02211366
# sig_hail (51908.0)         feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.020044014
# sig_hail (51908.0)         feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016710479
# tornado_life_risk (3093.0) feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0002561355
# tornado_life_risk (3093.0) feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0002560921
# tornado_life_risk (3093.0) feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00025400426
# tornado_life_risk (3093.0) feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0002485532
# tornado_life_risk (3093.0) feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00023604307
# tornado_life_risk (3093.0) feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00021626402
# tornado_life_risk (3093.0) feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00018335154

# tornado_life_risk
# ["TORPROB:calculated:hour fcst:calculated_prob:", "TORPROB:calculated:hour fcst:calculated_prob:15mi mean", "TORPROB:calculated:hour fcst:calculated_prob:25mi mean", "TORPROB:calculated:hour fcst:calculated_prob:35mi mean", "TORPROB:calculated:hour fcst:calculated_prob:50mi mean", "TORPROB:calculated:hour fcst:calculated_prob:70mi mean",
# "TORPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# tornado (101642.0)         feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04458257
# tornado (101642.0)         feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.044846594
# tornado (101642.0)         feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.044660322
# tornado (101642.0)         feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.043800037
# tornado (101642.0)         feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04156998
# tornado (101642.0)         feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.037759554
# tornado (101642.0)         feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03102177
# wind (874384.0)            feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.051420897
# wind (874384.0)            feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.051494535
# wind (874384.0)            feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.051235344
# wind (874384.0)            feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.050470702
# wind (874384.0)            feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04875791
# wind (874384.0)            feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.04557444
# wind (874384.0)            feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03978436
# wind_adj (278174.0)        feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012910408
# wind_adj (278174.0)        feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012873274
# wind_adj (278174.0)        feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012752202
# wind_adj (278174.0)        feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012476351
# wind_adj (278174.0)        feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011908886
# wind_adj (278174.0)        feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.010995837
# wind_adj (278174.0)        feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.009452826
# hail (405123.0)            feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02218699
# hail (405123.0)            feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022128839
# hail (405123.0)            feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021926094
# hail (405123.0)            feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.021447338
# hail (405123.0)            feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020441411
# hail (405123.0)            feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.018837493
# hail (405123.0)            feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016192595
# sig_tornado (13792.0)      feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027445951
# sig_tornado (13792.0)      feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.028389782
# sig_tornado (13792.0)      feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.028972166
# sig_tornado (13792.0)      feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.029353565
# sig_tornado (13792.0)      feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.028786186
# sig_tornado (13792.0)      feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.027141767
# sig_tornado (13792.0)      feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021901364
# sig_wind (84250.0)         feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0076630577
# sig_wind (84250.0)         feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.007648342
# sig_wind (84250.0)         feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00757643
# sig_wind (84250.0)         feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0074103763
# sig_wind (84250.0)         feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007064466
# sig_wind (84250.0)         feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0064972215
# sig_wind (84250.0)         feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005527759
# sig_wind_adj (31404.0)     feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0016301085
# sig_wind_adj (31404.0)     feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0016249514
# sig_wind_adj (31404.0)     feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0016100901
# sig_wind_adj (31404.0)     feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0015785949
# sig_wind_adj (31404.0)     feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0015126985
# sig_wind_adj (31404.0)     feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.001404068
# sig_wind_adj (31404.0)     feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0012119444
# sig_hail (51908.0)         feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0041780425
# sig_hail (51908.0)         feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0041632997
# sig_hail (51908.0)         feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004118933
# sig_hail (51908.0)         feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004012486
# sig_hail (51908.0)         feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0037959367
# sig_hail (51908.0)         feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0034512295
# sig_hail (51908.0)         feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0028937378
# tornado_life_risk (3093.0) feature 57 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0064463015
# tornado_life_risk (3093.0) feature 58 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0066459044
# tornado_life_risk (3093.0) feature 59 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0067401244
# tornado_life_risk (3093.0) feature 60 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0067231283
# tornado_life_risk (3093.0) feature 61 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0062892623
# tornado_life_risk (3093.0) feature 62 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0054824594
# tornado_life_risk (3093.0) feature 63 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0039054176



# collate by predicted hazard (rather than by predicting model)
# $ pbpaste > scratch.txt
# $ grep ' tornado (' scratch.txt | pbcopy

# find max after copying column text
# $ irb
# irb(main):001> `pbpaste`.split.map(&:to_f).max



println("Determining best blur radii to maximize area under precision-recall curve")

bests = []
for prediction_i in 1:length(HREFPrediction2024.models)
# for prediction_i in [3,7] # wind_adj and sig_wind_adj
  (event_name, _, _) = HREFPrediction2024.models[prediction_i]

  prediction_i_base = (prediction_i - 1) * length(blur_radii) # 0-indexed

  feature_names = readlines("validation_forecasts_with_blurs_and_forecast_hour_2024/features.txt")

  # oof it's just a little too big, so need to work in chunks
  # one hazard at a time, plus the forecast hour
  only_features = [prediction_i_base+1:prediction_i_base+length(blur_radii); length(HREFPrediction2024.models)*length(blur_radii)+1]

  println(event_name)
  println(only_features)
  println(feature_names[only_features])

  X, Ys, weights =
    TrainingShared.read_data_labels_weights_from_disk("validation_forecasts_with_blurs_and_forecast_hour_2024"; only_features = only_features)

  forecast_hour_j = size(X, 2)

  y = Ys[event_name]

  println("blur_radius_f1\tblur_radius_f36\tAU_PR_$event_name")

  best_blur_i_lo, best_blur_i_hi, best_au_pr = (0, 0, 0.0)

  for blur_i_lo in 1:length(blur_radii)
    for blur_i_hi in 1:length(blur_radii)
      X_blurred = zeros(Float32, length(y))

      Threads.@threads :static for i in 1:length(y)
        forecast_ratio = (X[i,forecast_hour_j] - 1f0) * (1f0/(36f0-1f0))
        X_blurred[i] = X[i,blur_i_lo] * (1f0 - forecast_ratio) + X[i,blur_i_hi] * forecast_ratio
      end

      au_pr_curve = Metrics.area_under_pr_curve(X_blurred, y, weights)

      if au_pr_curve > best_au_pr
        best_blur_i_lo, best_blur_i_hi, best_au_pr = (blur_i_lo, blur_i_hi, au_pr_curve)
      end

      println("$(blur_radii[blur_i_lo])\t$(blur_radii[blur_i_hi])\t$(Float32(au_pr_curve))")
    end
  end
  println("Best $event_name: $(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
  push!(bests, (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr))
  println()
end

# tornado
# ["TORPROB:calculated:hour fcst:calculated_prob:", "TORPROB:calculated:hour fcst:calculated_prob:15mi mean", "TORPROB:calculated:hour fcst:calculated_prob:25mi mean", "TORPROB:calculated:hour fcst:calculated_prob:
# 35mi mean", "TORPROB:calculated:hour fcst:calculated_prob:50mi mean", "TORPROB:calculated:hour fcst:calculated_prob:70mi mean", "TORPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated
# :hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_tornado
# 0       0       0.044582576
# 0       15      0.04480133
# 0       25      0.044821482
# 0       35      0.044658918
# 0       50      0.044125736
# 0       70      0.043133624
# 0       100     0.04146615
# 15      0       0.044728763
# 15      15      0.044846594
# 15      25      0.044844333
# 15      35      0.044661894
# 15      50      0.04410935
# 15      70      0.043102257
# 15      100     0.041417878
# 25      0       0.04459561
# 25      15      0.044694483
# 25      25      0.044660326
# 25      35      0.044445165
# 25      50      0.0438478
# 25      70      0.04279756
# 25      100     0.041066755
# 35      0       0.04406192
# 35      15      0.04414489
# 35      25      0.044076685
# 35      35      0.04380004
# 35      50      0.043121677
# 35      70      0.041969184
# 35      100     0.04011658
# 50      0       0.042780202
# 50      15      0.04284841
# 50      25      0.042738568
# 50      35      0.04237241
# 50      50      0.041569985
# 50      70      0.040228155
# 50      100     0.03808937
# 70      0       0.040728968
# 70      15      0.040801574
# 70      25      0.040670436
# 70      35      0.04021748
# 70      50      0.039155353
# 70      70      0.03775953
# 70      100     0.035143416
# 100     0       0.03735972
# 100     15      0.037436243
# 100     25      0.037281964
# 100     35      0.036774807
# 100     50      0.03552852
# 100     70      0.03350106
# 100     100     0.031021748

# Best tornado: 15        15      0.044846594     TORPROB:calculated:hour fcst:calculated_prob:15mi mean  TORPROB:calculated:hour fcst:calculated_prob:15mi mean

# wind
# ["WINDPROB:calculated:hour fcst:calculated_prob:", "WINDPROB:calculated:hour fcst:calculated_prob:15mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:25mi mean", "WINDPROB:calculated:hour fcst:calculated_p
# rob:35mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:50mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:70mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:cal
# culated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_wind
# 0       0       0.12782909
# 0       15      0.12854694
# 0       25      0.1288311
# 0       35      0.12894967
# 0       50      0.12873289
# 0       70      0.12792678
# 0       100     0.12606253
# 15      0       0.12859476
# 15      15      0.12906836
# 15      25      0.12930617
# 15      35      0.12939164
# 15      50      0.12915662
# 15      70      0.12835608
# 15      100     0.12652718
# 25      0       0.12867017
# 25      15      0.1290938
# 25      25      0.1292555
# 25      35      0.12926982
# 25      50      0.12896138
# 25      70      0.1281054
# 25      100     0.12624566
# 35      0       0.12813316
# 35      15      0.1285154
# 35      25      0.1285998
# 35      35      0.12847693
# 35      50      0.12800072
# 35      70      0.12698679
# 35      100     0.12499367
# 50      0       0.12638374
# 50      15      0.12672614
# 50      25      0.12671742
# 50      35      0.12640594
# 50      50      0.12560302
# 50      70      0.12421823
# 50      100     0.12184997
# 70      0       0.12293293
# 70      15      0.123239614
# 70      25      0.12312999
# 70      35      0.12260182
# 70      50      0.121368974
# 70      70      0.11934433
# 70      100     0.11616182
# 100     0       0.116882965
# 100     15      0.11716573
# 100     25      0.1169534
# 100     35      0.1161728
# 100     50      0.114396706
# 100     70      0.11143596
# 100     100     0.10668006

# Best wind: 15   35      0.12939164      WINDPROB:calculated:hour fcst:calculated_prob:15mi mean WINDPROB:calculated:hour fcst:calculated_prob:35mi mean

# wind_adj
# [15, 16, 17, 18, 19, 20, 21, 64]
# ["WINDPROB:calculated:hour fcst:calculated_prob:", "WINDPROB:calculated:hour fcst:calculated_prob:15mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:25mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:35mi mean", "WINDPROB:calc
# ulated:hour fcst:calculated_prob:50mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:70mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_wind_adj
# 0       0       0.08149846
# 0       15      0.0819842
# 0       25      0.08214298
# 0       35      0.08214686
# 0       50      0.08184287
# 0       70      0.08114912
# 0       100     0.07961836
# 15      0       0.08222223
# 15      15      0.082511716
# 15      25      0.082636714
# 15      35      0.08261672
# 15      50      0.082300484
# 15      70      0.08161317
# 15      100     0.08011718
# 25      0       0.08239792
# 25      15      0.08264794
# 25      25      0.08271261
# 25      35      0.0826386
# 25      50      0.08226616
# 25      70      0.08153497
# 25      100     0.080012284
# 35      0       0.082134105
# 35      15      0.08235645
# 35      25      0.08236622
# 35      35      0.082197204
# 35      50      0.08170555
# 35      70      0.08086437
# 35      100     0.079242215
# 50      0       0.080965914
# 50      15      0.08117376
# 50      25      0.081122674
# 50      35      0.08082395
# 50      50      0.08009962
# 50      70      0.07899857
# 50      100     0.07707879
# 70      0       0.07872702
# 70      15      0.078935966
# 70      25      0.07882211
# 70      35      0.07838706
# 70      50      0.0773673
# 70      70      0.075880885
# 70      100     0.07338514
# 100     0       0.07447552
# 100     15      0.074727006
# 100     25      0.07456818
# 100     35      0.07400938
# 100     50      0.072616816
# 100     70      0.07053216
# 100     100     0.067190096

# Best wind_adj: 25       25      0.08271261

# hail
# ["HAILPROB:calculated:hour fcst:calculated_prob:", "HAILPROB:calculated:hour fcst:calculated_prob:15mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:25mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:35mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:50mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:70mi mean", "HAILPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_hail
# 0       0       0.09149646
# 0       15      0.09212156
# 0       25      0.0922787
# 0       35      0.092147544
# 0       50      0.091541134
# 0       70      0.090346426
# 0       100     0.08821113
# 15      0       0.09222775
# 15      15      0.09264722
# 15      25      0.09276263
# 15      35      0.09259314
# 15      50      0.09196228
# 15      70      0.09077332
# 15      100     0.088676035
# 25      0       0.09231119
# 25      15      0.09268868
# 25      25      0.09272799
# 25      35      0.092474446
# 25      50      0.09175265
# 25      70      0.09050612
# 25      100     0.08839166
# 35      0       0.09177924
# 35      15      0.092119426
# 35      25      0.092070885
# 35      35      0.091651455
# 35      50      0.09070965
# 35      70      0.08928323
# 35      100     0.087059796
# 50      0       0.09010119
# 50      15      0.09042193
# 50      25      0.09026538
# 50      35      0.0896066
# 50      50      0.08822686
# 50      70      0.0863032
# 50      100     0.08367961
# 70      0       0.087010466
# 70      15      0.08733356
# 70      25      0.08708777
# 70      35      0.08619252
# 70      50      0.08427911
# 70      70      0.0815599
# 70      100     0.0781099
# 100     0       0.082003586
# 100     15      0.08235142
# 100     25      0.08204425
# 100     35      0.08094832
# 100     50      0.07848735
# 100     70      0.074675895
# 100     100     0.06952725

# Best hail: 15   25      0.09276263      HAILPROB:calculated:hour fcst:calculated_prob:15mi mean HAILPROB:calculated:hour fcst:calculated_prob:25mi mean

# sig_tornado
# ["STORPROB:calculated:hour fcst:calculated_prob:", "STORPROB:calculated:hour fcst:calculated_prob:15mi mean", "STORPROB:calculated:hour fcst:calculated_prob:25mi mean", "STORPROB:calculated:hour fcst:calculated_prob:35mi mean", "STORPROB:calculated:hour fcst:calculated_prob:50mi mean", "STORPROB:calculated:hour fcst:calculated_prob:70mi mean", "STORPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_sig_tornado
# 0       0       0.031367242
# 0       15      0.031583294
# 0       25      0.031554986
# 0       35      0.03139269
# 0       50      0.031022206
# 0       70      0.030473657
# 0       100     0.029565193
# 15      0       0.03157358
# 15      15      0.031650614
# 15      25      0.03157411
# 15      35      0.031342875
# 15      50      0.0309454
# 15      70      0.030410293
# 15      100     0.02960142
# 25      0       0.03134968
# 25      15      0.031392828
# 25      25      0.031269126
# 25      35      0.030961
# 25      50      0.030637357
# 25      70      0.030053802
# 25      100     0.029244522
# 35      0       0.03063031
# 35      15      0.030640762
# 35      25      0.030458132
# 35      35      0.0300357
# 35      50      0.029319063
# 35      70      0.028849639
# 35      100     0.02802408
# 50      0       0.029171433
# 50      15      0.029148236
# 50      25      0.028891636
# 50      35      0.028336221
# 50      50      0.02738366
# 50      70      0.026464887
# 50      100     0.025591118
# 70      0       0.027109405
# 70      15      0.027067557
# 70      25      0.02675049
# 70      35      0.026062172
# 70      50      0.024876283
# 70      70      0.023388447
# 70      100     0.022345375
# 100     0       0.02448187
# 100     15      0.024400042
# 100     25      0.024007479
# 100     35      0.023162965
# 100     50      0.02168727
# 100     70      0.019797347
# 100     100     0.017623859

# Best sig_tornado: 15    15      0.031650614     STORPROB:calculated:hour fcst:calculated_prob:15mi mean STORPROB:calculated:hour fcst:calculated_prob:15mi mean

# sig_wind
# ["SWINDPRO:calculated:hour fcst:calculated_prob:", "SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_sig_wind
# 0       0       0.02502842
# 0       15      0.025245862
# 0       25      0.025365705
# 0       35      0.025592085
# 0       50      0.025753872
# 0       70      0.025689758
# 0       100     0.025311602
# 15      0       0.025272004
# 15      15      0.02541912
# 15      25      0.025511334
# 15      35      0.025695192
# 15      50      0.026027227
# 15      70      0.026028465
# 15      100     0.025693418
# 25      0       0.025300851
# 25      15      0.025432006
# 25      25      0.025494264
# 25      35      0.025616013
# 25      50      0.025977153
# 25      70      0.026015101
# 25      100     0.025686858
# 35      0       0.025172995
# 35      15      0.02528631
# 35      25      0.025322236
# 35      35      0.025370907
# 35      50      0.025669916
# 35      70      0.025728617
# 35      100     0.025399331
# 50      0       0.024767866
# 50      15      0.024865665
# 50      25      0.024865575
# 50      35      0.024832252
# 50      50      0.024844542
# 50      70      0.025001906
# 50      100     0.024673507
# 70      0       0.024043621
# 70      15      0.024139874
# 70      25      0.024106236
# 70      35      0.023992406
# 70      50      0.023745565
# 70      70      0.023806818
# 70      100     0.02340155
# 100     0       0.022681111
# 100     15      0.022789313
# 100     25      0.022739934
# 100     35      0.022555487
# 100     50      0.022134734
# 100     70      0.021603536
# 100     100     0.0211607

# Best sig_wind: 15       70      0.026028465     SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean

# sig_wind_adj
# [43, 44, 45, 46, 47, 48, 49, 64]
# ["SWINDPRO:calculated:hour fcst:calculated_prob:", "SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_sig_wind_adj
# 0       0       0.017062822
# 0       15      0.017215442
# 0       25      0.017293176
# 0       35      0.017352778
# 0       50      0.01739703
# 0       70      0.017377157
# 0       100     0.017216396
# 15      0       0.017247155
# 15      15      0.017356664
# 15      25      0.017435541
# 15      35      0.017495638
# 15      50      0.017543487
# 15      70      0.017524935
# 15      100     0.017368006
# 25      0       0.017345412
# 25      15      0.017456252
# 25      25      0.017525434
# 25      35      0.017578002
# 25      50      0.01761635
# 25      70      0.017590664
# 25      100     0.017427826
# 35      0       0.01737602
# 35      15      0.017485317
# 35      25      0.017558016
# 35      35      0.017610498
# 35      50      0.017637556
# 35      70      0.017585032
# 35      100     0.017397262
# 50      0       0.017295217
# 50      15      0.017390262
# 50      25      0.01745348
# 50      35      0.017493015
# 50      50      0.017511109
# 50      70      0.017435387
# 50      100     0.01719377
# 70      0       0.016991103
# 70      15      0.017090097
# 70      25      0.017131787
# 70      35      0.017131967
# 70      50      0.017089665
# 70      70      0.016932802
# 70      100     0.016617335
# 100     0       0.016406197
# 100     15      0.016511602
# 100     25      0.016542042
# 100     35      0.01650199
# 100     50      0.016363626
# 100     70      0.016126648
# 100     100     0.015554884

# Best sig_wind_adj: 35   50      0.017637556

# sig_hail
# ["SHAILPRO:calculated:hour fcst:calculated_prob:", "SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean", "SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_sig_hail
# 0       0       0.023658833
# 0       15      0.02385503
# 0       25      0.02389502
# 0       35      0.023823215
# 0       50      0.02363681
# 0       70      0.023340043
# 0       100     0.022921173
# 15      0       0.023842653
# 15      15      0.02395763
# 15      25      0.023982298
# 15      35      0.023898004
# 15      50      0.023703074
# 15      70      0.023408083
# 15      100     0.023007764
# 25      0       0.023791503
# 25      15      0.023889007
# 25      25      0.023886377
# 25      35      0.02377506
# 25      50      0.023551397
# 25      70      0.023235917
# 25      100     0.02283137
# 35      0       0.023475155
# 35      15      0.023556963
# 35      25      0.023523886
# 35      35      0.023357054
# 35      50      0.02306713
# 35      70      0.0226924
# 35      100     0.022248661
# 50      0       0.022780843
# 50      15      0.022847379
# 50      25      0.022777347
# 50      35      0.022536298
# 50      50      0.022113658
# 50      70      0.021597566
# 50      100     0.021030804
# 70      0       0.021748312
# 70      15      0.021796376
# 70      25      0.02168998
# 70      35      0.021370295
# 70      50      0.020790959
# 70      70      0.020044014
# 70      100     0.019216752
# 100     0       0.020290112
# 100     15      0.020329118
# 100     25      0.020190774
# 100     35      0.019795246
# 100     50      0.019048763
# 100     70      0.018017316
# 100     100     0.01671048

# Best sig_hail: 15       25      0.023982298     SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean

# tornado_life_risk
# ["PRSIGSVR:calculated:hour fcst:calculated_prob:", "PRSIGSVR:calculated:hour fcst:calculated_prob:15mi mean", "PRSIGSVR:calculated:hour fcst:calculated_prob:25mi mean", "PRSIGSVR:calculated:hour fcst:calculated_prob:35mi mean", "PRSIGSVR:calculated:hour fcst:calculated_prob:50mi mean", "PRSIGSVR:calculated:hour fcst:calculated_prob:70mi mean", "PRSIGSVR:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_tornado_life_risk
# 0       0       0.0067399177
# 0       15      0.006764876
# 0       25      0.006718952
# 0       35      0.0066055283
# 0       50      0.0063984604
# 0       70      0.0061102416
# 0       100     0.0057503837
# 15      0       0.006756751
# 15      15      0.0067530987
# 15      25      0.0067009144
# 15      35      0.0065830597
# 15      50      0.006372003
# 15      70      0.0060794605
# 15      100     0.005716136
# 25      0       0.0066733602
# 25      15      0.006662851
# 25      25      0.0066018174
# 25      35      0.006473958
# 25      50      0.006255723
# 25      70      0.0059566535
# 25      100     0.0055856495
# 35      0       0.0064598573
# 35      15      0.0064415527
# 35      25      0.006369658
# 35      35      0.006223618
# 35      50      0.0059842383
# 35      70      0.005671027
# 35      100     0.0052873185
# 50      0       0.0060724756
# 50      15      0.0060468507
# 50      25      0.005966215
# 50      35      0.0058050323
# 50      50      0.0055439044
# 50      70      0.0051954496
# 50      100     0.0047757546
# 70      0       0.005543325
# 70      15      0.0055082235
# 70      25      0.005414757
# 70      35      0.0052328436
# 70      50      0.0049467892
# 70      70      0.004565
# 70      100     0.0041037975
# 100     0       0.0049474323
# 100     15      0.0049073617
# 100     25      0.004800827
# 100     35      0.004592278
# 100     50      0.004270465
# 100     70      0.003846182
# 100     100     0.0033183633

# Best tornado_life_risk: 0       15      0.006764876     PRSIGSVR:calculated:hour fcst:calculated_prob:  PRSIGSVR:calculated:hour fcst:calculated_prob:15mi mean


println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f35\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()



# event_name        best_blur_radius_f1 best_blur_radius_f36 AU_PR
# tornado           15                  15                   0.044846594
# wind              15                  35                   0.12939164
# wind_adj          25                  25                   0.08271261
# hail              15                  25                   0.09276263
# sig_tornado       15                  15                   0.031650614
# sig_wind          15                  70                   0.026028465
# sig_wind_adj      35                  50                   0.017637556
# sig_hail          15                  25                   0.023982298
# tornado_life_risk 0                   15                   0.006764876


# Now go back to HREFPrediction2024.jl and put those numbers in



# CHECKING that the blurred forecasts are correct

# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.

# The unblurred prediction forecasts should be in the lib/computation_cache, so don't waste time hitting disk
# FORECAST_DISK_PREFETCH=false make julia

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction2024.regular_forecasts(HREFPrediction2024.forecasts_blurred()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2024, 2, 28, 12)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100])

# rm("validation_forecasts_blurred_2024"; recursive=true)

ENV["FORECAST_DISK_PREFETCH"] = "false"

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_blurred_2024");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction2024.models)
    (event_name, _, _) = HREFPrediction2024.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    # au_pr_curve = area_under_pr_curve_interpolated(x, y, weights)
    println("$event_name ($(sum(y))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
  end
end
test_predictive_power(validation_forecasts_blurred, X, Ys, weights)

# EXPECTED:
# event_name        AU_PR
# tornado           0.044846594
# wind              0.12939164
# wind_adj          0.08271261
# hail              0.09276263
# sig_tornado       0.031650614
# sig_wind          0.026028465
# sig_wind_adj      0.017637556
# sig_hail          0.023982298
# tornado_life_risk 0.006764876


# ACTUAL:

# tornado (101642.0)            feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.044846594
# wind (874384.0)               feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.12939164
# wind_adj (278174.34)          feature 3 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.08271261
# hail (405123.0)               feature 4 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.09276263
# sig_tornado (13792.0)         feature 5 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.031650614
# sig_wind (84250.0)            feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.026028465
# sig_wind_adj (31404.064)      feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.017637556
# sig_hail (51908.0)            feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.023982298
# tornado_life_risk (3093.1033) feature 9 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.006764876

# yay!


Metrics.reliability_curves_midpoints(20, X, Ys, map(m -> m[1], HREFPrediction2024.models_with_gated), weights, map(m -> m[3], HREFPrediction2024.models_with_gated))

# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,
# 4.977491e-6,5.409505e-6,5.137307e-5,4.6773588e-5,1.7182001e-5,1.455972e-5,1.8662524e-5,2.1351121e-5,7.0779544e-7,7.194496e-7,4.8419984e-6,4.4645235e-6,1.4993236e-6,1.6296904e-6,2.2964487e-6,2.703989e-6,6.298349e-8,7.2044537e-7,
# 0.00025146615,0.00027031396,0.0024511963,0.002331398,0.00085950276,0.0007921487,0.0012373476,0.0012594636,0.00013447806,0.00019814324,0.00020872787,0.00022359591,9.06345e-5,8.069936e-5,0.0002560058,0.00028807682,1.0765029e-5,0.00017272728,
# 0.000640631,0.00062959397,0.005008284,0.00472252,0.0018011968,0.0016749598,0.00264853,0.0027951254,0.00028420042,0.0004487273,0.00046747556,0.00042056153,0.00024174337,0.00021006237,0.0005667947,0.0006362532,2.9626826e-5,0.00039837667,
# 0.0011936144,0.0013813013,0.007949347,0.008021718,0.0030199178,0.0026397621,0.0042287926,0.00457191,0.00052390853,0.0004908678,0.00082952384,0.0007903033,0.00043569555,0.00043396014,0.0009744628,0.0009255752,6.514139e-5,0.0005272371,
# 0.0018504255,0.0022364545,0.011200475,0.011518874,0.004553267,0.004275769,0.0060837986,0.0062203542,0.00092779385,0.0009041884,0.0012714951,0.001185325,0.0006691027,0.00064906763,0.0015260795,0.0013753948,0.0001341088,0.0008376753,
# 0.0026932503,0.003059213,0.014846678,0.015771018,0.0062691323,0.006699799,0.008341792,0.008334314,0.0014977277,0.0011677721,0.0018044093,0.0018316838,0.00096683874,0.00095115934,0.00219627,0.002057963,0.00023612232,0.0015001523,
# 0.003808044,0.004127698,0.018917656,0.020663528,0.008131922,0.00940504,0.010978972,0.011193747,0.0022553725,0.0020871975,0.002431887,0.0024234639,0.0013239982,0.0014092214,0.0029561562,0.0029628936,0.00036145613,0.0023329605,
# 0.005209509,0.005835179,0.023509009,0.026037088,0.010211787,0.012269134,0.013971722,0.014565542,0.003116983,0.002972473,0.003161491,0.0035203784,0.0017252406,0.0020401077,0.0038042,0.004014226,0.0005150754,0.0031942786,
# 0.006862989,0.007998768,0.028694207,0.032485217,0.01262279,0.0150713585,0.017392302,0.018476132,0.004158918,0.004310596,0.0039658807,0.004617191,0.0021806252,0.002642368,0.0047378098,0.0054604113,0.000702473,0.004361173,
# 0.008796359,0.010440413,0.03458931,0.039396394,0.015502325,0.018505596,0.021387925,0.02264472,0.005401025,0.005945097,0.0048708254,0.0059947036,0.002700162,0.0036240595,0.0057641817,0.0068483087,0.00091867754,0.0063870945,
# 0.011055802,0.013840616,0.04132271,0.048270237,0.01889301,0.02357547,0.026166383,0.02752715,0.006891214,0.00840477,0.00584985,0.008131803,0.003269162,0.0048874016,0.006991429,0.0077940524,0.0011731958,0.0077075786,
# 0.013697664,0.01724283,0.04906352,0.057128876,0.02283756,0.029040694,0.03192437,0.0335347,0.008545258,0.0124447085,0.0069440296,0.009435399,0.0039044612,0.0061027776,0.008532881,0.0092358105,0.0014959547,0.009367543,
# 0.016910655,0.020686502,0.058224823,0.06746239,0.027572965,0.035328515,0.0388291,0.041325595,0.010421541,0.015215635,0.008301392,0.010633212,0.0046504233,0.0072475984,0.010480731,0.010987732,0.0018892649,0.012021539,
# 0.02083242,0.026112901,0.06925795,0.08099383,0.033383194,0.04296763,0.047137067,0.05112994,0.012676806,0.01994305,0.010041561,0.012239118,0.0055603427,0.008646059,0.0128707,0.014408461,0.0024044653,0.013063961,
# 0.025734391,0.030324874,0.082867324,0.095519535,0.04080706,0.051666092,0.057291664,0.06352094,0.015144683,0.028423954,0.01232429,0.014489772,0.006681883,0.010484819,0.015723364,0.018703707,0.003179616,0.01564534,
# 0.03248483,0.035383243,0.10034661,0.113434546,0.05072856,0.06382445,0.07012392,0.07921522,0.017900564,0.03460746,0.015404123,0.01729798,0.008148398,0.012553232,0.019260408,0.022570536,0.0042959815,0.021756385,
# 0.04235359,0.043830927,0.1237082,0.138721,0.06403076,0.083994694,0.086925425,0.10086322,0.021560619,0.03739465,0.019965153,0.021196749,0.010414738,0.013544954,0.023944058,0.028960055,0.0058535826,0.027546283,
# 0.056999132,0.061973114,0.15678537,0.1766725,0.08323015,0.10880142,0.10990023,0.13170478,0.027121035,0.045397468,0.02745386,0.029188816,0.0143981995,0.018479308,0.030637613,0.03610884,0.008140826,0.03615846,
# 0.08068845,0.09154007,0.21017087,0.23664358,0.11707217,0.15687117,0.14598498,0.18045685,0.037467536,0.05397256,0.041969605,0.045057714,0.022521352,0.030954657,0.041877445,0.047935735,0.0120253805,0.045450047,
# 0.14041229,0.14667808,0.33990055,0.36636552,0.21377791,0.2650771,0.23395628,0.29742417,0.06688269,0.10971996,0.08302862,0.0939055,0.045344967,0.06998137,0.072786614,0.071633674,0.026059728,0.08097472,



# calibrate the hourlies

const bin_count = 6

function find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

  ŷ              = @view X[:,prediction_i]; # HREF prediction for event_name
  sort_perm      = Metrics.parallel_sort_perm(ŷ);
  y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
  ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
  weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

  bins_Σŷ      = zeros(Float64, bin_count)
  bins_Σy      = zeros(Float64, bin_count)
  bins_Σweight = zeros(Float64, bin_count)
  bins_max     = ones(Float32, bin_count)

  bin_i = 1
  for i in 1:length(y_sorted)
    if ŷ_sorted[i] > bins_max[bin_i]
      bin_i += 1
    end

    bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
    bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
    bins_Σweight[bin_i] += Float64(weights_sorted[i])

    if bins_Σy[bin_i] >= per_bin_pos_weight
      bins_max[bin_i] = ŷ_sorted[i]
    end
  end

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for bin_i in 1:bin_count
    Σŷ      = bins_Σŷ[bin_i]
    Σy      = bins_Σy[bin_i]
    Σweight = bins_Σweight[bin_i]

    mean_ŷ = Σŷ / Σweight
    mean_y = Σy / Σweight

    println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
  end

  bins_max
end

event_to_bins = Dict{String,Vector{Float32}}()
for prediction_i in 1:length(HREFPrediction2024.models)
  (event_name, _, _, _, _) = HREFPrediction2024.models[prediction_i]

  event_to_bins[event_name] = find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)

  # println("event_to_bins[\"$event_name\"] = $(event_to_bins[event_name])")
end

# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# tornado           1.7501354796653564e-5  1.6612787138194736e-5  9.074858595611498e8  0.0011201472
# tornado           0.0024167125032373625  0.0020956217342336663  6.571936703451216e6  0.0040470827
# tornado           0.007179017953625256   0.00624735513873309    2.212258956207752e6  0.009924359
# tornado           0.01734390212629259    0.013914922620331272   915710.4554556012    0.020150831
# tornado           0.03269118817939099    0.028881364294202214   485834.4418205023    0.04450299
# tornado           0.08168801878204363    0.07550402019477592    194392.89683753252   1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# wind              0.00015105371859579558 0.00016045336800442223 8.985119620855772e8  0.0075269686
# wind              0.012746369728047817   0.012191625172081211   1.064799517927134e7  0.019717596
# wind              0.03049249352749501    0.02712869346405187    4.4510424723523855e6 0.037927717
# wind              0.05809039019524568    0.04986950553491064    2.336415893607497e6  0.067318514
# wind              0.1032872767469163     0.09040409513779539    1.3140406404521465e6 0.12851062
# wind              0.22450222307361142    0.2021319329708971     604536.7384964824    1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# wind_adj          4.715671981624615e-5   5.258568115140553e-5   9.035483740155869e8  0.0028242674
# wind_adj          0.004826439926474433   0.00484010578870181    8.827940516079962e6  0.008482344
# wind_adj          0.014198717129509675   0.011915722178326468   3.0008001959676743e6 0.017187402
# wind_adj          0.02937275711164607    0.023166684129607606   1.450592324770689e6  0.032382138
# wind_adj          0.05717780692685101    0.044825428092086415   745186.8497611284    0.06685648
# wind_adj          0.14536133630543366    0.11197483749194498    293099.1121739745    1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# hail              6.93423359881619e-5    6.462175521516936e-5   9.021169586102781e8  0.003987495
# hail              0.006935951562136347   0.006781555881979186   9.019026801820993e6  0.011505866
# hail              0.01718840449497406    0.016337856632643217   3.639411119345665e6  0.023705853
# hail              0.03430094456383672    0.032355307910259166   1.8237266280366182e6 0.04564259
# hail              0.06937913848184941    0.06223903199179972    901643.6012685895    0.09055661
# hail              0.17127023123792567    0.1391641067039236     365226.25418168306   1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# sig_tornado       2.384386450155931e-6   1.8962338735085708e-6  9.146988187708921e8  0.00048080692
# sig_tornado       0.0009520157283366507  0.0010663773028390133  2.290899461978376e6  0.0023937228
# sig_tornado       0.0037750329794483104  0.00372735108903595    577742.8457904458    0.006148654
# sig_tornado       0.011796470457632435   0.008481687449911545   184898.29661512375   0.012262353
# sig_tornado       0.028870615942764116   0.016127371756997452   75539.41810506582    0.022267453
# sig_tornado       0.05714779598306871    0.03594229015485278    38094.21869623661    1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# sig_wind          1.4406305840667845e-5  1.5127618749596658e-5  9.027850034346155e8  0.00078459026
# sig_wind          0.00136906325517261    0.0014072677226638346  9.49990525327766e6   0.0025756117
# sig_wind          0.0042493676988290905  0.0036766042705208713  3.0608166912840605e6 0.005378738
# sig_wind          0.009496157129992277   0.007119871402227385   1.3695999124849439e6 0.009706564
# sig_wind          0.01564042319526102    0.013688993275156732   831581.0632736087    0.020927433
# sig_wind          0.04075107924289223    0.03763593467509922    319086.6598518491    1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# sig_wind_adj      5.276077872504237e-6   5.607901531806515e-6   9.08093031498491e8   0.00041078764
# sig_wind_adj      0.000746652195160004   0.0007447045575933897  6.417009086224198e6  0.0013922893
# sig_wind_adj      0.002446313080120781   0.0020063946913092827  1.958810379295051e6  0.002989863
# sig_wind_adj      0.006105444044360073   0.003950037529387319   784806.289450109     0.0053869793
# sig_wind_adj      0.011112869545079423   0.0073460096050324795  431181.0001280308    0.010809073
# sig_wind_adj      0.026437817481561675   0.019379419829542853   181154.75997298956   1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# sig_hail          8.879293227663033e-6   7.95215191033246e-6    9.096900670081505e8  0.00091282465
# sig_hail          0.0015848414358735864  0.0016860357258137659  5.096874928020716e6  0.0031095394
# sig_hail          0.004912206482770015   0.004394310515751464   1.6444091458676457e6 0.00634054
# sig_hail          0.009442767974912704   0.00872466554448439    855436.23040241      0.012477984
# sig_hail          0.020137071863751882   0.01707176028772785    401132.877597034     0.02483106
# sig_hail          0.04533856964099356    0.04108481052463147    178072.82027477026   1.0
# event_name        mean_y                 mean_ŷ                 Σweight              bin_max
# tornado_life_risk 5.370251833986228e-7   2.739560811282957e-7   9.157380286567521e8  0.00011407318
# tornado_life_risk 0.00033042528593478043 0.0002663905866201927  1.4884117221491933e6 0.00061691855
# tornado_life_risk 0.0013657614834284466  0.0009372564869139419  360205.4012424946    0.0014705167
# tornado_life_risk 0.003174712852798226   0.002050655458856236   154891.1291909814    0.0029623061
# tornado_life_risk 0.006111093283938664   0.004280699525806122   80475.20726698637    0.0065326905
# tornado_life_risk 0.011171033413829332   0.012557482548342188   43980.902911901474   1.0


println(event_to_bins)
# Dict{String, Vector{Float32}}("sig_wind" => [0.00078459026, 0.0025756117, 0.005378738, 0.009706564, 0.020927433, 1.0], "sig_hail" => [0.00091282465, 0.0031095394, 0.00634054, 0.012477984, 0.02483106, 1.0], "hail" => [0.003987495, 0.011505866, 0.023705853, 0.04564259, 0.09055661, 1.0], "sig_wind_adj" => [0.00041078764, 0.0013922893, 0.002989863, 0.0053869793, 0.010809073, 1.0], "tornado_life_risk" => [0.00011407318, 0.00061691855, 0.0014705167, 0.0029623061, 0.0065326905, 1.0], "tornado" => [0.0011201472, 0.0040470827, 0.009924359, 0.020150831, 0.04450299, 1.0], "wind_adj" => [0.0028242674, 0.008482344, 0.017187402, 0.032382138, 0.06685648, 1.0], "sig_tornado" => [0.00048080692, 0.0023937228, 0.006148654, 0.012262353, 0.022267453, 1.0], "wind" => [0.0075269686, 0.019717596, 0.037927717, 0.067318514, 0.12851062, 1.0])


# 4. combine bin-pairs (overlapping, 5 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + b)

import LogisticRegression

const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,prediction_i]; # HREF prediction for event_name

  bins_max = event_to_bins[event_name]
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_href_x  = X[bin_members, prediction_i]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 1))

    Threads.@threads :static for i in 1:length(bin_y)
      logit_href = logit(bin_href_x[i])

      bin_X_features[i,1] = logit_href
      # bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
      # bin_X_features[i,3] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,5] = max(logit_href, logit_sref)
      # bin_X_features[i,6] = min(logit_href, logit_sref)
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("event_name", event_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("HREF_ŷ_min", bin_min),
      ("HREF_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_HREF_ŷ", sum(bin_href_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("HREF_logloss", sum(logloss.(bin_y, bin_href_x) .* bin_weights) / bin_weight),
      ("HREF_au_pr", Metrics.area_under_pr_curve(bin_href_x, bin_y, bin_weights)),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights)),
      ("logistic_coeffs", coeffs)
    ]

    headers = map(first, stuff)
    row     = map(last, stuff)

    bin_i == 1 && println(join(headers, "\t"))
    println(join(row, "\t"))

    push!(bins_logistic_coeffs, coeffs)
  end

  bins_logistic_coeffs
end

event_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:length(HREFPrediction2024.models)
  (event_name, _, _, _, _) = HREFPrediction2024.models[prediction_i]

  event_to_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# tornado 1-2     -1.0    0.0040470827    991813832       34175.0 9.140578e8      3.1560547e-5    3.475132e-5     0.00028464381   0.0023222638    3.4751334e-5    0.00028447784   0.0023183043    Float32[1.0151445, 0.2071434]
# Float32[0.990645, 0.08936819]6]]
# tornado 2-3     0.0011201472    0.009924359     9386675 33951.0 8.784196e6      0.0031412167    0.003616077     0.023287237     0.0074321157    0.0036160771    0.02325282      0.0074321195    Float32[0.990645, 0.08936819]
# Float32[1.0937685, 0.61661005]]]
# tornado 3-4     0.0040470827    0.020150831     3326649 33832.0 3.1279695e6     0.008492029     0.010154779     0.05559735      0.017400458     0.010154781     0.05543317      0.01740044      Float32[1.0937685, 0.61661005]
# Float32[0.86215496, -0.3573002]]
# tornado 4-5     0.009924359     0.04450299      1484737 33759.0 1.4015449e6     0.019102922     0.022663917     0.107075796     0.032811765     0.022663917     0.10671539      0.032811753     Float32[0.86215496, -0.3573002]
# Float32[0.9471853, -0.049062375]
# tornado 5-6     0.020150831     1.0     718487  33635.0 680227.4        0.042205017     0.04669332      0.18149939      0.09635553      0.04669332      0.1812272       0.09635552      Float32[0.9471853, -0.049062375]
# Float32[1.052107, 0.26756564]]]]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# wind    1-2     -1.0    0.019717596     986512296       291269.0        9.0915994e8     0.00030136132   0.0002985689    0.0018736775    0.01249159      0.0002985689    0.0018729281    0.012491581     Float32[1.052107, 0.26756564]
# Float32[1.0934643, 0.45261195]5]
# wind    2-3     0.0075269686    0.037927717     16181757        291318.0        1.5099038e7     0.01659492      0.017977748     0.08783856      0.030573916     0.017977742     0.087764814     0.030573908     Float32[1.0934643, 0.45261195]
# Float32[1.0603855, 0.33706936]3]
# wind    3-4     0.019717596     0.067318514     7279979 291267.0        6.787458e6      0.03495666      0.039992392     0.16545215      0.05839914      0.039992392     0.16508158      0.05839749      Float32[1.0603855, 0.33706936]
# Float32[0.97554934, 0.09084179]]
# wind    4-5     0.037927717     0.12851062      3921334 291372.0        3.6504565e6     0.06446058      0.07435974      0.2611329       0.10503989      0.07435973      0.26034647      0.105039924     Float32[0.97554934, 0.09084179]
# Float32[0.98142743, 0.10940261]]
# wind    5-6     0.067318514     1.0     2066693 291848.0        1.9185775e6     0.12560913      0.14148164      0.38980708      0.25563306      0.14148167      0.38863906      0.25563303      Float32[0.98142743, 0.10940261]
# Float32[1.0665799, 0.36155185]]]]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# wind_adj        1-2     -1.0    0.008482344     989889656       92241.82        9.123763e8      9.890861e-5     9.3399954e-5    0.000680384     0.005162785     9.339994e-5     0.0006798179    0.005156657     Float32[1.0665799, 0.36155185]
# Float32[1.2188917, 1.1445187]]]]
# wind_adj        2-3     0.0028242674    0.017187402     12799133        92533.56        1.182874e7      0.0066350987    0.0072040665    0.041377626     0.014106335     0.007204068     0.041309148     0.014106285     Float32[1.2188917, 1.1445187]
# Float32[1.0872297, 0.5649324]]9]
# wind_adj        3-4     0.008482344     0.032382138     4833035 92617.42        4.451393e6      0.0155821135    0.019143537     0.09342104      0.029667573     0.019143537     0.09302372      0.029667579     Float32[1.0872297, 0.5649324]
# Float32[1.0266879, 0.34076434]]
# wind_adj        4-5     0.017187402     0.06685648      2391968 92773.35        2.1957792e6     0.030517064     0.038809024     0.16218314      0.059275445     0.038809024     0.16110326      0.05927543      Float32[1.0266879, 0.34076434]
# Float32[1.036534, 0.3751357]8]03]
# wind_adj        5-6     0.032382138     1.0     1136277 93315.08        1.038286e6      0.06378112      0.08207125      0.2716118       0.17947556      0.082071245     0.26892096      0.17947555      Float32[1.036534, 0.3751357]
# Float32[0.98223704, -0.05874168]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# hail    1-2     -1.0    0.011505866     988576867       135180.0        9.11136e8       0.00013111041   0.00013731257   0.0009463133    0.0067931553    0.00013731257   0.00094613153   0.006788934     Float32[0.98223704, -0.05874168]
# Float32[1.0158885, 0.1087229]]7]
# hail    2-3     0.003987495     0.023705853     13673229        134727.0        1.2658438e7     0.009529076     0.009883621     0.05416124      0.017288595     0.009883617     0.054154333     0.017288567     Float32[1.0158885, 0.1087229]
# Float32[1.0228801, 0.14139068]]
# hail    3-4     0.011505866     0.04564259      5912835 135127.0        5.4631375e6     0.021684866     0.022900984     0.10727531      0.035133652     0.022900986     0.10724 0.035133656     Float32[1.0228801, 0.14139068]
# Float32[1.0846623, 0.34556273]1]
# hail    4-5     0.023705853     0.09055661      2952224 135403.0        2.72537e6       0.042241845     0.045905985     0.18247357      0.0714904       0.04590598      0.18228622      0.07149042      Float32[1.0846623, 0.34556273]
# Float32[1.127743, 0.46917012]]
# hail    5-6     0.04564259      1.0     1369266 134816.0        1.2668698e6     0.08441579      0.09875336      0.30760708      0.1950368       0.09875335      0.30608276      0.1950368       Float32[1.127743, 0.46917012]
# Float32[1.003107, 0.06779992]]]]3]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# sig_tornado     1-2     -1.0    0.0023937228    994937379       4673.0  9.169897e8      4.555609e-6     4.756834e-6     4.1985368e-5    0.0012952677    4.756834e-6     4.198113e-5     0.0011739942    Float32[1.003107, 0.06779992]
# Float32[1.0754979, 0.41287646]44]
# sig_tornado     2-3     0.00048080692   0.006148654     3034195 4575.0  2.8686425e6     0.0016022957    0.0015205694    0.010947165     0.0042382064    0.00152057      0.010942917     0.004249924     Float32[1.0754979, 0.41287646]
# Float32[1.3813931, 2.1319306]]]]
# sig_tornado     3-4     0.0023937228    0.012262353     803128  4569.0  762641.1        0.0048800153    0.005719788     0.03422437      0.012302845     0.0057197874    0.034065705     0.012302827     Float32[1.3813931, 2.1319306]
# Float32[1.3965261, 2.2143273]]]
# sig_tornado     4-5     0.006148654     0.022267453     273033  4566.0  260437.72       0.010699303     0.01674879      0.084566325     0.028643638     0.016748786     0.08291048      0.02864361      Float32[1.3965261, 2.2143273]
# Float32[0.91918075, 0.24742183]
# sig_tornado     5-6     0.012262353     1.0     118461  4550.0  113633.64       0.02277007      0.038350176     0.16318595      0.06823607      0.038350172     0.15856825      0.06823605      Float32[0.91918075, 0.24742183]
# Float32[1.030558, 0.19321586]]6]7]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# sig_wind        1-2     -1.0    0.0025756117    989844779       28024.0 9.122849e8      2.9624403e-5    2.8512766e-5    0.00024450067   0.0013884482    2.8512763e-5    0.0002444512    0.0013884581    Float32[1.030558, 0.19321586]
# Float32[1.1623001, 1.0385767]]]]
# sig_wind        2-3     0.00078459026   0.005378738     13505255        28105.0 1.2560722e7     0.0019602631    0.0020709406    0.01446144      0.0043633007    0.0020709408    0.014450416     0.0043632924    Float32[1.1623001, 1.0385767]
# Float32[1.1844709, 1.1859512]]8]
# sig_wind        3-4     0.0025756117    0.009706564     4770566 28094.0 4.430417e6      0.0047410405    0.0058713374    0.035569113     0.0092791       0.0058713392    0.03542892      0.009261657     Float32[1.1844709, 1.1859512]
# Float32[0.7777289, -0.807949]]]]
# sig_wind        4-5     0.005378738     0.020927433     2372671 28020.0 2.201181e6      0.009601611     0.01181739      0.06396293      0.01620134      0.01181739      0.063678674     0.016201327     Float32[0.7777289, -0.807949]
# Float32[0.9798549, 0.035456475]]
# sig_wind        5-6     0.009706564     1.0     1243623 28132.0 1.1506678e6     0.020329615     0.02260375      0.10372395      0.058059417     0.022603748     0.103594765     0.05805942      Float32[0.9798549, 0.035456475]
# Float32[0.9896531, -0.11605113]4]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# sig_wind_adj    1-2     -1.0    0.0013922893    992199298       10404.409       9.1451e8        1.0794055e-5    1.0478227e-5    9.822712e-5     0.00085316633   1.0478226e-5    9.822022e-5     0.000853094     Float32[0.9896531, -0.11605113]
# Float32[1.1852539, 1.3334796]5]]]
# sig_wind_adj    2-3     0.00041078764   0.002989863     9089240 10456.594       8.3758195e6     0.0010397695    0.0011441432    0.008650997     0.0026035735    0.0011441433    0.008639854     0.0026034638    Float32[1.1852539, 1.3334796]
# Float32[1.3313962, 2.2548943]]8]
# sig_wind_adj    3-4     0.0013922893    0.0053869793    2988498 10484.82        2.7436165e6     0.0025623702    0.0034930005    0.022967456     0.0061835404    0.0034929996    0.022786843     0.006183524     Float32[1.3313962, 2.2548943]
# Float32[0.9650048, 0.245746]67]]
# sig_wind_adj    4-5     0.002989863     0.010809073     1327490 10455.254       1.2159872e6     0.0051542264    0.007881044     0.046171162     0.0112454705    0.007881044     0.045546494     0.011245461     Float32[0.9650048, 0.245746]
# Float32[1.0000306, 0.3686924]]]
# sig_wind_adj    5-6     0.0053869793    1.0     671172  10514.834       612335.75       0.010906001     0.015646635     0.07849636      0.039967928     0.015646635     0.07757143      0.039967924     Float32[1.0000306, 0.3686924]
# Float32[0.98084414, -0.12066075]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# sig_hail        1-2     -1.0    0.0031095394    992540710       17314.0 9.1478694e8     1.730185e-5     1.7660006e-5    0.00014777816   0.0017137745    1.7660004e-5    0.0001477653    0.0016712913    Float32[0.98084414, -0.12066075]
# Float32[1.166308, 0.9986842]]1]]
# sig_hail        2-3     0.00091282465   0.00634054      7275639 17342.0 6.741284e6      0.0023466684    0.0023964895    0.016377568     0.004971309     0.0023964902    0.01636745      0.0049713026    Float32[1.166308, 0.9986842]
# Float32[0.9872798, 0.031519376]
# sig_hail        3-4     0.0031095394    0.012477984     2697428 17358.0 2.4998455e6     0.005876139     0.0064625447    0.038559485     0.009748793     0.0064625437    0.038530882     0.009748772     Float32[0.9872798, 0.031519376]
# Float32[1.0985931, 0.55506366]]
# sig_hail        4-5     0.00634054      0.02483106      1352734 17294.0 1.2565692e6     0.0113892965    0.012856695     0.06768993      0.020571731     0.012856695     0.06758842      0.020571727     Float32[1.0985931, 0.55506366]
# Float32[0.87546086, -0.3075612]]
# sig_hail        5-6     0.012477984     1.0     620830  17236.0 579205.75       0.024454406     0.027885098     0.12443565      0.049019907     0.027885098     0.12412216      0.049019903     Float32[0.87546086, -0.3075612]
# Float32[0.97598267, 0.19340746]]]]
# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count   pos_count       weight  mean_HREF_ŷ     mean_y  HREF_logloss    HREF_au_pr      mean_logistic_ŷ logistic_logloss        logistic_au_pr  logistic_coeffs
# tornado_life_risk       1-2     -1.0    0.00061691855   995188886       1036.4462       9.172265e8      7.057917e-7     1.072345e-6     1.0685868e-5    0.00048378485   1.0723437e-6    1.0620789e-5    0.00048402627   Float32[0.97598267, 0.19340746]
# Float32[1.0696578, 0.8193858]]]]]
# tornado_life_risk       2-3     0.00011407318   0.0014705167    1952533 1028.7659       1.8486172e6     0.00039710966   0.00053216185   0.0044095335    0.0014357703    0.00053216197   0.0043881903    0.001435767     Float32[1.0696578, 0.8193858]
# Float32[1.0577359, 0.78586924]6]]
# tornado_life_risk       3-4     0.00061691855   0.0029623061    541089  1036.0891       515096.53       0.0012720589    0.0019097187    0.013789453     0.0030430956    0.0019097185    0.0136503205    0.003043094     Float32[1.0577359, 0.78586924]
# Float32[0.8839506, -0.27480733]]
# tornado_life_risk       4-5     0.0014705167    0.0065326905    245746  1032.4497       235366.33       0.0028131404    0.0041787047    0.027061464     0.006323261     0.004178704     0.026767269     0.006323267     Float32[0.8839506, -0.27480733]
# Float32[0.64814734, -1.5873486]]
# tornado_life_risk       5-6     0.0029623061    1.0     128993  1020.5681       124456.11       0.0072055897    0.0078992       0.04567751      0.013693101     0.007899201     0.045380574     0.013693108     Float32[0.64814734, -1.5873486]


# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# tornado      1-2 -1.0          0.004538168  663522521 25328.0   6.114865e8  3.7722053e-5  3.8569822e-5  0.00031404942 0.0025057626862979797 3.8569815e-5    0.00031371796    0.0025057810877705806 Float32[1.0710595,  0.5394862]0.89414734, -0.5616081]]
# tornado      2-3 0.0012153604  0.011742671  6826598   25086.0   6.395767e6  0.003641221   0.003687606   0.02373275    0.007081899006221959  0.003687606     0.023723962      0.007080886384121605  Float32[0.89414734, -0.5616081]
# tornado      3-4 0.004538168   0.023059275  2521603   25053.0   2.3784395e6 0.009743426   0.009916259   0.054313518   0.01738706205391112   0.009916258     0.054299463      0.01738732278874145   Float32[1.108447,   0.50613326]
# tornado      4-5 0.011742671   0.049979284  1096364   25059.0   1.0404369e6 0.022212109   0.022668853   0.10695663    0.0329149983068956    0.022668855     0.106901385      0.032915092429173884  Float32[0.83939207, -0.5758451]
# tornado      5-6 0.023059275   1.0          524116    24912.0   499438.75   0.043356903   0.047219485   0.18174142    0.10422838676781673   0.04721949      0.18125284       0.10422838362397668   Float32[1.2171175,  0.7334359]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# wind         1-2 -1.0          0.020091131  660130788 196117.0  6.083781e8  0.0003179921  0.0003000101  0.0018847216  0.012626556056294723  0.00030001017   0.001881708      0.012626548304852075  Float32[1.0940193,  0.44082266]
# wind         2-3 0.0078376075  0.037815828  10837072  196417.0  1.009659e7  0.016983436   0.018077338   0.08827049    0.03055191239147584   0.018077333     0.0882144        0.030551879728915397  Float32[1.1078424,  0.48961264]
# wind         3-4 0.020091131   0.065400586  4955623   196494.0  4.611817e6  0.034935802   0.039576545   0.16426766    0.056934783477593724  0.03957655      0.1639482        0.056934762904432244  Float32[1.0705312,  0.36004978]
# wind         4-5 0.037815828   0.11733462   2702100   196256.0  2.5110525e6 0.062353585   0.07268644    0.2572899     0.10137994491141147   0.07268643      0.25641176       0.10137737869120787   Float32[1.0221666,  0.22469698]
# wind         5-6 0.065400586   1.0          1481829   195965.0  1.3744602e6 0.118102394   0.13279128    0.3792063     0.22144316909244727   0.13279128      0.37804726       0.22144315901635722   Float32[0.9163128,  -0.022937609]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# wind_adj     1-2 -1.0          0.008021789  662276769 60416.832 6.1042534e8 0.00010135513 9.1453105e-5  0.0006765833  0.00449694161036154   9.145309e-5     0.000675008      0.004496931627945791  Float32[1.1026261,  0.554337]
# wind_adj     2-3 0.0025728762  0.016496103  9452124   60655.164 8.719148e6  0.006212037   0.0064026015  0.037491444   0.012887062939626086  0.0064026       0.037456684      0.012883091892627438  Float32[1.1907648,  0.9671046]
# wind_adj     3-4 0.008021789   0.030200316  3467153   60774.0   3.1872375e6 0.01470288    0.017515238   0.08678374    0.02777114206147479   0.017515238     0.08649106       0.02777035084044691   Float32[1.1658177,  0.862035]
# wind_adj     4-5 0.016496103   0.058084033  1666892   60854.504 1.5271355e6 0.028346961   0.036555316   0.15526581    0.055184535107434526  0.036555316     0.1541252        0.05518450849745459   Float32[1.0740721,  0.5209931]
# wind_adj     5-6 0.030200316   1.0          824318    61098.58  751848.25   0.05596935    0.07424825    0.25896335    0.12743653754297862   0.07424825      0.25600436       0.12743653176423161   Float32[0.9590977,  0.1968526]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# hail         1-2 -1.0          0.00951148   660934254 87998.0   6.091634e8  0.00012539385 0.00013385723 0.00093775016 0.00607737629900066   0.00013385725   0.0009369508     0.006075387218411241  Float32[1.0600259,  0.43388337]
# hail         2-3 0.0033965781  0.0200999    10070263  87700.0   9.320537e6  0.008125199   0.008748478   0.049155116   0.014854077757736648  0.0087484745    0.049130067      0.014854047295768147  Float32[0.96319646, -0.097637914]
# hail         3-4 0.00951148    0.037743744  4516233   87979.0   4.169741e6  0.018013459   0.019555109   0.094599225   0.03042918491899215   0.019555109     0.09452403       0.0304291760614459    Float32[1.0792756,  0.39443073]
# hail         4-5 0.0200999     0.07645232   2240170   88414.0   2.0655298e6 0.03584321    0.03947642    0.16378516    0.0579127469756851    0.03947642      0.16359496       0.057912802594265504  Float32[0.9629313,  -0.01864577]
# hail         5-6 0.037743744   1.0          1117753   88474.0   1.0313141e6 0.07284379    0.07906139    0.26499474    0.15609715817529107   0.07906139      0.26470444       0.1560971504887032    Float32[1.0167043,  0.13172606]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# sig_tornado  1-2 -1.0          0.0032272756 666002121 3457.0    6.1382285e8 5.9626673e-6  5.309203e-6   4.678147e-5   0.001833951472291395  5.309204e-6     4.671619e-5      0.0018338649169751405 Float32[1.0426707,  0.21755064]
# sig_tornado  2-3 0.00080238597 0.007747574  1607048   3394.0    1.5254722e6 0.0023659368  0.0021363355  0.014722086   0.0054325978772233    0.0021363357    0.014700975      0.005432595040627732  Float32[1.1516033,  0.7787872]
# sig_tornado  3-4 0.0032272756  0.0136314705 484467    3384.0    462877.25   0.006111807   0.007039877   0.040661324   0.01626752971401274   0.0070398776    0.04036978       0.01626747429930489   Float32[1.6196959,  3.2147853]
# sig_tornado  4-5 0.007747574   0.022335978  166347    3384.0    159775.9    0.012380175   0.020392418   0.10002828    0.03093794150156197   0.020392416     0.0976591        0.030937900454230553  Float32[1.4218588,  2.3287997]
# sig_tornado  5-6 0.0136314705  1.0          81652     3402.0    78647.305   0.022481717   0.041381754   0.17543556    0.06762628303699908   0.04138175      0.1688633        0.06762626155907245   Float32[1.032921,   0.7553395]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# sig_wind     1-2 -1.0          0.0022018508 661801163 19552.0   6.099468e8  2.7360013e-5  2.9757573e-5  0.0002545742  0.00138971324464957   2.9757566e-5    0.0002541716     0.001373367108601567  Float32[1.0894886,  0.7778136]
# sig_wind     2-3 0.0006871624  0.0047563836 9767449   19615.0   9.071754e6  0.0017652413  0.0020007533  0.014110417   0.003992581297500078  0.002000753     0.014094296      0.003992617027907525  Float32[1.0595194,  0.49330923]
# sig_wind     3-4 0.0022018508  0.008192127  3754983   19680.0   3.4820855e6 0.0040781545  0.0052126553  0.032138065   0.008741883872774291  0.005212655     0.03195264       0.008741855491867194  Float32[1.3312615,  2.0373042]
# sig_wind     4-5 0.0047563836  0.015201166  1772069   19628.0   1.641605e6  0.007951819   0.01105726    0.060907602   0.014663455649926242  0.011057259     0.060352236      0.014663448094255118  Float32[0.87163,    -0.28045976]
# sig_wind     5-6 0.008192127   1.0          1012094   19628.0   935518.7    0.01633615    0.019399984   0.09497683    0.030610951630046828  0.019399982     0.0942568        0.030610947143767372  Float32[0.66988313, -1.1392391]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# sig_wind_adj 1-2 -1.0          0.0012706288 664083552 6943.2876 6.120963e8  1.2320967e-5  1.048358e-5   9.9540746e-5  0.0008474856962220903 1.0483582e-5    9.9337114e-5     0.0007742466196269562 Float32[1.0503622,  0.2618211]
# sig_wind_adj 2-3 0.00042257016 0.002259613  6474711   7033.013  5.9486125e6 0.0009254576  0.001078762   0.008219483   0.0025793723448849478 0.0010787616    0.008177658      0.0025793693040923214 Float32[1.5049535,  3.5953336]
# sig_wind_adj 3-4 0.0012706288  0.003675646  2059537   7065.459  1.8815475e6 0.0020542839  0.0034107538  0.022884354   0.005600466228951639  0.003410754     0.022488942      0.005598384072679459  Float32[1.3739139,  2.7983332]
# sig_wind_adj 4-5 0.002259613   0.0054352577 943625    7024.4463 859270.7    0.0032737446  0.0074688867  0.045292616   0.013113337473667596  0.0074688895    0.04307581       0.013113187761986741  Float32[2.022486,   6.61417]
# sig_wind_adj 5-6 0.003675646   1.0          425151    7044.542  386567.47   0.0055550486  0.016597465   0.09072071    0.024179441332532796  0.01659747      0.08352649       0.024179445504042545  Float32[0.99839044, 1.0990105]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# sig_hail     1-2 -1.0          0.0021597594 663854239 10586.0   6.118555e8  1.4032314e-5  1.6142685e-5  0.00013715753 0.001572345618040608  1.614268e-5     0.00013676504    0.0014903592320110288 Float32[1.0996535,  0.9160373]
# sig_hail     2-3 0.00072055974 0.00463198   5029295   10606.0   4.652242e6  0.001794349   0.0021232117  0.014918671   0.0038137251844720414 0.0021232124    0.014890156      0.0038137240819630233 Float32[1.0066487,  0.20982021]
# sig_hail     3-4 0.0021597594  0.009187652  2187333   10671.0   2.0199059e6 0.004228157   0.0048900396  0.030582078   0.007277253005427299  0.0048900386    0.030530766      0.00727724696762096   Float32[0.9350299,  -0.20348155]
# sig_hail     4-5 0.00463198    0.01903059   1093147   10670.0   1.010515e6  0.008537743   0.009774541   0.054330025   0.015039050530950922  0.009774543     0.054242585      0.015038992180916953  Float32[0.9644414,  -0.02964372]
# sig_hail     5-6 0.009187652   1.0          526668    10593.0   489037.75   0.019050557   0.020192599   0.09657447    0.03679682643147697   0.020192597     0.096430674      0.03679682126227771   Float32[0.8333078,  -0.5743816]

print("event_to_bins_logistic_coeffs = $event_to_bins_logistic_coeffs")
# event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[1.030558, 0.19321586], [1.1623001, 1.0385767], [1.1844709, 1.1859512], [0.7777289, -0.807949], [0.9798549, 0.035456475]], "sig_hail" => [[0.98084414, -0.12066075], [1.166308, 0.9986842], [0.9872798, 0.031519376], [1.0985931, 0.55506366], [0.87546086, -0.3075612]], "hail" => [[0.98223704, -0.05874168], [1.0158885, 0.1087229], [1.0228801, 0.14139068], [1.0846623, 0.34556273], [1.127743, 0.46917012]], "sig_wind_adj" => [[0.9896531, -0.11605113], [1.1852539, 1.3334796], [1.3313962, 2.2548943], [0.9650048, 0.245746], [1.0000306, 0.3686924]], "tornado_life_risk" => [[0.97598267, 0.19340746], [1.0696578, 0.8193858], [1.0577359, 0.78586924], [0.8839506, -0.27480733], [0.64814734, -1.5873486]], "tornado" => [[1.0151445, 0.2071434], [0.990645, 0.08936819], [1.0937685, 0.61661005], [0.86215496, -0.3573002], [0.9471853, -0.049062375]], "wind_adj" => [[1.0665799, 0.36155185], [1.2188917, 1.1445187], [1.0872297, 0.5649324], [1.0266879, 0.34076434], [1.036534, 0.3751357]], "sig_tornado" => [[1.003107, 0.06779992], [1.0754979, 0.41287646], [1.3813931, 2.1319306], [1.3965261, 2.2143273], [0.91918075, 0.24742183]], "wind" => [[1.052107, 0.26756564], [1.0934643, 0.45261195], [1.0603855, 0.33706936], [0.97554934, 0.09084179], [0.98142743, 0.10940261]])


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)






import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_calibrated_with_sig_gated, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction2024.regular_forecasts(HREFPrediction2024.forecasts_calibrated_with_sig_gated()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2024, 2, 28, 12)
validation_forecasts_calibrated_with_sig_gated = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_calibrated_with_sig_gated);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_calibrated_with_sig_gated[100]);

# rm("validation_forecasts_calibrated_with_sig_gated"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_calibrated_with_sig_gated; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_calibrated_with_sig_gated");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction2024.models_with_gated)
    (event_name, _, model_name) = HREFPrediction2024.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
  end
end
test_predictive_power(validation_forecasts_calibrated_with_sig_gated, X, Ys, weights)

# Same as before, because calibration was monotonic
# tornado (75293.0)                         feature 1 TORPROB:calculated:hour  fcst:calculated_prob:                   AU-PR-curve: 0.047563255
# wind (588576.0)                           feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.11646805
# wind_adj (182289.4)                       feature 3 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.06342232
# hail (264451.0)                           feature 4 HAILPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.07536129
# sig_tornado (10243.0)                     feature 5 STORPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.03307567
# sig_wind (58860.0)                        feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.016221732
# sig_wind_adj (21053.29)                   feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.01225615
# sig_hail (31850.0)                        feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.018038029
# sig_tornado_gated_by_tornado (10243.0)    feature 9 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado   AU-PR-curve: 0.030468112
# sig_wind_gated_by_wind (58860.0)          feature 10 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind     AU-PR-curve: 0.01621948
# sig_wind_adj_gated_by_wind_adj (21053.29) feature 11 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.012388945
# sig_hail_gated_by_hail (31850.0)          feature 12 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail     AU-PR-curve: 0.017981928

Metrics.reliability_curves_midpoints(20, X, Ys, map(m -> m[1], HREFPrediction2024.models_with_gated), weights, map(m -> m[3], HREFPrediction2024.models_with_gated))
# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,ŷ_sig_wind_gated_by_wind,y_sig_wind_gated_by_wind,ŷ_sig_wind_adj_gated_by_wind_adj,y_sig_wind_adj_gated_by_wind_adj,ŷ_sig_hail_gated_by_hail,y_sig_hail_gated_by_hail,
# 6.1531127e-6,6.0045345e-6,4.4935874e-5,4.6977286e-5,1.3700466e-5,1.4281125e-5,2.0477783e-5,2.0870466e-5,6.8494865e-7,8.0508244e-7,4.5083425e-6,4.6533664e-6,1.5824395e-6,1.6324826e-6,2.138175e-6,2.472806e-6,6.653776e-7,8.0506925e-7,4.4942576e-6,4.6530813e-6,1.5755261e-6,1.6309828e-6,2.1315907e-6,2.4727815e-6,
# 0.00026546858,0.00029663646,0.002290551,0.0024005529,0.0006841353,0.0007571397,0.0011119856,0.001162719,8.458081e-5,0.00015346402,0.00020227442,0.00023276251,8.8660185e-5,7.6549535e-5,0.00021447489,0.00024548653,8.457267e-5,0.00015368158,0.00020209982,0.0002334415,8.828558e-5,7.8686746e-5,0.00021444552,0.00024552655,
# 0.0007241435,0.0007362149,0.004945054,0.0047397586,0.0015168466,0.0016048265,0.0023980946,0.0025161966,0.00030457432,0.00026138377,0.00047718835,0.00050955516,0.00022885864,0.00022712679,0.00052120467,0.0005501983,0.00030460488,0.00026185418,0.0004766413,0.00050895894,0.00022747555,0.00023213439,0.00052120606,0.00055055704,
# 0.0014086519,0.0014777686,0.008167128,0.008020744,0.0026165282,0.0024921226,0.003946606,0.0039206226,0.00079093565,0.00060432125,0.000842937,0.00084891566,0.00039754307,0.00037013332,0.0009400649,0.00082231394,0.00078836095,0.00061007845,0.00084247795,0.0008493521,0.00039594792,0.00037563677,0.0009395708,0.0008243345,
# 0.0022743954,0.0023369447,0.011813817,0.01156835,0.0040380885,0.003781158,0.005770756,0.0057041226,0.0014265027,0.0012039922,0.0012869171,0.0012531572,0.00061243656,0.0006563708,0.0014382264,0.0014449642,0.0014211469,0.0011923579,0.0012862285,0.0012543306,0.00061138987,0.00066342356,0.0014364207,0.0014473302,
# 0.0032991148,0.0031599442,0.015957268,0.015869418,0.00580325,0.005719665,0.0077474723,0.007746222,0.002088601,0.0025097656,0.001804329,0.0017502705,0.0009374502,0.0008770512,0.0019374617,0.0019365001,0.002087178,0.0024984886,0.0018034052,0.0017504246,0.00093657133,0.0008881968,0.0019348813,0.0019360838,
# 0.004397003,0.0044003646,0.020614753,0.021137327,0.007969011,0.0079894075,0.0098469565,0.009742342,0.0028011894,0.002971259,0.0023979256,0.0022582745,0.0014601588,0.001398697,0.002475206,0.002400055,0.0028016237,0.0029669846,0.0023967759,0.0022608023,0.0014581167,0.0014092462,0.002473384,0.0023912704,
# 0.0057230773,0.0057523674,0.025985755,0.02598212,0.010663667,0.010659683,0.012369373,0.012067936,0.0037991686,0.003585902,0.003152089,0.0029241217,0.0020421844,0.0022604896,0.0031618634,0.00313945,0.0038001798,0.0035857887,0.00314976,0.0029278467,0.0020391028,0.0023040012,0.003161496,0.0031364467,
# 0.0075928513,0.0072833244,0.03218833,0.031517666,0.013865874,0.01379582,0.015530437,0.015330491,0.005414933,0.0046489895,0.004161406,0.0040349807,0.0026658876,0.0026277062,0.003957348,0.004065092,0.0054104985,0.004656186,0.0041574244,0.004031934,0.0026587844,0.0026807904,0.003957444,0.0040664463,
# 0.01026361,0.009662776,0.039115857,0.039486628,0.017621074,0.01765114,0.019474972,0.01945359,0.007905099,0.00800732,0.0054371282,0.0057556187,0.0034000028,0.0033810283,0.0048506153,0.0047968896,0.007863663,0.008059345,0.0054337154,0.0057353093,0.003386512,0.0034583919,0.0048511517,0.004794828,
# 0.013689927,0.013830343,0.046869006,0.0480191,0.021985603,0.022760319,0.024301428,0.024507402,0.011180207,0.010953245,0.0069900597,0.0075062234,0.004148515,0.0043761227,0.0058626155,0.006379528,0.011056643,0.010964823,0.0069888067,0.007504863,0.0041328873,0.0044267685,0.005863383,0.0063857087,
# 0.017167425,0.018398501,0.055859745,0.055579707,0.027127184,0.02716937,0.029772168,0.03001216,0.015194971,0.013734752,0.008670861,0.0085381605,0.005140524,0.0048049865,0.0070963036,0.007151415,0.014540397,0.016793694,0.008667627,0.008564158,0.0051127737,0.0048993197,0.007096665,0.0071537625,
# 0.020688688,0.021057948,0.06628032,0.065868706,0.03316254,0.03290143,0.035898324,0.036441717,0.019473804,0.021624925,0.010292352,0.010347398,0.006804207,0.006104501,0.008720822,0.008206778,0.017953511,0.021738533,0.010284814,0.010384223,0.00674694,0.0061401105,0.00872098,0.008207917,
# 0.02426165,0.025226573,0.07840475,0.07834983,0.040299043,0.040148772,0.042901717,0.04263499,0.023741381,0.02770295,0.01176248,0.012122186,0.009291704,0.009332127,0.010851891,0.010240092,0.021657351,0.025139946,0.011751908,0.012097324,0.009220322,0.009229625,0.010852688,0.010235438,
# 0.028935073,0.028913405,0.09292727,0.093550175,0.04954558,0.048543308,0.051738426,0.052434433,0.028825084,0.031171057,0.013737846,0.0140913455,0.012047821,0.011704192,0.013352082,0.014180333,0.026416557,0.027197393,0.01372902,0.014043398,0.012013897,0.0117538115,0.013353988,0.014184162,
# 0.036926188,0.03251899,0.10976821,0.1108744,0.061063915,0.060316876,0.06336136,0.06260348,0.034392405,0.03469714,0.016040726,0.015734384,0.014415448,0.016678726,0.016306508,0.016113844,0.032183103,0.031557653,0.016037563,0.015708987,0.014393535,0.017267672,0.016307851,0.016128818,
# 0.050845355,0.048398882,0.12963742,0.13075574,0.074829385,0.07659403,0.07952274,0.07946183,0.040369272,0.034914173,0.01870779,0.017730925,0.01613597,0.017199518,0.02005569,0.019921625,0.038158704,0.037322007,0.018707719,0.01773912,0.016125347,0.01740088,0.020055834,0.019915525,
# 0.06879845,0.073836155,0.15680626,0.15682542,0.09176409,0.103652835,0.102680996,0.10188601,0.04932659,0.04450005,0.022033101,0.023077397,0.01851127,0.017795611,0.02527401,0.025431696,0.045600604,0.04233571,0.022033175,0.02307988,0.018512636,0.0180935,0.025274096,0.025433697,
# 0.093195826,0.10680757,0.20118624,0.199661,0.11701216,0.12639023,0.14015123,0.13652179,0.06467039,0.06496779,0.02732545,0.030587189,0.022415793,0.02518377,0.033233356,0.036842506,0.059167758,0.058898456,0.027325466,0.030587478,0.022418238,0.02552011,0.03323331,0.036842506,
# 0.15691693,0.15076944,0.30818096,0.30750546,0.18210845,0.16728972,0.22729546,0.23269866,0.09690755,0.11453006,0.0434624,0.043758858,0.03227213,0.029773861,0.053115238,0.050361976,0.09094789,0.09411095,0.0434624,0.043758858,0.03229444,0.029928342,0.053109474,0.050361976,


event_to_bins = Dict{String, Vector{Float32}}(
  "tornado"      => [0.0012153604,  0.004538168,  0.011742671,  0.023059275,  0.049979284,  1.0],
  "wind"         => [0.0078376075,  0.020091131,  0.037815828,  0.065400586,  0.11733462,   1.0],
  "wind_adj"     => [0.0025728762,  0.008021789,  0.016496103,  0.030200316,  0.058084033,  1.0],
  "hail"         => [0.0033965781,  0.00951148,   0.0200999,    0.037743744,  0.07645232,   1.0],
  "sig_tornado"  => [0.00080238597, 0.0032272756, 0.007747574,  0.0136314705, 0.022335978,  1.0],
  "sig_wind"     => [0.0006871624,  0.0022018508, 0.0047563836, 0.008192127,  0.015201166,  1.0],
  "sig_wind_adj" => [0.00042257016, 0.0012706288, 0.002259613,  0.003675646,  0.0054352577, 1.0],
  "sig_hail"     => [0.00072055974, 0.0021597594, 0.00463198,   0.009187652,  0.01903059,   1.0],
)
event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
  "tornado"      => [[1.0710595, 0.5394862],  [0.89414734, -0.5616081],   [1.108447,  0.50613326],  [0.83939207, -0.5758451],  [1.2171175,  0.7334359]],
  "wind"         => [[1.0940193, 0.44082266], [1.1078424,  0.48961264],   [1.0705312, 0.36004978],  [1.0221666,  0.22469698],  [0.9163128,  -0.022937609]],
  "wind_adj"     => [[1.1026261, 0.554337],   [1.1907648,  0.9671046],    [1.1658177, 0.862035],    [1.0740721,  0.5209931],   [0.9590977,  0.1968526]],
  "hail"         => [[1.0600259, 0.43388337], [0.96319646, -0.097637914], [1.0792756, 0.39443073],  [0.9629313,  -0.01864577], [1.0167043,  0.13172606]],
  "sig_tornado"  => [[1.0426707, 0.21755064], [1.1516033,  0.7787872],    [1.6196959, 3.2147853],   [1.4218588,  2.3287997],   [1.032921,   0.7553395]],
  "sig_wind"     => [[1.0894886, 0.7778136],  [1.0595194,  0.49330923],   [1.3312615, 2.0373042],   [0.87163,    -0.28045976], [0.66988313, -1.1392391]],
  "sig_wind_adj" => [[1.0503622, 0.2618211],  [1.5049535,  3.5953336],    [1.3739139, 2.7983332],   [2.022486,   6.61417],     [0.99839044, 1.0990105]],
  "sig_hail"     => [[1.0996535, 0.9160373],  [1.0066487,  0.20982021],   [0.9350299, -0.20348155], [0.9644414,  -0.02964372], [0.8333078,  -0.5743816]],
)

function plot_calibration_curves(model_names, event_to_bins, event_to_bins_logistic_coeffs)
  σ(x)     = 1.0f0 / (1.0f0 + exp(-x))
  logit(p) = log(p / (one(p) - p))
  ratio_between(x, lo, hi) = (x - lo) / (hi - lo)
  predict_one(coeffs, ŷ_in) = σ(coeffs[1]*logit(ŷ_in) + coeffs[2])

  for model_name in model_names
    print("ŷ_in_$model_name,ŷ_out_$model_name,")
  end
  println()

  for ŷ_in in (collect(0:0.01:1).^2)
    for model_name in model_names
      bin_maxes            = event_to_bins[model_name]
      bins_logistic_coeffs = event_to_bins_logistic_coeffs[model_name]
      @assert length(bin_maxes) == length(bins_logistic_coeffs) + 1

      if ŷ_in <= bin_maxes[1]
        # Bin 1-2 predictor only
        ŷ_out = predict_one(bins_logistic_coeffs[1], ŷ_in)
      elseif ŷ_in > bin_maxes[length(bin_maxes) - 1]
        # Bin 5-6 predictor only
        ŷ_out = predict_one(bins_logistic_coeffs[length(bins_logistic_coeffs)], ŷ_in)
      else
        # Overlapping bins
        higher_bin_i = findfirst(bin_max -> ŷ_in <= bin_max, bin_maxes)
        lower_bin_i  = higher_bin_i - 1
        coeffs_higher_bin = bins_logistic_coeffs[higher_bin_i]
        coeffs_lower_bin  = bins_logistic_coeffs[lower_bin_i]

        # Bin 1-2 and 2-3 predictors
        ratio = ratio_between(ŷ_in, bin_maxes[lower_bin_i], bin_maxes[higher_bin_i])
        ŷ_out = ratio*predict_one(coeffs_higher_bin, ŷ_in) + (1f0 - ratio)*predict_one(coeffs_lower_bin, ŷ_in)
      end
      print("$(Float32(ŷ_in)),$(Float32(ŷ_out)),")
    end
    println()
  end

  ()
end

plot_calibration_curves(map(m -> m[1], HREFPrediction2024.models), event_to_bins, event_to_bins_logistic_coeffs)
