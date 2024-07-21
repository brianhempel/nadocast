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

# for prediction_i in 1:length(HREFPrediction2024.models)
for prediction_i in [3,7] # wind_adj and sig_wind_adj
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




###################################


# tornado (101642.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.04458257
# tornado (101642.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.044846594 ***best tor***
# tornado (101642.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.044660322
# tornado (101642.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.043800037
# tornado (101642.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.04156998
# tornado (101642.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.037759554
# tornado (101642.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.03102177
# tornado (101642.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010589085
# tornado (101642.0) feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010687708
# tornado (101642.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010656607
# tornado (101642.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.010505419
# tornado (101642.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.010120345
# tornado (101642.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.009323221
# tornado (101642.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007895621
# tornado (101642.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010589085
# tornado (101642.0) feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010687708
# tornado (101642.0) feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010656607
# tornado (101642.0) feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.010505419
# tornado (101642.0) feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.010120345
# tornado (101642.0) feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.009323221
# tornado (101642.0) feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007895621
# tornado (101642.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.007297086
# tornado (101642.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0073290663
# tornado (101642.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.007298768
# tornado (101642.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0071581146
# tornado (101642.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0067913895
# tornado (101642.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.006192091
# tornado (101642.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0051869852
# tornado (101642.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.032265045
# tornado (101642.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.032318044
# tornado (101642.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.032005627
# tornado (101642.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03114941
# tornado (101642.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.029329598
# tornado (101642.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02635234
# tornado (101642.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021628633
# tornado (101642.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010221196
# tornado (101642.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010250007
# tornado (101642.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010174918
# tornado (101642.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009935485
# tornado (101642.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0094199
# tornado (101642.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008609302
# tornado (101642.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007291271
# tornado (101642.0) feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010221196
# tornado (101642.0) feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010250007
# tornado (101642.0) feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.010174918
# tornado (101642.0) feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009935485
# tornado (101642.0) feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0094199
# tornado (101642.0) feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008609302
# tornado (101642.0) feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007291271

# wind (874384.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.051420897
# wind (874384.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.051494535
# wind (874384.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.051235344
# wind (874384.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.050470702
# wind (874384.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.04875791
# wind (874384.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.04557444
# wind (874384.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.03978436
# wind (874384.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.12782907
# wind (874384.0) feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.12906836
# wind (874384.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.12925549 ***best wind***
# wind (874384.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.12847693
# wind (874384.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.12560302
# wind (874384.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.11934432
# wind (874384.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.10668006
# wind (874384.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.12782907
# wind (874384.0) feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.12906836
# wind (874384.0) feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.12925549
# wind (874384.0) feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.12847693
# wind (874384.0) feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.12560302
# wind (874384.0) feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.11934432
# wind (874384.0) feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.10668006
# wind (874384.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.037065726
# wind (874384.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.037033882
# wind (874384.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.03675523
# wind (874384.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03605872
# wind (874384.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.034615297
# wind (874384.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.032338165
# wind (874384.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.028555334
# wind (874384.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.046982918
# wind (874384.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.04694493
# wind (874384.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.046603665
# wind (874384.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.045765713
# wind (874384.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.044001147
# wind (874384.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.04083301
# wind (874384.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03541002
# wind (874384.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.08659278
# wind (874384.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.08715066
# wind (874384.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.087004766
# wind (874384.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0860994
# wind (874384.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0836549
# wind (874384.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.07904909
# wind (874384.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.070160866
# wind (874384.0) feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.08659278
# wind (874384.0) feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.08715066
# wind (874384.0) feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.087004766
# wind (874384.0) feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0860994
# wind (874384.0) feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0836549
# wind (874384.0) feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.07904909
# wind (874384.0) feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.070160866

# wind_adj (278174.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.012910408
# wind_adj (278174.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.012873274
# wind_adj (278174.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.012752202
# wind_adj (278174.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.012476351
# wind_adj (278174.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.011908886
# wind_adj (278174.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.010995837
# wind_adj (278174.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.009452826
# wind_adj (278174.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03774708
# wind_adj (278174.0) feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.03803853
# wind_adj (278174.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.037904996
# wind_adj (278174.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03756368
# wind_adj (278174.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.036224145
# wind_adj (278174.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03371297
# wind_adj (278174.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.028613193
# wind_adj (278174.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03774708
# wind_adj (278174.0) feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.03803853
# wind_adj (278174.0) feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.037904996
# wind_adj (278174.0) feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03756368
# wind_adj (278174.0) feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.036224145
# wind_adj (278174.0) feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03371297
# wind_adj (278174.0) feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.028613193
# wind_adj (278174.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.020148933
# wind_adj (278174.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.020234473
# wind_adj (278174.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.020149669
# wind_adj (278174.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.019857861
# wind_adj (278174.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.019138655
# wind_adj (278174.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.017918186
# wind_adj (278174.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015787683
# wind_adj (278174.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.011943924
# wind_adj (278174.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.011905571
# wind_adj (278174.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.011794113
# wind_adj (278174.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01155057
# wind_adj (278174.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011068058
# wind_adj (278174.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.010264817
# wind_adj (278174.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00889574
# wind_adj (278174.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0546422
# wind_adj (278174.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.05526082
# wind_adj (278174.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05528199
# wind_adj (278174.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.054841246
# wind_adj (278174.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.053170532
# wind_adj (278174.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.050040647
# wind_adj (278174.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04364945
# wind_adj (278174.0) feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0546422
# wind_adj (278174.0) feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.05526082
# wind_adj (278174.0) feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05528199
# wind_adj (278174.0) feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.054841246
# wind_adj (278174.0) feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.053170532
# wind_adj (278174.0) feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.050040647
# wind_adj (278174.0) feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04364945

# hail (405123.0)            feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02218699
# hail (405123.0)            feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022128839
# hail (405123.0)            feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021926094
# hail (405123.0)            feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.021447338
# hail (405123.0)            feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020441411
# hail (405123.0)            feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.018837493
# hail (405123.0)            feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016192595
# hail (405123.0)            feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.022103406
# hail (405123.0)            feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022114059
# hail (405123.0)            feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021948108
# hail (405123.0)            feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.02155953
# hail (405123.0)            feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020670062
# hail (405123.0)            feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.019258074
# hail (405123.0)            feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016872102
# hail (405123.0)            feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.022103406
# hail (405123.0)            feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.022114059
# hail (405123.0)            feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021948108
# hail (405123.0)            feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.02155953
# hail (405123.0)            feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.020670062
# hail (405123.0)            feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.019258074
# hail (405123.0)            feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016872102
# hail (405123.0)            feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.09149645
# hail (405123.0)            feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.092647225
# hail (405123.0)            feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.09272799
# hail (405123.0)            feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.091651455
# hail (405123.0)            feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.08822686
# hail (405123.0)            feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0815599
# hail (405123.0)            feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.06952726
# hail (405123.0)            feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.019043827
# hail (405123.0)            feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.018979628
# hail (405123.0)            feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0188036
# hail (405123.0)            feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.018415714
# hail (405123.0)            feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.017646324
# hail (405123.0)            feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01637637
# hail (405123.0)            feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014250935
# hail (405123.0)            feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027099736
# hail (405123.0)            feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02710439
# hail (405123.0)            feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.02690581
# hail (405123.0)            feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.026428387
# hail (405123.0)            feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.025340807
# hail (405123.0)            feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.023568414
# hail (405123.0)            feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02054999
# hail (405123.0)            feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027099736
# hail (405123.0)            feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02710439
# hail (405123.0)            feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.02690581
# hail (405123.0)            feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.026428387
# hail (405123.0)            feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.025340807
# hail (405123.0)            feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.023568414
# hail (405123.0)            feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02054999

# sig_tornado (13792.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.027445951
# sig_tornado (13792.0)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.028389782
# sig_tornado (13792.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.028972166
# sig_tornado (13792.0)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.029353565
# sig_tornado (13792.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.028786186
# sig_tornado (13792.0)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.027141767
# sig_tornado (13792.0)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021901364
# sig_tornado (13792.0)      feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004057158
# sig_tornado (13792.0)      feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004160798
# sig_tornado (13792.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0041581015
# sig_tornado (13792.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0040954268
# sig_tornado (13792.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004004421
# sig_tornado (13792.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0038498708
# sig_tornado (13792.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0034231232
# sig_tornado (13792.0)      feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004057158
# sig_tornado (13792.0)      feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004160798
# sig_tornado (13792.0)      feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0041581015
# sig_tornado (13792.0)      feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0040954268
# sig_tornado (13792.0)      feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004004421
# sig_tornado (13792.0)      feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0038498708
# sig_tornado (13792.0)      feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0034231232
# sig_tornado (13792.0)      feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0025716498
# sig_tornado (13792.0)      feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0026195375
# sig_tornado (13792.0)      feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0026439372
# sig_tornado (13792.0)      feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0026329819
# sig_tornado (13792.0)      feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0025185533
# sig_tornado (13792.0)      feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022743498
# sig_tornado (13792.0)      feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0017834442
# sig_tornado (13792.0)      feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.031367242
# sig_tornado (13792.0)      feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.031650614
# sig_tornado (13792.0)      feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.031269126
# sig_tornado (13792.0)      feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.030035706
# sig_tornado (13792.0)      feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.027383663
# sig_tornado (13792.0)      feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.023388445
# sig_tornado (13792.0)      feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.017623857
# sig_tornado (13792.0)      feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027873488
# sig_tornado (13792.0)      feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0027932958
# sig_tornado (13792.0)      feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0027619095
# sig_tornado (13792.0)      feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0026840423
# sig_tornado (13792.0)      feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0025364668
# sig_tornado (13792.0)      feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022947108
# sig_tornado (13792.0)      feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018972879
# sig_tornado (13792.0)      feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027873488
# sig_tornado (13792.0)      feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0027932958
# sig_tornado (13792.0)      feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0027619095
# sig_tornado (13792.0)      feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0026840423
# sig_tornado (13792.0)      feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0025364668
# sig_tornado (13792.0)      feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022947108
# sig_tornado (13792.0)      feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018972879

# sig_wind (84250.0)         feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0076630577
# sig_wind (84250.0)         feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.007648342
# sig_wind (84250.0)         feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00757643
# sig_wind (84250.0)         feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0074103763
# sig_wind (84250.0)         feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007064466
# sig_wind (84250.0)         feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0064972215
# sig_wind (84250.0)         feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005527759
# sig_wind (84250.0)         feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015860096
# sig_wind (84250.0)         feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016017107
# sig_wind (84250.0)         feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.015977671
# sig_wind (84250.0)         feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015863271
# sig_wind (84250.0)         feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015382411
# sig_wind (84250.0)         feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.014461649
# sig_wind (84250.0)         feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012575701
# sig_wind (84250.0)         feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015860096
# sig_wind (84250.0)         feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016017107
# sig_wind (84250.0)         feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.015977671
# sig_wind (84250.0)         feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015863271
# sig_wind (84250.0)         feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015382411
# sig_wind (84250.0)         feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.014461649
# sig_wind (84250.0)         feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012575701
# sig_wind (84250.0)         feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.006867611
# sig_wind (84250.0)         feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.006885819
# sig_wind (84250.0)         feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0068374043
# sig_wind (84250.0)         feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.006700213
# sig_wind (84250.0)         feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00639286
# sig_wind (84250.0)         feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005876067
# sig_wind (84250.0)         feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0050469614
# sig_wind (84250.0)         feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.006835445
# sig_wind (84250.0)         feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0068151625
# sig_wind (84250.0)         feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.006752512
# sig_wind (84250.0)         feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0066145915
# sig_wind (84250.0)         feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.006334261
# sig_wind (84250.0)         feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005851701
# sig_wind (84250.0)         feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0050128847
# sig_wind (84250.0)         feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.025028422
# sig_wind (84250.0)         feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02541912
# sig_wind (84250.0)         feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.025494266
# sig_wind (84250.0)         feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.025370913
# sig_wind (84250.0)         feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.024844557
# sig_wind (84250.0)         feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02380682
# sig_wind (84250.0)         feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021160657
# sig_wind (84250.0)         feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.025028422
# sig_wind (84250.0)         feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02541912
# sig_wind (84250.0)         feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.025494266
# sig_wind (84250.0)         feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.025370913
# sig_wind (84250.0)         feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.024844557
# sig_wind (84250.0)         feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02380682
# sig_wind (84250.0)         feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.021160657

# sig_wind_adj (31404.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0016301085
# sig_wind_adj (31404.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0016249514
# sig_wind_adj (31404.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0016100901
# sig_wind_adj (31404.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0015785949
# sig_wind_adj (31404.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0015126985
# sig_wind_adj (31404.0)     feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.001404068
# sig_wind_adj (31404.0)     feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0012119444
# sig_wind_adj (31404.0)     feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005745996
# sig_wind_adj (31404.0)     feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0058170967
# sig_wind_adj (31404.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0058026724
# sig_wind_adj (31404.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0058195647
# sig_wind_adj (31404.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005686624
# sig_wind_adj (31404.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0053757997
# sig_wind_adj (31404.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0045962296
# sig_wind_adj (31404.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005745996
# sig_wind_adj (31404.0)     feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0058170967
# sig_wind_adj (31404.0)     feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0058026724
# sig_wind_adj (31404.0)     feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0058195647
# sig_wind_adj (31404.0)     feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005686624
# sig_wind_adj (31404.0)     feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0053757997
# sig_wind_adj (31404.0)     feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0045962296
# sig_wind_adj (31404.0)     feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032326577
# sig_wind_adj (31404.0)     feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003250134
# sig_wind_adj (31404.0)     feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0032337126
# sig_wind_adj (31404.0)     feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.003179589
# sig_wind_adj (31404.0)     feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0030445037
# sig_wind_adj (31404.0)     feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0027965766
# sig_wind_adj (31404.0)     feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0023894587
# sig_wind_adj (31404.0)     feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017000715
# sig_wind_adj (31404.0)     feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0016952861
# sig_wind_adj (31404.0)     feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0016812349
# sig_wind_adj (31404.0)     feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.001651115
# sig_wind_adj (31404.0)     feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0015912488
# sig_wind_adj (31404.0)     feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0014923983
# sig_wind_adj (31404.0)     feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0013108619
# sig_wind_adj (31404.0)     feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012106531
# sig_wind_adj (31404.0)     feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012422113
# sig_wind_adj (31404.0)     feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012584977
# sig_wind_adj (31404.0)     feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012735926
# sig_wind_adj (31404.0)     feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.012779744
# sig_wind_adj (31404.0)     feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012806546
# sig_wind_adj (31404.0)     feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011711587
# sig_wind_adj (31404.0)     feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012106531
# sig_wind_adj (31404.0)     feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012422113
# sig_wind_adj (31404.0)     feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012584977
# sig_wind_adj (31404.0)     feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012735926
# sig_wind_adj (31404.0)     feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.012779744
# sig_wind_adj (31404.0)     feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012806546
# sig_wind_adj (31404.0)     feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011711587

# sig_hail (51908.0)         feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0041780425
# sig_hail (51908.0)         feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0041632997
# sig_hail (51908.0)         feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004118933
# sig_hail (51908.0)         feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004012486
# sig_hail (51908.0)         feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0037959367
# sig_hail (51908.0)         feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0034512295
# sig_hail (51908.0)         feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0028937378
# sig_hail (51908.0)         feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0030991847
# sig_hail (51908.0)         feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0030940569
# sig_hail (51908.0)         feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0030659959
# sig_hail (51908.0)         feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0030002496
# sig_hail (51908.0)         feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0028635242
# sig_hail (51908.0)         feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026494649
# sig_hail (51908.0)         feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0022984515
# sig_hail (51908.0)         feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0030991847
# sig_hail (51908.0)         feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0030940569
# sig_hail (51908.0)         feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0030659959
# sig_hail (51908.0)         feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0030002496
# sig_hail (51908.0)         feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0028635242
# sig_hail (51908.0)         feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026494649
# sig_hail (51908.0)         feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0022984515
# sig_hail (51908.0)         feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.022759821
# sig_hail (51908.0)         feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.023222687
# sig_hail (51908.0)         feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.023420911
# sig_hail (51908.0)         feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.023353716
# sig_hail (51908.0)         feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.022783604
# sig_hail (51908.0)         feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0215332
# sig_hail (51908.0)         feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.019015664
# sig_hail (51908.0)         feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0035185488
# sig_hail (51908.0)         feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003505364
# sig_hail (51908.0)         feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0034689524
# sig_hail (51908.0)         feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0033883622
# sig_hail (51908.0)         feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0032329846
# sig_hail (51908.0)         feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0029784683
# sig_hail (51908.0)         feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0025565021
# sig_hail (51908.0)         feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0055913213
# sig_hail (51908.0)         feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0056044515
# sig_hail (51908.0)         feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0055645113
# sig_hail (51908.0)         feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.005452066
# sig_hail (51908.0)         feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005188876
# sig_hail (51908.0)         feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0047719576
# sig_hail (51908.0)         feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004059665
# sig_hail (51908.0)         feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0055913213
# sig_hail (51908.0)         feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0056044515
# sig_hail (51908.0)         feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0055645113
# sig_hail (51908.0)         feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.005452066
# sig_hail (51908.0)         feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005188876
# sig_hail (51908.0)         feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0047719576
# sig_hail (51908.0)         feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004059665

# tornado_life_risk (3093.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0064463015
# tornado_life_risk (3093.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0066459044
# tornado_life_risk (3093.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0067401244
# tornado_life_risk (3093.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0067231283
# tornado_life_risk (3093.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0062892623
# tornado_life_risk (3093.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0054824594
# tornado_life_risk (3093.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0039054176
# tornado_life_risk (3093.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005027309
# tornado_life_risk (3093.0) feature 9  WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005034586
# tornado_life_risk (3093.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00049928884
# tornado_life_risk (3093.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.00048760482
# tornado_life_risk (3093.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0004645387
# tornado_life_risk (3093.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00042451956
# tornado_life_risk (3093.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00036548005
# tornado_life_risk (3093.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005027309
# tornado_life_risk (3093.0) feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005034586
# tornado_life_risk (3093.0) feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00049928884
# tornado_life_risk (3093.0) feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.00048760482
# tornado_life_risk (3093.0) feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0004645387
# tornado_life_risk (3093.0) feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00042451956
# tornado_life_risk (3093.0) feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00036548005
# tornado_life_risk (3093.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0006553588
# tornado_life_risk (3093.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0006675654
# tornado_life_risk (3093.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0006722062
# tornado_life_risk (3093.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.000663792
# tornado_life_risk (3093.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0006222865
# tornado_life_risk (3093.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0005455593
# tornado_life_risk (3093.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00041630407
# tornado_life_risk (3093.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010123185
# tornado_life_risk (3093.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010063962
# tornado_life_risk (3093.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00975427
# tornado_life_risk (3093.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009048249
# tornado_life_risk (3093.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0077871797
# tornado_life_risk (3093.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0061529996
# tornado_life_risk (3093.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0042618965
# tornado_life_risk (3093.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005923344
# tornado_life_risk (3093.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005936034
# tornado_life_risk (3093.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00058817817
# tornado_life_risk (3093.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0005728559
# tornado_life_risk (3093.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00054026407
# tornado_life_risk (3093.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00048483734
# tornado_life_risk (3093.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00039651856
# tornado_life_risk (3093.0) feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0005923344
# tornado_life_risk (3093.0) feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0005936034
# tornado_life_risk (3093.0) feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00058817817
# tornado_life_risk (3093.0) feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0005728559
# tornado_life_risk (3093.0) feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00054026407
# tornado_life_risk (3093.0) feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00048483734
# tornado_life_risk (3093.0) feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00039651856





println("Determining best blur radii to maximize area under precision-recall curve")

bests = []
# for prediction_i in 1:length(HREFPrediction2024.models)
for prediction_i in [3,7] # wind_adj and sig_wind_adj
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
# ["WINDPROB:calculated:hour fcst:calculated_prob:", "WINDPROB:calculated:hour fcst:calculated_prob:15mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:25mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:35mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:50mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:70mi mean", "WINDPROB:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour fcst::"]

# blur_radius_f1  blur_radius_f36 AU_PR_wind_adj
# 0       0       0.037747085
# 0       15      0.037929885
# 0       25      0.03794523
# 0       35      0.037949953
# 0       50      0.037749697
# 0       70      0.037359428
# 0       100     0.036497302
# 15      0       0.037931252
# 15      15      0.038038526
# 15      25      0.038038637
# 15      35      0.03804059
# 15      50      0.03784269
# 15      70      0.03747948
# 15      100     0.03663646
# 25      0       0.037840366
# 25      15      0.03792976
# 25      25      0.037904996
# 25      35      0.037889145
# 25      50      0.037675858
# 25      70      0.037310097
# 25      100     0.0364737
# 35      0       0.03758057
# 35      15      0.037661035
# 35      25      0.03761482
# 35      35      0.037563678
# 35      50      0.037307113
# 35      70      0.036905263
# 35      100     0.036055923
# 50      0       0.036686357
# 50      15      0.036755495
# 50      25      0.036680885
# 50      35      0.036574677
# 50      50      0.036224153
# 50      70      0.035712633
# 50      100     0.03478165
# 70      0       0.0351524
# 70      15      0.035202067
# 70      25      0.035087977
# 70      35      0.03491665
# 70      50      0.034426365
# 70      70      0.033712987
# 70      100     0.03250789
# 100     0       0.032526318
# 100     15      0.032541085
# 100     25      0.03237593
# 100     35      0.032091223
# 100     50      0.031386252
# 100     70      0.03034129
# 100     100     0.028613197

# Best wind_adj: 15       35      0.03804059      WINDPROB:calculated:hour fcst:calculated_prob:15mi mean WINDPROB:calculated:hour fcst:calculated_prob:35mi mean

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
# ["SWINDPRO:calculated:hour fcst:calculated_prob:", "SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean", "SWINDPRO:calculated:hour f
# cst:calculated_prob:35mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean", "SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean", "forecast_hour:calculated:hour f
# cst::"]

# blur_radius_f2    blur_radius_f35 AU_PR_sig_wind_adj
# 0       0       0.012106529
# 0       15      0.012255298
# 0       25      0.012395859
# 0       35      0.012726272
# 0       50      0.013015197
# 0       70      0.013063088
# 0       100     0.0128721045
# 15      0       0.012308566
# 15      15      0.012422113
# 15      25      0.01252641
# 15      35      0.012832229
# 15      50      0.013423114
# 15      70      0.013544044
# 15      100     0.013413029
# 25      0       0.012398286
# 25      15      0.012501342
# 25      25      0.012584977
# 25      35      0.012804719
# 25      50      0.013514677
# 25      70      0.01369382
# 25      100     0.01357985
# 35      0       0.012439277
# 35      15      0.0125298565
# 35      25      0.012593365
# 35      35      0.012735918
# 35      50      0.0133428695
# 35      70      0.013598547
# 35      100     0.013512778
# 50      0       0.012398974
# 50      15      0.012480889
# 50      25      0.012516248
# 50      35      0.012585066
# 50      50      0.01277974
# 50      70      0.013270556
# 50      100     0.013225817
# 70      0       0.01220589
# 70      15      0.012297878
# 70      25      0.012312948
# 70      35      0.012320729
# 70      50      0.012320369
# 70      70      0.012806544
# 70      100     0.0127709275
# 100     0       0.011642084
# 100     15      0.011759886
# 100     25      0.011779914
# 100     35      0.011748745
# 100     50      0.011620431
# 100     70      0.011500985
# 100     100     0.011711602

# Best sig_wind_adj: 25   70      0.01369382      SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean

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
# wind_adj          15                  35                   0.03804059
# hail              15                  25                   0.09276263
# sig_tornado       15                  15                   0.031650614
# sig_wind          15                  70                   0.026028465
# sig_wind_adj      25                  70                   0.01369382
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
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100])

# rm("validation_forecasts_blurred"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_blurred");

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
# event_name   AU_PR
# tornado      0.047551293
# wind         0.1164659
# wind_adj     0.06341821
# hail         0.07535621
# sig_tornado  0.033054337
# sig_wind     0.016219445
# sig_wind_adj 0.012251711
# sig_hail     0.018009394

# ACTUAL:
# tornado (75293.0)      feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.047563255
# wind (588576.0)        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.11646806
# wind_adj (182289.0)    feature 3 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.06342232
# hail (264451.0)        feature 4 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.0753613
# sig_tornado (10243.0)  feature 5 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.033075683
# sig_wind (58860.0)     feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016221734
# sig_wind_adj (21053.0) feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.012256149
# sig_hail (31850.0)     feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.01803805

# oh, i used the interpolated version, above is stairstep, below is interpolated:

# tornado (75293.0)       feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.047551293
# wind (588576.0)         feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.1164659
# wind_adj (182289.4)     feature 3 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.06341821
# hail (264451.0)         feature 4 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.07535621
# sig_tornado (10243.0)   feature 5 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.033054337
# sig_wind (58860.0)      feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016219445
# sig_wind_adj (21053.29) feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.012251711
# sig_hail (31850.0)      feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.018009394


Metrics.reliability_curves_midpoints(20, X, Ys, map(m -> m[1], HREFPrediction2024.models_with_gated), weights, map(m -> m[3], HREFPrediction2024.models_with_gated))

# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,
# 7.5513485e-6,6.0045345e-6,6.05324e-5,4.6977286e-5,2.0162386e-5,1.4281072e-5,2.2600307e-5,2.0870466e-5,9.4585124e-7,8.0508244e-7,5.4054067e-6,4.6533664e-6,2.1874018e-6,1.6324826e-6,2.6234475e-6,2.472806e-6,
# 0.0002759688,0.00029663634,0.002572171,0.0024005529,0.0008123714,0.00075714174,0.0010830434,0.0011627187,0.0001004696,0.0001534642,0.00019848593,0.00023276222,0.00010773685,7.654957e-5,0.00019956443,0.00024548642,
# 0.00070582196,0.0007362142,0.005206293,0.004739763,0.001676029,0.0016048274,0.0022396073,0.0025161982,0.00034350556,0.00026138325,0.00043737135,0.00050955504,0.00026638818,0.0002271265,0.00044886154,0.0005501983,
# 0.001312134,0.0014777698,0.008246645,0.008020731,0.0027614331,0.0024921221,0.0035844825,0.0039206203,0.0008669868,0.00060432125,0.0007403602,0.0008489142,0.0004532406,0.00037013306,0.000770035,0.00082231394,
# 0.0020645992,0.0023369417,0.011589878,0.01156835,0.0041484847,0.0037811517,0.005167507,0.0057041175,0.0015770886,0.0012039922,0.0011114064,0.0012531598,0.0006746584,0.0006563708,0.001156209,0.0014449654,
# 0.003033589,0.003159947,0.015278767,0.015869418,0.00578446,0.0057196687,0.0069779386,0.0077462215,0.0023119487,0.0025097656,0.0015572422,0.0017502714,0.0009245431,0.0008770526,0.0015706583,0.001936498,
# 0.0042863335,0.0044003646,0.019306187,0.021137327,0.007647419,0.0079894075,0.009092577,0.009742357,0.003074751,0.002971259,0.0020984514,0.002258271,0.0012093204,0.0013986935,0.0020691846,0.002400055,
# 0.0058559524,0.005752358,0.02378077,0.02598212,0.009769026,0.010659683,0.01163889,0.012067925,0.0040972787,0.003585902,0.0027622299,0.0029241245,0.0014834516,0.0022604896,0.0026626126,0.00313945,
# 0.007847848,0.0072833244,0.028895602,0.031517666,0.012182583,0.013795879,0.014643785,0.015330546,0.0054462543,0.0046490314,0.0035299824,0.0040349755,0.0017661299,0.0026277062,0.0033442255,0.0040651024,
# 0.010300395,0.009662776,0.034638915,0.039486628,0.014940399,0.017651044,0.018117992,0.019453503,0.006965963,0.008007195,0.0043343515,0.0057556187,0.002086982,0.0033810283,0.0041533047,0.0047968756,
# 0.013077056,0.013830291,0.041009862,0.048019182,0.018034624,0.022760319,0.022105578,0.024507402,0.008515468,0.010953245,0.005158326,0.0075062234,0.0024232774,0.0043761227,0.0050945855,0.006379554,
# 0.016049711,0.018398592,0.048314955,0.0555796,0.021612283,0.02716937,0.026729798,0.03001223,0.0103019355,0.013734752,0.0060825925,0.0085381605,0.0028047832,0.0048049865,0.006205634,0.007151382,
# 0.01949945,0.021057948,0.056925524,0.065868706,0.02590361,0.03290143,0.03221182,0.036441617,0.012166888,0.021624925,0.007161004,0.010347398,0.0032616688,0.006104501,0.0076402663,0.008206778,
# 0.023816045,0.025226573,0.06711332,0.07834983,0.031068433,0.040148772,0.038980186,0.04263499,0.013995491,0.02770295,0.00841677,0.012122186,0.0037450427,0.009332127,0.009472606,0.010240092,
# 0.029367073,0.028913405,0.07932666,0.093550175,0.037427608,0.048543055,0.047500946,0.052434433,0.016041039,0.031171057,0.009941696,0.0140913455,0.0042329347,0.011704192,0.011659375,0.0141804535,
# 0.03716632,0.03251899,0.0942802,0.1108744,0.045559887,0.060317267,0.05849034,0.06260348,0.018506147,0.03469714,0.011895192,0.015734471,0.0047439705,0.016678726,0.014425597,0.01611369,
# 0.04761403,0.048398882,0.11351546,0.13075574,0.05591579,0.07659403,0.07329336,0.07946183,0.021764362,0.034914173,0.01463568,0.017730923,0.0053336644,0.017199518,0.018365953,0.019921625,
# 0.060449388,0.073836155,0.14057702,0.15682542,0.069454566,0.103652835,0.09434363,0.10188601,0.026710223,0.04450005,0.018698562,0.023077209,0.0061947466,0.017795611,0.024291037,0.025431696,
# 0.07780886,0.10680757,0.18554574,0.199661,0.09012851,0.12639023,0.12856436,0.13652179,0.03497311,0.06496779,0.02583789,0.030587189,0.007523856,0.02518377,0.033750333,0.036842506,
# 0.12069207,0.15076944,0.2983427,0.30750546,0.14601271,0.16728972,0.20890835,0.23269866,0.052649133,0.11453006,0.05291693,0.043758858,0.010922647,0.029773861,0.059497926,0.050361976,



# To make HREF-only day 2 predictions, we need to calibrate the hourlies

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

# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# tornado      1.9433767699185833e-5  2.0002769098688164e-5  6.06803381293335e8   0.0012153604
# tornado      0.0025180521817733475  0.0023336320664977006  4.683162616125584e6  0.004538168
# tornado      0.006885780923916211   0.007216856966735946   1.7126045786351562e6 0.011742671
# tornado      0.017711000447631936   0.0162420525903571     665834.9625293612    0.023059275
# tornado      0.03148117549867956    0.03282356196542077    374601.9258365035    0.049979284
# tornado      0.09444591873851253    0.0749646351119608     124836.84526234865   1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# wind         0.00015177399412438905 0.00017351229729035874 6.012902948704313e8  0.0078376075
# wind         0.012875463196918378   0.012574780606948555   7.087850197094262e6  0.020091131
# wind         0.030331663485577545   0.027369138199619832   3.008740676535964e6  0.037815828
# wind         0.056927844297067896   0.04913733249491495    1.6030762623867393e6 0.065400586
# wind         0.1005090084579039     0.08568750899908971    907976.2785154581    0.11733462
# wind         0.1956263260259242     0.18119557688344326    466483.9331075549    1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# wind_adj     4.6221927747441394e-5  5.3810935916400845e-5  6.038882208607397e8  0.0025728762
# wind_adj     0.004269836290534732   0.004493410538482293   6.537115807275176e6  0.008021789
# wind_adj     0.012792117490484013   0.011360842451636058   2.1820324492643476e6 0.016496103
# wind_adj     0.027767884050310394   0.021957555261364864   1.0052048863343596e6 0.030200316
# wind_adj     0.05347934980557008    0.040652544908575734   521930.6376814842    0.058084033
# wind_adj     0.12139524992544105    0.09073968923106623    229917.5779030919    1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# hail         6.764960230567528e-5   6.531116938039389e-5   6.026699636046774e8  0.0033965781
# hail         0.006278763760885456   0.005701827717092226   6.493403477720737e6  0.00951148
# hail         0.014420953862893071   0.013691235395506226   2.8271335290228724e6 0.0200999
# hail         0.03036611534729695    0.027114783444007525   1.3426075132088661e6 0.037743744
# hail         0.0563960274097572     0.05205358097325824    722922.2855994105    0.07645232
# hail         0.1321928161649276     0.12157954443463385    308391.8084717989    1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# sig_tornado  2.6593520472600964e-6  2.9087044519127597e-6  6.126516836772509e8  0.00080238597
# sig_tornado  0.0013914225287293778  0.0016034634877866321  1.1712139657267928e6 0.0032272756
# sig_tornado  0.004599093472374555   0.004886750811805548   354258.3201146126    0.007747574
# sig_tornado  0.015000437717734967   0.010107300520468552   108618.95035761595   0.0136314705
# sig_tornado  0.03184093593075597    0.017206056993538065   51156.95207029581    0.022335978
# sig_tornado  0.05913631128691091    0.032299220579131115   27490.35437476635    1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# sig_wind     1.5042058545146386e-5  1.4179392635317313e-5  6.03333864125402e8   0.0006871624
# sig_wind     0.0013723297187506972  0.0012298956588724483  6.612953758494496e6  0.0022018508
# sig_wind     0.0036909008396189967  0.003205055684794471   2.4588003684316278e6 0.0047563836
# sig_wind     0.008869202286479402   0.006176078460737166   1.0232852561042309e6 0.008192127
# sig_wind     0.014678376293352925   0.01089057472730591    618319.7001114488    0.015201166
# sig_wind     0.028603864431963836   0.026951269043355437   317199.0110206008    1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# sig_wind_adj 5.282255305241205e-6   6.842631058343068e-6   6.074130114360392e8  0.00042257016
# sig_wind_adj 0.0006850837465814247  0.0007228487073264091  4.683295867521226e6  0.0012706288
# sig_wind_adj 0.0025358770408533495  0.0016753712242263612  1.2653165299318433e6 0.002259613
# sig_wind_adj 0.005207151979126613   0.0028323115432383945  616230.9108408093    0.003675646
# sig_wind_adj 0.01320354652849132    0.004393004738892404   243039.7882348299    0.0054352577
# sig_wind_adj 0.022344491351120654   0.007522772879297647   143527.68617641926   1.0
# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# sig_hail     8.114952808294503e-6   7.310395083599137e-6   6.085486595204935e8  0.00072055974
# sig_hail     0.0014934736428487768  0.0012510563425396038  3.306819054213047e6  0.0021597594
# sig_hail     0.0036710009451549203  0.003129669670864673   1.3454226473237872e6 0.00463198
# sig_hail     0.007321713057372265   0.006419359641547799   674483.2519410253    0.009187652
# sig_hail     0.014697855891048285   0.012789763046642049   336031.78822118044   0.01903059
# sig_hail     0.03226015752280175    0.03280051570749567    153005.95635402203   1.0


println(event_to_bins)
# Dict{String, Vector{Float32}}("sig_wind" => [0.0006871624, 0.0022018508, 0.0047563836, 0.008192127, 0.015201166, 1.0], "sig_hail" => [0.00072055974, 0.0021597594, 0.00463198, 0.009187652, 0.01903059, 1.0], "hail" => [0.0033965781, 0.00951148, 0.0200999, 0.037743744, 0.07645232, 1.0], "sig_wind_adj" => [0.00042257016, 0.0012706288, 0.002259613, 0.003675646, 0.0054352577, 1.0], "tornado" => [0.0012153604, 0.004538168, 0.011742671, 0.023059275, 0.049979284, 1.0], "wind_adj" => [0.0025728762, 0.008021789, 0.016496103, 0.030200316, 0.058084033, 1.0], "sig_tornado" => [0.00080238597, 0.0032272756, 0.007747574, 0.0136314705, 0.022335978, 1.0], "wind" => [0.0078376075, 0.020091131, 0.037815828, 0.065400586, 0.11733462, 1.0])


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
# event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[1.0894886, 0.7778136], [1.0595194, 0.49330923], [1.3312615, 2.0373042], [0.87163, -0.28045976], [0.66988313, -1.1392391]], "sig_hail" => [[1.0996535, 0.9160373], [1.0066487, 0.20982021], [0.9350299, -0.20348155], [0.9644414, -0.02964372], [0.8333078, -0.5743816]], "hail" => [[1.0600259, 0.43388337], [0.96319646, -0.097637914], [1.0792756, 0.39443073], [0.9629313, -0.01864577], [1.0167043, 0.13172606]], "sig_wind_adj" => [[1.0503622, 0.2618211], [1.5049535, 3.5953336], [1.3739139, 2.7983332], [2.022486, 6.61417], [0.99839044, 1.0990105]], "tornado" => [[1.0710595, 0.5394862], [0.89414734, -0.5616081], [1.108447, 0.50613326], [0.83939207, -0.5758451], [1.2171175, 0.7334359]], "wind_adj" => [[1.1026261, 0.554337], [1.1907648, 0.9671046], [1.1658177, 0.862035], [1.0740721, 0.5209931], [0.9590977, 0.1968526]], "sig_tornado" => [[1.0426707, 0.21755064], [1.1516033, 0.7787872], [1.6196959, 3.2147853], [1.4218588, 2.3287997], [1.032921, 0.7553395]], "wind" => [[1.0940193, 0.44082266], [1.1078424, 0.48961264], [1.0705312, 0.36004978], [1.0221666, 0.22469698], [0.9163128, -0.022937609]])


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
cutoff = Dates.DateTime(2022, 6, 1, 12)
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
