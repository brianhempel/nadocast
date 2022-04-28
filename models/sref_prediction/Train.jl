# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.


import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import SREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(SREFPrediction.forecasts_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# for testing
# validation_forecasts = validation_forecasts[1:300]
# validation_forecasts = rand(validation_forecasts, 30);

# rm("validation_forecasts_with_blurs_and_forecast_hour"; recursive=true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts;
    event_name_to_labeler = TrainingShared.event_name_to_labeler,
    save_dir = "validation_forecasts_with_blurs_and_forecast_hour"
  );

length(validation_forecasts) # 19959
size(X) # (101172171, 31)
length(weights) # 101172171

# Sanity check...tornado features should best predict tornadoes, etc
# (this did find a bug :D)

# function test_predictive_power(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   for prediction_i in 1:length(SREFPrediction.models)
#     (event_name, _, _) = SREFPrediction.models[prediction_i]
#     y = Ys[event_name]
#     for j in 1:size(X,2)
#       x = @view X[:,j]
#       auc = Metrics.roc_auc(x, y, weights)
#       println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAUC: $auc")
#     end
#   end
# end
# test_predictive_power(validation_forecasts, X, Ys, weights)

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(SREFPrediction.models)
    (event_name, _, _) = SREFPrediction.models[prediction_i]
    y = Ys[event_name]
    for j in 1:size(X,2)
      x = @view X[:,j]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (9554.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.020782699266332125 ***best tor***
# tornado (9554.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.020607184947527755
# tornado (9554.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.02012135228462739
# tornado (9554.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.01906040998023459
# tornado (9554.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.016734722113272673
# tornado (9554.0)     feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.008729082769789741
# tornado (9554.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.00889882055885013
# tornado (9554.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.008676999342587774
# tornado (9554.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.00831111457058183
# tornado (9554.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.007260073720642238
# tornado (9554.0)     feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.004790220802736399
# tornado (9554.0)     feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.004918421200854783
# tornado (9554.0)     feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.004812143717491475
# tornado (9554.0)     feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.004668362255184233
# tornado (9554.0)     feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.004185316627705799
# tornado (9554.0)     feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.015794702167198863
# tornado (9554.0)     feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.015854215228154788
# tornado (9554.0)     feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.015633346787162956
# tornado (9554.0)     feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.015083948811094197
# tornado (9554.0)     feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.013645079469362676
# tornado (9554.0)     feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.007843810762544456
# tornado (9554.0)     feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.007990598192245938
# tornado (9554.0)     feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007849767768100965
# tornado (9554.0)     feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.007686272444727404
# tornado (9554.0)     feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007023532601477734
# tornado (9554.0)     feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.003880939136248815
# tornado (9554.0)     feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0039044362451593137
# tornado (9554.0)     feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.003867154271132548
# tornado (9554.0)     feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0037500829165953305
# tornado (9554.0)     feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003444568566829963
# tornado (9554.0)     feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 9.632263685692656e-5
# wind (76241.0)       feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.042843912998900294
# wind (76241.0)       feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.04343943928128981
# wind (76241.0)       feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.04335901228170536
# wind (76241.0)       feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.042172059404344285
# wind (76241.0)       feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.038834719418305456
# wind (76241.0)       feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.08842399168992812
# wind (76241.0)       feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.09048772423009667
# wind (76241.0)       feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.09065510802946328 ***best wind***
# wind (76241.0)       feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.08849672733661118
# wind (76241.0)       feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0832519575518789
# wind (76241.0)       feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.02248224518135722
# wind (76241.0)       feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.02201637577875003
# wind (76241.0)       feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.021561237158861192
# wind (76241.0)       feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.02066002002640838
# wind (76241.0)       feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.018773362884374742
# wind (76241.0)       feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.02944664371564376
# wind (76241.0)       feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.029617009435959393
# wind (76241.0)       feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.029367005865423262
# wind (76241.0)       feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0286426675215702
# wind (76241.0)       feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.02661736603693505
# wind (76241.0)       feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05265932462200569
# wind (76241.0)       feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.053473984951376494
# wind (76241.0)       feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.053382549377969594
# wind (76241.0)       feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.05250124330919556
# wind (76241.0)       feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04967127463082073
# wind (76241.0)       feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014767961646069059
# wind (76241.0)       feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.014534542848686561
# wind (76241.0)       feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.014284773460881148
# wind (76241.0)       feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01383568027248423
# wind (76241.0)       feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012829848828795924
# wind (76241.0)       feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 0.0007532768859019603
# hail (33947.0)       feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.012519689873883378
# hail (33947.0)       feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.012283131741584954
# hail (33947.0)       feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.012014182733023841
# hail (33947.0)       feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.011515693320021258
# hail (33947.0)       feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.010369214003351503
# hail (33947.0)       feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.013442575010976792
# hail (33947.0)       feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.013184767682764371
# hail (33947.0)       feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.01290626880406864
# hail (33947.0)       feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.01239027436065652
# hail (33947.0)       feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.01126218109606531
# hail (33947.0)       feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.056932018106091394 ***best hail***
# hail (33947.0)       feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.05672849939527176
# hail (33947.0)       feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.05552942397718741
# hail (33947.0)       feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.05193967263620021
# hail (33947.0)       feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.04526882686961591
# hail (33947.0)       feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.008885118818392119
# hail (33947.0)       feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.008833606239823094
# hail (33947.0)       feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.008718454951969785
# hail (33947.0)       feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.008468372067316973
# hail (33947.0)       feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.007856384019226021
# hail (33947.0)       feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014232405014854654
# hail (33947.0)       feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.014132504155404154
# hail (33947.0)       feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01396398247552295
# hail (33947.0)       feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.013568538822652666
# hail (33947.0)       feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012578104888292136
# hail (33947.0)       feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04153192050251955
# hail (33947.0)       feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.04110800061984685
# hail (33947.0)       feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04024574112556847
# hail (33947.0)       feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03785934174793547
# hail (33947.0)       feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03320578279995697
# hail (33947.0)       feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 0.00033141154834544516
# sig_tornado (1456.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.00887455864077723
# sig_tornado (1456.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.009332984111944418
# sig_tornado (1456.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.009365142449056962
# sig_tornado (1456.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.008962501689296201                                                                                                                                                                                         sig_
# sig_tornado (1456.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.007989374378301096
# sig_tornado (1456.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0020183392678100143
# sig_tornado (1456.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.002162852740820606
# sig_tornado (1456.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0022144157834026647                                                                                                                                                                                  si
# sig_tornado (1456.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0020881646566595627
# sig_tornado (1456.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.001935086159001465
# sig_tornado (1456.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0012774453422948464
# sig_tornado (1456.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0013682276080212847
# sig_tornado (1456.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.001354179284361416
# sig_tornado (1456.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0013468425965915993
# sig_tornado (1456.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0012216792474914256
# sig_tornado (1456.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.013022853338869819
# sig_tornado (1456.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.013260132972206998 ***best sigtor***
# sig_tornado (1456.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.013135121146448697
# sig_tornado (1456.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.012671876725209913
# sig_tornado (1456.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.01161729686372112
# sig_tornado (1456.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002285822990932865
# sig_tornado (1456.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0023598851004624855
# sig_tornado (1456.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.002341456986124376
# sig_tornado (1456.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022374127914466992
# sig_tornado (1456.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0019994705270588325
# sig_tornado (1456.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0008957252790421215
# sig_tornado (1456.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0009128957892462618
# sig_tornado (1456.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0009234400394200565
# sig_tornado (1456.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0009107958607937905
# sig_tornado (1456.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0008791627366882373
# sig_tornado (1456.0) feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 1.5175022436710001e-5
# sig_wind (7763.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.005257098694985902
# sig_wind (7763.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.0052459946014711545
# sig_wind (7763.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.005157440239193201
# sig_wind (7763.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.005080250891698849
# sig_wind (7763.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.004681050883076476
# sig_wind (7763.0)    feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.008863283437957828
# sig_wind (7763.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.00895641631527723
# sig_wind (7763.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.008818891909697401
# sig_wind (7763.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.008608229733031221
# sig_wind (7763.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.007955487070257873
# sig_wind (7763.0)    feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.003689713059746794
# sig_wind (7763.0)    feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0036447550022363595
# sig_wind (7763.0)    feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0035788227564974772
# sig_wind (7763.0)    feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0034325625122283206
# sig_wind (7763.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0030964975988868673
# sig_wind (7763.0)    feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.004215948372841709
# sig_wind (7763.0)    feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0043048681510965275
# sig_wind (7763.0)    feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0041929103930657035
# sig_wind (7763.0)    feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.004141855242199331
# sig_wind (7763.0)    feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0038310374473061615
# sig_wind (7763.0)    feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.011886272397312634
# sig_wind (7763.0)    feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01224057314240419 ***best sigwind**
# sig_wind (7763.0)    feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.012217368543237675
# sig_wind (7763.0)    feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01211489220288776
# sig_wind (7763.0)    feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011465610189038576
# sig_wind (7763.0)    feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002833272447286402
# sig_wind (7763.0)    feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0028234808854716254
# sig_wind (7763.0)    feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.002782116646014624
# sig_wind (7763.0)    feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.002707992764372849
# sig_wind (7763.0)    feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002520695602895615
# sig_wind (7763.0)    feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 7.593908385192544e-5
# sig_hail (4210.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.002292193790287841
# sig_hail (4210.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.002251120321234128
# sig_hail (4210.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.0021962957702101674
# sig_hail (4210.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.0020824880736158647
# sig_hail (4210.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.0018447624706492522
# sig_hail (4210.0)    feature 6 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0017631638910707659
# sig_hail (4210.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.0017319378139064108
# sig_hail (4210.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0016819980102746378
# sig_hail (4210.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.001608474404724983
# sig_hail (4210.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0014643600760589166
# sig_hail (4210.0)    feature 11 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.014199820572411693
# sig_hail (4210.0)    feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.014309985500982302 ***best sighail***
# sig_hail (4210.0)    feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.01400973795594628
# sig_hail (4210.0)    feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.013208316481724082
# sig_hail (4210.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.011859939048920931
# sig_hail (4210.0)    feature 16 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0017002642162870013
# sig_hail (4210.0)    feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0016714551523526091
# sig_hail (4210.0)    feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0016428791787266785
# sig_hail (4210.0)    feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0015620511741973669
# sig_hail (4210.0)    feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.001413646560737021
# sig_hail (4210.0)    feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027598330267082512
# sig_hail (4210.0)    feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.002753304234365191
# sig_hail (4210.0)    feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0027397494723968288
# sig_hail (4210.0)    feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026850344017470013
# sig_hail (4210.0)    feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00251681386316898
# sig_hail (4210.0)    feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013486701386034809
# sig_hail (4210.0)    feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.013570744206282766 (not best sighail)
# sig_hail (4210.0)    feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.013288036900563738
# sig_hail (4210.0)    feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012526074412665879
# sig_hail (4210.0)    feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010965544045008829
# sig_hail (4210.0)    feature 31 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 4.094225562796943e-5



# tornado (9554.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:        AUC: 0.9853510253427634
# tornado (9554.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean       AUC: 0.9855161078753414
# tornado (9554.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean       AUC: 0.9854630553480989
# tornado (9554.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean       AUC: 0.985149685030206
# tornado (9554.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9841681006085854
# tornado (9554.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:       AUC: 0.9729687574644436
# tornado (9554.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9733565276163831
# tornado (9554.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9732654439546365
# tornado (9554.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9728871823747943
# tornado (9554.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean    AUC: 0.9717443114175286
# tornado (9554.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:      AUC: 0.95647612031941
# tornado (9554.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean     AUC: 0.9567701560245393
# tornado (9554.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean     AUC: 0.9564787886871897
# tornado (9554.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean     AUC: 0.9555578117001224
# tornado (9554.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean    AUC: 0.9531719601805859
# tornado (9554.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:      AUC: 0.9636861751794397
# tornado (9554.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean     AUC: 0.9646802194973608
# tornado (9554.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean     AUC: 0.9647229708362134
# tornado (9554.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean     AUC: 0.9645835925064128
# tornado (9554.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean    AUC: 0.9635834019561287
# tornado (9554.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:     AUC: 0.9712072919521524
# tornado (9554.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9716516190567763
# tornado (9554.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.9715575479035303
# tornado (9554.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.9712269863014199
# tornado (9554.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9702519126943745
# tornado (9554.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:     AUC: 0.9483616607443691
# tornado (9554.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9491787639173674
# tornado (9554.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.949288161127435
# tornado (9554.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.9486899509097323
# tornado (9554.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9467936648102897
# tornado (9554.0) feature 31 forecast_hour:calculated:hour fcst::        AUC: 0.4975478568003605
# wind (76241.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:  AUC: 0.9602153865319246
# wind (76241.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean AUC: 0.9609180443909543
# wind (76241.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean AUC: 0.9608641117087021
# wind (76241.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean AUC: 0.9602194023146089
# wind (76241.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean        AUC: 0.9583367002012261
# wind (76241.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob: AUC: 0.9841262384320523
# wind (76241.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean        AUC: 0.984474860226249
# wind (76241.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean        AUC: 0.9844744942213566
# wind (76241.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean        AUC: 0.984254741881639
# wind (76241.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9835250825593285
# wind (76241.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:        AUC: 0.9607415638386363
# wind (76241.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean       AUC: 0.9611109091532068
# wind (76241.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean       AUC: 0.9609691178284533
# wind (76241.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean       AUC: 0.9602676304816856
# wind (76241.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9583082279677898
# wind (76241.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:        AUC: 0.9233217844434465
# wind (76241.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean       AUC: 0.9252576456969921
# wind (76241.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean       AUC: 0.9257646975507423
# wind (76241.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean       AUC: 0.9256576481628719
# wind (76241.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9249664578174774
# wind (76241.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:       AUC: 0.9740165033124774
# wind (76241.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9744172897521022
# wind (76241.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9743433883378437
# wind (76241.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9738888147564924
# wind (76241.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.9725579346669867
# wind (76241.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:       AUC: 0.9417262918936158
# wind (76241.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.942950480963109
# wind (76241.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9430033226392229
# wind (76241.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9424113500566565
# wind (76241.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.9404054278172317
# wind (76241.0) feature 31 forecast_hour:calculated:hour fcst::  AUC: 0.49820418679731165
# hail (33947.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:  AUC: 0.9614375138640956
# hail (33947.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean AUC: 0.9631114113708558
# hail (33947.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean AUC: 0.9635697521790578
# hail (33947.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean AUC: 0.9635776873036093
# hail (33947.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean        AUC: 0.9624954922213323
# hail (33947.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob: AUC: 0.9718678398971429
# hail (33947.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean        AUC: 0.972109752420438
# hail (33947.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean        AUC: 0.9719372527292367
# hail (33947.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean        AUC: 0.9713502417619471
# hail (33947.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9697003709194171
# hail (33947.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:        AUC: 0.985635511904134
# hail (33947.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean       AUC: 0.9858894586276629
# hail (33947.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean       AUC: 0.985812829174833
# hail (33947.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean       AUC: 0.9854778768133305
# hail (33947.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9844860962723286
# hail (33947.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:        AUC: 0.9402035588225701
# hail (33947.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean       AUC: 0.9420821548813049
# hail (33947.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean       AUC: 0.9424457669744497
# hail (33947.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean       AUC: 0.9423835688241814
# hail (33947.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean      AUC: 0.9412729825515965
# hail (33947.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:       AUC: 0.9701444487735278
# hail (33947.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9705326084476049
# hail (33947.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9704300168735658
# hail (33947.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9700429800308505
# hail (33947.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.968692519750841
# hail (33947.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:       AUC: 0.9788848827482417
# hail (33947.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9793148202352601
# hail (33947.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9792597912793889
# hail (33947.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9788763231026227
# hail (33947.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.9777471739120115
# hail (33947.0) feature 31 forecast_hour:calculated:hour fcst::  AUC: 0.49680726514627177
# sig_tornado (1456.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:    AUC: 0.9878695461278791
# sig_tornado (1456.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean   AUC: 0.9881491547221682
# sig_tornado (1456.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean   AUC: 0.9881467390917699
# sig_tornado (1456.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean   AUC: 0.9880897210794519
# sig_tornado (1456.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean  AUC: 0.987423350566552
# sig_tornado (1456.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:   AUC: 0.9802610914265749
# sig_tornado (1456.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AUC: 0.9803467679016374
# sig_tornado (1456.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AUC: 0.9802728791932804
# sig_tornado (1456.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AUC: 0.9798921473224249
# sig_tornado (1456.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean        AUC: 0.979182603139641
# sig_tornado (1456.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:  AUC: 0.9691828832289122
# sig_tornado (1456.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean AUC: 0.970188770903669
# sig_tornado (1456.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean AUC: 0.9700733041232965
# sig_tornado (1456.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean AUC: 0.9697884211347385
# sig_tornado (1456.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean        AUC: 0.9684581608975318
# sig_tornado (1456.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:  AUC: 0.9788640198334441
# sig_tornado (1456.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean AUC: 0.9791474977687419
# sig_tornado (1456.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean AUC: 0.9791168006541603
# sig_tornado (1456.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean AUC: 0.9790323469989695
# sig_tornado (1456.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean        AUC: 0.978586468918341
# sig_tornado (1456.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob: AUC: 0.9855254635387074
# sig_tornado (1456.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean        AUC: 0.9862227515496658
# sig_tornado (1456.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean        AUC: 0.9862252129652698
# sig_tornado (1456.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean        AUC: 0.986126741632145
# sig_tornado (1456.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean       AUC: 0.9857565956354646
# sig_tornado (1456.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob: AUC: 0.9656434371065298
# sig_tornado (1456.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean        AUC: 0.9655371799969563
# sig_tornado (1456.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean        AUC: 0.9654047133132772
# sig_tornado (1456.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean        AUC: 0.9649893319646393
# sig_tornado (1456.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean       AUC: 0.963797844970546
# sig_tornado (1456.0) feature 31 forecast_hour:calculated:hour fcst::    AUC: 0.49675204950626045
# sig_wind (7763.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:       AUC: 0.9729883661280379
# sig_wind (7763.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9739408520397034
# sig_wind (7763.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9739452571652684
# sig_wind (7763.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.973589365391343
# sig_wind (7763.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.9721794408140543
# sig_wind (7763.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:      AUC: 0.985879758824423
# sig_wind (7763.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean     AUC: 0.9862951866349564
# sig_wind (7763.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean     AUC: 0.986338367158421
# sig_wind (7763.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean     AUC: 0.9861642535422618
# sig_wind (7763.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9855710336693106
# sig_wind (7763.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:     AUC: 0.9679807873825591
# sig_wind (7763.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9681983019839612
# sig_wind (7763.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.9680162997435665
# sig_wind (7763.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.9674315863979661
# sig_wind (7763.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9658822266753733
# sig_wind (7763.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:     AUC: 0.9588087454011973
# sig_wind (7763.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9597675845017517
# sig_wind (7763.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.9598255199363172
# sig_wind (7763.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.9594690609889371
# sig_wind (7763.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9583126933244799
# sig_wind (7763.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:    AUC: 0.9846436619261212
# sig_wind (7763.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean   AUC: 0.9849445066739609
# sig_wind (7763.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean   AUC: 0.9848964409772596
# sig_wind (7763.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean   AUC: 0.984690579366703
# sig_wind (7763.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean  AUC: 0.9839312311524977
# sig_wind (7763.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:    AUC: 0.9586501675216379
# sig_wind (7763.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean   AUC: 0.9593789280531142
# sig_wind (7763.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean   AUC: 0.9593076449067583
# sig_wind (7763.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean   AUC: 0.9588359632365482
# sig_wind (7763.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean  AUC: 0.9572738672364184
# sig_wind (7763.0) feature 31 forecast_hour:calculated:hour fcst::       AUC: 0.4964228442198636
# sig_hail (4210.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:       AUC: 0.9757318970505486
# sig_hail (4210.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean      AUC: 0.9775975650952637
# sig_hail (4210.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean      AUC: 0.9782580256966573
# sig_hail (4210.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean      AUC: 0.9788612423316122
# sig_hail (4210.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean     AUC: 0.9787755993952766
# sig_hail (4210.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:      AUC: 0.9760251470320117
# sig_hail (4210.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean     AUC: 0.9764609110706776
# sig_hail (4210.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean     AUC: 0.9763525034896288
# sig_hail (4210.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean     AUC: 0.9761117395102727
# sig_hail (4210.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9751381682640097
# sig_hail (4210.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:     AUC: 0.9916870767694093
# sig_hail (4210.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9921628447554207
# sig_hail (4210.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.9923088094258008
# sig_hail (4210.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.9923650567076646
# sig_hail (4210.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.992239694651554
# sig_hail (4210.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:     AUC: 0.9729689901008851
# sig_hail (4210.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean    AUC: 0.9738787907320694
# sig_hail (4210.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean    AUC: 0.9739784449902024
# sig_hail (4210.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean    AUC: 0.973795227449058
# sig_hail (4210.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean   AUC: 0.9729460933388578
# sig_hail (4210.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:    AUC: 0.9818517258202872
# sig_hail (4210.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean   AUC: 0.9823602288454409
# sig_hail (4210.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean   AUC: 0.9824186529530367
# sig_hail (4210.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean   AUC: 0.982271851725297
# sig_hail (4210.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean  AUC: 0.9816978318177089
# sig_hail (4210.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:    AUC: 0.9938288941580401
# sig_hail (4210.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean   AUC: 0.9939923854677758
# sig_hail (4210.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean   AUC: 0.9939817280415055
# sig_hail (4210.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean   AUC: 0.9938381481598034
# sig_hail (4210.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean  AUC: 0.9933883750975475
# sig_hail (4210.0) feature 31 forecast_hour:calculated:hour fcst::       AUC: 0.4922905351325514


println("Determining best blur radii to maximize area under precision-recall curve")

blur_radii = [0; SREFPrediction.blur_radii]
forecast_hour_j = size(X, 2)

bests = []
for prediction_i in 1:length(SREFPrediction.models)
  (event_name, _, _) = SREFPrediction.models[prediction_i]
  y = Ys[event_name]
  prediction_i_base = (prediction_i - 1) * length(blur_radii) # 0-indexed

  println("blur_radius_f2\tblur_radius_f38\tAU_PR_$event_name")

  best_blur_i_lo, best_blur_i_hi, best_au_pr = (0, 0, 0.0)

  for blur_i_lo in 1:length(blur_radii)
    for blur_i_hi in 1:length(blur_radii)
      X_blurred = zeros(Float32, length(y))

      Threads.@threads for i in 1:length(y)
        forecast_ratio = (X[i,forecast_hour_j] - 2f0) * (1f0/(38f0-2f0))
        X_blurred[i] = X[i,prediction_i_base+blur_i_lo] * (1f0 - forecast_ratio) + X[i,prediction_i_base+blur_i_hi] * forecast_ratio
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
# blur_radius_f2  blur_radius_f38 AU_PR_tornado
# 0       0       0.0207827
# 0       35      0.020941839
# 0       50      0.020957047
# 0       70      0.020886002
# 0       100     0.020635314
# 35      0       0.020583533
# 35      35      0.020607185
# 35      50      0.020582465
# 35      70      0.02047177
# 35      100     0.020177372
# 50      0       0.02017669
# 50      35      0.020197283
# 50      50      0.020121353
# 50      70      0.019979602
# 50      100     0.019623939
# 70      0       0.019450538
# 70      35      0.019383889
# 70      50      0.019281948
# 70      70      0.01906041
# 70      100     0.01860483
# 100     0       0.018032517
# 100     35      0.017916162
# 100     50      0.017733444
# 100     70      0.017416338
# 100     100     0.016734721
# Best tornado: 0 50      0.020957047

# blur_radius_f2  blur_radius_f38 AU_PR_wind
# 0       0       0.08842399
# 0       35      0.08970518
# 0       50      0.08994371
# 0       70      0.089598514
# 0       100     0.088719234
# 35      0       0.08976621
# 35      35      0.090487726
# 35      50      0.09067886
# 35      70      0.09025056
# 35      100     0.089363314
# 50      0       0.08993563
# 50      35      0.09059681
# 50      50      0.09065511
# 50      70      0.09010187
# 50      100     0.089054644
# 70      0       0.08863365
# 70      35      0.08928645
# 70      50      0.08934647
# 70      70      0.08849673
# 70      100     0.0871801
# 100     0       0.08575069
# 100     35      0.08641601
# 100     50      0.08642633
# 100     70      0.085259244
# 100     100     0.08325195
# Best wind: 35   50      0.09067886

# blur_radius_f2  blur_radius_f38 AU_PR_hail
# 0       0       0.056932017
# 0       35      0.057110623
# 0       50      0.05690895
# 0       70      0.056191016
# 0       100     0.054990213
# 35      0       0.057055406
# 35      35      0.0567285
# 35      50      0.056438867
# 35      70      0.05553659
# 35      100     0.054195996
# 50      0       0.05625833
# 50      35      0.05592779
# 50      50      0.055529423
# 50      70      0.054492947
# 50      100     0.05296945
# 70      0       0.054113727
# 70      35      0.05372238
# 70      50      0.05325764
# 70      70      0.051939674
# 70      100     0.05009843
# 100     0       0.05047296
# 100     35      0.04995172
# 100     50      0.049416218
# 100     70      0.047767516
# 100     100     0.045268826
# Best hail: 0    35      0.057110623

# blur_radius_f2  blur_radius_f38 AU_PR_sig_tornado
# 0       0       0.013022854
# 0       35      0.013147673
# 0       50      0.013161048
# 0       70      0.013095252
# 0       100     0.01290228
# 35      0       0.013245753
# 35      35      0.013260133
# 35      50      0.013245582
# 35      70      0.013155325
# 35      100     0.012947855
# 50      0       0.013160974
# 50      35      0.013167673
# 50      50      0.013135121
# 50      70      0.013114634
# 50      100     0.013034597
# 70      0       0.012959486
# 70      35      0.012908472
# 70      50      0.012828065
# 70      70      0.012671876
# 70      100     0.012387383
# 100     0       0.012598947
# 100     35      0.012499749
# 100     50      0.01233663
# 100     70      0.0120950835
# 100     100     0.011617296
# Best sig_tornado: 35    35      0.013260133

# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind
# 0       0       0.011886273
# 0       35      0.012140539
# 0       50      0.012182775
# 0       70      0.012224448
# 0       100     0.01217784
# 35      0       0.012062436
# 35      35      0.012240573
# 35      50      0.012276374
# 35      70      0.012308959
# 35      100     0.012250253
# 50      0       0.012025666
# 50      35      0.012199556
# 50      50      0.012217369
# 50      70      0.012239931
# 50      100     0.012159852
# 70      0       0.011950417
# 70      35      0.012114315
# 70      50      0.01212179
# 70      70      0.012114892
# 70      100     0.011994959
# 100     0       0.011601174
# 100     35      0.0117603615
# 100     50      0.011744328
# 100     70      0.011696961
# 100     100     0.01146561
# Best sig_wind: 35       70      0.012308959

# blur_radius_f2  blur_radius_f38 AU_PR_sig_hail
# 0       0       0.013486701
# 0       35      0.01365057
# 0       50      0.013652643
# 0       70      0.013584077
# 0       100     0.013379718
# 35      0       0.013523464
# 35      35      0.013570745
# 35      50      0.013558566
# 35      70      0.013456966
# 35      100     0.013228252
# 50      0       0.013294753
# 50      35      0.013327017
# 50      50      0.013288037
# 50      70      0.013157872
# 50      100     0.012895392
# 70      0       0.012778704
# 70      35      0.012770981
# 70      50      0.012710734
# 70      70      0.012526074
# 70      100     0.012195693
# 100     0       0.011879772
# 100     35      0.011828173
# 100     50      0.01172188
# 100     70      0.01144836
# 100     100     0.010965544
# Best sig_hail: 0        50      0.013652643




# blur_radius_f2  blur_radius_f38 AUC_tornado
# 0       0       0.985351
# 0       35      0.9854703
# 0       50      0.98548776
# 0       70      0.9854077
# 0       100     0.9850544
# 35      0       0.9855062
# 35      35      0.98551613
# 35      50      0.9855152
# 35      70      0.985422
# 35      100     0.98506546
# 50      0       0.98549074
# 50      35      0.9854852
# 50      50      0.9854631
# 50      70      0.9853618
# 50      100     0.9849952
# 70      0       0.9853206
# 70      35      0.985302
# 70      50      0.9852698
# 70      70      0.9851497
# 70      100     0.9847742
# 100     0       0.98477685
# 100     35      0.9847476
# 100     50      0.9847015
# 100     70      0.9845667
# 100     100     0.9841681
# Best tornado: 35        35      0.98551613

# blur_radius_f2  blur_radius_f38 AUC_wind
# 0       0       0.9841262
# 0       35      0.98439896
# 0       50      0.98447233
# 0       70      0.98448825
# 0       100     0.98440087
# 35      0       0.98435193
# 35      35      0.98447484
# 35      50      0.98451644
# 35      70      0.98449767
# 35      100     0.98438275
# 50      0       0.98437303
# 50      35      0.98446697
# 50      50      0.9844745
# 50      70      0.9844348
# 50      100     0.98428696
# 70      0       0.9842913
# 70      35      0.98435295
# 70      50      0.98434013
# 70      70      0.9842547
# 70      100     0.9840573
# 100     0       0.98398894
# 100     35      0.9840218
# 100     50      0.9839752
# 100     70      0.9838394
# 100     100     0.9835251
# Best wind: 35   50      0.98451644

# blur_radius_f2  blur_radius_f38 AUC_hail
# 0       0       0.9856355
# 0       35      0.985882
# 0       50      0.9859254
# 0       70      0.98589957
# 0       100     0.98568845
# 35      0       0.98578095
# 35      35      0.98588943
# 35      50      0.98590803
# 35      70      0.9858502
# 35      100     0.9856173
# 50      0       0.98573554
# 50      35      0.9858186
# 50      50      0.98581284
# 50      70      0.9857382
# 50      100     0.9854809
# 70      0       0.9855691
# 70      35      0.9856187
# 70      50      0.98559564
# 70      70      0.98547786
# 70      100     0.98517376
# 100     0       0.9850894
# 100     35      0.98510885
# 100     50      0.9850583
# 100     70      0.98488975
# 100     100     0.9844861
# Best hail: 0    50      0.9859254

# blur_radius_f2  blur_radius_f38 AUC_sig_tornado
# 0       0       0.978864
# 0       35      0.9790746
# 0       50      0.97909075
# 0       70      0.97914505
# 0       100     0.979021
# 35      0       0.97893083
# 35      35      0.9791475
# 35      50      0.9791623
# 35      70      0.9792136
# 35      100     0.9790876
# 50      0       0.97888714
# 50      35      0.97910017
# 50      50      0.9791168
# 50      70      0.97916746
# 50      100     0.97904307
# 70      0       0.9787542
# 70      35      0.97896194
# 70      50      0.9789786
# 70      70      0.97903234
# 70      100     0.97891176
# 100     0       0.97843254
# 100     35      0.9786318
# 100     50      0.97864777
# 100     70      0.978701
# 100     100     0.9785865
# Best sig_tornado: 35    70      0.9792136

# blur_radius_f2  blur_radius_f38 AUC_sig_wind
# 0       0       0.98464364
# 0       35      0.98489195
# 0       50      0.9849402
# 0       70      0.9849348
# 0       100     0.9847735
# 35      0       0.98482156
# 35      35      0.9849445
# 35      50      0.98496914
# 35      70      0.9849357
# 35      100     0.98475194
# 50      0       0.98479044
# 50      35      0.9848918
# 50      50      0.9848964
# 50      70      0.98485
# 50      100     0.98464876
# 70      0       0.98469204
# 70      35      0.9847719
# 70      50      0.9847651
# 70      70      0.9846906
# 70      100     0.98446184
# 100     0       0.984293
# 100     35      0.98435307
# 100     50      0.9843279
# 100     70      0.9842243
# 100     100     0.98393124
# Best sig_wind: 35       50      0.98496914

# blur_radius_f2  blur_radius_f38 AUC_sig_hail
# 0       0       0.9938289
# 0       35      0.9939737
# 0       50      0.99401325
# 0       70      0.9940117
# 0       100     0.99394655
# 35      0       0.993913
# 35      35      0.9939924
# 35      50      0.9940194
# 35      70      0.99400425
# 35      100     0.9939239
# 50      0       0.9939013
# 50      35      0.9939673
# 50      50      0.9939817
# 50      70      0.99395853
# 50      100     0.9938654
# 70      0       0.99382454
# 70      35      0.9938753
# 70      50      0.9938809
# 70      70      0.99383813
# 70      100     0.9937226
# 100     0       0.99359703
# 100     35      0.99363256
# 100     50      0.9936239
# 100     70      0.9935571
# 100     100     0.99338835
# Best sig_hail: 35       50      0.9940194

println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f38\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
# tornado     0                   50                   0.020957047
# wind        35                  50                   0.09067886
# hail        0                   35                   0.057110623
# sig_tornado 35                  35                   0.013260133
# sig_wind    35                  70                   0.012308959
# sig_hail    0                   50                   0.013652643

# event_name      best_blur_radius_f2     best_blur_radius_f38    AUC
# tornado         35      35      0.98551613
# wind            35      50      0.98451644
# hail            0       50      0.9859254
# sig_tornado     35      70      0.9792136
# sig_wind        35      50      0.98496914
# sig_hail        35      50      0.9940194



# Now go back to SREFPrediction.jl and put those numbers in

# CHECKING that the blurred forecasts are corrected

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
import SREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(SREFPrediction.forecasts_blurred(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100])

# rm("validation_forecasts_blurred"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_blurred");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(SREFPrediction.models)
    (event_name, _, _) = SREFPrediction.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_blurred, X, Ys, weights)

# EXPECTED:
# event_name  AU_PR
# tornado     0.020957047
# wind        0.09067886
# hail        0.057110623
# sig_tornado 0.013260133
# sig_wind    0.012308959
# sig_hail    0.013652643

# ACTUAL:
# tornado (9554.0)     feature 1 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.020957046595993258
# wind (76241.0)       feature 2 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.09067886564861938
# hail (33947.0)       feature 3 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.057110620698781485
# sig_tornado (1456.0) feature 4 STORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.013260132978854111
# sig_wind (7763.0)    feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.012308958537568998
# sig_hail (4210.0)    feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.013652642732904893

# Yay!
