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

# tornado (9554.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.020673989862393907 ***best tor***
# tornado (9554.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.020396528293089137
# tornado (9554.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.019897799317505682
# tornado (9554.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.018861477259179475
# tornado (9554.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.016536778462417308
# tornado (9554.0)     feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.008510502530366908
# tornado (9554.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.008672079553152445
# tornado (9554.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.008457751074549939
# tornado (9554.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.008097658115387957
# tornado (9554.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007061587003581422
# tornado (9554.0)     feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004743685714911955
# tornado (9554.0)     feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004890434228057318
# tornado (9554.0)     feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004776955092882925
# tornado (9554.0)     feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.004634980726076592
# tornado (9554.0)     feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004152761580895475
# tornado (9554.0)     feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015642172282896494
# tornado (9554.0)     feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015670675692129894
# tornado (9554.0)     feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015470310388042323
# tornado (9554.0)     feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.014866313920047083
# tornado (9554.0)     feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013356804710540992
# tornado (9554.0)     feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.007780831100549196
# tornado (9554.0)     feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.007922881262949296
# tornado (9554.0)     feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007777700676839251
# tornado (9554.0)     feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.007615336699719287
# tornado (9554.0)     feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.006961450389497822
# tornado (9554.0)     feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.003907571242034323
# tornado (9554.0)     feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0039348913170217
# tornado (9554.0)     feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.003900114132737239
# tornado (9554.0)     feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.003784801862654207
# tornado (9554.0)     feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003477903584002676
# tornado (9554.0)     feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 9.632263685692656e-5
# wind (76241.0)       feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.042126108750157855
# wind (76241.0)       feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.042601900687798844
# wind (76241.0)       feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.042422910246775254
# wind (76241.0)       feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.04105623722815699
# wind (76241.0)       feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.03751068461572154
# wind (76241.0)       feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0887295283018391
# wind (76241.0)       feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.09083817422836311
# wind (76241.0)       feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.09102787243082383 ***best wind***
# wind (76241.0)       feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.08886063305654229
# wind (76241.0)       feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.08354476443219633
# wind (76241.0)       feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02236917495543825
# wind (76241.0)       feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.021918360673617226
# wind (76241.0)       feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.021470317164241387
# wind (76241.0)       feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.020581705165705313
# wind (76241.0)       feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01871740055547519
# wind (76241.0)       feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.029925277075128705
# wind (76241.0)       feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03012816528060895
# wind (76241.0)       feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.029923568815830342
# wind (76241.0)       feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02918273220995161
# wind (76241.0)       feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.027178492272304417
# wind (76241.0)       feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.052651414258475256
# wind (76241.0)       feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.05347946367179884
# wind (76241.0)       feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.053393949021555545
# wind (76241.0)       feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.052518149510077494
# wind (76241.0)       feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04970473498190692
# wind (76241.0)       feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014627699450206408
# wind (76241.0)       feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0143936557263282
# wind (76241.0)       feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.014145215682201834
# wind (76241.0)       feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.013699318454336189
# wind (76241.0)       feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012703442973046986
# wind (76241.0)       feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.0007532768859019603
# hail (33947.0)       feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.012644917620250998
# hail (33947.0)       feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.012406404625097508
# hail (33947.0)       feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.012136066747501933
# hail (33947.0)       feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.011633607342053098
# hail (33947.0)       feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.010481023610546139
# hail (33947.0)       feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.013402093986976907
# hail (33947.0)       feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.01315591308238461
# hail (33947.0)       feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.012881109328361734
# hail (33947.0)       feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.012370902669171464
# hail (33947.0)       feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011250176608982976
# hail (33947.0)       feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05684401433073763 ***best hail***
# hail (33947.0)       feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.056711302472260944
# hail (33947.0)       feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.05552411383987455
# hail (33947.0)       feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.051966178822598685
# hail (33947.0)       feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.045322114173919154
# hail (33947.0)       feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.008854935869997945
# hail (33947.0)       feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.008797232216256919
# hail (33947.0)       feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.008678150228207206
# hail (33947.0)       feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008424847408190229
# hail (33947.0)       feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007808045882947874
# hail (33947.0)       feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014244000845978523
# hail (33947.0)       feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.014148693914454986
# hail (33947.0)       feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.013982791333803701
# hail (33947.0)       feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01359434201989666
# hail (33947.0)       feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012613619478352489
# hail (33947.0)       feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04140143659238543
# hail (33947.0)       feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.04093615574198445
# hail (33947.0)       feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0400856621676289
# hail (33947.0)       feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03769293738572532
# hail (33947.0)       feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.033097885841986176
# hail (33947.0)       feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00033141154834544516
# sig_tornado (1456.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.008993768396086286
# sig_tornado (1456.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.00948369281868588
# sig_tornado (1456.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.00951259572300951
# sig_tornado (1456.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.009084668151438583
# sig_tornado (1456.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.008068858299267496
# sig_tornado (1456.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0019758942450156937
# sig_tornado (1456.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.00211868218360701
# sig_tornado (1456.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0021710531150864304
# sig_tornado (1456.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0020376304628587916
# sig_tornado (1456.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018755403495441373
# sig_tornado (1456.0) feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0012754126846766384
# sig_tornado (1456.0) feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0013718098956261107
# sig_tornado (1456.0) feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0013579759380061456
# sig_tornado (1456.0) feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0013530103555645413
# sig_tornado (1456.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0012238942256533264
# sig_tornado (1456.0) feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012305727212815231
# sig_tornado (1456.0) feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012457302310741051
# sig_tornado (1456.0) feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.012564231767680087 ***best sigtor***
# sig_tornado (1456.0) feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.011861283109147734
# sig_tornado (1456.0) feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010566969687396576
# sig_tornado (1456.0) feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0022451295414142583
# sig_tornado (1456.0) feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0023204099115798392
# sig_tornado (1456.0) feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0023051370058046966
# sig_tornado (1456.0) feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0022017208027531572
# sig_tornado (1456.0) feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0019658385376909756
# sig_tornado (1456.0) feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0008976216238249938
# sig_tornado (1456.0) feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0009182637971446378
# sig_tornado (1456.0) feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0009318022599520912
# sig_tornado (1456.0) feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0009216733851832179
# sig_tornado (1456.0) feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.000894291986139887
# sig_tornado (1456.0) feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 1.5175022436710001e-5
# sig_wind (7763.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.005169852812461103
# sig_wind (7763.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.005143932802428631
# sig_wind (7763.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.005046062592678608
# sig_wind (7763.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0049591717253676315
# sig_wind (7763.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.004556622071437644
# sig_wind (7763.0)    feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.008722888517190816
# sig_wind (7763.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.008813257469202324
# sig_wind (7763.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.008677797721966493
# sig_wind (7763.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.008470369813875143
# sig_wind (7763.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00783407956701595
# sig_wind (7763.0)    feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0036668115564717995
# sig_wind (7763.0)    feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.003620418389664366
# sig_wind (7763.0)    feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0035547510428132383
# sig_wind (7763.0)    feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0034090248548837763
# sig_wind (7763.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003077846733002274
# sig_wind (7763.0)    feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004206975294573811
# sig_wind (7763.0)    feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0042403941293771815
# sig_wind (7763.0)    feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004175810884156483
# sig_wind (7763.0)    feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.004106196598641198
# sig_wind (7763.0)    feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0038185681869915555
# sig_wind (7763.0)    feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.011869790809924372
# sig_wind (7763.0)    feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012232485420622349 ***best sigwind***
# sig_wind (7763.0)    feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01221113354745665
# sig_wind (7763.0)    feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012113282289273922
# sig_wind (7763.0)    feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011475119126956538
# sig_wind (7763.0)    feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027972891635519724
# sig_wind (7763.0)    feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.002785975216845971
# sig_wind (7763.0)    feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.002744941048324694
# sig_wind (7763.0)    feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026710253475298595
# sig_wind (7763.0)    feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002486529400705616
# sig_wind (7763.0)    feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 7.593908385192544e-5
# sig_hail (4210.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0023024667683398344
# sig_hail (4210.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.0022585517380890544
# sig_hail (4210.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0022036928557057495
# sig_hail (4210.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0020897524648483307
# sig_hail (4210.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.0018527034305171864
# sig_hail (4210.0)    feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0017630699425827995
# sig_hail (4210.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0017327220645195086
# sig_hail (4210.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0016829993401467663
# sig_hail (4210.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.001609372109057884
# sig_hail (4210.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014653969388021496
# sig_hail (4210.0)    feature 11 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014182624303670474
# sig_hail (4210.0)    feature 12 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.014328518416410465 ***best sighail***
# sig_hail (4210.0)    feature 13 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.014065344028006294
# sig_hail (4210.0)    feature 14 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.013189333602148596
# sig_hail (4210.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011813906758478437
# sig_hail (4210.0)    feature 16 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0016959896688526375
# sig_hail (4210.0)    feature 17 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.001664715544286852
# sig_hail (4210.0)    feature 18 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0016339289464577737
# sig_hail (4210.0)    feature 19 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.001553667869971903
# sig_hail (4210.0)    feature 20 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014047752094093267
# sig_hail (4210.0)    feature 21 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027663530669807114
# sig_hail (4210.0)    feature 22 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.002759656941165803
# sig_hail (4210.0)    feature 23 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0027480843342910454
# sig_hail (4210.0)    feature 24 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0026962509843625757
# sig_hail (4210.0)    feature 25 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0025336090585764337
# sig_hail (4210.0)    feature 26 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013380280240415392
# sig_hail (4210.0)    feature 27 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01345255374886646 (not best sighail)
# sig_hail (4210.0)    feature 28 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.013164010929754716
# sig_hail (4210.0)    feature 29 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012385388871278304
# sig_hail (4210.0)    feature 30 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010838118374610104
# sig_hail (4210.0)    feature 31 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 4.094225562796943e-5




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
# 0       0       0.02067399
# 0       35      0.020831263
# 0       50      0.02083797
# 0       70      0.020763265
# 0       100     0.020512043
# 35      0       0.020385014
# 35      35      0.020396529
# 35      50      0.020366516
# 35      70      0.020253586
# 35      100     0.019962527
# 50      0       0.020002153
# 50      35      0.019977167
# 50      50      0.019897798
# 50      70      0.019750543
# 50      100     0.019399803
# 70      0       0.019281987
# 70      35      0.01920392
# 70      50      0.01908438
# 70      70      0.018861478
# 70      100     0.018419622
# 100     0       0.017888756
# 100     35      0.01775706
# 100     50      0.017560748
# 100     70      0.017220503
# 100     100     0.016536778
# Best tornado: 0 50      0.02083797

# blur_radius_f2  blur_radius_f38 AU_PR_wind
# 0       0       0.08872953
# 0       35      0.08997875
# 0       50      0.09019419
# 0       70      0.08981727
# 0       100     0.08885631
# 35      0       0.0901493
# 35      35      0.09083817
# 35      50      0.091019414
# 35      70      0.090549596
# 35      100     0.08957677
# 50      0       0.090362936
# 50      35      0.090986356
# 50      50      0.09102787
# 50      70      0.09044231
# 50      100     0.08930532
# 70      0       0.08907597
# 70      35      0.08969355
# 70      50      0.08973043
# 70      70      0.08886063
# 70      100     0.08744851
# 100     0       0.08622919
# 100     35      0.086864755
# 100     50      0.08684555
# 100     70      0.08562976
# 100     100     0.08354476
# Best wind: 50   50      0.09102787

# blur_radius_f2  blur_radius_f38 AU_PR_hail
# 0       0       0.056844015
# 0       35      0.057103045
# 0       50      0.056919135
# 0       70      0.05621726
# 0       100     0.05505335
# 35      0       0.056922175
# 35      35      0.05671131
# 35      50      0.05644463
# 35      70      0.055571552
# 35      100     0.0542755
# 50      0       0.056096923
# 50      35      0.055894583
# 50      50      0.055524115
# 50      70      0.054527577
# 50      100     0.053052455
# 70      0       0.053887956
# 70      35      0.053640585
# 70      50      0.05322457
# 70      70      0.05196618
# 70      100     0.05018413
# 100     0       0.050214622
# 100     35      0.049759787
# 100     50      0.049281113
# 100     70      0.04771897
# 100     100     0.045322113
# Best hail: 0    35      0.057103045

# blur_radius_f2  blur_radius_f38 AU_PR_sig_tornado
# 0       0       0.012305729
# 0       35      0.0124057615
# 0       50      0.012399225
# 0       70      0.012326375
# 0       100     0.012131729
# 35      0       0.012448443
# 35      35      0.012457302
# 35      50      0.012433973
# 35      70      0.01235207
# 35      100     0.0121428175
# 50      0       0.012457372
# 50      35      0.012623444 lol this is best?
# 50      50      0.012564232
# 50      70      0.012472704
# 50      100     0.012274923
# 70      0       0.012060871
# 70      35      0.012048844
# 70      50      0.011995949
# 70      70      0.011861283
# 70      100     0.0116055785
# 100     0       0.011491346
# 100     35      0.011380854
# 100     50      0.011238129
# 100     70      0.011009043
# 100     100     0.010566969
# Best sig_tornado: 50    35      0.012623444

# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind
# 0       0       0.011869791
# 0       35      0.012127075
# 0       50      0.012170782
# 0       70      0.012216457
# 0       100     0.012173215
# 35      0       0.012052469
# 35      35      0.012232485
# 35      50      0.012269737
# 35      70      0.012305792
# 35      100     0.012250941
# 50      0       0.012015845
# 50      35      0.012191487
# 50      50      0.012211134
# 50      70      0.012237056
# 50      100     0.012160974
# 70      0       0.0119412895
# 70      35      0.012107306
# 70      50      0.012116599
# 70      70      0.012113282
# 70      100     0.011998117
# 100     0       0.011597047
# 100     35      0.011757778
# 100     50      0.011743384
# 100     70      0.011700502
# 100     100     0.011475119
# Best sig_wind: 35       70      0.012305792

# blur_radius_f2  blur_radius_f38 AU_PR_sig_hail
# 0       0       0.01338028
# 0       35      0.013543452
# 0       50      0.013548006
# 0       70      0.013478646
# 0       100     0.013289827
# 35      0       0.013401165
# 35      35      0.013452554
# 35      50      0.013445974
# 35      70      0.013347655
# 35      100     0.013135119
# 50      0       0.01315902
# 50      35      0.0131950835
# 50      50      0.013164011
# 50      70      0.013041592
# 50      100     0.012792082
# 70      0       0.012620414
# 70      35      0.012616805
# 70      50      0.012563525
# 70      70      0.012385389
# 70      100     0.012079042
# 100     0       0.01168226
# 100     35      0.011644918
# 100     50      0.011551977
# 100     70      0.011286792
# 100     100     0.010838117
# Best sig_hail: 0        50      0.013548006


println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f38\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
# tornado     0                   50                   0.02083797
# wind        50                  50                   0.09102787
# hail        0                   35                   0.057103045
# sig_tornado 50                  35                   0.012623444
# sig_wind    35                  70                   0.012305792
# sig_hail    0                   50                   0.013548006

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
