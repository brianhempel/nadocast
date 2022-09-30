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
cutoff = Dates.DateTime(2022, 6, 1, 12)
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

length(validation_forecasts) # 23215
size(X) # (89284890, 41)
length(weights) # 89284890

# Sanity check...tornado features should best predict tornadoes, etc
# (this did find a bug :D)

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(SREFPrediction.models)
    (event_name, _, _) = SREFPrediction.models[prediction_i]
    y = Ys[event_name]
    for j in 1:size(X,2)
      x = @view X[:,j]
      au_pr_curve = Float32(Metrics.area_under_pr_curve(x, y, weights))
      println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (10686.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0287798423293815 ***best tor***
# tornado (10686.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.02816390256277358
# tornado (10686.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.027280145401243558
# tornado (10686.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.02586287622884925
# tornado (10686.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.021898484197448605
# tornado (10686.0)     feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.00889556269229842
# tornado (10686.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.008947487344217872
# tornado (10686.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.008705094327244579
# tornado (10686.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.00831737964518261
# tornado (10686.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007134107286950845
# tornado (10686.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004467951523153947
# tornado (10686.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0043710676983410865
# tornado (10686.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0042589671485522805
# tornado (10686.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.004060791040640725
# tornado (10686.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0035944194372653252
# tornado (10686.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005311711760930115
# tornado (10686.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.005424535516097632
# tornado (10686.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005294922481830561
# tornado (10686.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.005155322345525056
# tornado (10686.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004562450793774357
# tornado (10686.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.021088228030201668
# tornado (10686.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.020687564723253747
# tornado (10686.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.020153487913218842
# tornado (10686.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.019025915567114356
# tornado (10686.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016494925452679016
# tornado (10686.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.008649685564662111
# tornado (10686.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00863691044466454
# tornado (10686.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.00842171997331245
# tornado (10686.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.008114065042289892
# tornado (10686.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007277489385144834
# tornado (10686.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.003922106264257418
# tornado (10686.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0038452195419003154
# tornado (10686.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0037729988802290156
# tornado (10686.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0036281963137705176
# tornado (10686.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003290707949569142
# tornado (10686.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.003964722175670912
# tornado (10686.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.003915111580325947
# tornado (10686.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003822253443846411
# tornado (10686.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0036572720744306826
# tornado (10686.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003265679658019375
# tornado (10686.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00012127423817360149
# wind (80396.0)        feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.04549166280039331
# wind (80396.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.04526618122440255
# wind (80396.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.04466102978105957
# wind (80396.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.04307342613912038
# wind (80396.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.03900833394815928
# wind (80396.0)        feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.09339634521048745
# wind (80396.0)        feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.09516007592939708 ** best wind***
# wind (80396.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.09502641382951704
# wind (80396.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.09260719392976388
# wind (80396.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.08594016367926767
# wind (80396.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04692488487673821
# wind (80396.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.04665027304304932
# wind (80396.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.04604342043170001
# wind (80396.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0446519945187424
# wind (80396.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04122880990283079
# wind (80396.0)        feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.024757989285954716
# wind (80396.0)        feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.02401906439891135
# wind (80396.0)        feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.023430566618794103
# wind (80396.0)        feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.022388459721489874
# wind (80396.0)        feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.020212378380209752
# wind (80396.0)        feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.031937428502707046
# wind (80396.0)        feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.03172737574742659
# wind (80396.0)        feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03128940483586756
# wind (80396.0)        feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.030276745647836514
# wind (80396.0)        feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02758202355107999
# wind (80396.0)        feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05603139977135943
# wind (80396.0)        feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.05634965292080136
# wind (80396.0)        feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.05585205836532619
# wind (80396.0)        feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.05440799902687542
# wind (80396.0)        feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.050598539161717714
# wind (80396.0)        feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03358064310955259
# wind (80396.0)        feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.03363988814486978
# wind (80396.0)        feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03336472468356079
# wind (80396.0)        feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.03269079094017106
# wind (80396.0)        feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03076972043598155
# wind (80396.0)        feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.016792241043092283
# wind (80396.0)        feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0163455180732563
# wind (80396.0)        feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015981132593348463
# wind (80396.0)        feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015355208256868202
# wind (80396.0)        feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01406543993052253
# wind (80396.0)        feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.000894725430960612
# wind_adj (24503.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.007518613139405525
# wind_adj (24503.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.007342979622055159
# wind_adj (24503.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.007168601016374212
# wind_adj (24503.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.006873274965127069
# wind_adj (24503.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.006186514902176137
# wind_adj (24503.0)    feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.019181171353789175
# wind_adj (24503.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.018856621077821392
# wind_adj (24503.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.018465901060023653
# wind_adj (24503.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.01762648129695651
# wind_adj (24503.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015765207991967744
# wind_adj (24503.0)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04641216279640385
# wind_adj (24503.0)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.04763594601627121 ***best wind_adj***
# wind_adj (24503.0)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.04762447824755166
# wind_adj (24503.0)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.04722913902978673
# wind_adj (24503.0)    feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04525971020853071
# wind_adj (24503.0)    feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012687477101755123
# wind_adj (24503.0)    feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.012368432715620535
# wind_adj (24503.0)    feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.012103280741170367
# wind_adj (24503.0)    feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.011541029790295016
# wind_adj (24503.0)    feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010380190985357664
# wind_adj (24503.0)    feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.006016651740047032
# wind_adj (24503.0)    feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00588461296504909
# wind_adj (24503.0)    feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005749781138195477
# wind_adj (24503.0)    feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.005528268230126357
# wind_adj (24503.0)    feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005027767484501144
# wind_adj (24503.0)    feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02619038544588339
# wind_adj (24503.0)    feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.02651992851022336
# wind_adj (24503.0)    feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.026354093003992017
# wind_adj (24503.0)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.025951245453708496
# wind_adj (24503.0)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.024328453025414445
# wind_adj (24503.0)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03859463060095839
# wind_adj (24503.0)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.03959995919600432
# wind_adj (24503.0)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03962761822439627
# wind_adj (24503.0)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.03953609218509199
# wind_adj (24503.0)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03832507134410741
# wind_adj (24503.0)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010532661300325414
# wind_adj (24503.0)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.010349288582887257
# wind_adj (24503.0)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.010164334429431158
# wind_adj (24503.0)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.009794394516249956
# wind_adj (24503.0)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.008990781143395087
# wind_adj (24503.0)    feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.0002676237192161264
# hail (36833.0)        feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.014073554755664285
# hail (36833.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.01375083821036267
# hail (36833.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.013450447785145604
# hail (36833.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.012877164288605575
# hail (36833.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.011612073453822543
# hail (36833.0)        feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.015281247691360151
# hail (36833.0)        feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.014960512512822578
# hail (36833.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.014623876321738867
# hail (36833.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.014017569126682882
# hail (36833.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012750677875134384
# hail (36833.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.017959643764341338
# hail (36833.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.017683826327730304
# hail (36833.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01736050203337848
# hail (36833.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.016707941084732243
# hail (36833.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015295441432974278
# hail (36833.0)        feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05940978948648458 ***best hail***
# hail (36833.0)        feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.05833051151988928
# hail (36833.0)        feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.05685697934996585
# hail (36833.0)        feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.05314947419793287
# hail (36833.0)        feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04634916129938539
# hail (36833.0)        feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.009487997805112974
# hail (36833.0)        feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.009317602833215059
# hail (36833.0)        feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.00914905666153729
# hail (36833.0)        feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.008833732895922833
# hail (36833.0)        feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.008127598086420893
# hail (36833.0)        feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0161579096950482
# hail (36833.0)        feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015936482910774156
# hail (36833.0)        feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015666508725712507
# hail (36833.0)        feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01514847367685818
# hail (36833.0)        feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013934691954497534
# hail (36833.0)        feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014490482891193035
# hail (36833.0)        feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01429390528719185
# hail (36833.0)        feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014078757237108042
# hail (36833.0)        feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01362301991699066
# hail (36833.0)        feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012540450672085871
# hail (36833.0)        feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.04413193121469798
# hail (36833.0)        feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.04328332557314879
# hail (36833.0)        feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.042206495729699325
# hail (36833.0)        feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.039499281929620954
# hail (36833.0)        feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0344680712660775
# hail (36833.0)        feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00040571509629665984
# sig_tornado (1637.0)  feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.01430651240090263
# sig_tornado (1637.0)  feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.014831839711711014
# sig_tornado (1637.0)  feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.01531654311290099
# sig_tornado (1637.0)  feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.01480421688229282
# sig_tornado (1637.0)  feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.013473919205140955
# sig_tornado (1637.0)  feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.001982502438055496
# sig_tornado (1637.0)  feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.0021467656251314797
# sig_tornado (1637.0)  feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.0022544187664504723
# sig_tornado (1637.0)  feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.0021059492089885628
# sig_tornado (1637.0)  feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0019343395222238414
# sig_tornado (1637.0)  feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0009475290164619642
# sig_tornado (1637.0)  feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0009438947986207365
# sig_tornado (1637.0)  feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0009282088675978063
# sig_tornado (1637.0)  feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0008799362782317025
# sig_tornado (1637.0)  feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00077886531475817
# sig_tornado (1637.0)  feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0015860043605678844
# sig_tornado (1637.0)  feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0017355521055445779
# sig_tornado (1637.0)  feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.001747680848103543
# sig_tornado (1637.0)  feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0017989495942073292 ***best sigtor***
# sig_tornado (1637.0)  feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0016315182286133817
# sig_tornado (1637.0)  feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01568126479178295
# sig_tornado (1637.0)  feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015404555453958468
# sig_tornado (1637.0)  feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015090986599249418
# sig_tornado (1637.0)  feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01400874107450962
# sig_tornado (1637.0)  feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011840905396683825
# sig_tornado (1637.0)  feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0024854653308733136
# sig_tornado (1637.0)  feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0024923750984156773
# sig_tornado (1637.0)  feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0024569076717458663
# sig_tornado (1637.0)  feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0023384283261038987
# sig_tornado (1637.0)  feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0021145560023228096
# sig_tornado (1637.0)  feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0008866104450901806
# sig_tornado (1637.0)  feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0008575552421719564
# sig_tornado (1637.0)  feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0008317791874217267
# sig_tornado (1637.0)  feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0007851311064709801
# sig_tornado (1637.0)  feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.000696612955366715
# sig_tornado (1637.0)  feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0009455127132239787
# sig_tornado (1637.0)  feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0009495148262954361
# sig_tornado (1637.0)  feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.000946855371467743
# sig_tornado (1637.0)  feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0009131996156227875
# sig_tornado (1637.0)  feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0008421614570751103
# sig_tornado (1637.0)  feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 1.911416655095588e-5
# sig_wind (7943.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.005505888028709689
# sig_wind (7943.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.005443005944059548
# sig_wind (7943.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.005318819727504212
# sig_wind (7943.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.005162354786763048
# sig_wind (7943.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.004659744706485199
# sig_wind (7943.0)     feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.009014028030210269
# sig_wind (7943.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.00902456133905611
# sig_wind (7943.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.00884042935890506
# sig_wind (7943.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.008579695030621512
# sig_wind (7943.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007766477115247189
# sig_wind (7943.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013648771030030262
# sig_wind (7943.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0142441270891229
# sig_wind (7943.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014379343447368293
# sig_wind (7943.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014386853762729114 ***best sigwind***
# sig_wind (7943.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013824446252495431
# sig_wind (7943.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004036358856830791
# sig_wind (7943.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.003928398763287889
# sig_wind (7943.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003818923332792384
# sig_wind (7943.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0036603946601924046
# sig_wind (7943.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0032673781028023296
# sig_wind (7943.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004350974268963236
# sig_wind (7943.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.004340423105740298
# sig_wind (7943.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.004290872646193996
# sig_wind (7943.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0041808357240272365
# sig_wind (7943.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0038458877763751145
# sig_wind (7943.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012027123090896815
# sig_wind (7943.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.012203279671603805 (not best sigwind)
# sig_wind (7943.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.012136276659061663
# sig_wind (7943.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.012031589064954232
# sig_wind (7943.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011333586987892446
# sig_wind (7943.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012256712396563505
# sig_wind (7943.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.012677892957117923
# sig_wind (7943.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.012694378738975646
# sig_wind (7943.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.012867326742684437
# sig_wind (7943.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01259575493798323
# sig_wind (7943.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0033365424758469214
# sig_wind (7943.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.003253558540683329
# sig_wind (7943.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003174404518188377
# sig_wind (7943.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0030513688067911795
# sig_wind (7943.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027891534547807724
# sig_wind (7943.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 8.772770172793049e-5
# sig_wind_adj (2799.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0010006174027265059
# sig_wind_adj (2799.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.0009788890801545353
# sig_wind_adj (2799.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.0009593931258287666
# sig_wind_adj (2799.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.0009305534111104819
# sig_wind_adj (2799.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.0008489255499141925
# sig_wind_adj (2799.0) feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.003054727245079578
# sig_wind_adj (2799.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.0030342872340094647
# sig_wind_adj (2799.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.002989088170093843
# sig_wind_adj (2799.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.002877781979470108
# sig_wind_adj (2799.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0025903916415230413
# sig_wind_adj (2799.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012375790051327823
# sig_wind_adj (2799.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01363900983838428
# sig_wind_adj (2799.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01420761906008643
# sig_wind_adj (2799.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014709683290543731
# sig_wind_adj (2799.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01503049887572014
# sig_wind_adj (2799.0) feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0018188286613641982
# sig_wind_adj (2799.0) feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0017802240385975032
# sig_wind_adj (2799.0) feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0017427301599658462
# sig_wind_adj (2799.0) feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0016659772570859698
# sig_wind_adj (2799.0) feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.001490543639255167
# sig_wind_adj (2799.0) feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0008940739562018051
# sig_wind_adj (2799.0) feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0008737570787094186
# sig_wind_adj (2799.0) feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0008557953281063191
# sig_wind_adj (2799.0) feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0008243550807100619
# sig_wind_adj (2799.0) feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0007527648622627723
# sig_wind_adj (2799.0) feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005632034475431711
# sig_wind_adj (2799.0) feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.005714106573070175
# sig_wind_adj (2799.0) feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005728679092664684
# sig_wind_adj (2799.0) feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.005697101797584719
# sig_wind_adj (2799.0) feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005443828581188869
# sig_wind_adj (2799.0) feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013182671037487577
# sig_wind_adj (2799.0) feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.014249333092947764
# sig_wind_adj (2799.0) feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01448964659990564
# sig_wind_adj (2799.0) feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015332594199904976
# sig_wind_adj (2799.0) feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01592456556064055 *** best sig_wind_adj***
# sig_wind_adj (2799.0) feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017017144216155016
# sig_wind_adj (2799.0) feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0016527416096054342
# sig_wind_adj (2799.0) feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0016183076224663272
# sig_wind_adj (2799.0) feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0015549517144570067
# sig_wind_adj (2799.0) feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014259375295685615
# sig_wind_adj (2799.0) feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 3.0117466380601622e-5
# sig_hail (4395.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0027102633427632332
# sig_hail (4395.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.002637181521945838
# sig_hail (4395.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.00255577558110596
# sig_hail (4395.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.002404661698260406
# sig_hail (4395.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.0020819673742118495
# sig_hail (4395.0)     feature 6 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.001953117243154732
# sig_hail (4395.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.0019099614949305494
# sig_hail (4395.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.0018562933062651805
# sig_hail (4395.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.001772045915071798
# sig_hail (4395.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0016008941430076713
# sig_hail (4395.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032262769767971685
# sig_hail (4395.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0031973207117989427
# sig_hail (4395.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0031313089228836993
# sig_hail (4395.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.003030047586805547
# sig_hail (4395.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027702053637899257
# sig_hail (4395.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014790013239820763
# sig_hail (4395.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.014835580283037976
# sig_hail (4395.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014355203327260811
# sig_hail (4395.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.013766711491028392
# sig_hail (4395.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012490493111130865
# sig_hail (4395.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017389247904227355
# sig_hail (4395.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0016946598054366233
# sig_hail (4395.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0016536839654499684
# sig_hail (4395.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0015749781263582315
# sig_hail (4395.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014154409464524102
# sig_hail (4395.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002871480472750503
# sig_hail (4395.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.002831796633115907
# sig_hail (4395.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0027853271341658843
# sig_hail (4395.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.002711961686716909
# sig_hail (4395.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002510658224852951
# sig_hail (4395.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0029839990245756273
# sig_hail (4395.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0029544639348248964
# sig_hail (4395.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.002915050075040498
# sig_hail (4395.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0028447849667623927
# sig_hail (4395.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0026361586687641872
# sig_hail (4395.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.016152462858350786 ***best sighail***
# sig_hail (4395.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015981744395316896
# sig_hail (4395.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015466764010594125
# sig_hail (4395.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014425821154809835
# sig_hail (4395.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012368202248699322
# sig_hail (4395.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 4.819751699337258e-5

# tornado (10686.0)     feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.02886784 ***best tor***
# tornado (10686.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.028257657
# tornado (10686.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.027373588
# tornado (10686.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.025957579
# tornado (10686.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.021980641
# tornado (10686.0)     feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.0089139
# tornado (10686.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.008971877
# tornado (10686.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.008736457
# tornado (10686.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.008366387
# tornado (10686.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0071543013
# tornado (10686.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004470979
# tornado (10686.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.004373885
# tornado (10686.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.004261612
# tornado (10686.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0040633962
# tornado (10686.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0035969764
# tornado (10686.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005336669
# tornado (10686.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00544209
# tornado (10686.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005309541
# tornado (10686.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0051662032
# tornado (10686.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004569754
# tornado (10686.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.021116732
# tornado (10686.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.020713918
# tornado (10686.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.020177336
# tornado (10686.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.019048253
# tornado (10686.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016513726
# tornado (10686.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.008672547
# tornado (10686.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.008657047
# tornado (10686.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.008436934
# tornado (10686.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.008128993
# tornado (10686.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0072897566
# tornado (10686.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0039244588
# tornado (10686.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0038474936
# tornado (10686.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0037751244
# tornado (10686.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0036302106
# tornado (10686.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0032923939
# tornado (10686.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0039741076
# tornado (10686.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0039230688
# tornado (10686.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003829446
# tornado (10686.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.003663114
# tornado (10686.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0032701474
# tornado (10686.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00012347198
# wind (80396.0)        feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.04550908
# wind (80396.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.045283865
# wind (80396.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.04467888
# wind (80396.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.04309097
# wind (80396.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.03902438
# wind (80396.0)        feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.093413524
# wind (80396.0)        feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.09517683 ***best wind***
# wind (80396.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.09504187
# wind (80396.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.09262293
# wind (80396.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.08595775
# wind (80396.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.046938602
# wind (80396.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.046664733
# wind (80396.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.04606048
# wind (80396.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.044668585
# wind (80396.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.041243155
# wind (80396.0)        feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.024761712
# wind (80396.0)        feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.024022376
# wind (80396.0)        feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0234338
# wind (80396.0)        feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.022392042
# wind (80396.0)        feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02021586
# wind (80396.0)        feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.031951107
# wind (80396.0)        feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.03174505
# wind (80396.0)        feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.031302955
# wind (80396.0)        feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.030293606
# wind (80396.0)        feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.027594553
# wind (80396.0)        feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.056042116
# wind (80396.0)        feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.05636142
# wind (80396.0)        feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.055863343
# wind (80396.0)        feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.05442058
# wind (80396.0)        feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.05061
# wind (80396.0)        feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.033596367
# wind (80396.0)        feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.033655122
# wind (80396.0)        feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03337999
# wind (80396.0)        feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.032704215
# wind (80396.0)        feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.030783625
# wind (80396.0)        feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.016794698
# wind (80396.0)        feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.016347554
# wind (80396.0)        feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015983103
# wind (80396.0)        feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015356969
# wind (80396.0)        feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014066994
# wind (80396.0)        feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.0009072825
# wind_adj (24503.0)    feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.0075220764
# wind_adj (24503.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.007345924
# wind_adj (24503.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.007170853
# wind_adj (24503.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.0068753604
# wind_adj (24503.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.006187986
# wind_adj (24503.0)    feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.01918736
# wind_adj (24503.0)    feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.018862486
# wind_adj (24503.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.01847108
# wind_adj (24503.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.017631713
# wind_adj (24503.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015769187
# wind_adj (24503.0)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.046450417
# wind_adj (24503.0)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.047676966 ***best wind_adj***
# wind_adj (24503.0)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.04767443
# wind_adj (24503.0)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.047277834
# wind_adj (24503.0)    feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.045301344
# wind_adj (24503.0)    feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012694488
# wind_adj (24503.0)    feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.012373864
# wind_adj (24503.0)    feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0121086985
# wind_adj (24503.0)    feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.011546822
# wind_adj (24503.0)    feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010386267
# wind_adj (24503.0)    feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0060179904
# wind_adj (24503.0)    feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0058858357
# wind_adj (24503.0)    feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005750899
# wind_adj (24503.0)    feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.005529301
# wind_adj (24503.0)    feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.005028606
# wind_adj (24503.0)    feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.026199633
# wind_adj (24503.0)    feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.026528554
# wind_adj (24503.0)    feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.026362458
# wind_adj (24503.0)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.025959253
# wind_adj (24503.0)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.024335416
# wind_adj (24503.0)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.038642116
# wind_adj (24503.0)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.039646093
# wind_adj (24503.0)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.039673924
# wind_adj (24503.0)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.039576545
# wind_adj (24503.0)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.038367096
# wind_adj (24503.0)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01053688
# wind_adj (24503.0)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.010352783
# wind_adj (24503.0)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.010167602
# wind_adj (24503.0)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.009797234
# wind_adj (24503.0)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.008993104
# wind_adj (24503.0)    feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.0002714251
# hail (36833.0)        feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.014094461
# hail (36833.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.013771825
# hail (36833.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.01347093
# hail (36833.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.012897317
# hail (36833.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.011629823
# hail (36833.0)        feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.0152853895
# hail (36833.0)        feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.014963279
# hail (36833.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.014626511
# hail (36833.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.01401997
# hail (36833.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012752835
# hail (36833.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.017965587
# hail (36833.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.017689042
# hail (36833.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.017365534
# hail (36833.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01671287
# hail (36833.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015300179
# hail (36833.0)        feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.059443332 ***best hail***
# hail (36833.0)        feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.05836339
# hail (36833.0)        feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.05688998
# hail (36833.0)        feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.053175364
# hail (36833.0)        feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.046369996
# hail (36833.0)        feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.009490936
# hail (36833.0)        feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.009320329
# hail (36833.0)        feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.009151697
# hail (36833.0)        feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0088361055
# hail (36833.0)        feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0081297
# hail (36833.0)        feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01616297
# hail (36833.0)        feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015940925
# hail (36833.0)        feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015670758
# hail (36833.0)        feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01515217
# hail (36833.0)        feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013937571
# hail (36833.0)        feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0144941
# hail (36833.0)        feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01429705
# hail (36833.0)        feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014081716
# hail (36833.0)        feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0136257345
# hail (36833.0)        feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012542781
# hail (36833.0)        feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.044158667
# hail (36833.0)        feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.043309845
# hail (36833.0)        feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0422359
# hail (36833.0)        feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.039527375
# hail (36833.0)        feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.034497727
# hail (36833.0)        feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00041131355
# sig_tornado (1637.0)  feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.014760357
# sig_tornado (1637.0)  feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.0152919
# sig_tornado (1637.0)  feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.015758388 ***best sigtor***
# sig_tornado (1637.0)  feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.015242966
# sig_tornado (1637.0)  feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.013855155
# sig_tornado (1637.0)  feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.0020129248
# sig_tornado (1637.0)  feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.002187792
# sig_tornado (1637.0)  feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.002323431
# sig_tornado (1637.0)  feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.0021772068
# sig_tornado (1637.0)  feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0019961465
# sig_tornado (1637.0)  feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0009505071
# sig_tornado (1637.0)  feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0009468432
# sig_tornado (1637.0)  feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0009310448
# sig_tornado (1637.0)  feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0008825642
# sig_tornado (1637.0)  feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0007813011
# sig_tornado (1637.0)  feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0015962254
# sig_tornado (1637.0)  feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0017510995
# sig_tornado (1637.0)  feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0017624598
# sig_tornado (1637.0)  feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0018181723
# sig_tornado (1637.0)  feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0016466351
# sig_tornado (1637.0)  feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015747255 (not best sigtor)
# sig_tornado (1637.0)  feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01546682
# sig_tornado (1637.0)  feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0151535105
# sig_tornado (1637.0)  feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014065619
# sig_tornado (1637.0)  feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011890301
# sig_tornado (1637.0)  feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0024973927
# sig_tornado (1637.0)  feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0025042722
# sig_tornado (1637.0)  feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0024684113
# sig_tornado (1637.0)  feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0023472512
# sig_tornado (1637.0)  feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0021213507
# sig_tornado (1637.0)  feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0008889513
# sig_tornado (1637.0)  feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00085968623
# sig_tornado (1637.0)  feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0008337088
# sig_tornado (1637.0)  feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0007868963
# sig_tornado (1637.0)  feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0006981124
# sig_tornado (1637.0)  feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.000949269
# sig_tornado (1637.0)  feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0009534998
# sig_tornado (1637.0)  feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.00095091824
# sig_tornado (1637.0)  feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0009173759
# sig_tornado (1637.0)  feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0008458966
# sig_tornado (1637.0)  feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 1.9716565e-5
# sig_wind (7943.0)     feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.005512643
# sig_wind (7943.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.005449083
# sig_wind (7943.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.005324171
# sig_wind (7943.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.0051673944
# sig_wind (7943.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.004663894
# sig_wind (7943.0)     feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.009023562
# sig_wind (7943.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.00903241
# sig_wind (7943.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.008847869
# sig_wind (7943.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.008587244
# sig_wind (7943.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0077737058
# sig_wind (7943.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013723137
# sig_wind (7943.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.014333949
# sig_wind (7943.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014499431
# sig_wind (7943.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014507355 ***best sigwind***
# sig_wind (7943.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013924452
# sig_wind (7943.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004046195
# sig_wind (7943.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.003936334
# sig_wind (7943.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003826392
# sig_wind (7943.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0036705113
# sig_wind (7943.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0032762128
# sig_wind (7943.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004355683
# sig_wind (7943.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.004344841
# sig_wind (7943.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0042950884
# sig_wind (7943.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.004185053
# sig_wind (7943.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0038496507
# sig_wind (7943.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01204185
# sig_wind (7943.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.012217627
# sig_wind (7943.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.012150323
# sig_wind (7943.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.012044909
# sig_wind (7943.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011344552
# sig_wind (7943.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012363003
# sig_wind (7943.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0127855465
# sig_wind (7943.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01280437
# sig_wind (7943.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.012963097
# sig_wind (7943.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012696069 (not best sigwind)
# sig_wind (7943.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0033409002
# sig_wind (7943.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0032570488
# sig_wind (7943.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003177601
# sig_wind (7943.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0030541203
# sig_wind (7943.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027913041
# sig_wind (7943.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 8.91269e-5
# sig_wind_adj (2799.0) feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.0010022675
# sig_wind_adj (2799.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.0009805098
# sig_wind_adj (2799.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.00096092245
# sig_wind_adj (2799.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.00093209854
# sig_wind_adj (2799.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.00085023127
# sig_wind_adj (2799.0) feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.0030598487
# sig_wind_adj (2799.0) feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.0030389691
# sig_wind_adj (2799.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.002993749
# sig_wind_adj (2799.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.002882249
# sig_wind_adj (2799.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002594174
# sig_wind_adj (2799.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012551806
# sig_wind_adj (2799.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.013859353
# sig_wind_adj (2799.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014514721
# sig_wind_adj (2799.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015018062
# sig_wind_adj (2799.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015279751
# sig_wind_adj (2799.0) feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.001834195
# sig_wind_adj (2799.0) feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0017859178
# sig_wind_adj (2799.0) feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0017474432
# sig_wind_adj (2799.0) feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.001669711
# sig_wind_adj (2799.0) feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014931717
# sig_wind_adj (2799.0) feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.00089511817
# sig_wind_adj (2799.0) feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00087473966
# sig_wind_adj (2799.0) feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0008567712
# sig_wind_adj (2799.0) feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.00082530733
# sig_wind_adj (2799.0) feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00075364456
# sig_wind_adj (2799.0) feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005647589
# sig_wind_adj (2799.0) feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0057284436
# sig_wind_adj (2799.0) feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005743401
# sig_wind_adj (2799.0) feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.00571211
# sig_wind_adj (2799.0) feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0054563214
# sig_wind_adj (2799.0) feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01347219
# sig_wind_adj (2799.0) feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.014539996
# sig_wind_adj (2799.0) feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014786454
# sig_wind_adj (2799.0) feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0155849075
# sig_wind_adj (2799.0) feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016188215 ***best sig_wind_adj***
# sig_wind_adj (2799.0) feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017083118
# sig_wind_adj (2799.0) feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.001657572
# sig_wind_adj (2799.0) feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.001622609
# sig_wind_adj (2799.0) feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0015584171
# sig_wind_adj (2799.0) feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014284011
# sig_wind_adj (2799.0) feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 3.055796e-5
# sig_hail (4395.0)     feature 1 TORPROB:calculated:hour   fcst:calculated_prob:           AU-PR-curve: 0.0027192952
# sig_hail (4395.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:35mi    mean AU-PR-curve: 0.002647511
# sig_hail (4395.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.0025649886
# sig_hail (4395.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:70mi    mean AU-PR-curve: 0.0024139204
# sig_hail (4395.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:100mi   mean AU-PR-curve: 0.0020862897
# sig_hail (4395.0)     feature 6 WINDPROB:calculated:hour  fcst:calculated_prob:           AU-PR-curve: 0.0019556084
# sig_hail (4395.0)     feature 7 WINDPROB:calculated:hour fcst:calculated_prob:35mi   mean AU-PR-curve: 0.0019122335
# sig_hail (4395.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:50mi   mean AU-PR-curve: 0.001858434
# sig_hail (4395.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:70mi   mean AU-PR-curve: 0.0017739723
# sig_hail (4395.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.001602497
# sig_hail (4395.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032325778
# sig_hail (4395.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0032039506
# sig_hail (4395.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0031383429
# sig_hail (4395.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0030377666
# sig_hail (4395.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027776845
# sig_hail (4395.0)     feature 16 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0148871755
# sig_hail (4395.0)     feature 17 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01491283
# sig_hail (4395.0)     feature 18 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014420448
# sig_hail (4395.0)     feature 19 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.013833471
# sig_hail (4395.0)     feature 20 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012566853
# sig_hail (4395.0)     feature 21 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0017414918
# sig_hail (4395.0)     feature 22 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0016970313
# sig_hail (4395.0)     feature 23 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0016559977
# sig_hail (4395.0)     feature 24 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0015770365
# sig_hail (4395.0)     feature 25 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0014173018
# sig_hail (4395.0)     feature 26 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002880064
# sig_hail (4395.0)     feature 27 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0028390335
# sig_hail (4395.0)     feature 28 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.002792358
# sig_hail (4395.0)     feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0027182149
# sig_hail (4395.0)     feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0025152792
# sig_hail (4395.0)     feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002989897
# sig_hail (4395.0)     feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0029594514
# sig_hail (4395.0)     feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.002919753
# sig_hail (4395.0)     feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0028492496
# sig_hail (4395.0)     feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00263985
# sig_hail (4395.0)     feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.016200982 ***best sighail***
# sig_hail (4395.0)     feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.016030649
# sig_hail (4395.0)     feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.015519087
# sig_hail (4395.0)     feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014474395
# sig_hail (4395.0)     feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012430134
# sig_hail (4395.0)     feature 41 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 4.9210783e-5




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
# 0       0       0.02886784
# 0       35      0.02892696
# 0       50      0.028880775
# 0       70      0.02870506
# 0       100     0.028099779
# 35      0       0.028355595
# 35      35      0.028257657
# 35      50      0.028175287
# 35      70      0.027951468
# 35      100     0.027310172
# 50      0       0.027633166
# 50      35      0.02752214
# 50      50      0.027373588
# 50      70      0.027076522
# 50      100     0.026324878
# 70      0       0.026716607
# 70      35      0.026550317
# 70      50      0.026363844
# 70      70      0.025957579
# 70      100     0.025020616
# 100     0       0.02406251
# 100     35      0.02382601
# 100     50      0.02358275
# 100     70      0.023258822
# 100     100     0.021980641
# Best tornado: 0 35      0.02892696

# blur_radius_f2  blur_radius_f38 AU_PR_wind
# 0       0       0.093413524
# 0       35      0.09441722
# 0       50      0.09452699
# 0       70      0.094134964
# 0       100     0.09307711
# 35      0       0.09469125
# 35      35      0.09517683
# 35      50      0.09525015
# 35      70      0.09476187
# 35      100     0.09365039
# 50      0       0.094680354
# 50      35      0.09510661
# 50      50      0.09504187
# 50      70      0.09444839
# 50      100     0.09315806
# 70      0       0.09312559
# 70      35      0.093508475
# 70      50      0.09347098
# 70      70      0.09262293
# 70      100     0.09103005
# 100     0       0.0895945
# 100     35      0.089854054
# 100     50      0.08954784
# 100     70      0.08830738
# 100     100     0.08595776
# Best wind: 35   50      0.09525015

# blur_radius_f2  blur_radius_f38 AU_PR_wind_adj
# 0       0       0.046450417
# 0       35      0.04700118
# 0       50      0.04706072
# 0       70      0.047055595
# 0       100     0.04687252
# 35      0       0.04749802
# 35      35      0.047676966
# 35      50      0.047691677
# 35      70      0.04762277
# 35      100     0.047411308
# 50      0       0.04757678
# 50      35      0.04774611
# 50      50      0.047674417
# 50      70      0.047540195
# 50      100     0.04724601
# 70      0       0.04757133
# 70      35      0.047704108
# 70      50      0.047585417
# 70      70      0.047277834
# 70      100     0.04680666
# 100     0       0.046510953
# 100     35      0.046923783
# 100     50      0.046760954
# 100     70      0.046290584
# 100     100     0.04530133
# Best wind_adj: 50       35      0.04774611

# blur_radius_f2  blur_radius_f38 AU_PR_hail
# 0       0       0.059443332
# 0       35      0.059553593
# 0       50      0.059410267
# 0       70      0.058882035
# 0       100     0.058041345
# 35      0       0.05864445
# 35      35      0.05836339
# 35      50      0.058148004
# 35      70      0.05745334
# 35      100     0.056488242
# 50      0       0.05754262
# 50      35      0.057219427
# 50      50      0.05688998
# 50      70      0.056098394
# 50      100     0.054968353
# 70      0       0.055135135
# 70      35      0.054623764
# 70      50      0.05423152
# 70      70      0.053175364
# 70      100     0.05175181
# 100     0       0.051144976
# 100     35      0.05033756
# 100     50      0.04979259
# 100     70      0.04842869
# 100     100     0.046369996
# Best hail: 0    35      0.059553593

# blur_radius_f2  blur_radius_f38 AU_PR_sig_tornado
# 0       0       0.015747271
# 0       35      0.015818305
# 0       50      0.01586604
# 0       70      0.015722184
# 0       100     0.015309504
# 35      0       0.015487565
# 35      35      0.01546682
# 35      50      0.015471228
# 35      70      0.015283476
# 35      100     0.0148841925
# 50      0       0.015258747
# 50      35      0.015195998
# 50      50      0.0151535105
# 50      70      0.01493396
# 50      100     0.014505962
# 70      0       0.014593962
# 70      35      0.014474618
# 70      50      0.014381975
# 70      70      0.014065619
# 70      100     0.013495695
# 100     0       0.013579775
# 100     35      0.01338934
# 100     50      0.0132344635
# 100     70      0.012782078
# 100     100     0.011890301
# Best sig_tornado: 0     50      0.01586604

# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind
# 0       0       0.01204185
# 0       35      0.012225352
# 0       50      0.012266053
# 0       70      0.012349707
# 0       100     0.0123755885
# 35      0       0.012100015
# 35      35      0.012217627
# 35      50      0.012246445
# 35      70      0.012315949
# 35      100     0.0123342555
# 50      0       0.012037552
# 50      35      0.012140923
# 50      50      0.012150323
# 50      70      0.01220783
# 50      100     0.012199542
# 70      0       0.011940993
# 70      35      0.012025476
# 70      50      0.012019193
# 70      70      0.012044909
# 70      100     0.011992096
# 100     0       0.011544015
# 100     35      0.011600689
# 100     50      0.011559464
# 100     70      0.011533889
# 100     100     0.011344552
# Best sig_wind: 0        100     0.0123755885


# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind_adj
# 0       0       0.01347219
# 0       35      0.013800037
# 0       50      0.013893111
# 0       70      0.014022983
# 0       100     0.014214433
# 35      0       0.01431585
# 35      35      0.014539996
# 35      50      0.014742265
# 35      70      0.014740945
# 35      100     0.014961938
# 50      0       0.014549286
# 50      35      0.014707592
# 50      50      0.014786454
# 50      70      0.015031936
# 50      100     0.015134199
# 70      0       0.014966261
# 70      35      0.015367647
# 70      50      0.015451628
# 70      70      0.0155849075
# 70      100     0.015793735
# 100     0       0.015517541
# 100     35      0.015840136
# 100     50      0.01592401
# 100     70      0.01605181
# 100     100     0.016188215
# Best sig_wind_adj: 100  100     0.016188215

# blur_radius_f2  blur_radius_f38 AU_PR_sig_hail
# 0       0       0.016200982
# 0       35      0.016264368
# 0       50      0.016202657
# 0       70      0.016047658
# 0       100     0.015771447
# 35      0       0.016161673
# 35      35      0.01603066
# 35      50      0.015933473
# 35      70      0.015723698
# 35      100     0.015409165
# 50      0       0.015855143
# 50      35      0.015665317
# 50      50      0.015519087
# 50      70      0.015260816
# 50      100     0.014892859
# 70      0       0.015317029
# 70      35      0.015030983
# 70      50      0.0148308035
# 70      70      0.014474395
# 70      100     0.014003808
# 100     0       0.014445817
# 100     35      0.013970537
# 100     50      0.013686227
# 100     70      0.013153218
# 100     100     0.012430137
# Best sig_hail: 0        35      0.016264368


println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f38\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name   best_blur_radius_f2 best_blur_radius_f38 AU_PR
# tornado      0                   35                   0.02892696
# wind         35                  50                   0.09525015
# wind_adj     50                  35                   0.04774611
# hail         0                   35                   0.059553593
# sig_tornado  0                   50                   0.01586604
# sig_wind     0                   100                  0.0123755885
# sig_wind_adj 100                 100                  0.016188215
# sig_hail     0                   35                   0.016264368



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
cutoff = Dates.DateTime(2022, 6, 1, 12)
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
    au_pr_curve = Float32(Metrics.area_under_pr_curve(x, y, weights))
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_blurred, X, Ys, weights)

# EXPECTED:
# event_name   AU_PR
# tornado      0.02892696
# wind         0.09525015
# wind_adj     0.04774611
# hail         0.059553593
# sig_tornado  0.01586604
# sig_wind     0.0123755885
# sig_wind_adj 0.016188215
# sig_hail     0.016264368

# ACTUAL:
# tornado (10686.0)     feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.02892696
# wind (80396.0)        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.09525015
# wind_adj (24503.0)    feature 3 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.04774611
# hail (36833.0)        feature 4 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.059553593
# sig_tornado (1637.0)  feature 5 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.01586604
# sig_wind (7943.0)     feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.0123755885
# sig_wind_adj (2799.0) feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016188215
# sig_hail (4395.0)     feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016264368

# Yay!