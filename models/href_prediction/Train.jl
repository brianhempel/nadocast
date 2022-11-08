# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.


import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# disk2_date = Dates.DateTime(2021, 3, 1, 0)
# validation_forecasts = filter(forecast -> Forecasts.run_utc_datetime(forecast) >= disk2_date, validation_forecasts);

# for testing
# validation_forecasts = rand(validation_forecasts, 30);

# validation_forecasts1 = filter(forecast -> isodd(forecast.run_day),  validation_forecasts);
# validation_forecasts2 = filter(forecast -> iseven(forecast.run_day), validation_forecasts);

# # rm("validation_forecasts_with_blurs_and_forecast_hour1"; recursive=true)
# # rm("validation_forecasts_with_blurs_and_forecast_hour2"; recursive=true)

# # To double loading speed, manually run the other one of these in a separate process with USE_ALT_DISK=true
# # When it's done, run it in the main process and it will load from the save_dir

# function dictmap(f, dict)
#   out = Dict()
#   for (k, v) in dict
#     out[k] = f(v)
#   end
#   out
# end

# if get(ENV, "USE_ALT_DISK", "false") != "true"
#   X1, Ys1, weights1 =
#     TrainingShared.get_data_labels_weights(
#       validation_forecasts1;
#       event_name_to_labeler = TrainingShared.event_name_to_labeler,
#       save_dir = "validation_forecasts_with_blurs_and_forecast_hour1"
#     );

#   Ys1 = dictmap(y -> y .> 0.5, Ys1) # Convert to bitarrays. This saves memory
#   # Then wait for the X2, Ys2, weights2 to finish in the other process, then continue.
# end

# # blur_radii = [0; HREFPrediction.blur_radii]
# # X1 = X1[:,1:2*length(blur_radii)]

# GC.gc()

# X2, Ys2, weights2 =
#   TrainingShared.get_data_labels_weights(
#     validation_forecasts2;
#     event_name_to_labeler = TrainingShared.event_name_to_labeler,
#     save_dir = "validation_forecasts_with_blurs_and_forecast_hour2"
#   );
# Ys2 = dictmap(y -> y .> 0.5, Ys2) # Convert to bitarrays. This saves memory

# GC.gc()

# if get(ENV, "USE_ALT_DISK", "false") == "true"
#   exit(0)
# end

# # blur_radii = [0; HREFPrediction.blur_radii]
# # X2 = X2[:,1:2*length(blur_radii)]

# Ys = Dict{String, BitArray}();
# for event_name in keys(Ys1)
#   Ys[event_name] = vcat(Ys1[event_name], Ys2[event_name])
# end

# GC.gc()

# weights = vcat(weights1, weights2);

# # Free
# Ys1, weights1 = (nothing, nothing)
# Ys2, weights2 = (nothing, nothing)

# GC.gc()

# X = vcat(X1, X2);

# # Free
# X1 = nothing
# X2 = nothing

# GC.gc()

X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_forecasts;
      event_name_to_labeler = TrainingShared.event_name_to_labeler,
      save_dir = "validation_forecasts_with_blurs_and_forecast_hour"
    );

length(validation_forecasts) # 24370
size(X) # (666568240, 57)
length(weights) # 666568240

sum(Ys["tornado"]) # 75293.0f0
sum(weights) # 6.143644f8

# Sanity check...tornado features should best predict tornadoes, etc
# (this did find a bug :D)

# function test_predictive_power(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   for prediction_i in 1:length(HREFPrediction.models)
#     (event_name, _, _) = HREFPrediction.models[prediction_i]
#     y = Ys[event_name]
#     for j in 1:size(X,2)
#       x = @view X[:,j]
#       auc = Metrics.roc_auc(x, y, weights)
#       println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAUC: $auc")
#     end
#   end
# end
# test_predictive_power(validation_forecasts, X, Ys, weights)


# OKAY THIS HUGE MESS BELOW IS ME TRYING TO FIGURE OUT WHY THE OLD 2020 MODELS HAD BETTER AUC
# It's because 90% of AUC is the performance at false positive rates of 10%-100%, i.e. painting
# more than 10% of the US in torprob. That's unrealistic and thus AUC only really measures the
# model performance at super high POD.
#
# Better is to use the area under the precision-recall curve.
#
# The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
# Takaya Saito, Marc Rehmsmeier
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
#
# The area under the precision-recall curve as a performance metric for rare binary events
# Helen R. Sofaer, Jennifer A. Hoeting, Catherine S. Jarnevich
# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13140
#
# ...which is the same as the area to the left of the performance diagram curve (which is what we want to optimize for anyway).

# ROC AUC:
# 2020 tornado models on same dataset: 0.9845657137724223
# 2021:                                0.9840355246144981
# 2021 w/higher feat fraction:         0.9827257079052968

# y = Ys["tornado"];

# const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
# logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)
# sum(logloss.(y, (@view X[:,1])) .* weights) / sum(weights)
# # 2020 tornado models on same dataset: 0.0005586252f0
# # 2021 tornado models:                 0.00055437785f0 # OKAY so this is what you get for training with logloss. there is justice in the world
# # 2021 w/higher feat fraction:         0.00055674755f0


# σ(x) = 1.0f0 / (1.0f0 + exp(-x))
# logit(p) = log(p / (one(p) - p))

# import LogisticRegression

# function logloss_rescaled(x, y, weights)
#   X = reshape(logit.(x), (length(x),1))
#   a, b = LogisticRegression.fit(X, y, weights)
#   x_rescaled = σ.(logit.(x) .* a .+ b)
#   sum(logloss.(y, x_rescaled) .* weights) / sum(weights)
# end

#                                              # 2020:                  0.0005585756f0
# logloss_rescaled((@view X[:,1]), y, weights) # 2021:                  0.0005543472f0
# logloss_rescaled((@view X[:,8]), y, weights) # 2021 higher feat frac: 0.0005567189f0

# # HMM, should be using area under the precision-recall curve
# # b/c the precision-recall curve is a (flipped) performance diagram

# # Can't use SPC historical PODs because those are daily
# # SR  = true_pos / painted
# # POD = true_pos / total_pos
# function sr_for_target_pod(x, y, weights, target_pod)
#   x_positive       = x[y .>= 0.5f0]
#   weights_positive = weights[y .>= 0.5f0]
#   total_pos        = sum(weights_positive)
#   threshold = 0.5f0
#   Δ = 0.25f0
#   while Δ > 0.0000000001
#     true_pos = sum(weights_positive[x_positive .>= threshold])
#     pod = true_pos / total_pos
#     if pod > target_pod
#       threshold += Δ
#     else
#       threshold -= Δ
#     end
#     Δ *= 0.5f0
#   end
#   true_pos = sum(weights_positive[x_positive .>= threshold])
#   painted  = sum(weights[x .>= threshold])
#   println(threshold)
#   true_pos / painted
# end

#                                                     # 2020:                  0.0087980125f0
# sr_for_target_pod((@view X[:,1]), y, weights, 0.75) # 2021:                  0.009115579f0
# sr_for_target_pod((@view X[:,8]), y, weights, 0.75) # 2021 higher feat frac: 0.009057991f0

#                                                    # 2020:                  0.023044856f0
# sr_for_target_pod((@view X[:,1]), y, weights, 0.5) # 2021:                  0.023928536f0
# sr_for_target_pod((@view X[:,8]), y, weights, 0.5) # 2021 higher feat frac: 0.023018489f0


#                                                     # 2020:                  0.051686835f0
# sr_for_target_pod((@view X[:,1]), y, weights, 0.25) # 2021:                  0.053621564f0
# sr_for_target_pod((@view X[:,8]), y, weights, 0.25) # 2021 higher feat frac: 0.051918864f0

#                                                    # 2020:                  0.077482425f0
# sr_for_target_pod((@view X[:,1]), y, weights, 0.1) # 2021:                  0.09632159f0
# sr_for_target_pod((@view X[:,8]), y, weights, 0.1) # 2021 higher feat frac: 0.1018713f0

# function au_pr_curve_est(x, y, weights)
#   area = 0
#   for target_pod in 0.01:0.02:0.99
#     area += 0.02 * sr_for_target_pod(x, y, weights, target_pod)
#   end
#   area
# end

# au_pr_curve_est((@view X[:,1]), y, weights)     # 2021: 0.03809332829870984

#                                                 # 2020:                  0.03268739260172207
# area_under_pr_curve((@view X[:,1]), y, weights) # 2021:                  0.03809290465346103
# area_under_pr_curve((@view X[:,8]), y, weights) # 2021 higher feat frac: 0.037736602971328484



function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction.models)
    (event_name, _, _) = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    for j in 1:size(X,2)
      x = @view X[:,j]
      au_pr_curve = Float32(Metrics.area_under_pr_curve(x, y, weights))
      println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (75293.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.04717385
# tornado (75293.0)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.047551297 ***best tor***
# tornado (75293.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.04741936
# tornado (75293.0)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.04677319
# tornado (75293.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.04468108
# tornado (75293.0)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.040692568
# tornado (75293.0)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.03376709
# tornado (75293.0)      feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.009975404
# tornado (75293.0)      feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.009982798
# tornado (75293.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.009919028
# tornado (75293.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009725033
# tornado (75293.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.009318604
# tornado (75293.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008530009
# tornado (75293.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0072350176
# tornado (75293.0)      feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005358717
# tornado (75293.0)      feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0052384543
# tornado (75293.0)      feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0051098475
# tornado (75293.0)      feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004930653
# tornado (75293.0)      feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004643754
# tornado (75293.0)      feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0042573214
# tornado (75293.0)      feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003641266
# tornado (75293.0)      feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0067367465
# tornado (75293.0)      feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0067915833
# tornado (75293.0)      feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.006796195
# tornado (75293.0)      feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0067240954
# tornado (75293.0)      feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0064676907
# tornado (75293.0)      feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.005910502
# tornado (75293.0)      feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004953218
# tornado (75293.0)      feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03152867
# tornado (75293.0)      feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.03159829
# tornado (75293.0)      feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.031419236
# tornado (75293.0)      feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.03084188
# tornado (75293.0)      feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.029407797
# tornado (75293.0)      feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.026742406
# tornado (75293.0)      feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.022050139
# tornado (75293.0)      feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0095200455
# tornado (75293.0)      feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.009521937
# tornado (75293.0)      feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.009452423
# tornado (75293.0)      feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.009268862
# tornado (75293.0)      feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.008877417
# tornado (75293.0)      feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.008188491
# tornado (75293.0)      feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.007032415
# tornado (75293.0)      feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0045086
# tornado (75293.0)      feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0044937637
# tornado (75293.0)      feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0044585797
# tornado (75293.0)      feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0043809405
# tornado (75293.0)      feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0042159786
# tornado (75293.0)      feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.003941411
# tornado (75293.0)      feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003453775
# tornado (75293.0)      feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0045812014
# tornado (75293.0)      feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004578006
# tornado (75293.0)      feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004549704
# tornado (75293.0)      feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004468673
# tornado (75293.0)      feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0042879363
# tornado (75293.0)      feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.003971312
# tornado (75293.0)      feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003443835
# tornado (75293.0)      feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00011486774
# wind (588576.0)        feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.04973119
# wind (588576.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.049819887
# wind (588576.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.049596425
# wind (588576.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.04889884
# wind (588576.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.047323007
# wind (588576.0)        feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.044324093
# wind (588576.0)        feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.0388824
# wind (588576.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.1156223
# wind (588576.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.11639204 ***best wind***
# wind (588576.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.11633609
# wind (588576.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.115318336
# wind (588576.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.11255966
# wind (588576.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.106737256
# wind (588576.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.09593935
# wind (588576.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.065118514
# wind (588576.0)        feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.06521844
# wind (588576.0)        feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.06490341
# wind (588576.0)        feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.063937135
# wind (588576.0)        feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.06184721
# wind (588576.0)        feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.05828685
# wind (588576.0)        feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.05206174
# wind (588576.0)        feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.031251837
# wind (588576.0)        feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.031178381
# wind (588576.0)        feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.030897077
# wind (588576.0)        feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.030240685
# wind (588576.0)        feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.028942674
# wind (588576.0)        feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.026966082
# wind (588576.0)        feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.023751868
# wind (588576.0)        feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.042023346
# wind (588576.0)        feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.041921567
# wind (588576.0)        feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.04163715
# wind (588576.0)        feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.040985934
# wind (588576.0)        feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.039631754
# wind (588576.0)        feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.037120733
# wind (588576.0)        feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.032641623
# wind (588576.0)        feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07480313
# wind (588576.0)        feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.07518965
# wind (588576.0)        feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.07507878
# wind (588576.0)        feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.07441249
# wind (588576.0)        feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.07266358
# wind (588576.0)        feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.06915781
# wind (588576.0)        feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.062351573
# wind (588576.0)        feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05142782
# wind (588576.0)        feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.051559795
# wind (588576.0)        feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.051440638
# wind (588576.0)        feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.05100117
# wind (588576.0)        feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.04993065
# wind (588576.0)        feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.047823112
# wind (588576.0)        feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.04362749
# wind (588576.0)        feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.021261059
# wind (588576.0)        feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.02114532
# wind (588576.0)        feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.020917619
# wind (588576.0)        feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.020464793
# wind (588576.0)        feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.019625224
# wind (588576.0)        feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.018386204
# wind (588576.0)        feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01638418
# wind (588576.0)        feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00087867497
# wind_adj (182289.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.009881259
# wind_adj (182289.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.009846503
# wind_adj (182289.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.009756479
# wind_adj (182289.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.009557577
# wind_adj (182289.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0091614565
# wind_adj (182289.0)    feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.008525669
# wind_adj (182289.0)    feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.007444489
# wind_adj (182289.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.025380298
# wind_adj (182289.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.025406115
# wind_adj (182289.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.02523643
# wind_adj (182289.0)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.024833452
# wind_adj (182289.0)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.023910655
# wind_adj (182289.0)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.022371743
# wind_adj (182289.0)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.019690009
# wind_adj (182289.0)    feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.06287687
# wind_adj (182289.0)    feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0633626
# wind_adj (182289.0)    feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.063391596 ***best wind_adj***
# wind_adj (182289.0)    feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.06298552
# wind_adj (182289.0)    feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.06179719
# wind_adj (182289.0)    feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.05936777
# wind_adj (182289.0)    feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.054838225
# wind_adj (182289.0)    feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015411755
# wind_adj (182289.0)    feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.015440027
# wind_adj (182289.0)    feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.015351694
# wind_adj (182289.0)    feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01509444
# wind_adj (182289.0)    feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.014549391
# wind_adj (182289.0)    feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.013682536
# wind_adj (182289.0)    feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012191225
# wind_adj (182289.0)    feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0090909125
# wind_adj (182289.0)    feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.009034093
# wind_adj (182289.0)    feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0089417435
# wind_adj (182289.0)    feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.008758568
# wind_adj (182289.0)    feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00842084
# wind_adj (182289.0)    feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.007867892
# wind_adj (182289.0)    feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0069337296
# wind_adj (182289.0)    feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03718408
# wind_adj (182289.0)    feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.037474297
# wind_adj (182289.0)    feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.037474472
# wind_adj (182289.0)    feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.037255023
# wind_adj (182289.0)    feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.03648692
# wind_adj (182289.0)    feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.03493837
# wind_adj (182289.0)    feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03176083
# wind_adj (182289.0)    feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.05334476
# wind_adj (182289.0)    feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.053826585
# wind_adj (182289.0)    feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05400175
# wind_adj (182289.0)    feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.054005522
# wind_adj (182289.0)    feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.05363977
# wind_adj (182289.0)    feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.05242285
# wind_adj (182289.0)    feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.049427856
# wind_adj (182289.0)    feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012618135
# wind_adj (182289.0)    feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.012591669
# wind_adj (182289.0)    feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.012494314
# wind_adj (182289.0)    feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.012278571
# wind_adj (182289.0)    feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011859178
# wind_adj (182289.0)    feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01121301
# wind_adj (182289.0)    feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01009452
# wind_adj (182289.0)    feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00026842207
# hail (264451.0)        feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.01838816
# hail (264451.0)        feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.018350435
# hail (264451.0)        feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.018205777
# hail (264451.0)        feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.017866516
# hail (264451.0)        feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.017151238
# hail (264451.0)        feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.01601319
# hail (264451.0)        feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.014041341
# hail (264451.0)        feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.019074613
# hail (264451.0)        feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.01905839
# hail (264451.0)        feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0189254
# hail (264451.0)        feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.018604232
# hail (264451.0)        feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.017913926
# hail (264451.0)        feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.016845072
# hail (264451.0)        feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015095888
# hail (264451.0)        feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02105287
# hail (264451.0)        feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.021019049
# hail (264451.0)        feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.020853765
# hail (264451.0)        feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.020480622
# hail (264451.0)        feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.019691125
# hail (264451.0)        feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01843334
# hail (264451.0)        feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016272508
# hail (264451.0)        feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07440023
# hail (264451.0)        feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.075273976 ***best hail***
# hail (264451.0)        feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0751889
# hail (264451.0)        feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.07397763
# hail (264451.0)        feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.07073591
# hail (264451.0)        feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.06503451
# hail (264451.0)        feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.054932527
# hail (264451.0)        feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.013880857
# hail (264451.0)        feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.013826784
# hail (264451.0)        feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.013720977
# hail (264451.0)        feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.013491806
# hail (264451.0)        feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.013027595
# hail (264451.0)        feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012256049
# hail (264451.0)        feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0109131625
# hail (264451.0)        feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.021520682
# hail (264451.0)        feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.021514181
# hail (264451.0)        feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.021389006
# hail (264451.0)        feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.021090228
# hail (264451.0)        feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.02040426
# hail (264451.0)        feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01924813
# hail (264451.0)        feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.017174058
# hail (264451.0)        feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.018871376
# hail (264451.0)        feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.018867347
# hail (264451.0)        feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.018781073
# hail (264451.0)        feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.018576503
# hail (264451.0)        feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01807003
# hail (264451.0)        feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.017141758
# hail (264451.0)        feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.015331898
# hail (264451.0)        feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.055674136
# hail (264451.0)        feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.055770326
# hail (264451.0)        feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05533214
# hail (264451.0)        feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.054070957
# hail (264451.0)        feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.051289298
# hail (264451.0)        feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.046868403
# hail (264451.0)        feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.039552793
# hail (264451.0)        feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00039227225
# sig_tornado (10243.0)  feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.03396329
# sig_tornado (10243.0)  feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.035555333
# sig_tornado (10243.0)  feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.03675656
# sig_tornado (10243.0)  feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.03883886
# sig_tornado (10243.0)  feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.03970156 ***best sigtor***
# sig_tornado (10243.0)  feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.037842363
# sig_tornado (10243.0)  feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.031464607
# sig_tornado (10243.0)  feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0024883347
# sig_tornado (10243.0)  feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.0025042398
# sig_tornado (10243.0)  feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0024986872
# sig_tornado (10243.0)  feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0024596404
# sig_tornado (10243.0)  feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.002381613
# sig_tornado (10243.0)  feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0021872479
# sig_tornado (10243.0)  feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018546821
# sig_tornado (10243.0)  feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.00089321216
# sig_tornado (10243.0)  feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.00089207245
# sig_tornado (10243.0)  feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0008841684
# sig_tornado (10243.0)  feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.00086375367
# sig_tornado (10243.0)  feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.00082146714
# sig_tornado (10243.0)  feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0007495371
# sig_tornado (10243.0)  feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00063183985
# sig_tornado (10243.0)  feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0014356314
# sig_tornado (10243.0)  feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0014611315
# sig_tornado (10243.0)  feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0014752744
# sig_tornado (10243.0)  feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0014785667
# sig_tornado (10243.0)  feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0014493696
# sig_tornado (10243.0)  feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0013461255
# sig_tornado (10243.0)  feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0011269723
# sig_tornado (10243.0)  feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.032650158
# sig_tornado (10243.0)  feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.033054337 (not best sigtor)
# sig_tornado (10243.0)  feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.032925744
# sig_tornado (10243.0)  feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.032143027
# sig_tornado (10243.0)  feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.029842664
# sig_tornado (10243.0)  feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.02588302
# sig_tornado (10243.0)  feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.019283367
# sig_tornado (10243.0)  feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0024614965
# sig_tornado (10243.0)  feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0024659683
# sig_tornado (10243.0)  feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0024499625
# sig_tornado (10243.0)  feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0024032814
# sig_tornado (10243.0)  feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0023089428
# sig_tornado (10243.0)  feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0021245282
# sig_tornado (10243.0)  feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018066779
# sig_tornado (10243.0)  feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.00084724964
# sig_tornado (10243.0)  feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.00084361027
# sig_tornado (10243.0)  feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00083587377
# sig_tornado (10243.0)  feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.00081901817
# sig_tornado (10243.0)  feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0007844895
# sig_tornado (10243.0)  feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00072626007
# sig_tornado (10243.0)  feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0006240534
# sig_tornado (10243.0)  feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0007772464
# sig_tornado (10243.0)  feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0007762425
# sig_tornado (10243.0)  feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00077157735
# sig_tornado (10243.0)  feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0007607334
# sig_tornado (10243.0)  feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0007401514
# sig_tornado (10243.0)  feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.00070363434
# sig_tornado (10243.0)  feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.00063517917
# sig_tornado (10243.0)  feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 1.6320686e-5
# sig_wind (58860.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0065956656
# sig_wind (58860.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.006597002
# sig_wind (58860.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.006554054
# sig_wind (58860.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.0064395126
# sig_wind (58860.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0061846413
# sig_wind (58860.0)     feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0057420214
# sig_wind (58860.0)     feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.004958658
# sig_wind (58860.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.011155648
# sig_wind (58860.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.011174197
# sig_wind (58860.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.011107571
# sig_wind (58860.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.010923737
# sig_wind (58860.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.010535884
# sig_wind (58860.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.009885833
# sig_wind (58860.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.008779362
# sig_wind (58860.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.016106632
# sig_wind (58860.0)     feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016212452
# sig_wind (58860.0)     feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0162281
# sig_wind (58860.0)     feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.016137147
# sig_wind (58860.0)     feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015837165
# sig_wind (58860.0)     feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.015220447
# sig_wind (58860.0)     feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014167892
# sig_wind (58860.0)     feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005270951
# sig_wind (58860.0)     feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.00527757
# sig_wind (58860.0)     feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.005230141
# sig_wind (58860.0)     feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0051051606
# sig_wind (58860.0)     feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.004852495
# sig_wind (58860.0)     feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0044770204
# sig_wind (58860.0)     feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0038803534
# sig_wind (58860.0)     feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005722842
# sig_wind (58860.0)     feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.005701274
# sig_wind (58860.0)     feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.005654977
# sig_wind (58860.0)     feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.005553445
# sig_wind (58860.0)     feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.005355252
# sig_wind (58860.0)     feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0050024325
# sig_wind (58860.0)     feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0043875673
# sig_wind (58860.0)     feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01605378
# sig_wind (58860.0)     feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016186645 ***best sigwind***
# sig_wind (58860.0)     feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.016176753
# sig_wind (58860.0)     feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.016034747
# sig_wind (58860.0)     feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015631482
# sig_wind (58860.0)     feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.014853298
# sig_wind (58860.0)     feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013404692
# sig_wind (58860.0)     feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015479452
# sig_wind (58860.0)     feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.015638728
# sig_wind (58860.0)     feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.015716653
# sig_wind (58860.0)     feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015737994
# sig_wind (58860.0)     feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.015655823
# sig_wind (58860.0)     feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.015295205
# sig_wind (58860.0)     feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.014484877
# sig_wind (58860.0)     feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0042608203
# sig_wind (58860.0)     feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0042504678
# sig_wind (58860.0)     feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004209944
# sig_wind (58860.0)     feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004119949
# sig_wind (58860.0)     feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0039434107
# sig_wind (58860.0)     feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0036750797
# sig_wind (58860.0)     feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0032486215
# sig_wind (58860.0)     feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 8.7249486e-5
# sig_wind_adj (21053.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0011699245
# sig_wind_adj (21053.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.0011675529
# sig_wind_adj (21053.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.001159565
# sig_wind_adj (21053.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.0011421423
# sig_wind_adj (21053.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0011044495
# sig_wind_adj (21053.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0010418001
# sig_wind_adj (21053.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.0009311349
# sig_wind_adj (21053.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0033070864
# sig_wind_adj (21053.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.0033100399
# sig_wind_adj (21053.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0032934968
# sig_wind_adj (21053.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0032552532
# sig_wind_adj (21053.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0031650818
# sig_wind_adj (21053.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0030208954
# sig_wind_adj (21053.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002775856
# sig_wind_adj (21053.0) feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0115729105
# sig_wind_adj (21053.0) feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.011685652
# sig_wind_adj (21053.0) feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.011753528
# sig_wind_adj (21053.0) feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.011819169
# sig_wind_adj (21053.0) feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0118838595
# sig_wind_adj (21053.0) feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.011880476
# sig_wind_adj (21053.0) feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01205001
# sig_wind_adj (21053.0) feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0022303623
# sig_wind_adj (21053.0) feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.002240579
# sig_wind_adj (21053.0) feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.00222963
# sig_wind_adj (21053.0) feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0021932758
# sig_wind_adj (21053.0) feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0021105404
# sig_wind_adj (21053.0) feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0019752956
# sig_wind_adj (21053.0) feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0017401908
# sig_wind_adj (21053.0) feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0011895834
# sig_wind_adj (21053.0) feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0011839513
# sig_wind_adj (21053.0) feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.001174404
# sig_wind_adj (21053.0) feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.001155297
# sig_wind_adj (21053.0) feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0011207836
# sig_wind_adj (21053.0) feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0010632055
# sig_wind_adj (21053.0) feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0009625452
# sig_wind_adj (21053.0) feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0061551393
# sig_wind_adj (21053.0) feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.006225915
# sig_wind_adj (21053.0) feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0062476313
# sig_wind_adj (21053.0) feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0062523084
# sig_wind_adj (21053.0) feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.006202903
# sig_wind_adj (21053.0) feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0060779876
# sig_wind_adj (21053.0) feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0058395388
# sig_wind_adj (21053.0) feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.010768583
# sig_wind_adj (21053.0) feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.010965173
# sig_wind_adj (21053.0) feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.011108275
# sig_wind_adj (21053.0) feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.011292513
# sig_wind_adj (21053.0) feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.011566344
# sig_wind_adj (21053.0) feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0118137635
# sig_wind_adj (21053.0) feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.012251711 ***best sigwind_adj**
# sig_wind_adj (21053.0) feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0019733112
# sig_wind_adj (21053.0) feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0019713617
# sig_wind_adj (21053.0) feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.001956047
# sig_wind_adj (21053.0) feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0019213882
# sig_wind_adj (21053.0) feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0018482482
# sig_wind_adj (21053.0) feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0017324584
# sig_wind_adj (21053.0) feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0015443354
# sig_wind_adj (21053.0) feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 3.0573752e-5
# sig_hail (31850.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0030954592
# sig_hail (31850.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.0030845958
# sig_hail (31850.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.0030526386
# sig_hail (31850.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.002977147
# sig_hail (31850.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.0028282658
# sig_hail (31850.0)     feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.0025960747
# sig_hail (31850.0)     feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.0022115498
# sig_hail (31850.0)     feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0024253612
# sig_hail (31850.0)     feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.002418936
# sig_hail (31850.0)     feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.002396979
# sig_hail (31850.0)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0023490656
# sig_hail (31850.0)     feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0022502034
# sig_hail (31850.0)     feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0021010167
# sig_hail (31850.0)     feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0018674367
# sig_hail (31850.0)     feature 15 WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0035682975
# sig_hail (31850.0)     feature 16 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0035639643
# sig_hail (31850.0)     feature 17 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.003532481
# sig_hail (31850.0)     feature 18 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0034567628
# sig_hail (31850.0)     feature 19 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0033030112
# sig_hail (31850.0)     feature 20 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0030579942
# sig_hail (31850.0)     feature 21 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0026516137
# sig_hail (31850.0)     feature 22 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.014930886
# sig_hail (31850.0)     feature 23 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.015043112
# sig_hail (31850.0)     feature 24 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.014944986
# sig_hail (31850.0)     feature 25 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.014546624
# sig_hail (31850.0)     feature 26 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.013779763
# sig_hail (31850.0)     feature 27 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012619778
# sig_hail (31850.0)     feature 28 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010763833
# sig_hail (31850.0)     feature 29 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0023837932
# sig_hail (31850.0)     feature 30 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.002371184
# sig_hail (31850.0)     feature 31 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.002346316
# sig_hail (31850.0)     feature 32 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0022926475
# sig_hail (31850.0)     feature 33 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0021897159
# sig_hail (31850.0)     feature 34 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0020317107
# sig_hail (31850.0)     feature 35 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0017796424
# sig_hail (31850.0)     feature 36 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0036530579
# sig_hail (31850.0)     feature 37 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0036704845
# sig_hail (31850.0)     feature 38 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0036621026
# sig_hail (31850.0)     feature 39 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0036317494
# sig_hail (31850.0)     feature 40 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0035027915
# sig_hail (31850.0)     feature 41 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0032762373
# sig_hail (31850.0)     feature 42 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0028739816
# sig_hail (31850.0)     feature 43 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0036730436
# sig_hail (31850.0)     feature 44 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003684678
# sig_hail (31850.0)     feature 45 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0036652905
# sig_hail (31850.0)     feature 46 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0036176548
# sig_hail (31850.0)     feature 47 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.003491967
# sig_hail (31850.0)     feature 48 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0032745225
# sig_hail (31850.0)     feature 49 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.002871884
# sig_hail (31850.0)     feature 50 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.017947793
# sig_hail (31850.0)     feature 51 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.018001104 ***best sighail***
# sig_hail (31850.0)     feature 52 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.017792888
# sig_hail (31850.0)     feature 53 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.017218145
# sig_hail (31850.0)     feature 54 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01604816
# sig_hail (31850.0)     feature 55 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.01427211
# sig_hail (31850.0)     feature 56 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.011622308
# sig_hail (31850.0)     feature 57 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 4.7828773e-5





println("Determining best blur radii to maximize area under precision-recall curve")

blur_radii = [0; HREFPrediction.blur_radii]
forecast_hour_j = size(X, 2)

bests = []
for prediction_i in 1:length(HREFPrediction.models)
  (event_name, _, _) = HREFPrediction.models[prediction_i]
  y = Ys[event_name]
  prediction_i_base = (prediction_i - 1) * length(blur_radii) # 0-indexed

  println("blur_radius_f2\tblur_radius_f35\tAU_PR_$event_name")

  best_blur_i_lo, best_blur_i_hi, best_au_pr = (0, 0, 0.0)

  for blur_i_lo in 1:length(blur_radii)
    for blur_i_hi in 1:length(blur_radii)
      X_blurred = zeros(Float32, length(y))

      Threads.@threads :static for i in 1:length(y)
        forecast_ratio = (X[i,forecast_hour_j] - 2f0) * (1f0/(35f0-2f0))
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

# blur_radius_f2  blur_radius_f35 AU_PR_tornado
# 0       0       0.047173847
# 0       15      0.047420785
# 0       25      0.04742167
# 0       35      0.047233805
# 0       50      0.04665581
# 0       70      0.045591995
# 0       100     0.043800306
# 15      0       0.04743233
# 15      15      0.047551293
# 15      25      0.047532737
# 15      35      0.047330227
# 15      50      0.046721555
# 15      70      0.045634057
# 15      100     0.04383907
# 25      0       0.047350574
# 25      15      0.04746109
# 25      25      0.047419347
# 25      35      0.047213506
# 25      50      0.046551954
# 25      70      0.04536807
# 25      100     0.04350253
# 35      0       0.04694619
# 35      15      0.04708863
# 35      25      0.0470315
# 35      35      0.046773158
# 35      50      0.046105452
# 35      70      0.04482643
# 35      100     0.042631574
# 50      0       0.045823853
# 50      15      0.045953847
# 50      25      0.045858216
# 50      35      0.04550048
# 50      50      0.044681065
# 50      70      0.043307934
# 50      100     0.04093804
# 70      0       0.043849222
# 70      15      0.04396533
# 70      25      0.043830603
# 70      35      0.04338001
# 70      50      0.042348675
# 70      70      0.040692586
# 70      100     0.038126037
# 100     0       0.040675975
# 100     15      0.040781587
# 100     25      0.040604457
# 100     35      0.04005331
# 100     50      0.03882262
# 100     70      0.03679023
# 100     100     0.03376708
# Best tornado: 15        15      0.047551293

# blur_radius_f2  blur_radius_f35 AU_PR_wind
# 0       0       0.1156223
# 0       15      0.116068944
# 0       25      0.11618116
# 0       35      0.11606292
# 0       50      0.11564519
# 0       70      0.1146725
# 0       100     0.11295548
# 15      0       0.116118714
# 15      15      0.11639204
# 15      25      0.1164659
# 15      35      0.116311215
# 15      50      0.11586369
# 15      70      0.114871874
# 15      100     0.1131628
# 25      0       0.11609077
# 25      15      0.11632273
# 25      25      0.1163361
# 25      35      0.11611682
# 25      50      0.11559692
# 25      70      0.114540264
# 25      100     0.112786435
# 35      0       0.1155186
# 35      15      0.115713626
# 35      25      0.11565925
# 35      35      0.115318336
# 35      50      0.11464014
# 35      70      0.11342263
# 35      100     0.11153145
# 50      0       0.11399147
# 50      15      0.11415057
# 50      25      0.11401697
# 50      35      0.113513276
# 50      50      0.11255966
# 50      70      0.11100694
# 50      100     0.10875191
# 70      0       0.11095199
# 70      15      0.11107177
# 70      25      0.11085183
# 70      35      0.110160224
# 70      50      0.108840816
# 70      70      0.10673726
# 70      100     0.10377449
# 100     0       0.10598816
# 100     15      0.10609674
# 100     25      0.105785854
# 100     35      0.10485393
# 100     50      0.10305329
# 100     70      0.10014517
# 100     100     0.09593935
# Best wind: 15   25      0.1164659

# blur_radius_f2  blur_radius_f35 AU_PR_wind_adj
# 0       0       0.06287687
# 0       15      0.06315651
# 0       25      0.06323718
# 0       35      0.063215055
# 0       50      0.06310393
# 0       70      0.06282467
# 0       100     0.062314242
# 15      0       0.06320236
# 15      15      0.0633626
# 15      25      0.06341821
# 15      35      0.063375264
# 15      50      0.06324745
# 15      70      0.06296386
# 15      100     0.06247299
# 25      0       0.06323945
# 25      15      0.06337348
# 25      25      0.063391596
# 25      35      0.0633127
# 25      50      0.06314707
# 25      70      0.06283357
# 25      100     0.06233076
# 35      0       0.06303393
# 35      15      0.06314666
# 35      25      0.06312745
# 35      35      0.06298552
# 35      50      0.062743
# 35      70      0.06235761
# 35      100     0.06180366
# 50      0       0.062392578
# 50      15      0.062483374
# 50      25      0.062417798
# 50      35      0.062188756
# 50      50      0.06179719
# 50      70      0.061249312
# 50      100     0.060536083
# 70      0       0.061130203
# 70      15      0.06120682
# 70      25      0.061093606
# 70      35      0.06076419
# 70      50      0.06018381
# 70      70      0.05936777
# 70      100     0.058330536
# 100     0       0.05906441
# 100     15      0.059133876
# 100     25      0.058985095
# 100     35      0.058546983
# 100     50      0.05772534
# 100     70      0.056505054
# 100     100     0.054838225
# Best wind_adj: 15       25      0.06341821

# blur_radius_f2  blur_radius_f35 AU_PR_hail
# 0       0       0.07440023
# 0       15      0.07491264
# 0       25      0.07501864
# 0       35      0.07484354
# 0       50      0.07428693
# 0       70      0.07331995
# 0       100     0.071745135
# 15      0       0.074904345
# 15      15      0.07527398
# 15      25      0.07535621
# 15      35      0.0751627
# 15      50      0.07460407
# 15      70      0.073651165
# 15      100     0.07211594
# 25      0       0.07480726
# 25      15      0.075154
# 25      25      0.0751889
# 25      35      0.0749448
# 25      50      0.07433616
# 25      70      0.07334851
# 25      100     0.07180803
# 35      0       0.0740166
# 35      15      0.07434657
# 35      25      0.07432535
# 35      35      0.07397762
# 35      50      0.073245116
# 35      70      0.07214516
# 35      100     0.070531026
# 50      0       0.072009705
# 50      15      0.07232097
# 50      25      0.07222305
# 50      35      0.071717866
# 50      50      0.07073591
# 50      70      0.06937606
# 50      100     0.06749656
# 70      0       0.06876684
# 70      15      0.06903966
# 70      25      0.068852514
# 70      35      0.06814967
# 70      50      0.06681225
# 70      70      0.065034516
# 70      100     0.06262114
# 100     0       0.06404085
# 100     15      0.06423367
# 100     25      0.06392256
# 100     35      0.062959224
# 100     50      0.06106884
# 100     70      0.05848146
# 100     100     0.054932527
# Best hail: 15   25      0.07535621

# blur_radius_f2  blur_radius_f35 AU_PR_sig_tornado
# 0       0       0.032650154
# 0       15      0.032817807
# 0       25      0.03280885
# 0       35      0.032647766
# 0       50      0.03214054
# 0       70      0.031356324
# 0       100     0.029981095
# 15      0       0.03296353
# 15      15      0.033054337
# 15      25      0.033022128
# 15      35      0.032844614
# 15      50      0.03230738
# 15      70      0.031510882
# 15      100     0.030146483
# 25      0       0.032919906
# 25      15      0.032989293
# 25      25      0.032925747
# 25      35      0.03271272
# 25      50      0.03212623
# 25      70      0.031299274
# 25      100     0.029921139
# 35      0       0.032480564
# 35      15      0.032528453
# 35      25      0.032428488
# 35      35      0.032143027
# 35      50      0.031461857
# 35      70      0.030542873
# 35      100     0.029107753
# 50      0       0.03118597
# 50      15      0.031215066
# 50      25      0.031069243
# 50      35      0.030689519
# 50      50      0.029842662
# 50      70      0.028706128
# 50      100     0.027105384
# 70      0       0.029072119
# 70      15      0.029073048
# 70      25      0.028874835
# 70      35      0.028382191
# 70      50      0.02733499
# 70      70      0.025883023
# 70      100     0.023864044
# 100     0       0.02588612
# 100     15      0.025826106
# 100     25      0.025555424
# 100     35      0.02493371
# 100     50      0.023663169
# 100     70      0.021818511
# 100     100     0.019283365
# Best sig_tornado: 15    15      0.033054337

# blur_radius_f2  blur_radius_f35 AU_PR_sig_wind
# 0       0       0.01605378
# 0       15      0.016139263
# 0       25      0.01617151
# 0       35      0.016183123
# 0       50      0.016169276
# 0       70      0.016126059
# 0       100     0.0160506
# 15      0       0.016132528
# 15      15      0.016186645
# 15      25      0.016212652
# 15      35      0.016219445
# 15      50      0.016201459
# 15      70      0.016155893
# 15      100     0.016081689
# 25      0       0.016112316
# 25      15      0.016159981
# 25      25      0.016176753
# 25      35      0.016174695
# 25      50      0.01614681
# 25      70      0.016091686
# 25      100     0.016009359
# 35      0       0.016003335
# 35      15      0.016045377
# 35      25      0.016052945
# 35      35      0.016034747
# 35      50      0.015986143
# 35      70      0.015908776
# 35      100     0.015804153
# 50      0       0.01572707
# 50      15      0.015762968
# 50      25      0.015759122
# 50      35      0.015719134
# 50      50      0.015631482
# 50      70      0.015506058
# 50      100     0.015344408
# 70      0       0.015245822
# 70      15      0.015275865
# 70      25      0.015259723
# 70      35      0.015193939
# 70      50      0.015055689
# 70      70      0.014853298
# 70      100     0.014586027
# 100     0       0.014480586
# 100     15      0.014505475
# 100     25      0.014475012
# 100     35      0.014378043
# 100     50      0.014174218
# 100     70      0.013861837
# 100     100     0.013404692
# Best sig_wind: 15       35      0.016219445

# blur_radius_f2  blur_radius_f35 AU_PR_sig_wind_adj
# 0       0       0.010768583
# 0       15      0.010856167
# 0       25      0.010917695
# 0       35      0.010998158
# 0       50      0.0111227315
# 0       70      0.011255515
# 0       100     0.011455701
# 15      0       0.010896597
# 15      15      0.010965173
# 15      25      0.011024886
# 15      35      0.011106024
# 15      50      0.011234046
# 15      70      0.0113729285
# 15      100     0.011588068
# 25      0       0.010986695
# 25      15      0.011052717
# 25      25      0.011108275
# 25      35      0.011188326
# 25      50      0.011318124
# 25      70      0.011461026
# 25      100     0.011687476
# 35      0       0.011094514
# 35      15      0.011160287
# 35      25      0.011214856
# 35      35      0.011292513
# 35      50      0.011422338
# 35      70      0.01156882
# 35      100     0.011810416
# 50      0       0.011237536
# 50      15      0.011308015
# 50      25      0.011362307
# 50      35      0.011437356
# 50      50      0.011566344
# 50      70      0.011713376
# 50      100     0.011973715
# 70      0       0.0113278255
# 70      15      0.011406902
# 70      25      0.0114652235
# 70      35      0.011542085
# 70      50      0.011670388
# 70      70      0.0118137635
# 70      100     0.012080499
# 100     0       0.011395489
# 100     15      0.01149091
# 100     25      0.011562788
# 100     35      0.011657244
# 100     50      0.011807112
# 100     70      0.011963725
# 100     100     0.012251711
# Best sig_wind_adj: 100  100     0.012251711

# blur_radius_f2  blur_radius_f35 AU_PR_sig_hail
# 0       0       0.017947799
# 0       15      0.018009394
# 0       25      0.017984034
# 0       35      0.017889323
# 0       50      0.017734107
# 0       70      0.017532252
# 0       100     0.017345302
# 15      0       0.018000767
# 15      15      0.018001104
# 15      25      0.017961454
# 15      35      0.017852608
# 15      50      0.017684998
# 15      70      0.017476808
# 15      100     0.01729842
# 25      0       0.017878069
# 25      15      0.017848784
# 25      25      0.01779289
# 25      35      0.017660955
# 25      50      0.017465657
# 25      70      0.01723989
# 25      100     0.017056096
# 35      0       0.017546017
# 35      15      0.0175082
# 35      25      0.017399695
# 35      35      0.017218143
# 35      50      0.016966118
# 35      70      0.01669317
# 35      100     0.01647658
# 50      0       0.016839484
# 50      15      0.01682632
# 50      25      0.016653746
# 50      35      0.016400699
# 50      50      0.01604816
# 50      70      0.015660444
# 50      100     0.015344957
# 70      0       0.015742091
# 70      15      0.015701715
# 70      25      0.015578226
# 70      35      0.015296965
# 70      50      0.014836504
# 70      70      0.01427211
# 70      100     0.01372474
# 100     0       0.014501102
# 100     15      0.014441917
# 100     25      0.014287722
# 100     35      0.013941022
# 100     50      0.013350397
# 100     70      0.012568293
# 100     100     0.011622308
# Best sig_hail: 0        15      0.018009394

println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f35\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name   best_blur_radius_f2 best_blur_radius_f35 AU_PR
# tornado      15                  15                   0.047551293
# wind         15                  25                   0.1164659
# wind_adj     15                  25                   0.06341821
# hail         15                  25                   0.07535621
# sig_tornado  15                  15                   0.033054337
# sig_wind     15                  35                   0.016219445
# sig_wind_adj 100                 100                  0.012251711
# sig_hail     0                   15                   0.018009394


# Now go back to HREFPrediction.jl and put those numbers in


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
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.regular_forecasts(HREFPrediction.forecasts_blurred()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100])

# rm("validation_forecasts_blurred"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_blurred");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction.models)
    (event_name, _, _) = HREFPrediction.models[prediction_i]
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


Metrics.reliability_curves_midpoints(20, X, Ys, map(m -> m[1], HREFPrediction.models_with_gated), weights, map(m -> m[3], HREFPrediction.models_with_gated))

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
for prediction_i in 1:length(HREFPrediction.models)
  (event_name, _, _, _, _) = HREFPrediction.models[prediction_i]

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
for prediction_i in 1:length(HREFPrediction.models)
  (event_name, _, _, _, _) = HREFPrediction.models[prediction_i]

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
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_calibrated_with_sig_gated, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.regular_forecasts(HREFPrediction.forecasts_calibrated_with_sig_gated()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts_calibrated_with_sig_gated = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_calibrated_with_sig_gated);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_calibrated_with_sig_gated[100]);

# rm("validation_forecasts_calibrated_with_sig_gated"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_calibrated_with_sig_gated; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_calibrated_with_sig_gated");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction.models_with_gated)
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
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

Metrics.reliability_curves_midpoints(20, X, Ys, map(m -> m[1], HREFPrediction.models_with_gated), weights, map(m -> m[3], HREFPrediction.models_with_gated))
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

plot_calibration_curves(map(m -> m[1], HREFPrediction.models), event_to_bins, event_to_bins_logistic_coeffs)
