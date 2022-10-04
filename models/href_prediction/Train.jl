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

      Threads.@threads for i in 1:length(y)
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
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
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
# tornado (68134.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve:
# wind (562866.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve:
# hail (246689.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve:
# sig_tornado (9182.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve:
# sig_wind (57701.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve:
# sig_hail (30597.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve:

# Yay!




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

# event_name  mean_y                 mean_ŷ                 Σweight              bin_max
# tornado     1.529606123529999e-5   1.4148039907265803e-5  6.973074100372117e8  0.0009693373
# tornado     0.0018216224618967571  0.0019498115061567457  5.8549051383398175e6 0.003943406
# tornado     0.00568073556699246    0.006123562364300988   1.8775200630750656e6 0.009779687
# tornado     0.01318227137856676    0.014068343842865035   809092.026733458     0.021067958
# tornado     0.02940509128244388    0.02939055385222918    362711.1334466934    0.04314823
# tornado     0.07456508720599171    0.06506041208349338    143007.95882487297   1.0
# wind        0.00012590116795742026 0.00015252209584833615 6.924477708026028e8  0.007293307
# wind        0.011309064551886225   0.011832892523891256   7.7088925327656865e6 0.019115837
# wind        0.027523318115127964   0.026062746069332173   3.1674831753054857e6 0.036141146
# wind        0.05239926109230082    0.04730908940138771    1.663765046367824e6  0.06361171
# wind        0.09553100258326357    0.08368232783656926    912586.2284119129    0.11529552
# wind        0.19195656196864078    0.17572405890161805    454148.5604224205    1.0
# hail        5.455856829572775e-5   5.38953508157325e-5    6.947669272917984e8  0.0032933427
# hail        0.005850972375548683   0.005511708615476839   6.478618584934175e6  0.00916807
# hail        0.013514867701920667   0.013147210460493839   2.804765588622153e6  0.019258574
# hail        0.028805758077871106   0.025951271127335626   1.3159078453031182e6 0.03614172
# hail        0.054098382334513286   0.049773531285174734   700688.3589830399    0.07324672
# hail        0.13172348234093761    0.1143075361693153     287738.6857646704    1.0
# sig_tornado 2.0793742105664814e-6  2.2139589759300784e-6  7.043884747808976e8  0.00064585934
# sig_tornado 0.0010718664930749002  0.0012829256036335735  1.3670102717869878e6 0.0025964007
# sig_tornado 0.0039006527533272956  0.003821656631232155   375477.0924088359    0.005882601
# sig_tornado 0.010550654775277396   0.00801411081504029    138817.35471099615   0.011497159
# sig_tornado 0.023014612104714758   0.01584936109599417    63638.565164387226   0.023939667
# sig_tornado 0.06891064626011477    0.0379987235956591     21228.28565776348    1.0
# sig_wind    1.2794419283396139e-5  1.3774982586939223e-5  6.946559299164902e8  0.0007026063
# sig_wind    0.0012679894529567756  0.001236753060875541   7.008705636615396e6  0.002175051
# sig_wind    0.0033220323538792527  0.0031620202598151613  2.675361352635026e6  0.0047103437
# sig_wind    0.008096499223348373   0.006150974523453701   1.097703709167838e6  0.008226235
# sig_wind    0.01444389469749182    0.010793573909811303   615320.4450896978    0.014878796
# sig_wind    0.02945443889145807    0.02412952268810448    301625.2910477519    1.0
# sig_hail    6.76781241931757e-6    7.11150700002389e-6    7.003515833429558e8  0.00071826525
# sig_hail    0.001425119899308227   0.0012175292279273196  3.3260057993766665e6 0.0020554834
# sig_hail    0.0034634791264424344  0.0029141941909888456  1.3684366065796614e6 0.004203511
# sig_hail    0.0062328488670805315  0.005841556421350675   760383.5147241354    0.008379867
# sig_hail    0.01221663751689759    0.011808751415596546   387974.286046505     0.01785916
# sig_hail    0.029554956639004617   0.029523741209696606   160262.80071800947   1.0

println(event_to_bins)
# Dict{String, Vector{Float32}}("sig_hail" => [0.00071826525, 0.0020554834, 0.004203511, 0.008379867, 0.01785916, 1.0], "hail" => [0.0032933427, 0.00916807, 0.019258574, 0.03614172, 0.07324672, 1.0], "tornado" => [0.0009693373, 0.003943406, 0.009779687, 0.021067958, 0.04314823, 1.0], "sig_tornado" => [0.00064585934, 0.0025964007, 0.005882601, 0.011497159, 0.023939667, 1.0], "sig_wind" => [0.0007026063, 0.002175051, 0.0047103437, 0.008226235, 0.014878796, 1.0], "wind" => [0.007293307, 0.019115837, 0.036141146, 0.06361171, 0.11529552, 1.0])


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

    Threads.@threads for i in 1:length(bin_y)
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

# event_name  bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ   mean_y        HREF_logloss  HREF_au_pr            mean_logistic_ŷ logistic_logloss logistic_au_pr        logistic_coeffs
# tornado     1-2 -1.0          0.003943406  768580069 23015.0   7.031623e8  3.0265412e-5  3.0336501e-5  0.00026088595 0.0018272604432091135 3.03365e-5      0.0002607281     0.0018132768233895585 Float32[0.9410416,  -0.43989772]
# tornado     2-3 0.0009693373  0.009779687  8279143   22771.0   7.732425e6  0.0029632454  0.0027586587  0.018463962   0.005573121102516215  0.0027586583    0.018456165      0.005573121200265879  Float32[0.9696831,  -0.2419776]
# tornado     3-4 0.003943406   0.021067958  2851135   22641.0   2.686612e6  0.008516189   0.007939876   0.045417007   0.01359600740689414   0.007939874     0.045396835      0.013596007697703711  Float32[0.9994333,  -0.073430814]
# tornado     4-5 0.009779687   0.04314823   1232921   22614.0   1.1718032e6 0.018811064   0.01820376    0.08901149    0.030113523453788656  0.018203758     0.08898648       0.030113511859822965  Float32[1.094306,   0.3302174]
# tornado     5-6 0.021067958   1.0          528376    22478.0   505719.12   0.039477322   0.042175498   0.1684217     0.08418477705356607   0.042175494     0.168238         0.08418477701104629   Float32[1.1247456,  0.4527394]
# wind        1-2 -1.0          0.019115837  765289028 187584.0  7.001567e8  0.00028112577 0.00024903048 0.0016073652  0.011146547407329555  0.00024903056   0.0016036514     0.011146540215301246  Float32[1.0835536,  0.3302101]
# wind        2-3 0.007293307   0.036141146  11692832  187875.0  1.0876376e7 0.015976995   0.016031075   0.08006723    0.02740987237452326   0.016031077     0.080040686      0.02740987083511612   Float32[1.1271127,  0.5125384]
# wind        3-4 0.019115837   0.06361171   5198267   188131.0  4.831248e6  0.033379473   0.036089994   0.15287088    0.05317369660639551   0.036089994     0.15274402       0.05317357267282812   Float32[1.0838023,  0.35807598]
# wind        4-5 0.036141146   0.11529552   2774714   187896.0  2.5763512e6 0.0601931     0.06767724    0.24379614    0.09598473286184864   0.06767724      0.2433045        0.0959834031628096    Float32[1.0530577,  0.26911345]
# wind        5-6 0.06361171    1.0          1472285   187151.0  1.3667349e6 0.11426662    0.12757197    0.36758074    0.22774408096398865   0.12757199      0.3667041        0.22774408003422644   Float32[0.9993558,  0.12850402]
# hail        1-2 -1.0          0.00916807   766414551 82204.0   7.012456e8  0.00010431862 0.00010811006 0.00077062927 0.00562086417355665   0.00010811002   0.0007703596     0.005620862883802084  Float32[1.0387952,  0.27688286]
# hail        2-3 0.0032933427  0.019258574  10049511  81840.0   9.283384e6  0.007818604   0.008166446   0.04641192    0.013883571722057253  0.0081664445    0.04640389       0.013883574039793154  Float32[0.9819013,  -0.041472256]
# hail        3-4 0.00916807    0.03614172   4471349   81964.0   4.1206735e6 0.017236097   0.018397905   0.090045005   0.02910102256089555   0.018397907     0.08998896       0.02910101982705813   Float32[1.1090772,  0.4985385]
# hail        4-5 0.019258574   0.07324672   2192057   82469.0   2.0165962e6 0.03422858    0.037593953   0.15771052    0.055836359977871126  0.037593953     0.15754303       0.0558363645977709    Float32[0.98798597, 0.05851983]
# hail        5-6 0.03614172    1.0          1073680   82521.0   988427.0    0.06855988    0.07669564    0.25909775    0.1559476060797505    0.07669566      0.25851932       0.15594760280019088   Float32[1.0764973,  0.31471553]
# sig_tornado 1-2 -1.0          0.0025964007 771336077 3130.0    7.057555e8  4.6946284e-6  4.1514936e-6  3.9215014e-5  0.0013324690374889518 4.1514927e-6    3.918973e-5      0.0013305201167142498 Float32[0.92174333, -0.7631738]
# sig_tornado 2-3 0.00064585934 0.005882601  1834683   3051.0    1.7424874e6 0.0018299798  0.001681423   0.011962341   0.004599673664678605  0.0016814225    0.011940232      0.004599679410779465  Float32[1.226332,   1.2883613]
# sig_tornado 3-4 0.0025964007  0.011497159  535895    3030.0    514294.47   0.004953276   0.005695608   0.03428813    0.010599822095757315  0.005695608     0.034164835      0.010599770193332915  Float32[1.3703138,  2.0596972]
# sig_tornado 4-5 0.005882601   0.023939667  209844    3025.0    202455.92   0.010476989   0.014468487   0.07475995    0.025499862871274884  0.014468485     0.074024156      0.02549986168962322   Float32[1.2036767,  1.2354493]
# sig_tornado 5-6 0.011497159   1.0          87608     3022.0    84866.85    0.021389721   0.03449488    0.14647296    0.0769724855892172    0.034494873     0.1427091        0.076972482541521     Float32[1.2170192,  1.2941704]
# sig_wind    1-2 -1.0          0.002175051  766897086 19193.0   7.0166464e8 2.5990927e-5  2.5332163e-5  0.00022042012 0.0012700716249063712 2.5332158e-5    0.0002200686     0.001270103871160812  Float32[1.0980748,  0.74092144]
# sig_wind    2-3 0.0007026063  0.0047103437 10436303  19208.0   9.684068e6  0.0017686352  0.0018354479  0.0131185865  0.0033881630588428664 0.0018354482    0.013117139      0.0033811746452988966 Float32[1.0278105,  0.2092408]
# sig_wind    3-4 0.002175051   0.008226235  4070047   19274.0   3.773065e6  0.0040316014  0.0047110757  0.02942888    0.008283840539270558  0.004711076     0.029336378      0.008266362999770032  Float32[1.3354824,  1.9728225]
# sig_wind    4-5 0.0047103437  0.014878796  1849307   19261.0   1.7130241e6 0.007818602   0.010376492   0.05750749    0.014353149856674716  0.010376492     0.05712362       0.014353180859650769  Float32[1.0461866,  0.5070991]
# sig_wind    5-6 0.008226235   1.0          992447    19234.0   916945.75   0.0151803745  0.019381547   0.09460407    0.03153562723926698   0.019381545     0.09396499       0.031535627430103234  Float32[0.8209114,  -0.47899055]
# sig_hail    1-2 -1.0          0.0020554834 769058501 10189.0   7.036776e8  1.2832674e-5  1.3471802e-5  0.00011470124 0.0014578486110732047 1.3471806e-5    0.00011414103    0.001457837872468933  Float32[1.1735471,  1.4033803]
# sig_hail    2-3 0.00071826525 0.004203511  5085831   10191.0   4.6944425e6 0.0017121094  0.0020193043  0.01431179    0.0034794615114734934 0.0020193039    0.014285652      0.0034552337734657847 Float32[1.0141681,  0.25384367]
# sig_hail    3-4 0.0020554834  0.008379867  2309480   10234.0   2.12882e6   0.0039598052  0.004452658   0.028338917   0.006272377809412836  0.0044526574    0.028299749      0.006240622524798311  Float32[0.83706397, -0.7718604]
# sig_hail    4-5 0.004203511   0.01785916   1245358   10223.0   1.1483578e6 0.0078575825  0.008254481   0.04726873    0.012339396262257456  0.0082544815    0.04725294       0.012339396759446288  Float32[0.9086216,  -0.3855668]
# sig_hail    5-6 0.008379867   1.0          591599    10174.0   548237.1    0.016987264   0.01728504    0.08513529    0.03157749232532621   0.017285042     0.085097656      0.03157749170120359   Float32[0.8945266,  -0.3957873]

print("event_to_bins_logistic_coeffs = $event_to_bins_logistic_coeffs")
# event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.1735471, 1.4033803], [1.0141681, 0.25384367], [0.83706397, -0.7718604], [0.9086216, -0.3855668], [0.8945266, -0.3957873]], "hail" => [[1.0387952, 0.27688286], [0.9819013, -0.041472256], [1.1090772, 0.4985385], [0.98798597, 0.05851983], [1.0764973, 0.31471553]], "tornado" => [[0.9410416, -0.43989772], [0.9696831, -0.2419776], [0.9994333, -0.073430814], [1.094306, 0.3302174], [1.1247456, 0.4527394]], "sig_tornado" => [[0.92174333, -0.7631738], [1.226332, 1.2883613], [1.3703138, 2.0596972], [1.2036767, 1.2354493], [1.2170192, 1.2941704]], "sig_wind" => [[1.0980748, 0.74092144], [1.0278105, 0.2092408], [1.3354824, 1.9728225], [1.0461866, 0.5070991], [0.8209114, -0.47899055]], "wind" => [[1.0835536, 0.3302101], [1.1271127, 0.5125384], [1.0838023, 0.35807598], [1.0530577, 0.26911345], [0.9993558, 0.12850402]])


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

  for prediction_i in 1:length(HREFPrediction.models)
    (event_name, _, _) = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
  end
end
test_predictive_power(validation_forecasts_calibrated_with_sig_gated, X, Ys, weights)

# Same as before, because calibration was monotonic
# tornado (68134.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.038589306
# wind (562866.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.11570608
# hail (246689.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07425418
# sig_tornado (9182.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.033819992
# sig_wind (57701.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.016239488
# sig_hail (30597.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.015588201



function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  println("model_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_count = 20
    per_bin_pos_weight = Float64(sum(y .* weights)) / bin_count

    # bins = map(_ -> Int64[], 1:bin_count)
    bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
    bins_Σy      = map(_ -> 0.0, 1:bin_count)
    bins_Σweight = map(_ -> 0.0, 1:bin_count)
    bins_max     = map(_ -> 1.0f0, 1:bin_count)

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

    for bin_i in 1:bin_count
      Σŷ      = bins_Σŷ[bin_i]
      Σy      = bins_Σy[bin_i]
      Σweight = bins_Σweight[bin_i]

      mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      println("$model_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(validation_forecasts_calibrated_with_sig_gated, X, Ys, weights)

# event_name                   mean_y        mean_ŷ        Σweight     bin_max
# tornado                      4.7577023e-6  5.0776057e-6  6.726667e8  8.822641e-5
# tornado                      0.00020302102 0.00016350107 1.5761458e7 0.0003000085
# tornado                      0.00042865236 0.0004785654  7.4647785e6 0.00075306173
# tornado                      0.0009798631  0.0009926242  3.265949e6  0.0013058825
# tornado                      0.0015785117  0.0016265801  2.0273739e6 0.0020298408
# tornado                      0.0024154384  0.0024362206  1.3249922e6 0.0029350107
# tornado                      0.0035027056  0.003438191   913703.3    0.0040292507
# tornado                      0.0048957192  0.004606743   653651.0    0.005281379
# tornado                      0.0059676054  0.006045543   536161.94   0.006945363
# tornado                      0.0075219213  0.0079397755  425421.72   0.009119052
# tornado                      0.009690316   0.010305535   330234.44   0.011710509
# tornado                      0.01346105    0.013127627   237694.03   0.014796797
# tornado                      0.017307408   0.016604794   184907.88   0.018740496
# tornado                      0.02118981    0.021137537   151027.0    0.023900302
# tornado                      0.026993312   0.026827194   118535.86   0.030261038
# tornado                      0.032782637   0.034166776   97617.08    0.038810108
# tornado                      0.043053117   0.043917384   74317.92    0.049964588
# tornado                      0.058244523   0.056626145   54942.633   0.06514058
# tornado                      0.07824215    0.076699294   40898.242   0.09354243
# tornado                      0.1312069     0.12843598    24320.938   1.0
# wind                         3.9025166e-5  3.5552963e-5  6.702e8     0.001023272
# wind                         0.0018041574  0.0018325968  1.4496691e7 0.00312451
# wind                         0.004134872   0.0042761746  6.325239e6  0.0057691615
# wind                         0.006964759   0.0071469685  3.7552998e6 0.008803299
# wind                         0.010242495   0.010400374   2.553491e6  0.012276133
# wind                         0.014020917   0.014157137   1.8653774e6 0.016341235
# wind                         0.018865112   0.01849844    1.3863931e6 0.020964393
# wind                         0.023777591   0.023456464   1.0999704e6 0.026248872
# wind                         0.029288191   0.029076602   892990.7    0.032227375
# wind                         0.03493034    0.035501644   748756.44   0.039143797
# wind                         0.042730443   0.04298102    612086.2    0.047272224
# wind                         0.05136805    0.05173728    509152.84   0.056737307
# wind                         0.06225012    0.061877556   420148.72   0.06760992
# wind                         0.07506528    0.07360504    348415.72   0.08035976
# wind                         0.08970167    0.087473385   291572.2    0.09547654
# wind                         0.10498218    0.10420463    249132.23   0.11412729
# wind                         0.12361988    0.1251774     211568.9    0.1384551
# wind                         0.14966369    0.15507202    174753.56   0.17585532
# wind                         0.19542663    0.20388791    133831.77   0.2444367
# wind                         0.32773158    0.3162817     79774.57    1.0
# hail                         1.6801008e-5  1.5820162e-5  6.7684454e8 0.00054557895
# hail                         0.0009952725  0.0009636982  1.1425841e7 0.0016103927
# hail                         0.0021486226  0.0022045332  5.292855e6  0.0029754357
# hail                         0.003628722   0.0036911208  3.1338678e6 0.0045527727
# hail                         0.0053407014  0.0053622695  2.1292708e6 0.006295701
# hail                         0.007297265   0.0071850354  1.558415e6  0.008185917
# hail                         0.009084847   0.0091959955  1.2517269e6 0.010347905
# hail                         0.011276938   0.0115990015  1.0084563e6 0.013054606
# hail                         0.0145315835  0.014584941   782605.4    0.016366318
# hail                         0.018257974   0.018321542   622864.56   0.020623907
# hail                         0.022885127   0.022993619   496903.44   0.025654215
# hail                         0.028959394   0.028281879   392700.84   0.031198185
# hail                         0.03420509    0.034223042   332455.53   0.03755446
# hail                         0.04172632    0.040982887   272527.97   0.04498506
# hail                         0.04953626    0.049575552   229573.7    0.0549675
# hail                         0.059924424   0.061312187   189779.1    0.069058925
# hail                         0.07762781    0.07809931    146497.45   0.08907306
# hail                         0.10160117    0.10173436    111923.93   0.117873155
# hail                         0.13919899    0.13863233    81698.22    0.16830547
# hail                         0.22663823    0.22620103    50136.42    1.0
# sig_tornado                  6.303477e-7   7.2038415e-7  6.9721e8    3.3574852e-5
# sig_tornado                  8.845426e-5   7.785889e-5   4.9699965e6 0.00016840432
# sig_tornado                  0.0002450834  0.00026580694 1.7936699e6 0.00041199927
# sig_tornado                  0.0005134927  0.0005492066  857060.8    0.0007302659
# sig_tornado                  0.0008319362  0.00093923253 528898.6    0.001231456
# sig_tornado                  0.0016940493  0.0014945535  259441.08   0.0018364261
# sig_tornado                  0.0021714815  0.0022762588  202382.6    0.0027798968
# sig_tornado                  0.003079509   0.0032655848  142670.39   0.003864245
# sig_tornado                  0.004283152   0.0045055184  102679.26   0.005289895
# sig_tornado                  0.0068380344  0.006024709   64276.76    0.006899892
# sig_tornado                  0.008351387   0.007884817   52685.367   0.009023594
# sig_tornado                  0.009828747   0.0103205135  44777.113   0.011838434
# sig_tornado                  0.014543949   0.013168855   30207.785   0.014640002
# sig_tornado                  0.014615556   0.016604831   30074.26    0.01905926
# sig_tornado                  0.020052748   0.021651385   21923.078   0.024841756
# sig_tornado                  0.029392289   0.028060745   14950.955   0.032008216
# sig_tornado                  0.037813835   0.036915112   11618.702   0.043161657
# sig_tornado                  0.050977778   0.05074944    8636.439    0.060440518
# sig_tornado                  0.08372934    0.07116039    5257.4688   0.087302685
# sig_tornado                  0.12550399    0.13317385    3444.568    1.0
# sig_wind                     3.961684e-6   3.698574e-6   6.731528e8  8.695716e-5
# sig_wind                     0.00020420847 0.00015564042 1.3060035e7 0.00026710462
# sig_wind                     0.00038452615 0.00040125483 6.934748e6  0.00059058797
# sig_wind                     0.0007386655  0.0007636923  3.6094055e6 0.0009785821
# sig_wind                     0.0011495483  0.0011788368  2.3194235e6 0.0014143221
# sig_wind                     0.0015365441  0.0016512765  1.7351946e6 0.0019224314
# sig_wind                     0.0021286062  0.002173407   1.2528846e6 0.002466119
# sig_wind                     0.0027563218  0.0027933675  967334.06   0.0031891225
# sig_wind                     0.0037000172  0.003623283   720684.75   0.004152767
# sig_wind                     0.00453524    0.004806697   587936.2    0.005618073
# sig_wind                     0.0060018664  0.006419017   444325.28   0.0073132077
# sig_wind                     0.008196961   0.008096937   325332.03   0.008940162
# sig_wind                     0.010665658   0.00966769    249982.47   0.010425533
# sig_wind                     0.011979158   0.011265674   222626.69   0.012294381
# sig_wind                     0.0140962135  0.013374587   189175.36   0.01455917
# sig_wind                     0.015122527   0.015887551   176309.06   0.017349068
# sig_wind                     0.017545048   0.018884214   151972.52   0.02070126
# sig_wind                     0.023196883   0.02284445    114938.61   0.025555518
# sig_wind                     0.032804325   0.028975844   81290.97    0.03396688
# sig_wind                     0.045624822   0.045747764   58251.93    1.0
# sig_hail                     2.0565428e-6  1.8520311e-6  6.916354e8  0.00012762647
# sig_hail                     0.0002722307  0.00021408104 5.223518e6  0.0003455947
# sig_hail                     0.000487786   0.0004972943  2.9153055e6 0.000705616
# sig_hail                     0.0009044581  0.00089577783 1.572771e6  0.0011281127
# sig_hail                     0.0012851037  0.0013562975  1.1067718e6 0.0016188795
# sig_hail                     0.001778059   0.0018571168  799735.25   0.002116471
# sig_hail                     0.0023178202  0.0023510398  613404.94   0.0026450232
# sig_hail                     0.0030434355  0.00295975    467368.8    0.0033051437
# sig_hail                     0.003672403   0.00364622    387185.9    0.004014159
# sig_hail                     0.0043777893  0.0043652044  324800.38   0.004732903
# sig_hail                     0.0054701753  0.005127069   259978.28   0.005571102
# sig_hail                     0.0058137854  0.006111384   244702.9    0.0067338077
# sig_hail                     0.0073693227  0.0073966407  193047.25   0.00816771
# sig_hail                     0.008702842   0.00905629    163393.56   0.010077691
# sig_hail                     0.011695439   0.011109824   121624.31   0.012307547
# sig_hail                     0.013128468   0.013785124   108296.38   0.015555069
# sig_hail                     0.017100515   0.017555939   83184.47    0.019888023
# sig_hail                     0.024237033   0.022286052   58693.44    0.025278533
# sig_hail                     0.031395745   0.029240599   45288.848   0.034825012
# sig_hail                     0.04681459    0.047757152   30188.738   1.0
# sig_tornado_gated_by_tornado 6.302589e-7   7.182884e-7   6.9730816e8 3.3574852e-5
# sig_tornado_gated_by_tornado 8.967794e-5   7.7806995e-5  4.90218e6   0.00016840432
# sig_tornado_gated_by_tornado 0.00024817706 0.000265865   1.771311e6  0.00041199927
# sig_tornado_gated_by_tornado 0.00051613967 0.00054935203 852665.5    0.0007302659
# sig_tornado_gated_by_tornado 0.0008330432  0.00093932    528195.8    0.001231456
# sig_tornado_gated_by_tornado 0.0017015577  0.0014930425  258320.4    0.0018325506
# sig_tornado_gated_by_tornado 0.0021746664  0.002269171   202131.94   0.0027703033
# sig_tornado_gated_by_tornado 0.0031450652  0.0032401416  139781.14   0.0038170351
# sig_tornado_gated_by_tornado 0.0042994996  0.0044404706  102274.555  0.0052007576
# sig_tornado_gated_by_tornado 0.007062154   0.005880263   62300.543   0.006684648
# sig_tornado_gated_by_tornado 0.0088073695  0.0075628147  49902.71    0.008563295
# sig_tornado_gated_by_tornado 0.009809098   0.009749743   44861.94    0.011120642
# sig_tornado_gated_by_tornado 0.014496204   0.012330667   30325.43    0.013680984
# sig_tornado_gated_by_tornado 0.014378119   0.015405993   30600.826   0.017518433
# sig_tornado_gated_by_tornado 0.017365474   0.020096652   25344.516   0.023232374
# sig_tornado_gated_by_tornado 0.028734975   0.025981048   15294.354   0.029258389
# sig_tornado_gated_by_tornado 0.034875296   0.033482328   12610.21    0.038910735
# sig_tornado_gated_by_tornado 0.043232504   0.046440676   10172.941   0.056332543
# sig_tornado_gated_by_tornado 0.08532565    0.065858305   5154.6875   0.079786226
# sig_tornado_gated_by_tornado 0.14123648    0.10858203    3053.138    1.0
# sig_wind_gated_by_wind       3.9612237e-6  3.662285e-6   6.73231e8   8.695716e-5
# sig_wind_gated_by_wind       0.00020478632 0.00015562537 1.3023182e7 0.00026710462
# sig_wind_gated_by_wind       0.00038580652 0.00040120847 6.9116015e6 0.00059044163
# sig_wind_gated_by_wind       0.0007398425  0.00076363154 3.6037328e6 0.0009785821
# sig_wind_gated_by_wind       0.0011514088  0.0011788331  2.3156758e6 0.0014143221
# sig_wind_gated_by_wind       0.0015394853  0.0016512779  1.7318795e6 0.0019224314
# sig_wind_gated_by_wind       0.002132129   0.0021734545  1.2508146e6 0.002466119
# sig_wind_gated_by_wind       0.0027594941  0.0027933582  966222.0    0.0031891225
# sig_wind_gated_by_wind       0.0037032824  0.0036233203  720049.25   0.004152767
# sig_wind_gated_by_wind       0.00453644    0.0048067686  587780.7    0.005618073
# sig_wind_gated_by_wind       0.006005268   0.006418907   444093.22   0.0073130066
# sig_wind_gated_by_wind       0.008202279   0.008096789   325106.75   0.008940162
# sig_wind_gated_by_wind       0.010677682   0.009667546   249700.97   0.010425533
# sig_wind_gated_by_wind       0.011989084   0.011265771   222442.39   0.012294381
# sig_wind_gated_by_wind       0.014109744   0.013374586   188993.95   0.01455917
# sig_wind_gated_by_wind       0.015136881   0.015887503   176141.88   0.017349068
# sig_wind_gated_by_wind       0.017557878   0.018884273   151861.47   0.02070126
# sig_wind_gated_by_wind       0.023211686   0.022844542   114865.31   0.025555518
# sig_wind_gated_by_wind       0.032814723   0.028976081   81265.2     0.03396688
# sig_wind_gated_by_wind       0.04562977    0.045748796   58245.617   1.0
# sig_hail_gated_by_hail       2.0561513e-6  1.8400787e-6  6.916353e8  0.00012675468
# sig_hail_gated_by_hail       0.00027106964 0.00021342444 5.2468915e6 0.0003455947
# sig_hail_gated_by_hail       0.0004918202  0.0004963276  2.8923572e6 0.0007029206
# sig_hail_gated_by_hail       0.00091127097 0.0008909939  1.5602256e6 0.0011205256
# sig_hail_gated_by_hail       0.0012830608  0.0013479741  1.1087965e6 0.0016097318
# sig_hail_gated_by_hail       0.0017705082  0.0018484303  803382.9    0.0021084556
# sig_hail_gated_by_hail       0.002288989   0.0023458595  621372.44   0.0026427328
# sig_hail_gated_by_hail       0.003050013   0.0029568179  466339.38   0.0033016822
# sig_hail_gated_by_hail       0.0036771693  0.0036425344  386654.47   0.004010433
# sig_hail_gated_by_hail       0.004375538   0.004362241   325012.34   0.0047308113
# sig_hail_gated_by_hail       0.0054804753  0.005124128   259476.84   0.0055672345
# sig_hail_gated_by_hail       0.0057958863  0.006109305   245453.33   0.0067338394
# sig_hail_gated_by_hail       0.0073692654  0.0073973755  193041.38   0.008169025
# sig_hail_gated_by_hail       0.008707519   0.009057851   163305.88   0.010079546
# sig_hail_gated_by_hail       0.011702028   0.011111906   121554.73   0.0123103075
# sig_hail_gated_by_hail       0.013108428   0.013793052   108532.54   0.015570249
# sig_hail_gated_by_hail       0.017164396   0.017568279   82880.484   0.019894622
# sig_hail_gated_by_hail       0.024283849   0.022290794   58578.574   0.025279433
# sig_hail_gated_by_hail       0.031356428   0.029249929   45346.637   0.034850664
# sig_hail_gated_by_hail       0.046860833   0.04778715    30117.734   1.0
