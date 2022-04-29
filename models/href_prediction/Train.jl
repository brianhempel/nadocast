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
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# disk2_date = Dates.DateTime(2021, 3, 1, 0)
# validation_forecasts = filter(forecast -> Forecasts.run_utc_datetime(forecast) >= disk2_date, validation_forecasts);

# for testing
# validation_forecasts = rand(validation_forecasts, 30);

validation_forecasts1 = filter(forecast -> isodd(forecast.run_day),  validation_forecasts);
validation_forecasts2 = filter(forecast -> iseven(forecast.run_day), validation_forecasts);

# rm("validation_forecasts_with_blurs_and_forecast_hour1"; recursive=true)
# rm("validation_forecasts_with_blurs_and_forecast_hour2"; recursive=true)

# To double loading speed, manually run the other one of these in a separate process with USE_ALT_DISK=true
# When it's done, run it in the main process and it will load from the save_dir

function dictmap(f, dict)
  out = Dict()
  for (k, v) in dict
    out[k] = f(v)
  end
  out
end

# 2020 models:
# X, y, weights =
#   TrainingShared.get_data_labels_weights(
#     validation_forecasts;
#     save_dir = "validation_forecasts_with_blurs_and_forecast_hour"
#   );

if get(ENV, "USE_ALT_DISK", "false") != "true"
  X1, Ys1, weights1 =
    TrainingShared.get_data_labels_weights(
      validation_forecasts1;
      event_name_to_labeler = TrainingShared.event_name_to_labeler,
      save_dir = "validation_forecasts_with_blurs_and_forecast_hour1"
    );

  Ys1 = dictmap(y -> y .> 0.5, Ys1) # Convert to bitarrays. This saves memory
  # Then wait for the X2, Ys2, weights2 to finish in the other process, then continue.
end

# blur_radii = [0; HREFPrediction.blur_radii]
# X1 = X1[:,1:2*length(blur_radii)]

GC.gc()

X2, Ys2, weights2 =
  TrainingShared.get_data_labels_weights(
    validation_forecasts2;
    event_name_to_labeler = TrainingShared.event_name_to_labeler,
    save_dir = "validation_forecasts_with_blurs_and_forecast_hour2"
  );
Ys2 = dictmap(y -> y .> 0.5, Ys2) # Convert to bitarrays. This saves memory

GC.gc()

if get(ENV, "USE_ALT_DISK", "false") == "true"
  exit(0)
end

# blur_radii = [0; HREFPrediction.blur_radii]
# X2 = X2[:,1:2*length(blur_radii)]

Ys = Dict{String, BitArray}();
for event_name in keys(Ys1)
  Ys[event_name] = vcat(Ys1[event_name], Ys2[event_name])
end

GC.gc()

weights = vcat(weights1, weights2);

# Free
Ys1, weights1 = (nothing, nothing)
Ys2, weights2 = (nothing, nothing)

GC.gc()

X = vcat(X1, X2);

# Free
X1 = nothing
X2 = nothing

GC.gc()

# X, Ys, weights =
#     TrainingShared.get_data_labels_weights(
#       validation_forecasts;
#       event_name_to_labeler = TrainingShared.event_name_to_labeler,
#       save_dir = "validation_forecasts_with_blurs_and_forecast_hour"
#     );

length(validation_forecasts) # 21381
size(X) # (771959580, 43)
length(weights) # 771959580

sum(Ys["tornado"]) # 68134
sum(weights) # 7.063547f8

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
# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13140#:~:text=Guo%2C%202013).-,The%20area%20under%20the%20precision%2Drecall%20curve%20(AUC%2DPR,Davis%20%26%20Goadrich%2C%202006).
#
# ...which is the same as the area to the left of the performance diagram curve (which is what we want to optimize for anyway).

# ROC AUC:
# 2020 tornado models on same dataset: 0.9845657137724223
# 2021:                                0.9840355246144981
# 2021 w/higher feat fraction:         0.9827257079052968

y = Ys["tornado"];

const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)
sum(logloss.(y, (@view X[:,1])) .* weights) / sum(weights)
# 2020 tornado models on same dataset: 0.0005586252f0
# 2021 tornado models:                 0.00055437785f0 # OKAY so this is what you get for training with logloss. there is justice in the world
# 2021 w/higher feat fraction:         0.00055674755f0


σ(x) = 1.0f0 / (1.0f0 + exp(-x))
logit(p) = log(p / (one(p) - p))

import LogisticRegression

function logloss_rescaled(x, y, weights)
  X = reshape(logit.(x), (length(x),1))
  a, b = LogisticRegression.fit(X, y, weights)
  x_rescaled = σ.(logit.(x) .* a .+ b)
  sum(logloss.(y, x_rescaled) .* weights) / sum(weights)
end

                                             # 2020:                  0.0005585756f0
logloss_rescaled((@view X[:,1]), y, weights) # 2021:                  0.0005543472f0
logloss_rescaled((@view X[:,8]), y, weights) # 2021 higher feat frac: 0.0005567189f0

# HMM, should be using area under the precision-recall curve
# b/c the precision-recall curve is a (flipped) performance diagram

# Can't use SPC historical PODs because those are daily
# SR  = true_pos / painted
# POD = true_pos / total_pos
function sr_for_target_pod(x, y, weights, target_pod)
  x_positive       = x[y .>= 0.5f0]
  weights_positive = weights[y .>= 0.5f0]
  total_pos        = sum(weights_positive)
  threshold = 0.5f0
  Δ = 0.25f0
  while Δ > 0.0000000001
    true_pos = sum(weights_positive[x_positive .>= threshold])
    pod = true_pos / total_pos
    if pod > target_pod
      threshold += Δ
    else
      threshold -= Δ
    end
    Δ *= 0.5f0
  end
  true_pos = sum(weights_positive[x_positive .>= threshold])
  painted  = sum(weights[x .>= threshold])
  println(threshold)
  true_pos / painted
end

                                                    # 2020:                  0.0087980125f0
sr_for_target_pod((@view X[:,1]), y, weights, 0.75) # 2021:                  0.009115579f0
sr_for_target_pod((@view X[:,8]), y, weights, 0.75) # 2021 higher feat frac: 0.009057991f0

                                                   # 2020:                  0.023044856f0
sr_for_target_pod((@view X[:,1]), y, weights, 0.5) # 2021:                  0.023928536f0
sr_for_target_pod((@view X[:,8]), y, weights, 0.5) # 2021 higher feat frac: 0.023018489f0


                                                    # 2020:                  0.051686835f0
sr_for_target_pod((@view X[:,1]), y, weights, 0.25) # 2021:                  0.053621564f0
sr_for_target_pod((@view X[:,8]), y, weights, 0.25) # 2021 higher feat frac: 0.051918864f0

                                                   # 2020:                  0.077482425f0
sr_for_target_pod((@view X[:,1]), y, weights, 0.1) # 2021:                  0.09632159f0
sr_for_target_pod((@view X[:,8]), y, weights, 0.1) # 2021 higher feat frac: 0.1018713f0

function au_pr_curve_est(x, y, weights)
  area = 0
  for target_pod in 0.01:0.02:0.99
    area += 0.02 * sr_for_target_pod(x, y, weights, target_pod)
  end
  area
end

au_pr_curve_est((@view X[:,1]), y, weights)     # 2021: 0.03809332829870984

                                                # 2020:                  0.03268739260172207
area_under_pr_curve((@view X[:,1]), y, weights) # 2021:                  0.03809290465346103
area_under_pr_curve((@view X[:,8]), y, weights) # 2021 higher feat frac: 0.037736602971328484



function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPrediction.models)
    (event_name, _, _) = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    for j in 1:size(X,2)
      x = @view X[:,j]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)


# tornado (68134.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.038092913737913965
# tornado (68134.0)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.03856512528258033 ***best tor***
# tornado (68134.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.03845738122730976
# tornado (68134.0)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.037783075394596896
# tornado (68134.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.03586202958529153
# tornado (68134.0)    feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.032635863699013196
# tornado (68134.0)    feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.027478088541143805
# tornado (68134.0)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.009138211652795516
# tornado (68134.0)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.009153928145848759
# tornado (68134.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.009096792262488855
# tornado (68134.0)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0089207569549636
# tornado (68134.0)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0085562486094652
# tornado (68134.0)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.007815819662820645
# tornado (68134.0)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.006618790180379198
# tornado (68134.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.005503808613302046
# tornado (68134.0)    feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.005557626076828263
# tornado (68134.0)    feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.005568308265734539
# tornado (68134.0)    feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.005528017957230592
# tornado (68134.0)    feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.005366238417300644
# tornado (68134.0)    feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.005013933282289027
# tornado (68134.0)    feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.004325847107365612
# tornado (68134.0)    feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.030279414344508843
# tornado (68134.0)    feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.030781845313049006
# tornado (68134.0)    feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0307313441846042
# tornado (68134.0)    feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.03026258939257635
# tornado (68134.0)    feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.029036629860555364
# tornado (68134.0)    feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.02676152279379891
# tornado (68134.0)    feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.022927036019328127
# tornado (68134.0)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.008029663062378034
# tornado (68134.0)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.008056663805061742
# tornado (68134.0)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.008014100037020824
# tornado (68134.0)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.007871403222770698
# tornado (68134.0)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.007571982275867091
# tornado (68134.0)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.007027223460010957
# tornado (68134.0)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0061186815950853785
# tornado (68134.0)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0042083250880540855
# tornado (68134.0)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004245717403462443
# tornado (68134.0)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.004225796880890934
# tornado (68134.0)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004143293996058427
# tornado (68134.0)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0040000605534679465
# tornado (68134.0)    feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0037472009142315737
# tornado (68134.0)    feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003307826526581784
# tornado (68134.0)    feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 9.023639634213338e-5
# wind (562866.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.04918463256936391
# wind (562866.0)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.049519221319923104
# wind (562866.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.049432050846926125
# wind (562866.0)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.048922509880212954
# wind (562866.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.0475990396362673
# wind (562866.0)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.04486746284022893
# wind (562866.0)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.039745146699839556
# wind (562866.0)      feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.11387557670254292
# wind (562866.0)      feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.11519925200319703
# wind (562866.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.11548662037783414 ***best wind***
# wind (562866.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.11498115747699791
# wind (562866.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.11273335372993429
# wind (562866.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.10718166918997074
# wind (562866.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.09650616070474936
# wind (562866.0)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.029397657049463472
# wind (562866.0)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.029340638488842247
# wind (562866.0)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.029095213349960628
# wind (562866.0)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.028542748043704602
# wind (562866.0)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.027398446404583828
# wind (562866.0)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.025588920265580982
# wind (562866.0)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.02258128098611348
# wind (562866.0)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.03994307508533854
# wind (562866.0)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.04018644839544405
# wind (562866.0)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0400939445905491
# wind (562866.0)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.039643639889518176
# wind (562866.0)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.038520980257138716
# wind (562866.0)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0362207392906237
# wind (562866.0)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.03206042508756286
# wind (562866.0)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07179553395962818
# wind (562866.0)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.07231771148823514
# wind (562866.0)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.07225372527455799
# wind (562866.0)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.07165441882275721
# wind (562866.0)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0700819551077755
# wind (562866.0)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.06679649820937893
# wind (562866.0)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.060329869754028306
# wind (562866.0)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.02027317393706811
# wind (562866.0)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.020215979752234735
# wind (562866.0)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.02004178692738622
# wind (562866.0)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.019687677654435846
# wind (562866.0)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01898013902344102
# wind (562866.0)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.017872853472723476
# wind (562866.0)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01600741925450469
# wind (562866.0)      feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 0.0007303467435103636
# hail (246689.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.01637261248335978
# hail (246689.0)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.016370586636360776
# hail (246689.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.016261919954696846
# hail (246689.0)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.01598924533026332
# hail (246689.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.015386401080774938
# hail (246689.0)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.014384973485001446
# hail (246689.0)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.012661499570824016
# hail (246689.0)      feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.01725518624006352
# hail (246689.0)      feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.01727356725859199
# hail (246689.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.017172912058746048
# hail (246689.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.01692710551492894
# hail (246689.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.01633817289393868
# hail (246689.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.015403977471742478
# hail (246689.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.01382022306737807
# hail (246689.0)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.07338978299850449
# hail (246689.0)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.07443952832328785
# hail (246689.0)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.07445061357719689 ***best hail***
# hail (246689.0)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.07349545712340752
# hail (246689.0)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.07053449459238521
# hail (246689.0)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.06480901004735605
# hail (246689.0)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.05451709269494041
# hail (246689.0)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.01210756289590851
# hail (246689.0)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.012148114641810164
# hail (246689.0)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.012110632880191825
# hail (246689.0)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.011979329081814993
# hail (246689.0)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.011668818144941553
# hail (246689.0)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.011075362492995929
# hail (246689.0)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.009953654252707958
# hail (246689.0)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.020153606212438682
# hail (246689.0)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.020200490514324557
# hail (246689.0)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.020121779063159945
# hail (246689.0)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.01989453589038338
# hail (246689.0)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.019322016214765564
# hail (246689.0)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.018313401317087842
# hail (246689.0)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.016409278629683586
# hail (246689.0)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.052807506263340305
# hail (246689.0)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.053293503180016945
# hail (246689.0)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.05310005871427087
# hail (246689.0)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.05224484224404801
# hail (246689.0)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.049949320025603185
# hail (246689.0)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0459187345525047
# hail (246689.0)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.038791941883513786
# hail (246689.0)      feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 0.00031771735739933725
# sig_tornado (9182.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.020184379534294042
# sig_tornado (9182.0) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.021144926654606173
# sig_tornado (9182.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.021580260324859362
# sig_tornado (9182.0) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.021904851224180396
# sig_tornado (9182.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.02180155822768238
# sig_tornado (9182.0) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.021381120089437884
# sig_tornado (9182.0) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.019070311605259287
# sig_tornado (9182.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.002558607805941661
# sig_tornado (9182.0) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.0025719149597759395
# sig_tornado (9182.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.002559434398368522
# sig_tornado (9182.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.00250682586597828
# sig_tornado (9182.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0024204158409261993
# sig_tornado (9182.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.002198464887538905
# sig_tornado (9182.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.001852951677827093
# sig_tornado (9182.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0009792484051565032
# sig_tornado (9182.0) feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.0009883905826187278
# sig_tornado (9182.0) feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0009926773120351624
# sig_tornado (9182.0) feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0009923050240455804
# sig_tornado (9182.0) feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0009818430744605376
# sig_tornado (9182.0) feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0009380236570838718
# sig_tornado (9182.0) feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0008168493857262371
# sig_tornado (9182.0) feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.0324614175173982
# sig_tornado (9182.0) feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.0337699770470802
# sig_tornado (9182.0) feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.034055771154099815 ***best sigtor***
# sig_tornado (9182.0) feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.033751310075924794
# sig_tornado (9182.0) feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.03218272774563538
# sig_tornado (9182.0) feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.028903409792467895
# sig_tornado (9182.0) feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.02339701297718336
# sig_tornado (9182.0) feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0020895934407504377
# sig_tornado (9182.0) feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.002092873006957537
# sig_tornado (9182.0) feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0020745528726702107
# sig_tornado (9182.0) feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0020270281192032147
# sig_tornado (9182.0) feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0019469680859835168
# sig_tornado (9182.0) feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0017953926942175426
# sig_tornado (9182.0) feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0015543784638183717
# sig_tornado (9182.0) feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0007247091495270593
# sig_tornado (9182.0) feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0007232785583409606
# sig_tornado (9182.0) feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0007184958944572933
# sig_tornado (9182.0) feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0007084174491098745
# sig_tornado (9182.0) feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0006882672783247524
# sig_tornado (9182.0) feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0006538859882123982
# sig_tornado (9182.0) feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0005932834146331394
# sig_tornado (9182.0) feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 1.2735864479380344e-5
# sig_wind (57701.0)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.0060210072176445145
# sig_wind (57701.0)   feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.006048969453979045
# sig_wind (57701.0)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.006022840470670077
# sig_wind (57701.0)   feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.0059372341635195195
# sig_wind (57701.0)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.0057279642086105744
# sig_wind (57701.0)   feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.005342553684812942
# sig_wind (57701.0)   feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.004656146055115123
# sig_wind (57701.0)   feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.010766316380641511
# sig_wind (57701.0)   feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.010824801872930988
# sig_wind (57701.0)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.010781795360701119
# sig_wind (57701.0)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.010648535541028771
# sig_wind (57701.0)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.010331358814089487
# sig_wind (57701.0)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.009769018181861192
# sig_wind (57701.0)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.008766611596835257
# sig_wind (57701.0)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.00500354359593779
# sig_wind (57701.0)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.005032446282118618
# sig_wind (57701.0)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0050209789298699455
# sig_wind (57701.0)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.004965501899952111
# sig_wind (57701.0)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0048105855576805415
# sig_wind (57701.0)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.004507640259407646
# sig_wind (57701.0)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.003957855311428943
# sig_wind (57701.0)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.005491124046034227
# sig_wind (57701.0)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.005509467028734762
# sig_wind (57701.0)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.005489005027342889
# sig_wind (57701.0)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.005418662009577087
# sig_wind (57701.0)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.00525597087691415
# sig_wind (57701.0)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.00493524817965473
# sig_wind (57701.0)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.004360755760358743
# sig_wind (57701.0)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01601545251926452
# sig_wind (57701.0)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.016222622433558598
# sig_wind (57701.0)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.016241121316946938 ***best sigwind***
# sig_wind (57701.0)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.016137694990441976
# sig_wind (57701.0)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.01581277364880481
# sig_wind (57701.0)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.015141527379450011
# sig_wind (57701.0)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013747707743660425
# sig_wind (57701.0)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0041438678826870865
# sig_wind (57701.0)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.004146304300789253
# sig_wind (57701.0)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.0041226884886457335
# sig_wind (57701.0)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.004067287481384556
# sig_wind (57701.0)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.003942654070018957
# sig_wind (57701.0)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0037227515226612275
# sig_wind (57701.0)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003325584685750076
# sig_wind (57701.0)   feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 7.421561014472997e-5
# sig_hail (30597.0)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:              AU-PR-curve: 0.0027471818967157225
# sig_hail (30597.0)   feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     AU-PR-curve: 0.002749050920758793
# sig_hail (30597.0)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     AU-PR-curve: 0.00273302580973768
# sig_hail (30597.0)   feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     AU-PR-curve: 0.002685779527336153
# sig_hail (30597.0)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     AU-PR-curve: 0.0025799616482731326
# sig_hail (30597.0)   feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     AU-PR-curve: 0.0023900383225517316
# sig_hail (30597.0)   feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    AU-PR-curve: 0.002064537744714158
# sig_hail (30597.0)   feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0021542919258318014
# sig_hail (30597.0)   feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.0021502725642520326
# sig_hail (30597.0)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0021314589974564153
# sig_hail (30597.0)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0020923526682915593
# sig_hail (30597.0)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.002009398184703421
# sig_hail (30597.0)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0018811438978874185
# sig_hail (30597.0)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0016824989046255432
# sig_hail (30597.0)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.01518139135734051
# sig_hail (30597.0)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.01534259206035175
# sig_hail (30597.0)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.015273089283459905
# sig_hail (30597.0)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.014948374469927993
# sig_hail (30597.0)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.01428404617962779
# sig_hail (30597.0)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.012984404855172174
# sig_hail (30597.0)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.010837741356282807
# sig_hail (30597.0)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.002048044614833259
# sig_hail (30597.0)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.0020522768842563778
# sig_hail (30597.0)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   AU-PR-curve: 0.0020404803563425014
# sig_hail (30597.0)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   AU-PR-curve: 0.0020068326342661306
# sig_hail (30597.0)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   AU-PR-curve: 0.0019360064615904814
# sig_hail (30597.0)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   AU-PR-curve: 0.0018136608623679378
# sig_hail (30597.0)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  AU-PR-curve: 0.0016051746945991898
# sig_hail (30597.0)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032941122590802105
# sig_hail (30597.0)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.003313157617177453
# sig_hail (30597.0)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.003304428458238803
# sig_hail (30597.0)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0032713038054582705
# sig_hail (30597.0)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.0031818200623134783
# sig_hail (30597.0)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0030361237740777584
# sig_hail (30597.0)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027028669361492726
# sig_hail (30597.0)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015346508970791128
# sig_hail (30597.0)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.015531681729515856 ***best sighail***
# sig_hail (30597.0)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.01546989425889235
# sig_hail (30597.0)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.015140278728833254
# sig_hail (30597.0)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.014363193551997125
# sig_hail (30597.0)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.012943015408153992
# sig_hail (30597.0)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01065905099940301
# sig_hail (30597.0)   feature 43 forecast_hour:calculated:hour fcst::                      AU-PR-curve: 3.993461625611225e-5


# tornado (68134)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9840355246144981
# tornado (68134)    feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.9842087298847911
# tornado (68134)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.9842905084142878
# tornado (68134)    feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9843224648698559
# tornado (68134)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9842305685061311
# tornado (68134)    feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.9838451074882943
# tornado (68134)    feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.9828250760453755
# tornado (68134)    feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.9752727668762812
# tornado (68134)    feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9754622419305661
# tornado (68134)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9755315499831028
# tornado (68134)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9755140633388613
# tornado (68134)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9752878942227615
# tornado (68134)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.974624000316682
# tornado (68134)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9730014983644707
# tornado (68134)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9569290994454451
# tornado (68134)    feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.957091236588984
# tornado (68134)    feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9570794246317931
# tornado (68134)    feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9569062443284628
# tornado (68134)    feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9563821842888789
# tornado (68134)    feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9551897262497847
# tornado (68134)    feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9524699076237695
# tornado (68134)    feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9653198225106664
# tornado (68134)    feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9660965252829078
# tornado (68134)    feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9666084436143448
# tornado (68134)    feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.967202273251932
# tornado (68134)    feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9677996134238476
# tornado (68134)    feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9680848598389654
# tornado (68134)    feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9676042525937988
# tornado (68134)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9743953882993837
# tornado (68134)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9746443638269222
# tornado (68134)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9747420652190675
# tornado (68134)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9747738155135705
# tornado (68134)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9745529416207602
# tornado (68134)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9738864516875314
# tornado (68134)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9723123359151281
# tornado (68134)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9508896905502625
# tornado (68134)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9512800106218378
# tornado (68134)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9513793082733829
# tornado (68134)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.951414849835245
# tornado (68134)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.951277965806476
# tornado (68134)    feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9506144569951479
# tornado (68134)    feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9486244731293917
# tornado (68134)    feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5016091986286819
# wind (562866)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9696341214753226
# wind (562866)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.9698418182243459
# wind (562866)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.9698689674702873
# wind (562866)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9697583069377972
# wind (562866)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9693579376237849
# wind (562866)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.9683343612024371
# wind (562866)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.9659040947728529
# wind (562866)      feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.9877188638325693
# wind (562866)      feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9878209269831755
# wind (562866)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9878423433566331
# wind (562866)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9878022368286115
# wind (562866)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9876065341548111
# wind (562866)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9871437833579424
# wind (562866)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9860516088518202
# wind (562866)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9700601139048584
# wind (562866)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.97015179249165
# wind (562866)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9700604109381846
# wind (562866)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9697894631923102
# wind (562866)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.969102550437933
# wind (562866)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9677823271175066
# wind (562866)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9650265666015942
# wind (562866)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9369067880022091
# wind (562866)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.938546777913046
# wind (562866)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9394556344428143
# wind (562866)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9404443526767422
# wind (562866)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9413671620243393
# wind (562866)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9415408633791745
# wind (562866)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9401051519138512
# wind (562866)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9815294901496647
# wind (562866)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9816454585842116
# wind (562866)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9816321773627967
# wind (562866)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9815167529937698
# wind (562866)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9811557698603398
# wind (562866)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9803662316632074
# wind (562866)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9785851406514754
# wind (562866)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9488592796475761
# wind (562866)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9495671394136685
# wind (562866)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9498375210957769
# wind (562866)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9500163827162234
# wind (562866)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9498515041531287
# wind (562866)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9489564675968126
# wind (562866)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9464494585416365
# wind (562866)      feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5008397681596708
# hail (246689)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9722849575949352
# hail (246689)      feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.9726417525949125
# hail (246689)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.9728386911996808
# hail (246689)      feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9729724086032853
# hail (246689)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9729092725982724
# hail (246689)      feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.9722653158032915
# hail (246689)      feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.9703039782601859
# hail (246689)      feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.9770329087900397
# hail (246689)      feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9771346438209834
# hail (246689)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9771203461767355
# hail (246689)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9769674906375759
# hail (246689)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9765359948302414
# hail (246689)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9756419076014462
# hail (246689)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9737088291224677
# hail (246689)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9894175813042464
# hail (246689)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9895152594626779
# hail (246689)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9895427840813774
# hail (246689)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.989514604831788
# hail (246689)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9893616012893902
# hail (246689)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9889833370620407
# hail (246689)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9880732708658999
# hail (246689)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9422904519313006
# hail (246689)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9441333105205225
# hail (246689)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9453118990552611
# hail (246689)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.946625900313264
# hail (246689)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.948005039082355
# hail (246689)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9486906529694806
# hail (246689)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9478278951388955
# hail (246689)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9785022479377982
# hail (246689)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9786954726767476
# hail (246689)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.978757297968017
# hail (246689)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9787255484855124
# hail (246689)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9784936775237554
# hail (246689)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.977871219392515
# hail (246689)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9763647842359479
# hail (246689)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9835310129250558
# hail (246689)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9837753523877246
# hail (246689)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9838726491324181
# hail (246689)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9839163739350312
# hail (246689)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9838039853259698
# hail (246689)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9833612877608698
# hail (246689)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9821922401061375
# hail (246689)      feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5013302604061118
# sig_tornado (9182) feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9814091126359091
# sig_tornado (9182) feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.981555227734666
# sig_tornado (9182) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.9815639305881898
# sig_tornado (9182) feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9814419362270701
# sig_tornado (9182) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9809641871896386
# sig_tornado (9182) feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.979939847827281
# sig_tornado (9182) feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.9775154480151279
# sig_tornado (9182) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.9805910077570996
# sig_tornado (9182) feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9806364342757449
# sig_tornado (9182) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9805669674659064
# sig_tornado (9182) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9803757379817456
# sig_tornado (9182) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9798575845567538
# sig_tornado (9182) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9788262952729234
# sig_tornado (9182) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9764574682347447
# sig_tornado (9182) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9615248820532448
# sig_tornado (9182) feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9616389884118705
# sig_tornado (9182) feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9617313214367759
# sig_tornado (9182) feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9618214732740484
# sig_tornado (9182) feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.961756096687927
# sig_tornado (9182) feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9610995615569137
# sig_tornado (9182) feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9591730012191421
# sig_tornado (9182) feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9760490043282434
# sig_tornado (9182) feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9761076109782018
# sig_tornado (9182) feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9762215533108984
# sig_tornado (9182) feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9763829116844673
# sig_tornado (9182) feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9764341286304696
# sig_tornado (9182) feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9762542859130083
# sig_tornado (9182) feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9757399067368727
# sig_tornado (9182) feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9840856770372935
# sig_tornado (9182) feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9844842325504081
# sig_tornado (9182) feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9847800885887714
# sig_tornado (9182) feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9850841557961958
# sig_tornado (9182) feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.985022344302668
# sig_tornado (9182) feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9842920209713709
# sig_tornado (9182) feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9821883454303242
# sig_tornado (9182) feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9635258719704443
# sig_tornado (9182) feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.963467080679084
# sig_tornado (9182) feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9633952333387609
# sig_tornado (9182) feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9632392973618394
# sig_tornado (9182) feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9628675973806194
# sig_tornado (9182) feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9620043091914576
# sig_tornado (9182) feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9603247221075685
# sig_tornado (9182) feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5089708075049733
# sig_wind (57701)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9785330931942395
# sig_wind (57701)   feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.9787139326315362
# sig_wind (57701)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.978708500381532
# sig_wind (57701)   feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9785401543877433
# sig_wind (57701)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9780901870389365
# sig_wind (57701)   feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.9770882305464198
# sig_wind (57701)   feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.9749458602633532
# sig_wind (57701)   feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.9896065772305997
# sig_wind (57701)   feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9896911812277263
# sig_wind (57701)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9897066121042192
# sig_wind (57701)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9896640517666431
# sig_wind (57701)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.989494029000095
# sig_wind (57701)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9890677719099696
# sig_wind (57701)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9880384550890516
# sig_wind (57701)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9754838842452599
# sig_wind (57701)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.975617582810376
# sig_wind (57701)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.975602049140786
# sig_wind (57701)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9754790137890281
# sig_wind (57701)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9751305832134394
# sig_wind (57701)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9743085240938175
# sig_wind (57701)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9723762730060685
# sig_wind (57701)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9610459950612732
# sig_wind (57701)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9619890820712477
# sig_wind (57701)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9624744259869521
# sig_wind (57701)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.962873097294136
# sig_wind (57701)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9630606533866749
# sig_wind (57701)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9625222565311132
# sig_wind (57701)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9607389889069148
# sig_wind (57701)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9893708858909429
# sig_wind (57701)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9894742230277107
# sig_wind (57701)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9894937093969121
# sig_wind (57701)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9894589035268253
# sig_wind (57701)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9892910847090777
# sig_wind (57701)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9888472155979994
# sig_wind (57701)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9877318434061383
# sig_wind (57701)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9653669518365001
# sig_wind (57701)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9659540697801268
# sig_wind (57701)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9661820597601171
# sig_wind (57701)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9663710618743391
# sig_wind (57701)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9663319600889273
# sig_wind (57701)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9656132881593775
# sig_wind (57701)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9636049972874006
# sig_wind (57701)   feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5004555265116033
# sig_hail (30597)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:              ROC: 0.9817888157266031
# sig_hail (30597)   feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean     ROC: 0.9821093905909697
# sig_hail (30597)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean     ROC: 0.9823347100610099
# sig_hail (30597)   feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean     ROC: 0.9825671340981093
# sig_hail (30597)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean     ROC: 0.9827944397496
# sig_hail (30597)   feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean     ROC: 0.9826557110176243
# sig_hail (30597)   feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean    ROC: 0.98166116799069
# sig_hail (30597)   feature 8 WINDPROB:calculated:hour fcst:calculated_prob:             ROC: 0.981624693303902
# sig_hail (30597)   feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean    ROC: 0.9817258658237329
# sig_hail (30597)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9817139858465291
# sig_hail (30597)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9815539987418039
# sig_hail (30597)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9811461989102662
# sig_hail (30597)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.980272680936157
# sig_hail (30597)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9785257713292488
# sig_hail (30597)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9942728154891438
# sig_hail (30597)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9943682139319756
# sig_hail (30597)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.9944268184245656
# sig_hail (30597)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9944609929199903
# sig_hail (30597)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9944426169509802
# sig_hail (30597)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.994249006288922
# sig_hail (30597)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9937124349135704
# sig_hail (30597)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:            ROC: 0.9659558176029186
# sig_hail (30597)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean   ROC: 0.9672168515584484
# sig_hail (30597)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean   ROC: 0.968037679792683
# sig_hail (30597)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean   ROC: 0.9689740957876222
# sig_hail (30597)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean   ROC: 0.9700648802407524
# sig_hail (30597)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean   ROC: 0.9706981818212058
# sig_hail (30597)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean  ROC: 0.9702549268871546
# sig_hail (30597)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9866705848185333
# sig_hail (30597)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9868322475068843
# sig_hail (30597)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9869125925109479
# sig_hail (30597)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9869321143040413
# sig_hail (30597)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9868287587644955
# sig_hail (30597)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9864054755293961
# sig_hail (30597)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9853683916622112
# sig_hail (30597)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           ROC: 0.9949766105963571
# sig_hail (30597)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  ROC: 0.9950623064218848
# sig_hail (30597)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  ROC: 0.9951076768665803
# sig_hail (30597)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  ROC: 0.9951336831133981
# sig_hail (30597)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  ROC: 0.9950758966862242
# sig_hail (30597)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  ROC: 0.9948403959347146
# sig_hail (30597)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean ROC: 0.9942304556830814
# sig_hail (30597)   feature 43 forecast_hour:calculated:hour fcst::                      ROC: 0.5050937219215438

println("Determining best blur radii to maximize area under precision-recall curve")

blur_radii = [0; HREFPrediction.blur_radii]
forecast_hour_j = size(X, 2)

bests = []
for prediction_i in 1:length(HREFPrediction.models)
  (event_name, _, _) = HREFPrediction.models[prediction_i]
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
# 0       0       0.0380929
# 0       15      0.03830932
# 0       25      0.03828504
# 0       35      0.03811438
# 0       50      0.037652444
# 0       70      0.036949728
# 0       100     0.03583728
# 15      0       0.03849197
# 15      15      0.03856513
# 15      25      0.038526017
# 15      35      0.03833931
# 15      50      0.037866782
# 15      70      0.03715156
# 15      100     0.03603181
# 25      0       0.03848133
# 25      15      0.03853256
# 25      25      0.03845738
# 25      35      0.038236875
# 25      50      0.037724618
# 25      70      0.036972165
# 25      100     0.03582087
# 35      0       0.03815995
# 35      15      0.038194023
# 35      25      0.038077522
# 35      35      0.037783075
# 35      50      0.037182696
# 35      70      0.036343995
# 35      100     0.03511797
# 50      0       0.037174955
# 50      15      0.03718365
# 50      25      0.037015006
# 50      35      0.03661892
# 50      50      0.03586203
# 50      70      0.034865133
# 50      100     0.03349937
# 70      0       0.0356374
# 70      15      0.035625294
# 70      25      0.035397198
# 70      35      0.034873012
# 70      50      0.033898324
# 70      70      0.032635864
# 70      100     0.031073641
# 100     0       0.033395763
# 100     15      0.0334185
# 100     25      0.03315884
# 100     35      0.032520637
# 100     50      0.031279597
# 100     70      0.02964025
# 100     100     0.027478088
# Best tornado: 15        15      0.03856513

# blur_radius_f2  blur_radius_f38 AU_PR_wind
# 0       0       0.11387558
# 0       15      0.11443071
# 0       25      0.11459447
# 0       35      0.11458798
# 0       50      0.11428203
# 0       70      0.11338613
# 0       100     0.111703716
# 15      0       0.11489694
# 15      15      0.115199246
# 15      25      0.11531972
# 15      35      0.115287215
# 15      50      0.11496686
# 15      70      0.11407072
# 15      100     0.11240323
# 25      0       0.11518349
# 25      15      0.11543674
# 25      25      0.115486614
# 25      35      0.115392156
# 25      50      0.11501253
# 25      70      0.11405865
# 25      100     0.11235156
# 35      0       0.11500097
# 35      15      0.115215465
# 35      25      0.11519623
# 35      35      0.11498116
# 35      50      0.11445734
# 35      70      0.113362774
# 35      100     0.11152597
# 50      0       0.11381371
# 50      15      0.11399506
# 50      25      0.113894254
# 50      35      0.113522165
# 50      50      0.11273336
# 50      70      0.1113322
# 50      100     0.10917358
# 70      0       0.11087611
# 70      15      0.11102812
# 70      25      0.11083625
# 70      35      0.11027317
# 70      50      0.10912439
# 70      70      0.10718167
# 70      100     0.10435845
# 100     0       0.10597582
# 100     15      0.1060986
# 100     25      0.10580449
# 100     35      0.10500043
# 100     50      0.103393205
# 100     70      0.10065172
# 100     100     0.09650616
# Best wind: 25   25      0.115486614

# blur_radius_f2  blur_radius_f38 AU_PR_hail
# 0       0       0.073389776
# 0       15      0.07391185
# 0       25      0.0739978
# 0       35      0.07384685
# 0       50      0.07331552
# 0       70      0.07239307
# 0       100     0.070906155
# 15      0       0.07415522
# 15      15      0.07443954
# 15      25      0.0744809
# 15      35      0.07430244
# 15      50      0.07376281
# 15      70      0.07284978
# 15      100     0.071408145
# 25      0       0.074251525
# 25      15      0.0744874
# 25      25      0.074450605
# 25      35      0.07420511
# 25      50      0.07360149
# 25      70      0.07264397
# 25      100     0.07118752
# 35      0       0.07379301
# 35      15      0.0739977
# 35      25      0.07387899
# 35      35      0.073495455
# 35      50      0.07272919
# 35      70      0.071638405
# 35      100     0.0700942
# 50      0       0.07223941
# 50      15      0.07243
# 50      25      0.07221675
# 50      35      0.07163351
# 50      50      0.0705345
# 50      70      0.06909127
# 50      100     0.06722426
# 70      0       0.06932287
# 70      15      0.06952071
# 70      25      0.06920731
# 70      35      0.06839888
# 70      50      0.06685042
# 70      70      0.06480901
# 70      100     0.0622709
# 100     0       0.064614646
# 100     15      0.06480104
# 100     25      0.06440657
# 100     35      0.06336205
# 100     50      0.06126437
# 100     70      0.058369793
# 100     100     0.05451709
# Best hail: 25   15      0.0744874

# blur_radius_f2  blur_radius_f38 AU_PR_sig_tornado
# 0       0       0.03246142
# 0       15      0.033061862
# 0       25      0.033272173
# 0       35      0.033423796
# 0       50      0.033405352
# 0       70      0.033174813
# 0       100     0.032493047
# 15      0       0.033410273
# 15      15      0.033769976
# 15      25      0.033960946
# 15      35      0.034111153
# 15      50      0.034105085
# 15      70      0.03391962
# 15      100     0.033307504
# 25      0       0.033593286
# 25      15      0.033930384
# 25      25      0.03405577
# 25      35      0.034162313
# 25      50      0.034111384
# 25      70      0.0339117
# 25      100     0.03331883
# 35      0       0.033320095
# 35      15      0.03363533
# 35      25      0.03372108
# 35      35      0.03375131
# 35      50      0.033606347
# 35      70      0.03331573
# 35      100     0.032698747
# 50      0       0.032271024
# 50      15      0.032555506
# 50      25      0.032572202
# 50      35      0.032482762
# 50      50      0.032182727
# 50      70      0.03172961
# 50      100     0.030975612
# 70      0       0.030250784
# 70      15      0.030488998
# 70      25      0.030429272
# 70      35      0.030191215
# 70      50      0.029655166
# 70      70      0.028903408
# 70      100     0.02781331
# 100     0       0.027218424
# 100     15      0.027477967
# 100     25      0.027382178
# 100     35      0.027008526
# 100     50      0.026203575
# 100     70      0.025011359
# 100     100     0.023397012
# Best sig_tornado: 25    35      0.034162313

# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind
# 0       0       0.016015451
# 0       15      0.016128669
# 0       25      0.01616838
# 0       35      0.016189512
# 0       50      0.016188394
# 0       70      0.016154964
# 0       100     0.0160643
# 15      0       0.0161626
# 15      15      0.016222622
# 15      25      0.016254812
# 15      35      0.016272869
# 15      50      0.016269665
# 15      70      0.016236434
# 15      100     0.016149253
# 25      0       0.016174493
# 25      15      0.016223341
# 25      25      0.016241122
# 25      35      0.016247837
# 25      50      0.016234364
# 25      70      0.016193287
# 25      100     0.016102593
# 35      0       0.016103344
# 35      15      0.01614545
# 35      25      0.01615179
# 35      35      0.016137695
# 35      50      0.01610135
# 35      70      0.016038712
# 35      100     0.015926668
# 50      0       0.015902225
# 50      15      0.015938943
# 50      25      0.015932225
# 50      35      0.01589259
# 50      50      0.015812773
# 50      70      0.015703445
# 50      100     0.015538966
# 70      0       0.015522251
# 70      15      0.015554685
# 70      25      0.015534708
# 70      35      0.015466662
# 70      50      0.0153316595
# 70      70      0.015141527
# 70      100     0.014873189
# 100     0       0.014837958
# 100     15      0.014869529
# 100     25      0.014834235
# 100     35      0.014731016
# 100     50      0.014522398
# 100     70      0.014213325
# 100     100     0.013747708
# Best sig_wind: 15       35      0.016272869

# blur_radius_f2  blur_radius_f38 AU_PR_sig_hail
# 0       0       0.0153465085
# 0       15      0.015466008
# 0       25      0.015493708
# 0       35      0.015465641
# 0       50      0.015371837
# 0       70      0.01518398
# 0       100     0.014952504
# 15      0       0.015470042
# 15      15      0.0155316815
# 15      25      0.015544395
# 15      35      0.015506284
# 15      50      0.015408426
# 15      70      0.015219534
# 15      100     0.015000783
# 25      0       0.015427134
# 25      15      0.015476295
# 25      25      0.015469894
# 25      35      0.015413173
# 25      50      0.0152959535
# 25      70      0.01509281
# 25      100     0.014866378
# 35      0       0.015222007
# 35      15      0.015258313
# 35      25      0.015230694
# 35      35      0.015140278
# 35      50      0.014981992
# 35      70      0.014737622
# 35      100     0.014478411
# 50      0       0.014770379
# 50      15      0.014792764
# 50      25      0.014739467
# 50      35      0.014599809
# 50      50      0.014363194
# 50      70      0.014037584
# 50      100     0.013688218
# 70      0       0.0140114585
# 70      15      0.014028065
# 70      25      0.013948858
# 70      35      0.013750541
# 70      50      0.013408101
# 70      70      0.012943015
# 70      100     0.012451445
# 100     0       0.012915809
# 100     15      0.012934895
# 100     25      0.012841559
# 100     35      0.012589661
# 100     50      0.012114886
# 100     70      0.011460391
# 100     100     0.010659051
# Best sig_hail: 15       25      0.015544395





# blur_radius_f2  blur_radius_f38 AUC_tornado
# 0       0       0.98403555
# 0       15      0.98411876
# 0       25      0.9841687
# 0       35      0.98421746
# 0       50      0.98424155
# 0       70      0.98416454
# 0       100     0.98387045
# 15      0       0.984159
# 15      15      0.9842087
# 15      25      0.984248
# 15      35      0.9842861
# 15      50      0.98430127
# 15      70      0.98421955
# 15      100     0.98392636
# 25      0       0.98422414
# 25      15      0.98426384
# 25      25      0.9842905
# 25      35      0.98431575
# 25      50      0.98431885
# 25      70      0.98422915
# 25      100     0.9839323
# 35      0       0.98427296
# 35      15      0.9843031
# 35      25      0.9843173
# 35      35      0.9843225
# 35      50      0.98430395
# 35      70      0.98419803
# 35      100     0.98389053
# 50      0       0.9842711
# 50      15      0.98429376
# 50      25      0.9842965
# 50      35      0.9842809
# 50      50      0.9842306
# 50      70      0.98409545
# 50      100     0.9837624
# 70      0       0.9841186
# 70      15      0.9841366
# 70      25      0.9841309
# 70      35      0.98409843
# 70      50      0.9840181
# 70      70      0.9838451
# 70      100     0.9834682
# 100     0       0.9836814
# 100     15      0.9836973
# 100     25      0.98368424
# 100     35      0.9836349
# 100     50      0.98351973
# 100     70      0.9832926
# 100     100     0.9828251
# Best tornado: 35        35      0.9843225

# blur_radius_f2  blur_radius_f38 AUC_wind
# 0       0       0.9877189
# 0       15      0.9877767
# 0       25      0.9877994
# 0       35      0.9878116
# 0       50      0.9877904
# 0       70      0.9876978
# 0       100     0.98743767
# 15      0       0.9877884
# 15      15      0.9878209
# 15      25      0.98783594
# 15      35      0.98784065
# 15      50      0.98781234
# 15      70      0.98771536
# 15      100     0.9874545
# 25      0       0.98781097
# 25      15      0.98783636
# 25      25      0.9878423
# 25      35      0.9878376
# 25      50      0.98779935
# 25      70      0.9876946
# 25      100     0.98742825
# 35      0       0.98780584
# 35      15      0.987825
# 35      25      0.9878221
# 35      35      0.9878022
# 35      50      0.9877457
# 35      70      0.9876247
# 35      100     0.98734385
# 50      0       0.98772836
# 50      15      0.9877417
# 50      25      0.98772997
# 50      35      0.9876927
# 50      50      0.9876065
# 50      70      0.9874537
# 50      100     0.9871385
# 70      0       0.9875251
# 70      15      0.98753464
# 70      25      0.9875154
# 70      35      0.987462
# 70      50      0.9873438
# 70      70      0.98714375
# 70      100     0.98676693
# 100     0       0.98704904
# 100     15      0.987057
# 100     25      0.9870308
# 100     35      0.98696053
# 100     50      0.98680484
# 100     70      0.9865399
# 100     100     0.9860516
# Best wind: 25   25      0.9878423

# blur_radius_f2  blur_radius_f38 AUC_hail
# 0       0       0.98941755
# 0       15      0.9894806
# 0       25      0.9895114
# 0       35      0.989533
# 0       50      0.9895336
# 0       70      0.9894821
# 0       100     0.9892977
# 15      0       0.9894754
# 15      15      0.98951524
# 15      25      0.9895393
# 15      35      0.9895551
# 15      50      0.9895505
# 15      70      0.98949623
# 15      100     0.98931295
# 25      0       0.98949367
# 25      15      0.98952705
# 25      25      0.9895428
# 25      35      0.9895509
# 25      50      0.98953813
# 25      70      0.98947775
# 25      100     0.989291
# 35      0       0.9894823
# 35      15      0.9895102
# 35      25      0.98951846
# 35      35      0.9895146
# 35      50      0.9894876
# 35      70      0.98941463
# 35      100     0.98921776
# 50      0       0.9894085
# 50      15      0.98943144
# 50      25      0.98943186
# 50      35      0.9894138
# 50      50      0.9893616
# 50      70      0.98926187
# 50      100     0.98903835
# 70      0       0.9892258
# 70      15      0.989245
# 70      25      0.98923844
# 70      35      0.98920625
# 70      50      0.98912555
# 70      70      0.98898333
# 70      100     0.9887081
# 100     0       0.98881626
# 100     15      0.9888336
# 100     25      0.9888201
# 100     35      0.9887721
# 100     50      0.9886558
# 100     70      0.98845077
# 100     100     0.9880733
# Best hail: 15   35      0.9895551

# blur_radius_f2  blur_radius_f38 AUC_sig_tornado
# 0       0       0.976049
# 0       15      0.9760343
# 0       25      0.9760699
# 0       35      0.97614837
# 0       50      0.9762031
# 0       70      0.97612983
# 0       100     0.9758608
# 15      0       0.976134
# 15      15      0.9761076
# 15      25      0.97613305
# 15      35      0.9761902
# 15      50      0.9762218
# 15      70      0.9761378
# 15      100     0.9758654
# 25      0       0.97623456
# 25      15      0.97620344
# 25      25      0.97622156
# 25      35      0.9762633
# 25      50      0.9762777
# 25      70      0.97618425
# 25      100     0.97590685
# 35      0       0.9763801
# 35      15      0.9763446
# 35      25      0.97635686
# 35      35      0.9763829
# 35      50      0.97637266
# 35      70      0.9762623
# 35      100     0.97597367
# 50      0       0.9764661
# 50      15      0.97642845
# 50      25      0.9764388
# 50      35      0.9764589
# 50      50      0.9764341
# 50      70      0.9763104
# 50      100     0.97601
# 70      0       0.9764149
# 70      15      0.97637653
# 70      25      0.9763864
# 70      35      0.976405
# 70      50      0.9763784
# 70      70      0.9762543
# 70      100     0.97595286
# 100     0       0.9761978
# 100     15      0.9761598
# 100     25      0.97617024
# 100     35      0.97618884
# 100     50      0.976162
# 100     70      0.9760387
# 100     100     0.9757399
# Best sig_tornado: 50    0       0.9764661

# blur_radius_f2  blur_radius_f38 AUC_sig_wind
# 0       0       0.9893709
# 0       15      0.9894301
# 0       25      0.98944986
# 0       35      0.98945606
# 0       50      0.9894255
# 0       70      0.9892954
# 0       100     0.98892605
# 15      0       0.9894416
# 15      15      0.98947424
# 15      25      0.98948723
# 15      35      0.98948735
# 15      50      0.9894513
# 15      70      0.9893183
# 15      100     0.9889494
# 25      0       0.98946226
# 25      15      0.98948854
# 25      25      0.9894937
# 25      35      0.9894864
# 25      50      0.98944306
# 25      70      0.98930514
# 25      100     0.9889331
# 35      0       0.9894587
# 35      15      0.9894797
# 35      25      0.98947805
# 35      35      0.9894589
# 35      50      0.9894023
# 35      70      0.98925436
# 35      100     0.9888739
# 50      0       0.98938864
# 50      15      0.9894059
# 50      25      0.9893983
# 50      35      0.9893675
# 50      50      0.9892911
# 50      70      0.9891247
# 50      100     0.9887262
# 70      0       0.9891639
# 70      15      0.98917955
# 70      25      0.98916835
# 70      35      0.9891294
# 70      50      0.9890363
# 70      70      0.9888472
# 70      100     0.9884202
# 100     0       0.9885827
# 100     15      0.9885992
# 100     25      0.98858505
# 100     35      0.98853827
# 100     50      0.9884277
# 100     70      0.98820937
# 100     100     0.9877318
# Best sig_wind: 25       25      0.9894937

# blur_radius_f2  blur_radius_f38 AUC_sig_hail
# 0       0       0.99497664
# 0       15      0.99502707
# 0       25      0.9950619
# 0       35      0.9951043
# 0       50      0.99513257
# 0       70      0.9950967
# 0       100     0.9949345
# 15      0       0.9950291
# 15      15      0.9950623
# 15      25      0.9950905
# 15      35      0.99512637
# 15      50      0.99514925
# 15      70      0.9951113
# 15      100     0.99495095
# 25      0       0.9950597
# 25      15      0.99508697
# 25      25      0.99510765
# 25      35      0.99513555
# 25      50      0.99515074
# 25      70      0.99510837
# 25      100     0.99494666
# 35      0       0.9950828
# 35      15      0.9951046
# 35      25      0.995118
# 35      35      0.9951337
# 35      50      0.9951356
# 35      70      0.99508446
# 35      100     0.99491733
# 50      0       0.9950665
# 50      15      0.9950838
# 50      25      0.9950905
# 50      35      0.9950936
# 50      50      0.9950759
# 50      70      0.99500793
# 50      100     0.99482614
# 70      0       0.99495345
# 70      15      0.99496835
# 70      25      0.9949706
# 70      35      0.9949646
# 70      50      0.9949299
# 70      70      0.9948404
# 70      100     0.99463284
# 100     0       0.9946548
# 100     15      0.9946698
# 100     25      0.9946691
# 100     35      0.9946553
# 100     50      0.9946029
# 100     70      0.994485
# 100     100     0.99423045
# Best sig_hail: 25       50      0.99515074



println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f38\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
# tornado     15                  15                   0.03856513
# wind        25                  25                   0.115486614
# hail        25                  15                   0.0744874
# hail        15                  25                   0.0744809 # 0.01% difference, let's use it for justice
# sig_tornado 25                  35                   0.034162313
# sig_wind    15                  35                   0.016272869
# sig_hail    15                  25                   0.015544395


# event_name      best_blur_radius_f2     best_blur_radius_f38    AUC
# tornado         35      35      0.9843225
# wind            25      25      0.9878423
# hail            15      35      0.9895551
# sig_tornado     50      0       0.9764661
# sig_wind        25      25      0.9894937
# sig_hail        25      50      0.99515074

# if using regular tornado predictor for sig_tor, then:
# Best sig_tornado: 25    15      0.98158324


# Now go back to HREFPrediction.jl and put those numbers in


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
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_blurred(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
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
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_blurred, X, Ys, weights)

# EXPECTED:
# event_name  AU_PR
# tornado     0.03856513
# wind        0.115486614
# hail        0.0744809 # 0.01% difference, let's use it for justice
# sig_tornado 0.034162313
# sig_wind    0.016272869
# sig_hail    0.015544395

# ACTUAL:
# tornado (68134.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.03856512649512859
# wind (562866.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.11548661874975621
# hail (246689.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.07447986343604622
# sig_tornado (9182.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.03416580036765368
# sig_wind (57701.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016274761350642975
# sig_hail (30597.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.015544502650765795

# Yay!