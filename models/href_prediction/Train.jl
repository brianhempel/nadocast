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

# tornado (68134.0)    feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.03809341923679216
# tornado (68134.0)    feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.03858930727538041 ***best tor***
# tornado (68134.0)    feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.03849788709844083
# tornado (68134.0)    feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.037841004621810365
# tornado (68134.0)    feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.035916983157636764
# tornado (68134.0)    feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.032578160798125616
# tornado (68134.0)    feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.027195448840758918
# tornado (68134.0)    feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.009395700931404427
# tornado (68134.0)    feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.009430582678922323
# tornado (68134.0)    feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.009372667178212118
# tornado (68134.0)    feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.009184660747417448
# tornado (68134.0)    feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.008788058691009711
# tornado (68134.0)    feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.008013845601424721
# tornado (68134.0)    feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.006775424952470916
# tornado (68134.0)    feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005491191319329745
# tornado (68134.0)    feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.005544830649903299
# tornado (68134.0)    feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.005554961775140985
# tornado (68134.0)    feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.005514144197587153
# tornado (68134.0)    feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005352276664250288
# tornado (68134.0)    feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.004999772642223141
# tornado (68134.0)    feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004312186860941715
# tornado (68134.0)    feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.029975491734984813
# tornado (68134.0)    feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.030437580246747913
# tornado (68134.0)    feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.030378770669097867
# tornado (68134.0)    feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.029919143756071843
# tornado (68134.0)    feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.02874023385401975
# tornado (68134.0)    feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.02653325820253399
# tornado (68134.0)    feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02279371568483029
# tornado (68134.0)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.008015426182622156
# tornado (68134.0)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.008043342655526592
# tornado (68134.0)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.008000340971535186
# tornado (68134.0)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.007857179445884014
# tornado (68134.0)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.007557328177332453
# tornado (68134.0)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0070119495235905355
# tornado (68134.0)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.006103273184647692
# tornado (68134.0)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004218347687955302
# tornado (68134.0)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.004260599913850644
# tornado (68134.0)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.004240587809628718
# tornado (68134.0)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0041587762417662445
# tornado (68134.0)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.004015641871789456
# tornado (68134.0)    feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.003762491633090937
# tornado (68134.0)    feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0033225369061155303
# tornado (68134.0)    feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 9.023639634213338e-5
# wind (562866.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.04951877712146706
# wind (562866.0)      feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.04986577539922757
# wind (562866.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.0497909675884383
# wind (562866.0)      feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.04930766219301333
# wind (562866.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.04801431329613518
# wind (562866.0)      feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.04531221042064368
# wind (562866.0)      feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.040197105559454153
# wind (562866.0)      feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.11425857172613606
# wind (562866.0)      feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.11550551929169148
# wind (562866.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.11570608229578414 ***best wind***
# wind (562866.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.11504890954246082
# wind (562866.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.11261633942775649
# wind (562866.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.10689794250957872
# wind (562866.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.09627795425148059
# wind (562866.0)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.029186549460025846
# wind (562866.0)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.02912924606127656
# wind (562866.0)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.028885081027534752
# wind (562866.0)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.02833634091533701
# wind (562866.0)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.027200237258412644
# wind (562866.0)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.025403159779908584
# wind (562866.0)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02241454706790836
# wind (562866.0)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.03963689872998119
# wind (562866.0)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.039875357122756694
# wind (562866.0)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.03978328545735312
# wind (562866.0)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.03934036946102498
# wind (562866.0)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03822971996958463
# wind (562866.0)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.03594286332240919
# wind (562866.0)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03180594821964994
# wind (562866.0)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07181419414894027
# wind (562866.0)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.07234172807389372
# wind (562866.0)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.07227699119027357
# wind (562866.0)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.07168056920137697
# wind (562866.0)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.07011647348923615
# wind (562866.0)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.066837985357281
# wind (562866.0)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.060383713434271506
# wind (562866.0)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.020194223363102424
# wind (562866.0)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.020137788229596103
# wind (562866.0)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.019964386232662897
# wind (562866.0)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.019611400886410786
# wind (562866.0)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01890605806747935
# wind (562866.0)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.017801867671298703
# wind (562866.0)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01594217507198899
# wind (562866.0)      feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.0007303467435103636
# hail (246689.0)      feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.016366443637846687
# hail (246689.0)      feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.016363472003528307
# hail (246689.0)      feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.016254229963658395
# hail (246689.0)      feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015980176018150074
# hail (246689.0)      feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.01537246577318613
# hail (246689.0)      feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.014364716931832791
# hail (246689.0)      feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.012628569564300474
# hail (246689.0)      feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01751366697974395
# hail (246689.0)      feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.017529754534793787
# hail (246689.0)      feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.017426824086990506
# hail (246689.0)      feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.017171745270266166
# hail (246689.0)      feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01657035396381233
# hail (246689.0)      feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015608939391916182
# hail (246689.0)      feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013999127338429009
# hail (246689.0)      feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07318164087797681
# hail (246689.0)      feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.07421523124137194 ***best hail***
# hail (246689.0)      feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.074210069125944
# hail (246689.0)      feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.07324084265752918
# hail (246689.0)      feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.07026437660731606
# hail (246689.0)      feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.06454941343296884
# hail (246689.0)      feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.05430872665070556
# hail (246689.0)      feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.012055260716347486
# hail (246689.0)      feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.012093386152189026
# hail (246689.0)      feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.012054515095979938
# hail (246689.0)      feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.011922344232779247
# hail (246689.0)      feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01161260575790171
# hail (246689.0)      feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.011023223569139397
# hail (246689.0)      feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.009912012344971266
# hail (246689.0)      feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.020199316312547556
# hail (246689.0)      feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.020245456797655968
# hail (246689.0)      feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.02016611759077021
# hail (246689.0)      feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.019937283540232227
# hail (246689.0)      feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.019363277498103012
# hail (246689.0)      feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.018352379479905506
# hail (246689.0)      feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01644525816139401
# hail (246689.0)      feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.052777219134225686
# hail (246689.0)      feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.053263537532633375
# hail (246689.0)      feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.05307112167403091
# hail (246689.0)      feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.05221884518306501
# hail (246689.0)      feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.049923962367745274
# hail (246689.0)      feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.045900650991045594
# hail (246689.0)      feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.03880359409494361
# hail (246689.0)      feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 0.00031771735739933725
# sig_tornado (9182.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.019408982194337104
# sig_tornado (9182.0) feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.020339854241600046
# sig_tornado (9182.0) feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.02075142147683586
# sig_tornado (9182.0) feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.021028647463596922
# sig_tornado (9182.0) feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.020804835080558602
# sig_tornado (9182.0) feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.019910501580679203
# sig_tornado (9182.0) feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.017384060023300334
# sig_tornado (9182.0) feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0027652332208500468
# sig_tornado (9182.0) feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.0027798020953503787
# sig_tornado (9182.0) feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.002765715711404773
# sig_tornado (9182.0) feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0027063844800939846
# sig_tornado (9182.0) feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0026095094040228506
# sig_tornado (9182.0) feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0023613610570926845
# sig_tornado (9182.0) feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.001975697067069967
# sig_tornado (9182.0) feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0009807275584009475
# sig_tornado (9182.0) feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0009898340550745723
# sig_tornado (9182.0) feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.0009938698471880973
# sig_tornado (9182.0) feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0009931525421620934
# sig_tornado (9182.0) feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0009824533000562393
# sig_tornado (9182.0) feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0009388192550265813
# sig_tornado (9182.0) feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0008179714210316302
# sig_tornado (9182.0) feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.032372271249524916
# sig_tornado (9182.0) feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.03352600321260325
# sig_tornado (9182.0) feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.03373667786251336 ***best sigtor***
# sig_tornado (9182.0) feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.033368880793342
# sig_tornado (9182.0) feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.03180527072806252
# sig_tornado (9182.0) feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.028561029753815158
# sig_tornado (9182.0) feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.02316841523563225
# sig_tornado (9182.0) feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.002075768534447326
# sig_tornado (9182.0) feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0020791739130354247
# sig_tornado (9182.0) feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.0020605750689877736
# sig_tornado (9182.0) feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0020123398950101204
# sig_tornado (9182.0) feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0019312824565669237
# sig_tornado (9182.0) feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0017790987798680415
# sig_tornado (9182.0) feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.001538221026817875
# sig_tornado (9182.0) feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0007297079849924315
# sig_tornado (9182.0) feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0007284317971438842
# sig_tornado (9182.0) feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.000723788044158793
# sig_tornado (9182.0) feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0007139218327491783
# sig_tornado (9182.0) feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0006942640213690924
# sig_tornado (9182.0) feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0006605474180031003
# sig_tornado (9182.0) feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0006008873319494659
# sig_tornado (9182.0) feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 1.2735864479380344e-5
# sig_wind (57701.0)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.006094259814489804
# sig_wind (57701.0)   feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.006123523078937593
# sig_wind (57701.0)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.006096825173807593
# sig_wind (57701.0)   feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00600962860395727
# sig_wind (57701.0)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.0057970860008838
# sig_wind (57701.0)   feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.005406145430564985
# sig_wind (57701.0)   feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.004707681932102453
# sig_wind (57701.0)   feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.011083927529895232
# sig_wind (57701.0)   feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.011129491434742298
# sig_wind (57701.0)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.011074623121302772
# sig_wind (57701.0)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.010926747431581067
# sig_wind (57701.0)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.010589837733394772
# sig_wind (57701.0)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.01000279620816406
# sig_wind (57701.0)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.008974209387622728
# sig_wind (57701.0)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.00500779018633884
# sig_wind (57701.0)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.005036027016565651
# sig_wind (57701.0)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.005024064047501076
# sig_wind (57701.0)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0049681960224662826
# sig_wind (57701.0)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.004813197121675836
# sig_wind (57701.0)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.004510175833185109
# sig_wind (57701.0)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.003959203649605776
# sig_wind (57701.0)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.005470766362163563
# sig_wind (57701.0)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0054888260966505405
# sig_wind (57701.0)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.005468978791141284
# sig_wind (57701.0)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.00539998386036541
# sig_wind (57701.0)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.005240256322716095
# sig_wind (57701.0)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.004923300914914417
# sig_wind (57701.0)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.004351323413906337
# sig_wind (57701.0)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015982477433103947
# sig_wind (57701.0)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.016188654524609786
# sig_wind (57701.0)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.01621067405530351 ***best sigwind***
# sig_wind (57701.0)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.01611063170708923
# sig_wind (57701.0)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.01579234069946467
# sig_wind (57701.0)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.015124744299665118
# sig_wind (57701.0)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.013735235017930055
# sig_wind (57701.0)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.004135178733046252
# sig_wind (57701.0)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.004137935092992308
# sig_wind (57701.0)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.0041147087708857735
# sig_wind (57701.0)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.004059891206968263
# sig_wind (57701.0)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003935937315956243
# sig_wind (57701.0)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.003716473079527257
# sig_wind (57701.0)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0033191743870137637
# sig_wind (57701.0)   feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 7.421561014472997e-5
# sig_hail (30597.0)   feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.0027281072285540017
# sig_hail (30597.0)   feature 2 TORPROB:calculated:hour   fcst:calculated_prob:15mi  mean AU-PR-curve: 0.00272940075653071
# sig_hail (30597.0)   feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi    mean AU-PR-curve: 0.0027131088660382924
# sig_hail (30597.0)   feature 4 TORPROB:calculated:hour   fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0026656223228365573
# sig_hail (30597.0)   feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi    mean AU-PR-curve: 0.0025596238797890914
# sig_hail (30597.0)   feature 6 TORPROB:calculated:hour   fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0023698549557959617
# sig_hail (30597.0)   feature 7 TORPROB:calculated:hour   fcst:calculated_prob:100mi mean AU-PR-curve: 0.0020457844169989225
# sig_hail (30597.0)   feature 8  WINDPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0021993606677390737
# sig_hail (30597.0)   feature 9  WINDPROB:calculated:hour  fcst:calculated_prob:15mi mean AU-PR-curve: 0.002195652296800663
# sig_hail (30597.0)   feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.0021768396125813384
# sig_hail (30597.0)   feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0021379153149256517
# sig_hail (30597.0)   feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0020559799953049017
# sig_hail (30597.0)   feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0019287135532202759
# sig_hail (30597.0)   feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0017282565401535706
# sig_hail (30597.0)   feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.01516031831443958
# sig_hail (30597.0)   feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.01531364481444899
# sig_hail (30597.0)   feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.015234390253753794
# sig_hail (30597.0)   feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.014895086627373507
# sig_hail (30597.0)   feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014209915840622115
# sig_hail (30597.0)   feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.012885311313063268
# sig_hail (30597.0)   feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.010733071657578306
# sig_hail (30597.0)   feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0020384095822184
# sig_hail (30597.0)   feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0020420512396392013
# sig_hail (30597.0)   feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.0020297474889828426
# sig_hail (30597.0)   feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.001995538558719344
# sig_hail (30597.0)   feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.0019241915792183094
# sig_hail (30597.0)   feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.00180164100314552
# sig_hail (30597.0)   feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0015935803623303374
# sig_hail (30597.0)   feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.0032930838756027353
# sig_hail (30597.0)   feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.0033111390163943757
# sig_hail (30597.0)   feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.003301501020561235
# sig_hail (30597.0)   feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.0032682356286284765
# sig_hail (30597.0)   feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.003180087716377742
# sig_hail (30597.0)   feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0030328626050877255
# sig_hail (30597.0)   feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0027043900882929755
# sig_hail (30597.0)   feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.015385460547444888
# sig_hail (30597.0)   feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi  mean AU-PR-curve: 0.01557374595014855 ***best sighail***
# sig_hail (30597.0)   feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi  mean AU-PR-curve: 0.015514858453339856
# sig_hail (30597.0)   feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi  mean AU-PR-curve: 0.015189997280659078
# sig_hail (30597.0)   feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi  mean AU-PR-curve: 0.014414316345688587
# sig_hail (30597.0)   feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi  mean AU-PR-curve: 0.0129907158916543
# sig_hail (30597.0)   feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.01069562879026595
# sig_hail (30597.0)   feature 43 forecast_hour:calculated:hour fcst::                     AU-PR-curve: 3.993461625611225e-5



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
# 0       0       0.03809342
# 0       15      0.038320705
# 0       25      0.038313318
# 0       35      0.03815591
# 0       50      0.03770487
# 0       70      0.03701286
# 0       100     0.03591176
# 15      0       0.03850322
# 15      15      0.038589306
# 15      25      0.03856
# 15      35      0.038383294
# 15      50      0.03791692
# 15      70      0.037213624
# 15      100     0.03610798
# 25      0       0.038503136
# 25      15      0.038566124
# 25      25      0.038497888
# 25      35      0.038285214
# 25      50      0.03777889
# 25      70      0.037036855
# 25      100     0.03590152
# 35      0       0.038191244
# 35      15      0.03823522
# 35      25      0.0381263
# 35      35      0.037841003
# 35      50      0.037245423
# 35      70      0.036412958
# 35      100     0.03519801
# 50      0       0.03720261
# 50      15      0.03722399
# 50      25      0.037064433
# 50      35      0.036677442
# 50      50      0.03591698
# 50      70      0.034904987
# 50      100     0.033516776
# 70      0       0.035612933
# 70      15      0.03561966
# 70      25      0.035404064
# 70      35      0.034890126
# 70      50      0.03390078
# 70      70      0.03257816
# 70      100     0.030886205
# 100     0       0.03325529
# 100     15      0.033287812
# 100     25      0.03303947
# 100     35      0.03241653
# 100     50      0.031177698
# 100     70      0.02945765
# 100     100     0.027195448
# Best tornado: 15        15      0.038589306

# blur_radius_f2  blur_radius_f38 AU_PR_wind
# 0       0       0.11425857
# 0       15      0.1148392
# 0       25      0.11502065
# 0       35      0.11503661
# 0       50      0.11478107
# 0       70      0.11396844
# 0       100     0.112427205
# 15      0       0.115185045
# 15      15      0.11550552
# 15      25      0.11564684
# 15      35      0.11564439
# 15      50      0.11537782
# 15      70      0.11456819
# 15      100     0.113056764
# 25      0       0.115367495
# 25      15      0.115639046
# 25      25      0.115706086
# 25      35      0.115644746
# 25      50      0.1153247
# 25      70      0.1144664
# 25      100     0.11292721
# 35      0       0.115006104
# 35      15      0.11524059
# 35      25      0.11523594
# 35      35      0.11504892
# 35      50      0.1146022
# 35      70      0.113621235
# 35      100     0.11197347
# 50      0       0.11361388
# 50      15      0.11380467
# 50      25      0.11370797
# 50      35      0.11334789
# 50      50      0.11261634
# 50      70      0.11136181
# 50      100     0.10942233
# 70      0       0.11053029
# 70      15      0.110678054
# 70      25      0.11047181
# 70      35      0.10987993
# 70      50      0.10873095
# 70      70      0.10689795
# 70      100     0.1043184
# 100     0       0.105620965
# 100     15      0.105760865
# 100     25      0.10547316
# 100     35      0.104660586
# 100     50      0.10300152
# 100     70      0.10023475
# 100     100     0.09627795
# Best wind: 25   25      0.115706086

# blur_radius_f2  blur_radius_f38 AU_PR_hail
# 0       0       0.07318163
# 0       15      0.0737054
# 0       25      0.07379321
# 0       35      0.07364843
# 0       50      0.073127076
# 0       70      0.07222181
# 0       100     0.07075998
# 15      0       0.07392695
# 15      15      0.07421523
# 15      25      0.07425974
# 15      35      0.07408572
# 15      50      0.073557705
# 15      70      0.07266323
# 15      100     0.07124361
# 25      0       0.074002996
# 25      15      0.074242875
# 25      25      0.07421008
# 25      35      0.07397271
# 25      50      0.07338229
# 25      70      0.07243879
# 25      100     0.071004264
# 35      0       0.07351991
# 35      15      0.07372961
# 35      25      0.07361579
# 35      35      0.07324083
# 35      50      0.0724898
# 35      70      0.07141548
# 35      100     0.0698963
# 50      0       0.071930416
# 50      15      0.07212494
# 50      25      0.07191672
# 50      35      0.071345136
# 50      50      0.07026436
# 50      70      0.06884639
# 50      100     0.06701053
# 70      0       0.068984434
# 70      15      0.06917874
# 70      25      0.06887235
# 70      35      0.068080544
# 70      50      0.0665504
# 70      70      0.06454941
# 70      100     0.06204971
# 100     0       0.06426858
# 100     15      0.06445081
# 100     25      0.06405096
# 100     35      0.06302279
# 100     50      0.060948372
# 100     70      0.058095183
# 100     100     0.054308724
# Best hail: 15   25      0.07425974

# blur_radius_f2  blur_radius_f38 AU_PR_sig_tornado
# 0       0       0.03237228
# 0       15      0.03292119
# 0       25      0.033107318
# 0       35      0.033235956
# 0       50      0.033199463
# 0       70      0.032973863
# 0       100     0.03234522
# 15      0       0.03320229
# 15      15      0.033526
# 15      25      0.03368453
# 15      35      0.03380525
# 15      50      0.03377654
# 15      70      0.033599578
# 15      100     0.033050727
# 25      0       0.033346925
# 25      15      0.033632178
# 25      25      0.033736676
# 25      35      0.03381651
# 25      50      0.03374406
# 25      70      0.033543672
# 25      100     0.03301898
# 35      0       0.033047143
# 35      15      0.033310197
# 35      25      0.03336976
# 35      35      0.033368878
# 35      50      0.033209093
# 35      70      0.032922104
# 35      100     0.032363143
# 50      0       0.03202782
# 50      15      0.03225422
# 50      25      0.032249294
# 50      35      0.032124363
# 50      50      0.031805273
# 50      70      0.031339016
# 50      100     0.030610599
# 70      0       0.030093733
# 70      15      0.030264659
# 70      25      0.03017111
# 70      35      0.029892702
# 70      50      0.02932476
# 70      70      0.028561028
# 70      100     0.02747392
# 100     0       0.02718754
# 100     15      0.027370635
# 100     25      0.027224505
# 100     35      0.026794223
# 100     50      0.025927367
# 100     70      0.024721023
# 100     100     0.023168415
# Best sig_tornado: 25    35      0.03381651

# blur_radius_f2  blur_radius_f38 AU_PR_sig_wind
# 0       0       0.015982477
# 0       15      0.016093707
# 0       25      0.016134672
# 0       35      0.016156817
# 0       50      0.016158082
# 0       70      0.016126579
# 0       100     0.01603801
# 15      0       0.016129857
# 15      15      0.016188655
# 15      25      0.016220633
# 15      35      0.016237844
# 15      50      0.016237346
# 15      70      0.016207851
# 15      100     0.016123284
# 25      0       0.016143903
# 25      15      0.016192678
# 25      25      0.016210673
# 25      35      0.016216163
# 25      50      0.016204923
# 25      70      0.016165476
# 25      100     0.016076013
# 35      0       0.016075268
# 35      15      0.01611751
# 35      25      0.01612431
# 35      35      0.016110633
# 35      50      0.016077263
# 35      70      0.016017968
# 35      100     0.015909143
# 50      0       0.015877284
# 50      15      0.015913932
# 50      25      0.01590801
# 50      35      0.01586983
# 50      50      0.01579234
# 50      70      0.01568642
# 50      100     0.015526874
# 70      0       0.015497869
# 70      15      0.01553034
# 70      25      0.015511257
# 70      35      0.015444803
# 70      50      0.015312641
# 70      70      0.015124745
# 70      100     0.014860065
# 100     0       0.014813265
# 100     15      0.014844882
# 100     25      0.014810547
# 100     35      0.014709301
# 100     50      0.014504516
# 100     70      0.014197887
# 100     100     0.013735235
# Best sig_wind: 15       35      0.016237844

# blur_radius_f2  blur_radius_f38 AU_PR_sig_hail
# 0       0       0.015385461
# 0       15      0.015506115
# 0       25      0.0155356
# 0       35      0.0155116925
# 0       50      0.015421791
# 0       70      0.015237438
# 0       100     0.015011334
# 15      0       0.015510789
# 15      15      0.015573746
# 15      25      0.015587974
# 15      35      0.01555286
# 15      50      0.015458695
# 15      70      0.015273759
# 15      100     0.015061713
# 25      0       0.015469778
# 25      15      0.015519621
# 25      25      0.015514858
# 25      35      0.015461082
# 25      50      0.01534767
# 25      70      0.015147802
# 25      100     0.01492863
# 35      0       0.015266805
# 35      15      0.015304009
# 35      25      0.015278034
# 35      35      0.015189998
# 35      50      0.015034933
# 35      70      0.014793917
# 35      100     0.014541294
# 50      0       0.014813177
# 50      15      0.014836785
# 50      25      0.014785355
# 50      35      0.014648365
# 50      50      0.014414316
# 50      70      0.014090983
# 50      100     0.013747987
# 70      0       0.014046268
# 70      15      0.0140640475
# 70      25      0.013987178
# 70      35      0.013792071
# 70      50      0.013453575
# 70      70      0.012990715
# 70      100     0.012501941
# 100     0       0.012933281
# 100     15      0.012950038
# 100     25      0.012857055
# 100     35      0.012609563
# 100     50      0.012142011
# 100     70      0.011493708
# 100     100     0.010695629
# Best sig_hail: 15       25      0.015587974





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
# tornado     15                  15                   0.038589306
# wind        25                  25                   0.115706086
# hail        15                  25                   0.07425974
# sig_tornado 25                  35                   0.03381651
# sig_wind    15                  35                   0.016237844
# sig_hail    15                  25                   0.015587974


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