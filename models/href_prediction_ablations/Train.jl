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
size(X) #
length(weights) #

sum(Ys["tornado"]) #
sum(weights) #

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
# 0       0       0.03809342
# 0       15      0.03833245
# 0       25      0.038320392
# 0       35      0.038138837
# 0       50      0.03762647
# 0       70      0.03683251
# 0       100     0.035555083
# 15      0       0.038487047
# 15      15      0.038589306
# 15      25      0.038554613
# 15      35      0.038354363
# 15      50      0.037827887
# 15      70      0.03702315
# 15      100     0.03574093
# 25      0       0.038488097
# 25      15      0.038570095
# 25      25      0.038497888
# 25      35      0.038262453
# 25      50      0.037694894
# 25      70      0.03685101
# 25      100     0.03553859
# 35      0       0.038193204
# 35      15      0.038259614
# 35      25      0.03814807
# 35      35      0.037841003
# 35      50      0.037183996
# 35      70      0.036247954
# 35      100     0.034855004
# 50      0       0.03725682
# 50      15      0.03730489
# 50      25      0.037145507
# 50      35      0.036738563
# 50      50      0.035916984
# 50      70      0.034799792
# 50      100     0.033230897
# 70      0       0.035751272
# 70      15      0.035792924
# 70      25      0.03558344
# 70      35      0.03505568
# 70      50      0.034006774
# 70      70      0.03257816
# 70      100     0.030731909
# 100     0       0.03350377
# 100     15      0.033584286
# 100     25      0.03335196
# 100     35      0.032732237
# 100     50      0.0314521
# 100     70      0.02963512
# 100     100     0.027195448
# Best tornado: 15        15      0.038589306

# blur_radius_f2  blur_radius_f35 AU_PR_wind
# 0       0       0.11425857
# 0       15      0.11487553
# 0       25      0.11506089
# 0       35      0.11505345
# 0       50      0.114719905
# 0       70      0.1137273
# 0       100     0.11183025
# 15      0       0.1151375
# 15      15      0.11550552
# 15      25      0.11565481
# 15      35      0.11563217
# 15      50      0.1152908
# 15      70      0.11430385
# 15      100     0.11243332
# 25      0       0.11530363
# 25      15      0.115627155
# 25      25      0.11570608
# 25      35      0.11562999
# 25      50      0.11523922
# 25      70      0.11420522
# 25      100     0.11230791
# 35      0       0.11493821
# 35      15      0.115230374
# 35      25      0.11524302
# 35      35      0.11504892
# 35      50      0.11454087
# 35      70      0.11338941
# 35      100     0.111385085
# 50      0       0.11356733
# 50      15      0.11382539
# 50      25      0.11375178
# 50      35      0.113393635
# 50      50      0.11261634
# 50      70      0.11120868
# 50      100     0.10892003
# 70      0       0.11056573
# 70      15      0.11079354
# 70      25      0.110620454
# 70      35      0.1100415
# 70      50      0.10886046
# 70      70      0.106897935
# 70      100     0.103996836
# 100     0       0.10577907
# 100     15      0.10600986
# 100     25      0.10577274
# 100     35      0.10500525
# 100     50      0.103359774
# 100     70      0.10050264
# 100     100     0.09627795
# Best wind: 25   25      0.11570608

# blur_radius_f2  blur_radius_f35 AU_PR_hail
# 0       0       0.07318162
# 0       15      0.07373512
# 0       25      0.07381648
# 0       35      0.073629946
# 0       50      0.07299951
# 0       70      0.07190046
# 0       100     0.07010953
# 15      0       0.07388143
# 15      15      0.074215226
# 15      25      0.07425418
# 15      35      0.074047066
# 15      50      0.07341247
# 15      70      0.072326355
# 15      100     0.070574865
# 25      0       0.073946565
# 25      15      0.07424145
# 25      25      0.07421007
# 25      35      0.073940955
# 25      50      0.07324884
# 25      70      0.07211829
# 25      100     0.070350885
# 35      0       0.07346742
# 35      15      0.07374107
# 35      25      0.07363912
# 35      35      0.07324084
# 35      50      0.07239695
# 35      70      0.071139425
# 35      100     0.069290064
# 50      0       0.071907304
# 50      15      0.07218139
# 50      25      0.0719952
# 50      35      0.07141567
# 50      50      0.070264354
# 50      70      0.068678185
# 50      100     0.06651561
# 70      0       0.06902741
# 70      15      0.069311745
# 70      25      0.06904332
# 70      35      0.06826525
# 70      50      0.06668998
# 70      70      0.06454942
# 70      100     0.061738383
# 100     0       0.06444595
# 100     15      0.064716436
# 100     25      0.0643686
# 100     35      0.063372806
# 100     50      0.061288495
# 100     70      0.058345076
# 100     100     0.054308724
# Best hail: 15   25      0.07425418

# blur_radius_f2  blur_radius_f35 AU_PR_sig_tornado
# 0       0       0.03237227
# 0       15      0.032958195
# 0       25      0.033152383
# 0       35      0.0332831
# 0       50      0.033218533
# 0       70      0.032929715
# 0       100     0.032157406
# 15      0       0.033156913
# 15      15      0.033526
# 15      25      0.03369552
# 15      35      0.033816796
# 15      50      0.03376468
# 15      70      0.03352364
# 15      100     0.032838657
# 25      0       0.03328689
# 25      15      0.033618513
# 25      25      0.033736676
# 25      35      0.03381999
# 25      50      0.033720843
# 25      70      0.033457663
# 25      100     0.03280141
# 35      0       0.032975577
# 35      15      0.033288684
# 35      25      0.033365272
# 35      35      0.033368878
# 35      50      0.033184376
# 35      70      0.032836597
# 35      100     0.032150667
# 50      0       0.03198282
# 50      15      0.03225435
# 50      25      0.032262743
# 50      35      0.032144863
# 50      50      0.031805266
# 50      70      0.031279944
# 50      100     0.030424474
# 70      0       0.030103818
# 70      15      0.030324932
# 70      25      0.030245673
# 70      35      0.029969255
# 70      50      0.029378466
# 70      70      0.028561028
# 70      100     0.027344132
# 100     0       0.027286839
# 100     15      0.027531506
# 100     25      0.027411342
# 100     35      0.026994074
# 100     50      0.026105745
# 100     70      0.024830295
# 100     100     0.023168413
# Best sig_tornado: 25    35      0.03381999

# blur_radius_f2  blur_radius_f35 AU_PR_sig_wind
# 0       0       0.015982477
# 0       15      0.016100902
# 0       25      0.016143078
# 0       35      0.016165817
# 0       50      0.01616156
# 0       70      0.016117385
# 0       100     0.01599725
# 15      0       0.016121585
# 15      15      0.016188655
# 15      25      0.016222881
# 15      35      0.016239488
# 15      50      0.016234523
# 15      70      0.01619136
# 15      100     0.016076315
# 25      0       0.016132716
# 25      15      0.016190287
# 25      25      0.016210673
# 25      35      0.01621575
# 25      50      0.016199801
# 25      70      0.016147181
# 25      100     0.016026575
# 35      0       0.016063936
# 35      15      0.016115203
# 35      25      0.016124519
# 35      35      0.016110633
# 35      50      0.016072333
# 35      70      0.01600009
# 35      100     0.01586066
# 50      0       0.015869593
# 50      15      0.015915971
# 50      25      0.015913049
# 50      35      0.015874794
# 50      50      0.01579234
# 50      70      0.015673311
# 50      100     0.015482166
# 70      0       0.0155005
# 70      15      0.01554367
# 70      25      0.015528235
# 70      35      0.01546246
# 70      50      0.015325937
# 70      70      0.015124745
# 70      100     0.014827551
# 100     0       0.01483683
# 100     15      0.014881223
# 100     25      0.014851967
# 100     35      0.014753425
# 100     50      0.014547106
# 100     70      0.014229765
# 100     100     0.013735235
# Best sig_wind: 15       35      0.016239488

# blur_radius_f2  blur_radius_f35 AU_PR_sig_hail
# 0       0       0.015385461
# 0       15      0.01551403
# 0       25      0.015543372
# 0       35      0.01551327
# 0       50      0.015403725
# 0       70      0.015183927
# 0       100     0.014900879
# 15      0       0.015501743
# 15      15      0.0155737465
# 15      25      0.015588201
# 15      35      0.015546589
# 15      50      0.015434055
# 15      70      0.015212937
# 15      100     0.014943915
# 25      0       0.015459389
# 25      15      0.015518905
# 25      25      0.015514858
# 25      35      0.015454577
# 25      50      0.015323038
# 25      70      0.015087394
# 25      100     0.014810198
# 35      0       0.015262133
# 35      15      0.015309558
# 35      25      0.015284565
# 35      35      0.015189998
# 35      50      0.015017431
# 35      70      0.014740298
# 35      100     0.014429382
# 50      0       0.014824726
# 50      15      0.0148601355
# 50      25      0.014810139
# 50      35      0.014666012
# 50      50      0.014414316
# 50      70      0.01405533
# 50      100     0.013652732
# 70      0       0.014089459
# 70      15      0.014121531
# 70      25      0.014047797
# 70      35      0.013846563
# 70      50      0.013488993
# 70      70      0.012990716
# 70      100     0.01244347
# 100     0       0.013019429
# 100     15      0.013049809
# 100     25      0.012961261
# 100     35      0.012713867
# 100     50      0.012233336
# 100     70      0.011551532
# 100     100     0.010695629
# Best sig_hail: 15       25      0.015588201




# blur_radius_f2  blur_radius_f35 AUC_tornado
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

# blur_radius_f2  blur_radius_f35 AUC_wind
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

# blur_radius_f2  blur_radius_f35 AUC_hail
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

# blur_radius_f2  blur_radius_f35 AUC_sig_tornado
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

# blur_radius_f2  blur_radius_f35 AUC_sig_wind
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

# blur_radius_f2  blur_radius_f35 AUC_sig_hail
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


# waiting on this
println("event_name\tbest_blur_radius_f2\tbest_blur_radius_f35\tAU_PR")
for (event_name, best_blur_i_lo, best_blur_i_hi, best_au_pr) in bests
  println("$event_name\t$(blur_radii[best_blur_i_lo])\t$(blur_radii[best_blur_i_hi])\t$(Float32(best_au_pr))")
end
println()

# event_name  best_blur_radius_f2 best_blur_radius_f35 AU_PR
# tornado     15                  15                   0.038589306
# wind        25                  25                   0.11570608
# hail        15                  25                   0.07425418
# sig_tornado 25                  35                   0.03381999
# sig_wind    15                  35                   0.016239488
# sig_hail    15                  25                   0.015588201

# event_name      best_blur_radius_f2     best_blur_radius_f35    AUC
# tornado         35      35      0.9843225
# wind            25      25      0.9878423
# hail            15      35      0.9895551
# sig_tornado     50      0       0.9764661
# sig_wind        25      25      0.9894937
# sig_hail        25      50      0.99515074

# if using regular tornado predictor for sig_tor, then:
# Best sig_tornado: 25    15      0.98158324


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
# event_name  AU_PR
# tornado     0.038589306
# wind        0.11570608
# hail        0.07425418
# sig_tornado 0.03381999
# sig_wind    0.016239488
# sig_hail    0.015588201

# ACTUAL:
# tornado (68134.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.038589306
# wind (562866.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.11570608
# hail (246689.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.07425418
# sig_tornado (9182.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.03381999
# sig_wind (57701.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.016239488
# sig_hail (30597.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.015588201

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
