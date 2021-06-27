# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_with_blurs_and_forecast_hour");

length(validation_forecasts) #
size(X) #
length(y) #

println("Dividing into bins of equal positive weight...")

ŷ = X[:, 1]

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]
y       = y[sort_perm]
ŷ       = ŷ[sort_perm]
weights = weights[sort_perm]

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights)

bin_count = 10 # 40 equal logloss bins
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count

# bins = map(_ -> Int64[], 1:bin_count)
bins_Σŷ      = map(_ -> 0.0f0, 1:bin_count)
bins_Σy      = map(_ -> 0.0f0, 1:bin_count)
bins_Σweight = map(_ -> 0.0f0, 1:bin_count)
bins_max     = map(_ -> 1.0f0, 1:bin_count)

bin_i = 1
# bin_logloss = 0.0
for i in 1:length(y)
  global bin_i
  # global bin_logloss

  if ŷ[i] > bins_max[bin_i]
    bin_i += 1
    # bin_logloss = 0.0
  end

  bins_Σŷ[bin_i]      += ŷ[i] * weights[i]
  bins_Σy[bin_i]      += y[i] * weights[i]
  bins_Σweight[bin_i] += weights[i]

  # bin_logloss += logloss(y[i], ŷ[i])

  # if bin_logloss >= per_bin_logloss
  #   bins_max[bin_i] = ŷ[i]
  # end
  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("mean_y\tmean_ŷ\tΣweight\tbin_max")
for bin_i in 1:length(bins_Σy)
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end

# mean_y          mean_ŷ          Σweight         bin_max

function roc_auc(ŷ, y, weights; sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort), total_weight = sum(Float64.(weights)), positive_weight = sum(y .* Float64.(weights)))
  y       = y[sort_perm]
  ŷ       = ŷ[sort_perm]
  weights = Float64.(weights[sort_perm])

  negative_weight  = total_weight - positive_weight
  true_pos_weight  = positive_weight
  false_pos_weight = negative_weight

  # tpr = true_pos/total_pos
  # fpr = false_pos/total_neg
  # ROC is tpr vs fpr

  auc = 0.0

  last_fpr = false_pos_weight / negative_weight # = 1.0
  for i in 1:length(y)
    if y[i] > 0.5f0
      true_pos_weight -= weights[i]
    else
      false_pos_weight -= weights[i]
    end
    fpr = false_pos_weight / negative_weight
    tpr = true_pos_weight  / positive_weight
    if fpr != last_fpr
      auc += (last_fpr - fpr) * tpr
    end
    last_fpr = fpr
  end

  auc
end

roc_auc(X[:,1], y, weights) #

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


# CSI = hits / (hits + false alarms + misses)
#     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)
#     = 1 / (1/POD + 1/(1-FAR) - 1)

function csi(ŷ, y, weights; sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort), total_weight = sum(Float64.(weights)), positive_weight = sum(y .* Float64.(weights)))
  y       = y[sort_perm]
  ŷ       = ŷ[sort_perm]
  weights = Float64.(weights[sort_perm])

  negative_weight = total_weight - positive_weight

  true_pos_weight  = positive_weight
  false_pos_weight = negative_weight
  false_neg_weight = 0.0

  # CSI = hits / (hits + false alarms + misses)
  #     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)

  pods = Float64[true_pos_weight / positive_weight]
  csis = Float64[true_pos_weight / (true_pos_weight + false_pos_weight + false_neg_weight)]

  for i in 1:length(y)
    if y[i] > 0.5f0
      true_pos_weight  -= weights[i]
      false_neg_weight += weights[i]
    else
      false_pos_weight -= weights[i]
    end

    pod = true_pos_weight / positive_weight
    csi = true_pos_weight / (true_pos_weight + false_pos_weight + false_neg_weight)

    push!(pods, pod)
    push!(csis, csi)
  end

  # CSIs for PODs 0.9, 0.8, ..., 0.1
  map(collect(0.9:-0.1:0.1)) do pod_threshold
    i = findfirst(pod -> pod < pod_threshold, pods)
    csis[i]
  end
end

function mean_csi(ŷ, y, weights)
  csis = csi(ŷ, y, weights)
  Float32(sum(csis) / length(csis))
end


# Plan:
# 0. x Use all SREF forecasts, not just 0Z
# 1. blur SREF to maximize AUC (hour-based)
# 2. blur HREF to maximize AUC (hour-based)
# 3. bin HREF predictions into 10 bins of equal weight of positive labels
# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# (5.5. add a4*hour/36*logit(HREF) + a5*hour/36*logit(SREF) + a6*hour/36*logit(HREF)*logit(SREF) + a7*hour/36 terms + a8*logit(hour in day tor prob) + a9*logit(hour in day tor prob given severe) + a10*logit(geomean previous two)? check via cross-validation)
# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)


# For days:
# 1. Try both independent events total prob and max hourly prob as the main descriminator
# 2. bin predictions into 10 bins of equal weight of positive labels
# 3. combine bin-pairs (overlapping, 9 bins total)
# 4. train a logistic regression for each bin,
#   σ(a1*logit(independent events total prob) +
#     a2*logit(max hourly prob) +
#     a3*logit(2nd highest hourly prob) +
#     a4*logit(3rd highest hourly prob) +
#     a5*logit(4th highest hourly prob) +
#     a6*logit(5th highest hourly prob) +
#     a7*logit(6th highest hourly prob) +
#     a8*logit(tornado day climatological prob) +
#     a9*logit(tornado day given severe day climatological prob) +
#     a10*logit(geomean(above two)) +
#     a11*logit(tornado prob for given month) +
#     a12*logit(tornado prob given severe day for given month) +
#     a13*logit(geomean(above two)) +
#     b)
#   Check & eliminate terms via 3-fold cross-validation.
# 5. prediction is weighted mean of the two overlapping logistic models
# 6. should thereby be absolutely calibrated (check)
# 7. calibrate to SPC thresholds (linear interpolation)

# Don't worry about tiny regions (yet)
#
# Compare to SPC over validation set

println("Determining best blur radii")

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
inventory = Forecasts.inventory(validation_forecasts[1])

# for j in 1:(length(inventory) - 5)
#   println("$j $(Inventories.inventory_line_description(inventory[j]))")
#   println("AUC: $(roc_auc((@view X[:,j]), y, weights))")
# end

blur_radii = [0; HREFPrediction.blur_radii]
forecast_hour_j = length(inventory)

window_size = 6
println("$window_size hour windows")

println("forecast_hour\tbest_blur_radius\tAUC")
for forecast_hour in (2 + window_size):38
  mask           = ((@view X[:,forecast_hour_j]) .>= forecast_hour - window_size) .& ((@view X[:,forecast_hour_j]) .<= forecast_hour)
  masked_y       = y[mask]
  masked_weights = weights[mask]

  best_blur_j = nothing
  best_auc    = 0.0

  total_weight    = sum(Float64.(masked_weights))
  positive_weight = sum(masked_y .* Float64.(masked_weights))

  for blur_j in 1:length(blur_radii)
    auc = roc_auc(X[mask, blur_j], masked_y, masked_weights; total_weight = total_weight, positive_weight = positive_weight)
    # println("forecast_hour\tblur_radius\tAUC")
    # println("$forecast_hour\t$(blur_radii[blur_j])\t$(Float32(auc))")

    if auc > best_auc
      best_blur_j = blur_j
      best_auc    = auc
    end
  end

  println("$forecast_hour\t$(blur_radii[best_blur_j])\t$(Float32(best_auc))")
end

# forecast_hour   best_blur_radius        AUC


println("blur_radius_f2\tblur_radius_f38\tAUC")

total_weight    = sum(Float64.(weights))
positive_weight = sum(y .* Float64.(weights))

for blur_i_lo in 1:length(blur_radii)
  for blur_i_hi in 1:length(blur_radii)
    X_blurred = zeros(Float32, length(y))

    Threads.@threads for i in 1:length(y)
      forecast_ratio = (X[i,forecast_hour_j] - 2f0) * (1f0/(38f0-2f0))
      X_blurred[i] = X[i,blur_i_lo] * (1f0 - forecast_ratio) + X[i,blur_i_hi] * forecast_ratio
    end

    auc = roc_auc(X_blurred, y, weights; total_weight = total_weight, positive_weight = positive_weight)

    println("$(blur_radii[blur_i_lo])\t$(blur_radii[blur_i_hi])\t$(Float32(auc))")
  end
end

# blur_radius_f2  blur_radius_f38 AUC


# Sanity check. Should be ...

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import HREFPrediction

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_blurred_and_forecast_hour(); just_hours_near_storm_events = false);

# Make sure a forecast loads
import Forecasts
Forecasts.data(validation_forecasts_blurred[100])

X2, y2, weights2 = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; save_dir = "validation_forecasts_blurred_and_forecast_hour");

roc_auc((@view X2[:,1]), y, weights) # Expected: ...



# # Checking:

# (_, validation_forecasts2, _) = TrainingShared.forecasts_train_validation_test(SREFPrediction.forecasts_blurred_and_hour_climatology(); just_hours_near_storm_events = false);

# import ForecastCombinators
# push!(LOAD_PATH, (@__DIR__) * "/../../lib")
# import Forecasts
# import Inventories
# inventory = Forecasts.inventory(validation_forecasts2[1])

# # For the purposes of AUC checking, we only need the first two features (the predictions)
# # Trying to save some memory
# validation_forecasts2_trimmed =
#   ForecastCombinators.filter_features_forecasts(validation_forecasts2, line -> line.abbrev == "tornado probability");

# X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts2_trimmed; save_dir = "validation_forecasts_blurred_and_hour_climatology_predictions_only");

# println("HREF AUC: $(roc_auc(X[:,1], y, weights))")
# println("SREF AUC: $(roc_auc(X[:,2], y, weights))")




# n_folds = 3 # Choose something not divisible by 7, they're already partitioned by that
# folds = map(1:n_folds) do n
#   fold_forecasts = filter(forecast -> Forecasts.valid_time_in_convective_days_since_epoch_utc(forecast) % n_folds == n-1, validation_forecasts2)
#   fold_X, fold_y, fold_weights = TrainingShared.get_data_labels_weights(fold_forecasts; save_dir = "validation_forecasts_blurred_and_hour_climatology_fold_$(n)");
#   (X = fold_X, y = fold_y, weights = fold_weights)
# end

