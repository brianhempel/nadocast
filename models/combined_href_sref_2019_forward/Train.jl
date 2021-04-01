import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer_with_blurs_and_hour_climatology());

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_href_newer_with_blurs_and_hour_climatology");

function try_combine(combiner)
  ŷ = map(i -> combiner(X[i, :]), 1:length(y))

  sum(logloss.(y, ŷ) .* weights) / sum(weights)
end

println( try_combine(minimum) )
println( try_combine(maximum) )
println( try_combine(x -> x[2]) )
println( try_combine(x -> x[1]) )
println( try_combine(x -> 0.9f0*x[1] + 0.1f0*x[2]) ) # best so far, via logloss; best for AUC is 0.8 and 0.2


ŷ = map(i -> 0.9*X[i, 1] + 0.1*X[i, 2], 1:length(y)); # minimizes logloss
# ŷ = map(i -> X[i, 1], 1:length(y));
# ŷ = map(i -> X[i, 2], 1:length(y));

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
bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
bins_Σy      = map(_ -> 0.0, 1:bin_count)
bins_Σweight = map(_ -> 0.0, 1:bin_count)
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
  if bins_Σy >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

for bin_i in 1:length(bins_Σy)
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end

function roc_auc(ŷ, y, weights)
  sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort)
  y       = y[sort_perm]
  ŷ       = ŷ[sort_perm]
  weights = Float64.(weights[sort_perm])

  total_weight    = sum(weights)
  positive_weight = sum(y .* weights)
  negative_weight = total_weight - positive_weight

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

roc_auc(map(i -> 0.9*X[i, 1] + 0.1*X[i, 2], 1:length(y)), y, weights) # 0.9875262383941
roc_auc(map(i -> 0.8*X[i, 1] + 0.2*X[i, 2], 1:length(y)), y, weights) # 0.987625756842824, v hard to beat

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


# CSI = hits / (hits + false alarms + misses)
#     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)
#     = 1 / (1/POD + 1/(1-FAR) - 1)

function csi(ŷ, y, weights)
  sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort)
  y       = y[sort_perm]
  ŷ       = ŷ[sort_perm]
  weights = Float64.(weights[sort_perm])

  total_weight    = sum(weights)
  positive_weight = sum(y .* weights)
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
# 0. x Use all SREF/HREF forecasts, not just 0Z
# 1. blur HREF to maximize AUC (hour-based)
# 2. blue SREF to maximize AUC (hour-based)
# 3. bin HREF predictions into 10 bins of equal weight of positive labels
# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + b)
# (5.5. add a4*hour/36*logit(HREF) + a5*hour/36*logit(SREF) + a6*hour/36*logit(HREF)*logit(SREF) + a7*hour/36 terms + a8*logit(hour in day tor prob) + a9*logit(hour in day tor prob given severe) + a10*logit(geomean previous two)? check via cross-validation)
# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)

# blurs for SREF should be 26mi, 40mi, 55mi, 80mi, 110mi NOPE BLUR AFTER UPSAMPLING
# blurs for HREF should be 10mi, 15mi, 25mi, 35mi, 50mi, 70mi, 100mi


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
