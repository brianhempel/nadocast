# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.

# Any number that is 1.6777216e7 has had addition round-off error

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
using Metrics

push!(LOAD_PATH, @__DIR__)
import HRRRPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HRRRPrediction.forecasts_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts); # 20440
validation_forecasts = filter(forecast -> forecast.forecast_hour in 2:17, validation_forecasts); # 19726
validation_forecasts = filter(forecast -> forecast.run_hour in [8,9,10,12,13,14], validation_forecasts); # 19726


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_with_blurs_and_forecast_hour");

length(validation_forecasts) # 19726
size(X) # (1911262400, 8)
length(y) # 1911262400

println("Dividing into bins of equal positive weight...")

ŷ = X[:, 1]

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]
y       = y[sort_perm]
ŷ       = ŷ[sort_perm]
weights = weights[sort_perm]

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights)

bin_count = 6 # equal logloss bins
per_bin_pos_weight = total_positive_weight / bin_count

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
# 0.0019516714    0.0011239782    1.6777216e7     0.0010939918 # This line is wrong, addition underflow, but we aren't using this downstream so it's okay
# 0.0019686848    0.002285054     1.6631918e7     0.005531816 # This line is likely wrong, addition underflow, but we aren't using this downstream so it's okay
# 0.007366764     0.008583214     4.4447365e6     0.014440106
# 0.018769274     0.020027727     1.744537e6      0.029168867
# 0.039853472     0.038245406     821593.94       0.052170333
# 0.06926268      0.08326917      472691.6        1.0

roc_auc(X[:,1], y, weights) # 0.9872143259000387

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


# Plan:
# 0. x Use all SREF forecasts, not just 0Z
# 1. blur SREF to maximize AUC (hour-based)
# 2. blur HRRR to maximize AUC (hour-based)
# 3. bin HRRR predictions into 10 bins of equal weight of positive labels
# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HRRR) + a2*logit(SREF) + a3*logit(HRRR)*logit(SREF) + a4*max(logit(HRRR),logit(SREF)) + a5*min(logit(HRRR),logit(SREF)) + b)
# (5.5. add a4*hour/36*logit(HRRR) + a5*hour/36*logit(SREF) + a6*hour/36*logit(HRRR)*logit(SREF) + a7*hour/36 terms + a8*logit(hour in day tor prob) + a9*logit(hour in day tor prob given severe) + a10*logit(geomean previous two)? check via cross-validation)
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

blur_radii = [0; HRRRPrediction.blur_radii]
forecast_hour_j = length(inventory)

println("blur_radius_f2\tblur_radius_f17\tAUC")

total_weight    = sum(Float64.(weights))
positive_weight = sum(y .* Float64.(weights))

X_blurred = Array{Float32}(undef, length(y))

# Mutates X_blurred
function does_this_need_to_be_a_function_to_be_fast(X_blurred, X, blur_i_lo, blur_i_hi, forecast_hour_j)
  Threads.@threads for i in 1:length(y)
    forecast_ratio = (X[i,forecast_hour_j] - 2f0) * (1f0/(17f0-2f0))
    X_blurred[i] = X[i,blur_i_lo] * (1f0 - forecast_ratio) + X[i,blur_i_hi] * forecast_ratio
  end
end

for blur_i_lo in 1:length(blur_radii)
  for blur_i_hi in 1:length(blur_radii)

    does_this_need_to_be_a_function_to_be_fast(X_blurred, X, blur_i_lo, blur_i_hi, forecast_hour_j)

    auc = Metrics.roc_auc(X_blurred, y, weights; total_weight = total_weight, positive_weight = positive_weight)

    println("$(blur_radii[blur_i_lo])\t$(blur_radii[blur_i_hi])\t$(Float32(auc))")
  end
end

X_blurred = nothing

# blur_radius_f2  blur_radius_f17 AUC
# 0       0       0.9872143
# 0       10      0.987271
# 0       15      0.9873093
# 0       25      0.9873899
# 0       35      0.98743355
# 0       50      0.9874199
# 0       70      0.98723793
# 10      0       0.98724425
# 10      10      0.9872866
# 10      15      0.9873214
# 10      25      0.98739755
# 10      35      0.9874394 # BEST
# 10      50      0.98742443
# 10      70      0.9872425
# 15      0       0.98725706
# 15      10      0.98729616
# 15      15      0.98732686
# 15      25      0.98739773
# 15      35      0.98743725
# 15      50      0.98742044
# 15      70      0.98723763
# 25      0       0.9872664
# 25      10      0.98730123
# 25      15      0.98732656
# 25      25      0.987383
# 25      35      0.9874146
# 25      50      0.9873908
# 25      70      0.98720396
# 35      0       0.9872478
# 35      10      0.9872806
# 35      15      0.9873033
# 35      25      0.98735124
# 35      35      0.9873749
# 35      50      0.98734295
# 35      70      0.9871508
# 50      0       0.9871574
# 50      10      0.9871884
# 50      15      0.9872085
# 50      25      0.9872473
# 50      35      0.98726165
# 50      50      0.9872167
# 50      70      0.9870148
# 70      0       0.98693454
# 70      10      0.9869645
# 70      15      0.9869827
# 70      25      0.98701423
# 70      35      0.9870206
# 70      50      0.9869635
# 70      70      0.98674834


# Sanity check. Should be 0.9874394

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HRRRPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HRRRPrediction.forecasts_blurred(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);
validation_forecasts_blurred = filter(forecast -> forecast.forecast_hour in 2:17, validation_forecasts_blurred);
validation_forecasts_blurred = filter(forecast -> forecast.run_hour in [8,9,10,12,13,14], validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100])

X2, y2, weights2 = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; save_dir = "validation_forecasts_blurred");

Float32(Metrics.roc_auc((@view X2[:,1]), y2, weights2)) # Expected: 0.9874394
# 0.9874394f0

