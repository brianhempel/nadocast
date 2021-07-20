import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer(); just_hours_near_storm_events = false);

length(validation_forecasts) # 14640

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 12464


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_href_newer");


Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9833407781228322
Metrics.roc_auc((@view X[:,2]), y, weights) # 0.9789812699518358

Metrics.roc_auc(0.8f0 .* X[:,1] .+ 0.2f0 .* X[:,2], y, weights) # 0.9831764833007209
Metrics.roc_auc(0.9f0 .* X[:,1] .+ 0.1f0 .* X[:,2], y, weights) # 0.9833499318886384
Metrics.roc_auc(0.92f0 .* X[:,1] .+ 0.08f0 .* X[:,2], y, weights) # 0.9833739737072544
Metrics.roc_auc(0.95f0 .* X[:,1] .+ 0.05f0 .* X[:,2], y, weights) # 0.9833991831782952
Metrics.roc_auc(0.98f0 .* X[:,1] .+ 0.02f0 .* X[:,2], y, weights) # 0.98340183553223 ****
Metrics.roc_auc(0.99f0 .* X[:,1] .+ 0.01f0 .* X[:,2], y, weights) # 0.9833909538718132

Metrics.roc_auc(X[:,1] .* X[:,2], y, weights) # 0.9828690536162518


# 3. bin HREF predictions into 10 bins of equal weight of positive labels

ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; # 450075040×2 Array{Float32,2}
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 41618.656f0

bin_count = 6
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count # 4161.8657f0

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
  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("bins_max = ")
println(bins_max)

println("mean_y\tmean_ŷ\tΣweight\tbin_max")
for bin_i in 1:bin_count
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end

# mean_y  mean_ŷ  Σweight bin_max
# 1.7093242783136054e-5   1.6127353600648782e-5   4.05814086045141e8      0.000962529
# 0.001770997144582146    0.00197414754830567     3.916877433709204e6     0.0040673064
# 0.005556945875638406    0.006311162159425093    1.2482885660216212e6    0.009957244
# 0.012634757611182369    0.013949859332435916    549063.3355151415       0.020302918
# 0.03304568464045081     0.026607782036935347    209932.50258797407      0.037081156
# 0.07952174760387734     0.05468136008199715     87194.71405380964       1.0

# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms

bins_logistic_coeffs = []

# Paired, overlapping bins
for bin_i in 1:(bin_count - 1)
  bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
  bin_max = bins_max[bin_i+1]

  bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

  bin_X       = X[bin_members,:]
  bin_ŷ       = ŷ[bin_members]
  bin_y       = y[bin_members]
  bin_weights = weights[bin_members]
  bin_weight  = Float32(bins_Σweight[bin_i] + bins_Σweight[bin_i+1])

  println("Bin $bin_i-$(bin_i+1) --------")
  println("$(bin_min) < HREF_ŷ <= $(bin_max)")
  println("Data count: $(length(bin_y))")
  println("Positive count: $(sum(bin_y))")
  println("Weight: $(bin_weight)")
  println("Mean HREF_ŷ: $(sum((@view bin_X[:,1]) .* bin_weights) / bin_weight)")
  println("Mean SREF_ŷ: $(sum((@view bin_X[:,2]) .* bin_weights) / bin_weight)")
  println("Mean y:      $(sum(bin_y .* bin_weights) / bin_weight)")
  println("HREF logloss: $(sum(logloss.(bin_y, (@view bin_X[:,1])) .* bin_weights) / bin_weight)")
  println("SREF logloss: $(sum(logloss.(bin_y, (@view bin_X[:,2])) .* bin_weights) / bin_weight)")
  println("HREF AUC: $(Metrics.roc_auc((@view bin_X[:,1]), bin_y, bin_weights))")
  println("SREF AUC: $(Metrics.roc_auc((@view bin_X[:,2]), bin_y, bin_weights))")
  println("0.98*HREF+0.02*SREF AUC: $(Metrics.roc_auc(0.98f0 .* bin_X[:,1] .+ 0.02f0 .* bin_X[:,2], bin_y, bin_weights))")

  # logit(HREF), logit(SREF), HREF*SREF, logit(HREF*SREF), max(logit(HREF),logit(SREF)), min(logit(HREF),logit(SREF))
  bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

  Threads.@threads for i in 1:length(bin_y)
    logit_href = logit(bin_X[i,1])
    logit_sref = logit(bin_X[i,2])

    bin_X_features[i,1] = logit_href
    bin_X_features[i,2] = logit_sref
    # bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
    # bin_X_features[i,3] = logit(bin_X[i,1]*bin_X[i,2])
    # bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
    # bin_X_features[i,5] = max(logit_href, logit_sref)
    # bin_X_features[i,6] = min(logit_href, logit_sref)
  end

  coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

  println("Fit logistic coefficients: $(coeffs)")

  logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

  println("Mean logistic_ŷ: $(sum(logistic_ŷ .* bin_weights) / bin_weight)")
  println("Logistic logloss: $(sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight)")
  println("Logistic AUC: $(Metrics.roc_auc(logistic_ŷ, bin_y, bin_weights))")

  push!(bins_logistic_coeffs, coeffs)
end

# Bin 1-2 --------
# -1.0 < HREF_ŷ <= 0.0040673064
# Data count: 447858428
# Positive count: 14999.0
# Weight: 4.0973098e8
# Mean HREF_ŷ: 3.4845303e-5
# Mean SREF_ŷ: 5.346712e-5
# Mean y:      3.3859917e-5
# HREF logloss: 0.00029107
# SREF logloss: 0.00031317156
# HREF AUC: 0.957299691712132
# SREF AUC: 0.9474509456092955
# 0.98*HREF+0.02*SREF AUC: 0.9574794268903781
# Fit logistic coefficients: Float32[0.77540565, 0.19299681, -0.21271989]
# Mean logistic_ŷ: 3.3859917e-5
# Logistic logloss: 0.00028985497
# Logistic AUC: 0.9576177024448136
# Bin 2-3 --------
# 0.000962529 < HREF_ŷ <= 0.009957244
# Data count: 5529664
# Positive count: 14787.0
# Weight: 5.165166e6
# Mean HREF_ŷ: 0.003022293
# Mean SREF_ŷ: 0.0028288637
# Mean y:      0.002685964
# HREF logloss: 0.01803897
# SREF logloss: 0.019109778
# HREF AUC: 0.6781641448257744
# SREF AUC: 0.6270299202849181
# 0.98*HREF+0.02*SREF AUC: 0.6790409071908659
# Fit logistic coefficients: Float32[0.84564245, 0.14841641, -0.06817224]
# Mean logistic_ŷ: 0.0026859634
# Logistic logloss: 0.017981809
# Logistic AUC: 0.683341801973895
# Bin 3-4 --------
# 0.0040673064 < HREF_ŷ <= 0.020302918
# Data count: 1906594
# Positive count: 14632.0
# Weight: 1.7973519e6
# Mean HREF_ŷ: 0.0086446665
# Mean SREF_ŷ: 0.006166528
# Mean y:      0.0077191084
# HREF logloss: 0.044290453
# SREF logloss: 0.046770364
# HREF AUC: 0.6450960668830347
# SREF AUC: 0.5935822709669953
# 0.98*HREF+0.02*SREF AUC: 0.6460015914750974
# Fit logistic coefficients: Float32[0.9977281, 0.14388186, 0.64254296]
# Mean logistic_ŷ: 0.0077191065
# Logistic logloss: 0.044153214
# Logistic AUC: 0.6511050423058052
# Bin 4-5 --------
# 0.009957244 < HREF_ŷ <= 0.037081156
# Data count: 798447
# Positive count: 14562.0
# Weight: 758995.8
# Mean HREF_ŷ: 0.017450947
# Mean SREF_ŷ: 0.010616818
# Mean y:      0.018280266
# HREF logloss: 0.08886718
# SREF logloss: 0.096353196
# HREF AUC: 0.6535971012169608
# SREF AUC: 0.5730014427563903
# 0.98*HREF+0.02*SREF AUC: 0.653592212160162
# Fit logistic coefficients: Float32[1.3795987, 0.091625534, 1.9759048]
# Mean logistic_ŷ: 0.018280266
# Logistic logloss: 0.0885154
# Logistic AUC: 0.6553196275557029
# Bin 5-6 --------
# 0.020302918 < HREF_ŷ <= 1.0
# Data count: 310018
# Positive count: 14462.0
# Weight: 297127.22
# Mean HREF_ŷ: 0.03484623
# Mean SREF_ŷ: 0.019427333
# Mean y:      0.046684507
# HREF logloss: 0.1849636
# SREF logloss: 0.20539832
# HREF AUC: 0.6429831812856853
# SREF AUC: 0.602931863588318
# 0.98*HREF+0.02*SREF AUC: 0.6433563836083285
# Fit logistic coefficients: Float32[0.9358031, 0.1812378, 0.836498]
# Mean logistic_ŷ: 0.046684515
# Logistic logloss: 0.1825411
# Logistic AUC: 0.6446689518090329


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)



import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer_combined(); just_hours_near_storm_events = false);

length(combined_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) # Expected: 12464
# 12464

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])


X, y, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; save_dir = "combined_validation_forecasts_href_newer");


ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) #

bin_count = 20
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count #

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
  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("bins_max = ")
println(bins_max)

println("mean_y\tmean_ŷ\tΣweight\tbin_max")
for bin_i in 1:bin_count
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end

# mean_y  mean_ŷ  Σweight bin_max
# 5.322560722124211e-6    5.8811573125446005e-6   3.911319965931541e8     9.887987e-5
# 0.00019897465672958535  0.00019454206020272648  1.0462843866977692e7    0.00037690063
# 0.0005686291530310367   0.0005456260122432716   3.659761417550981e6     0.0007842968
# 0.001124113270554064    0.0010030568344838555   1.8515379332851768e6    0.0012828964
# 0.0017085736510547634   0.0015725956070797495   1.2181251957098246e6    0.001929728
# 0.0022484146804597534   0.002339690964324273    925741.6350624561       0.002841651
# 0.002882528989945521    0.0034183730723036767   722101.1699277163       0.0041100453
# 0.004501347402880451    0.004745127229969608    462419.2483088374       0.0055032303
# 0.006560868100279065    0.006242639048840438    317262.77945637703      0.0071106516
# 0.008274812728123419    0.008048372308300503    251514.2555526495       0.009134761
# 0.01040401562636757     0.010259118930113867    200084.54374402761      0.011555698
# 0.01253964938577878     0.013084069069929535    165975.35050690174      0.01495919
# 0.016783502945246494    0.017087142267366313    124033.5082758069       0.019724188
# 0.02196217402350504     0.02273457447872943     94769.14799720049       0.026396887
# 0.031611153830540376    0.02990545021054135     65836.33591234684       0.03387627
# 0.03894577068013097     0.03799737080267973     53439.49512767792       0.042718615
# 0.04518538143221551     0.04810086833999881     46055.43351483345       0.05448793
# 0.06681750111822628     0.06099447983555109     31156.07918536663       0.068961926
# 0.0943207881878254      0.07820774570388521     22066.47932779789       0.0906716
# 0.11065026340006642     0.12204214860594781     18722.128450989723      1.0

Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9834577307320753








# n_folds = 3 # Choose something not divisible by 7, they're already partitioned by that
# folds = map(1:n_folds) do n
#   fold_forecasts = filter(forecast -> Forecasts.valid_time_in_convective_days_since_epoch_utc(forecast) % n_folds == n-1, validation_forecasts2)
#   fold_X, fold_y, fold_weights = TrainingShared.get_data_labels_weights(fold_forecasts; save_dir = "validation_forecasts_href_newer_blurred_and_hour_climatology_fold_$(n)");
#   (X = fold_X, y = fold_y, weights = fold_weights)
# end


























#
#
# Calibrating SREF newer
#
#


import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_sref_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_sref_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_sref_newer(); just_hours_near_storm_events = false);

length(validation_forecasts) # 13455

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 11347


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_sref_newer");


Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9829061663930163
Metrics.roc_auc((@view X[:,2]), y, weights) # 0.9793759050730833

Metrics.roc_auc(0.8f0 .* X[:,1] .+ 0.2f0 .* X[:,2], y, weights) # 0.9830208225628564
Metrics.roc_auc(0.9f0 .* X[:,1] .+ 0.1f0 .* X[:,2], y, weights) # 0.0.983134353268765
Metrics.roc_auc(0.92f0 .* X[:,1] .+ 0.08f0 .* X[:,2], y, weights) # 0.983138148522048 ****
Metrics.roc_auc(0.95f0 .* X[:,1] .+ 0.05f0 .* X[:,2], y, weights) # 0.9831219110071081
Metrics.roc_auc(0.98f0 .* X[:,1] .+ 0.02f0 .* X[:,2], y, weights) # 0.9830570120374379
Metrics.roc_auc(0.99f0 .* X[:,1] .+ 0.01f0 .* X[:,2], y, weights) # 0.9830106697665038

Metrics.roc_auc(X[:,1] .* X[:,2], y, weights) # 0.9827395411324908


# 3. bin HREF predictions into 10 bins of equal weight of positive labels

ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; # 409740170×2 Array{Float32,2}
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 38244.39f0

bin_count = 6
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
  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("bins_max = ")
println(bins_max)
# Float32[0.0009233353, 0.0038515618, 0.00954726, 0.01923272, 0.035036117, 1.0]

println("mean_y\tmean_ŷ\tΣweight\tbin_max")
for bin_i in 1:bin_count
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end

# mean_y  mean_ŷ  Σweight bin_max
# 1.7264677587925844e-5   1.61261532234735e-5     3.692360973807265e8     0.0009233353
# 0.0017351929753948545   0.0018818112425679743   3.673934999193132e6     0.0038515618
# 0.005292660143932808    0.00601392616502343     1.204344835934639e6     0.00954726
# 0.012305667039831332    0.0133210959328097      518006.0649062395       0.01923272
# 0.031393511640927706    0.025171920335517555    203041.68478280306      0.035036117
# 0.07676594810361057     0.051474065903578045    83003.88141471148       1.0



# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms

bins_logistic_coeffs = []

# Paired, overlapping bins
for bin_i in 1:(bin_count - 1)
  bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
  bin_max = bins_max[bin_i+1]

  bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

  bin_X       = X[bin_members,:]
  bin_ŷ       = ŷ[bin_members]
  bin_y       = y[bin_members]
  bin_weights = weights[bin_members]
  bin_weight  = Float32(bins_Σweight[bin_i] + bins_Σweight[bin_i+1])

  println("Bin $bin_i-$(bin_i+1) --------")
  println("$(bin_min) < HREF_ŷ <= $(bin_max)")
  println("Data count: $(length(bin_y))")
  println("Positive count: $(sum(bin_y))")
  println("Weight: $(bin_weight)")
  println("Mean HREF_ŷ: $(sum((@view bin_X[:,1]) .* bin_weights) / bin_weight)")
  println("Mean SREF_ŷ: $(sum((@view bin_X[:,2]) .* bin_weights) / bin_weight)")
  println("Mean y:      $(sum(bin_y .* bin_weights) / bin_weight)")
  println("HREF logloss: $(sum(logloss.(bin_y, (@view bin_X[:,1])) .* bin_weights) / bin_weight)")
  println("SREF logloss: $(sum(logloss.(bin_y, (@view bin_X[:,2])) .* bin_weights) / bin_weight)")
  println("HREF AUC: $(Metrics.roc_auc((@view bin_X[:,1]), bin_y, bin_weights))")
  println("SREF AUC: $(Metrics.roc_auc((@view bin_X[:,2]), bin_y, bin_weights))")
  println("0.98*HREF+0.02*SREF AUC: $(Metrics.roc_auc(0.98f0 .* bin_X[:,1] .+ 0.02f0 .* bin_X[:,2], bin_y, bin_weights))")

  # logit(HREF), logit(SREF), HREF*SREF, logit(HREF*SREF), max(logit(HREF),logit(SREF)), min(logit(HREF),logit(SREF))
  bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

  Threads.@threads for i in 1:length(bin_y)
    logit_href = logit(bin_X[i,1])
    logit_sref = logit(bin_X[i,2])

    bin_X_features[i,1] = logit_href
    bin_X_features[i,2] = logit_sref
    # bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
    # bin_X_features[i,3] = logit(bin_X[i,1]*bin_X[i,2])
    # bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
    # bin_X_features[i,5] = max(logit_href, logit_sref)
    # bin_X_features[i,6] = min(logit_href, logit_sref)
  end

  coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

  println("Fit logistic coefficients: $(coeffs)")

  logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

  println("Mean logistic_ŷ: $(sum(logistic_ŷ .* bin_weights) / bin_weight)")
  println("Logistic logloss: $(sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight)")
  println("Logistic AUC: $(Metrics.roc_auc(logistic_ŷ, bin_y, bin_weights))")

  push!(bins_logistic_coeffs, coeffs)
end

# Bin 1-2 --------
# -1.0 < HREF_ŷ <= 0.0038515618
# Data count: 407613435
# Positive count: 13795.0
# Weight: 3.7291005e8
# Mean HREF_ŷ: 3.4507008e-5
# Mean SREF_ŷ: 5.235415e-5
# Mean y:      3.4189823e-5
# HREF logloss: 0.00029473202
# SREF logloss: 0.00031075315
# HREF AUC: 0.9563687879352794
# SREF AUC: 0.947897012376162
# 0.98*HREF+0.02*SREF AUC: 0.9568097124469311
# Fit logistic coefficients: Float32[0.68609446, 0.29776412, -0.06935765]
# Mean logistic_ŷ: 3.4189816e-5
# Logistic logloss: 0.00029202303
# Logistic AUC: 0.9569031225763764
# Bin 2-3 --------
# 0.0009233353 < HREF_ŷ <= 0.00954726
# Data count: 5224621
# Positive count: 13594.0
# Weight: 4.87828e6
# Mean HREF_ŷ: 0.0029019434
# Mean SREF_ŷ: 0.0028575049
# Mean y:      0.0026134567
# HREF logloss: 0.017626058
# SREF logloss: 0.018356862
# HREF AUC: 0.6769379783509994
# SREF AUC: 0.6571324438631635
# 0.98*HREF+0.02*SREF AUC: 0.6792089024934399
# Fit logistic coefficients: Float32[0.73588675, 0.26439694, 0.021931686]
# Mean logistic_ŷ: 0.0026134565
# Logistic logloss: 0.01748634
# Logistic AUC: 0.6929151892095925
# Bin 3-4 --------
# 0.0038515618 < HREF_ŷ <= 0.01923272
# Data count: 1828148
# Positive count: 13438.0
# Weight: 1.7223509e6
# Mean HREF_ŷ: 0.008211596
# Mean SREF_ŷ: 0.006386808
# Mean y:      0.0074018594
# HREF logloss: 0.042738765
# SREF logloss: 0.043982364
# HREF AUC: 0.6487671072550764
# SREF AUC: 0.6367833629662765
# 0.98*HREF+0.02*SREF AUC: 0.651493716512653
# Fit logistic coefficients: Float32[0.8676892, 0.3169111, 0.9392679]
# Mean logistic_ŷ: 0.00740186
# Logistic logloss: 0.04231037
# Logistic AUC: 0.6741230493453069
# Bin 4-5 --------
# 0.00954726 < HREF_ŷ <= 0.035036117
# Data count: 758928
# Positive count: 13373.0
# Weight: 721047.75
# Mean HREF_ŷ: 0.0166582
# Mean SREF_ŷ: 0.011239922
# Mean y:      0.017680662
# HREF logloss: 0.08663094
# SREF logloss: 0.09124912
# HREF AUC: 0.6492170385710861
# SREF AUC: 0.6098851312615089
# 0.98*HREF+0.02*SREF AUC: 0.6509072809498203
# Fit logistic coefficients: Float32[1.2143207, 0.26236853, 2.1367016]
# Mean logistic_ŷ: 0.01768066
# Logistic logloss: 0.08585571
# Logistic AUC: 0.6638099455529567
# Bin 5-6 --------
# 0.01923272 < HREF_ŷ <= 1.0
# Data count: 298587
# Positive count: 13286.0
# Weight: 286045.56
# Mean HREF_ŷ: 0.032804202
# Mean SREF_ŷ: 0.020830723
# Mean y:      0.044559553
# HREF logloss: 0.1783302
# SREF logloss: 0.19152443
# HREF AUC: 0.6506293527981011
# SREF AUC: 0.6348837063148317
# 0.98*HREF+0.02*SREF AUC: 0.652039754698053
# Fit logistic coefficients: Float32[0.851503, 0.31797588, 1.0932944]
# Mean logistic_ŷ: 0.044559553
# Logistic logloss: 0.17479251
# Logistic AUC: 0.6614826891786213


# 7. predictions should thereby be calibrated (check)



import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_sref_newer_combined(); just_hours_near_storm_events = false);

length(combined_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) # Expected: 11347
#

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])


X, y, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; save_dir = "combined_validation_forecasts_sref_newer");


ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) #

bin_count = 20
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count #

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
  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("bins_max = ")
println(bins_max)

println("mean_y\tmean_ŷ\tΣweight\tbin_max")
for bin_i in 1:bin_count
  Σŷ      = bins_Σŷ[bin_i]
  Σy      = bins_Σy[bin_i]
  Σweight = bins_Σweight[bin_i]

  mean_ŷ = Σŷ / Σweight
  mean_y = Σy / Σweight

  println("$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
end
