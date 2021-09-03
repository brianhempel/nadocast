import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHRRRRAPHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import ForecastCombinators

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHRRRRAPHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_separate(); just_hours_near_storm_events = false);

length(validation_forecasts) # 2933

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

# Only 10Z and 14Z HRRRs were downloaded, 2019-01-12 through 2020-10-17
length(validation_forecasts) # 2636

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

# @time Forecasts.data(validation_forecasts[100]); # make sure a forecast loads
# @time Forecasts.data(validation_forecasts[101]); # baseline load time
ForecastCombinators.turn_forecast_caching_on()
@time Forecasts.data(validation_forecasts[1]); # 87462×8 Array{Float32,2}
# @time Forecasts.data(validation_forecasts[102]);
# @time Forecasts.data(validation_forecasts[103]); # make sure caching helps
# @time Forecasts.data(validation_forecasts[104]); # make sure caching helps
# @time Forecasts.data(validation_forecasts[105]); # make sure caching helps

ForecastCombinators.turn_forecast_caching_on()
X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_separate");
ForecastCombinators.clear_cached_forecasts()

Metrics.roc_auc((@view X[:,1]), y, weights) # HRRR-0 # 0.9814505780075439
Metrics.roc_auc((@view X[:,2]), y, weights) # HRRR-1 # 0.9806793021079172
Metrics.roc_auc((@view X[:,3]), y, weights) # HRRR-2 # 0.9798568458863672
Metrics.roc_auc((@view X[:,4]), y, weights) # RAP-0  # 0.9803516889213414
Metrics.roc_auc((@view X[:,5]), y, weights) # RAP-1  # 0.9793563661736295
Metrics.roc_auc((@view X[:,6]), y, weights) # RAP-2  # 0.9784186738340481
Metrics.roc_auc((@view X[:,7]), y, weights) # HREF   # 0.9796398968079756
Metrics.roc_auc((@view X[:,8]), y, weights) # SREF   # 0.97422052207036

Metrics.roc_auc((@view X[:,1]) .+ (@view X[:,2]) .+ (@view X[:,3]) .+ (@view X[:,4]) .+ (@view X[:,5]) .+ (@view X[:,6]) .+ (@view X[:,7]) .+ (@view X[:,8]), y, weights)
# 0.9809890672836681
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,2]) .* (@view X[:,3]) .* (@view X[:,4]) .* (@view X[:,5]) .* (@view X[:,6]) .* (@view X[:,7]) .* (@view X[:,8]), y, weights)
# 0.9814265002781438
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,4]), y, weights)
# 0.9819461611454262
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,7]), y, weights)
# 0.9825032267824557
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,4]) .* (@view X[:,7]), y, weights)
# 0.9827705476598608
Metrics.roc_auc((@view X[:,1]) .+ (@view X[:,4]) .+ (@view X[:,7]), y, weights)
# 0.982413863759543
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,4]) .* (@view X[:,7]) .* (@view X[:,8]), y, weights)
# 0.9818037547220301
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,2]) .* (@view X[:,3]) .* (@view X[:,4]) .* (@view X[:,7]), y, weights)
# 0.9823492107043525


# 3. bin predictions into 6 bins of equal weight of positive labels

ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

total_positive_weight = sum(y .* weights) # 11478.343f0

bin_count = 6
per_bin_pos_weight = total_positive_weight / bin_count # 1913.0571f0

bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
bins_Σy      = map(_ -> 0.0, 1:bin_count)
bins_Σweight = map(_ -> 0.0, 1:bin_count)
bins_max     = map(_ -> 1.0f0, 1:bin_count)

bin_i = 1
for i in 1:length(y)
  global bin_i

  if ŷ[i] > bins_max[bin_i]
    bin_i += 1
  end

  bins_Σŷ[bin_i]      += ŷ[i] * weights[i]
  bins_Σy[bin_i]      += y[i] * weights[i]
  bins_Σweight[bin_i] += weights[i]

  if bins_Σy[bin_i] >= per_bin_pos_weight
    bins_max[bin_i] = ŷ[i]
  end
end

println("bins_max = ")
println(bins_max)
# Float32[0.0010772932, 0.0056677116, 0.016188348, 0.03184764, 0.05504267, 1.0]

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
# 2.23751317246056e-5     2.0173983911533358e-5   8.552338460757327e7     0.0010772932
# 0.0018904718660167287   0.0024668729488375937   1.0120553528988361e6    0.0056677116
# 0.006664794986525737    0.009440528929746525    287185.4086174965       0.016188348
# 0.0195033080276471      0.022474489491229273    98094.33709013462       0.03184764
# 0.04006907385241231     0.04123673880165891     47751.739669561386      0.05504267
# 0.06591080555584303     0.0883819969908915      28992.478892087936      1.0


# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin

bins_logistic_coeffs = Vector{Float32}[]

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
  println("$(bin_min) < HRRRminus0_ŷ <= $(bin_max)")
  println("Data count: $(length(bin_y))")
  println("Positive count: $(sum(bin_y))")
  println("Weight: $(bin_weight)")
  # println("Mean HRRRminus0_ŷ: $(sum((@view bin_X[:,1]) .* bin_weights) / bin_weight)")
  # println("Mean HRRRminus1_ŷ: $(sum((@view bin_X[:,2]) .* bin_weights) / bin_weight)")
  # println("Mean HRRRminus2_ŷ: $(sum((@view bin_X[:,3]) .* bin_weights) / bin_weight)")
  # println("Mean RAPminus0_ŷ:  $(sum((@view bin_X[:,4]) .* bin_weights) / bin_weight)")
  # println("Mean RAPminus1_ŷ:  $(sum((@view bin_X[:,5]) .* bin_weights) / bin_weight)")
  # println("Mean RAPminus2_ŷ:  $(sum((@view bin_X[:,6]) .* bin_weights) / bin_weight)")
  # println("Mean HREF_ŷ:       $(sum((@view bin_X[:,7]) .* bin_weights) / bin_weight)")
  # println("Mean SREF_ŷ:       $(sum((@view bin_X[:,8]) .* bin_weights) / bin_weight)")
  println("Mean y:            $(sum(bin_y .* bin_weights) / bin_weight)")
  # println("HRRRminus0 logloss: $(sum(logloss.(bin_y, (@view bin_X[:,1])) .* bin_weights) / bin_weight)")
  # println("HRRRminus1 logloss: $(sum(logloss.(bin_y, (@view bin_X[:,2])) .* bin_weights) / bin_weight)")
  # println("HRRRminus2 logloss: $(sum(logloss.(bin_y, (@view bin_X[:,3])) .* bin_weights) / bin_weight)")
  # println("RAPminus0 logloss:  $(sum(logloss.(bin_y, (@view bin_X[:,4])) .* bin_weights) / bin_weight)")
  # println("RAPminus1 logloss:  $(sum(logloss.(bin_y, (@view bin_X[:,5])) .* bin_weights) / bin_weight)")
  # println("RAPminus2 logloss:  $(sum(logloss.(bin_y, (@view bin_X[:,6])) .* bin_weights) / bin_weight)")
  # println("HREF logloss:       $(sum(logloss.(bin_y, (@view bin_X[:,7])) .* bin_weights) / bin_weight)")
  # println("SREF logloss:       $(sum(logloss.(bin_y, (@view bin_X[:,8])) .* bin_weights) / bin_weight)")
  # println("HRRRminus0 AUC: $(Metrics.roc_auc((@view bin_X[:,1]), bin_y, bin_weights))")
  # println("HRRRminus1 AUC: $(Metrics.roc_auc((@view bin_X[:,2]), bin_y, bin_weights))")
  # println("HRRRminus2 AUC: $(Metrics.roc_auc((@view bin_X[:,3]), bin_y, bin_weights))")
  # println("RAPminus0 AUC:  $(Metrics.roc_auc((@view bin_X[:,4]), bin_y, bin_weights))")
  # println("RAPminus1 AUC:  $(Metrics.roc_auc((@view bin_X[:,5]), bin_y, bin_weights))")
  # println("RAPminus2 AUC:  $(Metrics.roc_auc((@view bin_X[:,6]), bin_y, bin_weights))")
  # println("HREF AUC:       $(Metrics.roc_auc((@view bin_X[:,7]), bin_y, bin_weights))")
  # println("SREF AUC:       $(Metrics.roc_auc((@view bin_X[:,8]), bin_y, bin_weights))")

  bin_X_features = Array{Float32}(undef, (length(bin_y), 8))
  # bin_X_features = bin_X

  Threads.@threads for i in 1:length(bin_y)
    bin_X_features[i,1] = logit(bin_X[i,1])
    bin_X_features[i,2] = logit(bin_X[i,2])
    bin_X_features[i,3] = logit(bin_X[i,3])
    bin_X_features[i,4] = logit(bin_X[i,4])
    bin_X_features[i,5] = logit(bin_X[i,5])
    bin_X_features[i,6] = logit(bin_X[i,6])
    bin_X_features[i,7] = logit(bin_X[i,7])
    bin_X_features[i,8] = logit(bin_X[i,8])
    # bin_X_features[i,9] = logit(bin_X[i,1] * bin_X[i,4] * bin_X[i,7]) # no convergence
    # bin_X_features[i,9] = logit((bin_X[i,1] * bin_X[i,4] * bin_X[i,7]) .^ (1/3f0)) # no convergence
  end

  # l2_regularization = 0.00001
  l2_regularization = 0.0
  coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300, l2_regularization = l2_regularization)

  println("Fit logistic coefficients: $(coeffs)")

  logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

  println("Mean logistic_ŷ: $(sum(logistic_ŷ .* bin_weights) / bin_weight)")
  println("Logistic logloss: $(sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight)")
  println("Logistic AUC: $(Metrics.roc_auc(logistic_ŷ, bin_y, bin_weights))")

  push!(bins_logistic_coeffs, coeffs)
end

# Bin 1-2 --------
# -1.0 < HRRRminus0_ŷ <= 0.0056677116
# Data count: 94587722
# Positive count: 4165.0
# Weight: 8.653544e7
# Mean HRRRminus0_ŷ: 4.8788792e-5
# Mean HRRRminus1_ŷ: 5.6648078e-5
# Mean HRRRminus2_ŷ: 6.132281e-5
# Mean RAPminus0_ŷ:  7.524401e-5
# Mean RAPminus1_ŷ:  7.738698e-5
# Mean RAPminus2_ŷ:  7.989792e-5
# Mean HREF_ŷ:       6.201495e-5
# Mean SREF_ŷ:       7.436201e-5
# Mean y:            4.4223027e-5
# HRRRminus0 logloss: 0.00038146673
# HRRRminus1 logloss: 0.00038770228
# HRRRminus2 logloss: 0.00039035443
# RAPminus0 logloss:  0.0004016827
# RAPminus1 logloss:  0.00040416777
# RAPminus2 logloss:  0.0004071871
# HREF logloss:       0.00038501437
# SREF logloss:       0.00041688388
# HRRRminus0 AUC: 0.9520201728817965
# HRRRminus1 AUC: 0.9502150443574227
# HRRRminus2 AUC: 0.9476048242637636
# RAPminus0 AUC:  0.9500613419708978
# RAPminus1 AUC:  0.9472434940679192
# RAPminus2 AUC:  0.9446179225816171
# HREF AUC:       0.9476234121173652
# SREF AUC:       0.9344604181411339
# Float32[0.2958114, 0.093472, 0.014974893, 0.25723657, -0.08649167, -0.03651152, 0.51546174, -0.17759001, -0.9377826]]2]]3013]6656]]566]
# Fit logistic coefficients: Float32[0.2958114, 0.093472, 0.014974893, 0.25723657, -0.08649167, -0.03651152, 0.51546174, -0.17759001, -0.9377826]
# Mean logistic_ŷ: 4.4223034e-5
# Logistic logloss: 0.0003693823
# Logistic AUC: 0.9563615530660051
# Bin 2-3 --------
# 0.0010772932 < HRRRminus0_ŷ <= 0.016188348
# Data count: 1394766
# Positive count: 4120.0
# Weight: 1.2992408e6
# Mean HRRRminus0_ŷ: 0.0040083365
# Mean HRRRminus1_ŷ: 0.0040849145
# Mean HRRRminus2_ŷ: 0.00414849
# Mean RAPminus0_ŷ:  0.0043264325
# Mean RAPminus1_ŷ:  0.0041909707
# Mean RAPminus2_ŷ:  0.0040460997
# Mean HREF_ŷ:       0.0036779023
# Mean SREF_ŷ:       0.0032898334
# Mean y:            0.0029457926
# HRRRminus0 logloss: 0.019470988
# HRRRminus1 logloss: 0.020010771
# HRRRminus2 logloss: 0.0200337
# RAPminus0 logloss:  0.020264372
# RAPminus1 logloss:  0.020219946
# RAPminus2 logloss:  0.020290175
# HREF logloss:       0.019373855
# SREF logloss:       0.020942459
# HRRRminus0 AUC: 0.705862360071405
# HRRRminus1 AUC: 0.6686690744276761
# HRRRminus2 AUC: 0.6732500605813102
# RAPminus0 AUC:  0.672487582314471
# RAPminus1 AUC:  0.6705756192675089
# RAPminus2 AUC:  0.6667179433952568
# HREF AUC:       0.732570172330835
# SREF AUC:       0.6275999030257781
# Float32[0.55761576, -0.08032898, 0.029375209, -0.043275304, 0.2413465, -0.029677542, 0.5895104, -0.27158892, -0.28064802]300988]4]322]
# Fit logistic coefficients: Float32[0.55761576, -0.08032898, 0.029375209, -0.043275304, 0.2413465, -0.029677542, 0.5895104, -0.27158892, -0.28064802]
# Mean logistic_ŷ: 0.002945792
# Logistic logloss: 0.01872671
# Logistic AUC: 0.7557408722926326
# Bin 3-4 --------
# 0.0056677116 < HRRRminus0_ŷ <= 0.03184764
# Data count: 409868
# Positive count: 4057.0
# Weight: 385279.75
# Mean HRRRminus0_ŷ: 0.012759047
# Mean HRRRminus1_ŷ: 0.012010568
# Mean HRRRminus2_ŷ: 0.011803538
# Mean RAPminus0_ŷ:  0.010948555
# Mean RAPminus1_ŷ:  0.010299844
# Mean RAPminus2_ŷ:  0.009776067
# Mean HREF_ŷ:       0.009536379
# Mean SREF_ŷ:       0.007351375
# Mean y:            0.00993355
# HRRRminus0 logloss: 0.05446645
# HRRRminus1 logloss: 0.054912858
# HRRRminus2 logloss: 0.0543538
# RAPminus0 logloss:  0.056117564
# RAPminus1 logloss:  0.05597554
# RAPminus2 logloss:  0.055795167
# HREF logloss:       0.05255689
# SREF logloss:       0.058003545
# HRRRminus0 AUC: 0.657521848421291
# HRRRminus1 AUC: 0.6453050946039411
# HRRRminus2 AUC: 0.6669127232200801
# RAPminus0 AUC:  0.6127638293160151
# RAPminus1 AUC:  0.6180847788241574
# RAPminus2 AUC:  0.6296270430672114
# HREF AUC:       0.7367202678200507
# SREF AUC:       0.5905943029131259
# Float32[0.5240835, -0.15477853, 0.30952054, -0.16907471, 0.26656932, 0.04274331, 0.7430586, -0.30433056, 0.9741227]]87]1]42]803]
# Fit logistic coefficients: Float32[0.5240835, -0.15477853, 0.30952054, -0.16907471, 0.26656932, 0.04274331, 0.7430586, -0.30433056, 0.9741227]
# Mean logistic_ŷ: 0.009933552
# Logistic logloss: 0.05166098
# Logistic AUC: 0.7468593335072604
# Bin 4-5 --------
# 0.016188348 < HRRRminus0_ŷ <= 0.05504267
# Data count: 153299
# Positive count: 4023.0
# Weight: 145846.08
# Mean HRRRminus0_ŷ: 0.028617471
# Mean HRRRminus1_ŷ: 0.025752008
# Mean HRRRminus2_ŷ: 0.024672367
# Mean RAPminus0_ŷ:  0.021020614
# Mean RAPminus1_ŷ:  0.019519612
# Mean RAPminus2_ŷ:  0.018241417
# Mean HREF_ŷ:       0.017125282
# Mean SREF_ŷ:       0.013095667
# Mean y:            0.026236784
# HRRRminus0 logloss: 0.11940481
# HRRRminus1 logloss: 0.11965044
# HRRRminus2 logloss: 0.11812625
# RAPminus0 logloss:  0.12087032
# RAPminus1 logloss:  0.12130313
# RAPminus2 logloss:  0.12084382
# HREF logloss:       0.11499897
# SREF logloss:       0.1288669
# HRRRminus0 AUC: 0.6158309132938782
# HRRRminus1 AUC: 0.6219072631875336
# HRRRminus2 AUC: 0.6457782813967978
# RAPminus0 AUC:  0.6236279146864662
# RAPminus1 AUC:  0.6290187422649232
# RAPminus2 AUC:  0.6440421254916995
# HREF AUC:       0.7403572324388772
# SREF AUC:       0.5946706957930683
# Float32[0.4446033, -0.31768143, 0.402504, 0.055888213, 0.10179166, 0.26728168, 0.8905479, -0.28445807, 2.323705]3]39]222]]
# Fit logistic coefficients: Float32[0.4446033, -0.31768143, 0.402504, 0.055888213, 0.10179166, 0.26728168, 0.8905479, -0.28445807, 2.323705]
# Mean logistic_ŷ: 0.026236786
# Logistic logloss: 0.11115688
# Logistic AUC: 0.7567593194474369
# Bin 5-6 --------
# 0.03184764 < HRRRminus0_ŷ <= 1.0
# Data count: 80040
# Positive count: 3990.0
# Weight: 76744.22
# Mean HRRRminus0_ŷ: 0.0590473
# Mean HRRRminus1_ŷ: 0.050603677
# Mean HRRRminus2_ŷ: 0.046447236
# Mean RAPminus0_ŷ:  0.037029617
# Mean RAPminus1_ŷ:  0.03389058
# Mean RAPminus2_ŷ:  0.03025478
# Mean HREF_ŷ:       0.025173621
# Mean SREF_ŷ:       0.021366598
# Mean y:            0.049831584
# HRRRminus0 logloss: 0.19753355
# HRRRminus1 logloss: 0.19835939
# HRRRminus2 logloss: 0.19436929
# RAPminus0 logloss:  0.19778784
# RAPminus1 logloss:  0.19948697
# RAPminus2 logloss:  0.19911201
# HREF logloss:       0.19732708
# SREF logloss:       0.21505101
# HRRRminus0 AUC: 0.5889328612319006
# HRRRminus1 AUC: 0.5911467388013477
# HRRRminus2 AUC: 0.6270467260365329
# RAPminus0 AUC:  0.6237099652483581
# RAPminus1 AUC:  0.6237527625134083
# RAPminus2 AUC:  0.6467448285144056
# HREF AUC:       0.7068605497213987
# SREF AUC:       0.59662471719963
# Float32[0.16820262, -0.698822, 0.5922206, 0.16202259, 0.031716842, 0.39105946, 0.7149798, -0.2696285, 0.8772054]]]7]0817]
# Fit logistic coefficients: Float32[0.16820262, -0.698822, 0.5922206, 0.16202259, 0.031716842, 0.39105946, 0.7149798, -0.2696285, 0.8772054]
# Mean logistic_ŷ: 0.049831573
# Logistic logloss: 0.18342927
# Logistic AUC: 0.7303982759524867

println(bins_logistic_coeffs)

# Array{Float32,1}[[0.2958114, 0.093472, 0.014974893, 0.25723657, -0.08649167, -0.03651152, 0.51546174, -0.17759001, -0.9377826], [0.55761576, -0.08032898, 0.029375209, -0.043275304, 0.2413465, -0.029677542, 0.5895104, -0.27158892, -0.28064802], [0.5240835, -0.15477853, 0.30952054, -0.16907471, 0.26656932, 0.04274331, 0.7430586, -0.30433056, 0.9741227], [0.4446033, -0.31768143, 0.402504, 0.055888213, 0.10179166, 0.26728168, 0.8905479, -0.28445807, 2.323705], [0.16820262, -0.698822, 0.5922206, 0.16202259, 0.031716842, 0.39105946, 0.7149798, -0.2696285, 0.8772054]]


# just for fun
coeffs = LogisticRegression.fit(logit.(X), y, weights; iteration_count = 300)
println("Fit logistic coefficients: $(coeffs)")
println("Logistic AUC: $(Metrics.roc_auc(LogisticRegression.predict(X, coeffs), y, weights))");
# Fit logistic coefficients: Float32[0.39407995, -0.083142385, 0.1771561, 0.10479377, 0.0016176531, 0.06640092, 0.5703708, -0.2080386, 0.1512659]
# Logistic AUC: 0.9809899045778002


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)



import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHRRRRAPHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import ForecastCombinators


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts(); just_hours_near_storm_events = false);

length(combined_validation_forecasts) # 2961

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) # Expected: 2636
# 2636

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])


ForecastCombinators.turn_forecast_caching_on()
X, y, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; save_dir = "combined_validation_forecasts");
ForecastCombinators.clear_cached_forecasts()


ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 11478.342f0

bin_count = 20
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count # 573.9171f0

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
# Float32[9.96191f-5, 0.00040074784, 0.00074802537, 0.0013861385, 0.0023897155, 0.00345245, 0.0050529176, 0.007601612, 0.010993889, 0.016017333, 0.022238974, 0.02969854, 0.038574178, 0.047355093, 0.05609508, 0.06658636, 0.07913155, 0.09622029, 0.124933906, 1.0]

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
# 7.046450725111844e-6    7.276030260643027e-6    8.149382910901582e7     9.96191e-5
# 0.00018569760664097521  0.0001992050167294557   3.0917846637243032e6    0.00040074784
# 0.0007083221273481578   0.000547043320881087    811604.7255518436       0.00074802537
# 0.001004968604574856    0.0010154845111336255   571666.5000259876       0.0013861385
# 0.001615284805189093    0.0018141624219989128   355754.3150098324       0.0023897155
# 0.003243819902635385    0.002864499934662217    177011.33592289686      0.00345245
# 0.00417040778502528     0.004160083251711094    137680.79975754023      0.0050529176
# 0.005494325888930752    0.006165368414754303    104514.26054441929      0.007601612
# 0.008520016266732528    0.009110663996769706    67447.08611220121       0.010993889
# 0.011208618890038251    0.013233758175118635    51217.874188542366      0.016017333
# 0.01617714764916813     0.018841202709561244    35490.04291969538       0.022238974
# 0.022507225720713116    0.02567859833427495     25499.464930951595      0.02969854
# 0.02938825999561008     0.03379664949075832     19539.090526640415      0.038574178
# 0.04242381279651925     0.04275958996910579     13547.885240018368      0.047355093
# 0.05654047274437263     0.0515019379424905      10162.466443002224      0.05609508
# 0.06651922052078427     0.06104852031037582     8634.96290642023        0.06658636
# 0.07956938515208833     0.0724341302610855      7214.635666847229       0.07913155
# 0.09561960571668754     0.08699347351125025     6008.735875487328       0.09622029
# 0.11719265615162906     0.1084831716212397      4897.901657998562       0.124933906
# 0.14299085635775274     0.16909156607273287     3958.068720936775       1.0

Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9830358055177714


