import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_accumulators(); just_hours_near_storm_events = false);

length(validation_forecasts) # 903

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 735

@time Forecasts.data(validation_forecasts[10]) # Check if a forecast loads

validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, validation_forecasts);
length(validation_forecasts_0z) # 92


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

compute_forecast_labels(forecast) = begin
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  # Annoying that we have to recalculate this.
  # The end_seconds will always be the last hour of the convective day
  # start_seconds depends on whether the run started during the day or not
  # I suppose for 0Z the answer is always "no" but whatev here's the right math
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 2*HOUR) - 30*MINUTE
  println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  utc_datetime = Dates.unix2datetime(start_seconds)
  println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_conus_tornado_neighborhoods(forecast.grid, TrainingShared.TORNADO_SPACIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts_0z; save_dir = "day_accumulators_validation_forecasts_0z", compute_forecast_labels = compute_forecast_labels);


# should do some checks here.

aug29 = validation_forecasts_0z[85]; Forecasts.time_title(aug29) # "2020-08-29 00Z +35"
aug29_data = Forecasts.data(aug29);
PlotMap.plot_debug_map("aug29_0z_day_accs_1", aug29.grid, aug29_data[:,1]);
PlotMap.plot_debug_map("aug29_0z_day_accs_2", aug29.grid, aug29_data[:,2]);
PlotMap.plot_debug_map("aug29_0z_day_accs_3", aug29.grid, aug29_data[:,3]);
PlotMap.plot_debug_map("aug29_0z_day_accs_7", aug29.grid, aug29_data[:,7]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_2.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_3.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_7.pdf ./

aug29_labels = compute_forecast_labels(aug29);
PlotMap.plot_debug_map("aug29_0z_day_tornadoes", aug29.grid, aug29_labels);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_tornadoes.pdf ./

july11 = validation_forecasts_0z[78]; Forecasts.time_title(july11) # "2020-07-11 00Z +35"
PlotMap.plot_debug_map("july11_0z_day_tornadoes", july11.grid, compute_forecast_labels(july11));
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_tornadoes.pdf ./
july11_data = Forecasts.data(july11);
PlotMap.plot_debug_map("july11_0z_day_accs_1", july11.grid, july11_data[:,1]);
PlotMap.plot_debug_map("july11_0z_day_accs_2", july11.grid, july11_data[:,2]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_2.pdf ./

(_, all_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_accumulators(); just_hours_near_storm_events = false);

length(all_validation_forecasts) # 903

all_validation_forecast_0z = filter(forecast -> forecast.run_hour == 0, all_validation_forecasts);
length(all_validation_forecast_0z) # 113

Forecasts.time_title(all_validation_forecast_0z[113]) # "2021-08-07 00Z +35"
aug7_21 = all_validation_forecast_0z[113];
aug7_21_data = Forecasts.data(aug7_21);
PlotMap.plot_debug_map("aug7_21_0z_day_accs_1", aug7_21.grid, aug7_21_data[:,1]);
PlotMap.plot_debug_map("aug7_21_0z_day_accs_2", aug7_21.grid, aug7_21_data[:,2]);
PlotMap.plot_debug_map("aug7_21_0z_day_accs_1times2", aug7_21.grid, aug7_21_data[:,1] .* aug7_21_data[:,2]);
PlotMap.plot_debug_map("aug7_21_0z_day_accs_1times2_w1", aug7_21.grid, (aug7_21_data[:,1] .^ 2 .* aug7_21_data[:,2]) .^ (1f0/3f0));
PlotMap.plot_debug_map("aug7_21_0z_day_accs_sqrt1times2", aug7_21.grid, sqrt.(aug7_21_data[:,1] .* aug7_21_data[:,2]));
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug7_21_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug7_21_0z_day_accs_2.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug7_21_0z_day_accs_1times2.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug7_21_0z_day_accs_1times2_w1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug7_21_0z_day_accs_sqrt1times2.pdf ./


Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9702920184229288
Metrics.roc_auc((@view X[:,2]), y, weights) # 0.9702491120791028
Metrics.roc_auc((@view X[:,3]), y, weights) # 0.970192917641865
Metrics.roc_auc((@view X[:,7]), y, weights) # 0.9676332184207135

Metrics.roc_auc((@view X[:,1]) .+ (@view X[:,2]), y, weights) # 0.9703356900528725
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,2]), y, weights) # 0.97050115490524
Metrics.roc_auc((@view X[:,1]).^2 .* (@view X[:,2]), y, weights) # 0.9704892805445616 # even weighting seems to be best


# 3. bin predictions into 6 bins of equal weight of positive labels

ŷ = X[:,1]; # indep_events_ŷ

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 5821.808f0

total_max_hour = sum(X[:,2] .* weights) # 1653.1519f0
max_hour_multipler = total_positive_weight / total_max_hour

bin_count = 6
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count # 970.30133f0

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
# bins_max = Float32[0.008833055, 0.025307992, 0.06799701, 0.11479675, 0.18474162, 1.0]

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
# 0.000333467112118504    0.00041960028949406604  2.9117239340932965e6    0.008833055
# 0.013936748693091678    0.014909639502841415    69664.75792533159       0.025307992
# 0.027251130651393677    0.041216979264066575    35606.99930393696       0.06799701
# 0.08142417377185326     0.08795948971931956     11917.636298596859      0.11479675
# 0.15839284952354146     0.14423929368937344     6129.9213426709175      0.18474162
# 0.2039969708265159      0.2702648722874661      4746.603324890137       1.0


# 4. combine bin-pairs (overlapping, 5 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms

bins_logistic_coeffs = []

# Paired, overlapping bins
for bin_i in 1:(bin_count - 1)
  global bins_logistic_coeffs

  bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
  bin_max = bins_max[bin_i+1]

  bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

  bin_X       = X[bin_members,:]
  bin_ŷ       = ŷ[bin_members]
  bin_y       = y[bin_members]
  bin_weights = weights[bin_members]
  bin_weight  = Float32(bins_Σweight[bin_i] + bins_Σweight[bin_i+1])

  println("Bin $bin_i-$(bin_i+1) --------")
  println("$(bin_min) < indep_events_ŷ <= $(bin_max)")
  println("Data count: $(length(bin_y))")
  println("Positive count: $(sum(bin_y))")
  println("Weight: $(bin_weight)")
  println("Mean indep_events_ŷ: $(sum((@view bin_X[:,1]) .* bin_weights) / bin_weight)")
  println("Mean y:      $(sum(bin_y .* bin_weights) / bin_weight)")
  println("indep_events_ŷ logloss: $(sum(logloss.(bin_y, (@view bin_X[:,1])) .* bin_weights) / bin_weight)")
  println("indep_events_ŷ AUC: $(Metrics.roc_auc((@view bin_X[:,1]), bin_y, bin_weights))")
  println("max_hour AUC: $(Metrics.roc_auc((@view bin_X[:,2]), bin_y, bin_weights))")

  bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

  Threads.@threads for i in 1:length(bin_y)
    bin_X_features[i,1] = logit(bin_X[i,1])
    bin_X_features[i,2] = logit(bin_X[i,2])
    # bin_X_features[i,3] = logit(sqrt.(bin_X[i,1] .* bin_X[i,2] .* max_hour_multipler))
    # bin_X_features[i,3] = logit(bin_X[i,3])
    # bin_X_features[i,4] = logit(bin_X[i,4])
    # bin_X_features[i,5] = logit(bin_X[i,5])
    # bin_X_features[i,6] = logit(bin_X[i,6])
    # bin_X_features[i,7] = logit(bin_X[i,7])
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
# -1.0 < indep_events_ŷ <= 0.025307992
# Data count: 3260245
# Positive count: 2106.0
# Weight: 2.9813888e6
# Mean indep_events_ŷ: 0.00075818243
# Mean y:      0.0006513288
# indep_events_ŷ logloss: 0.004137885
# indep_events_ŷ AUC: 0.936794428389541
# max_hour AUC: 0.9373933956547336
# Fit logistic coefficients: Float32[0.8790791, 0.17466258, 0.42071092]
# Mean logistic_ŷ: 0.0006513288
# Logistic logloss: 0.0041245907
# Logistic AUC: 0.9372048039156227
# Bin 2-3 --------
# 0.008833055 < indep_events_ŷ <= 0.06799701
# Data count: 113052
# Positive count: 2085.0
# Weight: 105271.76
# Mean indep_events_ŷ: 0.023807803
# Mean y:      0.01844019
# indep_events_ŷ logloss: 0.090819865
# indep_events_ŷ AUC: 0.6317487654364228
# max_hour AUC: 0.6255392540142458
# Fit logistic coefficients: Float32[0.5856237, 0.17571865, -0.84646785]
# Mean logistic_ŷ: 0.01844019
# Logistic logloss: 0.08993409
# Logistic AUC: 0.6323592711366193
# Bin 3-4 --------
# 0.025307992 < indep_events_ŷ <= 0.11479675
# Data count: 50583
# Positive count: 2056.0
# Weight: 47524.637
# Mean indep_events_ŷ: 0.05293848
# Mean y:      0.040835973
# indep_events_ŷ logloss: 0.16468208
# indep_events_ŷ AUC: 0.6775749703489325
# max_hour AUC: 0.6462210887399996
# Fit logistic coefficients: Float32[1.4197825, 0.00979548, 0.91951996]
# Mean logistic_ŷ: 0.040835973
# Logistic logloss: 0.16236585
# Logistic AUC: 0.6776229005704377
# Bin 4-5 --------
# 0.06799701 < indep_events_ŷ <= 0.18474162
# Data count: 18954
# Positive count: 2033.0
# Weight: 18047.559
# Mean indep_events_ŷ: 0.10707513
# Mean y:      0.10756687
# indep_events_ŷ logloss: 0.33271375
# indep_events_ŷ AUC: 0.6324209720505071
# max_hour AUC: 0.5747030265464277
# Bin 4-5 --------
# 0.06799701 < indep_events_ŷ <= 0.18474162
# Data count: 18954
# Positive count: 2033.0
# Weight: 18047.559
# Mean indep_events_ŷ: 0.10707513
# Mean y:      0.10756687
# indep_events_ŷ logloss: 0.33271375
# indep_events_ŷ AUC: 0.6324209720505071
# max_hour AUC: 0.5747030265464277
# Fit logistic coefficients: Float32[1.5459903, -0.15001805, 0.5937435]
# Mean logistic_ŷ: 0.10756687
# Logistic logloss: 0.33181626
# Logistic AUC: 0.6336697730176891
# Bin 5-6 --------
# 0.11479675 < indep_events_ŷ <= 1.0
# Data count: 11292
# Positive count: 2016.0
# Weight: 10876.524
# Mean indep_events_ŷ: 0.19923788
# Mean y:      0.17829485
# indep_events_ŷ logloss: 0.46269885
# indep_events_ŷ AUC: 0.5782792292499737
# max_hour AUC: 0.5441913787386747
# Fit logistic coefficients: Float32[1.1762913, -0.5233394, -1.4415414]
# Mean logistic_ŷ: 0.17829485
# Logistic logloss: 0.45726532
# Logistic AUC: 0.5826572935419169

println(bins_logistic_coeffs)
# Any[Float32[0.8790791, 0.17466258, 0.42071092], Float32[0.5856237, 0.17571865, -0.84646785], Float32[1.4197825, 0.00979548, 0.91951996], Float32[1.5459903, -0.15001805, 0.5937435], Float32[1.1762913, -0.5233394, -1.4415414]]


# 6. prediction is weighted mean of the two overlapping logistic models

# Metrics.roc_auc((@view X_day[:,1]), y, weights) # 0.9704562360340072

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


(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts) # Expected:
#

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
length(day_validation_forecasts_0z) # Expected: 92
#

X, y, weights = TrainingShared.get_data_labels_weights(day_validation_forecasts_0z; save_dir = "day_validation_forecasts_0z");


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



# Expected: 0.9704562360340072
Metrics.roc_auc((@view X[:,1]), y, weights) #






# Calibrate to SPC