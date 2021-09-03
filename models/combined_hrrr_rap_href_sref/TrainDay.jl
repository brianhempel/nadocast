import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHRRRRAPHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHRRRRAPHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_day_accumulators(); just_hours_near_storm_events = false);

length(validation_forecasts) #

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) #

@time Forecasts.data(validation_forecasts[10]) # Check if a forecast loads

validation_forecasts_10z_14z = filter(forecast -> forecast.run_hour == 10 || forecast.run_hour == 14, validation_forecasts);
length(validation_forecasts_10z_14z) #


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
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 2*HOUR) - 30*MINUTE
  println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  utc_datetime = Dates.unix2datetime(start_seconds)
  println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_conus_tornado_neighborhoods(forecast.grid, TrainingShared.TORNADO_SPACIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

ForecastCombinators.turn_forecast_caching_on()
X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts_10z_14z; save_dir = "day_accumulators_validation_forecasts_10z_14z", compute_forecast_labels = compute_forecast_labels);
ForecastCombinators.clear_cached_forecasts()


# should do some checks here.

# aug29 = validation_forecasts_10z_14z[85]; Forecasts.time_title(aug29) # "2020-08-29 00Z +35"
# aug29_data = Forecasts.data(aug29);
# PlotMap.plot_debug_map("aug29_0z_day_accs_1", aug29.grid, aug29_data[:,1]);
# PlotMap.plot_debug_map("aug29_0z_day_accs_2", aug29.grid, aug29_data[:,2]);
# PlotMap.plot_debug_map("aug29_0z_day_accs_3", aug29.grid, aug29_data[:,3]);
# PlotMap.plot_debug_map("aug29_0z_day_accs_7", aug29.grid, aug29_data[:,7]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_2.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_3.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_7.pdf ./

# aug29_labels = compute_forecast_labels(aug29);
# PlotMap.plot_debug_map("aug29_0z_day_tornadoes", aug29.grid, aug29_labels);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_tornadoes.pdf ./

# july11 = validation_forecasts_10z_14z[78]; Forecasts.time_title(july11) # "2020-07-11 00Z +35"
# PlotMap.plot_debug_map("july11_0z_day_tornadoes", july11.grid, compute_forecast_labels(july11));
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_tornadoes.pdf ./
# july11_data = Forecasts.data(july11);
# PlotMap.plot_debug_map("july11_0z_day_accs_1", july11.grid, july11_data[:,1]);
# PlotMap.plot_debug_map("july11_0z_day_accs_2", july11.grid, july11_data[:,2]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_2.pdf ./


Metrics.roc_auc((@view X[:,1]), y, weights) #
Metrics.roc_auc((@view X[:,2]), y, weights) #

Metrics.roc_auc((@view X[:,1]) .+ (@view X[:,2]), y, weights) #
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,2]), y, weights) #
Metrics.roc_auc((@view X[:,1]).^2 .* (@view X[:,2]), y, weights) #


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
#

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


# 4. combine bin-pairs (overlapping, 5 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms

bins_logistic_coeffs = Vector{Float32}[]

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



println(bins_logistic_coeffs)
#


# 6. prediction is weighted mean of the two overlapping logistic models


# 7. predictions should thereby be calibrated (check)



import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHRRRRAPHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_day(); just_hours_near_storm_events = false);

length(day_validation_forecasts)
# 903

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts) # Expected: 735
# 735

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

validation_forecasts_10z_14z = filter(forecast -> forecast.run_hour == 10 || forecast.run_hour == 14, validation_forecasts);
length(day_validation_forecasts_10z_14z) # Expected: 92
# 92

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

ForecastCombinators.turn_forecast_caching_on()
X, y, weights = TrainingShared.get_data_labels_weights(day_validation_forecasts_10z_14z; save_dir = "day_validation_forecasts_10z_14z", compute_forecast_labels = compute_forecast_labels);
ForecastCombinators.clear_cached_forecasts()


ŷ = X[:,1];

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 5821.8076f0

bin_count = 20
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count # 291.0904f0

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
# Float32[0.00097577664, 0.0035424125, 0.0068334113, 0.010702337, 0.014457697, 0.018155066, 0.021734202, 0.027651712, 0.03753781, 0.05559306, 0.076415844, 0.0933851, 0.10967304, 0.12784216, 0.14595553, 0.1621963, 0.18518762, 0.2340388, 0.30211022, 1.0]Float32[0.0070357043, 0.01022332, 0.021027338, 0.023034172, 0.025326166, 0.0274165, 0.02900777, 0.030673556, 0.032790024, 0.03441065, 0.037875757, 0.04150515, 0.047454756, 0.058034927, 0.073863976, 0.08908757, 0.12828249, 0.17552827, 1.0, 1.0]

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
# 0.0001089659531147851   0.0001039783132379927   2.6742326298045516e6    0.00097577664
# 0.001760340660025297    0.0019026741997471292   165492.52153635025      0.0035424125
# 0.0048276268251249395   0.004946116178554242    60406.159088253975      0.0068334113
# 0.008558151635295863    0.008590979826240427    34042.4254655242        0.010702337
# 0.0133678045274549      0.012463140159311574    21802.550265967846      0.014457697
# 0.016984762764517724    0.016286686422954007    17190.005229771137      0.018155066
# 0.01987269476873383     0.019829021253987382    14691.411134302616      0.021734202
# 0.023482472772874543    0.024303937577457836    12433.217101633549      0.027651712
# 0.029936281457548806    0.03196438758504661     9738.050294101238       0.03753781
# 0.041338372255156755    0.04526684442033121     7049.755621552467       0.05559306
# 0.05315587254576701     0.06531589970701189     5476.31123059988        0.076415844
# 0.08852993524984791     0.08458178404722025     3297.176026880741       0.0933851
# 0.12224831704323326     0.1010973370449811      2385.681490957737       0.10967304
# 0.14548918026167934     0.11838054484039393     2006.3990591168404      0.12784216
# 0.15331280352287868     0.13717977935837336     1900.389076769352       0.14595553
# 0.14773744775283826     0.15395207248511505     1971.3568314313889      0.1621963
# 0.15843580980211655     0.17308492055260047     1840.2607005238533      0.18518762
# 0.13703903462860317     0.20657700500765952     2126.0651206970215      0.2340388
# 0.2538073639221181      0.2602755820478401      1148.4315833449364      0.30211022
# 0.5049433181045647      0.34571517923806827     559.0556263923645       1.0




# Expected: 0.9704562360340072
Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9704562387563738




# Calibrate to SPC

function success_ratio(ŷ, y, weights, threshold)
  painted_weight       = 0.0
  true_positive_weight = 0.0

  for i in 1:length(y)
    if ŷ[i] >= threshold
      painted_weight += Float64(weights[i])
      if y[i] > 0.5f0
        true_positive_weight += Float64(weights[i])
      end
    end
  end

  true_positive_weight / painted_weight
end

target_success_ratios = [
  (0.02, 0.0485357084712472),
  (0.05, 0.11142399542325385),
  (0.1, 0.22373785045573905),
  (0.15, 0.33311809995812625),
  (0.3, 0.42931275151328113)
]

println("nominal_prob\tthreshold\tsuccess_ratio")

thresholds_to_match_success_ratio =
  map(target_success_ratios) do (nominal_prob, target_success_ratio)
    threshold = 0.5f0
    step = 0.25f0
    while step > 0.00000001f0
      sr = success_ratio(ŷ, y, weights, threshold)
      if isnan(sr) || sr > target_success_ratio
        threshold -= step
      else
        threshold += step
      end
      step *= 0.5f0
    end
    println("$nominal_prob\t$threshold\t$(success_ratio(ŷ, y, weights, threshold))")
    threshold
  end

# nominal_prob    threshold       success_ratio
# 0.02    0.0127295405    0.04853525616997882
# 0.05    0.040039346     0.11142667300267296
# 0.1     0.18400295      0.2237468737640593
# 0.15    0.23302902      0.33317687621379216
# 0.3     0.26344         0.42958206936270776


function probability_of_detection(ŷ, y, weights, threshold)
  positive_weight      = 0.0
  true_positive_weight = 0.0

  for i in 1:length(y)
    if y[i] > 0.5f0
      positive_weight += Float64(weights[i])
      if ŷ[i] >= threshold
        true_positive_weight += Float64(weights[i])
      end
    end
  end

  true_positive_weight / positive_weight
end

target_PODs = [
  (0.02, 0.672673334503246),
  (0.05, 0.4098841063979484),
  (0.1, 0.14041993254469878),
  (0.15, 0.029338579808637542),
  (0.3, 0.00679180457194558)
]

println("nominal_prob\tthreshold\tPOD")

thresholds_to_match_POD =
  map(target_PODs) do (nominal_prob, target_POD)
    threshold = 0.5f0
    step = 0.25f0
    while step > 0.00000001f0
      pod = probability_of_detection(ŷ, y, weights, threshold)
      if isnan(pod) || pod > target_POD
        threshold += step
      else
        threshold -= step
      end
      step *= 0.5f0
    end
    println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
    threshold
  end

# nominal_prob    threshold       POD
# 0.02    0.019777253     0.672573803962084
# 0.05    0.08982225      0.40978062116159975
# 0.1     0.19142316      0.14033469816696242
# 0.15    0.33357763      0.02943007176491364
# 0.3     0.3842491       0.006698613290046718

println("nominal_prob\tthreshold_to_match_succes_ratio\tthreshold_to_match_POD\tmean_threshold\tsuccess_ratio\tPOD")

for i in 1:length(target_PODs)
  nominal_prob, _ = target_PODs[i]
  threshold_to_match_succes_ratio = thresholds_to_match_success_ratio[i]
  threshold_to_match_POD = thresholds_to_match_POD[i]
  mean_threshold = (threshold_to_match_succes_ratio + threshold_to_match_POD) * 0.5f0
  sr  = success_ratio(ŷ, y, weights, mean_threshold)
  pod = probability_of_detection(ŷ, y, weights, mean_threshold)
  println("$nominal_prob\t$threshold_to_match_succes_ratio\t$threshold_to_match_POD\t$mean_threshold\t$sr\t$pod")
end

# nominal_prob    threshold_to_match_succes_ratio threshold_to_match_POD  mean_threshold  success_ratio   POD
# 0.02    0.0127295405    0.019777253     0.016253397     0.05632502012753043     0.7290002254123179
# 0.05    0.040039346     0.08982225      0.0649308       0.1382464577013378      0.47516603256458734
# 0.1     0.18400295      0.19142316      0.18771306      0.2310717260852238      0.145134634956839
# 0.15    0.23302902      0.33357763      0.28330332      0.4875050416418053      0.062033164663175995
# 0.3     0.26344         0.3842491       0.32384455      0.5220695437305608      0.036442226839582884

calibration = [
  (0.02, 0.016253397),
  (0.05, 0.0649308),
  (0.1,  0.18771306),
  (0.15, 0.28330332),
  (0.3,  0.32384455),
]