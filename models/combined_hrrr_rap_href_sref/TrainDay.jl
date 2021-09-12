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
import ForecastCombinators
import PlotMap
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHRRRRAPHREFSREF.forecasts_day_accumulators(); just_hours_near_storm_events = false);

length(validation_forecasts) # 207

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 184

validation_forecasts_10z_14z = filter(forecast -> forecast.run_hour == 10 || forecast.run_hour == 14, validation_forecasts);
length(validation_forecasts_10z_14z) # 184 # same as above because rn we only have the appropriate HRRRs for 10z 14z

ForecastCombinators.turn_forecast_caching_on()
# ForecastCombinators.turn_forecast_gc_circumvention_on()
@time Forecasts.data(validation_forecasts_10z_14z[10]); # Check if a forecast loads
# 1545.059279 seconds (247.87 M allocations: 1.442 TiB, 43.71% gc time)
# With GC circumvention: 1715.983475 seconds (247.65 M allocations: 1.442 TiB, 45.75% gc time)


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
# ForecastCombinators.turn_forecast_gc_circumvention_on()
X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts_10z_14z; save_dir = "day_accumulators_validation_forecasts_10z_14z", compute_forecast_labels = compute_forecast_labels);
ForecastCombinators.clear_cached_forecasts()


# should do some checks here.

Forecasts.time_title(validation_forecasts_10z_14z[169]) # "2020-08-29 10Z +25"
aug29 = validation_forecasts_10z_14z[169];
aug29_data = Forecasts.data(aug29);
PlotMap.plot_debug_map("aug29_10z_day_accs_1", aug29.grid, aug29_data[:,1]);
PlotMap.plot_debug_map("aug29_10z_day_accs_2", aug29.grid, aug29_data[:,2]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/aug29_10z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/aug29_10z_day_accs_2.pdf ./

aug29_labels = compute_forecast_labels(aug29);
PlotMap.plot_debug_map("aug29_10z_day_tornadoes", aug29.grid, aug29_labels);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/aug29_10z_day_tornadoes.pdf ./

# july11 = validation_forecasts_10z_14z[78]; Forecasts.time_title(july11) # "2020-07-11 00Z +35"
# PlotMap.plot_debug_map("july11_0z_day_tornadoes", july11.grid, compute_forecast_labels(july11));
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/july11_0z_day_tornadoes.pdf ./
# july11_data = Forecasts.data(july11);
# PlotMap.plot_debug_map("july11_0z_day_accs_1", july11.grid, july11_data[:,1]);
# PlotMap.plot_debug_map("july11_0z_day_accs_2", july11.grid, july11_data[:,2]);
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/july11_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_hrrr_rap_href_sref/july11_0z_day_accs_2.pdf ./


Metrics.roc_auc((@view X[:,1]), y, weights) # 0.9750816398134108
Metrics.roc_auc((@view X[:,2]), y, weights) # 0.9743185918044269

Metrics.roc_auc((@view X[:,1]) .+ (@view X[:,2]), y, weights) # 0.9750095149631679
Metrics.roc_auc((@view X[:,1]) .* (@view X[:,2]), y, weights) # 0.9748560155404963
Metrics.roc_auc((@view X[:,1]).^2 .* (@view X[:,2]), y, weights) # 0.974970653133173


# 3. bin predictions into 6 bins of equal weight of positive labels

ŷ = X[:,1]; # indep_events_ŷ

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]; #
y       = y[sort_perm];
ŷ       = ŷ[sort_perm];
weights = weights[sort_perm];

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights) # 11190.123f0

bin_count = 6
# per_bin_logloss = total_logloss / bin_count
per_bin_pos_weight = total_positive_weight / bin_count # 1865.0205f0

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
# Float32[0.01005014, 0.029228631, 0.07709654, 0.16022275, 0.27256465, 1.0]

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
# 0.0003191580597977087   0.0004332917027417096   5.843902203816056e6     0.01005014
# 0.016356833477520344    0.01685010109246095     114069.47458165884      0.029228631
# 0.03673086894963807     0.046837177305630785    50789.80803579092       0.07709654
# 0.09015270720128846     0.11164417863639488     20695.623482227325      0.16022275
# 0.1722911548678199      0.20809881728055737     10825.711207389832      0.27256465
# 0.29775838445694713     0.35620784405517136     6255.689407706261       1.0


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
# Array{Float32,1}[[1.1575506, -0.050190955, 0.29937217], [0.781756, 5.8343867f-5, -0.90717673], [1.0664908, -0.11734534, -0.5342406], [1.0162495, -0.06751505, -0.4148333], [1.0685753, -0.24417035, -0.76769584]]



# 6. prediction is weighted mean of the two overlapping logistic models


# 7. predictions should thereby be calibrated (check)

ŷ_absolutely_calibrated =
  CombinedHRRRRAPHREFSREF.make_combined_prediction(
    X;
    first_guess_feature_i = 1,
    bin_maxes             = Float32[0.01005014, 0.029228631, 0.07709654, 0.16022275, 0.27256465, 1.0],
    bins_logistic_coeffs  = Vector{Float32}[[1.1575506, -0.050190955, 0.29937217], [0.781756, 5.8343867f-5, -0.90717673], [1.0664908, -0.11734534, -0.5342406], [1.0162495, -0.06751505, -0.4148333], [1.0685753, -0.24417035, -0.76769584]]
  );

Metrics.roc_auc(ŷ_absolutely_calibrated, y, weights) # 0.9750841124257028

ŷ = ŷ_absolutely_calibrated;

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
# 0.00010284110043995907  0.00010572652295101056  5.446894796482623e6     0.0011950788
# 0.0021146208492433486   0.0021478412875673755   264858.8125824332       0.0037842996
# 0.0051960905486618724   0.005314976502329646    107704.89507895708      0.0074427454
# 0.009775968877650781    0.009406125490504043    57247.54726266861       0.011904001
# 0.015339870984524913    0.0141080502349321      36530.28287798166       0.016644988
# 0.019555650924999186    0.019215043593932984    28615.720175862312      0.022021696
# 0.02349969266280992     0.024653811937666217    23833.006240546703      0.027710585
# 0.0342538067213098      0.03082056058679128     16336.551253437996      0.034530364
# 0.037530363535884285    0.03933639676480956     14917.532215714455      0.04509478
# 0.04707618171300135     0.0520973040502739      11904.184662997723      0.06116276
# 0.0744770172774424      0.06891024143057069     7516.384555518627       0.07784144
# 0.08565805374513213     0.08820425391585056     6532.727254927158       0.09993576
# 0.1099525586887527      0.1111847945113449      5096.132573068142       0.12306256
# 0.13595345277366466     0.1342071536702206      4116.949266374111       0.146957
# 0.16522762178441663     0.15943438713183958     3388.5922436118126      0.17293704
# 0.1983603214771481      0.1865050419913375      2823.00670927763        0.20116605
# 0.19985800916747004     0.21802355341168936     2802.3515010476112      0.23558396
# 0.240871016659729       0.2530661370602951      2325.0548011660576      0.27437007
# 0.34357171309214357     0.295000083304399       1628.5017993450165      0.323545
# 0.3765266454417768      0.3839909304383223      1465.480993270874       1.0



# Calibrate to SPC (see models/spc_outlooks/Stats.jl for SPC stats)

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
# 0.02    0.010013118     0.048535860767389596
# 0.05    0.03852339      0.11142474987211093
# 0.1     0.13697089      0.22374267282134933
# 0.15    0.25369757      0.33315036263231873
# 0.3     0.52313256      0.0

println("threshold\tsuccess_ratio\tweight")
for threshold in 0.25f0:0.01f0:0.53f0
  println("$threshold\t$(success_ratio(ŷ, y, weights, threshold))\t$(sum(weights[ŷ .>= threshold]))")
end

# 0.25    0.3295237608815819      4391.211
# 0.26    0.3417913234714222      3818.8374
# 0.27    0.35255703081156536     3279.7898
# 0.28    0.365525305381746       2826.877
# 0.29    0.37473195691358885     2389.3037
# 0.3     0.3802889554730957      2034.7942
# 0.31    0.376269246592432       1774.4778
# 0.32    0.3772915246673419      1544.5347
# 0.33    0.3769892556480005      1353.4761
# 0.34    0.38253726113299186     1177.3013
# 0.35    0.3939671014569983      1020.4347
# 0.36    0.40490059495758024     871.9632
# 0.37    0.4079316397583762      759.05865
# 0.38    0.40624745113047406     645.8125
# 0.39    0.4082167694373917      552.8268   ***** we'll use this number
# 0.4     0.40156209001509857     475.1206
# 0.41    0.3980749598107955      399.55606
# 0.42    0.3957194555008884      338.22946
# 0.43    0.3619898886228898      268.51706
# 0.44    0.35169536578763616     210.03757
# 0.45    0.34624858738829056     174.03833
# 0.46    0.35180907726809457     140.91675
# 0.47    0.31320516793092584     111.74357
# 0.48    0.2974165432103353      71.896194
# 0.49    0.3002384721685388      48.584827
# 0.5     0.3799579891035744      28.18682
# 0.51    0.44580411523020147     8.734745
# 0.52    0.0     0.96780515
# 0.53    NaN     0.0

thresholds_to_match_success_ratio[5] = 0.39f0
thresholds_to_match_success_ratio
# 0.010013118
# 0.03852339
# 0.13697089
# 0.25369757
# 0.39


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
# 0.02    0.024865106     0.6726705661479192
# 0.05    0.09510009      0.40990829387439714
# 0.1     0.2420405       0.1403951157826877
# 0.15    0.36541605      0.02938601630678431
# 0.3     0.43843162      0.006861865473951445


println("nominal_prob\tthreshold_to_match_succes_ratio\tthreshold_to_match_POD\tmean_threshold\tsuccess_ratio\tPOD")

calibration = []

for i in 1:length(target_PODs)
  nominal_prob, _ = target_PODs[i]
  threshold_to_match_succes_ratio = thresholds_to_match_success_ratio[i]
  threshold_to_match_POD = thresholds_to_match_POD[i]
  mean_threshold = (threshold_to_match_succes_ratio + threshold_to_match_POD) * 0.5f0
  sr  = success_ratio(ŷ, y, weights, mean_threshold)
  pod = probability_of_detection(ŷ, y, weights, mean_threshold)
  push!(calibration, (nominal_prob, mean_threshold))
  println("$nominal_prob\t$threshold_to_match_succes_ratio\t$threshold_to_match_POD\t$mean_threshold\t$sr\t$pod")
end

# nominal_prob    threshold_to_match_succes_ratio threshold_to_match_POD  mean_threshold  success_ratio   POD
# 0.02    0.010013118     0.024865106     0.017439112     0.06472411905857228     0.7434338172141978
# 0.05    0.03852339      0.09510009      0.06681174      0.15533263366251956     0.48261538793030245
# 0.1     0.13697089      0.2420405       0.1895057       0.2629948044070861      0.21913907331601418
# 0.15    0.25369757      0.36541605      0.3095568       0.37767216707603646     0.06044300369330901
# 0.3     0.39    0.43843162      0.4142158       0.40010543678136457     0.013172743441568865


println(calibration)
calibration = [
  (0.02, 0.017439112),
  (0.05, 0.06681174),
  (0.1,  0.1895057),
  (0.15, 0.3095568),
  (0.3,  0.4142158)
]

