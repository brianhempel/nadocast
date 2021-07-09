import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts; save_dir = "validation_forecasts_href_newer_with_blurs_and_forecast_hour");

function try_combine(combiner)
  ŷ = map(i -> combiner(X[i, :]), 1:length(y))

  sum(logloss.(y, ŷ) .* weights) / sum(weights)
end

println( try_combine(minimum) )
println( try_combine(maximum) )
println( try_combine(x -> x[2]) )
println( try_combine(x -> x[1]) )
println( try_combine(x -> 0.8f0*x[1] + 0.2f0*x[2]) )
println( try_combine(x -> 0.9f0*x[1] + 0.1f0*x[2]) ) # best so far, via logloss; best for AUC is 0.8 and 0.2

# 3. bin HREF predictions into 10 bins of equal weight of positive labels

ŷ = map(i -> X[i, 1], 1:length(y));

sort_perm = sortperm(ŷ);
X       = X[sort_perm, :]
y       = y[sort_perm]
ŷ       = ŷ[sort_perm]
weights = weights[sort_perm]

# total_logloss = sum(logloss.(y, ŷ) .* weights)
total_positive_weight = sum(y .* weights)

bin_count = 10
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


# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)

bins_logistic_coeffs = []

# Paired, overlapping bins
for bin_i in 1:(bin_count - 1)
  bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
  bin_max = bins_max[bin_i+1]

  bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

  bin_X       = X[bin_members]
  bin_ŷ       = ŷ[bin_members]
  bin_y       = y[bin_members]
  bin_weights = weights[bin_members]
  bin_weight  = sum(bin_weights)

  println("Bin $bin_i-$(bin_i+1) --------")
  println("$(bin_min) < HREF_ŷ <= $(bin_max)")
  println("Data count: $(length(y))")
  println("Weight: $(sum(bin_weights))")
  println("Mean HREF_ŷ: $(sum(@view bin_X[:,1] .* bin_weights) / bin_weight)")
  println("Mean SREF_ŷ: $(sum(@view bin_X[:,2] .* bin_weights) / bin_weight)")
  println("Mean y:      $(sum(bin_y .* bin_weights) / bin_weight)")
  println("HREF logloss: $(sum(logloss.(y, @view bin_X[:,1]) .* bin_weights) / bin_weight)")
  println("SREF logloss: $(sum(logloss.(y, @view bin_X[:,2]) .* bin_weights) / bin_weight)")

  # logit(HREF), logit(SREF), HREF*SREF, logit(HREF*SREF), max(logit(HREF),logit(SREF)), min(logit(HREF),logit(SREF))
  bin_X_features = Array{Float32}(undef, (length(bin_y), 5))

  Threads.@threads for i in 1:length(y)
    logit_href = logit(bin_X[i,1])
    logit_sref = logit(bin_X[i,2])

    bin_X_features[i,1] = logit_href
    bin_X_features[i,2] = logit_sref
    bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
    bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
    bin_X_features[i,5] = max(logit_href, logit_sref)
    bin_X_features[i,6] = min(logit_href, logit_sref)
  end

  coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights)

  println("Fit logistic coefficients: $(coeffs)")

  logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

  println("Mean logistic_ŷ: $(sum(logistic_ŷ .* bin_weights) / bin_weight)")
  println("Logistic logloss: $(sum(logloss.(y, logistic_ŷ) .* bin_weights) / bin_weight)")

  push!(bins_logistic_coeffs, coeffs)
end


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)




# n_folds = 3 # Choose something not divisible by 7, they're already partitioned by that
# folds = map(1:n_folds) do n
#   fold_forecasts = filter(forecast -> Forecasts.valid_time_in_convective_days_since_epoch_utc(forecast) % n_folds == n-1, validation_forecasts2)
#   fold_X, fold_y, fold_weights = TrainingShared.get_data_labels_weights(fold_forecasts; save_dir = "validation_forecasts_href_newer_blurred_and_hour_climatology_fold_$(n)");
#   (X = fold_X, y = fold_y, weights = fold_weights)
# end

