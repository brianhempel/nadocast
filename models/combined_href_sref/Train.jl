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


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 12464


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts;
    event_name_to_labeler = TrainingShared.event_name_to_labeler,
    save_dir = "validation_forecasts_href_newer"
  );

@assert length(HREFPrediction.models) == length(SREFPrediction.models)

event_types_count = length(HREFPrediction.models)


# Sanity check...tornado features should best predict tornadoes, etc
# and HREF should do best

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:event_types_count
    (event_name, _, _) = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    for j in 1:size(X,2)
      x = @view X[:,j]
      auc = Metrics.roc_auc(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAUC: $auc")
    end
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)



# 3. bin HREF predictions into 6 bins of equal weight of positive labels

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

bins_maxes = Dict{String,Vector{Float32}}()
for prediction_i in 1:event_types_count
  (event_name, _, _) = HREFPrediction.models[prediction_i]

  bins_maxes[event_name] = find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)

  println("bins_maxes[\"$event_name\"] = $(bins_maxes[event_name])")
end

# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + b)
# For the 2020 models, adding more terms resulted in dangerously large coefficients
# There's more data this year...try interaction terms this time?

function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,prediction_i]; # HREF prediction for event_name

  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_href_x  = X[bin_members, prediction_i]
    bin_sref_x  = X[bin_members, prediction_i + event_types_count]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
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
      ("mean_HREF_ŷ", sum((@view bin_X[:,1]) .* bin_weights) / bin_weight),
      ("mean_SREF_ŷ", sum((@view bin_X[:,2]) .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("HREF_logloss", sum(logloss.(bin_y, (@view bin_X[:,1])) .* bin_weights) / bin_weight),
      ("SREF_logloss", sum(logloss.(bin_y, (@view bin_X[:,2])) .* bin_weights) / bin_weight),
      ("HREF_auc", Metrics.roc_auc((@view bin_X[:,1]), bin_y, bin_weights)),
      ("SREF_auc", Metrics.roc_auc((@view bin_X[:,2]), bin_y, bin_weights)),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_auc", Metrics.roc_auc(logistic_ŷ, bin_y, bin_weights)),
      ("logistic_coeffs", coeffs)
    ]

    headers = map(first, stuff)
    row     = map(last, stuff)

    bin_i == 1 && println(join(headers, "\t"))
    println(join(row, "\t"))

    push!(bins_logistic_coeffs, coeffs)
  end
end

for prediction_i in 1:event_types_count
  (event_name, _, _) = HREFPrediction.models[prediction_i]

  find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end




# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)


# CHECKING

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







