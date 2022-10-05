import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 716

validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_forecasts);
length(validation_forecasts_0z_12z) # 358

@time Forecasts.data(validation_forecasts[10]); # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


compute_day_labels(events, forecast) = begin
  # Annoying that we have to recalculate this.
  # The end_seconds will always be the last hour of the convective day
  # start_seconds depends on whether the run started during the day or not
  # I suppose for 0Z the answer is always "no" but whatev here's the right math
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 2*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime = Dates.unix2datetime(start_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_day_labeler = Dict(
  "tornado" => (forecast -> compute_day_labels(StormEvents.conus_tornado_events(), forecast))
)

# rm("day_accumulators_validation_forecasts_0z_12z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts_0z_12z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_accumulators_validation_forecasts_0z_12z",
  );



# should do some checks here.
import PlotMap

dec11 = filter(f -> Forecasts.time_title(f) == "2021-12-11 00Z +35", validation_forecasts_0z_12z)[1];
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  prediction_i = div(i - 1, 2) + 1
  model_name, _ = HREFPredictionAblations.models[prediction_i]
  PlotMap.plot_debug_map("dec11_0z_12z_day_accs_$i_$model_name", dec11.grid, dec11_data[:,i]);
end
PlotMap.plot_debug_map("dec11_0z_12z_day_tornado", dec11.grid, event_name_to_day_labeler["tornado"](dec11));

# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_tornado.pdf ./


# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    model_name, _ = HREFPredictionAblations.models[prediction_i]
    y = Ys["tornado"]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z_12z, X, Ys, weights)

# tornadao_mean_58                                           (20606.0) feature 1 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13162558944528638
# tornadao_mean_58                                           (20606.0) feature 2 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.12449276362093863
# tornadao_prob_80                                           (20606.0) feature 3 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13009477393713162
# tornadao_prob_80                                           (20606.0) feature 4 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.11937844108793211
# tornadao_mean_prob_138                                     (20606.0) feature 5 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13792107012674604
# tornadao_mean_prob_138                                     (20606.0) feature 6 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.13220794055741025
# tornadao_mean_prob_computed_no_sv_219                      (20606.0) feature 7 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.1369357573174645
# tornadao_mean_prob_computed_no_sv_219                      (20606.0) feature 8 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.13028198157841184
# tornadao_mean_prob_computed_220                            (20606.0) feature 9 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.1358151410361583
# tornadao_mean_prob_computed_220                            (20606.0) feature 10 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13025419582526923
# tornadao_mean_prob_computed_partial_climatology_227        (20606.0) feature 11 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14589679961936527
# tornadao_mean_prob_computed_partial_climatology_227        (20606.0) feature 12 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13918545049521447
# tornadao_mean_prob_computed_climatology_253                (20606.0) feature 13 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.1476615184683567
# tornadao_mean_prob_computed_climatology_253                (20606.0) feature 14 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13952586413567675
# tornadao_mean_prob_computed_climatology_blurs_910          (20606.0) feature 15 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14386786713427835
# tornadao_mean_prob_computed_climatology_blurs_910          (20606.0) feature 16 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1358966592876078
# tornadao_mean_prob_computed_climatology_grads_1348         (20606.0) feature 17 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14783769477377198
# tornadao_mean_prob_computed_climatology_grads_1348         (20606.0) feature 18 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1396541560734088
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   (20606.0) feature 19 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15169494866457822
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   (20606.0) feature 20 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1420868134833007
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 (20606.0) feature 21 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15109026383083607
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 (20606.0) feature 22 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.14853730824111055
# tornadao_mean_prob_computed_climatology_3hr_1567           (20606.0) feature 23 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15952592485504877 ***best**
# tornadao_mean_prob_computed_climatology_3hr_1567           (20606.0) feature 24 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.15389499088128772 (second best)
# tornado_full_13831                                         (20606.0) feature 25 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15398281422151097 (third best)
# tornado_full_13831                                         (20606.0) feature 26 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.14738984560906757


# event_names is list of keys of Ys, one per column in Ŷ
Metrics.reliability_curves_midpoints(20, X, Ys, fill("tornado", size(X,2)), weights, map(i -> HREFPredictionAblations.models[div(i - 1, 2) + 1][1] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2)))



# 3. bin predictions into 4 bins of equal weight of positive labels

const bin_count = 4

function find_ŷ_bin_splits(event_name, model_name, ŷ, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

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

  for bin_i in 1:bin_count
    Σŷ      = bins_Σŷ[bin_i]
    Σy      = bins_Σy[bin_i]
    Σweight = bins_Σweight[bin_i]

    mean_ŷ = Σŷ / Σweight
    mean_y = Σy / Σweight

    println("$model_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
  end

  bins_max
end

nmodels = length(HREFPredictionAblations.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("model_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:nmodels
  model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[model_name] = find_ŷ_bin_splits("tornado", model_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# model_name                                                 mean_y        mean_ŷ        Σweight    bin_max
# tornadao_mean_58                                           0.0005525441  0.00064435945 8.74989e6  0.017304773
# tornadao_mean_58                                           0.026186043   0.030275125   184641.98  0.0553223
# tornadao_mean_58                                           0.07631268    0.084108755   63355.395  0.13471735
# tornadao_mean_58                                           0.1773909     0.24363838    27244.582  1.0
# tornadao_prob_80                                           0.00055105187 0.0007312233  8.773089e6 0.019274753
# tornadao_prob_80                                           0.029058345   0.033250812   166377.2   0.059863195
# tornadao_prob_80                                           0.08667382    0.086673744   55785.69   0.13051617
# tornadao_prob_80                                           0.16175802    0.22859141    29879.643  1.0
# tornadao_mean_prob_138                                     0.0005505699  0.0006848417  8.780765e6 0.019996958
# tornadao_mean_prob_138                                     0.029467909   0.034681678   164072.67  0.06293694
# tornadao_mean_prob_138                                     0.08824085    0.093168594   54791.113  0.14385402
# tornadao_mean_prob_138                                     0.18952441    0.25917518    25502.709  1.0
# tornadao_mean_prob_computed_no_sv_219                      0.0005502851  0.0006740746  8.786296e6 0.019619932
# tornadao_mean_prob_computed_no_sv_219                      0.030165914   0.034514353   160281.48  0.06317351
# tornadao_mean_prob_computed_no_sv_219                      0.0892953     0.09441178    54143.47   0.14767814
# tornadao_mean_prob_computed_no_sv_219                      0.19797844    0.2680879     24410.443  1.0
# tornadao_mean_prob_computed_220                            0.0005500478  0.00067421497 8.790041e6 0.020045973
# tornadao_mean_prob_computed_220                            0.031090373   0.035085127   155508.97  0.06332173
# tornadao_mean_prob_computed_220                            0.088539734   0.09354061    54611.535  0.14715679
# tornadao_mean_prob_computed_220                            0.19352877    0.26402542    24970.133  1.0
# tornadao_mean_prob_computed_partial_climatology_227        0.0005498372  0.00066107186 8.793964e6 0.019988786
# tornadao_mean_prob_computed_partial_climatology_227        0.03144185    0.03438385    153782.98  0.061916392
# tornadao_mean_prob_computed_partial_climatology_227        0.08897645    0.09635408    54334.35   0.16038308
# tornadao_mean_prob_computed_partial_climatology_227        0.20965223    0.2826052     23050.35   1.0
# tornadao_mean_prob_computed_climatology_253                0.0005492325  0.00068179093 8.803066e6 0.02144064
# tornadao_mean_prob_computed_climatology_253                0.033442535   0.03624927    144564.94  0.063346006
# tornadao_mean_prob_computed_climatology_253                0.08659002    0.100448      55831.27   0.16949469
# tornadao_mean_prob_computed_climatology_253                0.22305384    0.28914496    21669.77   1.0
# tornadao_mean_prob_computed_climatology_blurs_910          0.0005490907  0.00065349595 8.805662e6 0.02166239
# tornadao_mean_prob_computed_climatology_blurs_910          0.03356391    0.037512593   144040.5   0.06790343
# tornadao_mean_prob_computed_climatology_blurs_910          0.091271184   0.10380857    52967.86   0.16562855
# tornadao_mean_prob_computed_climatology_blurs_910          0.21518865    0.28201032    22461.203  1.0
# tornadao_mean_prob_computed_climatology_grads_1348         0.0005495181  0.00065491704 8.798646e6 0.020726241
# tornadao_mean_prob_computed_climatology_grads_1348         0.0316612     0.03734225    152710.67  0.06985498
# tornadao_mean_prob_computed_climatology_grads_1348         0.09449947    0.106237754   51163.77   0.16851029
# tornadao_mean_prob_computed_climatology_grads_1348         0.2137203     0.28244874    22611.502  1.0
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   0.00054904306 0.00064249634 8.805825e6 0.021495968
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   0.032774426   0.038583875   147511.77  0.071746476
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   0.09908231    0.10698437    48797.91   0.16894186
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   0.21016328    0.28666922    22996.9    1.0
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 0.0005491873  0.0006503939  8.803094e6 0.020654399
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 0.033055868   0.035830382   146264.4   0.06460478
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 0.0905178     0.10031668    53417.71   0.16610569
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 0.2161817     0.2877253     22355.297  1.0
# tornadao_mean_prob_computed_climatology_3hr_1567           0.0005491144  0.0006493895  8.80407e6  0.021308538
# tornadao_mean_prob_computed_climatology_3hr_1567           0.032205112   0.037208974   150132.1   0.067999676
# tornadao_mean_prob_computed_climatology_3hr_1567           0.096681885   0.10643049    50004.45   0.17814146
# tornadao_mean_prob_computed_climatology_3hr_1567           0.23098622    0.2981269     20925.564  1.0
# tornado_full_13831                                         0.000549145   0.0006369488  8.804865e6 0.021043906
# tornado_full_13831                                         0.0320742     0.038657922   150732.98  0.074019335
# tornado_full_13831                                         0.100802325   0.10940487    47960.055  0.17095083
# tornado_full_13831                                         0.22403128    0.29084346    21573.916  1.0


println("event_to_day_bins = $event_to_day_bins")
# event_to_day_bins = Dict{String, Vector{Float32}}("tornadao_mean_prob_computed_partial_climatology_227" => [0.019988786, 0.061916392, 0.16038308, 1.0], "tornadao_mean_58" => [0.017304773, 0.0553223, 0.13471735, 1.0], "tornadao_mean_prob_computed_climatology_prior_next_hrs_691" => [0.020654399, 0.06460478, 0.16610569, 1.0], "tornadao_prob_80" => [0.019274753, 0.059863195, 0.13051617, 1.0], "tornado_full_13831" => [0.021043906, 0.074019335, 0.17095083, 1.0], "tornadao_mean_prob_computed_no_sv_219" => [0.019619932, 0.06317351, 0.14767814, 1.0], "tornadao_mean_prob_computed_220" => [0.020045973, 0.06332173, 0.14715679, 1.0], "tornadao_mean_prob_computed_climatology_253" => [0.02144064, 0.063346006, 0.16949469, 1.0], "tornadao_mean_prob_138" => [0.019996958, 0.06293694, 0.14385402, 1.0], "tornadao_mean_prob_computed_climatology_grads_1348" => [0.020726241, 0.06985498, 0.16851029, 1.0], "tornadao_mean_prob_computed_climatology_blurs_910" => [0.02166239, 0.06790343, 0.16562855, 1.0], "tornadao_mean_prob_computed_climatology_blurs_grads_2005" => [0.021495968, 0.071746476, 0.16894186, 1.0], "tornadao_mean_prob_computed_climatology_3hr_1567" => [0.021308538, 0.067999676, 0.17814146, 1.0])











# 4. combine bin-pairs (overlapping, 3 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms


function find_logistic_coeffs(event_name, model_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  bins_max = event_to_day_bins[model_name]
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_total_prob_x  = X[bin_members, prediction_i*2 - 1]
    bin_max_hourly_x  = X[bin_members, prediction_i*2]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

    Threads.@threads for i in 1:length(bin_y)
      logit_total_prob = logit(bin_total_prob_x[i])
      logit_max_hourly = logit(bin_max_hourly_x[i])

      bin_X_features[i,1] = logit_total_prob
      bin_X_features[i,2] = logit_max_hourly
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("model_name", model_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("total_prob_ŷ_min", bin_min),
      ("total_prob_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_total_prob_ŷ", sum(bin_total_prob_x .* bin_weights) / bin_weight),
      ("mean_max_hourly_ŷ", sum(bin_max_hourly_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("total_prob_logloss", sum(logloss.(bin_y, bin_total_prob_x) .* bin_weights) / bin_weight),
      ("max_hourly_logloss", sum(logloss.(bin_y, bin_max_hourly_x) .* bin_weights) / bin_weight),
      ("total_prob_au_pr", Float32(Metrics.area_under_pr_curve(bin_total_prob_x, bin_y, bin_weights))),
      ("max_hourly_au_pr", Float32(Metrics.area_under_pr_curve(bin_max_hourly_x, bin_y, bin_weights))),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Float32(Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights))),
      ("logistic_coeffs", coeffs)
    ]

    headers = map(first, stuff)
    row     = map(last, stuff)

    bin_i == 1 && println(join(headers, "\t"))
    println(join(row, "\t"))

    push!(bins_logistic_coeffs, coeffs)
  end

  bins_logistic_coeffs
end

event_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:nmodels
  model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]

  event_to_day_bins_logistic_coeffs[model_name] = find_logistic_coeffs("tornado", model_name, prediction_i, X, Ys, weights)
end

# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_58                                           1-2 -1.0             0.0553223        9696254 10423.0   8.934532e6 0.0012567121      0.00032494482     0.0010822886 0.0060854536       0.0066588083       0.02568117       0.025274718      0.0010822888    0.0060613793     0.026141258    Float32[0.6559432,  0.33183938,   0.2690549]
# tornadao_mean_58                                           2-3 0.017304773      0.13471735       265695  10312.0   247997.38  0.044027895       0.011814468       0.038991798  0.15783298         0.1784407          0.07641898       0.073937856      0.038991798     0.15708669       0.077856414    Float32[0.6666154,  0.36046487,   0.4614278]
# tornadao_mean_58                                           3-4 0.0553223        1.0              95762   10183.0   90599.98   0.13208137        0.034078375       0.10670821   0.3278625          0.3775572          0.20934382       0.19714081       0.106708206     0.32114118       0.21226463     Float32[0.42497587, 0.44417486,   0.20855717]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_prob_80                                           1-2 -1.0             0.059863195      9702047 10437.0   8.939466e6 0.0013364629      0.00038399108     0.0010816164 0.0060175555       0.0064752693       0.030860346      0.02929963       0.0010816163    0.005984786      0.031192083    Float32[0.8796285,  0.16921686,   0.22449368]
# tornadao_prob_80                                           2-3 0.019274753      0.13051617       236468  10324.0   222162.9   0.04666545        0.014008047       0.04352574   0.17031741         0.19181994         0.084860384      0.08604295       0.043525744     0.16960227       0.08983052     Float32[0.8071765,  0.3629977,    0.88375133]
# tornadao_prob_80                                           3-4 0.059863195      1.0              89969   10169.0   85665.336  0.13617392        0.037335          0.11286281   0.34404176         0.39736992         0.20111248       0.18362024       0.1128628       0.33861682       0.20195282     Float32[0.51591927, 0.29846162,   -0.097147524]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_138                                     1-2 -1.0             0.06293694       9707455 10417.0   8.944838e6 0.0013084363      0.00036521038     0.0010809927 0.0059776995       0.006496915        0.028291194      0.026154425      0.0010809926    0.0059552435     0.028430644    Float32[0.93436664, 0.07926717,   -0.026851058]
# tornadao_mean_prob_138                                     2-3 0.019996958      0.14385402       233659  10320.0   218863.8   0.049323488       0.014423123       0.04418133   0.17289616         0.19545229         0.087084584      0.08254773       0.04418133      0.17245421       0.08872876     Float32[0.9042307,  0.17261904,   0.32986057]
# tornadao_mean_prob_138                                     3-4 0.06293694       1.0              84561   10189.0   80293.81   0.14589518        0.04049197        0.12041028   0.35680637         0.41135052         0.215481         0.209414         0.12041028      0.35006297       0.22002642     Float32[0.46952277, 0.3558928,    0.015089741]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_no_sv_219                      1-2 -1.0             0.06317351       9709203 10415.0   8.946578e6 0.0012803365      0.0003509757      0.0010808608 0.005960811        0.0065027666       0.029830607      0.02745438       0.0010808606    0.005941358      0.029989375    Float32[0.8970471,  0.12673378,   0.10686361]
# tornadao_mean_prob_computed_no_sv_219                      2-3 0.019619932      0.14767814       229041  10315.0   214424.97  0.049638774       0.0143582625      0.045096405  0.17584834         0.19896401         0.08727907       0.08571333       0.045096405     0.17540172       0.089979455    Float32[0.77730256, 0.24956441,   0.30919433]
# tornadao_mean_prob_computed_no_sv_219                      3-4 0.06317351       1.0              82813   10191.0   78553.914  0.1483812         0.040372815       0.12306833   0.36492777         0.4197913          0.21194836       0.20339726       0.12306832      0.3567272        0.21477623     Float32[0.38325134, 0.41652393,   0.074524485]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_220                            1-2 -1.0             0.06332173       9708100 10412.0   8.94555e6  0.0012724124      0.00035469467     0.0010809592 0.0059478595       0.0064870096       0.029999185      0.028186163      0.0010809592    0.005930681      0.030236766    Float32[0.9081216,  0.107832894,  0.050864797]
# tornadao_mean_prob_computed_220                            2-3 0.020045973      0.14715679       224454  10314.0   210120.52  0.05027804        0.014654327       0.046021793  0.17881362         0.20252019         0.089375         0.08498586       0.046021793     0.1784264        0.090924874    Float32[0.800826,   0.23256853,   0.31022668]
# tornadao_mean_prob_computed_220                            3-4 0.06332173       1.0              83916   10194.0   79581.67   0.14703318        0.03939797        0.12148187   0.36165562         0.4170215          0.20962848       0.20314537       0.12148187      0.35378066       0.21338512     Float32[0.40040252, 0.4055984,    0.067884356]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_partial_climatology_227        1-2 -1.0             0.061916392      9710511 10405.0   8.947747e6 0.001240658       0.0003388053      0.0010807714 0.0059278994       0.0065114894       0.03022069       0.028243082      0.0010807713    0.005915461      0.03050957     Float32[0.9746325,  0.05395953,   0.059440296]
# tornadao_mean_prob_computed_partial_climatology_227        2-3 0.019988786      0.16038308       221807  10317.0   208117.34  0.05056276        0.014658076       0.046462726  0.17946826         0.204413           0.096210934      0.0893217        0.046462722     0.179183         0.0984764      Float32[0.8418499,  0.16475138,   0.1517096]
# tornadao_mean_prob_computed_partial_climatology_227        3-4 0.061916392      1.0              81505   10201.0   77384.7    0.15183213        0.044288047       0.12492177   0.36279795         0.4149026          0.22880438       0.21963352       0.124921784     0.35611352       0.23174424     Float32[0.48570332, 0.34678283,   -0.001083027]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_253                1-2 -1.0             0.063346006      9710126 10383.0   8.947631e6 0.0012564469      0.00034074739     0.0010806825 0.005891865        0.006469967        0.031067852      0.02872078       0.0010806825    0.005876925      0.030992843    Float32[0.982298,   0.05099982,   0.06179544]
# tornadao_mean_prob_computed_climatology_253                2-3 0.02144064       0.16949469       213944  10316.0   200396.22  0.054135315       0.01560994        0.04824966   0.18579943         0.21097252         0.09405763       0.08667607       0.04824966      0.18534118       0.09519578     Float32[0.80724055, 0.13905986,   -0.0778435]
# tornadao_mean_prob_computed_climatology_253                3-4 0.063346006      1.0              81890   10223.0   77501.03   0.15320885        0.04611948        0.12474617   0.3605683          0.41195595         0.2311477        0.21925905       0.124746144     0.3547717        0.23343967     Float32[0.58624,    0.255771,     -0.13458002]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_blurs_910          1-2 -1.0             0.06790343       9712335 10378.0   8.949703e6 0.0012467225      0.00032986148     0.0010804458 0.0058266623       0.00642974         0.033769023      0.031561345      0.0010804457    0.0058138226     0.033726607    Float32[0.98352575, 0.03981221,   0.010008344]
# tornadao_mean_prob_computed_climatology_blurs_910          2-3 0.02166239       0.16562855       210172  10314.0   197008.36  0.055336993       0.015332372       0.04907914   0.18692967         0.21396725         0.097825475      0.089538544      0.049079135     0.18652098       0.09890015     Float32[0.9775675,  0.045042068,  -0.0055927183]
# tornadao_mean_prob_computed_climatology_blurs_910          3-4 0.06790343       1.0              79681   10228.0   75429.06   0.15687333        0.043077637       0.12817124   0.3720117          0.43263972         0.22026254       0.20930332       0.12817124      0.36554968       0.22139385     Float32[0.57631963, 0.20392773,   -0.28303716]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_grads_1348         1-2 -1.0             0.06985498       9714340 10397.0   8.951356e6 0.0012808052      0.00034945284     0.0010802852 0.005839013        0.006387385        0.033812318      0.029529627      0.0010802853    0.0058203507     0.033697013    Float32[0.94020444, 0.0850467,    0.051697694]
# tornadao_mean_prob_computed_climatology_grads_1348         2-3 0.020726241      0.16851029       217332  10320.0   203874.44  0.05463208        0.01564935        0.047430914  0.18148142         0.20685415         0.09502037       0.08189804       0.047430906     0.18089765       0.09562675     Float32[1.1303655,  -0.08044045,  -0.13138889]
# tornadao_mean_prob_computed_climatology_grads_1348         3-4 0.06985498       1.0              77676   10209.0   73775.266  0.16024496        0.04653408        0.13103966   0.37649903         0.43550298         0.22815737       0.21831112       0.13103966      0.37059173       0.23022527     Float32[0.63580066, 0.1437665,    -0.37667832]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   1-2 -1.0             0.071746476      9716472 10399.0   8.953336e6 0.0012676042      0.00034184597     0.0010799763 0.00582184         0.006394267        0.03330716       0.029734429      0.0010799761    0.005806496      0.033193134    Float32[0.95560277, 0.05098174,   -0.06451891]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   2-3 0.021495968      0.16894186       209192  10321.0   196309.67  0.05558661        0.015846616       0.049256988  0.1866256          0.21373942         0.097892374      0.08648004       0.049257        0.18617484       0.0975887      Float32[1.1089742,  -0.043572877, -0.017801635]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   3-4 0.071746476      1.0              75544   10207.0   71794.81   0.16453998        0.04728531        0.13466312   0.38415837         0.447142           0.23444557       0.2223931        0.13466312      0.3776991        0.23589237     Float32[0.64432305, 0.10341309,   -0.47638172]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 1-2 -1.0             0.06460478       9712035 10394.0   8.949358e6 0.0012253602      0.00031497504     0.0010804625 0.0058915494       0.0065076104       0.031272214      0.030206645      0.0010804624    0.005879357      0.031604104    Float32[0.81858176, 0.18513452,   0.15438604]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 2-3 0.020654399      0.16610569       213084  10320.0   199682.1   0.053081363       0.014722611       0.04842773   0.18551108         0.2118019          0.09631815       0.090434216      0.048427735     0.1850077        0.09710584     Float32[0.6977342,  0.27534917,   0.21286853]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 3-4 0.06460478       1.0              79981   10212.0   75773.01   0.15560782        0.045160126       0.1275924    0.36658564         0.42145917         0.23740172       0.23477384       0.12759241      0.36017954       0.24182731     Float32[0.5196121,  0.31482816,   -0.040216677]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornadao_mean_prob_computed_climatology_3hr_1567           1-2 -1.0             0.067999676      9717107 10376.0   8.954202e6 0.0012623717      0.0003241106      0.0010798799 0.0058977003       0.006511817        0.031293306      0.027593525      0.0010798799    0.0058833743     0.031110743    Float32[0.96216094, 0.034555722,  -0.12364115]
# tornadao_mean_prob_computed_climatology_3hr_1567           2-3 0.021308538      0.17814146       213541  10303.0   200136.53  0.054504093       0.015043081       0.048314743  0.18396774         0.2112294          0.10122298       0.09050967       0.04831475      0.18355963       0.1007994      Float32[1.0485787,  -0.011567661, -0.047783367]
# tornadao_mean_prob_computed_climatology_3hr_1567           3-4 0.067999676      1.0              74909   10230.0   70930.016  0.1629842         0.04952672        0.13630395   0.38097858         0.44215336         0.25189593       0.24691355       0.13630396      0.3757996        0.25742134     Float32[0.6019041,  0.21778609,   -0.17962365]
# model_name                                                 bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_full_13831                                         1-2 -1.0             0.074019335      9718800 10390.0   8.955598e6 0.0012768853      0.00033077784     0.0010797478 0.0058558607       0.0064523746       0.032501902      0.029135438      0.0010797479    0.0058391923     0.032301545    Float32[0.95958227, 0.04161413,   -0.10651286]
# tornado_full_13831                                         2-3 0.021043906      0.17095083       211661  10323.0   198693.03  0.055734657       0.015497775       0.048663635  0.1845327          0.2118171          0.10270019       0.089645624      0.048663624     0.18389481       0.10116889     Float32[1.2272763,  -0.15624464,  -0.18067063]
# tornado_full_13831                                         3-4 0.074019335      1.0              73216   10216.0   69533.97   0.1656988         0.04697025        0.13903588   0.39089894         0.45915875         0.23874679       0.23317282       0.13903588      0.38513187       0.24201544     Float32[0.5964124,  0.17200926,   -0.3083448]

println("event_to_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("tornadao_mean_prob_computed_partial_climatology_227" => [[0.9746325, 0.05395953, 0.059440296], [0.8418499, 0.16475138, 0.1517096], [0.48570332, 0.34678283, -0.001083027]], "tornadao_mean_58" => [[0.6559432, 0.33183938, 0.2690549], [0.6666154, 0.36046487, 0.4614278], [0.42497587, 0.44417486, 0.20855717]], "tornadao_mean_prob_computed_climatology_prior_next_hrs_691" => [[0.81858176, 0.18513452, 0.15438604], [0.6977342, 0.27534917, 0.21286853], [0.5196121, 0.31482816, -0.040216677]], "tornadao_prob_80" => [[0.8796285, 0.16921686, 0.22449368], [0.8071765, 0.3629977, 0.88375133], [0.51591927, 0.29846162, -0.097147524]], "tornado_full_13831" => [[0.95958227, 0.04161413, -0.10651286], [1.2272763, -0.15624464, -0.18067063], [0.5964124, 0.17200926, -0.3083448]], "tornadao_mean_prob_computed_no_sv_219" => [[0.8970471, 0.12673378, 0.10686361], [0.77730256, 0.24956441, 0.30919433], [0.38325134, 0.41652393, 0.074524485]], "tornadao_mean_prob_computed_220" => [[0.9081216, 0.107832894, 0.050864797], [0.800826, 0.23256853, 0.31022668], [0.40040252, 0.4055984, 0.067884356]], "tornadao_mean_prob_computed_climatology_253" => [[0.982298, 0.05099982, 0.06179544], [0.80724055, 0.13905986, -0.0778435], [0.58624, 0.255771, -0.13458002]], "tornadao_mean_prob_138" => [[0.93436664, 0.07926717, -0.026851058], [0.9042307, 0.17261904, 0.32986057], [0.46952277, 0.3558928, 0.015089741]], "tornadao_mean_prob_computed_climatology_grads_1348" => [[0.94020444, 0.0850467, 0.051697694], [1.1303655, -0.08044045, -0.13138889], [0.63580066, 0.1437665, -0.37667832]], "tornadao_mean_prob_computed_climatology_blurs_910" => [[0.98352575, 0.03981221, 0.010008344], [0.9775675, 0.045042068, -0.0055927183], [0.57631963, 0.20392773, -0.28303716]], "tornadao_mean_prob_computed_climatology_blurs_grads_2005" => [[0.95560277, 0.05098174, -0.06451891], [1.1089742, -0.043572877, -0.017801635], [0.64432305, 0.10341309, -0.47638172]], "tornadao_mean_prob_computed_climatology_3hr_1567" => [[0.96216094, 0.034555722, -0.12364115], [1.0485787, -0.011567661, -0.047783367], [0.6019041, 0.21778609, -0.17962365]])





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
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected:
# 157

compute_day_labels(events, forecast) = begin
  # Annoying that we have to recalculate this.
  # The end_seconds will always be the last hour of the convective day
  # start_seconds depends on whether the run started during the day or not
  # I suppose for 0Z the answer is always "no" but whatev here's the right math
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 2*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime = Dates.unix2datetime(start_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_day_labeler = Dict(
  "tornado"     => (forecast -> compute_day_labels(StormEvents.conus_tornado_events(),     forecast)),
  "wind"        => (forecast -> compute_day_labels(StormEvents.conus_severe_wind_events(), forecast)),
  "hail"        => (forecast -> compute_day_labels(StormEvents.conus_severe_hail_events(), forecast)),
  "sig_tornado" => (forecast -> compute_day_labels(StormEvents.conus_sig_tornado_events(), forecast)),
  "sig_wind"    => (forecast -> compute_day_labels(StormEvents.conus_sig_wind_events(),    forecast)),
  "sig_hail"    => (forecast -> compute_day_labels(StormEvents.conus_sig_hail_events(),    forecast)),
)

# rm("day_validation_forecasts_0z_12z_with_sig_gated"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_12z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_12z_with_sig_gated",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPredictionAblations.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (9446.0)                      feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  AU-PR-curve: 0.12701213629264377
# wind (72111.0)                        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.3834656173824258
# hail (31894.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.23064103438576106
# sig_tornado (1268.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.09451181604245633
# sig_wind (8732.0)                     feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.07854469228892626
# sig_hail (4478.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.0691045922967536
# sig_tornado_gated_by_tornado (1268.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.09363142784066525
# sig_wind_gated_by_wind (8732.0)       feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.07855136828846834
# sig_hail_gated_by_hail (4478.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.06912157017181443


function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, HREFPredictionAblations.models_with_gated))

  # Feature order is all HREF severe probs then all SREF severe probs
  for event_name in event_names
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = HREFPredictionAblations.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      y = Ys[event_name]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (9446.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.12701213629264377
# tornado (9446.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.04168648434556345
# tornado (9446.0)     feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.030765254579900772
# tornado (9446.0)     feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.11531809758773186
# tornado (9446.0)     feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.039822720343631504
# tornado (9446.0)     feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.032309275402615086
# tornado (9446.0)     feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.11624433417007461
# tornado (9446.0)     feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.03983155452623281
# tornado (9446.0)     feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.03231108262426286
# wind (72111.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.20397678719531137
# wind (72111.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.3834656173824258
# wind (72111.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.15227293177135814
# wind (72111.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.18899000884190237
# wind (72111.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.2871189699820319
# wind (72111.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.1232035272300651
# wind (72111.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.19067429713597928
# wind (72111.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.28719210932407674
# wind (72111.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.12326955300059134
# hail (31894.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.09099255929593081
# hail (31894.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.09452327656751952
# hail (31894.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.23064103438576106
# hail (31894.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.08234931731140827
# hail (31894.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.09955065202449667
# hail (31894.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.188595511108089
# hail (31894.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.0824502686306494
# hail (31894.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.09958118472086727
# hail (31894.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.1886802022992225
# sig_tornado (1268.0) feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.05728594701652062
# sig_tornado (1268.0) feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.011385789618738
# sig_tornado (1268.0) feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.005588672134719318
# sig_tornado (1268.0) feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.09451181604245633
# sig_tornado (1268.0) feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.008881777469212735
# sig_tornado (1268.0) feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.005321406702197539
# sig_tornado (1268.0) feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.09363142784066525
# sig_tornado (1268.0) feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.008882564887699068
# sig_tornado (1268.0) feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.005324294230821452
# sig_wind (8732.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.03572679857698384
# sig_wind (8732.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.05250853163761901
# sig_wind (8732.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.03161747330844581
# sig_wind (8732.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.034007477644701585
# sig_wind (8732.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.07854469228892626
# sig_wind (8732.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.028783730365721183
# sig_wind (8732.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.03429306032063456
# sig_wind (8732.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.07855136828846834
# sig_wind (8732.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.02879889359659011
# sig_hail (4478.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.01798515458351701
# sig_hail (4478.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.013479598274345111
# sig_hail (4478.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.06589495461034388
# sig_hail (4478.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.014544889120801473
# sig_hail (4478.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.01836176630607075
# sig_hail (4478.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.0691045922967536
# sig_hail (4478.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.01464675890085453
# sig_hail (4478.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.018365716755777285
# sig_hail (4478.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.06912157017181443



# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  total_weight = sum(Float64.(weights))

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPredictionAblations.models_with_gated[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

    total_pos_weight = sum(Float64.(y .* weights))

    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_count = 20
    per_bin_pos_weight = Float64(sum(y .* weights)) / bin_count

    # bins = map(_ -> Int64[], 1:bin_count)
    bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
    bins_Σy      = map(_ -> 0.0, 1:bin_count)
    bins_Σweight = map(_ -> 0.0, 1:bin_count)
    bins_max     = map(_ -> 1.0f0, 1:bin_count)

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

    for bin_i in 1:bin_count
      Σŷ      = bins_Σŷ[bin_i]
      Σy      = bins_Σy[bin_i]
      Σweight = Float32(bins_Σweight[bin_i])

      mean_ŷ = Float32(Σŷ / Σweight)
      mean_y = Float32(Σy / Σweight)

      pos_weight_in_and_after = sum(bins_Σy[bin_i:bin_count])
      weight_in_and_after     = sum(bins_Σweight[bin_i:bin_count])

      sr  = Float32(pos_weight_in_and_after / weight_in_and_after)
      pod = Float32(pos_weight_in_and_after / total_pos_weight)

      println("$model_name\t$mean_y\t$mean_ŷ\t$Σweight\t$sr\t$pod\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(day_validation_forecasts_0z_12z, X, Ys, weights)

# event_name                   mean_y        mean_ŷ        Σweight     SR            POD         bin_max
# tornado                      9.700865e-5   9.9049845e-5  4.573599e6  0.0017090585  1.0         0.0008555745
# tornado                      0.0018642666  0.00142832    237875.36   0.013719558   0.9499555   0.0023576757
# tornado                      0.0031376549  0.0036643934  141532.73   0.02121991    0.89993536  0.005619943
# tornado                      0.006996247   0.007203153   63402.914   0.032135308   0.84984547  0.009180412
# tornado                      0.00960002    0.011391198   46223.64    0.04145316    0.79981184  0.014106784
# tornado                      0.016310018   0.01626344    27224.174   0.053247765   0.74975955  0.018732127
# tornado                      0.02485353    0.020887442   17840.715   0.06355       0.6996758   0.023426436
# tornado                      0.028516073   0.026392281   15566.109   0.072204635   0.64966226  0.029872384
# tornado                      0.031154005   0.034372345   14257.465   0.08279699    0.59959453  0.039933395
# tornado                      0.041785702   0.046009205   10617.009   0.097539      0.54949385  0.053307083
# tornado                      0.04763488    0.0617777     9316.975    0.112589985   0.4994538   0.07096115
# tornado                      0.06391539    0.07893289    6942.6855   0.13275504    0.4493942   0.08757907
# tornado                      0.11442482    0.09269282    3880.694    0.15347265    0.3993423   0.09807844
# tornado                      0.14126927    0.10239867    3138.6887   0.16136983    0.3492562   0.10644376
# tornado                      0.16568106    0.10938814    2675.5706   0.16530076    0.2992431   0.11252215
# tornado                      0.1327265     0.11989863    3344.8777   0.16522467    0.24924229  0.12869819
# tornado                      0.11034642    0.14242381    4017.9119   0.17606342    0.19916676  0.15996142
# tornado                      0.13868733    0.18497059    3196.6716   0.21998936    0.14915797  0.21840203
# tornado                      0.32516035    0.23600286    1364.0076   0.31233206    0.099151924 0.2564116
# tornado                      0.3002684     0.3199752     1450.4642   0.3002684     0.049125206 1.0
# wind                         0.0007333039  0.0006652036  4.5677065e6 0.012911272   1.0         0.011141602
# wind                         0.017665606   0.017918272   189598.22   0.10266422    0.9499899   0.027255114
# wind                         0.034650616   0.03490376    96670.13    0.14012814    0.89998204  0.044160966
# wind                         0.05201806    0.05276521    64395.637   0.17070308    0.8499695   0.06265394
# wind                         0.066318914   0.07287459    50508.055   0.19910471    0.79995614  0.0843732
# wind                         0.08921028    0.09491389    37541.48    0.22978672    0.7499442   0.10652433
# wind                         0.11898993    0.116981484   28143.969   0.25893623    0.6999405   0.12846802
# wind                         0.14411765    0.13994569    23240.582   0.28469524    0.6499404   0.15250224
# wind                         0.1720042     0.16488889    19471.506   0.30989215    0.5999324   0.17824909
# wind                         0.19195645    0.19248405    17449.746   0.33425775    0.5499273   0.20799732
# wind                         0.21295361    0.22392222    15728.158   0.36103234    0.49991605  0.24049737
# wind                         0.2602031     0.25579062    12870.99    0.39127383    0.44990817  0.27177548
# wind                         0.2981486     0.2870136     11232.168   0.41757473    0.39990473  0.30342957
# wind                         0.3278786     0.31958458    10215.239   0.44292727    0.3499045   0.33644634
# wind                         0.3542443     0.35344136    9454.063    0.47045377    0.29989678  0.3715729
# wind                         0.3776399     0.39261442    8868.662    0.50350475    0.24989367  0.41578835
# wind                         0.43711454    0.44158578    7661.592    0.5493044     0.19988889  0.47099006
# wind                         0.52115136    0.50405425    6426.139    0.6007411     0.14988661  0.54199266
# wind                         0.57601786    0.58847904    5814.9453   0.65047044    0.099884346 0.64767545
# wind                         0.7473285     0.7519671     4469.818    0.7473285     0.049874317 1.0
# hail                         0.00031078092 0.00030401986 4.7322955e6 0.005669666   1.0         0.006891266
# hail                         0.01044888    0.011080333   140805.23   0.061384488   0.94999504  0.016829325
# hail                         0.021978628   0.021263007   66916.24    0.0841986     0.8999713   0.026533825
# hail                         0.03235512    0.031273644   45454.387   0.10102429    0.84996563  0.03663534
# hail                         0.045308188   0.041578356   32474.467   0.11647665    0.7999615   0.04720469
# hail                         0.055931125   0.052971583   26297.406   0.13011006    0.7499344   0.05915104
# hail                         0.06446846    0.06542572    22810.768   0.14373004    0.6999247   0.07218698
# hail                         0.078798346   0.07871479    18669.86    0.15874511    0.6499242   0.08571534
# hail                         0.08890267    0.09247879    16542.453   0.17341526    0.59990406  0.09952751
# hail                         0.10120925    0.10641503    14538.252   0.18982401    0.5499004   0.11399117
# hail                         0.11987173    0.12220042    12273.574   0.20805569    0.49987164  0.13112558
# hail                         0.13827227    0.14001511    10638.314   0.22659214    0.44984806  0.14941467
# hail                         0.15829067    0.15864913    9296.101    0.24626866    0.39983365  0.16864169
# hail                         0.18365733    0.17857906    8008.6064   0.26753646    0.34980217  0.18962103
# hail                         0.19555932    0.20265509    7520.521    0.2896        0.29979268  0.2174353
# hail                         0.2370742     0.23240577    6207.0747   0.32044885    0.24978767  0.24881782
# hail                         0.28289422    0.26606965    5198.3096   0.35140282    0.19975445  0.2847625
# hail                         0.3352303     0.30533865    4388.1494   0.3823156     0.14975406  0.328701
# hail                         0.38238993    0.35775426    3847.2227   0.41128483    0.09973774  0.39531642
# hail                         0.44512427    0.4873659     3285.0786   0.44512427    0.04971806  1.0
# sig_tornado                  1.2047162e-5  1.4437429e-5  5.0279775e6 0.00023340616 1.0         0.00046008435
# sig_tornado                  0.0015489991  0.000620892   39503.88    0.0072118156  0.94997233  0.0008245053
# sig_tornado                  0.0012606728  0.0012867345  48650.09    0.009076222   0.89943373  0.0019798223
# sig_tornado                  0.001833995   0.003183769   33270.344   0.014406291   0.8487792   0.0052594747
# sig_tornado                  0.00690796    0.006452647   8801.481    0.025394725   0.79838413  0.0078367125
# sig_tornado                  0.00995231    0.00908278    6115.261    0.03095475    0.74816865  0.010524806
# sig_tornado                  0.01157955    0.012097383   5262.185    0.03650292    0.69790304  0.013926878
# sig_tornado                  0.01346963    0.01593964    4508.503    0.04383515    0.6475773   0.018244676
# sig_tornado                  0.024001416   0.019918984   2547.455    0.054068234   0.5974216   0.021780202
# sig_tornado                  0.025120918   0.024143767   2422.7295   0.06113995    0.5469234   0.026980141
# sig_tornado                  0.03003723    0.03010828    2039.1814   0.07151833    0.49665758  0.033948936
# sig_tornado                  0.033580653   0.038780004   1817.3179   0.08479924    0.44606954  0.045137547
# sig_tornado                  0.05013078    0.05185326    1221.1023   0.10524846    0.395667    0.059880484
# sig_tornado                  0.080563776   0.06687405    760.7198    0.12545581    0.3451091   0.07437804
# sig_tornado                  0.09334492    0.08271323    656.9288    0.13874404    0.29449207  0.09200536
# sig_tornado                  0.11114166    0.10164373    552.5639    0.15433393    0.24384652  0.11262035
# sig_tornado                  0.12330799    0.12747169    498.82028   0.17187674    0.19312507  0.14483048
# sig_tornado                  0.14542684    0.1704235     417.73834   0.1999938     0.14232461  0.19986959
# sig_tornado                  0.30896956    0.21962947    196.64925   0.25134343    0.09215033  0.23971567
# sig_tornado                  0.20551312    0.27465805    247.26294   0.20551312    0.041969217 1.0
# sig_wind                     8.762574e-5   7.956794e-5   4.615673e6  0.0015565632  1.0         0.0010795709
# sig_wind                     0.0020825795  0.001814825   193981.45   0.013414203   0.94991076  0.0028963166
# sig_wind                     0.0039179027  0.0039368668  103105.52   0.019232223   0.8998797   0.005224411
# sig_wind                     0.005834378   0.006467304   69213.65    0.02498012    0.84985167  0.007945357
# sig_wind                     0.008835517   0.009320285   45742.48    0.03142871    0.7998408   0.01091732
# sig_wind                     0.011483245   0.012639817   35174.387   0.037897937   0.74978787  0.014694147
# sig_wind                     0.01744931    0.01655319    23148.463   0.045356132   0.69976497  0.01871097
# sig_wind                     0.020649865   0.021098858   19558.607   0.05172514    0.649741    0.02387907
# sig_wind                     0.025003763   0.027136791   16180.071   0.059148967   0.5997222   0.031004418
# sig_wind                     0.033851463   0.03475053    11945.531   0.06755926    0.5496192   0.038893435
# sig_wind                     0.039434146   0.043058835   10251.556   0.07505133    0.4995396   0.047464754
# sig_wind                     0.04473416    0.05181563    9045.808    0.08344653    0.44947395  0.056373745
# sig_wind                     0.06492861    0.0598693     6231.9253   0.093612395   0.39935932  0.063569665
# sig_wind                     0.065945596   0.06775919    6133.274    0.099947825   0.34924793  0.072085015
# sig_wind                     0.08764464    0.07611215    4606.5415   0.10939199    0.29915738  0.08034126
# sig_wind                     0.090064      0.08515918    4488.718    0.115124635   0.24915643  0.09031043
# sig_wind                     0.09548842    0.096013896   4236.5854   0.12378663    0.19908945  0.102228664
# sig_wind                     0.111371435   0.109113194   3625.3943   0.13748801    0.14898866  0.117370516
# sig_wind                     0.1494803     0.12685496    2704.816    0.15596397    0.09898441  0.13964514
# sig_wind                     0.16321121    0.18001986    2419.8394   0.16321121    0.04891188  1.0
# sig_hail                     4.2421325e-5  3.220337e-5   4.924094e6  0.0008022721  1.0         0.001318197
# sig_hail                     0.0025382582  0.0022416795  82272.734   0.015008614   0.9498081   0.0036093877
# sig_hail                     0.0043337564  0.005106818   48057.76    0.020673798   0.89962995  0.006999807
# sig_hail                     0.0067134313  0.008847184   31044.45    0.026576137   0.84958607  0.011068663
# sig_hail                     0.010588703   0.012986053   19688.49    0.032621574   0.79950756  0.015182176
# sig_hail                     0.017071232   0.016824745   12210.837   0.0378918     0.74941444  0.018555325
# sig_hail                     0.021248048   0.02004428    9799.877    0.041518603   0.6993265   0.02158984
# sig_hail                     0.026669838   0.022903282   7809.4185   0.04481297    0.6492928   0.02426124
# sig_hail                     0.025347624   0.025871573   8213.022    0.047512285   0.5992477   0.027540982
# sig_hail                     0.027341636   0.029312754   7630.691    0.051623642   0.54922545  0.031174533
# sig_hail                     0.030267406   0.033157837   6887.0146   0.05667976    0.49909386  0.035307348
# sig_hail                     0.037927397   0.03742463    5501.4717   0.0627922     0.4490064   0.039753683
# sig_hail                     0.048469674   0.041902665   4302.0317   0.068431295   0.3988698   0.044243563
# sig_hail                     0.0492872     0.04702346    4225.762    0.072734565   0.34876648  0.050134685
# sig_hail                     0.056120433   0.0533297     3716.3848   0.0790335     0.2987213   0.056848884
# sig_hail                     0.059499297   0.06103613    3501.9128   0.08612154    0.24860668  0.0658436
# sig_hail                     0.085206375   0.069975354   2444.3845   0.09707439    0.198541    0.07495049
# sig_hail                     0.095850974   0.08108813    2172.8608   0.10185565    0.14849555  0.088182054
# sig_hail                     0.11374573    0.09691916    1830.0184   0.10520578    0.09845164  0.1077964
# sig_hail                     0.09763601    0.1416834     2064.5598   0.09763601    0.04843512  1.0
# sig_tornado_gated_by_tornado 1.2044688e-5  1.4464373e-5  5.02901e6   0.00023340616 1.0         0.00046008435
# sig_tornado_gated_by_tornado 0.0015538402  0.00062055915 39380.8     0.0072588217  0.94997233  0.0008245053
# sig_tornado_gated_by_tornado 0.0012766533  0.0012856277  48041.113   0.0091455635  0.89943373  0.0019798223
# sig_tornado_gated_by_tornado 0.0018461995  0.0031905377  33050.406   0.014467288   0.8487792   0.0052594747
# sig_tornado_gated_by_tornado 0.0069257417  0.006443854   8780.715    0.025448764   0.79838413  0.007811773
# sig_tornado_gated_by_tornado 0.01077961    0.008947689   5649.973    0.03101798    0.74815816  0.010257343
# sig_tornado_gated_by_tornado 0.01272586    0.011599156   4794.479    0.035872545   0.6978566   0.013152388
# sig_tornado_gated_by_tornado 0.013809002   0.0149549795  4414.7456   0.041788157   0.64746463  0.017007226
# sig_tornado_gated_by_tornado 0.022520622   0.018557316   2726.78     0.050398786   0.59711456  0.020294942
# sig_tornado_gated_by_tornado 0.029579157   0.021911496   2063.1477   0.056941662   0.5463965   0.023793506
# sig_tornado_gated_by_tornado 0.030604053   0.026178267   2002.1079   0.06284972    0.49599442  0.028877527
# sig_tornado_gated_by_tornado 0.031029046   0.032508727   1955.7277   0.071397096   0.4453888   0.036711503
# sig_tornado_gated_by_tornado 0.04090295    0.04133125    1495.6295   0.08550168    0.39526904  0.047611304
# sig_tornado_gated_by_tornado 0.05159111    0.055509657   1186.4545   0.10176375    0.3447435   0.06523254
# sig_tornado_gated_by_tornado 0.079230666   0.073610626   772.46344   0.1221827     0.29418918  0.083278306
# sig_tornado_gated_by_tornado 0.08917256    0.09419476    687.0461    0.13766626    0.24364123  0.10639991
# sig_tornado_gated_by_tornado 0.092989005   0.12223298    660.36676   0.16055223    0.19304135  0.14342135
# sig_tornado_gated_by_tornado 0.15301146    0.16837253    396.83765   0.21664305    0.14232488  0.1984245
# sig_tornado_gated_by_tornado 0.34620395    0.21400294    176.84094   0.279994      0.092175096 0.22993922
# sig_tornado_gated_by_tornado 0.22719409    0.2585972     221.7547    0.22719409    0.041610427 1.0
# sig_wind_gated_by_wind       8.759892e-5   7.887138e-5   4.617086e6  0.0015565632  1.0         0.0010795709
# sig_wind_gated_by_wind       0.0020938707  0.0018146777  192935.42   0.013447442   0.94991076  0.0028963166
# sig_wind_gated_by_wind       0.003926863   0.003937269   102870.26   0.019250939   0.8998797   0.005224411
# sig_wind_gated_by_wind       0.0058410047  0.00646799    69135.125   0.024992133   0.84985167  0.007945357
# sig_wind_gated_by_wind       0.008835756   0.009320174   45741.242   0.031436898   0.7998408   0.01091732
# sig_wind_gated_by_wind       0.0114916805  0.012639401   35148.566   0.037910346   0.74978787  0.014694147
# sig_wind_gated_by_wind       0.017467922   0.016553186   23123.799   0.04536577    0.69976497  0.01871097
# sig_wind_gated_by_wind       0.020651773   0.021099042   19556.8     0.05172606    0.649741    0.02387907
# sig_wind_gated_by_wind       0.024994126   0.027136952   16186.31    0.059148967   0.5997222   0.031004418
# sig_wind_gated_by_wind       0.033828873   0.03475125    11953.508   0.06756567    0.5496192   0.038893435
# sig_wind_gated_by_wind       0.039471738   0.04305987    10241.793   0.075071186   0.4995396   0.047464754
# sig_wind_gated_by_wind       0.044756185   0.0518161     9041.356    0.08345507    0.44947395  0.056373745
# sig_wind_gated_by_wind       0.06492861    0.0598693     6231.9253   0.093612395   0.39935932  0.063569665
# sig_wind_gated_by_wind       0.065945596   0.06775919    6133.274    0.099947825   0.34924793  0.072085015
# sig_wind_gated_by_wind       0.08764464    0.07611215    4606.5415   0.10939199    0.29915738  0.08034126
# sig_wind_gated_by_wind       0.090064      0.08515918    4488.718    0.115124635   0.24915643  0.09031043
# sig_wind_gated_by_wind       0.09548842    0.096013896   4236.5854   0.12378663    0.19908945  0.102228664
# sig_wind_gated_by_wind       0.111371435   0.109113194   3625.3943   0.13748801    0.14898866  0.117370516
# sig_wind_gated_by_wind       0.1494803     0.12685496    2704.816    0.15596397    0.09898441  0.13964514
# sig_wind_gated_by_wind       0.16321121    0.18001986    2419.8394   0.16321121    0.04891188  1.0
# sig_hail_gated_by_hail       4.241907e-5   3.2193075e-5  4.924356e6  0.0008022721  1.0         0.001318197
# sig_hail_gated_by_hail       0.002543777   0.0022429482  82094.24    0.01502357    0.9498081   0.0036093877
# sig_hail_gated_by_hail       0.0043354956  0.005106204   48038.484   0.020683357   0.89962995  0.006999807
# sig_hail_gated_by_hail       0.0067161745  0.008847436   31031.768   0.02658901    0.84958607  0.011068663
# sig_hail_gated_by_hail       0.010594521   0.012986352   19677.68    0.032638125   0.79950756  0.015182176
# sig_hail_gated_by_hail       0.017215528   0.016809182   12091.6045  0.037910648   0.74941444  0.018515944
# sig_hail_gated_by_hail       0.021046346   0.020027013   9907.607    0.041476414   0.6993964   0.02158984
# sig_hail_gated_by_hail       0.026670078   0.022903148   7809.348    0.044834845   0.6492928   0.02426124
# sig_hail_gated_by_hail       0.02535061    0.025872072   8212.055    0.047538865   0.5992477   0.027540982
# sig_hail_gated_by_hail       0.027338756   0.029313533   7631.4946   0.051656753   0.54922545  0.031174533
# sig_hail_gated_by_hail       0.030258834   0.033157654   6888.9653   0.056724932   0.49909386  0.035307348
# sig_hail_gated_by_hail       0.037934076   0.03742634    5500.503    0.062857956   0.4490064   0.039753683
# sig_hail_gated_by_hail       0.048634436   0.04190139    4287.4575   0.0685165     0.3988698   0.044243563
# sig_hail_gated_by_hail       0.049389396   0.04702226    4217.0186   0.07279144    0.34876648  0.050134685
# sig_hail_gated_by_hail       0.056194518   0.053331286   3711.485    0.07906792    0.2987213   0.056848884
# sig_hail_gated_by_hail       0.059532415   0.061038014   3499.9646   0.08613551    0.24860668  0.0658436
# sig_hail_gated_by_hail       0.085206375   0.069975354   2444.3845   0.09707439    0.198541    0.07495049
# sig_hail_gated_by_hail       0.095850974   0.08108813    2172.8608   0.10185565    0.14849555  0.088182054
# sig_hail_gated_by_hail       0.11374573    0.09691916    1830.0184   0.10520578    0.09845164  0.1077964
# sig_hail_gated_by_hail       0.09763601    0.1416834     2064.5598   0.09763601    0.04843512  1.0




# Calibrate to SPC
# The targets below are computing in and copied from models/spc_outlooks/Stats.jl

target_success_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.050700042),
    (0.05, 0.11771048),
    (0.1,  0.22891627),
    (0.15, 0.30348805),
    (0.3,  0.32036155), # lol
    (0.45, 0.5009283),
  ],
  "wind" => [
    (0.05, 0.13740821),
    (0.15, 0.26610714),
    (0.3,  0.45119646),
    (0.45, 0.64423746),
  ],
  "hail" => [
    (0.05, 0.08125695),
    (0.15, 0.16056664),
    (0.3,  0.30656156),
    (0.45, 0.56523347),
  ],
  "sig_tornado" => [(0.1, 0.09901425)],
  "sig_wind"    => [(0.1, 0.13118134)],
  "sig_hail"    => [(0.1, 0.08935113)],
)

target_PODs = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.69082415),
    (0.05, 0.424347),
    (0.1,  0.1553744),
    (0.15, 0.035544172),
    (0.3,  0.008358953),
    (0.45, 0.001138251),
  ],
  "wind" => [
    (0.05, 0.7425363),
    (0.15, 0.42171422),
    (0.3,  0.11319263),
    (0.45, 0.017438034),
  ],
  "hail" => [
    (0.05, 0.7651715),
    (0.15, 0.43377835),
    (0.3,  0.083377354),
    (0.45, 0.008335298),
  ],
  "sig_tornado" => [(0.1, 0.27615425)],
  "sig_wind"    => [(0.1, 0.13106804)],
  "sig_hail"    => [(0.1, 0.28429562)],
)

target_warning_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.019616714),
    (0.05, 0.005190068),
    (0.1,  0.0009771693),
    (0.15, 0.00016861407),
    (0.3,  3.7564576e-5),
    (0.45, 3.271369e-6),
  ],
  "wind" => [
    (0.05, 0.058618702),
    (0.15, 0.017190674),
    (0.3,  0.0027213453),
    (0.45, 0.000293618),
  ],
  "hail" => [
    (0.05, 0.04349142),
    (0.15, 0.012477219),
    (0.3,  0.0012561331),
    (0.45, 6.8108064e-5),
  ],
  "sig_tornado" => [(0.1, 0.00060326053)],
  "sig_wind"    => [(0.1, 0.0011946002)],
  "sig_hail"    => [(0.1, 0.002233119)],
)



# Assumes weights are proportional to gridpoint areas
# (here they are because we do not do any fancy subsetting)
function spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  # println("nominal_prob\tthreshold\tsuccess_ratio")

  thresholds_to_match_success_ratio =
    map(target_success_ratios[event_name]) do (nominal_prob, target_success_ratio)
      # Can't binary search, not monotonic.
      # Backtrack when SR exceeded
      threshold = 0.0f0
      step      = 0.02f0
      while step > 0.000001f0
        sr = Metrics.success_ratio(ŷ, y, weights, threshold)
        if isnan(sr) || sr > target_success_ratio
          step *= 0.5f0
          threshold -= step
        else
          threshold += step
        end
      end
      # println("$nominal_prob\t$threshold\t$(success_ratio(ŷ, y, weights, threshold))")
      threshold
    end

  # println("nominal_prob\tthreshold\tPOD")

  thresholds_to_match_POD =
    map(target_PODs[event_name]) do (nominal_prob, target_POD)
      threshold = 0.5f0
      step = 0.25f0
      while step > 0.000001f0
        pod = Metrics.probability_of_detection(ŷ, y, weights, threshold)
        if isnan(pod) || pod > target_POD
          threshold += step
        else
          threshold -= step
        end
        step *= 0.5f0
      end
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_success_ratio = thresholds_to_match_success_ratio[i]
    threshold_to_match_POD = thresholds_to_match_POD[i]
    mean_threshold = (threshold_to_match_success_ratio + threshold_to_match_POD) * 0.5f0
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, mean_threshold))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, mean_threshold))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, mean_threshold))
    println("$event_name\t$nominal_prob\t$threshold_to_match_success_ratio\t$threshold_to_match_POD\t$mean_threshold\t$sr\t$pod\t$wr")
    push!(thresholds, (Float32(nominal_prob), Float32(mean_threshold)))
  end

  thresholds
end


# Assumes weights are proportional to gridpoint areas
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  thresholds_to_match_warning_ratio =
    map(target_warning_ratios[event_name]) do (nominal_prob, target_warning_ratio)
      threshold = 0.5f0
      step = 0.25f0
      while step > 0.000001f0
        wr = Metrics.warning_ratio(ŷ, weights, threshold)
        if isnan(wr) || wr > target_warning_ratio
          threshold += step
        else
          threshold -= step
        end
        step *= 0.5f0
      end
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  wr_thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, threshold_to_match_warning_ratio))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, threshold_to_match_warning_ratio))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, threshold_to_match_warning_ratio))
    println("$event_name\t$nominal_prob\t$threshold_to_match_warning_ratio\t$sr\t$pod\t$wr")
    push!(wr_thresholds, (Float32(nominal_prob), Float32(threshold_to_match_warning_ratio)))
  end

  wr_thresholds
end

# Assumes weights are proportional to gridpoint areas
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_all(prediction_i, X, Ys, weights)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  # println("nominal_prob\tthreshold\tsuccess_ratio")

  thresholds_to_match_success_ratio =
    map(target_success_ratios[event_name]) do (nominal_prob, target_success_ratio)
      # Can't binary search, not monotonic.
      # Backtrack when SR exceeded
      threshold = 0.0f0
      step      = 0.02f0
      while step > 0.000001f0
        sr = Metrics.success_ratio(ŷ, y, weights, threshold)
        if isnan(sr) || sr > target_success_ratio
          step *= 0.5f0
          threshold -= step
        else
          threshold += step
        end
      end
      # println("$nominal_prob\t$threshold\t$(success_ratio(ŷ, y, weights, threshold))")
      threshold
    end

  # println("nominal_prob\tthreshold\tPOD")

  thresholds_to_match_POD =
    map(target_PODs[event_name]) do (nominal_prob, target_POD)
      threshold = 0.5f0
      step = 0.25f0
      while step > 0.000001f0
        pod = Metrics.probability_of_detection(ŷ, y, weights, threshold)
        if isnan(pod) || pod > target_POD
          threshold += step
        else
          threshold -= step
        end
        step *= 0.5f0
      end
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  thresholds_to_match_warning_ratio =
    map(target_warning_ratios[event_name]) do (nominal_prob, target_warning_ratio)
      threshold = 0.5f0
      step = 0.25f0
      while step > 0.000001f0
        wr = Metrics.warning_ratio(ŷ, weights, threshold)
        if isnan(wr) || wr > target_warning_ratio
          threshold += step
        else
          threshold -= step
        end
        step *= 0.5f0
      end
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_success_ratio = thresholds_to_match_success_ratio[i]
    threshold_to_match_POD = thresholds_to_match_POD[i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    mean_threshold = (threshold_to_match_success_ratio + threshold_to_match_POD + threshold_to_match_warning_ratio) / 3f0
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, mean_threshold))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, mean_threshold))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, mean_threshold))
    println("$event_name\t$nominal_prob\t$threshold_to_match_success_ratio\t$threshold_to_match_POD\t$threshold_to_match_warning_ratio\t$mean_threshold\t$sr\t$pod\t$wr")
    push!(thresholds, (Float32(nominal_prob), Float32(mean_threshold)))
  end

  thresholds
end


println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tmean_threshold\tSR\tPOD\tWR")
calibrations_sr_pod = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  calibrations_sr_pod[event_name] = spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tthreshold_to_match_warning_ratio\tmean_threshold\tSR\tPOD\tWR")
calibrations_all = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  event_name, _ = HREFPredictionAblations.models[prediction_i]
  calibrations_all[event_name] = spc_calibrate_all(prediction_i, X, Ys, weights)
end

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD mean_threshold SR          POD          WR
# tornado     0.02         0.01297058                       0.019601822            0.016286202    0.05811968  0.7255002    0.02133395
# tornado     0.05         0.05831237                       0.079683304            0.06899784     0.13066258  0.4559308    0.0059635467
# tornado     0.1          0.16381283                       0.15509605             0.15945444     0.21934602  0.14991193   0.0011680551
# tornado     0.15         0.21046689                       0.26961708             0.24004199     0.30903193  0.06797499   0.00037592635
# tornado     0.3          0.3783576                        0.3603382              0.3693479      0.2993795   0.008200346  4.681306e-5
# tornado     0.45         0.63601375                       0.5975361              0.6167749      0.54525286  0.0006632183 2.0788132e-6
# wind        0.05         0.025938107                      0.0877018              0.056819953    0.19041657  0.8146683    0.05523891
# wind        0.15         0.11240294                       0.25773048             0.18506671     0.34040433  0.5379147    0.020402689
# wind        0.3          0.31343934                       0.521204               0.41732168     0.5510731   0.19818887   0.004643432
# wind        0.45         0.5346906                        0.8000431              0.66736686     0.769832    0.042777162  0.0007174391
# hail        0.05         0.015346069                      0.04394722             0.029646644    0.106049754 0.8351591    0.044649545
# hail        0.15         0.07387878                       0.13708305             0.10548092     0.19760264  0.52892953   0.015176184
# hail        0.3          0.20529476                       0.3468647              0.2760797      0.37571222  0.16215828   0.0024470412
# hail        0.45         0.5594928                        0.59866524             0.57907903     0.58626467  0.009759113  9.437872e-5
# sig_tornado 0.1          0.040588986                      0.07995033             0.06026966     0.12599954  0.3443003    0.0006377945
# sig_wind    0.1          0.098119505                      0.106687546            0.10240352     0.13775808  0.14839154   0.0016767136
# sig_hail    0.1          0.059057                         0.051927567            0.055492282    0.08464337  0.25743744   0.0024400598

# this is the one we are using
# event_name  nominal_prob threshold_to_match_warning_ratio SR          POD          WR
# tornado     0.02         0.017892838                      0.06162817  0.7073454    0.019615944
# tornado     0.05         0.07787514                       0.14183877  0.4307434    0.0051901583
# tornado     0.1          0.17152214                       0.2464791   0.14091049   0.0009770577
# tornado     0.15         0.2814541                        0.26955867  0.026589876  0.0001685854
# tornado     0.3          0.3905239                        0.34836945  0.007654115  3.7550166e-5
# tornado     0.45         0.6009083                        0.47030112  0.0008843078 3.213545e-6
# wind        0.05         0.051660538                      0.18246952  0.8284546    0.058620222
# wind        0.15         0.21513557                       0.3675667   0.48940307   0.017190937
# wind        0.3          0.49578285                       0.6181463   0.13029508   0.0027214838
# wind        0.45         0.78172493                       0.8917094   0.020278241  0.0002936135
# hail        0.05         0.030927658                      0.108049504 0.8288017    0.04348959
# hail        0.15         0.12172127                       0.21620022  0.47577965   0.012476915
# hail        0.3          0.33656883                       0.41593274  0.09215542   0.0012561898
# hail        0.45         0.61953926                       0.6029355   0.007252866  6.820187e-5
# sig_tornado 0.1          0.063589096                      0.129504    0.33463305   0.00060311204
# sig_wind    0.1          0.11205864                       0.15010695  0.11518542   0.0011944375
# sig_hail    0.1          0.057775497                      0.08727195  0.242912     0.0022330375

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD threshold_to_match_warning_ratio mean_threshold SR         POD          WR
# tornado     0.02         0.01297058                       0.019601822            0.017892838                      0.016821748    0.05935366 0.7197565    0.020725023
# tornado     0.05         0.05831237                       0.079683304            0.07787514                       0.07195694     0.13404815 0.44696563   0.0056986273
# tornado     0.1          0.16381283                       0.15509605             0.17152214                       0.163477       0.22779244 0.1466783    0.0011004834
# tornado     0.15         0.21046689                       0.26961708             0.2814541                        0.25384602     0.30016658 0.051623434  0.00029392834
# tornado     0.3          0.3783576                        0.3603382              0.3905239                        0.37640658     0.31507045 0.007982044  4.3297554e-5
# tornado     0.45         0.63601375                       0.5975361              0.6009083                        0.611486       0.4998252  0.0006632183 2.2677507e-6
# wind        0.05         0.025938107                      0.0877018              0.051660538                      0.055100147    0.1878509  0.8194134    0.056319505
# wind        0.15         0.11240294                       0.25773048             0.21513557                       0.19508965     0.349676   0.52065736   0.019224508
# wind        0.3          0.31343934                       0.521204               0.49578285                       0.4434754      0.5752322  0.17310232   0.0038853372
# wind        0.45         0.5346906                        0.8000431              0.78172493                       0.70548624     0.8269018  0.03264143   0.00050966436
# hail        0.05         0.015346069                      0.04394722             0.030927658                      0.030073648    0.10670167 0.8331348    0.04426918
# hail        0.15         0.07387878                       0.13708305             0.12172127                       0.11089436     0.20442617 0.509736     0.014137293
# hail        0.3          0.20529476                       0.3468647              0.33656883                       0.29624274     0.38949108 0.13536102   0.0019703961
# hail        0.45         0.5594928                        0.59866524             0.61953926                       0.5925658      0.58640826 0.008812057  8.519903e-5
# sig_tornado 0.1          0.040588986                      0.07995033             0.063589096                      0.06137614     0.1267123  0.3394527    0.0006252775
# sig_wind    0.1          0.098119505                      0.106687546            0.11205864                       0.1056219      0.14019963 0.13421097   0.0014900743
# sig_hail    0.1          0.059057                         0.051927567            0.057775497                      0.056253355    0.0856354  0.25276944   0.002368061


println(calibrations_sr_pod)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.055492282)], "hail" => [(0.05, 0.029646644), (0.15, 0.10548092), (0.3, 0.2760797), (0.45, 0.57907903)], "tornado" => [(0.02, 0.016286202), (0.05, 0.06899784), (0.1, 0.15945444), (0.15, 0.24004199), (0.3, 0.3693479), (0.45, 0.6167749)], "sig_tornado" => [(0.1, 0.06026966)], "sig_wind" => [(0.1, 0.10240352)], "wind" => [(0.05, 0.056819953), (0.15, 0.18506671), (0.3, 0.41732168), (0.45, 0.66736686)])

println(calibrations_wr)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.057775497)], "hail" => [(0.05, 0.030927658), (0.15, 0.12172127), (0.3, 0.33656883), (0.45, 0.61953926)], "tornado" => [(0.02, 0.017892838), (0.05, 0.07787514), (0.1, 0.17152214), (0.15, 0.2814541), (0.3, 0.3905239), (0.45, 0.6009083)], "sig_tornado" => [(0.1, 0.063589096)], "sig_wind" => [(0.1, 0.11205864)], "wind" => [(0.05, 0.051660538), (0.15, 0.21513557), (0.3, 0.49578285), (0.45, 0.78172493)])

println(calibrations_all)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.056253355)], "hail" => [(0.05, 0.030073648), (0.15, 0.11089436), (0.3, 0.29624274), (0.45, 0.5925658)], "tornado" => [(0.02, 0.016821748), (0.05, 0.07195694), (0.1, 0.163477), (0.15, 0.25384602), (0.3, 0.37640658), (0.45, 0.611486)], "sig_tornado" => [(0.1, 0.06137614)], "sig_wind" => [(0.1, 0.1056219)], "wind" => [(0.05, 0.055100147), (0.15, 0.19508965), (0.3, 0.4434754), (0.45, 0.70548624)])


# using the warning ratio calibrations rn
