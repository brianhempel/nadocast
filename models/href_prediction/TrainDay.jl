import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 628

validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, validation_forecasts);
length(validation_forecasts_0z) # 157

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
  "tornado"     => (forecast -> compute_day_labels(StormEvents.conus_tornado_events(),     forecast)),
  "wind"        => (forecast -> compute_day_labels(StormEvents.conus_severe_wind_events(), forecast)),
  "hail"        => (forecast -> compute_day_labels(StormEvents.conus_severe_hail_events(), forecast)),
  "sig_tornado" => (forecast -> compute_day_labels(StormEvents.conus_sig_tornado_events(), forecast)),
  "sig_wind"    => (forecast -> compute_day_labels(StormEvents.conus_sig_wind_events(),    forecast)),
  "sig_hail"    => (forecast -> compute_day_labels(StormEvents.conus_sig_hail_events(),    forecast)),
)

# rm("day_accumulators_validation_forecasts_0z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts_0z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_accumulators_validation_forecasts_0z",
  );



# should do some checks here.
import PlotMap

dec11 = filter(f -> Forecasts.time_title(f) == "2021-12-11 00Z +35", validation_forecasts_0z)[1];
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  PlotMap.plot_debug_map("dec11_0z_day_accs_$i", dec11.grid, dec11_data[:,i]);
end
for (event_name, labeler) in event_name_to_day_labeler
  dec11_labels = event_name_to_day_labeler[event_name](dec11);
  PlotMap.plot_debug_map("dec11_0z_day_$event_name", dec11.grid, dec11_labels);
end
# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_tornado.pdf ./


# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    event_name, _ = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z, X, Ys, weights)

# tornado (9446.0)     feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.12680832038522866
# tornado (9446.0)     feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.11642798952925312
# wind (72111.0)       feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.38313786770146624
# wind (72111.0)       feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.3597382825677279
# hail (31894.0)       feature 5 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.2281233952093273
# hail (31894.0)       feature 6 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.21234088728521894
# sig_tornado (1268.0) feature 7 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.09055666618983317
# sig_tornado (1268.0) feature 8 highest hourly STORPROB:calculated:day            fcst:: AU-PR-curve: 0.08725820073587578
# sig_wind (8732.0)    feature 9 independent events total SWINDPRO:calculated:day  fcst:: AU-PR-curve: 0.0777788483628452
# sig_wind (8732.0)    feature 10 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.075931385644641
# sig_hail (4478.0)    feature 11 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.06721734789273422
# sig_hail (4478.0)    feature 12 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.05879058492533432




# 3. bin predictions into 4 bins of equal weight of positive labels

const bin_count = 4

function find_ŷ_bin_splits(event_name, ŷ, Ys, weights)
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

    println("$event_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
  end

  bins_max
end

event_types_count = length(HREFPrediction.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:event_types_count
  (event_name, _, model_name) = HREFPrediction.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# event_name  mean_y        mean_ŷ        Σweight     bin_max
# tornado     0.0004378549  0.00056639145 5.062415e6  0.017401028
# tornado     0.025866624   0.030743996   85690.64    0.057005595
# tornado     0.08508345    0.083929636   26056.912   0.13199422
# tornado     0.16652103    0.24147587    13304.882   1.0
# wind        0.0033698757  0.0045183003  4.968903e6  0.105925485
# wind        0.13308668    0.16104285    125818.55   0.24237353
# wind        0.2809759     0.317453      59594.195   0.41793627
# wind        0.5050372     0.55661386    33151.844   1.0
# hail        0.0014652495  0.0018948776  5.0184295e6 0.057583164
# hail        0.07475628    0.08927173    98359.9     0.13694991
# hail        0.15669605    0.18665451    46926.02    0.26336262
# hail        0.3095211     0.38435712    23752.223   1.0
# sig_tornado 5.8779195e-5  8.302416e-5   5.1578135e6 0.00944276
# sig_tornado 0.015303657   0.017065546   19832.14    0.03155332
# sig_tornado 0.03825248    0.057183858   7928.9473   0.1166033
# sig_tornado 0.15892082    0.1726563     1892.8201   1.0
# sig_wind    0.00040164683 0.0004937433  5.026788e6  0.014537211
# sig_wind    0.018932946   0.025429603   106628.81   0.044868514
# sig_wind    0.05650823    0.060135435   35736.594   0.080929644
# sig_wind    0.11015707    0.10808131    18314.014   1.0
# sig_hail    0.00020400203 0.0002989909  5.1035735e6 0.017137118
# sig_hail    0.02375851    0.023858236   43831.246   0.032750417
# sig_hail    0.036847323   0.046266176   28253.066   0.06910927
# sig_hail    0.08791025    0.10530451    11809.844   1.0


println("event_to_0z_day_bins = $event_to_day_bins")
# event_to_0z_day_bins = Dict{String, Vector{Float32}}("sig_hail" => [0.017137118, 0.032750417, 0.06910927, 1.0], "hail" => [0.057583164, 0.13694991, 0.26336262, 1.0], "tornado" => [0.017401028, 0.057005595, 0.13199422, 1.0], "sig_tornado" => [0.00944276, 0.03155332, 0.1166033, 1.0], "sig_wind" => [0.014537211, 0.044868514, 0.080929644, 1.0], "wind" => [0.105925485, 0.24237353, 0.41793627, 1.0])










# 4. combine bin-pairs (overlapping, 3 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms


function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  bins_max = event_to_day_bins[event_name]
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
      ("event_name", event_name),
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
for prediction_i in 1:event_types_count
  event_name, _ = HREFPrediction.models[prediction_i]

  event_to_day_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name  bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado     1-2 -1.0             0.057005595      5627908 4772.0    5.148106e6  0.0010687001      0.0002511296      0.0008611188  0.0050174943       0.0055120285       0.02417658       0.02172301       0.0008611187    0.0049950834     0.024044603    Float32[0.93318164,   0.06823707,   -0.10806717]
# tornado     2-3 0.017401028      0.13199422       119297  4718.0    111747.555  0.04314564        0.01091816        0.0396746     0.15748583         0.18396132         0.09766345       0.07593608       0.039674606     0.15699896       0.09769626     Float32[1.3238057,    -0.114339165, 0.34774518]
# tornado     3-4 0.057005595      1.0              41362   4674.0    39361.797   0.13718264        0.035017274       0.11261058    0.34227386         0.4042841          0.20022316       0.18475614       0.11261056      0.33662492       0.20026325     Float32[0.6581387,    0.06536053,   -0.60667735]
# wind        1-2 -1.0             0.24237353       5569373 36113.0   5.0947215e6 0.008383809       0.0020293228      0.0065733446  0.025187163        0.028603999        0.12886713       0.115364924      0.0065733446    0.02494336       0.12816711     Float32[0.9533327,    0.08198542,   -0.049861502]
# wind        2-3 0.105925485      0.41793627       199914  36148.0   185412.75   0.2113152         0.054536834       0.18062028    0.45407182         0.5633424          0.28151384       0.24929999       0.18062028      0.45087236       0.28202492     Float32[1.1315389,    -0.06399821,  -0.22480214]
# wind        3-4 0.24237353       1.0              99897   35998.0   92746.05    0.40294045        0.11670191        0.361066      0.6175525          0.84184647         0.54253256       0.5127974        0.361066        0.61342573       0.5430452      Float32[0.9192797,    0.07381143,   -0.066281155]
# hail        1-2 -1.0             0.13694991       5592431 15961.0   5.116789e6  0.0035745208      0.0008128005      0.002874121   0.012652684        0.014255003        0.070959955      0.06905122       0.002874121     0.012551144      0.07130111     Float32[0.78442734,   0.27793196,   0.407107]
# hail        2-3 0.057583164      0.26336262       157679  15940.0   145285.9    0.12072548        0.02856196        0.10122208    0.31994775         0.38296047         0.1576388        0.1397978        0.10122208      0.31793103       0.16050415     Float32[1.0553415,    -0.1081846,   -0.47820166]
# hail        3-4 0.13694991       1.0              76839   15933.0   70678.24    0.2530947         0.06851147        0.20805466    0.49648744         0.61157775         0.32870942       0.3037502        0.20805463      0.4880406        0.33385178     Float32[1.2830826,    -0.47052482,  -1.2408078]
# sig_tornado 1-2 -1.0             0.03155332       5659086 644.0     5.177646e6  0.00014807297     3.8360264e-5      0.00011717224 0.00077690964      0.0008255258       0.014033859      0.014120079      0.00011717227   0.0007706992     0.014496199    Float32[0.5930697,    0.4203289,    0.43342784]
# sig_tornado 2-3 0.00944276       0.1166033        29030   629.0     27761.088   0.028523887       0.008058258       0.021858158   0.10256433         0.10823592         0.05090796       0.055792473      0.021858154     0.09987361       0.055865478    Float32[-0.032218266, 0.8780866,    0.3627351]
# sig_tornado 3-4 0.03155332       1.0              10184   624.0     9821.768    0.07943734        0.021637168       0.061507307   0.21461318         0.23677906         0.15210854       0.14430982       0.061507307     0.20862293       0.15645918     Float32[0.6223907,    0.6945181,    1.4135333]
# sig_wind    1-2 -1.0             0.044868514      5610943 4365.0    5.133417e6  0.0010116986      0.00022963232     0.00078656984 0.0046597593       0.005062965        0.020260252      0.018085005      0.0007865698    0.004622475      0.019494735    Float32[0.53879964,   0.4059111,    0.116683125]
# sig_wind    2-3 0.014537211      0.080929644      153594  4362.0    142365.4    0.03414147        0.008040776       0.0283651     0.12389084         0.13986644         0.056721367      0.0615863        0.0283651       0.12294913       0.059668314    Float32[1.0535268,    0.2508744,    1.1575938]
# sig_wind    3-4 0.044868514      1.0              58327   4367.0    54050.61    0.07638098        0.01728467        0.07468611    0.2593019          0.31307268         0.116249576      0.115519434      0.07468612      0.25815567       0.118530795    Float32[0.619009,     0.59856415,   1.4483397]
# sig_hail    1-2 -1.0             0.032750417      5625917 2244.0    5.147405e6  0.0004996029      0.0001211714      0.00040457366 0.0023200924       0.0025646547       0.021306612      0.016311225      0.00040457366   0.0022970182     0.024614729    Float32[1.7509274,    -0.6264722,   -0.59064865]
# sig_hail    2-3 0.017137118      0.06910927       78179   2241.0    72084.31    0.03264091        0.0077077076      0.028888602   0.12973951         0.14995398         0.04186439       0.035388827      0.028888598     0.12867774       0.043021       Float32[1.5742372,    -0.7514552,   -1.8797148]
# sig_hail    3-4 0.032750417      1.0              43353   2234.0    40062.91    0.06366964        0.016053006       0.051899783   0.19876958         0.22689283         0.100673005      0.08796213       0.051899783     0.19708908       0.10304778     Float32[1.4796587,    -0.48270363,  -0.9502378]


println("event_to_0z_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.7509274, -0.6264722, -0.59064865], [1.5742372, -0.7514552, -1.8797148], [1.4796587, -0.48270363, -0.9502378]], "hail" => [[0.78442734, 0.27793196, 0.407107], [1.0553415, -0.1081846, -0.47820166], [1.2830826, -0.47052482, -1.2408078]], "tornado" => [[0.93318164, 0.06823707, -0.10806717], [1.3238057, -0.114339165, 0.34774518], [0.6581387, 0.06536053, -0.60667735]], "sig_tornado" => [[0.5930697, 0.4203289, 0.43342784], [-0.032218266, 0.8780866, 0.3627351], [0.6223907, 0.6945181, 1.4135333]], "sig_wind" => [[0.53879964, 0.4059111, 0.116683125], [1.0535268, 0.2508744, 1.1575938], [0.619009, 0.59856415, 1.4483397]], "wind" => [[0.9533327, 0.08198542, -0.049861502], [1.1315389, -0.06399821, -0.22480214], [0.9192797, 0.07381143, -0.066281155]])




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
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
length(day_validation_forecasts_0z) # Expected: 157
#

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

# rm("day_validation_forecasts_0z_with_sig_gated"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_with_sig_gated",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z, X, Ys, weights)



function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, HREFPrediction.models_with_gated))

  # Feature order is all HREF severe probs then all SREF severe probs
  for event_name in event_names
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      y = Ys[event_name]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z, X, Ys, weights)





# rm("day_accumulators_validation_forecasts_0z"; recursive = true)

# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  total_weight = sum(Float64.(weights))

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
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
test_calibration(day_validation_forecasts_0z, X, Ys, weights)






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
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
  event_name, _ = HREFPrediction.models[prediction_i]
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
  event_name, _ = HREFPrediction.models[prediction_i]
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
  event_name, _ = HREFPrediction.models[prediction_i]
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
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_sr_pod[event_name] = spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tthreshold_to_match_warning_ratio\tmean_threshold\tSR\tPOD\tWR")
calibrations_all = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_all[event_name] = spc_calibrate_all(prediction_i, X, Ys, weights)
end



println(calibrations_sr_pod)

println(calibrations_wr)

println(calibrations_all)

# using the warning ratio calibrations rn
