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
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_fourhourly_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 16328

@time Forecasts.data(validation_forecasts[10]) # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


compute_fourhourly_labels(events, forecast) = begin
  # The original hourlies are ±30min, so four consecutive forecasts is -3:30 to +0:30 from the last valid time.
  end_seconds   = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  start_seconds = end_seconds - 4*HOUR
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime   = Dates.unix2datetime(end_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_fourhourly_labeler = Dict(
  "tornado"     => (forecast -> compute_fourhourly_labels(StormEvents.conus_tornado_events(),     forecast)),
  "wind"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_wind_events(), forecast)),
  "hail"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_hail_events(), forecast)),
  "sig_tornado" => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_tornado_events(), forecast)),
  "sig_wind"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_wind_events(),    forecast)),
  "sig_hail"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_hail_events(),    forecast)),
)

# rm("four-hourly_accumulators_validation_forecasts"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts;
    event_name_to_labeler = event_name_to_fourhourly_labeler,
    save_dir = "four-hourly_accumulators_validation_forecasts",
  );



# should do some checks here.
import PlotMap

(train_forecasts, _, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_fourhourly_accumulators(); just_hours_near_storm_events = false);
dec10 = filter(f -> Forecasts.time_title(f) == "2021-12-10 12Z +19", train_forecasts)[1];
dec10_data = Forecasts.data(dec10);

for i in 1:size(dec10_data,2)
  PlotMap.plot_debug_map("dec10_12Z_f19_four-hourly_accs_$i", dec10.grid, dec10_data[:,i]);
end
for (event_name, labeler) in event_name_to_fourhourly_labeler
  dec10_labels = event_name_to_fourhourly_labeler[event_name](dec10);
  PlotMap.plot_debug_map("dec10_12Z_f19_four-hourly_$event_name", dec10.grid, dec10_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec10_12Z_f19_four-hourly_accs_1.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec10_12Z_f19_four-hourly_tornado.pdf ./


# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (183930.0)    feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.09770359069227193
# tornado (183930.0)    feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.09639019463314036
# wind (1.480368e6)     feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.27048484658639294
# wind (1.480368e6)     feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.255617726417922
# hail (655960.0)       feature 5 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.15824351640016804
# hail (655960.0)       feature 6 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.14795403998953127
# sig_tornado (26911.0) feature 7 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.11078754977632764
# sig_tornado (26911.0) feature 8 highest hourly STORPROB:calculated:day            fcst:: AU-PR-curve: 0.10679188249598848
# sig_wind (166548.0)   feature 9 independent events total SWINDPRO:calculated:day  fcst:: AU-PR-curve: 0.06078920550898103
# sig_wind (166548.0)   feature 10 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.05776580264860349
# sig_hail (85061.0)    feature 11 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.05066403688398091
# sig_hail (85061.0)    feature 12 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.0481790491190767





# # 3. bin predictions into 4 bins of equal weight of positive labels

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

event_types_count = length(CombinedHREFSREF.models)
event_to_fourhourly_bins = Dict{String,Vector{Float32}}()
println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:event_types_count
  (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_fourhourly_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_fourhourly_bins[\"$event_name\"] = $(event_to_fourhourly_bins[event_name])")
end

# event_name  mean_y        mean_ŷ       Σweight     bin_max
# tornado     8.0768514e-5  9.449162e-5  5.3470163e8 0.00753908
# tornado     0.012443697   0.014732003  3.4706212e6 0.030358605
# tornado     0.044014603   0.04937559   981196.1    0.08693349
# tornado     0.12584485    0.14870805   343164.34   1.0
# wind        0.000647787   0.0007410317 5.3108173e8 0.043117322
# wind        0.06417034    0.07396143   5.361171e6  0.12710203
# wind        0.16235676    0.18382843   2.1189708e6 0.2744185
# wind        0.36804596    0.41048136   934732.8    1.0
# hail        0.00028496954 0.0003171573 5.319337e8  0.020449053
# hail        0.030311637   0.03531422   5.000868e6  0.0625404
# hail        0.08210492    0.09509294   1.8462332e6 0.15308933
# hail        0.21175805    0.25883663   715829.8    1.0
# sig_tornado 1.1980959e-5  1.474753e-5  5.385473e8  0.00470224
# sig_tornado 0.009347465   0.01013833   690232.1    0.022806743
# sig_tornado 0.030583877   0.04166681   210958.39   0.08246323
# sig_tornado 0.1340592     0.1552074    48117.406   1.0
# sig_wind    7.230618e-5   7.799446e-5  5.3350173e8 0.00543831
# sig_wind    0.009082949   0.010429465  4.2470395e6 0.021594528
# sig_wind    0.030344412   0.033370398  1.2712386e6 0.05260824
# sig_wind    0.080931164   0.08111184   476620.47   1.0
# sig_hail    3.688956e-5   4.026061e-5  5.3606326e8 0.006030224
# sig_hail    0.008938219   0.010017799  2.2124185e6 0.016691525
# sig_hail    0.021739667   0.025007816  909645.7    0.04071429
# sig_hail    0.06352594    0.07042984   311276.28   1.0


println("event_to_fourhourly_bins = $event_to_fourhourly_bins")
# event_to_fourhourly_bins = Dict{String, Vector{Float32}}("sig_hail" => [0.006030224, 0.016691525, 0.04071429, 1.0], "hail" => [0.020449053, 0.0625404, 0.15308933, 1.0], "tornado" => [0.00753908, 0.030358605, 0.08693349, 1.0], "sig_tornado" => [0.00470224, 0.022806743, 0.08246323, 1.0], "sig_wind" => [0.00543831, 0.021594528, 0.05260824, 1.0], "wind" => [0.043117322, 0.12710203, 0.2744185, 1.0])


# 4. combine bin-pairs (overlapping, 3 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + b)


function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  bins_max = event_to_fourhourly_bins[event_name]
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
      ("total_prob_au_pr", Metrics.area_under_pr_curve(bin_total_prob_x, bin_y, bin_weights)),
      ("max_hourly_au_pr", Metrics.area_under_pr_curve(bin_max_hourly_x, bin_y, bin_weights)),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights)),
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

event_to_fourhourly_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:event_types_count
  event_name, _ = CombinedHREFSREF.models[prediction_i]

  event_to_fourhourly_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name  bin total_prob_ŷ_min total_prob_ŷ_max count     pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr     max_hourly_au_pr      mean_logistic_ŷ logistic_logloss logistic_au_pr       logistic_coeffs
# tornado     1-2 -1.0             0.030358605      588215075 92930.0   5.381723e8  0.00018888751     8.261044e-5       0.00016049582 0.0010562381       0.0010914825       0.012880728276687427 0.0114017162663552    0.00016049582   0.0010538345     0.012964571243015302 Float32[1.1338079,  -0.12365579,  -0.21844162]
# tornado     2-3 0.00753908       0.08693349       4729573   91926.0   4.451817e6  0.022367574       0.009774388       0.019402036   0.09081951         0.095443204        0.046241878127403595 0.04104723399680213   0.01940203      0.09056037       0.04626996591630185  Float32[1.2712703,  -0.23105474,  -0.21188106]
# tornado     3-4 0.030358605      1.0              1389005   91000.0   1.3243605e6 0.07511432        0.032344993       0.06521821    0.226359           0.24114573         0.16624221486915353  0.1655198067466649    0.06521821      0.22556075       0.16722579225602965  Float32[0.88705343, 0.104377,     -0.08031504]`
# wind        1-2 -1.0             0.12710203       586318395 740818.0  5.3644294e8 0.0014727908      0.0006772136      0.0012826266  0.0059958207       0.006255835        0.0634350126717256   0.0579178901383007    0.0012826269    0.0059821764     0.06333398518097927  Float32[0.94788873, 0.04973387,   -0.11124777]
# wind        2-3 0.043117322      0.2744185        8046332   741864.0  7.480142e6  0.1050845         0.04746717        0.09198453    0.2934906          0.3144947          0.16429322694500617  0.147557977964065     0.09198455      0.29248342       0.16478742302734267  Float32[1.0852667,  -0.08753359,  -0.23767674]
# wind        3-4 0.12710203       1.0              3285685   739550.0  3.0537035e6 0.25320646        0.117020406       0.22531784    0.49654448         0.55196434         0.41797704180005696  0.3947048351583871    0.22531784      0.4941526        0.41797863324096923  Float32[1.080012,   -0.108289875, -0.3081989]`
# hail        1-2 -1.0             0.0625404        586831796 327792.0  5.369345e8  0.0006431109      0.00028537703     0.00056463014 0.003055213        0.003181994        0.030411082758001258 0.027365776110687415  0.00056463      0.0030500775     0.0303544737945698   Float32[0.9889517,  0.015257805,  -0.10187187]
# hail        2-3 0.020449053      0.15308933       7401637   327293.0  6.8471015e6 0.051432785       0.022254124       0.04427703    0.17440484         0.18488975         0.08339872992566728  0.07391672782123569   0.04427704      0.17378087       0.08407567571896929  Float32[1.1896296,  -0.2049748,   -0.39164874]
# hail        3-4 0.0625404        1.0              2772284   328168.0  2.562063e6  0.14084226        0.06220434        0.11832947    0.34257683         0.36795664         0.25461772933915733  0.2375710856842236    0.11832947      0.33978638       0.25510685984520215  Float32[1.1730652,  -0.2797999,   -0.66866827]`
# sig_tornado 1-2 -1.0             0.022806743      589336522 13645.0   5.392375e8  2.770587e-5       1.2680617e-5      2.3930521e-5  0.00017395488      0.00017939939      0.00930849627642124  0.008276901922942337  2.3930514e-5    0.00017340003    0.009179143720860712 Float32[1.5224317,  -0.49362254,  -0.39044833]
# sig_tornado 2-3 0.00470224       0.08246323       938756    13328.0   901190.5    0.017518789       0.007952066       0.0143186655  0.07190923         0.07428777         0.03270552694412787  0.030871684652317155  0.0143186655    0.071487755      0.03269514527174705  Float32[0.96165395, -0.110244565, -0.8577433]
# sig_tornado 3-4 0.022806743      1.0              267558    13266.0   259075.8    0.06275438        0.028716482       0.049802054   0.17904362         0.18571186         0.20013897594503147  0.19351258073808883   0.04980205      0.17727081       0.20071268275390963  Float32[0.94144565, 0.13474676,   0.0427606]`
# sig_wind    1-2 -1.0             0.021594528      587718585 83350.0   5.3774874e8 0.00015974844     7.165392e-5       0.00014347055 0.0009682866       0.0010014552       0.009521284392622956 0.008708883787359414  0.00014347058   0.00096738327    0.009485282122216315 Float32[0.9259525,  0.06745399,   -0.08790273]
# sig_wind    2-3 0.00543831       0.05260824       5941704   83149.0   5.518278e6  0.01571434        0.006982215       0.013980924   0.070463024        0.07363769         0.03009352917862939  0.02853544552718887   0.013980924     0.070358105      0.030175168998843902 Float32[0.97327083, 0.05183811,   0.023566378]
# sig_wind    3-4 0.021594528      1.0              1885495   83198.0   1.7478591e6 0.046388924       0.019271692       0.04413882    0.17292258         0.18615338         0.10091091451233196  0.09610471179421629   0.044138823     0.17278062       0.10117130946685401  Float32[0.9228902,  0.17639892,   0.3989572]`
# sig_hail    1-2 -1.0             0.016691525      588288657 42568.0   5.382757e8  8.127025e-5       3.756e-5          7.347576e-5   0.00048597963      0.00050454645      0.008692978734457764 0.0071978372439680536 7.347576e-5     0.00048524616    0.008942354730121666 Float32[1.326802,   -0.31069145,  -0.27217078]
# sig_hail    2-3 0.006030224      0.04071429       3373715   42657.0   3.1220642e6 0.014385297       0.00610205        0.012668054   0.06641107         0.069916904        0.022295979063247498 0.019454365032167225  0.012668058     0.06618348       0.02248221186115115  Float32[1.4527221,  -0.5298689,   -0.9338967]
# sig_hail    3-4 0.016691525      1.0              1315423   42493.0   1.220922e6  0.036588244       0.015088901       0.03239315    0.13640158         0.14503749         0.08393107876659227  0.08078457811978787   0.032393154     0.13611495       0.08352211713438026  Float32[1.1630019,  -0.154732,    -0.25065106]


print("event_to_fourhourly_bins_logistic_coeffs = $event_to_fourhourly_bins_logistic_coeffs")
# event_to_fourhourly_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.326802, -0.31069145, -0.27217078], [1.4527221, -0.5298689, -0.9338967], [1.1630019, -0.154732, -0.25065106]], "hail" => [[0.9889517, 0.015257805, -0.10187187], [1.1896296, -0.2049748, -0.39164874], [1.1730652, -0.2797999, -0.66866827]], "tornado" => [[1.1338079, -0.12365579, -0.21844162], [1.2712703, -0.23105474, -0.21188106], [0.88705343, 0.104377, -0.08031504]], "sig_tornado" => [[1.5224317, -0.49362254, -0.39044833], [0.96165395, -0.110244565, -0.8577433], [0.94144565, 0.13474676, 0.0427606]], "sig_wind" => [[0.9259525, 0.06745399, -0.08790273], [0.97327083, 0.05183811, 0.023566378], [0.9228902, 0.17639892, 0.3989572]], "wind" => [[0.94788873, 0.04973387, -0.11124777], [1.0852667, -0.08753359, -0.23767674], [1.080012, -0.108289875, -0.3081989]])






# # 6. prediction is weighted mean of the two overlapping logistic models
# # 7. predictions should thereby be calibrated (check)



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
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, fourhourly_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_fourhourly_with_sig_gated(); just_hours_near_storm_events = false);

length(fourhourly_validation_forecasts)

# # We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
fourhourly_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, fourhourly_validation_forecasts);

length(fourhourly_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(fourhourly_validation_forecasts[10])


compute_fourhourly_labels(events, forecast) = begin
  # The original hourlies are ±30min, so four consecutive forecasts is -3:30 to +0:30 from the last valid time.
  end_seconds   = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  start_seconds = end_seconds - 4*HOUR
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime   = Dates.unix2datetime(end_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_fourhourly_labeler = Dict(
  "tornado"     => (forecast -> compute_fourhourly_labels(StormEvents.conus_tornado_events(),     forecast)),
  "wind"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_wind_events(), forecast)),
  "hail"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_hail_events(), forecast)),
  "sig_tornado" => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_tornado_events(), forecast)),
  "sig_wind"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_wind_events(),    forecast)),
  "sig_hail"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_hail_events(),    forecast)),
)

# rm("four-hourly_validation_forecasts_with_sig_gated"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    fourhourly_validation_forecasts;
    event_name_to_labeler = event_name_to_fourhourly_labeler,
    save_dir = "four-hourly_validation_forecasts_with_sig_gated",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(fourhourly_validation_forecasts, X, Ys, weights)



# # rm("day_accumulators_validation_forecasts_0z"; recursive = true)

# # test y vs ŷ

# function test_calibration(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   total_weight = sum(Float64.(weights))

#   println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
#   for feature_i in 1:length(inventory)
#     prediction_i = feature_i
#     (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
#     y = Ys[event_name]
#     ŷ = @view X[:, feature_i]

#     total_pos_weight = sum(Float64.(y .* weights))

#     sort_perm      = Metrics.parallel_sort_perm(ŷ);
#     y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
#     ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
#     weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

#     bin_count = 20
#     per_bin_pos_weight = Float64(sum(y .* weights)) / bin_count

#     # bins = map(_ -> Int64[], 1:bin_count)
#     bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
#     bins_Σy      = map(_ -> 0.0, 1:bin_count)
#     bins_Σweight = map(_ -> 0.0, 1:bin_count)
#     bins_max     = map(_ -> 1.0f0, 1:bin_count)

#     bin_i = 1
#     for i in 1:length(y_sorted)
#       if ŷ_sorted[i] > bins_max[bin_i]
#         bin_i += 1
#       end

#       bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
#       bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
#       bins_Σweight[bin_i] += Float64(weights_sorted[i])

#       if bins_Σy[bin_i] >= per_bin_pos_weight
#         bins_max[bin_i] = ŷ_sorted[i]
#       end
#     end

#     for bin_i in 1:bin_count
#       Σŷ      = bins_Σŷ[bin_i]
#       Σy      = bins_Σy[bin_i]
#       Σweight = Float32(bins_Σweight[bin_i])

#       mean_ŷ = Float32(Σŷ / Σweight)
#       mean_y = Float32(Σy / Σweight)

#       pos_weight_in_and_after = sum(bins_Σy[bin_i:bin_count])
#       weight_in_and_after     = sum(bins_Σweight[bin_i:bin_count])

#       sr  = Float32(pos_weight_in_and_after / weight_in_and_after)
#       pod = Float32(pos_weight_in_and_after / total_pos_weight)

#       println("$model_name\t$mean_y\t$mean_ŷ\t$Σweight\t$sr\t$pod\t$(bins_max[bin_i])")
#     end
#   end
# end
# test_calibration(day_validation_forecasts_0z, X, Ys, weights)

# # event_name                   mean_y        mean_ŷ        Σweight     SR            POD         bin_max