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

length(fourhourly_validation_forecasts) # 19676

# # We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
fourhourly_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, fourhourly_validation_forecasts);

length(fourhourly_validation_forecasts) # 16328

# Make sure a forecast loads
@time Forecasts.data(fourhourly_validation_forecasts[10]);


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

# the gated sig predictors are worse, actually
# tornado (183930.0)                     feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  AU-PR-curve: 0.09828581160141926
# wind (1.480368e6)                      feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.2704869296357274
# hail (655960.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.1585022976082706
# sig_tornado (26911.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.1111356629131234
# sig_wind (166548.0)                    feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.06091301155354759
# sig_hail (85061.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.05056927129636332
# sig_tornado_gated_by_tornado (26911.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.10290330046070635
# sig_wind_gated_by_wind (166548.0)      feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.060701066315442954
# sig_hail_gated_by_hail (85061.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.05045019533120539

# vs accumulators:
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




# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  total_weight = sum(Float64.(weights))

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
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
test_calibration(fourhourly_validation_forecasts, X, Ys, weights)

# event_name                   mean_y        mean_ŷ        Σweight     SR            POD         bin_max
# tornado                      1.6833654e-5  1.7418653e-5  5.1313574e8 0.00032020002 1.0         0.00032649178
# tornado                      0.0006947653  0.0006233424  1.243274e7  0.0062254732  0.9499965   0.0011688905
# tornado                      0.0017943743  0.001714566   4.814045e6  0.011162379   0.8999936   0.002486952
# tornado                      0.0032447374  0.0032600276  2.662183e6  0.016110545   0.84998864  0.004256585
# tornado                      0.0052048503  0.0052333134  1.6595296e6 0.021419235   0.79998434  0.0064239
# tornado                      0.0076694516  0.007588201   1.1262029e6 0.027034046   0.74998283  0.008973025
# tornado                      0.009888774   0.010481356   873506.3    0.032982618   0.69998276  0.012261856
# tornado                      0.013504482   0.014080421   639658.06   0.040206056   0.6499795   0.016227018
# tornado                      0.018205948   0.018421128   474453.53   0.048139106   0.5999742   0.020970616
# tornado                      0.024290035   0.023521155   355617.94   0.05659992    0.5499711   0.026501307
# tornado                      0.029157776   0.029764192   296231.0    0.06528515    0.49996746  0.033458054
# tornado                      0.036922686   0.03732991    233933.33   0.07570888    0.4499669   0.04178811
# tornado                      0.04645741    0.046393186   185919.83   0.08715409    0.3999663   0.05165807
# tornado                      0.05952306    0.056888763   145119.31   0.09962233    0.34996623  0.062738284
# tornado                      0.07278508    0.06866687    118676.414  0.11222537    0.29996273  0.075343624
# tornado                      0.07986121    0.08379431    108161.66   0.12586947    0.24995966  0.09348917
# tornado                      0.09896828    0.104165286   87282.67    0.14705525    0.19995631  0.11701525
# tornado                      0.1259929     0.13185641    68558.59    0.17548986    0.14995125  0.15040454
# tornado                      0.17150733    0.17513418    50365.727   0.21841829    0.09994804  0.21163619
# tornado                      0.30079103    0.28874564    28683.09    0.30079103    0.049943704 1.0
# wind                         0.00013502034 0.00012646934 5.0959936e8 0.0025507296  1.0         0.003689402
# wind                         0.0063305628  0.006194306   1.0868918e7 0.04372656    0.94999945  0.009970632
# wind                         0.013251209   0.01341532    5.192404e6  0.065087035   0.8999988   0.017819172
# wind                         0.021297133   0.022028496   3.2307762e6 0.0845402     0.8499987   0.02709741
# wind                         0.031428084   0.031898413   2.189321e6  0.10380672    0.79999816  0.03747046
# wind                         0.042101834   0.04297191    1.6342761e6 0.12263554    0.7499977   0.049240578
# wind                         0.05478074    0.055409595   1.2560228e6 0.14204325    0.69999725  0.06234178
# wind                         0.06971907    0.0691559     986896.75   0.16187914    0.649997    0.07671535
# wind                         0.08453194    0.08440993    813958.8    0.18191878    0.599997    0.0929013
# wind                         0.10254335    0.101471536   670992.56   0.2032009     0.549997    0.1109725
# wind                         0.12162692    0.12071206    565715.0    0.22531866    0.49999672  0.13135372
# wind                         0.14148058    0.14234634    486327.22   0.24889617    0.44999623  0.15435714
# wind                         0.16759495    0.16654223    410551.6    0.2749944     0.39999598  0.17989483
# wind                         0.1948805     0.19369884    353066.7    0.30270696    0.34999532  0.2088437
# wind                         0.2253032     0.22467968    305392.66   0.3334577     0.2999951   0.24231453
# wind                         0.2608353     0.2618976     263792.38   0.36887348    0.24999477  0.28411585
# wind                         0.2971772     0.30974317    231531.08   0.41148457    0.19999415  0.33967203
# wind                         0.36836258    0.3736466     186789.86   0.47200516    0.14999396  0.414995
# wind                         0.4677881     0.46613675    147088.62   0.5492848     0.0999933   0.5334499
# wind                         0.66519076    0.65336794    103422.164  0.66519076    0.04999271  1.0
# hail                         5.9037153e-5  5.7065132e-5  5.135325e8  0.0011238939  1.0         0.0018457191
# hail                         0.0032605685  0.0031056488  9.298105e6  0.022185227   0.9499989   0.004997519
# hail                         0.006975272   0.0066548544  4.3464315e6 0.03274346    0.8999985   0.0087401895
# hail                         0.010772934   0.01075105    2.8142472e6 0.041834664   0.84999734  0.013137447
# hail                         0.0156026585  0.01538763    1.9430824e6 0.051031135   0.79999596  0.017977705
# hail                         0.020527184   0.020506436   1.4769476e6 0.060134325   0.7499953   0.023363091
# hail                         0.025329433   0.026305795   1.1969196e6 0.06974729    0.6999941   0.029618384
# hail                         0.03167945    0.032952577   957010.94   0.08062303    0.64999336  0.03669318
# hail                         0.039746936   0.040428177   762760.3    0.09253738    0.59999216  0.044605058
# hail                         0.04936909    0.04882312    614095.7    0.10524536    0.54999125  0.05364658
# hail                         0.059688516   0.05887546    507933.25   0.11867788    0.49999043  0.06472575
# hail                         0.07131737    0.07081122    425104.7    0.13331832    0.44998887  0.077552274
# hail                         0.084766544   0.084527545   357656.25   0.14957334    0.39998806  0.09222923
# hail                         0.1027322     0.0999893     295107.78   0.16791363    0.34998733  0.10852433
# hail                         0.11871479    0.11775396    255379.17   0.18777072    0.299987    0.12821712
# hail                         0.13736756    0.14042953    220704.19   0.2124938     0.24998626  0.15482856
# hail                         0.16949862    0.17163837    178862.83   0.24615231    0.19998503  0.19214667
# hail                         0.21491012    0.21719255    141070.56   0.2898508     0.14998476  0.24886964
# hail                         0.29450893    0.28773916    102942.81   0.35107288    0.09998371  0.34044
# hail                         0.43456802    0.4378746     69738.766   0.434568      0.0499825   1.0
# sig_tornado                  2.4261035e-6  2.8304066e-6  5.3194614e8 4.7834852e-5  1.0         0.00013263774
# sig_tornado                  0.0004057528  0.00023579424 3.1811122e6 0.0032469763  0.9499915   0.00040488812
# sig_tornado                  0.00059533695 0.0007281197  2.1677678e6 0.005315532   0.89997566  0.0012846164
# sig_tornado                  0.0014494961  0.0018466851  890369.6    0.009963221   0.84996736  0.00266355
# sig_tornado                  0.0033571855  0.0033455908  384587.03   0.015744388   0.79995763  0.0042526196
# sig_tornado                  0.005503555   0.0051897806  234480.75   0.020885557   0.7499269   0.006373711
# sig_tornado                  0.008005591   0.007611952   161288.22   0.026096554   0.6999215   0.009088422
# sig_tornado                  0.0113946     0.010493921   113257.195  0.031593025   0.64988774  0.012088198
# sig_tornado                  0.01543821    0.013566258   83581.49    0.037070997   0.59988064  0.015191154
# sig_tornado                  0.017654495   0.016838979   73106.29    0.04248412    0.5498802   0.0185897
# sig_tornado                  0.019828377   0.020732673   65086.04    0.049441174   0.4998679   0.023348691
# sig_tornado                  0.02711613    0.025992747   47586.285   0.059283312   0.4498596   0.029071588
# sig_tornado                  0.032319684   0.032506485   39927.387   0.06960904    0.39985886  0.03656427
# sig_tornado                  0.03674656    0.041633368   35115.22    0.08335467    0.34985486  0.047926597
# sig_tornado                  0.05163865    0.05498613    25005.69    0.105713196   0.29985383  0.0640893
# sig_tornado                  0.07474401    0.07334692    17264.646   0.1337696     0.24981806  0.08491799
# sig_tornado                  0.09472478    0.09996526    13628.161   0.1667167     0.19981451  0.11924904
# sig_tornado                  0.1412089     0.14204147    9139.917    0.22342238    0.14979173  0.174671
# sig_tornado                  0.25262052    0.22395414    5109.6367   0.31548607    0.09978008  0.30462146
# sig_tornado                  0.42072245    0.43105873    3052.3677   0.42072245    0.04976218  1.0
# sig_wind                     1.5078165e-5  1.3624353e-5  5.117169e8  0.0002860065  1.0         0.00032096036
# sig_wind                     0.00071845454 0.0005645285  1.0739296e7 0.005276647   0.949995    0.0009491181
# sig_wind                     0.0012641251  0.001429301   6.103135e6  0.008149331   0.8999903   0.0021022833
# sig_wind                     0.0026019572  0.002662677   2.965356e6  0.011991356   0.84998935  0.003354564
# sig_wind                     0.0039153774  0.004035426   1.9705401e6 0.01548397    0.7999845   0.0048506362
# sig_wind                     0.005621209   0.0056454116  1.3725514e6 0.01928248    0.7499817   0.0065851933
# sig_wind                     0.0072294283  0.007627342   1.0672359e6 0.023333339   0.699979    0.008883022
# sig_wind                     0.009796009   0.010221315   787632.06   0.028158871   0.64997554  0.011814225
# sig_wind                     0.013368689   0.013422279   577123.56   0.03337275    0.5999712   0.015281362
# sig_wind                     0.017281417   0.01716644    446460.0    0.038627904   0.54996854  0.01928957
# sig_wind                     0.021896418   0.021392534   352358.66   0.044072587   0.4999654   0.023738382
# sig_wind                     0.027012363   0.026082711   285625.1    0.04966183    0.44996274  0.028686933
# sig_wind                     0.031446893   0.03143836    245335.45   0.05547736    0.39995992  0.03449683
# sig_wind                     0.03580058    0.03782338    215523.5    0.062276676   0.34995952  0.041535772
# sig_wind                     0.043560848   0.045412015   177116.64   0.07103456    0.2999538   0.0498296
# sig_wind                     0.05270052    0.054386865   146407.31   0.081291065   0.24995136  0.059511732
# sig_wind                     0.066083      0.06493575    116751.66   0.09405175    0.1999464   0.07134952
# sig_wind                     0.08311509    0.078806564   92826.47    0.10950729    0.14994432  0.0880986
# sig_wind                     0.10975363    0.10076215    70300.79    0.1301902     0.09994236  0.11971675
# sig_wind                     0.16002867    0.16833912    48149.46    0.16002867    0.04993724  1.0
# sig_hail                     7.490777e-6   6.886346e-6   5.2799104e8 0.00014661766 1.0         0.0005291442
# sig_hail                     0.0009715011  0.0008870694  4.071248e6  0.0065311478  0.94999903  0.0014320236
# sig_hail                     0.0020442908  0.0019332805  1.9349276e6 0.009575764   0.89999604  0.0025752466
# sig_hail                     0.0031772682  0.0032107758  1.2449678e6 0.012225659   0.8499889   0.0039723096
# sig_hail                     0.0045570815  0.0047042     867973.7    0.01487347    0.7999812   0.00554126
# sig_hail                     0.0063848905  0.006297126   619443.44   0.017517628   0.7499756   0.007133517
# sig_hail                     0.007980753   0.00794038    495671.6    0.020009873   0.6999744   0.008820482
# sig_hail                     0.009687476   0.009671533   408268.5    0.022634959   0.64996374  0.010592608
# sig_hail                     0.011014988   0.011562333   359119.12   0.025472218   0.5999624   0.01263557
# sig_hail                     0.013351578   0.013738701   296227.66   0.02892434    0.5499535   0.014975866
# sig_hail                     0.016296674   0.016245637   242713.5    0.03274395    0.499952    0.017654015
# sig_hail                     0.019343762   0.019115977   204508.36   0.036880612   0.44994646  0.020767877
# sig_hail                     0.022585388   0.022557475   175148.94   0.04159641    0.39993414  0.024577565
# sig_hail                     0.026073143   0.026864454   151697.67   0.04728475    0.3499238   0.02950372
# sig_hail                     0.032282103   0.032488618   122518.28   0.054704595   0.29992065  0.035950314
# sig_hail                     0.038388345   0.039973374   103037.08   0.063533664   0.2499186   0.044875458
# sig_hail                     0.052908365   0.050108675   74756.59    0.07598317    0.19991308  0.056436777
# sig_hail                     0.06872969    0.06376386    57549.14    0.088918395   0.14990976  0.07290501
# sig_hail                     0.08853333    0.084757045   44681.184   0.10424471    0.09990536  0.100859016
# sig_hail                     0.12679838    0.13820235    31125.896   0.12679838    0.049895406 1.0
# sig_tornado_gated_by_tornado 2.4260205e-6  2.8232369e-6  5.3196432e8 4.7834852e-5  1.0         0.00013263774
# sig_tornado_gated_by_tornado 0.00040748646 0.00023586328 3.1675782e6 0.0032548145  0.9499915   0.00040488812
# sig_tornado_gated_by_tornado 0.00059639604 0.00072825706 2.1639182e6 0.005321194   0.89997566  0.0012846164
# sig_tornado_gated_by_tornado 0.0014504304  0.0018467634  889796.2    0.009966842   0.84996736  0.00266355
# sig_tornado_gated_by_tornado 0.0033574367  0.0033456606  384558.25   0.01574711    0.79995763  0.0042526196
# sig_tornado_gated_by_tornado 0.005502034   0.0051898044  234545.56   0.020890014   0.7499269   0.006373711
# sig_tornado_gated_by_tornado 0.008036677   0.007604031   160557.83   0.026106456   0.6999215   0.009070039
# sig_tornado_gated_by_tornado 0.011518989   0.010453398   112077.69   0.03156682    0.6499209   0.012020752
# sig_tornado_gated_by_tornado 0.015848855   0.0134452535  81424.25    0.03692618    0.5998944   0.015002809
# sig_tornado_gated_by_tornado 0.017917529   0.016612634   72040.86    0.04200632    0.5498888   0.018316887
# sig_tornado_gated_by_tornado 0.019800762   0.020350475   65191.227   0.048535567   0.49987108  0.022847794
# sig_tornado_gated_by_tornado 0.02737677    0.025333768   47137.508   0.05787411    0.44985175  0.02820508
# sig_tornado_gated_by_tornado 0.03437752    0.031199455   37544.48    0.06724203    0.3998465   0.034642883
# sig_tornado_gated_by_tornado 0.037977044   0.03886607    33995.445   0.07788699    0.34983295  0.04392035
# sig_tornado_gated_by_tornado 0.049632657   0.049590085   26010.797   0.09444963    0.29980546  0.0565758
# sig_tornado_gated_by_tornado 0.06634517    0.064493276   19461.611   0.11530119    0.24978037  0.07386831
# sig_tornado_gated_by_tornado 0.0899809     0.08468975    14348.237   0.14144419    0.19974755  0.09869439
# sig_tornado_gated_by_tornado 0.11007084    0.11758893    11723.839   0.17486227    0.14971925  0.14273505
# sig_tornado_gated_by_tornado 0.17615989    0.17992707    7324.954    0.24809682    0.09971476  0.25012338
# sig_tornado_gated_by_tornado 0.4210177     0.35996675    3047.259    0.4210177     0.049713757 1.0
# sig_wind_gated_by_wind       1.5076574e-5  1.342633e-5   5.117709e8  0.0002860065  1.0         0.00032096036
# sig_wind_gated_by_wind       0.0007205084  0.00056455564 1.0708683e7 0.0052869283  0.949995    0.0009491181
# sig_wind_gated_by_wind       0.0012674294  0.0014293983  6.087223e6  0.008160542   0.8999903   0.0021022833
# sig_wind_gated_by_wind       0.002605859   0.0026628117  2.9609158e6 0.01199958    0.84998935  0.003354564
# sig_wind_gated_by_wind       0.003917304   0.0040355204  1.9695711e6 0.015489909   0.7999845   0.0048506362
# sig_wind_gated_by_wind       0.005623632   0.0056454064  1.3719599e6 0.01928919    0.7499817   0.0065851933
# sig_wind_gated_by_wind       0.007232331   0.0076272995  1.0668075e6 0.023340883   0.699979    0.008883022
# sig_wind_gated_by_wind       0.009802316   0.01022068    787058.2    0.028167315   0.64997554  0.011812949
# sig_wind_gated_by_wind       0.013384319   0.013419264   576483.1    0.033378925   0.59997547  0.01527613
# sig_wind_gated_by_wind       0.01729987    0.01715819    446006.25   0.03862542    0.5499699   0.019277949
# sig_wind_gated_by_wind       0.02192068    0.021376021   351975.97   0.044057373   0.49996427  0.023717308
# sig_wind_gated_by_wind       0.027116623   0.026047839   284513.3    0.049626693   0.44996053  0.028637126
# sig_wind_gated_by_wind       0.031572536   0.03136771    244358.2    0.055373106   0.39996013  0.034400728
# sig_wind_gated_by_wind       0.036150586   0.037674353   213418.47   0.062056873   0.34995994  0.04131752
# sig_wind_gated_by_wind       0.043364022   0.045173235   177916.44   0.07047568    0.29995847  0.049561795
# sig_wind_gated_by_wind       0.05237359    0.054107923   147308.7    0.080549784   0.2499572   0.059209336
# sig_wind_gated_by_wind       0.065259695   0.064640366   118224.15   0.093070276   0.1999565   0.07106108
# sig_wind_gated_by_wind       0.08164209    0.07859402    94502.58    0.108486064   0.14995459  0.088000946
# sig_wind_gated_by_wind       0.109294236   0.1006903     70594.17    0.12984383    0.09995193  0.11968649
# sig_wind_gated_by_wind       0.15995133    0.16830483    48183.42    0.15995133    0.049948312 1.0
# sig_hail_gated_by_hail       7.4906125e-6  6.887794e-6   5.2800262e8 0.00014661766 1.0         0.0005291442
# sig_hail_gated_by_hail       0.00097227615 0.0008869933  4.0680025e6 0.006537741   0.94999903  0.0014320236
# sig_hail_gated_by_hail       0.002047208   0.001933334   1.9321704e6 0.009586541   0.89999604  0.0025752466
# sig_hail_gated_by_hail       0.0031804577  0.0032107302  1.2437192e6 0.012238122   0.8499889   0.0039723096
# sig_hail_gated_by_hail       0.004563891   0.0047038463  866709.4    0.0148887     0.7999812   0.0055405716
# sig_hail_gated_by_hail       0.0064018597  0.0062950407  617850.25   0.017533572   0.74997383  0.007129837
# sig_hail_gated_by_hail       0.0080205705  0.007932051   493136.94   0.020020522   0.6999687   0.008806337
# sig_hail_gated_by_hail       0.009751042   0.009651275   405646.97   0.022624658   0.6499655   0.010564698
# sig_hail_gated_by_hail       0.011046294   0.011528596   358055.34   0.025422119   0.59995925  0.012594114
# sig_hail_gated_by_hail       0.013297848   0.01369805    297426.66   0.028833913   0.5499568   0.014935859
# sig_hail_gated_by_hail       0.016059881   0.016221223   246277.16   0.032648806   0.4999549   0.017649554
# sig_hail_gated_by_hail       0.019330181   0.019113405   204625.34   0.03688252    0.44995242  0.020767156
# sig_hail_gated_by_hail       0.022590252   0.022556499   175075.06   0.041606136   0.39994663  0.024576362
# sig_hail_gated_by_hail       0.026091587   0.026862184   151586.72   0.04729431    0.3499466   0.029500116
# sig_hail_gated_by_hail       0.032292645   0.032483343   122477.72   0.0547051     0.2999447   0.03594419
# sig_hail_gated_by_hail       0.038367998   0.039968662   103093.46   0.063525274   0.24994288  0.044872575
# sig_hail_gated_by_hail       0.052920353   0.050103772   74738.54    0.07598662    0.19993652  0.056429513
# sig_hail_gated_by_hail       0.06872123    0.06375475    57556.844   0.088910736   0.14993395  0.07289584
# sig_hail_gated_by_hail       0.08849558    0.08474765    44699.934   0.10423459    0.09992902  0.10085272
# sig_hail_gated_by_hail       0.12683265    0.1381481     31132.455   0.12683265    0.049919404 1.0
