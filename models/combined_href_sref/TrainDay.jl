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

# forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer());

# (train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z);
# (_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer());
(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) #

validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, validation_forecasts);
length(validation_forecasts_0z) # 132

@time Forecasts.data(validation_forecasts[10]) # Check if a forecast loads


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

aug29 = validation_forecasts_0z[85]; Forecasts.time_title(aug29) # "2020-08-29 00Z +35"
aug29_data = Forecasts.data(aug29);
for i in 1:size(aug29_data,2)
  PlotMap.plot_debug_map("aug29_0z_day_accs_$(i)_recalib", aug29.grid, aug29_data[:,i]);
end
for (event_name, labeler) in event_name_to_day_labeler
  aug29_labels = event_name_to_day_labeler[event_name](aug29);
  PlotMap.plot_debug_map("aug29_0z_day_$(event_name)_recalib", aug29.grid, aug29_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_tornadoes.pdf ./

july11 = validation_forecasts_0z[78]; Forecasts.time_title(july11) # "2020-07-11 00Z +35"
july11_data = Forecasts.data(july11);
for i in 1:size(july11_data,2)
  PlotMap.plot_debug_map("july11_0z_day_accs_$i", july11.grid, july11_data[:,i]);
end
for (event_name, labeler) in event_name_to_day_labeler
  july11_labels = event_name_to_day_labeler[event_name](july11);
  PlotMap.plot_debug_map("july11_0z_day_$event_name", july11.grid, july11_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_1.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_tornado.pdf ./

dec11 = validation_forecasts_0z[130]; Forecasts.time_title(dec11) # "2021-12-11 00Z +35"
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  PlotMap.plot_debug_map("dec11_0z_day_accs_$i", dec11.grid, dec11_data[:,i]);
end
for (event_name, labeler) in event_name_to_day_labeler
  dec11_labels = event_name_to_day_labeler[event_name](dec11);
  PlotMap.plot_debug_map("dec11_0z_day_$event_name", dec11.grid, dec11_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_1.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_tornado.pdf ./


# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z, X, Ys, weights)

# tornado (8326.0)     feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.1281600368558516
# tornado (8326.0)     feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.11516567319301846
# wind (63336.0)       feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.406340228363044
# wind (63336.0)       feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.3836862069575323
# hail (28152.0)       feature 5 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.23835160168216338
# hail (28152.0)       feature 6 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.2212560788699771
# sig_tornado (1138.0) feature 7 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.087023893911395
# sig_tornado (1138.0) feature 8 highest hourly STORPROB:calculated:day            fcst:: AU-PR-curve: 0.07438944074529161
# sig_wind (7555.0)    feature 9 independent events total SWINDPRO:calculated:day  fcst:: AU-PR-curve: 0.08116342557841788 (only exception. oh well)
# sig_wind (7555.0)    feature 10 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.08192557235022217
# sig_hail (3887.0)    feature 11 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.06977732067660723
# sig_hail (3887.0)    feature 12 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.06093859309168115



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

    println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
  end

  bins_max
end

event_types_count = length(CombinedHREFSREF.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:event_types_count
  (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# event_name  mean_y                 mean_ŷ                 Σweight              bin_max
# tornado     0.00045932070255013616 0.000568258613904859   4.255912233469188e6  0.018898962
# tornado     0.027194184107956362   0.03337949038151059    71904.6996653676     0.061602164
# tornado     0.09047003657078737    0.08870623541941813    21606.823536157608   0.13501288
# tornado     0.16258279128921443    0.232958366409826      12013.857482671738   1.0
# wind        0.003514607089375809   0.00466129449096829    4.1864473609743714e6 0.11713328
# wind        0.14947836983484927    0.17574799820622752    98431.81638890505    0.2604162
# wind        0.2996934884080039     0.336842152353507      49095.1207510829     0.43730876
# wind        0.535667626515728      0.5841822917943789     27463.316039025784   1.0
# hail        0.0015438308781922342  0.0019161113764439232  4.212883760402143e6  0.058389254
# hail        0.07416538223460661    0.09170379692460888    87689.68762636185    0.14193675
# hail        0.16016809889308087    0.1925210906485888     40608.371112167835   0.26998883
# hail        0.32097365761196783    0.3921518894010636     20255.79501271248    1.0
# sig_tornado 6.28590695016493e-5    8.676200336322855e-5   4.336369025113583e6  0.010414468
# sig_tornado 0.01862001068148361    0.016828821407176533   14642.676899790764   0.027291382
# sig_tornado 0.03215173166058256    0.05148501001940698    8481.730343222618    0.11116843
# sig_tornado 0.13976613478586977    0.1631882674947276     1944.1817967891693   1.0
# sig_wind    0.0004144339826161722  0.00046735999004277057 4.223293962513983e6  0.01371814
# sig_wind    0.018749483001086528   0.025276126944477558   93309.74567508698    0.047133457
# sig_wind    0.05837724111103845    0.06378847797192107    29975.81201481819    0.08599372
# sig_wind    0.11764334637147038    0.11740937862984384    14858.093949496746   1.0
# sig_hail    0.00021072991662466427 0.00028818851273148654 4.291198541218936e6  0.01797507
# sig_hail    0.026259995053052224   0.024432634474340492   34450.15455287695    0.03298726
# sig_hail    0.03419223450099387    0.04749899401962511    26448.976460278034   0.07266835
# sig_hail    0.09664042009613436    0.11174816431497328    9339.941921293736    1.0

# event_name   mean_y                  mean_ŷ                  Σweight               bin_max
# tornado      0.000311160678598464    0.00036850454137231375  4.190607527739048e6   0.009306263
# tornado      0.013286336082292436    0.01670974577747424     98085.16124790907     0.028845591
# tornado      0.03330603740185989     0.041799274059122704    39124.24414759874     0.061602164
# tornado      0.07695905530855118     0.08018580928565454     16941.86586087942     0.106333174
# tornado      0.12715464294778728     0.1420522765983369      10250.771253168583    0.19435626
# tornado      0.20235603054254397     0.2956958599845698      6428.043904781342     1.0
# wind         0.002385713064358371    0.0030878624636602845   4.1116624385302067e6  0.07021518
# wind         0.08314986573430275     0.10820914409334015     117964.128947258      0.16140154
# wind         0.17753363630942604     0.20546383600294335     55252.609885811806    0.2604162
# wind         0.28122733210036754     0.3104431503465966      34879.72892636061     0.36931401
# wind         0.3760720124378656      0.4360516479394957      26082.807379186153    0.52724814
# wind         0.6287902618270512      0.6655123881021469      15595.90048456192     1.0
# hail         0.0010423511934540477   0.0013362067722615363   4.1594546736198664e6  0.03750208
# hail         0.04778742214124542     0.05618090813682159     90741.20787340403     0.082046196
# hail         0.0860624794954102      0.10834220894789728     50377.56653523445     0.14193675
# hail         0.14734114289649233     0.17423873517759667     29425.947029590607    0.21595338
# hail         0.21695407021677846     0.2679360633104456      19986.04788339138     0.34294724
# hail         0.3784465872065912      0.46097751256105934     11452.171211898327    1.0
# sig_tornado  4.224220600309771e-5    6.0742860882730115e-5   4.319971681737959e6   0.0046208645
# sig_tornado  0.0076582373637859025   0.00878969641585096     23738.2446616292      0.015867244
# sig_tornado  0.02467751356485312     0.020822452805932538    7370.360729336739     0.027446639
# sig_tornado  0.02685758963888132     0.04265024628913676     6777.87734913826      0.07221879
# sig_tornado  0.07514201697856404     0.10123745802130857     2425.1071608662605    0.14377913
# sig_tornado  0.15526994658525806     0.18840376447867133     1154.3425144553185    1.0
# sig_wind     0.00028145248127747465  0.00029034862435809464  4.1458331863900423e6  0.0071317935
# sig_wind     0.008881485862477431    0.013476774065854666    131327.8200699687     0.025168268
# sig_wind     0.029575682225171353    0.03444731773982316     39442.70172905922     0.047133457
# sig_wind     0.04858420291160842     0.059659180451577545    24023.24373370409     0.07531253
# sig_wind     0.100968906796281       0.08624279504693705     11554.40983992815     0.10004484
# sig_wind     0.1257609203929453      0.13254817589161744     9256.252390682697     1.0
# sig_hail     0.00014120228022351924  0.00021735687280660613  4.2710769115380645e6  0.012925774
# sig_hail     0.018329123523694986    0.017134699331423937    32921.908029675484    0.02218357
# sig_hail     0.027823301630779125    0.02707398338771938     21687.203794956207    0.033009935
# sig_hail     0.032048652643170826    0.04168036189670904     18827.726617455482    0.053475916
# sig_hail     0.053249374025245856    0.06824998540400107     11327.838861048222    0.09119451
# sig_hail     0.10710844796403486     0.13240077542150402     5596.0253121852875    1.0

println(event_to_day_bins)
# Dict{String, Vector{Float32}}("sig_hail" => [0.01797507, 0.03298726, 0.07266835, 1.0], "hail" => [0.058389254, 0.14193675, 0.26998883, 1.0], "tornado" => [0.018898962, 0.061602164, 0.13501288, 1.0], "sig_tornado" => [0.010414468, 0.027291382, 0.11116843, 1.0], "sig_wind" => [0.01371814, 0.047133457, 0.08599372, 1.0], "wind" => [0.11713328, 0.2604162, 0.43730876, 1.0])
# Dict{String, Vector{Float32}}("sig_hail" => [0.012925774, 0.02218357, 0.033009935, 0.053475916, 0.09119451, 1.0], "hail" => [0.03750208, 0.082046196, 0.14193675, 0.21595338, 0.34294724, 1.0], "tornado" => [0.009306263, 0.028845591, 0.061602164, 0.106333174, 0.19435626, 1.0], "sig_tornado" => [0.0046208645, 0.015867244, 0.027446639, 0.07221879, 0.14377913, 1.0], "sig_wind" => [0.0071317935, 0.025168268, 0.047133457, 0.07531253, 0.10004484, 1.0], "wind" => [0.07021518, 0.16140154, 0.2604162, 0.36931401, 0.52724814, 1.0])


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

event_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:event_types_count
  event_name, _ = CombinedHREFSREF.models[prediction_i]

  event_to_day_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name  bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr     max_hourly_au_pr     mean_logistic_ŷ logistic_logloss logistic_au_pr       logistic_coeffs
# tornado     1-2 -1.0             0.061602164      4731273 4209.0    4.327817e6  0.0011134022      0.00024953068     0.0009035082  0.0051579303       0.0057157436       0.026692282420898487 0.02314910262149825  0.0009035081    0.005135906      0.026116179553460395 Float32[0.915529,    0.075384595, -0.13371886]
# tornado     2-3 0.018898962      0.13501288       99627   4161.0    93511.52    0.046163313       0.011292203       0.041814737   0.16413356         0.19283397         0.0962565752711827   0.07540747000717556  0.041814737     0.1635127        0.09683596772813384  Float32[1.3713769,   -0.14764006, 0.3025914]
# tornado     3-4 0.061602164      1.0              35247   4117.0    33620.68    0.14025263        0.03499804        0.116238475   0.35077775         0.41896638         0.19964915178928266  0.18000191591109055  0.116238475     0.34602785       0.1996282709172069   Float32[0.7308742,   -0.01088467, -0.7200494]
# wind        1-2 -1.0             0.2604162        4684085 31709.0   4.2848795e6 0.0085914815      0.0020453343      0.0068676714  0.025646808        0.029395983        0.14005624222334634  0.12718100010983155  0.006867672     0.025428742      0.13970361586617938  Float32[0.8752906,   0.14241132,  0.019422961]
# wind        2-3 0.11713328       0.43730876       158996  31762.0   147526.94   0.2293581         0.05869442        0.1994681     0.48277742         0.60856056         0.2921690467592043   0.26438822915217985  0.19946808      0.47999385       0.29267077150132276  Float32[1.0762116,   -0.07859125, -0.31812784]
# wind        3-4 0.2604162        1.0              82435   31627.0   76558.44    0.42556888        0.12491704        0.38434294    0.6271495          0.8677727          0.5716532810041381   0.5425964559868411   0.38434297      0.6231382        0.5720873029568192   Float32[0.8967682,   0.0795651,   -0.053071257]
# hail        1-2 -1.0             0.14193675       4700656 14108.0   4.3005735e6 0.003746903       0.0008246585      0.0030246011  0.013220488        0.014967729        0.07183032811646992  0.07133696842462632  0.0030246011    0.013116469      0.07254503320323895  Float32[0.69415,     0.3440887,   0.4457177]
# hail        2-3 0.058389254      0.26998883       138852  14074.0   128298.055  0.12361407        0.028034544       0.101386614   0.32032704         0.38457015         0.15735357202747363  0.13849845010973116  0.1013866       0.31769595       0.1630493631922233   Float32[1.1330315,   -0.17600663, -0.6018588]
# hail        3-4 0.14193675       1.0              65864   14044.0   60864.168   0.25895888        0.06771119        0.21368471    0.50153005         0.62741184         0.3465903799789047   0.31890086992964656  0.21368471      0.4924614        0.354143863850937    Float32[1.4685977,   -0.6171552,  -1.4658885]
# sig_tornado 1-2 -1.0             0.027291382      4755736 579.0     4.3510115e6 0.0001431049      3.4930188e-5      0.00012531038 0.0008154274       0.0008881768       0.018449631807736816 0.019159588090648404 0.00012531038   0.00081216113    0.02114136905891041  Float32[0.9246038,   0.14572951,  0.4728381]
# sig_tornado 2-3 0.010414468      0.11116843       24091   564.0     23124.406   0.02954026        0.007996856       0.02358327    0.11005916         0.11708307         0.04899531336795075  0.057967113876265686 0.02358327      0.10669346       0.05813556935374382  Float32[-0.27371246, 1.0642519,   0.48847285]
# sig_tornado 3-4 0.027291382      1.0              10784   559.0     10425.912   0.072314985       0.018098857       0.05221923    0.1902982          0.21011928         0.14452761953831217  0.11803562342468982  0.05221923      0.18464838       0.1344448561028822   Float32[0.77029693,  0.587577,    1.354599]
# sig_wind    1-2 -1.0             0.047133457      4718220 3784.0    4.316604e6  0.0010036379      0.0002244146      0.00081077305 0.004767808        0.00519899         0.02233480092482336  0.019438854132948973 0.0008107731    0.0047234986     0.020428984492083635 Float32[0.28025302,  0.6347386,   0.37337035]
# sig_wind    2-3 0.01371814       0.08599372       132745  3772.0    123285.555  0.03464007        0.00806825        0.028384628   0.122832276        0.13845            0.06525466466811923  0.068942286831688    0.028384631     0.12159428       0.07037209534773976  Float32[1.0196984,   0.3275573,   1.3824697]
# sig_wind    3-4 0.047133457      1.0              48300   3771.0    44833.906   0.08155861        0.018383339       0.07801821    0.26706594         0.32243758         0.12108428956826832  0.1252189924049733   0.0780182       0.26607645       0.12502188609240958  Float32[0.59162444,  0.5482808,   1.1521151]
# sig_hail    1-2 -1.0             0.03298726       4727984 1949.0    4.3256485e6 0.00048047877     0.00011195232     0.0004181904  0.0023050203       0.0025967043       0.023283310014022587 0.018172292746630257 0.00041819026   0.0022790162     0.02666639382695081  Float32[1.9733405,   -0.7847382,  -0.51065785]
# sig_hail    2-3 0.01797507       0.07266835       65826   1955.0    60899.133   0.03445054        0.00779719        0.029705029   0.13395527         0.15440375         0.03825078042948839  0.03258042902226003  0.029705029     0.13243617       0.03940314191209964  Float32[1.3154331,   -0.7666408,  -2.8496652]
# sig_hail    3-4 0.03298726       1.0              38536   1938.0    35788.918   0.064266294       0.015569708       0.050489526   0.19443694         0.22026618         0.10390857002319569  0.0896882406063745   0.050489534     0.19257256       0.10569746828816608  Float32[1.3292868,   -0.31144178, -0.6903727]



print("event_to_0z_day_bins_logistic_coeffs = ")
println(event_to_day_bins_logistic_coeffs)
# event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.9733405, -0.7847382, -0.51065785], [1.3154331, -0.7666408, -2.8496652], [1.3292868, -0.31144178, -0.6903727]], "hail" => [[0.69415, 0.3440887, 0.4457177], [1.1330315, -0.17600663, -0.6018588], [1.4685977, -0.6171552, -1.4658885]], "tornado" => [[0.915529, 0.075384595, -0.13371886], [1.3713769, -0.14764006, 0.3025914], [0.7308742, -0.01088467, -0.7200494]], "sig_tornado" => [[0.9246038, 0.14572951, 0.4728381], [-0.27371246, 1.0642519, 0.48847285], [0.77029693, 0.587577, 1.354599]], "sig_wind" => [[0.28025302, 0.6347386, 0.37337035], [1.0196984, 0.3275573, 1.3824697], [0.59162444, 0.5482808, 1.1521151]], "wind" => [[0.8752906, 0.14241132, 0.019422961], [1.0762116, -0.07859125, -0.31812784], [0.8967682, 0.0795651, -0.053071257]])





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
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
length(day_validation_forecasts_0z) # Expected: 132
# 132

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
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z, X, Ys, weights)

# tornado (8326.0)     feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.1281600368558516
# tornado (8326.0)     feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.11516567319301846
# wind (63336.0)       feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.406340228363044
# wind (63336.0)       feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.3836862069575323
# hail (28152.0)       feature 5 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.23835160168216338
# hail (28152.0)       feature 6 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.2212560788699771
# sig_tornado (1138.0) feature 7 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.087023893911395
# sig_tornado (1138.0) feature 8 highest hourly STORPROB:calculated:day            fcst:: AU-PR-curve: 0.07438944074529161
# sig_wind (7555.0)    feature 9 independent events total SWINDPRO:calculated:day  fcst:: AU-PR-curve: 0.08116342557841788 (only exception. oh well)
# sig_wind (7555.0)    feature 10 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.08192557235022217
# sig_hail (3887.0)    feature 11 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.06977732067660723
# sig_hail (3887.0)    feature 12 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.06093859309168115

# tornado     (8326.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.1282209032860132
# wind        (63336.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.40661148740938
# hail        (28152.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.2418397986132723
# sig_tornado (1138.0)  feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.08536426405725345 (worse)
# sig_wind    (7555.0)  feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.0827883454964689
# sig_hail    (3887.0)  feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07163298981221203

# tornado                      (8326.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob:                 AU-PR-curve: 0.1282209032860132
# wind                         (63336.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.40661148740938
# hail                         (28152.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.2418397986132723
# sig_tornado                  (1138.0)  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.08536426405725345 (worse)
# sig_wind                     (7555.0)  feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.0827883454964689
# sig_hail                     (3887.0)  feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.07163298981221203
# sig_tornado_gated_by_tornado (1138.0)  feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.08848865823430829 (not worse)
# sig_wind_gated_by_wind       (7555.0)  feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.08279453700032868
# sig_hail_gated_by_hail       (3887.0)  feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.07167567544365759


function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, CombinedHREFSREF.models_with_gated))

  # Feature order is all HREF severe probs then all SREF severe probs
  for event_name in event_names
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      y = Ys[event_name]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z, X, Ys, weights)

# tornado (8326.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.1282209032860132
# tornado (8326.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.04732475800518112
# tornado (8326.0)     feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0339861586064038
# tornado (8326.0)     feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.11354375124612497
# tornado (8326.0)     feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.04459240364825072
# tornado (8326.0)     feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.034606936188546086
# tornado (8326.0)     feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.11493414538242039
# tornado (8326.0)     feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.04460273433204098
# tornado (8326.0)     feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.03462280395968384
# wind (63336.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.22089828570139206
# wind (63336.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.40661148740938
# wind (63336.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.15709995382179076
# wind (63336.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.2041958973994816
# wind (63336.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.30328126499935354
# wind (63336.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.1235252528946495
# wind (63336.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.20551911780837648
# wind (63336.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.30337063802019065
# wind (63336.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.12360930749052748
# hail (28152.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.09398262996782632
# hail (28152.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.09154321042267055
# hail (28152.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.2418397986132723
# hail (28152.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.08277889336965975
# hail (28152.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.09804952096183232
# hail (28152.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.18950324759485923
# hail (28152.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.08293647789688759
# hail (28152.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.09808661640778452
# hail (28152.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.18967656144171155
# sig_tornado (1138.0) feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.06041743003588988
# sig_tornado (1138.0) feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.013325360966308942
# sig_tornado (1138.0) feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.00533633718947968
# sig_tornado (1138.0) feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.08536426405725345
# sig_tornado (1138.0) feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.010622012320281402
# sig_tornado (1138.0) feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.005855293276099606
# sig_tornado (1138.0) feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.08848865823430829
# sig_tornado (1138.0) feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.010622744401640164
# sig_tornado (1138.0) feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.005861561543487743
# sig_wind (7555.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.03874712794642739
# sig_wind (7555.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0547703704664618
# sig_wind (7555.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.03027950059212661
# sig_wind (7555.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.03517850327738662
# sig_wind (7555.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.0827883454964689
# sig_wind (7555.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.026815068269763223
# sig_wind (7555.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.03541232905664782
# sig_wind (7555.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.08279453700032868
# sig_wind (7555.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.026840688278534006
# sig_hail (3887.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.01878808917978697
# sig_hail (3887.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.01229898477844842
# sig_hail (3887.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0767373527800814 (nice, the hail predictor is better at predicting sig_hail)
# sig_hail (3887.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.015452677450187456
# sig_hail (3887.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.01806388715156955
# sig_hail (3887.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.07163298981221203
# sig_hail (3887.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.015517669027858151
# sig_hail (3887.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.018068098872738927
# sig_hail (3887.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.07167567544365759


# rm("day_accumulators_validation_forecasts_0z"; recursive = true)

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
test_calibration(day_validation_forecasts_0z, X, Ys, weights)

# event_name                   mean_y        mean_ŷ        Σweight     SR            POD         bin_max
# tornado                      0.00010167745 9.478093e-5   3.8498672e6 0.0017925821  1.0         0.0008280874
# tornado                      0.0016477421  0.0016213172  237275.0    0.014517632   0.94993186  0.0031164377
# tornado                      0.0044999975  0.0045181573  86934.234   0.025650535   0.89992464  0.0064718667
# tornado                      0.007867265   0.008332609   49779.47    0.035464235   0.84988725  0.01063581
# tornado                      0.012190737   0.012809818   32089.846   0.045449305   0.7997956   0.015324517
# tornado                      0.016798053   0.017595688   23272.555   0.055566322   0.7497589   0.02022394
# tornado                      0.02640414    0.022535294   14840.12    0.06653987    0.6997561   0.02520302
# tornado                      0.027268358   0.028917618   14344.13    0.075379685   0.6496373   0.033224784
# tornado                      0.037735596   0.037635308   10373.523   0.08839214    0.599608    0.0428201
# tornado                      0.04326288    0.04921671    9054.567    0.10070974    0.549539    0.05690336
# tornado                      0.054456584   0.06507926    7187.278    0.1161874     0.49943477  0.0737913
# tornado                      0.07529733    0.080781825   5197.8423   0.13298085    0.449373    0.08804794
# tornado                      0.10501838    0.09392409    3729.179    0.14710926    0.39931265  0.10004951
# tornado                      0.13975067    0.10466423    2803.5896   0.15608245    0.34922048  0.10917482
# tornado                      0.14345174    0.11262955    2728.888    0.15919958    0.29910642  0.1155536
# tornado                      0.13576221    0.12252757    2880.648    0.1627927     0.24903582  0.13064986
# tornado                      0.09405034    0.14701247    4166.264    0.17136866    0.19901389  0.16760442
# tornado                      0.15710877    0.18806145    2490.931    0.23693241    0.14889535  0.21246456
# tornado                      0.32849082    0.22814818    1190.8087   0.3190184     0.09883966  0.2468049
# tornado                      0.30985874    0.30463162    1231.4708   0.30985874    0.048806667 1.0
# wind                         0.0007630626  0.00067959237 3.8571765e6 0.013493679   1.0         0.012142547
# wind                         0.019295642   0.01938194    152508.0    0.11087229    0.94998854  0.029342826
# wind                         0.035849784   0.03807136    82102.34    0.15057679    0.899986    0.048816368
# wind                         0.053140033   0.05901149    55381.06    0.18550847    0.84997314  0.07083556
# wind                         0.07501576    0.08210199    39230.312   0.219721      0.799967    0.09475872
# wind                         0.1047291     0.10596249    28101.234   0.25215274    0.7499619   0.1180387
# wind                         0.13683659    0.12922041    21508.717   0.28034684    0.6999547   0.14149615
# wind                         0.16161464    0.15380143    18208.06    0.3049561     0.6499447   0.16670153
# wind                         0.1811455     0.18017727    16246.431   0.329298      0.5999431   0.19482878
# wind                         0.20558606    0.20979135    14316.877   0.3557553     0.5499367   0.22617787
# wind                         0.24525946    0.24084637    11999.055   0.38380137    0.4999238   0.25593793
# wind                         0.27721983    0.27095932    10618.091   0.40951124    0.4499189   0.28650844
# wind                         0.31233156    0.30205852    9423.59     0.4355042     0.3999027   0.31804705
# wind                         0.33587414    0.33457202    8762.351    0.46151945    0.34989092  0.35229212
# wind                         0.35419852    0.3710563     8309.988    0.49222514    0.2998832   0.39135832
# wind                         0.39383218    0.4130885     7472.3784   0.5338662     0.24986972  0.43753636
# wind                         0.46196976    0.46527117    6370.6504   0.5859965     0.19986512  0.49825397
# wind                         0.5602162     0.53370863    5252.7007   0.64366245    0.14985737  0.57201576
# wind                         0.6167607     0.6217451     4772.341    0.6955395     0.09985642  0.68595064
# wind                         0.79779035    0.79397136    3676.8337   0.79779035    0.04984283  1.0
# hail                         0.00032671893 0.0003403519  3.9823778e6 0.005964378   1.0         0.007700762
# hail                         0.0120669855  0.011806619   107822.04   0.06519324    0.9499825   0.017368201
# hail                         0.0238847     0.02176208    54490.125   0.08631191    0.8999662   0.026822938
# hail                         0.032964785   0.031642444   39473.387   0.10200604    0.8499348   0.037096377
# hail                         0.04673541    0.041890673   27847.195   0.117379345   0.7999129   0.04731959
# hail                         0.052423768   0.053343054   24819.2     0.13054453    0.7498827   0.05992865
# hail                         0.06502401    0.0663432     20015.164   0.14610448    0.6998653   0.07325162
# hail                         0.078037396   0.07987019    16675.627   0.16162027    0.6498344   0.08687793
# hail                         0.09113246    0.093409814   14276.248   0.1774738     0.59980905  0.10018706
# hail                         0.10153657    0.107238814   12811.347   0.19421218    0.549795    0.11511048
# hail                         0.1236484     0.12355505    10520.143   0.21373065    0.4997889   0.13255808
# hail                         0.1483821     0.14112027    8767.335    0.23256764    0.44978368  0.15019691
# hail                         0.1542748     0.16021946    8435.242    0.25033477    0.399774    0.17093635
# hail                         0.18052077    0.18222788    7205.432    0.27480972    0.34974775  0.19452891
# hail                         0.20172232    0.20835963    6449.458    0.3010397     0.29974517  0.22377613
# hail                         0.24703225    0.2396799     5268.2666   0.33396924    0.24973224  0.25738972
# hail                         0.29103062    0.27554595    4471.4077   0.36626038    0.1997027   0.29621494
# hail                         0.34358785    0.3196851     3785.6433   0.4008952     0.14967756  0.34621853
# hail                         0.3971097     0.38063544    3276.4258   0.43750042    0.09967611  0.42327878
# hail                         0.48743522    0.5193493     2650.1987   0.48743522    0.049659293 1.0
# sig_tornado                  1.2932645e-5  1.42862045e-5 4.2178535e6 0.00024983965 1.0         0.00039908354
# sig_tornado                  0.0016676823  0.0005244036  32809.695   0.0072091077  0.9499403   0.0006862384
# sig_tornado                  0.0008702255  0.0015147728  62801.09    0.008850397   0.8997264   0.0032016796
# sig_tornado                  0.0034456397  0.004515462   15864.439   0.019297147   0.8495721   0.006478205
# sig_tornado                  0.007202703   0.008140471   7628.2705   0.027129143   0.79940677  0.010057206
# sig_tornado                  0.012257586   0.011401906   4483.798    0.03333836    0.74898356  0.012852915
# sig_tornado                  0.011730036   0.01448564    4710.299    0.03806525    0.69854534  0.016282119
# sig_tornado                  0.015196064   0.018047128   3603.1704   0.04618013    0.6478396   0.020143945
# sig_tornado                  0.03143372    0.021392673   1756.2347   0.05573583    0.5975909   0.022742532
# sig_tornado                  0.034040663   0.02434751    1622.4409   0.06003527    0.54692835  0.026145607
# sig_tornado                  0.023448752   0.029782394   2353.756    0.06511381    0.49624375  0.034343038
# sig_tornado                  0.03174656    0.03937564    1746.864    0.08159403    0.4455925   0.04568409
# sig_tornado                  0.048008986   0.052753594   1139.7488   0.102307506   0.3946987   0.061254423
# sig_tornado                  0.076820984   0.06771925    711.13403   0.12250471    0.34448287  0.07447778
# sig_tornado                  0.10807117    0.08026701    504.8168    0.13631153    0.29434794  0.08681021
# sig_tornado                  0.13988143    0.09249852    389.4981    0.14402522    0.24428083  0.0989296
# sig_tornado                  0.13354357    0.10636648    409.6866    0.1451317     0.19428033  0.114751205
# sig_tornado                  0.104149036   0.13588387    524.1591    0.1496575     0.14407107  0.16516839
# sig_tornado                  0.21403179    0.18622042    254.78905   0.19510818    0.093972266 0.21173073
# sig_tornado                  0.17725311    0.25077733    270.03683   0.17725311    0.043926425 1.0
# sig_wind                     9.01868e-5    8.2532104e-5  3.884849e6  0.001604436   1.0         0.0011887929
# sig_wind                     0.002291226   0.0018994362  152943.86   0.013947636   0.94993144  0.0029072782
# sig_wind                     0.003447475   0.004077951   101657.77   0.01945607    0.89985335  0.0055708494
# sig_wind                     0.0072220536  0.0066357534  48508.48    0.02678712    0.8497704   0.007874628
# sig_wind                     0.009581156   0.009111441   36592.938   0.03225795    0.79970634  0.010541309
# sig_wind                     0.011960679   0.012138069   29272.924   0.038320024   0.7496034   0.014022063
# sig_wind                     0.013513088   0.016543712   25945.258   0.045490324   0.69956887  0.019717678
# sig_wind                     0.021835351   0.022593033   16024.334   0.055649303   0.6494662   0.025874559
# sig_wind                     0.028992878   0.029211031   12080.038   0.06390373    0.5994641   0.03311475
# sig_wind                     0.03838238    0.03702531    9135.829    0.07177714    0.54941374  0.04127369
# sig_wind                     0.034732573   0.046698023   10083.45    0.078644305   0.49930334  0.052489027
# sig_wind                     0.049267955   0.05707676    7114.2617   0.09153694    0.44925448  0.061772134
# sig_wind                     0.07072625    0.06535926    4947.375    0.10258057    0.39916548  0.06909165
# sig_wind                     0.08414633    0.072600305   4160.8545   0.10965328    0.34916162  0.0761856
# sig_wind                     0.09637864    0.07962023    3635.0266   0.115509965   0.29912758  0.083233014
# sig_wind                     0.088595316   0.087806694   3957.9934   0.120310575   0.24906234  0.09299112
# sig_wind                     0.095256664   0.0988815     3676.2483   0.1322336     0.19895126  0.10557378
# sig_wind                     0.120346494   0.11356478    2907.3557   0.15207249    0.1489077   0.12311013
# sig_wind                     0.18081075    0.1325512     1937.0975   0.17545567    0.0989066   0.14328133
# sig_wind                     0.17028858    0.19422394    2007.5653   0.17028858    0.048854344 1.0
# sig_hail                     4.3443615e-5  3.7514972e-5  4.1813212e6 0.0008290639  1.0         0.002131442
# sig_hail                     0.003381      0.0035074942  53506.86    0.0190669     0.94976324  0.0054865396
# sig_hail                     0.005873211   0.007479992   30791.934   0.025695976   0.8997325   0.009960949
# sig_hail                     0.011891556   0.011696454   15210.576   0.032066226   0.8497181   0.013622783
# sig_hail                     0.013144904   0.015849095   13768.598   0.0358732     0.79969543  0.018294519
# sig_hail                     0.02240262    0.019809669   8079.5366   0.040555198   0.7496425   0.02130978
# sig_hail                     0.024748169   0.022662168   7316.8066   0.04305124    0.6995852   0.024028635
# sig_hail                     0.0264934     0.02530682    6846.937    0.045654565   0.64950716  0.026579775
# sig_hail                     0.026997052   0.02792544    6713.871    0.04859649    0.5993404   0.029312588
# sig_hail                     0.03255045    0.030573744   5564.403    0.052424673   0.5492134   0.03189743
# sig_hail                     0.03895502    0.033167113   4657.1387   0.055846684   0.4991226   0.034466434
# sig_hail                     0.03771013    0.036023602   4812.309    0.058690786   0.44895017  0.03764904
# sig_hail                     0.034657083   0.039754532   5240.4404   0.06310993    0.39876288  0.042106494
# sig_hail                     0.037930347   0.044996675   4785.1343   0.07157853    0.3485353   0.048422232
# sig_hail                     0.048757788   0.052650712   3726.201    0.08413622    0.29833996  0.057476085
# sig_hail                     0.0720744     0.0616579     2510.0889   0.09862987    0.248095    0.06640047
# sig_hail                     0.106706835   0.070674635   1694.5775   0.10875171    0.19806248  0.07556127
# sig_hail                     0.104662575   0.082178935   1729.7078   0.109460205   0.14805487  0.08971065
# sig_hail                     0.14311163    0.09713503    1264.5778   0.11208537    0.09798846  0.10622937
# sig_hail                     0.09139786    0.14559959    1896.5615   0.09139787    0.047938596 1.0
# sig_tornado_gated_by_tornado 1.2932563e-5  1.4285014e-5  4.2178805e6 0.00024983965 1.0         0.00039908354
# sig_tornado_gated_by_tornado 0.0016685951  0.00052441104 32791.746   0.00721045    0.9499403   0.0006862384
# sig_tornado_gated_by_tornado 0.0008703471  0.0015148158  62792.316   0.008851097   0.8997264   0.0032016796
# sig_tornado_gated_by_tornado 0.003444364   0.0045157736  15870.314   0.019297147   0.8495721   0.006478205
# sig_tornado_gated_by_tornado 0.0073095034  0.0081107365  7518.906    0.027134107   0.79940677  0.009991831
# sig_tornado_gated_by_tornado 0.012451438   0.011302842   4419.453    0.03319739    0.74896955  0.01271772
# sig_tornado_gated_by_tornado 0.0144717675  0.014012038   3822.0083   0.03774429    0.69846886  0.015388581
# sig_tornado_gated_by_tornado 0.015438776   0.016927898   3555.7712   0.04318703    0.64770883  0.018686714
# sig_tornado_gated_by_tornado 0.022730814   0.020221898   2407.7751   0.05090337    0.5973291   0.021899484
# sig_tornado_gated_by_tornado 0.040852904   0.023047244   1348.4341   0.05743904    0.5471018   0.024283629
# sig_tornado_gated_by_tornado 0.027136924   0.02678088    2025.7551   0.059915688   0.49654707  0.029632349
# sig_tornado_gated_by_tornado 0.03773724    0.032446314   1465.2151   0.06939525    0.4460976   0.0357152
# sig_tornado_gated_by_tornado 0.03288833    0.040876485   1659.5396   0.07776888    0.3953541   0.047445253
# sig_tornado_gated_by_tornado 0.051779054   0.054441854   1056.597    0.096965164   0.34526557  0.06277819
# sig_tornado_gated_by_tornado 0.06972147    0.07064572    782.7588    0.113875255   0.2950576   0.07914998
# sig_tornado_gated_by_tornado 0.12135007    0.08554337    449.1574    0.13081218    0.24497308  0.0922872
# sig_tornado_gated_by_tornado 0.117129155   0.10121726    473.38025   0.13348268    0.19495264  0.11120656
# sig_tornado_gated_by_tornado 0.085807584   0.13290901    636.1634    0.14040656    0.1440683   0.16516839
# sig_tornado_gated_by_tornado 0.19215532    0.18768853    283.79623   0.21248133    0.093972266 0.21173073
# sig_tornado_gated_by_tornado 0.24159756    0.24380437    198.11818   0.24159756    0.043926425 1.0
# sig_wind_gated_by_wind       9.014945e-5   8.1475046e-5  3.8864585e6 0.001604436   1.0         0.0011887929
# sig_wind_gated_by_wind       0.0023085468  0.0018997594  151796.34   0.013994899   0.94993144  0.0029072782
# sig_wind_gated_by_wind       0.0034607395  0.004079468   101268.13   0.019483883   0.89985335  0.0055708494
# sig_wind_gated_by_wind       0.0072283484  0.006636009   48466.234   0.026795855   0.8497704   0.007874628
# sig_wind_gated_by_wind       0.009581359   0.009111589   36592.16    0.03226355    0.79970634  0.010541309
# sig_wind_gated_by_wind       0.0119620515  0.012138187   29269.564   0.038328238   0.7496034   0.014022063
# sig_wind_gated_by_wind       0.013526184   0.016543481   25920.139   0.045501307   0.69956887  0.019717678
# sig_wind_gated_by_wind       0.021835357   0.02259304    16024.33    0.055649884   0.6494662   0.025874559
# sig_wind_gated_by_wind       0.028994923   0.029211247   12079.186   0.06390455    0.5994641   0.03311475
# sig_wind_gated_by_wind       0.03838238    0.03702531    9135.829    0.07177714    0.54941374  0.04127369
# sig_wind_gated_by_wind       0.034732573   0.046698023   10083.45    0.078644305   0.49930334  0.052489027
# sig_wind_gated_by_wind       0.049267955   0.05707676    7114.2617   0.09153694    0.44925448  0.061772134
# sig_wind_gated_by_wind       0.07072625    0.06535926    4947.375    0.10258057    0.39916548  0.06909165
# sig_wind_gated_by_wind       0.08414633    0.072600305   4160.8545   0.10965328    0.34916162  0.0761856
# sig_wind_gated_by_wind       0.09637864    0.07962023    3635.0266   0.115509965   0.29912758  0.083233014
# sig_wind_gated_by_wind       0.088595316   0.087806694   3957.9934   0.120310575   0.24906234  0.09299112
# sig_wind_gated_by_wind       0.095256664   0.0988815     3676.2483   0.1322336     0.19895126  0.10557378
# sig_wind_gated_by_wind       0.120346494   0.11356478    2907.3557   0.15207249    0.1489077   0.12311013
# sig_wind_gated_by_wind       0.18081075    0.1325512     1937.0975   0.17545567    0.0989066   0.14328133
# sig_wind_gated_by_wind       0.17028858    0.19422394    2007.5653   0.17028858    0.048854344 1.0
# sig_hail_gated_by_hail       4.344245e-5   3.7527032e-5  4.1814335e6 0.0008290639  1.0         0.002131442
# sig_hail_gated_by_hail       0.0033828572  0.003507996   53477.484   0.01907877    0.94976324  0.0054865396
# sig_hail_gated_by_hail       0.00587647    0.007479365   30774.86    0.02571277    0.8997325   0.009960949
# sig_hail_gated_by_hail       0.011897539   0.011696605   15202.927   0.0320882     0.8497181   0.013622783
# sig_hail_gated_by_hail       0.013206575   0.015847255   13755.366   0.035899017   0.79969543  0.018289192
# sig_hail_gated_by_hail       0.022441218   0.019803042   8071.396    0.040572267   0.749456    0.021301799
# sig_hail_gated_by_hail       0.024725448   0.022652488   7318.7715   0.04306439    0.69936293  0.024014909
# sig_hail_gated_by_hail       0.026091982   0.025316637   6959.6787   0.045675475   0.6493175   0.026614295
# sig_hail_gated_by_hail       0.027261103   0.027944217   6647.594    0.048742156   0.59909725  0.029323425
# sig_hail_gated_by_hail       0.03251509    0.03058334    5573.9214   0.052520253   0.5489797   0.031907417
# sig_hail_gated_by_hail       0.039122637   0.033174288   4637.348    0.055980828   0.4988577   0.034469076
# sig_hail_gated_by_hail       0.037468676   0.036039513   4842.615    0.058814894   0.44868353  0.037684303
# sig_hail_gated_by_hail       0.034984414   0.039785787   5192.671    0.06336025    0.39850354  0.042134486
# sig_hail_gated_by_hail       0.03818001    0.045019582   4753.2915   0.07175627    0.34826374  0.04845632
# sig_hail_gated_by_hail       0.04895833    0.052690554   3710.9084   0.08422849    0.29807425  0.05750161
# sig_hail_gated_by_hail       0.07229423    0.06168038    2502.8496   0.09863457    0.24782968  0.06642584
# sig_hail_gated_by_hail       0.105284646   0.07077988    1725.6705   0.1086499     0.19778928  0.0757371
# sig_hail_gated_by_hail       0.105445445   0.082404524   1717.2372   0.1098456     0.14754285  0.08992363
# sig_hail_gated_by_hail       0.14176302    0.09750575    1276.784    0.11225232    0.097465605 0.10679258
# sig_hail_gated_by_hail       0.09202548    0.14630783    1862.8114   0.09202548    0.04740884  1.0



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
  event_name, _ = CombinedHREFSREF.models[prediction_i]
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
  event_name, _ = CombinedHREFSREF.models[prediction_i]
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
  event_name, _ = CombinedHREFSREF.models[prediction_i]
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
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations_sr_pod[event_name] = spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tthreshold_to_match_warning_ratio\tmean_threshold\tSR\tPOD\tWR")
calibrations_all = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations_all[event_name] = spc_calibrate_all(prediction_i, X, Ys, weights)
end

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD mean_threshold SR          POD          WR
# tornado     0.02         0.013012083                      0.020887375            0.016949728    0.05950327  0.7356839    0.022163047
# tornado     0.05         0.05849914                       0.08197975             0.07023945     0.12929934  0.46040323   0.006382945
# tornado     0.1          0.16379696                       0.16168022             0.16273859     0.22669719  0.15410714   0.0012185845
# tornado     0.15         0.20259823                       0.25695992             0.22977906     0.3307749   0.07367811   0.00039928683
# tornado     0.3          0.2128558                        0.31739616             0.265126       0.2730277   0.02881738   0.00018920247
# tornado     0.45         0.5045417                        0.5705242              0.5375329      0.4997716   0.0018807576 6.7459064e-6
# wind        0.05         0.022922972                      0.098394394            0.060658682    0.20406981  0.8215586    0.054323807
# wind        0.15         0.10629699                       0.27374458             0.19002078     0.35123366  0.5577441    0.021427387
# wind        0.3          0.30596498                       0.55200005             0.4289825      0.5758059   0.2080737    0.004876088
# wind        0.45         0.49876654                       0.84303856             0.67090255     0.7800731   0.054841843  0.00094865245
# hail        0.05         0.014685668                      0.043951035            0.029318351    0.105873905 0.8370955    0.04715755
# hail        0.15         0.07232482                       0.1376667              0.10499576     0.20084587  0.53308463   0.015830638
# hail        0.3          0.19926694                       0.36832237             0.28379464     0.3919302   0.1650003    0.0025109681
# hail        0.45         0.5387201                        0.64518166             0.5919509      0.60983324  0.011408552  0.000111579546
# sig_tornado 0.1          0.043976437                      0.07899284             0.061484642    0.12216851  0.3417908    0.0006989763
# sig_wind    0.1          0.09259216                       0.112127304            0.10235973     0.14755915  0.16145596   0.0017555384
# sig_hail    0.1          0.051618036                      0.05091667             0.051267356    0.0888395   0.2828028    0.0026391589

# event_name  nominal_prob threshold_to_match_warning_ratio SR         POD          WR
# tornado     0.02         0.019361496                      0.06481828 0.7092841    0.01961561
# tornado     0.05         0.08399773                       0.14432065 0.4178407    0.0051899273
# tornado     0.1          0.17687798                       0.25749397 0.14037727   0.000977257
# tornado     0.15         0.2711239                        0.2659266  0.025009554  0.00016858664
# tornado     0.3          0.37184715                       0.26168418 0.0055128406 3.7763915e-5
# tornado     0.45         0.57912636                       0.46637073 0.0008775264 3.372935e-6
# wind        0.05         0.05350685                       0.19299479 0.83842576   0.05862049
# wind        0.15         0.23026466                       0.38716137 0.4932454    0.017191011
# wind        0.3          0.52155495                       0.66289306 0.13369814   0.0027215248
# wind        0.45         0.8279896                        0.9097415  0.019793663  0.00029358818
# hail        0.05         0.033460617                      0.11226271 0.8185733    0.043489777
# hail        0.15         0.1253109                        0.22510515 0.47089842   0.012476906
# hail        0.3          0.35450554                       0.44244266 0.09317733   0.0012560833
# hail        0.45         0.647604                         0.70404613 0.0080282595 6.801198e-5
# sig_tornado 0.1          0.069143295                      0.13080691 0.31584063   0.00060325186
# sig_wind    0.1          0.11427498                       0.16778375 0.124935105  0.0011946948
# sig_hail    0.1          0.05564308                       0.09666148 0.26031342   0.0022327038

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD threshold_to_match_warning_ratio mean_threshold SR                  POD                  WR
# tornado 0.02    0.013012083     0.020887375     0.019361496     0.017753651     0.06133527      0.7272919       0.021255804
# tornado 0.05    0.05849914      0.08197975      0.08399773      0.07482554      0.13421077      0.44672015      0.0059666047
# tornado 0.1     0.16379696      0.16168022      0.17687798      0.16745172      0.23657304      0.14925042      0.0011309134
# tornado 0.15    0.20259823      0.25695992      0.2711239       0.24356067      0.3085607       0.0516589       0.00030011215
# tornado 0.3     0.2128558       0.31739616      0.37184715      0.3006997       0.2380756       0.011839801     8.9147376e-5
# tornado 0.45    0.5045417       0.5705242       0.57912636      0.5513974       0.48126388      0.0016298362    6.0707134e-6
# wind    0.05    0.022922972     0.098394394     0.05350685      0.058274735     0.2003704       0.8267769       0.055678196
# wind    0.15    0.10629699      0.27374458      0.23026466      0.20343542      0.36393848      0.53599864      0.019873122
# wind    0.3     0.30596498      0.55200005      0.52155495      0.45983997      0.61152744      0.1801016       0.003974038
# wind    0.45    0.49876654      0.84303856      0.8279896       0.7232649       0.8471201       0.038826246     0.00061845884
# hail    0.05    0.014685668     0.043951035     0.033460617     0.030699106     0.10806746      0.83122927      0.04587658
# hail    0.15    0.07232482      0.1376667       0.1253109       0.11176747      0.20974542      0.5104774       0.014516074
# hail    0.3     0.19926694      0.36832237      0.35450554      0.30736494      0.41000995      0.13714822      0.0019950827
# hail    0.45    0.5387201       0.64518166      0.647604        0.61050195      0.64393663      0.010434036     9.6643875e-5
# sig_tornado     0.1     0.043976437     0.07899284      0.069143295     0.064037524     0.123192        0.32835937      0.0006659295
# sig_wind        0.1     0.09259216      0.112127304     0.11427498      0.106331475     0.15405066      0.14742465      0.0015354261
# sig_hail        0.1     0.051618036     0.05091667      0.05564308      0.05272593      0.091144875     0.27381667      0.002490667

println(calibrations_sr_pod)
# calibrations_sr_pod = Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.051267356)], "hail" => [(0.05, 0.029318351), (0.15, 0.10499576), (0.3, 0.28379464), (0.45, 0.5919509)], "tornado" => [(0.02, 0.016949728), (0.05, 0.07023945), (0.1, 0.16273859), (0.15, 0.22977906), (0.3, 0.265126), (0.45, 0.5375329)], "sig_tornado" => [(0.1, 0.061484642)], "sig_wind" => [(0.1, 0.10235973)], "wind" => [(0.05, 0.060658682), (0.15, 0.19002078), (0.3, 0.4289825), (0.45, 0.67090255)])

println(calibrations_wr)
# calibrations_wr = Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.05564308)], "hail" => [(0.05, 0.033460617), (0.15, 0.1253109), (0.3, 0.35450554), (0.45, 0.647604)], "tornado" => [(0.02, 0.019361496), (0.05, 0.08399773), (0.1, 0.17687798), (0.15, 0.2711239), (0.3, 0.37184715), (0.45, 0.57912636)], "sig_tornado" => [(0.1, 0.069143295)], "sig_wind" => [(0.1, 0.11427498)], "wind" => [(0.05, 0.05350685), (0.15, 0.23026466), (0.3, 0.52155495), (0.45, 0.8279896)])

println(calibrations_all)
# calibrations_all = Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.05272593)], "hail" => [(0.05, 0.030699106), (0.15, 0.11176747), (0.3, 0.30736494), (0.45, 0.61050195)], "tornado" => [(0.02, 0.017753651), (0.05, 0.07482554), (0.1, 0.16745172), (0.15, 0.24356067), (0.3, 0.3006997), (0.45, 0.5513974)], "sig_tornado" => [(0.1, 0.064037524)], "sig_wind" => [(0.1, 0.106331475)], "wind" => [(0.05, 0.058274735), (0.15, 0.20343542), (0.3, 0.45983997), (0.45, 0.7232649)])

# using the warning ratio calibrations rn
# spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
#   "tornado" => [
#     (0.02, 0.019361496),
#     (0.05, 0.08399773),
#     (0.1,  0.17687798),
#     (0.15, 0.2711239),
#     (0.3,  0.37184715),
#     (0.45, 0.57912636),
#   ],
#   "wind" => [
#     (0.05, 0.05350685),
#     (0.15, 0.23026466),
#     (0.3,  0.52155495),
#     (0.45, 0.8279896)
#   ],
#   "hail" => [
#     (0.05, 0.033460617),
#     (0.15, 0.1253109),
#     (0.3,  0.35450554),
#     (0.45, 0.647604)
#   ],
#   "sig_tornado" => [(0.1, 0.069143295)],
#   "sig_wind"    => [(0.1, 0.11427498)],
#   "sig_hail"    => [(0.1, 0.05564308)],
# )


# See how gating on SPC calibrated affects AU-PR


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

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts) # 628

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts) # 528

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10]);

day_validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
length(day_validation_forecasts_0z) # Expected: 132
# 132

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

# rm("day_validation_forecasts_0z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_spc_calibrated_with_sig_gated",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z, X, Ys, weights)

# tornado (8326.0)                      feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  AU-PR-curve: 0.12822090429073685
# wind (63336.0)                        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.40661148757985616
# hail (28152.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.24183979659718088
# sig_tornado (1138.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.08536426405725345
# sig_wind (7555.0)                     feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.0827883459131995
# sig_hail (3887.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.07163298416892598
# sig_tornado_gated_by_tornado (1138.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.07877760337620006
# sig_wind_gated_by_wind (7555.0)       feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.08279349944861077
# sig_hail_gated_by_hail (3887.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.07167777942623012


function test_threshold(forecasts, X, Ys, weights, threshold)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:,feature_i]
    model_csi = Metrics.csi(ŷ, y, weights, threshold)
    println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tThreshold: $threshold\tCSI: $model_csi")
  end
end
test_threshold(day_validation_forecasts_0z, X, Ys, weights, 0.1)

# tornado (8326.0)                      feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  Threshold: 0.1  CSI: 0.0999275885395187
# wind (63336.0)                        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.2621078607407563
# hail (28152.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.1533293040297049
# sig_tornado (1138.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.1019263018731568
# sig_wind (7555.0)                     feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.07713544881757496
# sig_hail (3887.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.07583280034351035
# sig_tornado_gated_by_tornado (1138.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado Threshold: 0.1  CSI: 0.10237996019129936
# sig_wind_gated_by_wind (7555.0)       feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    Threshold: 0.1  CSI: 0.07713544881757496
# sig_hail_gated_by_hail (3887.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    Threshold: 0.1  CSI: 0.07584469907328449













# tried some post-processing blurring below
# didn't help AU-PR

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

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_spc_calibrated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)
# 903

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_with_blurs_and_forecasts_hour = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
length(day_validation_forecasts_0z_with_blurs_and_forecasts_hour) # Expected: 132
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

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_with_blurs_and_forecasts_hour;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_with_blurs_and_forecasts_hour",
  );

println("Determining best blur radii to maximize area under precision-recall curve")

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:(length(inventory)-1)
    prediction_i = 1 + div(feature_i-1, 1 + length(CombinedHREFSREF.blur_radii))
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z_with_blurs_and_forecasts_hour, X, Ys, weights)

# tornado (8326.0)     feature 1 TORPROB:calculated:hour fcst:calculated_prob:             AU-PR-curve: 0.12910289299673042
# tornado (8326.0)     feature 2 TORPROB:calculated:hour fcst:calculated_prob:15mi mean    AU-PR-curve: 0.1289217023896551
# tornado (8326.0)     feature 3 TORPROB:calculated:hour fcst:calculated_prob:25mi mean    AU-PR-curve: 0.12855908567739005
# tornado (8326.0)     feature 4 TORPROB:calculated:hour fcst:calculated_prob:35mi mean    AU-PR-curve: 0.1276909837160792
# tornado (8326.0)     feature 5 TORPROB:calculated:hour fcst:calculated_prob:50mi mean    AU-PR-curve: 0.1257292040430739
# tornado (8326.0)     feature 6 TORPROB:calculated:hour fcst:calculated_prob:70mi mean    AU-PR-curve: 0.12313387528809555
# tornado (8326.0)     feature 7 TORPROB:calculated:hour fcst:calculated_prob:100mi mean   AU-PR-curve: 0.11846516730272345
# wind (63336.0)       feature 8 WINDPROB:calculated:hour fcst:calculated_prob:            AU-PR-curve: 0.407616606429577
# wind (63336.0)       feature 9 WINDPROB:calculated:hour fcst:calculated_prob:15mi mean   AU-PR-curve: 0.40755924309707947
# wind (63336.0)       feature 10 WINDPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.40727334595487796
# wind (63336.0)       feature 11 WINDPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.40641155183353506
# wind (63336.0)       feature 12 WINDPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.4042658521238372
# wind (63336.0)       feature 13 WINDPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.39995487770551125
# wind (63336.0)       feature 14 WINDPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.3909760337380804
# hail (28152.0)       feature 15 HAILPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.24281549180460507
# hail (28152.0)       feature 16 HAILPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.24208054472267743
# hail (28152.0)       feature 17 HAILPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.24082552832989604
# hail (28152.0)       feature 18 HAILPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.23807982360404809
# hail (28152.0)       feature 19 HAILPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.2323965848387333
# hail (28152.0)       feature 20 HAILPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.22298719203063502
# hail (28152.0)       feature 21 HAILPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.20709992125171317
# sig_tornado (1138.0) feature 22 STORPROB:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.09322716799320344
# sig_tornado (1138.0) feature 23 STORPROB:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.0933847036985924
# sig_tornado (1138.0) feature 24 STORPROB:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.09335577047000712
# sig_tornado (1138.0) feature 25 STORPROB:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.0932838322078577
# sig_tornado (1138.0) feature 26 STORPROB:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.09339258977120611
# sig_tornado (1138.0) feature 27 STORPROB:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.09485998312492377
# sig_tornado (1138.0) feature 28 STORPROB:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.0890668815603778
# sig_wind (7555.0)    feature 29 SWINDPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.08475474530886153
# sig_wind (7555.0)    feature 30 SWINDPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.08438349231988675
# sig_wind (7555.0)    feature 31 SWINDPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.08400587484598891
# sig_wind (7555.0)    feature 32 SWINDPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.08310618991612609
# sig_wind (7555.0)    feature 33 SWINDPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.08168829200078392
# sig_wind (7555.0)    feature 34 SWINDPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.07995998955198079
# sig_wind (7555.0)    feature 35 SWINDPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.07721169925889398
# sig_hail (3887.0)    feature 36 SHAILPRO:calculated:hour fcst:calculated_prob:           AU-PR-curve: 0.07277620173520509
# sig_hail (3887.0)    feature 37 SHAILPRO:calculated:hour fcst:calculated_prob:15mi mean  AU-PR-curve: 0.07259306402521629
# sig_hail (3887.0)    feature 38 SHAILPRO:calculated:hour fcst:calculated_prob:25mi mean  AU-PR-curve: 0.07228935492454468
# sig_hail (3887.0)    feature 39 SHAILPRO:calculated:hour fcst:calculated_prob:35mi mean  AU-PR-curve: 0.07162547245086157
# sig_hail (3887.0)    feature 40 SHAILPRO:calculated:hour fcst:calculated_prob:50mi mean  AU-PR-curve: 0.07041989216931623
# sig_hail (3887.0)    feature 41 SHAILPRO:calculated:hour fcst:calculated_prob:70mi mean  AU-PR-curve: 0.0677679578882266
# sig_hail (3887.0)    feature 42 SHAILPRO:calculated:hour fcst:calculated_prob:100mi mean AU-PR-curve: 0.056915442960505755

# Well darn. Any blurring decreases the AU-PR

