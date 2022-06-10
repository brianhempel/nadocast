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
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
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



# 3. bin predictions into 6 bins of equal weight of positive labels

const bin_count = 6

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
  (event_name, _) = CombinedHREFSREF.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

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
# Dict{String, Vector{Float32}}("sig_hail" => [0.012925774, 0.02218357, 0.033009935, 0.053475916, 0.09119451, 1.0], "hail" => [0.03750208, 0.082046196, 0.14193675, 0.21595338, 0.34294724, 1.0], "tornado" => [0.009306263, 0.028845591, 0.061602164, 0.106333174, 0.19435626, 1.0], "sig_tornado" => [0.0046208645, 0.015867244, 0.027446639, 0.07221879, 0.14377913, 1.0], "sig_wind" => [0.0071317935, 0.025168268, 0.047133457, 0.07531253, 0.10004484, 1.0], "wind" => [0.07021518, 0.16140154, 0.2604162, 0.36931401, 0.52724814, 1.0])


# 4. combine bin-pairs (overlapping, 5 bins total)
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

# event_name  bin  total_prob_ŷ_min total_prob_ŷ_max count    pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr     max_hourly_au_pr     mean_logistic_ŷ logistic_logloss logistic_au_pr       logistic_coeffs
# tornado     1-2  -1.0             0.028845591      4689564  2814.0    4.288693e6  0.0007422392      0.00016006072     0.00060791126 0.003871063        0.0042581437       0.014462792800350928 0.012635911772633497 0.0006079113    0.0038562466     0.014231701972089278 Float32[0.7936844,    0.19060811,   0.025705637]
# tornado     2-3  0.009306263      0.061602164      147009   2791.0    137209.4    0.023863837       0.005636964       0.018994804   0.09162543         0.10247716         0.03454350849804378  0.029432821207813285 0.018994799     0.091061324      0.03383744795603824  Float32[1.1202875,    -0.06516523,  -0.13911094]
# tornado     3-4  0.028845591      0.106333174      59587    2778.0    56066.11    0.053398788       0.013147442       0.046496954   0.1830729          0.21367657         0.0839970416857711   0.0622048591745176   0.04649697      0.18211001       0.08678483919318042  Float32[1.5175567,    -0.2130138,   0.37245846]
# tornado     4-5  0.061602164      0.19435626       28587    2764.0    27192.637   0.10350753        0.026276197       0.09588122    0.31304207         0.3791161          0.11458737231243346  0.10609080990906264  0.09588122      0.3122851        0.11550827806806185  Float32[0.9057919,    -0.1376394,   -0.7896346]
# tornado     5-6  0.106333174      1.0              17369    2734.0    16678.816   0.20126675        0.049943827       0.15613738    0.43468177         0.5144694          0.2292826134361234   0.21053903156276357  0.15613738      0.42420557       0.22953397641274584  Float32[0.5802065,    0.06708273,   -0.6733654]
# wind        1-2  -1.0             0.16140154       4624531  21119.0   4.229627e6  0.006019691       0.001404398       0.0046382216  0.01985602         0.02216605         0.08411796591760189  0.07722915812735964  0.0046382216    0.019648202      0.08278253419329806  Float32[0.6955889,    0.29533672,   0.15469939]
# wind        2-3  0.07021518       0.2604162        186415   21154.0   173216.73   0.13923141        0.03412005        0.11325637    0.3441893          0.40872324         0.1722143685794772   0.15362272025001836  0.11325637      0.34094813       0.17198825561390424  Float32[1.1739702,    -0.03139503,  -0.04425332]
# wind        3-4  0.16140154       0.36931401       97168    21176.0   90132.34    0.24608909        0.06265442        0.21766138    0.5171974          0.6587107          0.27909190799965994  0.2510702850440768   0.2176614       0.5148581        0.27803151549909155  Float32[1.0905153,    -0.07237125,  -0.26282614]
# wind        4-5  0.2604162        0.52724814       65736    21156.0   60962.535   0.3641847         0.098410726       0.3218066     0.62493676         0.8268598          0.3819448138802237   0.34888888974144633  0.32180658      0.6206163        0.38153565941395445  Float32[0.8646443,    -0.028500073, -0.3300517]
# wind        5-6  0.36931401       1.0              44821    21041.0   41678.71    0.52191436        0.16171792        0.4706375     0.6551853          0.942167           0.6439001392281727   0.6160938831642527   0.47063747      0.64927155       0.644782806958358    Float32[0.9196351,    0.12227182,   0.0019935018]
# hail        1-2  -1.0             0.082046196      4646124  9402.0    4.250196e6  0.0025071348      0.00054752285     0.0020403531  0.009877746        0.011058615        0.04544640980990685  0.04538698664281539  0.0020403531    0.0098004155     0.04663056881339708  Float32[0.67516387,   0.39498883,   0.6715193]
# hail        2-3  0.03750208       0.14193675       152689   9412.0    141118.78   0.07480181        0.016530145       0.061451122   0.22844316         0.26651338         0.08641852391054368  0.08596422511745538  0.061451115     0.22697197       0.08673107419469861  Float32[0.78884,      0.11251911,   -0.2715415]
# hail        3-4  0.082046196      0.21595338       86382    9383.0    79803.516   0.13264024        0.029983902       0.10865775    0.34118998         0.4095901          0.14372156017787538  0.12900418228892985  0.10865775      0.33843616       0.14782315900185797  Float32[1.1522207,    -0.12645392,  -0.39178276]
# hail        4-5  0.14193675       0.34294724       53514    9362.0    49411.996   0.2121372         0.050252527       0.17549801    0.463454           0.5782509          0.2204115925776582   0.1856974735250258   0.175498        0.45611164       0.24111646299591785  Float32[1.5031779,    -0.6665933,   -1.5814598]
# hail        5-6  0.21595338       1.0              34014    9367.0    31438.219   0.3382563         0.09376451        0.27578184    0.5779708          0.7274174          0.39863386036383147  0.36903534257310916  0.27578184      0.5642137        0.406531428684357    Float32[1.5379245,    -0.69794226,  -1.6085442]
# sig_tornado 1-2  -1.0             0.015867244      4748128  390.0     4.34371e6   0.00010844632     2.5392064e-5      8.3863386e-5  0.0006223221       0.0006690699       0.008880386392798145 0.006172449602071954 8.386338e-5     0.00061880564    0.007379170134549084 Float32[1.1938772,    -0.13410978,  -0.09725317]
# sig_tornado 2-3  0.0046208645     0.027446639      32754    379.0     31108.605   0.01164054        0.0031384283      0.011690505   0.06106865         0.06781364         0.02483337522720498  0.026646548666253534 0.011690505     0.060605284      0.029185760823168684 Float32[0.8071592,    0.551987,     2.28532]
# sig_tornado 3-4  0.015867244      0.07221879       14704    377.0     14148.238   0.03127931        0.008678908       0.025721904   0.11995059         0.1260357          0.03533286568783807  0.08911305902202721  0.025721904     0.113770515      0.10934501375602027  Float32[-1.0015389,   1.529142,     0.14288497]
# sig_tornado 4-5  0.027446639      0.14377913       9525     374.0     9202.984    0.058088742       0.01505509        0.039581172   0.16167928         0.17018336         0.09101479929226293  0.08313799540225217  0.039581172     0.15442018       0.086275821770572    Float32[0.6298735,    0.917871,     2.3416257]
# sig_tornado 5-6  0.07221879       1.0              3688     371.0     3579.4497   0.12934783        0.030058177       0.10098261    0.31616318         0.38154572         0.17843359320930569  0.12431756150506458  0.10098261      0.31031105       0.1764376582926154   Float32[1.6475935,    -0.2897538,   -0.1320373]
# sig_wind    1-2  -1.0             0.025168268      4675724  2526.0    4.277161e6  0.0006952304      0.00015001454     0.00054551166 0.0035856464       0.0038558803       0.009536931974211346 0.008460207845250681 0.0005455118    0.0035311906     0.008450174164615399 Float32[-0.060351126, 0.9043533,    0.35478225]
# sig_wind    2-3  0.0071317935     0.047133457      183926   2516.0    170770.53   0.01832032        0.0042028744      0.013661203   0.07023827         0.07698917         0.029101739657142797 0.024351077490795035 0.013661202     0.06951773       0.02845724937074389  Float32[1.0681612,    0.08553178,   0.41871232]
# sig_wind    3-4  0.025168268      0.07531253       68340    2518.0    63465.945   0.043990556       0.010438218       0.03677082    0.15630089         0.17583603         0.05401885354228505  0.06175325539281782  0.036770828     0.1550702        0.06257931213972666  Float32[0.46654662,   0.55500215,   0.7106551]
# sig_wind    4-5  0.047133457      0.10004484       38272    2517.0    35577.652   0.06829264        0.015696067       0.065596975   0.23818375         0.28149912         0.09111915714261841  0.10951173210533217  0.06559698      0.23476739       0.10714482029585953  Float32[0.8931556,    1.0609696,    4.039079]
# sig_wind    5-6  0.07531253       1.0              22456    2511.0    20810.662   0.10683871        0.023487872       0.111996      0.35069424         0.44430575         0.13411506642169316  0.1384452756306478   0.111996        0.34924227       0.13625251358904888  Float32[0.3618952,    0.17861357,   -0.61968464]
# sig_hail    1-2  -1.0             0.02218357       4704590  1299.0    4.303999e6  0.00034676003     8.250561e-5       0.00028032428 0.0016777442       0.0018481778       0.017051591682444987 0.012860554888894867 0.00028032428   0.0016562227     0.02136738814401973  Float32[1.9470923,    -0.7360531,   -0.30602866]
# sig_hail    2-3  0.012925774      0.033009935      59078    1300.0    54609.11    0.021081941       0.004738537       0.022099597   0.10531084         0.124962874        0.027856359074819502 0.021873910317067452 0.022099588     0.104085326      0.03490307122775223  Float32[2.0595398,    -1.1718963,   -2.2438788]
# sig_hail    3-4  0.02218357       0.053475916      43805    1307.0    40514.93    0.033861727       0.007541726       0.029786868   0.13404477         0.15515667         0.03572546478684759  0.030328331355828672 0.029786864     0.13318996       0.035795082592082646 Float32[1.2635527,    -0.6335322,   -2.3711135]
# sig_hail    4-5  0.033009935      0.09119451       32553    1302.0    30155.564   0.051661156       0.012027143       0.040012635   0.16640975         0.18801752         0.06699533823827789  0.051493035675145925 0.04001265      0.16421013       0.07106589188924511  Float32[2.0416996,    -0.59624666,  0.07782969]
# sig_hail    5-6  0.053475916      1.0              18125    1281.0    16923.863   0.08946202        0.022502713       0.071058355   0.25275388         0.28804302         0.12515506619167865  0.10665370085731583  0.07105836      0.25023583       0.1278968660990404   Float32[1.308884,     -0.365359,    -0.92940474]


# sig_tor with one fewer bin
# sig_tornado 1-2  -1.0             0.021278545      4752637  465.0     4.348028e6  0.00012603919     3.030781e-5       0.00010042195 0.00070720527      0.0007608925       0.014199921503762477 0.010587076316694567 0.000100421945  0.00070387236   0.01443408233067957     Float32[0.9629071, 0.071850866, 0.08534911]
# sig_tornado 2-3  0.006523651      0.05566356       30132    454.0     28779.32    0.018333986       0.0050506913      0.015189859   0.077535965        0.0828197          0.024081905255219103 0.04326837530765084  0.01518986      0.07597819      0.04548438749117086     Float32[-0.20521086, 0.95460045, 0.09853549]
# sig_tornado 3-4  0.021278545      0.13020535       12502    451.0     12067.658   0.048245702       0.012979303       0.03620263    0.15516266         0.16438662         0.059853987637833284 0.06540540812511837  0.036202632     0.14990279      0.06513822220471986     Float32[-0.12533844, 0.9560777, 0.5157306]
# sig_tornado 4-5  0.05566356       1.0              5017     446.0     4866.9185   0.109773226       0.026342222       0.08928919    0.2904483          0.34440392         0.1661787170429437   0.12158929409747414  0.08928918      0.2876767       0.16700476701583156     Float32[1.0730737, 0.14802879, 0.44363177]

# sig_tor with two fewer bins
# sig_tornado 1-2  -1.0             0.02879601       4756366  579.0     4.3516215e6 0.00014641577     3.59034e-5        0.00012538236 0.0008202097       0.0008887818       0.016770842668968083 0.018119259372910784 0.00012538236   0.00081660494    0.01796746725717596     Float32[0.7700436, 0.2685168, 0.45433167]
# sig_tornado 2-3  0.010408389      0.11002444       24143    564.0     23169.781   0.02950164        0.008097223       0.02353592    0.109691575        0.11672291         0.04324491094403623  0.05395874530684064  0.023535926     0.106585525      0.05426780902263358     Float32[-0.2203767, 1.006804, 0.38971102]
# sig_tornado 3-4  0.02879601       1.0              10154    559.0     9816.059    0.07418769        0.018606294       0.05542382    0.19981378         0.22363822         0.14687125266873996  0.11497895297785568  0.055423807     0.19499768       0.14321728513452953     Float32[0.83568263, 0.53293943, 1.3344601]

# Don't change the bin_count for sig_tor. It probably doesn't matter if we only have a 10% contour anyway.


print("event_to_0z_day_bins_logistic_coeffs = ")
println(event_to_day_bins_logistic_coeffs)
# event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.9470923, -0.7360531, -0.30602866], [2.0595398, -1.1718963, -2.2438788], [1.2635527, -0.6335322, -2.3711135], [2.0416996, -0.59624666, 0.07782969], [1.308884, -0.365359, -0.92940474]], "hail" => [[0.67516387, 0.39498883, 0.6715193], [0.78884, 0.11251911, -0.2715415], [1.1522207, -0.12645392, -0.39178276], [1.5031779, -0.6665933, -1.5814598], [1.5379245, -0.69794226, -1.6085442]], "tornado" => [[0.7936844, 0.19060811, 0.025705637], [1.1202875, -0.06516523, -0.13911094], [1.5175567, -0.2130138, 0.37245846], [0.9057919, -0.1376394, -0.7896346], [0.5802065, 0.06708273, -0.6733654]], "sig_tornado" => [[1.1938772, -0.13410978, -0.09725317], [0.8071592, 0.551987, 2.28532], [-1.0015389, 1.529142, 0.14288497], [0.6298735, 0.917871, 2.3416257], [1.6475935, -0.2897538, -0.1320373]], "sig_wind" => [[-0.060351126, 0.9043533, 0.35478225], [1.0681612, 0.08553178, 0.41871232], [0.46654662, 0.55500215, 0.7106551], [0.8931556, 1.0609696, 4.039079], [0.3618952, 0.17861357, -0.61968464]], "wind" => [[0.6955889, 0.29533672, 0.15469939], [1.1739702, -0.03139503, -0.04425332], [1.0905153, -0.07237125, -0.26282614], [0.8646443, -0.028500073, -0.3300517], [0.9196351, 0.12227182, 0.0019935018]])





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

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day(); just_hours_near_storm_events = false);

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

# rm("day_accumulators_validation_forecasts_0z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z",
  );

# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z, X, Ys, weights)

# tornado     (8326.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.12836913463206878
# wind        (63336.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.4069140205489688
# hail        (28152.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.2420451273585057
# sig_tornado (1138.0)  feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.09322992381110502
# sig_wind    (7555.0)  feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.08473176338436793
# sig_hail    (3887.0)  feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07212143119615202


# rm("day_accumulators_validation_forecasts_0z"; recursive = true)

# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  total_weight = sum(Float64.(weights))

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
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
      Σweight = bins_Σweight[bin_i]

      mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      pos_weight_in_and_after = sum(bins_Σy[bin_i:bin_count])
      weight_in_and_after     = sum(bins_Σweight[bin_i:bin_count])

      sr  = pos_weight_in_and_after / weight_in_and_after
      pod = pos_weight_in_and_after / total_pos_weight

      println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$sr\t$pod\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(day_validation_forecasts_0z, X, Ys, weights)

# event_name  mean_y                 mean_ŷ                 Σweight              SR                    POD                  bin_max
# tornado     0.00010158509480873166 9.390493680294686e-5   3.854919375448823e6  0.0017925820632635695 1.0                  0.00085540616
# tornado     0.0016835595922604957  0.0016651843061069556  232672.24124258757   0.01466212254455693   0.9499116669695081   0.0031741688
# tornado     0.00452105209434993    0.004591751374465388   86559.29108393192    0.02568931066563998   0.8998085948880513   0.0065524527
# tornado     0.007983884880147849   0.008372361722991882   49062.99536931515    0.03547275707701353   0.8497539377447659   0.0105718635
# tornado     0.01192507299283904    0.012688744700454852   32784.969016849995   0.045230029541050897  0.7996514126784546   0.015141781
# tornado     0.01701752491910662    0.017399058182366894   23006.06199759245    0.055585824269944524  0.7496448374429632   0.02001289
# tornado     0.02549542051015475    0.02233872215727473    15349.35876661539    0.06634981600318184   0.6995687983512499   0.024970705
# tornado     0.02744559817911145    0.02836459514019185    14262.183460652828   0.07569772487441127   0.6495142317558157   0.032296836
# tornado     0.03840247721623231    0.03633814520578238    10190.412433803082   0.0887262341490591    0.5994474117420906   0.041030705
# tornado     0.04368590623215632    0.04698578514923635    8960.0092304945      0.10075558233292455   0.5493930082273241   0.05434151
# tornado     0.054870623567619584   0.06297533239110333    7132.497891783714    0.11594221581836198   0.4993272157262507   0.072221056
# tornado     0.07328135971161581    0.07948680274440958    5336.508499443531    0.13235601570176184   0.4492692915185983   0.08679409
# tornado     0.10254791500514812    0.09222887876486081    3821.1031135320663   0.14722521295438903   0.3992494843503153   0.09781244
# tornado     0.14264257400395047    0.10253685127601182    2744.245426297188    0.15704745715144033   0.3491299669051205   0.10777641
# tornado     0.14814991821709933    0.1137963927777963     2643.190289080143    0.15974830159154468   0.29906160243948215  0.12027094
# tornado     0.1335363828655451     0.12798688612427622    2931.451932489872    0.1623044841561477    0.24897505051122384  0.13626921
# tornado     0.09195120814038012    0.15117092568871218    4255.802161514759    0.17161092000864542   0.19890550210784214  0.17010313
# tornado     0.16618079226779384    0.1876540348952982     2357.441847860813    0.242152294083271     0.14885249672905357  0.20820199
# tornado     0.3199683432447803     0.22114580933008723    1224.5362378954887   0.31529921286373325   0.09874380056403172  0.23675627
# tornado     0.31062780298195614    0.28318044590199326    1223.9387028217316   0.31062780298195614   0.04862854572164576  1.0
# wind        0.0007628483147496947  0.0007006327063574261  3.858066771569133e6  0.013493679734745061  1.0                  0.01230654
# wind        0.01946409855242839    0.019346380972783513   151202.44416177273   0.11106865532088359   0.9499910348182923   0.028873874
# wind        0.03529676972853432    0.0373340834898203     83380.86593061686    0.15039879664472294   0.899983776640928    0.047748838
# wind        0.054231372111923276   0.057280168400195376   54271.40076345205    0.18610471866300524   0.8499755677713893   0.06815834
# wind        0.0746418877006175     0.07891647575465742    39431.25773507357    0.21946794959126226   0.799965025834374    0.091209024
# wind        0.1041025448323461     0.10245749660026947    28267.162171840668   0.252084552287524     0.749954297294579    0.11508566
# wind        0.1353792834920697     0.12768934715609       21736.833072185516   0.280575879058342     0.6999527442445678   0.14208809
# wind        0.163919019164853      0.15566520476568427    17956.051456272602   0.30580846610147616   0.6499506241872951   0.16997153
# wind        0.18228003451735644    0.18441498144347324    16146.427366316319   0.3295916942725393    0.5999379444591296   0.19973257
# wind        0.2052272948352602     0.2147659838298436     14340.427252352238   0.3557358891183967    0.5499281024521814   0.23071185
# wind        0.2451024941646597     0.24588539074193386    12006.854484856129   0.3838989194263858    0.49992037160908254  0.2614823
# wind        0.27858441079601426    0.2760387024961762     10562.98784172535    0.4096838435901479    0.44991497052264473  0.29062456
# wind        0.3127458313120847     0.30403230411225696    9409.877666413784    0.43529600561154413   0.3999134136179491   0.31727675
# wind        0.3338824404328331     0.33055865935446826    8815.749122738838    0.4611183806243612    0.34990818190569517  0.3451302
# wind        0.3528340017218528     0.3620489758626988     8339.949866592884    0.4924130731474248    0.2998940495712152   0.3809266
# wind        0.39650364343179156    0.40228646355163683    7422.455709218979    0.5347394138628928    0.24989361172312494  0.42751628
# wind        0.46153039328212736    0.45864775789521195    6376.666961491108    0.5858372164897662    0.1998861588639432   0.4955455
# wind        0.560201621875504      0.5330100395174592     5253.988496541977    0.6436813586905109    0.1498787930258314   0.5734172
# wind        0.614220638111561      0.625915425665927      4791.84612762928     0.6955902953610114    0.09986688584285677  0.6935886
# wind        0.8021933304306803     0.8031444633118673     3657.5963971614838   0.8021933304306803    0.049855693859806344 1.0
# hail        0.0003267612403996816  0.00032551604014587987 3.982413546228349e6  0.005964377746736281  1.0                  0.0077075996
# hail        0.012022493417496428   0.012004257724236115   108257.93837124109   0.06519892838143601   0.9499755771620737   0.017877249
# hail        0.024131034920325357   0.02251148072902746    53933.76646721363    0.08645997468961078   0.8999422372834927   0.027896967
# hail        0.03275143309904753    0.03327890247725313    39736.00143682957    0.1019633545071492    0.8499109251958165   0.039338037
# hail        0.04667641336993955    0.044334457384785676   27870.477489352226   0.11749278144115487   0.7998821736991356   0.04971889
# hail        0.05280183217722742    0.05514764221743547    24648.858701467514   0.13071894527533942   0.7498732991629958   0.06088075
# hail        0.06586509270413439    0.06595305155767461    19751.82122963667    0.1461356555724942    0.6998409376678808   0.07139729
# hail        0.07771721995521792    0.07735674070230221    16745.77843338251    0.16126073755478199   0.6498296947914609   0.08393406
# hail        0.08975880476175573    0.09071405984532377    14497.605015397072   0.17714414304506407   0.5998000071128021   0.09816896
# hail        0.1017905743963123     0.1064713461877492     12785.855912446976   0.19436141363078635   0.5497759952983077   0.11540992
# hail        0.12470250864016649    0.12357291399501524    10430.42940723896    0.21382976839273088   0.49974460573201696  0.13214844
# hail        0.14316972411635356    0.14064555884170926    9085.474101483822    0.23228754080801214   0.44974315874717097  0.14971362
# hail        0.15827792903176932    0.15944824523999707    8217.738314211369    0.25190179497504317   0.39973924735397026  0.17048073
# hail        0.1761677476771446     0.18268459812434945    7387.613867521286    0.27517228414291944   0.34973834284544897  0.19647086
# hail        0.20758734079622465    0.21046870874540782    6266.51290088892     0.3036597455385802    0.2997077344855763   0.22592762
# hail        0.24361368677895343    0.24174912979221225    5339.834268987179    0.3346795762922191    0.24970059671550565  0.25958523
# hail        0.29804613575574496    0.277433920171316      4366.388456404209    0.36924484816191505   0.19969315279790278  0.2974602
# hail        0.3439843163221874     0.32125540201133673    3782.9503222703934   0.4012879406643093    0.14966539256647568  0.3482993
# hail        0.3984396434573551     0.3823225977951217     3264.4850445985794   0.43791168005223935   0.09964184658373092  0.4246016
# hail        0.4864534090422358     0.5201742081283158     2654.538184463978    0.4864534090422358    0.0496404171297562   1.0
# sig_tornado 1.3022278497742202e-5  1.5745622631599445e-5  4.217433150374651e6  0.0002498396427244399 1.0                  0.00036549053
# sig_tornado 0.0015035282141337362  0.0004933728952373021  36741.4005138278     0.007185467721647326  0.9495984176170549   0.0006640795
# sig_tornado 0.0008938492345409595  0.00148320137177481    61554.24628943205    0.00913173336786815   0.8989021164548261   0.0031850245
# sig_tornado 0.004223253932116711   0.004160697411060583   12913.68414670229    0.02022536436511546   0.848409104729363    0.0054826955
# sig_tornado 0.006359903810272938   0.0070978138197027845  8613.414980828762    0.02652648840359165   0.7983588444778583   0.009181244
# sig_tornado 0.012540242672801849   0.010738672226583056   4390.937982797623    0.033709731667800676  0.74808583272409     0.012478315
# sig_tornado 0.009818222783315386   0.015017072788957137   5618.66762316227     0.038406560895246196  0.6975531646504591   0.017703766
# sig_tornado 0.016228241714967784   0.019629832922255765   3366.0224453806877   0.04974067729277773   0.6469269787798332   0.021944363
# sig_tornado 0.030068051881317653   0.02358125578785842    1840.3754978179932   0.060179569850356364  0.5967970069579738   0.025286011
# sig_tornado 0.033668181080729546   0.02706654322442566    1646.5642812252045   0.06636050513621937   0.5460137303641058   0.029121827
# sig_tornado 0.04262589663757118    0.031189468020203164   1288.9276103973389   0.07371518606650013   0.4951383897957441   0.0338217
# sig_tornado 0.03310690409806793    0.03915538130867662    1675.2664709687233   0.08036035449096597   0.4447174380094772   0.046756312
# sig_tornado 0.04323492840298834    0.05644474397668932    1266.4942452311516   0.09853783372167413   0.3938181798784324   0.06747354
# sig_tornado 0.07484088621759073    0.07530582822469974    729.9910952448845    0.1212160529580135    0.3435669205431587   0.083285406
# sig_tornado 0.10167998943798877    0.08956949566540029    537.3568024039268    0.13557004101386516   0.29342909798133254  0.09590599
# sig_tornado 0.08028834314709732    0.10544071870162243    679.3891060948372    0.14556999259019265   0.24328645419663675  0.11842266
# sig_tornado 0.1895634616131176     0.1260794249469163     287.8256697654724    0.1844161714948332    0.19322769741434395  0.13279438
# sig_tornado 0.13001792737051665    0.14745361189744755    419.5798325538635    0.18268116279299498   0.14315590480333787  0.17273259
# sig_tornado 0.2581063773806567     0.19292340394782737    211.28467506170273   0.23355718270777223   0.0930917568765621   0.21225753
# sig_tornado 0.2103012810840145     0.2558631393448241     223.0345098376274    0.2103012810840145    0.04304502550048126  1.0
# sig_wind    8.999379366238849e-5   9.102030928631656e-5   3.8932922517692447e6 0.0016044359095530234 1.0                  0.0012931518
# sig_wind    0.0023777263053332146  0.001967480475742761   147424.92167782784   0.014199168719726377  0.9499300074306162   0.0028920358
# sig_wind    0.0033797666134274145  0.003985156002549471   103750.35929811001   0.019633107442879803  0.899836582018875    0.005404309
# sig_wind    0.008308194263616961   0.00625755930300869    42153.22316598892    0.02740510042875963   0.8497265958761175   0.007238741
# sig_wind    0.008543126754953589   0.008560981065201038   41036.49090951681    0.03200989744578338   0.7996787497038991   0.010205837
# sig_wind    0.010877083988595273   0.012339450571020255   32177.709282398224   0.03920821765771024   0.7495790608892372   0.015032627
# sig_wind    0.016050871086411594   0.017744259266731153   21838.907962858677   0.048180728813808534  0.6995622995190228   0.02070501
# sig_wind    0.02240643318192459    0.023570511894292222   15625.96782642603    0.05697771999086774   0.6494692482266335   0.026907485
# sig_wind    0.03240901976153769    0.029732897442310433   10814.644120156765   0.06540036568192491   0.5994349727437074   0.03274574
# sig_wind    0.037398726145798804   0.03575326801326003    9358.998250126839    0.07209145097085681   0.5493478493546478   0.038995907
# sig_wind    0.035411666439393696   0.04298903743259204    9897.393170952797    0.07947676890808567   0.49932894924694216  0.04764224
# sig_wind    0.04595754761150309    0.05287136654986286    7620.957300007343    0.09227897887073438   0.4492430876269724   0.059467852
# sig_wind    0.07600358004171391    0.06533642991856965    4611.55403649807     0.10562755561762184   0.3991919058964918   0.07244191
# sig_wind    0.09372398233542781    0.07967595003493022    3739.353805422783    0.11188436112265276   0.3491044107645498   0.08716396
# sig_wind    0.09663725177146203    0.09422148594838253    3628.5913411974907   0.11563725009480595   0.2990208435706444   0.10086389
# sig_wind    0.07736853129181254    0.1065653474306822     4533.680104911327    0.12040303512130691   0.2489101293934338   0.11124379
# sig_wind    0.10081058643490716    0.1142933073431904     3476.74697804451     0.1400458753464592    0.19878411349452565  0.117595986
# sig_wind    0.13900545550765067    0.12082582795846034    2517.397510945797    0.16117568694691806   0.14869686338597238  0.12472014
# sig_wind    0.17425610969106423    0.1291895184891597     2011.0009972453117   0.17534650637377752   0.09868977079744809  0.1345561
# sig_wind    0.17648416086795632    0.1545444863102951     1927.4646455049515   0.17648416086795632   0.04861162251648032  1.0
# sig_hail    4.343952161916368e-5   3.6766455948559764e-5  4.1817153905524015e6 0.0008290638627045212 1.0                  0.0021585145
# sig_hail    0.003357468309345073   0.0036022763299618375  53894.627737760544   0.019108703033000482  0.9497632130579347   0.005692235
# sig_hail    0.006115318769730393   0.007700627254365139   29616.974423468113   0.025855290902042128  0.899720626573948    0.010149463
# sig_hail    0.011622517126609035   0.011906044981458865   15627.129934608936   0.03193194066288728   0.8496316520696979   0.013822436
# sig_hail    0.014762700710773473   0.015708537280602543   12294.877278506756   0.035870439660267615  0.7994018151917      0.01773994
# sig_hail    0.0157546972995107     0.019914047709428288   11511.758663654327   0.03967073730797637   0.7492054392560101   0.022188077
# sig_hail    0.030924464054924623   0.023342941448534216   5846.480033040047    0.04451982023160427   0.6990481462607325   0.024484802
# sig_hail    0.025186576899346308   0.02582772605465628    7193.171595990658    0.04608047981052867   0.6490471103216425   0.027130838
# sig_hail    0.02898991082807675    0.028303351338755677   6249.197564303875    0.04951676323589476   0.5989431658780322   0.029518073
# sig_hail    0.03631819833172287    0.030548537321432475   4988.18860322237     0.05293856045831553   0.5488413479850656   0.031643145
# sig_hail    0.037226621809984706   0.032926860905541096   4879.281880140305    0.05548951325248592   0.4987399883801064   0.03427232
# sig_hail    0.03519468040073909    0.03597277297452064    5161.4110187888145   0.05871572854610752   0.4485066680220277   0.037877843
# sig_hail    0.03560083428159058    0.04026828657245059    5093.279887378216    0.06412118391390105   0.3982691910434799   0.043074872
# sig_hail    0.0352538783862315     0.0474143112217813     5132.951234996319    0.07248600136617755   0.3481227555936501   0.052795175
# sig_hail    0.056886310996942045   0.05715993139638046    3187.451330959797    0.08810868849264915   0.2980782514936466   0.061711468
# sig_hail    0.0772534981770518     0.06555014323062906    2348.5155401825905   0.09911089315525133   0.24793255320803875  0.06940585
# sig_hail    0.09641960749020274    0.0732414395413395     1875.1829680204391   0.10677596591775092   0.1977567951885883   0.07770982
# sig_hail    0.10964583402746571    0.08310517352982227    1654.0318952798843   0.11080356823608281   0.14775433661486026  0.08936867
# sig_hail    0.1419709819100177     0.09625138269759345    1279.355123758316    0.1114080831985706    0.09759885961177511  0.10464452
# sig_hail    0.09070182722854422    0.1402519298565085     1888.3568869233131   0.09070182722854422   0.04736771799266581  1.0



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
function spc_calibrate(prediction_i, X, Ys, weights)
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

  println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tmean_threshold\tSR\tPOD\tWR")
  thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_success_ratio = thresholds_to_match_success_ratio[i]
    threshold_to_match_POD = thresholds_to_match_POD[i]
    mean_threshold = (threshold_to_match_success_ratio + threshold_to_match_POD) * 0.5f0
    sr  = Metrics.success_ratio(ŷ, y, weights, mean_threshold)
    pod = Metrics.probability_of_detection(ŷ, y, weights, mean_threshold)
    wr  = Metrics.warning_ratio(ŷ, weights, mean_threshold)
    println("$event_name\t$nominal_prob\t$threshold_to_match_success_ratio\t$threshold_to_match_POD\t$mean_threshold\t$sr\t$pod\t$wr")
    push!(thresholds, (Float32(nominal_prob), Float32(mean_threshold)))
  end

  println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
  wr_thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    sr  = Metrics.success_ratio(ŷ, y, weights, threshold_to_match_warning_ratio)
    pod = Metrics.probability_of_detection(ŷ, y, weights, threshold_to_match_warning_ratio)
    wr  = Metrics.warning_ratio(ŷ, weights, threshold_to_match_warning_ratio)
    println("$event_name\t$nominal_prob\t$threshold_to_match_warning_ratio\t$sr\t$pod\t$wr")
    push!(wr_thresholds, (Float32(nominal_prob), Float32(threshold_to_match_warning_ratio)))
  end

  thresholds
end

calibrations = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations[event_name] = spc_calibrate(prediction_i, X, Ys, weights)
end

# event_name   nominal_prob  threshold_to_match_success_ratio  threshold_to_match_POD  mean_threshold  success_ratio         POD
# tornado      0.02           0.013106078                      0.022882462             0.01799427      0.062248498633836996  0.7237450145071389
# tornado      0.05           0.05241271                       0.09069252              0.07155262      0.13041010693869282   0.4595979609764453
# tornado      0.1            0.16216                          0.1885128               0.17533639      0.2602604408764249    0.14314764321503318
# tornado      0.15           0.19976498                       0.25847054              0.22911775      0.3278878657777642    0.06873094450782086
# tornado      0.3            0.20561947                       0.30122948              0.25342447      0.28779861751854896   0.033010873389855366
# wind         0.05           0.021777954                      0.10754967              0.06466381      0.21349353908937124   0.8077435939062366
# wind         0.15           0.09295714                       0.2965603               0.19475871      0.35260570410877334   0.5574172101278139
# wind         0.3            0.29035583                       0.56562996              0.42799288      0.587179141844545     0.19983921956088188
# wind         0.45           0.45901677                       0.8673687               0.66319275      0.7595734431193403    0.06063552925564667
# hail         0.05           0.014443969                      0.050340652             0.03239231      0.10834954044348313   0.8305503616488299
# hail         0.15           0.070988156                      0.14591789              0.10845302      0.2070218904495598    0.5193486074366591
# hail         0.3            0.19329038                       0.3788433               0.28606683      0.3932871977659952    0.16613831846637464
# hail         0.45           0.50763494                       0.62512016              0.5663775       0.5888789612140184    0.013460490246840578
# sig_tornado  0.1            0.040612184                      0.121248245             0.08093022      0.13623988675028018   0.2807708016672483
# sig_wind     0.1            0.09281066                       0.12635994              0.1095853       0.13657851645778404   0.20438348540838872
# sig_hail     0.1            0.049650267                      0.064489365             0.057069816     0.09399092050668835   0.28431260828958804



# prediction_i = 1
# event_name, _ = CombinedHREFSREF.models[prediction_i];
# y = Ys[event_name];
# ŷ = @view X[:, prediction_i];
# for threshold in 0.25:0.01:0.6
#   sr  = success_ratio(ŷ, y, weights, threshold)
#   pod = probability_of_detection(ŷ, y, weights, threshold)
#   println("$threshold\t$sr\t$pod")
# end


println(calibrations)

# spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
#   "tornado" => [
#     (0.02, 0.01799427),
#     (0.05, 0.07155262),
#     (0.1,  0.17533639),
#     (0.15, 0.22911775),
#     (0.3,  0.25342447)
#   ],
#   "wind" => [
#     (0.05, 0.06466381),
#     (0.15, 0.19475871),
#     (0.3,  0.42799288),
#     (0.45, 0.66319275)
#   ],
#   "hail" => [
#     (0.05, 0.03239231),
#     (0.15, 0.10845302),
#     (0.3,  0.28606683),
#     (0.45, 0.5663775)
#   ],
#   "sig_tornado" => [(0.1, 0.08093022)],
#   "sig_wind"    => [(0.1, 0.1095853)],
#   "sig_hail"    => [(0.1, 0.057069816)],
# )




# CHECK

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

