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
# 903

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

# tornado (8326.0)     feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.12910289299673042
# wind (63336.0)       feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.407616606429577
# hail (28152.0)       feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.24281549180460507
# sig_tornado (1138.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.09322716799320344
# sig_wind (7555.0)    feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.08475474530886153
# sig_hail (3887.0)    feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07277620173520509

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

    bin_count = 40
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

# event_name      mean_y                  mean_ŷ                  Σweight                 SR                      POD                     bin_max
# tornado         5.382947269016896e-5    6.448005152838625e-5    3.634818565661609e6     0.0017925820632635695   1.0                     0.00040439222
# tornado         0.0008859982026943782   0.000589069082265124    221076.37919431925      0.010490468817576957    0.9749738437398083      0.00086057256
# tornado         0.0014366498095367965   0.0012087136322585627   136387.24079489708      0.014690552641494764    0.9499204552571546      0.001700473
# tornado         0.0020656831329180533   0.002332991687718857    94776.85343521833       0.019587306413603754    0.924858442207921       0.0031772556
# tornado         0.0038159702121628378   0.0039058996043948003   51233.537968695164      0.02563968977193661     0.8998171187366331      0.0047719465
# tornado         0.005734639994202583    0.005575177617870856    34233.1779949069        0.03065036076521324     0.8748107510815382      0.006503056
# tornado         0.007351714916302905    0.007408235832706979    26679.735600709915      0.03516539919674453     0.8497008694078892      0.008415136
# tornado         0.008429918404505096    0.00945777746007498     23195.824107050896      0.039739473225122446    0.8246131335872492      0.010609463
# tornado         0.009964571602087306    0.011803346098456027    19655.68997436762       0.044962935944169134    0.7996025130801834      0.013110504
# tornado         0.013895737051773539    0.01416611472218273     14115.28737872839       0.050725320660169694    0.774550754695192       0.015322382
# tornado         0.013951785118014698    0.016574927426763145    14013.088239431381      0.05566389061447657     0.7494629534318681      0.017923575
# tornado         0.02540870661824172     0.018752243493322603    7712.122001588345       0.06206938219108152     0.724456337484913       0.01962445
# tornado         0.02663042517600796     0.020557855938302838    7345.520790696144       0.06545376361672484     0.6993924900278486      0.021548603
# tornado         0.022535121088236517    0.022896511838575922    8705.276270747185       0.06919651800251482     0.6743722206359227      0.024305282
# tornado         0.02441064212964973     0.025949570889576213    8040.316584169865       0.07521525375172387     0.649280309360111       0.027750857
# tornado         0.03061120786508809     0.02944984561882837     6396.684827804565       0.08208644038305535     0.6241762679961922      0.031312197
# tornado         0.03754949933070936     0.03313307358515334     5223.85741764307        0.08829297160096643     0.599130940887273       0.035070717
# tornado         0.03622741311089039     0.037505134788465094    5400.1649579405785      0.09383520914487345     0.574041744324024       0.040267188
# tornado         0.03996236962241982     0.04341795100536715     4901.7700852155685      0.1011673829011186      0.5490189596580967      0.046943676
# tornado         0.04419982976725078     0.05105659091898469     4434.766051769257       0.10916207295546386     0.5239638990172858      0.055775974
# tornado         0.05646084666108782     0.06043410584325421     3468.753687977791       0.11786793797654904     0.49889226729630437     0.065375894
# tornado         0.05702711754223489     0.07044454282046031     3433.8780167102814      0.12505850160475238     0.4738420130943241      0.07542028
# tornado         0.06937924106722314     0.07946918930578696     2819.0962336063385      0.13397867328010146     0.4487949059962459      0.08347204
# tornado         0.09263647693282907     0.08627689110543174     2114.4635712504387      0.14177121956164324     0.423778164844766       0.089019515
# tornado         0.09758756157786916     0.09154262785291464     2006.4761206507683      0.1466590516023234      0.3987243694647541      0.09402186
# tornado         0.12883636396132664     0.09604414379990679     1524.185082912445       0.15177413950176746     0.373679443588711       0.09812938
# tornado         0.12314361073738642     0.10029908609597474     1591.9902200102806      0.15374658567445879     0.3485624615363399      0.1025251
# tornado         0.12644834314301204     0.10509334114914975     1548.3754054903984      0.15676647676151712     0.32348730991534763     0.10767612
# tornado         0.13898806371616182     0.11012835712790091     1408.4697750806808      0.1599852182352989      0.2984446361321679      0.11283314
# tornado         0.1503874781336828      0.11542824108741176     1300.8243907094002      0.1622297315769014      0.2734056738960384      0.11808852
# tornado         0.14661306379557015     0.12148221767648956     1338.295544743538       0.1635269423382357      0.24838369595756649     0.12491806
# tornado         0.12026170690709823     0.12925096778000636     1631.475820839405       0.16567517073865293     0.22328703230327743     0.13376965
# tornado         0.11425630929313803     0.1392840470166267      1714.6031048893929      0.17399488438792388     0.19819133345589973     0.14503756
# tornado         0.07841461717220106     0.15627848035173728     2500.566632926464       0.18823905471672636     0.1731339865294906      0.16993305
# tornado         0.13209995495682192     0.18135229529515368     1483.6576597690582      0.2467903740927034      0.14805403186988178     0.19425467
# tornado         0.23174062398489312     0.20314562254653265     847.4157806634903       0.29985561701400193     0.12298557151935213     0.21218765
# tornado         0.30457716980324595     0.21957648287405362     642.8411839008331       0.32432197875500357     0.09786728511896864     0.22716966
# tornado         0.3628771032922369      0.23442733585872522     541.012929558754        0.33171703560446086     0.07282394061689357     0.24274874    # Yeah, SR starts dropping after ~23%
# tornado         0.36802246840615477     0.2532705173469768      532.2051868438721       0.31737436300416566     0.04771325913819805     0.26570705
# tornado         0.275464372438063       0.32343905345227586     643.1684673428535       0.275464372438063       0.02266112516678776     1.0
# wind            0.0003967675946731406   0.0003881758013356758   3.710216119995475e6     0.013493679734745061    1.0                     0.0057232226
# wind            0.009924101242652241    0.008593560498781011    148330.43967574835      0.08811095662161089     0.9749864495841611      0.012408127
# wind            0.016869280264814192    0.015980350034195453    87246.19009917974       0.11117259299763925     0.9499737016060537      0.020205893
# wind            0.023531248938233548    0.024297211826152028    62538.911459863186      0.13096738767622668     0.9249654712076569      0.028963635
# wind            0.031653411510619095    0.03322950181576271     46499.46528184414       0.14999549839626267     0.8999599890483242      0.03795665
# wind            0.04063125497859663     0.04252473479173368     36223.06697922945       0.16794307609709141     0.8749502926504584      0.047530666
# wind            0.04893658444949763     0.052451647465248954    30074.191174149513      0.18499894665428443     0.8499419222782018      0.05771228
# wind            0.058670344688653994    0.06263423858026297     25078.150372624397      0.2020268631486829      0.8249345794760828      0.06788205
# wind            0.06695619817979606     0.07339731927981816     21974.608297228813      0.2187303817186524      0.79993376951326        0.0793224
# wind            0.08147887382511132     0.08487059313892371     18064.81615716219       0.23598816469885728     0.7749330869626443      0.090763375
# wind            0.10061902993721483     0.09621367995819642     14631.383243381977      0.25192032733303427     0.7499228100094053      0.10190772
# wind            0.11010732936664944     0.10776221006886473     13365.15973085165       0.2657079676245907      0.7249075253582982      0.11405978
# wind            0.12644270526605983     0.1203164901242653      11642.688966870308      0.2798363402363594      0.6999023261020109      0.12705496
# wind            0.1421915516840073      0.13391858007275448     10347.622352957726      0.2930114229506596      0.6748881034235428      0.14096478
# wind            0.15927573598560382     0.14774093208663597     9240.525711476803       0.30547601749189734     0.6498872806813168      0.15474236
# wind            0.1640058856091474      0.1618199587483839      8972.036297738552       0.31712587659116964     0.6248788616606341      0.1691668
# wind            0.17246587880613407     0.1765716721706341      8534.156206190586       0.32996604273862806     0.599875961156422       0.18422753
# wind            0.186749173604537       0.19178438677279944     7881.8708127141         0.34361785244413773     0.5748665359415785      0.19954073
# wind            0.20252114603970445     0.20706977127255774     7266.297641038895       0.3572684417423745      0.5498557155153139      0.21464002
# wind            0.21993145576015394     0.22213911639912182     6693.7441021203995      0.3707655138171598      0.5248509090894468      0.23003304
# wind            0.23421207762417        0.23799586337313303     6284.021524608135       0.3839434707921421      0.49983614615052757     0.24618532
# wind            0.2561858857193019      0.2539206153416057      5743.387681782246       0.3973216388615741      0.47482768968795636     0.26175496
# wind            0.264399110569406       0.2693550491474549      5567.076189041138       0.40987176896428107     0.4498263507056551      0.27729037
# wind            0.2932611837149452      0.2843729245903824      5018.845736205578       0.4235931221191804      0.42481557900985273     0.29132813
# wind            0.3014734147360394      0.29808004950431194     4880.996782183647       0.43570574271390594     0.39980646149119003     0.30508775
# wind            0.3226787631288391      0.3114881682437239      4559.86860960722        0.4490437516014844      0.3748031530386193      0.3179704
# wind            0.3276666101686571      0.32439207545198157     4490.600885212421       0.4619742325601564      0.34980184969341194     0.33111018
# wind            0.33629556596429966     0.33822675887109693     4377.415079057217       0.4770254248474828      0.32479974422038727     0.34578496
# wind            0.34763386278741576     0.3538908461203032      4234.564692080021       0.4942841727214611      0.2997859940516839      0.36271334
# wind            0.36383597802095796     0.3716479217542127      4046.4107327461243      0.5140239126478483      0.27477270606632326     0.38114834
# wind            0.3914233422164609      0.39133817659773934     3759.7579516768456      0.5361930468788666      0.24975683999188078     0.40220964
# wind            0.40505787865349346     0.4144489368826323      3632.901775300503       0.5592046690766995      0.22475070654483048     0.42804933
# wind            0.4269821654147667      0.4433730070181424      3445.886916100979       0.58717625463936        0.19974663823126937     0.4607638
# wind            0.48985714730906055     0.4785217081886018      3004.402008533478       0.6204812037833652      0.1747460231528223      0.49736395
# wind            0.544585602356014       0.5169482925131172      2702.729073405266       0.6494013380767737      0.14973868712830407     0.5374659
# wind            0.5901563084390451      0.556821505970346       2493.7329571843147      0.6754693064871307      0.12472898001105316     0.57657295
# wind            0.6364935303505448      0.5995065239670265      2312.2193413972855      0.7008763681682246      0.09972224823312642     0.62528664
# wind            0.6082755542928739      0.6596973983517972      2419.062203526497       0.7254364417658022      0.07471516956196277     0.7006147
# wind            0.740737723476823       0.7443616171108124      1987.15547144413        0.8032499922948545      0.0497124450252072      0.8021412
# wind            0.8783022830347351      0.8809399474576618      1655.1339844465256      0.8783022830347351      0.024701146119784818    1.0
# hail            0.0001688766746174388   0.00016986352016793734  3.8561814796536565e6    0.005964377746736281    1.0                     0.0032460222
# hail            0.005056197381197998    0.005178516528119907    128648.97746747732      0.050196406560687454    0.9749658801428607      0.007872191
# hail            0.010203195536666897    0.010107680353831559    63830.05483341217       0.06561629873525927     0.9499603796128998      0.012781623
# hail            0.015821904513234814    0.015104063119349472    41117.35542935133       0.07692473976495817     0.9249242828573117      0.017752057
# hail            0.022147232291703338    0.020072625025441538    29403.899218082428      0.08617302562498311     0.8999156974210145      0.022598635
# hail            0.0259968901054229      0.025117546173503653    25023.030090630054      0.09394418158905143     0.8748817358962153      0.02781728
# hail            0.028361003971031782    0.030727630999250587    22940.767663419247      0.10177102441715521     0.8498744539515698      0.0338735
# hail            0.03903456414525181     0.0364492693270628      16677.66209191084       0.11043881203367296     0.8248632417258195      0.039151985
# hail            0.04407472435410093     0.041678273992984925    14758.519374191761      0.11714353868107179     0.7998373420715237      0.044298634
# hail            0.048301107357003624    0.04688631602879335     13469.439606487751      0.1237652682075777      0.7748317242810914      0.049554877
# hail            0.05065636373808272     0.05232625004193206     12852.339295744896      0.13056950046420285     0.749821832702088       0.05516261
# hail            0.0556633464109231      0.05792051684806667     11684.663540303707      0.13809194963192725     0.7247941069203352      0.060721464
# hail            0.05912976810452062     0.06343117424131829     11013.629515111446      0.14580644641778448     0.6997911877132116      0.06621122
# hail            0.0691549418583564      0.06868568818197651     9411.968672692776       0.15419242739172814     0.6747565198966244      0.071343645
# hail            0.07944395373175758     0.07403235497497444     8186.373473584652       0.1618570588026457      0.6497352764976398      0.076881915
# hail            0.07978270298785045     0.07997968401473039     8156.490838050842       0.16886747343429626     0.6247342639669534      0.08326042
# hail            0.08127802117471436     0.08681865290468002     8007.773299753666       0.17711691901250792     0.599718297121917       0.090595216
# hail            0.09099243211896378     0.0943117155658535      7148.7051075696945      0.18670135253295675     0.5746981369593849      0.09825146
# hail            0.1027766890422804      0.1022485423320787      6336.298111617565       0.19608359017745713     0.5496925041402002      0.10645311
# hail            0.10849301818929195     0.1108954613760138      6003.002557218075       0.20496234734522562     0.524658206078123       0.11529455
# hail            0.11641808941729413     0.11975218121381716     5592.059327960014       0.21452088275585054     0.49962159669749634     0.12428135
# hail            0.13552766298329122     0.12829131705233046     4804.105116486549       0.22449661409928418     0.47459525141858105     0.13238806
# hail            0.14385289149081704     0.13659859608756195     4522.07204324007        0.23301274361094212     0.4495661276993548      0.14090589
# hail            0.14059523872883764     0.14558430165422917     4628.663004398346       0.2418416076767666      0.42455914654055826     0.15041065
# hail            0.1436911832119879      0.15565528705086537     4529.06968832016        0.25326107426321504     0.39954236879591853     0.16129908
# hail            0.16583231676961216     0.16676701021848717     3925.3305953741074      0.2668535240189481      0.3745248451907266      0.17264391
# hail            0.17784239646922867     0.17843488184597522     3659.9545714259148      0.27902339391714986     0.34950119838742805     0.18470751
# hail            0.1911353823274965      0.19114882022021606     3405.8285197615623      0.29182655913665395     0.3244795342177464      0.19793293
# hail            0.20293857844390475     0.20487018607790106     3206.1061900258064      0.30526552450751615     0.2994548237435567      0.21192935
# hail            0.22175135010463226     0.21931380917669682     2935.7103304862976      0.3199693217885283      0.2744428646009568      0.22705068
# hail            0.22733616323575315     0.23516663323541886     2861.3684262037277      0.3348503743338633      0.24941725485333374     0.24365705
# hail            0.25412515829614346     0.2524287657342456      2559.733422458172       0.353478247445716       0.22441106634150426     0.2615881
# hail            0.282239078511908       0.27067067977287246     2306.315605401993       0.37170211606268755     0.19940487238779536     0.28002745
# hail            0.3259429458422585      0.2894690275969313      1995.792701125145       0.38941456751889053     0.174381774361945       0.29961887
# hail            0.32736190178973607     0.311604282230668       1988.1018089056015      0.4025374534679948      0.14937473887142463     0.3244648
# hail            0.3657107322144948      0.3365850333047497      1778.5936949849129      0.4220361067609475      0.12435562351078296     0.35006618
# hail            0.37925476687540344     0.36606605704448336     1717.0783013105392      0.43905505024036223     0.09935103868529417     0.3846177
# hail            0.42076591864563806     0.40345849928064315     1546.3336944580078      0.4636831041597191      0.07431726381122974     0.42595303
# hail            0.44556774535546084     0.4594315569448972      1460.94160425663        0.48898433104529204     0.04930523220800086     0.50141275
# hail            0.5435692667964874      0.5995343111176749      1162.0256665349007      0.5435692667964874      0.024281516594142315    1.0
# sig_tornado     5.902142122649731e-5    3.5431760981227327e-6   466833.6241362095       0.0002498396427244399   1.0                     4.1808744e-6
# sig_tornado     7.450206362469907e-6    1.80720156545955e-5     3.752140585887313e6     0.00027271240774020145  0.974713962532957       0.00037249012
# sig_tornado     0.0013917241950795426   0.00043309325709115333  19689.806038320065      0.007259075521624042    0.9490598857808176      0.0005021928
# sig_tornado     0.002171955185050353    0.0005591377425178753   12555.034772992134      0.008200051524397876    0.923911877158343       0.0006225577
# sig_tornado     0.0013529429970616132   0.0007663748197507676   20733.379400491714      0.008886714069043082    0.8988866651737083      0.00094311364
# sig_tornado     0.0006822283418760961   0.0016354264285560391   41135.29234713316       0.010632260394285852    0.8731436994120624      0.0028064288
# sig_tornado     0.003622313391435235    0.0032176006134409527   7693.9399029016495      0.019097583977045317    0.8473891870899481      0.0036818348
# sig_tornado     0.0035773725272929013   0.004359712735147206    7728.471186101437       0.02202620320699366     0.8218125290936567      0.005193146
# sig_tornado     0.006185693902550271    0.005895838104469548    4501.127290725708       0.026356363129299572    0.7964398255794303      0.0066875396
# sig_tornado     0.007453415507030091    0.00752271259310325     3753.020353794098       0.029550256806693216    0.7708881899809065      0.008446714
# sig_tornado     0.01161616869061502     0.009160416598696302    2397.1697034835815      0.03291137124140301     0.7452170451184864      0.009940591
# sig_tornado     0.010080006518648392    0.010982602328220395    2776.7493512034416      0.03520297837668052     0.7196623516681505      0.012088997
# sig_tornado     0.011977752369270338    0.013081530842779122    2344.9661700725555      0.03878053186543864     0.6939757602835319      0.014129498
# sig_tornado     0.007406870819624168    0.016081079842055677    3805.0789388418198      0.04244439867139198     0.6681994413350031      0.018178908
# sig_tornado     0.01227962953541955     0.019720452224535067    2276.087203860283       0.05243142971050706     0.6423347424719342      0.02135899
# sig_tornado     0.025825594275038544    0.02234731821971619     1072.6076238751411      0.06068453867721677     0.6166849918509548      0.023452088
# sig_tornado     0.041816755602833676    0.024112758957461046    656.4869470000267       0.06442328369595611     0.591263549454294       0.024830054
# sig_tornado     0.044269576798377445    0.025582736601435667    621.0995225310326       0.06601153217283619     0.5660702287541947      0.026339987
# sig_tornado     0.03566010084510677     0.02739002211513693     776.3140092492104       0.06755959853043027     0.540836841618363       0.02849795
# sig_tornado     0.048287862552233635    0.02935670349340504     577.1021230816841       0.07067583524646331     0.5154312697253216      0.030273693
# sig_tornado     0.03589848640328322     0.031677724272314516    780.982861995697        0.07242898625486087     0.4898572121228853      0.033089906
# sig_tornado     0.03485963738191695     0.034908028197132       805.2758322358131       0.0767590884951274      0.46412799176123287     0.036785007
# sig_tornado     0.032697086350241794    0.039596660953809634    862.1222693324089       0.08259312075097514     0.43836617329456246     0.042832177
# sig_tornado     0.04689540258385372     0.04555807628589651     581.336351275444        0.09133401591366667     0.4124967407289791      0.048332512
# sig_tornado     0.04629962851242861     0.051643423111276635    591.3698193430901       0.09728656327922719     0.3874779258259107      0.05505985
# sig_tornado     0.057926084087261874    0.058418289145909526    471.77237832546234      0.10533016948277746     0.36235063606123535     0.062005945
# sig_tornado     0.055468180396352006    0.06592133985341887     493.3640748858452       0.11215507733239925     0.3372713213450941      0.07022435
# sig_tornado     0.0683621751365723      0.07401179255066474     400.46127313375473      0.12220278862906202     0.3121570589852979      0.077955574
# sig_tornado     0.0638384251681796      0.08247684098829895     427.8435660600662       0.13125071302057276     0.28703325586747513     0.08687711
# sig_tornado     0.08883925407400375     0.09064779654551341     306.92751175165176      0.14600253236142127     0.26196776838285946     0.0940938
# sig_tornado     0.06940828762612233     0.09916739963517254     393.63376247882843      0.15664736905593218     0.23694417311798718     0.1058584
# sig_tornado     0.115572127291456       0.11181991738515362     236.1888673901558       0.18401924300023736     0.2118708061757883      0.118976094
# sig_tornado     0.2888603963345528      0.12260373587280317     94.33913218975067       0.19989371570259826     0.18682001100670975     0.12656204
# sig_tornado     0.2213007866039423      0.13136534324283114     127.28506582975388      0.1908108660118203      0.16181144068942477     0.13572225
# sig_tornado     0.13951485478236705     0.1433854634368887      202.21086925268173      0.18594005139379075     0.1359609141068821      0.15170556
# sig_tornado     0.15330210916808792     0.16129448771417598     177.7861635684967       0.2017294033830379      0.11007080190355874     0.17286865
# sig_tornado     0.25568786852267045     0.1822663465400607      106.57849729061127      0.22238749536742        0.085058418707297       0.19134726
# sig_tornado     0.24489072460152103     0.20204597145470693     111.51940643787384      0.21094589505207959     0.06004985785315041     0.2137211
# sig_tornado     0.17924453289667158     0.24161105434056773     152.70696115493774      0.19189198774361219     0.0349869319252323      0.29151195
# sig_tornado     0.2339084785202554      0.3218716515358371      45.966580271720886      0.2339084785202554      0.009867273005631118    1.0
# sig_wind        4.677428732043673e-5    6.131121562578891e-5    3.758567874748051e6     0.0016044359095530234   1.0                     0.00066432636
# sig_wind        0.0012856383740159798   0.0009437035606012093   136614.40850830078      0.011315616535116091    0.974876650603298       0.001302207
# sig_wind        0.0023311928315697893   0.0015927032123458859   75382.36618733406       0.014254434481054958    0.9497772532096131      0.0019336545
# sig_wind        0.0025445217970113303   0.0023632622110747598   68945.2407822609        0.016553908349320835    0.9246644076298126      0.0028589459
# sig_wind        0.0024304400979359223   0.0035476135297549775   72323.12841272354       0.019554210768998284    0.8995941711880094      0.0043693255
# sig_wind        0.005535576733607262    0.004838439611021384    31613.78411334753       0.024515836826400995    0.8744747234433752      0.005347013
# sig_wind        0.0072799444611335      0.005805189829202848    24102.833062410355      0.02726842002535047     0.8494662420692087      0.006298599
# sig_wind        0.00939996582875876     0.006743361481179613    18651.429261505604      0.029753251183134474    0.8243910585246093      0.0072145984
# sig_wind        0.008682811853328263    0.007804909552620404    20177.919882535934      0.03191956787919423     0.7993365231110267      0.008447081
# sig_wind        0.008389489181365858    0.009275341794723092    20954.245629310608      0.03494338799068617     0.7742993815130925      0.010208897
# sig_wind        0.009937405900965061    0.011240814695407629    17636.428233087063      0.039092520951684084    0.7491773063555576      0.0124175055
# sig_wind        0.011643718060430273    0.013674016405018672    15078.55777746439       0.04350740136021686     0.7241316956161001      0.0150820315
# sig_wind        0.0141406741590453      0.016530592811864014    12421.587271809578      0.048246145264654955    0.6990417657622722      0.01812666
# sig_wind        0.018103968854281827    0.019378844910212967    9703.675610244274       0.053007913419179264    0.6739405260168931      0.020700391
# sig_wind        0.021091205041195316    0.022113903785826054    8330.048471450806       0.05728092487603791     0.6488356532116217      0.023667058
# sig_wind        0.0263128246501436      0.025122848564277406    6656.356948912144       0.061530815676574965    0.623728534037783       0.026668768
# sig_wind        0.027145290823495975    0.028325135695889108    6453.271069705486       0.06517785675940688     0.5986990419719189      0.030047383
# sig_wind        0.03560742570270696     0.03150495066115904     4939.096835196018       0.0694223315919152      0.5736654961514128      0.032998197
# sig_wind        0.04314404419451607     0.03429405967797846     4055.661785185337       0.07258039065407557     0.5485329736841431      0.03562565
# sig_wind        0.03431225400971848     0.037401674749507126    5118.967091262341       0.07502529337723451     0.5235277614804097      0.039275263
# sig_wind        0.0353982854321186      0.041193106302336524    4955.383614897728       0.07979320040886531     0.4984274247594198      0.043196816
# sig_wind        0.03767735976355986     0.045408316806579606    4653.02173024416        0.08546967709536735     0.47336012992705984     0.04780317
# sig_wind        0.040003627107200075    0.050561039579102865    4373.937284708023       0.09199060132030486     0.4483069126704845      0.053615205
# sig_wind        0.05787281175702347     0.05634171863324459     3033.246574282646       0.09963944359002327     0.4233023142688235      0.059325617
# sig_wind        0.06442154041825512     0.06251375076789324     2721.866678714752       0.10438519421810274     0.39821638111733065     0.065936975
# sig_wind        0.08036038340146115     0.06928572692815786     2180.697160243988       0.10892256887509685     0.3731584093286595      0.072886504
# sig_wind        0.09117088416602971     0.07642407464419408     1920.096725165844       0.11178067371568917     0.3481154689318769      0.08020817
# sig_wind        0.10423021262728034     0.08351283875681005     1686.583735704422       0.11377200967567089     0.3230989293750773      0.08685597
# sig_wind        0.09508322595618789     0.09045478355927325     1844.460866689682       0.11465692824582911     0.29797720228090646     0.09406444
# sig_wind        0.09284098255839757     0.09745778980868677     1891.769082903862       0.11686621531512766     0.272914879729996       0.10046095
# sig_wind        0.08448889945772363     0.1032531032072217      2076.341696023941       0.1200116294061339      0.24781591470898154     0.105739795
# sig_wind        0.0666056407488826      0.10841888785303871     2633.647204518318       0.12597262303628542     0.22274637038201708     0.11086762
# sig_wind        0.09149090955122577     0.11261049539177773     1914.8697921037674      0.1420256848474179      0.19767855023913677     0.114373446
# sig_wind        0.10895047692405334     0.11603272388132761     1607.8294390439987      0.1543924401609871      0.17264253801793591     0.117656514
# sig_wind        0.13670095533096874     0.119244882927943       1285.8106350898743      0.16614459568463244     0.14760929740768022     0.12082111
# sig_wind        0.14991595759656484     0.12248170691891941     1169.1597928404808      0.17382207262739086     0.12249063408541655     0.124390624
# sig_wind        0.17072976466325301     0.12642990010478924     1027.9549351930618      0.18125162892668142     0.09744282776058877     0.12870768
# sig_wind        0.22189588669913718     0.13056017829124467     790.6413577795029       0.18520765769286987     0.07236261136437643     0.13262717
# sig_wind        0.2642592684935109      0.13526667630250508     664.4515901207924       0.17028174446430036     0.04729131782799529     0.13866127
# sig_wind        0.12145817722588818     0.1630329827303914      1278.962575018406       0.12145817722588818     0.022198956351700102    1.0
# sig_hail        2.216464571310139e-5    1.2931126533864667e-5   4.091973664226413e6     0.0008290638627045212   1.0                     0.0005326919
# sig_hail        0.001014425064379637    0.001150422202914929    89696.39141362906       0.013082318320336534    0.9749172024122722      0.0021832057
# sig_hail        0.0028037775206557897   0.002925954442177407    32422.853137373924      0.019103685501244534    0.9497533405630142      0.0038285695
# sig_hail        0.004377732157536748    0.004668177674086281    20741.131660819054      0.022690441613442983    0.9246126543030243      0.005650947
# sig_hail        0.005321907007939204    0.006682531074733373    17158.144751012325      0.025690564807009868    0.8995016556507074      0.007823558
# sig_hail        0.0078036737752282875   0.008782842528213207    11626.283546566963      0.028883830914220313    0.8742482497776712      0.009790572
# sig_hail        0.01000335073038368     0.0107157807510814      9041.597701311111       0.03138931062433119     0.8491569958184833      0.011692269
# sig_hail        0.012382953266616467    0.012564693078434596    7317.498324275017       0.033567375357944365    0.8241435787031112      0.013479922
# sig_hail        0.012088132650290172    0.01455536321174635     7482.74041980505        0.035470356933711485    0.7990842625486805      0.015723065
# sig_hail        0.016288559628618447    0.01668620386548764     5571.905107557774       0.03783544857034159     0.7740691623680455      0.01767531
# sig_hail        0.01837637880437862     0.018633764209747192    4928.403512835503       0.03959053243453405     0.7489694500758021      0.01961534
# sig_hail        0.01623106650772815     0.020718049992409737    5626.115225851536       0.04123761576697365     0.7239228637518518      0.02181292
# sig_hail        0.0224403043472901      0.02258980182077352     4055.426232814789       0.04366955310427076     0.6986684070736354      0.023345487
# sig_hail        0.02608662414830645     0.023978018634780697    3498.112974882126       0.04526994194904406     0.6735004712052034      0.024604267
# sig_hail        0.023112545122351495    0.02526914839998517     3944.66820448637        0.046604115982980795    0.6482636842149223      0.02592919
# sig_hail        0.027074576959455914    0.02645794664879133     3360.9723170995712      0.04860327670503485     0.6230497508155662      0.027018148
# sig_hail        0.031094427109227087    0.02753040876232571     2912.1436977386475      0.050286330754657965    0.5978840537372946      0.02804718
# sig_hail        0.029713264718009582    0.028621266563468615    3054.766616523266       0.051680796634013225    0.5728415521141466      0.029203814
# sig_hail        0.034467399440368744    0.02973957344673712     2640.0087084770203      0.053493249153874274    0.5477394115152611      0.030288022
# sig_hail        0.03626886641420836     0.030813746706969174    2494.340350151062       0.05495402335743124     0.5225744525572831      0.031374626
# sig_hail        0.0380107712936996      0.031959134675532797    2390.603925704956       0.056415499966371054    0.49755533176561534     0.0325462
# sig_hail        0.040640492242555545    0.033153365564469434    2227.264269053936       0.057906979699387       0.47242509037427766     0.033786863
# sig_hail        0.032252792398381754    0.03467513077395327     2808.842256963253       0.05931707854687369     0.44739208115811113     0.035615657
# sig_hail        0.03295476968700482     0.03666531713333499     2744.015417635441       0.06242450710636616     0.42233808415784263     0.03776552
# sig_hail        0.03937312693416367     0.038838913534414266    2304.6278325915337      0.06614764042749215     0.39732961098795605     0.04000773
# sig_hail        0.03661178207653541     0.04147013037527796     2469.2615703344345      0.06932585292208306     0.3722348502568036      0.04309425
# sig_hail        0.03518264770351698     0.04524570777840991     2584.6404529213905      0.07409278345106053     0.3472331055731339      0.047773357
# sig_hail        0.03724704689095137     0.050721502970150434    2449.8306228518486      0.08109559511158611     0.32208466809398917     0.054074824
# sig_hail        0.05763793324012543     0.05649777926024        1571.3345357179642      0.0901140043547899      0.2968492645229239      0.058987055
# sig_hail        0.06201337136962035     0.06154126325603639     1472.258781015873       0.09504927012407281     0.2718020517412568      0.064124
# sig_hail        0.06876230126591695     0.06642680644917877     1315.4910994768143      0.1005340100305079      0.24655260621377048     0.06882672
# sig_hail        0.10253107670201983     0.07033535248133924     888.348667383194        0.10606815684837491     0.22153644502436287     0.07185836
# sig_hail        0.08270267579497688     0.073976118833599       1100.6037876605988      0.10653967471265656     0.1963468420169747      0.07619421
# sig_hail        0.10100294919155503     0.0783084246738722      899.6180948615074       0.11125539807033594     0.17117396152721334     0.08070509
# sig_hail        0.12792598389062493     0.08269425103159381     712.0610821247101       0.11323307228050387     0.14604499736520057     0.08513206
# sig_hail        0.12285705365500178     0.08782285614222743     735.9584183096886       0.11058550293291629     0.12085324575158445     0.090688035
# sig_hail        0.13507382307205562     0.09376680093788502     674.7859163880348       0.1077769720997611      0.09584773538185287     0.09708124
# sig_hail        0.13936377164421354     0.10090547998200972     650.9123640656471       0.10052775500159407     0.0706408288125252      0.10543228
# sig_hail        0.08042217363580478     0.11803805561497323     1125.6676247119904      0.08715261307872944     0.045553480356638275    0.13493054
# sig_hail        0.09706501191706869     0.16904671204293673     764.3193039894104       0.09706501191706869     0.02051728496571095     1.0



# Calibrate to SPC

target_success_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.050728736928645046),
    (0.05, 0.11486038942521988),
    (0.1,  0.22690348338037045),
    (0.15, 0.3046782911820712),
    (0.3,  0.31560627664076046),
  ],
  "wind" => [
    (0.05, 0.13461838497658143),
    (0.15, 0.25485816629457203),
    (0.3,  0.43476976026058434),
    (0.45, 0.6185653041815565),
  ],
  "hail" => [
    (0.05, 0.08021216469509043),
    (0.15, 0.16136397567347519),
    (0.3,  0.3011316257061734),
    (0.45, 0.5444239242570899),
  ],
  "sig_tornado" => [(0.1, 0.08721705058958064)],
  "sig_wind"    => [(0.1, 0.1163905700911105)],
  "sig_hail"    => [(0.1, 0.08379410166467652)],
)

target_PODs = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.6615090274606944),
    (0.05, 0.3915418238912139),
    (0.1,  0.13042915979366457),
    (0.15, 0.027773066438696918),
    (0.3,  0.008323827309277956),
  ],
  "wind" => [
    (0.05, 0.7129517911854782),
    (0.15, 0.38983512353226496),
    (0.3,  0.10680248115276206),
    (0.45, 0.015289658660478881),
  ],
  "hail" => [
    (0.05, 0.7457921051915138),
    (0.15, 0.41143213280714935),
    (0.3,  0.0778527553093847),
    (0.45, 0.009683467798086529),
  ],
  "sig_tornado" => [(0.1, 0.17764733377105618)],
  "sig_wind"    => [(0.1, 0.08622702385365015)],
  "sig_hail"    => [(0.1, 0.2454541775633861)],
)





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

  thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_succes_ratio = thresholds_to_match_success_ratio[i]
    threshold_to_match_POD = thresholds_to_match_POD[i]
    mean_threshold = (threshold_to_match_succes_ratio + threshold_to_match_POD) * 0.5f0
    sr  = Metrics.success_ratio(ŷ, y, weights, mean_threshold)
    pod = Metrics.probability_of_detection(ŷ, y, weights, mean_threshold)
    println("$event_name\t$nominal_prob\t$threshold_to_match_succes_ratio\t$threshold_to_match_POD\t$mean_threshold\t$sr\t$pod")
    push!(thresholds, (Float32(nominal_prob), Float32(mean_threshold)))
  end
  thresholds
end
println("event_name\tnominal_prob\tthreshold_to_match_succes_ratio\tthreshold_to_match_POD\tmean_threshold\tsuccess_ratio\tPOD")
calibrations = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations[event_name] = spc_calibrate(prediction_i, X, Ys, weights)
end

# event_name   nominal_prob  threshold_to_match_succes_ratio  threshold_to_match_POD  mean_threshold  success_ratio         POD
# tornado      0.02           0.013106078                     0.022882462             0.01799427      0.062248498633836996  0.7237450145071389
# tornado      0.05           0.05241271                      0.09069252              0.07155262      0.13041010693869282   0.4595979609764453
# tornado      0.1            0.16216                         0.1885128               0.17533639      0.2602604408764249    0.14314764321503318
# tornado      0.15           0.19976498                      0.25847054              0.22911775      0.3278878657777642    0.06873094450782086
# tornado      0.3            0.20561947                      0.30122948              0.25342447      0.28779861751854896   0.033010873389855366
# wind         0.05           0.021777954                     0.10754967              0.06466381      0.21349353908937124   0.8077435939062366
# wind         0.15           0.09295714                      0.2965603               0.19475871      0.35260570410877334   0.5574172101278139
# wind         0.3            0.29035583                      0.56562996              0.42799288      0.587179141844545     0.19983921956088188
# wind         0.45           0.45901677                      0.8673687               0.66319275      0.7595734431193403    0.06063552925564667
# hail         0.05           0.014443969                     0.050340652             0.03239231      0.10834954044348313   0.8305503616488299
# hail         0.15           0.070988156                     0.14591789              0.10845302      0.2070218904495598    0.5193486074366591
# hail         0.3            0.19329038                      0.3788433               0.28606683      0.3932871977659952    0.16613831846637464
# hail         0.45           0.50763494                      0.62512016              0.5663775       0.5888789612140184    0.013460490246840578
# sig_tornado  0.1            0.040612184                     0.121248245             0.08093022      0.13623988675028018   0.2807708016672483
# sig_wind     0.1            0.09281066                      0.12635994              0.1095853       0.13657851645778404   0.20438348540838872
# sig_hail     0.1            0.049650267                     0.064489365             0.057069816     0.09399092050668835   0.28431260828958804



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

