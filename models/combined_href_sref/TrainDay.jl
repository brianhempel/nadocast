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

aug29 = validation_forecasts_0z[85]; Forecasts.time_title(aug29) # "2020-08-29 00Z +35"
aug29_data = Forecasts.data(aug29);
for i in 1:size(aug29_data,2)
  PlotMap.plot_debug_map("aug29_0z_day_accs_$i", aug29.grid, aug29_data[:,i]);
end
for (event_name, labeler) in event_name_to_day_labeler
  aug29_labels = event_name_to_day_labeler[event_name](aug29);
  PlotMap.plot_debug_map("aug29_0z_day_$event_name", aug29.grid, aug29_labels);
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

# tornado (8326.0)     feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.12862590219455652
# tornado (8326.0)     feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.11601147163983896
# wind (63336.0)       feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.4070694067680044
# wind (63336.0)       feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.38303104871692784
# hail (28152.0)       feature 5 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.23917144549298752
# hail (28152.0)       feature 6 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.222086339604639
# sig_tornado (1138.0) feature 7 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.08828535008120504
# sig_tornado (1138.0) feature 8 highest hourly STORPROB:calculated:day            fcst:: AU-PR-curve: 0.07309166844703058
# sig_wind (7555.0)    feature 9 independent events total SWINDPRO:calculated:day  fcst:: AU-PR-curve: 0.0807134420010289 (only exception. oh well)
# sig_wind (7555.0)    feature 10 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.08173225765702283
# sig_hail (3887.0)    feature 11 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.07042914553475686
# sig_hail (3887.0)    feature 12 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.06251880831100963



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

# event_name      mean_y                  mean_ŷ                  Σweight                 bin_max
# tornado         0.0003111797725399882   0.00037156756246816227  4.1902059462661743e6    0.009299877
# tornado         0.01336803361195303     0.0165849822594149      97491.02201145887       0.028508391
# tornado         0.03197687600179042     0.04178745617366761     40772.718268334866      0.062506445
# tornado         0.08114330883714369     0.08031162107209787     16065.28214508295       0.10407963
# tornado         0.12304151372522987     0.14032956374879546     10594.889869987965      0.1940539
# tornado         0.20610796208592153     0.29823922815669723     6307.755592346191       1.0
# wind            0.002386163947536112    0.0030770378719691366   4.1106334608077407e6    0.06972501
# wind            0.0826641672950054      0.1070416523157663      118662.17303037643      0.1592437
# wind            0.17604040935462295     0.20288025245353855     55720.645763874054      0.2574959
# wind            0.28163299983173845     0.3065729209800772      34828.944473445415      0.36441022
# wind            0.37677208103931625     0.43005849551901676     26033.56202661991       0.5204422
# wind            0.6303368404724048      0.660146007614328       15558.828051328659      1.0
# hail            0.0010425533104730588   0.0013505200136412255   4.1594016295077205e6    0.03756313
# hail            0.04783366417800802     0.056172324711248015    90641.15363228321       0.08204184
# hail            0.08542346627457732     0.10852262181613116     50759.941532194614      0.14231205
# hail            0.1476410416743679      0.1747681465532697      29366.719541311264      0.21637398
# hail            0.21833825839794685     0.2680870433729829      19859.435877144337      0.34320852
# hail            0.37982015571839106     0.46089000842522476     11408.734062731266      1.0
# sig_tornado     4.2115539978953755e-5   5.722979435784368e-5    4.316894313991845e6     0.0041370555
# sig_tornado     0.006613373354782284    0.008471009546958133    27461.040125072002      0.016406734
# sig_tornado     0.02507474406279913     0.021671332773658535    7266.2019882798195      0.02879601
# sig_tornado     0.029288389191126335    0.04401029716577926     6203.844566822052       0.07119626
# sig_tornado     0.07091782663279547     0.10154532444080051     2568.561889767647       0.14319168
# sig_tornado     0.1726495455941821      0.18624242572918537     1043.6515915989876      1.0
# sig_wind        0.0002814290183971363   0.0002897967450470186   4.1456266263979673e6    0.007097877
# sig_wind        0.008886846722621982    0.013459602818289017    131291.1367612481       0.02519015
# sig_wind        0.02921449057288732     0.03461229755450752     39930.60130119324       0.047340583
# sig_wind        0.04936999540633218     0.059497230867807054    23631.09668815136       0.07464776
# sig_wind        0.10023785441624197     0.08556494405215118     11644.13838160038       0.099326596
# sig_wind        0.12494939871098372     0.13168433427206228     9314.014623224735       1.0
# sig_hail        0.00014124886710636     0.0002151746707838891   4.270415205631971e6     0.01275157
# sig_hail        0.017906739660818075    0.017028496374880114    33678.47697079182       0.022190364
# sig_hail        0.02903275497034974     0.026867635193698704    20777.733484745026      0.03249627
# sig_hail        0.02973914735748192     0.04155306457861286     20276.20506322384       0.05409533
# sig_hail        0.05589286136215945     0.06866356905095351     10796.949942946434      0.09080737
# sig_hail        0.10921868953742886     0.13182396438543684     5493.043059706688       1.0

println(event_to_day_bins)
# Dict{String, Vector{Float32}}("sig_hail" => [0.01275157, 0.022190364, 0.03249627, 0.05409533, 0.09080737, 1.0], "hail" => [0.03756313, 0.08204184, 0.14231205, 0.21637398, 0.34320852, 1.0], "tornado" => [0.009299877, 0.028508391, 0.062506445, 0.10407963, 0.1940539, 1.0], "sig_tornado" => [0.0041370555, 0.016406734, 0.02879601, 0.07119626, 0.14319168, 1.0], "sig_wind" => [0.007097877, 0.02519015, 0.047340583, 0.07464776, 0.099326596, 1.0], "wind" => [0.06972501, 0.1592437, 0.2574959, 0.36441022, 0.5204422, 1.0])


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
# tornado     1-2  -1.0             0.028508391      4688491  2813.0    4.287697e6  0.00074021827     0.00015941972     0.00060805853 0.0038716984       0.004260177        0.014668874513289451 0.012710065607328827 0.0006080585    0.0038571511     0.014433284956395877 Float32[0.7984841, 0.19232666, 0.06514242]
# tornado     2-3  0.009299877      0.062506445      148129   2791.0    138263.75   0.024016963       0.005688873       0.01885561    0.0913576          0.10183169         0.03299388872092238  0.028745575986792962 0.018855613     0.09073429       0.03263049822087755  Float32[1.07512, -0.0651204, -0.31374842]
# tornado     3-4  0.028508391      0.10407963       60417    2778.0    56838.0     0.052676328       0.012986715       0.04587379    0.18069385         0.21069968         0.08632303491447675  0.06335024950076024  0.04587379      0.17956096       0.08703334662278019  Float32[1.5962993, -0.2032799, 0.6382291]
# tornado     4-5  0.062506445      0.1940539        28027    2765.0    26660.172   0.10416306        0.026533954       0.09779387    0.3187074          0.38772678         0.11204681730225982  0.10252364584219112  0.09779388      0.31750065       0.11547449106884143  Float32[0.83288527, -0.19397551, -1.1356705]
# tornado     5-6  0.10407963       1.0              17612    2735.0    16902.645   0.19925855        0.049278524       0.1540404     0.42977545         0.5085254          0.23101723157723933  0.2132159421654902   0.1540404       0.41955665       0.23174164746088968  Float32[0.60632825, 0.068164125, -0.642197]
# wind        1-2  -1.0             0.1592437        4624213  21120.0   4.2292955e6 0.0059939935      0.0013979143      0.00463854    0.019859761        0.022192718        0.08398383377815363  0.0773294387016005   0.00463854      0.019658804      0.08309355326500503  Float32[0.700852, 0.29223275, 0.16097507]
# wind        2-3  0.06972501       0.2574959        187599   21152.0   174382.81   0.13766502        0.033623766       0.11250074    0.34245422         0.40741473         0.17199638176562018  0.15266377078805488  0.11250074      0.33934718       0.1717781805890173   Float32[1.1891124, -0.0350956, -0.022556616]
# wind        3-4  0.1592437        0.36441022       97590    21176.0   90549.586   0.24276456        0.061562862       0.21665551    0.51525795         0.6592687          0.2780757495183039   0.2483229530000953   0.21665548      0.51318884       0.2775820970499318   Float32[1.1438317, -0.10413974, -0.27913937]
# wind        4-5  0.2574959        0.5204422        65610    21155.0   60862.508   0.35939312        0.09664003        0.32232815    0.6244063          0.83303183         0.38338104888139785  0.3492622647624472   0.32232815      0.6210337        0.3825677561733355   Float32[0.8871091, -0.04730344, -0.34016654]
# wind        5-6  0.36441022       1.0              44717    21040.0   41592.39    0.5161293         0.15908156        0.47162524    0.6532767          0.9501514          0.6462144735334092   0.616562904969527    0.47162524      0.64876914       0.6470169330433779   Float32[0.9359743, 0.11857994, 0.024626805]
# hail        1-2  -1.0             0.08204184       4645964  9402.0    4.250043e6  0.002519711       0.0005503036      0.0020404723  0.009882573        0.011050798        0.045548393679595706 0.045292429160235556 0.0020404723    0.009800431      0.04666562546694774  Float32[0.67091036, 0.4049075, 0.702673]
# hail        2-3  0.03756313       0.14231205       152964   9411.0    141401.1    0.07496495        0.016594393       0.061327595   0.22819261         0.26585725         0.08679103501763114  0.08594474863536322  0.061327588     0.22665747       0.0868966739101486   Float32[0.78364456, 0.113389224, -0.28517193]
# hail        3-4  0.08204184       0.21637398       86732    9385.0    80126.66    0.13280186        0.030097123       0.10822645    0.34007683         0.40740603         0.1460160816754119   0.12881572778103342  0.108226456     0.3371733        0.1501395223153059   Float32[1.185529, -0.12618165, -0.33549955]
# hail        4-5  0.14231205       0.34320852       53300    9360.0    49226.156   0.21241604        0.05041142        0.1761626     0.46405438         0.5795195          0.22276651164135017  0.18693559265030946  0.17616262      0.4568764        0.24201254652936796  Float32[1.5202497, -0.6613805, -1.5395572]
# hail        5-6  0.21637398       1.0              33824    9365.0    31268.17    0.33843455        0.09418756        0.2772577     0.5790014          0.7299978          0.4000110581459369   0.3710505651110603   0.27725774      0.5657868        0.40757679024573523  Float32[1.5235753, -0.6790004, -1.5636766]
# sig_tornado 1-2  -1.0             0.016406734      4748798  389.0     4.3443555e6 0.000110414       2.5919026e-5      8.365302e-5   0.0006254932       0.00067050644      0.007955390680067736 0.00541937554620857  8.3653016e-5    0.0006217336     0.006451588581857264 Float32[1.1255901, -0.09031829, -0.2014342]
# sig_tornado 2-3  0.0041370555     0.02879601       36595    379.0     34727.242   0.011232997       0.0030230873      0.0104761645  0.055572875        0.06100611         0.02284651265204892  0.025084199340085704 0.0104761645    0.05511161       0.025753734363094312 Float32[0.7604227, 0.56255525, 2.073743]
# sig_tornado 3-4  0.016406734      0.07119626       14005    377.0     13470.047   0.0319599         0.009032625       0.027015405   0.124497265        0.13128208         0.03822021979634261  0.0893163159310164   0.02701541      0.11823708       0.11004389624102595  Float32[-1.00439, 1.5047965, 0.034532372]
# sig_tornado 4-5  0.02879601       0.14319168       9080     374.0     8772.406    0.060856566       0.01616405        0.041477494   0.16963358         0.17936319         0.08255115946475505  0.07033400045795353  0.0414775       0.16401978       0.0729372701853981   Float32[0.6268594, 0.6715986, 1.3238566]
# sig_tornado 5-6  0.07119626       1.0              3717     372.0     3612.2134   0.12601627        0.02895928        0.10031047    0.3145655          0.38247845         0.18066816939485536  0.12113902781588674  0.10031048      0.3092849        0.17739075621575026  Float32[1.653894, -0.29033047, -0.088192254]
# sig_wind    1-2  -1.0             0.02519015       4675467  2527.0    4.276918e6  0.0006940782      0.00014988276     0.0005455947  0.0035863968       0.0038595467       0.009579995947585912 0.008388221023522085 0.00054559484   0.0035339529     0.008381953909720435 Float32[-0.034680896, 0.88001555, 0.32633656]
# sig_wind    2-3  0.007097877      0.047340583      184406   2517.0    171221.75   0.01839262        0.004233513       0.013627454   0.07016324         0.07680257         0.028780942756754656 0.024076352006618597 0.013627454     0.06942523       0.028159197982352023 Float32[1.0652515, 0.0716145, 0.32573313]
# sig_wind    3-4  0.02519015       0.07464776       68440    2517.0    63561.703   0.043864064       0.01039809        0.03670794    0.15587577         0.17554973         0.05505735921863313  0.06238642447716812  0.036707934     0.15469983       0.06360936746165409  Float32[0.5629955, 0.5333647, 0.90708613]
# sig_wind    4-5  0.047340583      0.099326596      37946    2516.0    35275.234   0.068102024       0.015598245       0.06616116    0.23976162         0.28418005         0.09034861488657818  0.11156281829073866  0.06616113      0.23633926       0.10855596593942987  Float32[0.8774715, 1.0666903, 4.0418673]
# sig_wind    5-6  0.07464776       1.0              22613    2511.0    20958.152   0.10606087        0.023367401       0.11121991    0.3492696          0.44185692         0.13261681245534132  0.137665891432449    0.11121991      0.34765676       0.13614113422935148  Float32[0.30341336, 0.22051479, -0.58996856]
# sig_hail    1-2  -1.0             0.022190364      4704680  1299.0    4.304094e6  0.00034673477     8.252288e-5       0.00028025947 0.0016795164       0.0018490457       0.017191866105179227 0.013107731122682157 0.00028025944   0.0016593802     0.020693157887292746 Float32[1.8899088, -0.68954426, -0.28587562]
# sig_hail    2-3  0.01275157       0.03249627       58924    1299.0    54456.21    0.020782614       0.0046694977      0.022151863   0.105312295        0.1253043          0.030061315839576618 0.02242345696357653  0.022151878     0.10407473       0.03649710485775409  Float32[2.1869068, -1.1520596, -1.635874]
# sig_hail    3-4  0.022190364      0.05409533       44389    1306.0    41053.938   0.034120653       0.007600716       0.029381638   0.1329779          0.15331925         0.03444730059053422  0.02971774147018954  0.02938164      0.13186601       0.03499653179325579  Float32[1.1694325, -0.6678911, -2.8746517]
# sig_hail    4-5  0.03249627       0.09080737       33550    1302.0    31073.154   0.05097312        0.011861129       0.038826745   0.16233699         0.18264328         0.06723652959440497  0.05119513542367985  0.038826738     0.15975034       0.07163487390232491  Float32[2.2023127, -0.62016976, 0.41733724]
# sig_hail    5-6  0.05409533       1.0              17451    1282.0    16289.993   0.08996148        0.022639653       0.07387452    0.2600398          0.2990931          0.1268056570917822   0.11137833747845276  0.0738745       0.258015         0.12917071261268284  Float32[1.2492256, -0.3587903, -1.0017315]


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


print("event_to_day_bins_logistic_coeffs = ")
println(event_to_day_bins_logistic_coeffs)
# Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.8899088, -0.68954426, -0.28587562], [2.1869068, -1.1520596, -1.635874], [1.1694325, -0.6678911, -2.8746517], [2.2023127, -0.62016976, 0.41733724], [1.2492256, -0.3587903, -1.0017315]], "hail" => [[0.67091036, 0.4049075, 0.702673], [0.78364456, 0.113389224, -0.28517193], [1.185529, -0.12618165, -0.33549955], [1.5202497, -0.6613805, -1.5395572], [1.5235753, -0.6790004, -1.5636766]], "tornado" => [[0.7984841, 0.19232666, 0.06514242], [1.07512, -0.0651204, -0.31374842], [1.5962993, -0.2032799, 0.6382291], [0.83288527, -0.19397551, -1.1356705], [0.60632825, 0.068164125, -0.642197]], "sig_tornado" => [[1.1255901, -0.09031829, -0.2014342], [0.7604227, 0.56255525, 2.073743], [-1.00439, 1.5047965, 0.034532372], [0.6268594, 0.6715986, 1.3238566], [1.653894, -0.29033047, -0.088192254]], "sig_wind" => [[-0.034680896, 0.88001555, 0.32633656], [1.0652515, 0.0716145, 0.32573313], [0.5629955, 0.5333647, 0.90708613], [0.8774715, 1.0666903, 4.0418673], [0.30341336, 0.22051479, -0.58996856]], "wind" => [[0.700852, 0.29223275, 0.16097507], [1.1891124, -0.0350956, -0.022556616], [1.1438317, -0.10413974, -0.27913937], [0.8871091, -0.04730344, -0.34016654], [0.9359743, 0.11857994, 0.024626805]])




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
    save_dir = "day_validation_forecasts_0z_spc_calibrated",
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

# tornado (8326.0)     feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.12910289118631163
# wind (63336.0)       feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.4076166080130104
# hail (28152.0)       feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.24281549137715708
# sig_tornado (1138.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.09322716799320344
# sig_wind (7555.0)    feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.08475474579815966
# sig_hail (3887.0)    feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.0727762015599357

# Yep, that's unchanged!


target_success_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.050728736928645046),
    (0.05, 0.11486038942521988),
    (0.1,  0.22690348338037045),
    (0.15, 0.3046782911820712),
    # (0.3,  0.31560627664076046), # lol, too close! this is probably a consequence of lack of data, so let's target 40% SR
    (0.3,  0.4),
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


function spc_calibrate(prediction_i, X, Ys, weights)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  # println("nominal_prob\tthreshold\tsuccess_ratio")

  thresholds_to_match_success_ratio =
    map(target_success_ratios[event_name]) do (nominal_prob, target_success_ratio)
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
      # println("$nominal_prob\t$threshold\t$(success_ratio(ŷ, y, weights, threshold))")
      threshold
    end

  # println("nominal_prob\tthreshold\tPOD")

  thresholds_to_match_POD =
    map(target_PODs[event_name]) do (nominal_prob, target_POD)
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
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_PODs[event_name])
    nominal_prob, _ = target_PODs[event_name][i]
    threshold_to_match_succes_ratio = thresholds_to_match_success_ratio[i]
    threshold_to_match_POD = thresholds_to_match_POD[i]
    mean_threshold = (threshold_to_match_succes_ratio + threshold_to_match_POD) * 0.5f0
    sr  = success_ratio(ŷ, y, weights, mean_threshold)
    pod = probability_of_detection(ŷ, y, weights, mean_threshold)
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

# event_name      nominal_prob    threshold_to_match_succes_ratio threshold_to_match_POD  mean_threshold  success_ratio           POD
# tornado         0.02            0.0145724565                    0.022738352             0.018655404     0.05927062614766337     0.7357301796004513
# tornado         0.05            0.03927873                      0.059221342             0.049250036     0.12933375779388187     0.4633115774305384
# tornado         0.1             0.093651995                     0.10508071              0.09936635      0.2577960043775589      0.14462476933577742
# tornado         0.15            0.2715841                       0.13205521              0.20181966      0.2348807750696644      0.006138932381166391    # something is wrong here
# tornado         0.3             0.37645632                      0.1485431               0.26249972      0.29040593698579        0.005512840471253729    # something is wrong here
# wind            0.05            0.01683949                      0.08296512              0.049902305     0.21327360357599012     0.8079320037289051
# wind            0.15            0.07174833                      0.21547179              0.14361006      0.3458116242820063      0.5714318508074652
# wind            0.3             0.21148114                      0.38777882              0.29963         0.586480634971858       0.20031806041584502
# wind            0.45            0.31978613                      0.78341365              0.55159986      0.8352297195856766      0.042701978011109115
# hail            0.05            0.022294775                     0.07359822              0.047946498     0.10646251427884203     0.8360045859843113
# hail            0.15            0.10074367                      0.18163772              0.1411907       0.2002707524909718      0.5396125414257729
# hail            0.3             0.22166397                      0.34964204              0.285653        0.3796638051091829      0.18973000759737998
# hail            0.45            0.41856432                      0.52450764              0.47153598      0.6026694365021374      0.01205135608020926
# sig_tornado     0.1             0.050181612                     0.13948123              0.09483142      0.13142122243009974     0.29421793546267744
# sig_wind        0.1             0.08469264                      0.116955206             0.100823924     0.1400181395832151      0.20048482143175006
# sig_hail        0.1             0.087000266                     0.10708104              0.09704065      0.09149244333150056     0.2904158717211753

# looks right enough























# blurrrrr

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

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_with_blurs_and_forecast_hour(); just_hours_near_storm_events = false);

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

