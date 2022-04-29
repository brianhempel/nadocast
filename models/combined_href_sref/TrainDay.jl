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

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_count = 10
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

      println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(day_validation_forecasts_0z, X, Ys, weights)

# event_name      mean_y                  mean_ŷ                  Σweight                 bin_max
# tornado         0.00019144406754754303  0.0001832241002183749   4.086482674550891e6     0.003163401
# tornado         0.005762722391111711    0.0059539586538002965   135795.3601744771       0.010594597
# tornado         0.014085254526087958    0.014560240166900624    55555.9188156724        0.019611735
# tornado         0.025749083150050013    0.024464624404798668    30378.764231026173      0.031200647
# tornado         0.03901497303780279     0.04068398627041344     20055.702448546886      0.0556092
# tornado         0.065860549666099       0.07232758857712564     11876.44122749567       0.08892811
# tornado         0.11792354227887294     0.09764554820728881     6632.709095716476       0.10743219
# tornado         0.13714207066720346     0.11922264653996906     5701.757938027382       0.13338953
# tornado         0.11950217629916997     0.1629445887353792      6545.895178794861       0.21110983
# tornado         0.3223328187072219      0.2578566699321067      2412.3904927372932      1.0
# wind            0.001468393779175532    0.001403827470378238    4.008272053004682e6     0.028952872
# wind            0.042684383100977365    0.04519850326760836     137889.67413407564      0.067867324
# wind            0.08652308964665463     0.08806486755178454     68020.54129821062       0.11400817
# wind            0.1462680052638156      0.13932669910442497     40238.03479319811       0.1691322
# wind            0.19387856701171788     0.19778705117992362     30358.51892721653       0.22992225
# wind            0.26006468302927577     0.2599617751010517      22630.43121212721       0.2912621
# wind            0.3216356207214249      0.31738174725452556     18299.949753046036      0.34566957
# wind            0.37530013253877215     0.3813419102150456      15683.16629344225       0.42783338
# wind            0.5051760111478776      0.4935101982184035      11650.013535499573      0.5762383
# wind            0.7005335272862856      0.7065631723542611      8395.23120188713        1.0
# hail            0.0006361628101032178   0.000632607502902406    4.08976477201128e6      0.017750435
# hail            0.027668078984610604    0.026912153209588983    94021.54924637079       0.039141823
# hail            0.049324641333907404    0.04917863959617647     52744.689136207104      0.060695738
# hail            0.07076239578858932     0.07077144068676647     36765.249469578266      0.08320593
# hail            0.0945895789869977      0.09751802311160625     27510.14194124937       0.11524189
# hail            0.13315121865397708     0.13177904001076032     19543.177692770958      0.15030804
# hail            0.16760804177128757     0.17147839525232203     15522.64737278223       0.19774066
# hail            0.22509880666894788     0.22628648956572045     11557.537741959095      0.26114574
# hail            0.32135948067427483     0.29953004735426947     8096.390666127205       0.34962097
# hail            0.43937859376043        0.4447459271823616      5911.4588750600815      1.0
# sig_tornado     2.571871549061692e-5    1.9986018385744473e-5   4.251143661282301e6     0.0006216399
# sig_tornado     0.001428204109318158    0.0018039419639181778   76732.54330050945       0.0050245933
# sig_tornado     0.008097979718414744    0.007703846572923989    13551.042606294155      0.011668011
# sig_tornado     0.011247669363099348    0.016459588651393253    9704.275691211224       0.022787407
# sig_tornado     0.04009750246890077     0.025934578686081566    2729.2030975818634      0.029620264
# sig_tornado     0.03658756842066715     0.03613689887174313     2991.6218940615654      0.046061274
# sig_tornado     0.0519084164793314      0.059100801103409734    2107.990620672703       0.076002166
# sig_tornado     0.07994975637014516     0.09154583990714128     1366.5978977680206      0.11275637
# sig_tornado     0.16979111384982726     0.13710055728824463     647.3261821866035       0.1662113
# sig_tornado     0.22524476619828537     0.21917497324751414     463.35158079862595      1.0
# sig_wind        0.00017331051707854     0.0001581066155393539   4.038138275197029e6     0.0028377278
# sig_wind        0.0047390320219361836   0.004583017978492242    147848.3398333788       0.007203269
# sig_wind        0.009499470345166806    0.010207280582933466    73677.81734418869       0.014998861
# sig_wind        0.01880426208865085     0.019974838924628386    37250.23594087362       0.026544504
# sig_wind        0.03413324505167252     0.032354517222172786    20504.770110189915      0.039056044
# sig_wind        0.040920646498821954    0.04715748584821698     17106.25998610258       0.058905784
# sig_wind        0.08090383866161632     0.0711214670676529      8658.907377064228       0.08654974
# sig_wind        0.08416943865290621     0.10041986582952954     8314.231711268425       0.11052245
# sig_wind        0.11545896491469304     0.11655473676112905     6068.401515364647       0.12404624
# sig_wind        0.1797980379304038      0.14082417659091237     3870.375137925148       1.0
# sig_hail        8.558498271245034e-5    8.17697466598171e-5     4.23456178406769e6      0.005622294
# sig_hail        0.007994276778490537    0.008954205778869398    45310.74785798788       0.013451036
# sig_hail        0.015417645442972493    0.017320155666033284    23497.35624051094       0.02173031
# sig_hail        0.024363196379011007    0.024430214146091225    14878.557882726192      0.026939921
# sig_hail        0.03239713531442493     0.029028804919280133    11162.627854585648      0.031307817
# sig_hail        0.03590073703696473     0.03412947338812053     10096.31495910883       0.037602838
# sig_hail        0.036933492455797495    0.043795592147574974    9790.91767668724        0.05339017
# sig_hail        0.06697919386624233     0.062257185093183384    5410.661112964153       0.07159327
# sig_hail        0.10515508426842482     0.07950517096073371     3446.4898949861526      0.09021059
# sig_hail        0.10845700375706654     0.12097012510236035     3282.1566061377525      1.0







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






# Calibrate to SPC

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

# event_name   nominal_prob  threshold_to_match_succes_ratio  threshold_to_match_POD  mean_threshold  success_ratio         POD
# tornado      0.02          0.01310648                       0.022883907             0.017995194     0.062252367844859956  0.7237450145071389
# tornado      0.05          0.052413                         0.09069385              0.071553424     0.13041471700846416   0.4595979609764453
# tornado      0.1           0.1621605                        0.18851317              0.17533684      0.2602604408764249    0.14314764321503318
# tornado      0.15          0.3515455                        0.2584691               0.3050073       0.2477242309012129    0.008127872749010829
# tornado      0.3           0.4236154                        0.30122894              0.36242217      0.34169923357087156   0.005138386437942132
# wind         0.05          0.021778211                      0.10755004              0.064664125     0.21349036193848545   0.8077283104148264
# wind         0.15          0.09295757                       0.29656035              0.19475895      0.35260570410877334   0.5574172101278139
# wind         0.3           0.29035532                       0.56562936              0.42799234      0.587179141844545     0.19983921956088188
# wind         0.45          0.45901692                       0.8673674               0.66319215      0.7595734431193403    0.06063552925564667
# hail         0.05          0.01444374                       0.05034159              0.032392666     0.10835004167125925   0.8305503616488299
# hail         0.15          0.07098855                       0.14591776              0.108453155     0.20702496087073188   0.5193486074366591
# hail         0.3           0.19331579                       0.37884295              0.28607938      0.39332204971781815   0.16613831846637464
# hail         0.45          0.5076345                        0.625119                0.56637675      0.5888789612140184    0.013460490246840578
# sig_tornado  0.1           0.04061179                       0.121247366             0.08092958      0.13623988675028018   0.2807708016672483
# sig_wind     0.1           0.092810735                      0.12636001              0.109585375     0.13657851645778404   0.20438348540838872
# sig_hail     0.1           0.04965006                       0.06448765              0.057068855     0.09399092050668835   0.28431260828958804


# prediction_i = 1
# event_name, _ = CombinedHREFSREF.models[prediction_i];
# y = Ys[event_name];
# ŷ = @view X[:, prediction_i];
# for threshold in 0.25:0.01:0.6
#   sr  = success_ratio(ŷ, y, weights, threshold)
#   pod = probability_of_detection(ŷ, y, weights, threshold)
#   println("$threshold\t$sr\t$pod")
# end


# calibration = [
#   (0.02, 0.016253397),
#   (0.05, 0.0649308),
#   (0.1,  0.18771306),
#   (0.15, 0.28330332),
#   (0.3,  0.32384455),
# ]

println(calibrations)

# calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
#   "tornado" => [
#     (0.02, 0.017995194),
#     (0.05, 0.071553424),
#     (0.1,  0.17533684),
#     (0.15, 0.3050073),
#     (0.3,  0.36242217)
#   ],
#   "wind" => [
#     (0.05, 0.064664125),
#     (0.15, 0.19475895),
#     (0.3,  0.42799234),
#     (0.45, 0.66319215)
#   ],
#   "hail" => [
#     (0.05, 0.032392666),
#     (0.15, 0.108453155),
#     (0.3,  0.28607938),
#     (0.45, 0.56637675)
#   ],
#   "sig_tornado" => [(0.1, 0.08092958)],
#   "sig_wind"    => [(0.1, 0.109585375)],
#   "sig_hail"    => [(0.1, 0.057068855)],
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
# tornado         0.15            0.2715841                       0.13205521              0.20181966      0.2348807750696644      0.006138932381166391
# tornado         0.3             0.37645632                      0.1485431               0.26249972      0.29040593698579        0.005512840471253729
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