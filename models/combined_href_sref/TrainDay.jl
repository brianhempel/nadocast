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
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 616

validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_forecasts);
length(validation_forecasts_0z_12z) # 308

@time Forecasts.data(validation_forecasts[10]); # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


# rm("day_accumulators_validation_forecasts_0z_12z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts_0z_12z;
    event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
    save_dir = "day_accumulators_validation_forecasts_0z_12z",
  );



# should do some checks here.
import PlotMap

dec11 = validation_forecasts_0z_12z[259]; Forecasts.time_title(dec11) # "2021-12-11 00Z +35"
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  prediction_i = div(i - 1, 2) + 1
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  PlotMap.plot_debug_map("dec11_0z_day_accs_$(i)_$event_name", dec11.grid, dec11_data[:,i]);
end
for (event_name, labeler) in TrainingShared.event_name_to_day_labeler
  dec11_labels = labeler(dec11);
  PlotMap.plot_debug_map("dec11_0z_day_$event_name", dec11.grid, dec11_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_1_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_2_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_3_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_4_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_5_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_6_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_7_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_8_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_9_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_10_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_11_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_12_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_13_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_14_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_15_sig_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_16_sig_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_sig_hail.pdf ./


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
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z_12z, X, Ys, weights)

# tornado (18464.0)        feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.14686555
# tornado (18464.0)        feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.13910441
# wind (132107.0)          feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.4199436
# wind (132107.0)          feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.39290816
# wind_adj (45947.96)      feature 5 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.25088573
# wind_adj (45947.96)      feature 6 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.23519003
# hail (60620.0)           feature 7 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.26877668
# hail (60620.0)           feature 8 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.24924032
# sig_tornado (2453.0)     feature 9 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.08099959
# sig_tornado (2453.0)     feature 10 highest hourly STORPROB:calculated:day           fcst:: AU-PR-curve: 0.080312535
# sig_wind (15350.0)       feature 11 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.093816966
# sig_wind (15350.0)       feature 12 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.08649487
# sig_wind_adj (5766.4995) feature 13 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.106180646
# sig_wind_adj (5766.4995) feature 14 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.10117872
# sig_hail (8202.0)        feature 15 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.092503265
# sig_hail (8202.0)        feature 16 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.08510517


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

# event_name   mean_y                 mean_ŷ                 Σweight              bin_max
# tornado      0.0005722087751186967  0.0006358079789730076  7.5759277820932865e6 0.022898283
# tornado      0.03450376684670141    0.04126565697241027    125635.04517567158   0.07626015
# tornado      0.10104661298962272    0.10994672028759699    42900.56347054243    0.16708241
# tornado      0.21468905200727117    0.2770817498668438     20175.18288511038    1.0
# wind         0.004135521405694254   0.005290016978515359   7.429321955440998e6  0.13070628
# wind         0.16888291313049308    0.19424919678345737    181925.1034783125    0.2815187
# wind         0.31518044472264406    0.3598383147010067     97479.2459025383     0.46197507
# wind         0.5494606987330766     0.6063086927574567     55912.26880276203    1.0
# wind_adj     0.0014073590733128518  0.0015595208997444445  7.502320085881352e6  0.044802103
# wind_adj     0.06618157071095644    0.07453323377641596    159537.75696367025   0.122619815
# wind_adj     0.14595342843268744    0.17710226906256957    72343.3106443882     0.26531604
# wind_adj     0.34683110797057193    0.37012429072649966    30437.420135200024   1.0
# hail         0.0018777040570915257  0.002366908676200889   7.482577994220674e6  0.06832412
# hail         0.08542666191910385    0.10526386356265817    164473.20186179876   0.16137148
# hail         0.178627355590095      0.2178865422061829     78654.62258785963    0.302417
# hail         0.3608544192424186     0.42977400632265383    38932.75495427847    1.0
# sig_tornado  7.585152421990364e-5   8.109445002059634e-5   7.715216754996419e6  0.010960604
# sig_tornado  0.01652576687904753    0.02125682066576666    35430.6884662509     0.047013957
# sig_tornado  0.05852878403513349    0.07958172653582864    10002.797970473766   0.13788173
# sig_tornado  0.14627763825417575    0.20587454106446854    3988.332191467285    1.0
# sig_wind     0.0004729250913414752  0.0005369393755615952  7.524172598376453e6  0.016674783
# sig_wind     0.02261233638514802    0.030243108459245365   157374.9769256711    0.053257223
# sig_wind     0.06357723066555106    0.07145538644764637    55969.94667571783    0.09693365
# sig_wind     0.13112454711110128    0.13214804806177566    27121.051646769047   1.0
# sig_wind_adj 0.00017312250116607929 0.00019865404526796132 7.627311558152139e6  0.008988211
# sig_wind_adj 0.013977080716853841   0.016537915806305287   94472.32980048656    0.02997692
# sig_wind_adj 0.03859675008746257    0.04855956571213662    34213.65082883835    0.08362562
# sig_wind_adj 0.15267131304255674    0.1243573221587458     8641.034843146801    1.0
# sig_hail     0.00025050572439512937 0.0002915816136380767  7.625256641854286e6  0.015938511
# sig_hail     0.020615693973035008   0.026741968704312256   92694.20390033722    0.04517084
# sig_hail     0.05841030088536394    0.06330639696647453    32705.93824136257    0.09306535
# sig_hail     0.13650642813533811    0.14000707461020861    13981.789628624916   1.0

println("event_to_day_bins = $event_to_day_bins")
# event_to_day_bins = Dict{String, Vector{Float32}}("sig_wind" => [0.016674783, 0.053257223, 0.09693365, 1.0], "sig_hail" => [0.015938511, 0.04517084, 0.09306535, 1.0], "hail" => [0.06832412, 0.16137148, 0.302417, 1.0], "sig_wind_adj" => [0.008988211, 0.02997692, 0.08362562, 1.0], "tornado" => [0.022898283, 0.07626015, 0.16708241, 1.0], "wind_adj" => [0.044802103, 0.122619815, 0.26531604, 1.0], "sig_tornado" => [0.010960604, 0.047013957, 0.13788173, 1.0], "wind" => [0.13070628, 0.2815187, 0.46197507, 1.0])



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
      ("total_prob_au_pr", Metrics.area_under_pr_curve_fast(bin_total_prob_x, bin_y, bin_weights)),
      ("max_hourly_au_pr", Metrics.area_under_pr_curve_fast(bin_max_hourly_x, bin_y, bin_weights)),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Metrics.area_under_pr_curve_fast(logistic_ŷ, bin_y, bin_weights)),
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

# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado      1-2 -1.0             0.07626015       8358051 9310.0    7.701563e6  0.0012985998      0.00033090718     0.0011257319  0.0059756134       0.006628513        0.03530084       0.031198628      0.0011257319    0.0059625222     0.03516895     Float32[0.89827985, 0.08686848,   -0.081391975]
# tornado      2-3 0.022898283      0.16708241       179421  9266.0    168535.62   0.058748342       0.015907697       0.051442172   0.19309793         0.2222641          0.10248497       0.093642585      0.05144217      0.19251166       0.10156963     Float32[1.148995,   -0.06917223,  -0.03546825]
# tornado      3-4 0.07626015       1.0              66365   9154.0    63075.746   0.16340594        0.043742545       0.13739587    0.3898923          0.46192595         0.22210684       0.21380454       0.13739587      0.38465714       0.22422877     Float32[0.60322016, 0.1676681,    -0.30535954]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind         1-2 -1.0             0.2815187        8259282 66078.0   7.611247e6  0.009806546       0.0025530525      0.008073336   0.028679347        0.032908533        0.1611215        0.14657311       0.008073336     0.028488716      0.16110326     Float32[0.9534275,  0.06817708,   -0.06704924]
# wind         2-3 0.13070628       0.46197507       300311  66117.0   279404.38   0.2520203         0.07104193        0.21992353    0.5106927          0.63625956         0.31169105       0.27879          0.21992353      0.5076161        0.31312683     Float32[1.0759317,  -0.11009742,  -0.3944708]
# wind         3-4 0.2815187        1.0              165134  66029.0   153391.52   0.44967845        0.14340067        0.4005772     0.63903964         0.86214536         0.5768069        0.53934246       0.40057722      0.63347197       0.5767191      Float32[0.95407456, -0.031302005, -0.28859687]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind_adj     1-2 -1.0             0.122619815      8311938 22819.45  7.661858e6  0.003079004       0.00080840505     0.0027561092  0.0125278095       0.014097166        0.062548995      0.057081066      0.002756109     0.012505848      0.06219148     Float32[0.8554493,  0.14431913,   0.08940227]
# wind_adj     2-3 0.044802103      0.26531604       251531  22974.477 231881.08   0.10653318        0.03036918        0.09106916    0.29617175         0.34160262         0.1499125        0.13531469       0.091069154     0.2946373        0.14998621     Float32[0.9335435,  -0.04865307,  -0.47948343]
# wind_adj     3-4 0.122619815      1.0              112478  23128.508 102780.734  0.23426367        0.0713346         0.2054412     0.47592828         0.57934463         0.38302815       0.3581152        0.2054412       0.4728324        0.38284662     Float32[1.1919755,  -0.05395548,  -0.10846542]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# hail         1-2 -1.0             0.16137148       8297468 30348.0   7.647051e6  0.0045800223      0.0010763744      0.00367468    0.015448958        0.01750515         -0.3853131       0.07704641       0.0036746797    0.015342001      0.085015535    Float32[0.9860878,  0.049258705,  -0.05329424]
# hail         2-3 0.06832412       0.302417         262522  30300.0   243127.83   0.14169858        0.034936387       0.115578145   0.34866542         0.41445842         0.17760482       0.16150278       0.11557814      0.34550518       0.17980167     Float32[1.136249,   -0.14193086,  -0.47682077]
# hail         3-4 0.16137148       1.0              126948  30272.0   117587.375  0.2880417         0.08137089        0.23896208    0.52802193         0.6546696          0.38866416       0.3595268        0.23896208      0.5202895        0.39265144     Float32[1.2677307,  -0.3622067,   -0.94742197]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_tornado  1-2 -1.0             0.047013957      8409839 1239.0    7.7506475e6 0.00017789546     5.0344523e-5      0.00015104935 0.00096891326      0.001041482        0.015948392      0.01563456       0.00015104935   0.00096336694    0.015612346    Float32[0.48711854, 0.40669206,   -0.15710305]
# sig_tornado  2-3 0.010960604      0.13788173       47648   1222.0    45433.49    0.03409784        0.010884164       0.0257733     0.11354815         0.11973138         0.07077905       0.085406534      0.0257733       0.111716956      0.09161759     Float32[0.4327529,  0.54704756,   0.31344774]
# sig_tornado  3-4 0.047013957      1.0              14577   1214.0    13991.131   0.115582936       0.033114433       0.0835426     0.28255722         0.3058572          0.12555668       0.124640875      0.0835426       0.27507812       0.12815252     Float32[0.52736497, 0.50139576,   0.3873906]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind     1-2 -1.0             0.053257223      8334639 7684.0    7.681548e6  0.0011455417      0.00028609237     0.000926504   0.0053053973       0.0057347543       0.022761863      0.022178898      0.00092650414   0.0052557266     0.023225786    Float32[0.28422165, 0.6058833,    0.1703358]
# sig_wind     2-3 0.016674783      0.09693365       229628  7672.0    213344.92   0.04105494        0.010572196       0.033359267   0.14124396         0.1567431          2.163317         0.06364868       0.033359263     0.14001511       0.066456445    Float32[0.78268194, 0.4231073,    1.0043924]
# sig_wind     3-4 0.053257223      1.0              89777   7666.0    83091.0     0.091265574       0.021731766       0.0856248     0.28414002         0.34129137         0.1421623        0.12985703       0.0856248       0.2831936        0.13823211     Float32[0.8006246,  0.42102247,   1.0746965]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind_adj 1-2 -1.0             0.02997692       8377102 2862.5508 7.721784e6  0.00039855705     9.397876e-5       0.00034200732 0.0021713434       0.0023651219       0.011507183      0.011363501      0.00034200732   0.0021629222     0.01167132     Float32[0.4749642,  0.4533547,    0.14967395]
# sig_wind_adj 2-3 0.008988211      0.08362562       140669  2888.1807 128685.984  0.025051488       0.006213596       0.020522693   0.09726903         0.107263066        0.045148544      0.04309167       0.020522693     0.09665315       0.045431405    Float32[0.46408254, 0.43720487,   0.07941379]
# sig_wind_adj 3-4 0.02997692       1.0              47314   2903.9487 42854.688   0.0638431         0.016088799       0.061598253   0.21245395         0.25236997         0.18353157       0.17386593       0.061598253     0.21012315       0.18458796     Float32[1.194624,   0.28879985,   1.5873092]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_hail     1-2 -1.0             0.04517084       8374334 4113.0    7.717951e6  0.00060925627     0.00015367863     0.00049509585 0.0027780822       0.0030430474       0.7119745        0.0208722        0.00049509597   0.0027645666     0.020328494    Float32[1.2652406,  -0.22227533,  -0.34498304]
# sig_hail     2-3 0.015938511      0.09306535       135167  4123.0    125400.14   0.03627843        0.00923679        0.030473003   0.13186373         0.14872126         0.059983674      0.053631697      0.030473009     0.1311022        0.06284146     Float32[1.5420237,  -0.37746048,  -0.21447094]
# sig_hail     3-4 0.04517084       1.0              50082   4089.0    46687.73    0.0862763         0.02363578        0.08179811    0.27327767         0.32266483         0.14158726       0.12934656       0.08179811      0.27281204       0.14230464     Float32[1.409416,   -0.404029,    -0.62075996]


println("event_to_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[0.28422165, 0.6058833, 0.1703358], [0.78268194, 0.4231073, 1.0043924], [0.8006246, 0.42102247, 1.0746965]], "sig_hail" => [[1.2652406, -0.22227533, -0.34498304], [1.5420237, -0.37746048, -0.21447094], [1.409416, -0.404029, -0.62075996]], "hail" => [[0.9860878, 0.049258705, -0.05329424], [1.136249, -0.14193086, -0.47682077], [1.2677307, -0.3622067, -0.94742197]], "sig_wind_adj" => [[0.4749642, 0.4533547, 0.14967395], [0.46408254, 0.43720487, 0.07941379], [1.194624, 0.28879985, 1.5873092]], "tornado" => [[0.89827985, 0.08686848, -0.081391975], [1.148995, -0.06917223, -0.03546825], [0.60322016, 0.1676681, -0.30535954]], "wind_adj" => [[0.8554493, 0.14431913, 0.08940227], [0.9335435, -0.04865307, -0.47948343], [1.1919755, -0.05395548, -0.10846542]], "sig_tornado" => [[0.48711854, 0.40669206, -0.15710305], [0.4327529, 0.54704756, 0.31344774], [0.52736497, 0.50139576, 0.3873906]], "wind" => [[0.9534275, 0.06817708, -0.06704924], [1.0759317, -0.11009742, -0.3944708], [0.95407456, -0.031302005, -0.28859687]])




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
cutoff = Dates.DateTime(2022, 6, 1, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected:
#

# rm("day_validation_forecasts_0z_12z_with_sig_gated"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_12z;
    event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_12z_with_sig_gated",
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
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

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
test_predictive_power_all(day_validation_forecasts_0z_12z, X, Ys, weights)

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


# rm("day_accumulators_validation_forecasts_0z_12z"; recursive = true)

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
test_calibration(day_validation_forecasts_0z_12z, X, Ys, weights)

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
cutoff = Dates.DateTime(2022, 6, 1, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts) # 528

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10]);

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected:
#

# rm("day_validation_forecasts_0z_12z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_12z;
    event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_12z_spc_calibrated_with_sig_gated",
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
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

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
test_threshold(day_validation_forecasts_0z_12z, X, Ys, weights, 0.1)

# tornado (8326.0)                      feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  Threshold: 0.1  CSI: 0.0999275885395187
# wind (63336.0)                        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.2621078607407563
# hail (28152.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.1533293040297049
# sig_tornado (1138.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.1019263018731568
# sig_wind (7555.0)                     feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.07713544881757496
# sig_hail (3887.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 Threshold: 0.1  CSI: 0.07583280034351035
# sig_tornado_gated_by_tornado (1138.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado Threshold: 0.1  CSI: 0.10237996019129936
# sig_wind_gated_by_wind (7555.0)       feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    Threshold: 0.1  CSI: 0.07713544881757496
# sig_hail_gated_by_hail (3887.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    Threshold: 0.1  CSI: 0.07584469907328449



