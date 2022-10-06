import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories


(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 17918


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts;
    event_name_to_labeler = TrainingShared.event_name_to_labeler,
    save_dir = "validation_forecasts_href_newer"
  );

# ...

event_types_count = length(CombinedHREFSREF.models)

# Confirm that HREF models are better
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  # Feature order is all HREF severe probs then all SREF severe probs
  for feature_i in 1:length(inventory)
    prediction_i = 1 + (feature_i - 1) % event_types_count
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (66484.0)        feature 1 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.04096050922959824
# wind (524123.0)          feature 2 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.12359914323642289
# wind_adj (158758.72)     feature 3 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.06579056292038551
# hail (237780.0)          feature 4 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.07644116022386543
# sig_tornado (9355.0)     feature 5 STORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.02930912296157666
# sig_wind (51376.0)       feature 6 SWINDPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.016991991337000804
# sig_wind_adj (18092.895) feature 7 SWINDPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.013898139435587277
# sig_hail (27885.0)       feature 8 SHAILPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.018954516980244043
# tornado (66484.0)        feature 9 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.023198579852888944
# wind (524123.0)          feature 10 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.09053563878797445
# wind_adj (158758.72)     feature 11 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.04608418454423105
# hail (237780.0)          feature 12 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.05272323284307826
# sig_tornado (9355.0)     feature 13 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.013200836534184881
# sig_wind (51376.0)       feature 14 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011702071814021137
# sig_wind_adj (18092.895) feature 15 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.015107926205925437
# sig_hail (27885.0)       feature 16 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.013757293287310518



# 3. bin HREF predictions into 6 bins of equal weight of positive labels
# (we will combine adjacent pairs of bins into 5 pairwise overlapping bins)

const bin_count = 6

function find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Metrics.parallel_iterate(is -> sum(Float64.(view(y, is) .* view(weights, is))), length(y)))
  per_bin_pos_weight = total_positive_weight / bin_count

  ŷ              = @view X[:,prediction_i]; # HREF prediction for event_name
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

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
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

event_to_bins = Dict{String,Vector{Float32}}()
for prediction_i in 1:event_types_count
  (event_name, _) = CombinedHREFSREF.models[prediction_i]

  event_to_bins[event_name] = find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)
end

# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# tornado      2.0017512048824645e-5  2.082888175135988e-5  5.2064060230394334e8 0.0013122079
# tornado      0.002655889902000432   0.002478021926246398  3.9244070574436784e6 0.004710326
# tornado      0.0069640493880342375  0.007407600221261315  1.4966173550154567e6 0.011888903
# tornado      0.017362635745529827   0.016314321153062895  600276.2435635328    0.022864908
# tornado      0.02993764406144195    0.032014090804743975  348149.42763745785   0.04753879
# tornado      0.08124910991229588    0.07064408763579479   128235.19191151857   1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# wind         0.00015741266281200494 0.0001847099208734292 5.164470169485555e8  0.008744916
# wind         0.014206964919374733   0.013853129679784285  5.722189787199557e6  0.021842286
# wind         0.032677585289046206   0.02959411232914133   2.487775991825223e6  0.04064058
# wind         0.0610121840626526     0.05248833196099973   1.3324318704804182e6 0.06938802
# wind         0.10708215059052975    0.0902320184595865    759185.4683158398    0.122678265
# wind         0.2086090465186428     0.18870530573215794   389687.51313841343   1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# wind_adj     4.6861304088148505e-5  5.7563253031589765e-5 5.1869171649894786e8 0.0028606635
# wind_adj     0.004667626540476744   0.00489083305847296   5.2076608349071145e6 0.008563636
# wind_adj     0.013630166347820426   0.012057438354943648  1.7833216788041592e6 0.017410113
# wind_adj     0.029274485125305144   0.023154653615100468  830326.4349123836    0.031802237
# wind_adj     0.056361043895736714   0.04275826682844031   431268.7670508623    0.060865648
# wind_adj     0.12528409842259042    0.0949236436978124    193993.36489260197   1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# hail         7.107382862383238e-5   6.692442480027312e-5  5.168657223346699e8  0.003458654
# hail         0.006457659294796132   0.0057813912997365675 5.688788030076444e6  0.009600723
# hail         0.014870803766749464   0.01374327077317342   2.470369201535642e6  0.020057917
# hail         0.031030847719972318   0.026941203482532337  1.1838481899712682e6 0.037324984
# hail         0.05633563361289792    0.05160697245250427   652090.3467494845    0.076065265
# hail         0.1323882140579507     0.12057726119107835   277469.4765122533    1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# sig_tornado  2.837780059547531e-6   2.958048051348655e-6  5.255118766364023e8  0.0007784463
# sig_tornado  0.0013153812203207443  0.001625880945018886  1.1339622947739959e6 0.0033980992
# sig_tornado  0.004742355433771681   0.0050626061843344035 314396.9325340986    0.007858597
# sig_tornado  0.014811241988098994   0.010144010653834806  100693.46283829212   0.013500211
# sig_tornado  0.03097861945639863    0.01684425238698523   48140.36363977194    0.021512743
# sig_tornado  0.05095276702824725    0.03128486659398223   29217.889326512814   1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# sig_wind     1.5325299167499523e-5  1.5086300260890803e-5 5.180116382436528e8  0.00076082407
# sig_wind     0.0014118681391051162  0.0013829152656214556 5.622756191015422e6  0.0025200176
# sig_wind     0.004175745303523315   0.003585880442542822  1.9010839804090858e6 0.0051898635
# sig_wind     0.009559409602331962   0.006678366386970533  830492.9002404213    0.008781022
# sig_wind     0.015464049811301612   0.011620243492001886  513391.43561410904   0.016167704
# sig_wind     0.030648287561612413   0.028665423763122     258924.8285831213    1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# sig_wind_adj 5.286534758688006e-6   8.135203866678671e-6  5.2239671219077384e8 0.0005631988
# sig_wind_adj 0.0009064378093370669  0.0008920361052785085 3.046676583100617e6  0.0014582027
# sig_wind_adj 0.0028822160533730147  0.0019082603474613292 958203.8290531039    0.00254442
# sig_wind_adj 0.006194209307918673   0.0031377993361207126 445882.5478153229    0.0040013813
# sig_wind_adj 0.01517405535053632    0.004729783573926283  182026.16273784637   0.0057724603
# sig_wind_adj 0.02537620484000769    0.007939886255773203  108786.26603424549   1.0
# event_name   mean_y                 mean_ŷ                Σweight              bin_max
# sig_hail     8.286829228855413e-6   7.859713907548184e-6  5.221983799890184e8  0.0007729447
# sig_hail     0.0015600304941201866  0.0013179577646831022 2.7740064467158914e6 0.0022382603
# sig_hail     0.003751650318916818   0.00322102967581788   1.153474297762096e6  0.0047318107
# sig_hail     0.007472737305743317   0.00650323545254852   579104.410691917     0.009228747
# sig_hail     0.014066989688945986   0.013087991077025648  307668.43133723736   0.020026933
# sig_hail     0.034422544845844824   0.034143674029227834  125654.00398945808   1.0

println("href_newer_event_to_bins = $event_to_bins")
# href_newer_event_to_bins = Dict{String, Vector{Float32}}("sig_wind" => [0.00076082407, 0.0025200176, 0.0051898635, 0.008781022, 0.016167704, 1.0], "sig_hail" => [0.0007729447, 0.0022382603, 0.0047318107, 0.009228747, 0.020026933, 1.0], "hail" => [0.003458654, 0.009600723, 0.020057917, 0.037324984, 0.076065265, 1.0], "sig_wind_adj" => [0.0005631988, 0.0014582027, 0.00254442, 0.0040013813, 0.0057724603, 1.0], "tornado" => [0.0013122079, 0.004710326, 0.011888903, 0.022864908, 0.04753879, 1.0], "wind_adj" => [0.0028606635, 0.008563636, 0.017410113, 0.031802237, 0.060865648, 1.0], "sig_tornado" => [0.0007784463, 0.0033980992, 0.007858597, 0.013500211, 0.021512743, 1.0], "wind" => [0.008744916, 0.021842286, 0.04064058, 0.06938802, 0.122678265, 1.0])

# href_newer_event_to_bins = Dict{String, Vector{Float32}}(
#   "tornado"     => [0.0010191497,  0.0040240395, 0.009904478,  0.021118658, 0.04198461,  1.0],
#   "wind"        => [0.008220518,   0.020898983,  0.039187722,  0.067865305, 0.1211984,   1.0],
#   "hail"        => [0.0033015092,  0.00926606,   0.019275624,  0.03602357,  0.07374218,  1.0],
#   "sig_tornado" => [0.0006729238,  0.0026572358, 0.0057637123, 0.010840383, 0.02260619,  1.0],
#   "sig_wind"    => [0.0007965934,  0.0025051977, 0.0053047705, 0.008843536, 0.015871514, 1.0],
#   "sig_hail"    => [0.00077555457, 0.002129781,  0.004338203,  0.00847202,  0.018383306, 1.0],
# )



# 4. combine bin-pairs (overlapping, 5 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + b)
# For the 2020 models, adding more terms resulted in dangerously large coefficients
# There's more data this year...try interaction terms this time? nah

function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,prediction_i]; # HREF prediction for event_name

  bins_max = event_to_bins[event_name]
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_href_x  = X[bin_members, prediction_i]
    bin_sref_x  = X[bin_members, prediction_i + event_types_count]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

    Threads.@threads for i in 1:length(bin_y)
      logit_href = logit(bin_href_x[i])
      logit_sref = logit(bin_sref_x[i])

      bin_X_features[i,1] = logit_href
      bin_X_features[i,2] = logit_sref
      # bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
      # bin_X_features[i,3] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,5] = max(logit_href, logit_sref)
      # bin_X_features[i,6] = min(logit_href, logit_sref)
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("event_name", event_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("HREF_ŷ_min", bin_min),
      ("HREF_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_HREF_ŷ", sum(bin_href_x .* bin_weights) / bin_weight),
      ("mean_SREF_ŷ", sum(bin_sref_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("HREF_logloss", sum(logloss.(bin_y, bin_href_x) .* bin_weights) / bin_weight),
      ("SREF_logloss", sum(logloss.(bin_y, bin_sref_x) .* bin_weights) / bin_weight),
      ("HREF_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_href_x, bin_y, bin_weights))),
      ("SREF_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_sref_x, bin_y, bin_weights))),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Float32(Metrics.area_under_pr_curve_fast(logistic_ŷ, bin_y, bin_weights))),
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

event_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:event_types_count
  event_name, _ = CombinedHREFSREF.models[prediction_i]

  event_to_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end
print("href_newer_event_to_bins_logistic_coeffs = $event_to_bins_logistic_coeffs")

# href_newer_event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[1.0237541, 0.07430771, 0.77363414], [0.9801676, 0.22527456, 1.4178667], [1.1523683, 0.2936303, 2.783076], [0.7023142, 0.22948036, 0.17595705], [0.49640635, 0.20968291, -0.77673304]], "sig_hail" => [[0.8787013, 0.3119461, 1.6104017], [0.7609628, 0.27349025, 0.5037077], [0.6508231, 0.3390255, 0.28728625], [0.562786, 0.35293543, -0.048666567], [0.63449466, 0.3219732, 0.13853967]], "hail" => [[0.8451918, 0.2678841, 0.7947575], [0.82366365, 0.1894165, 0.23550989], [0.9017725, 0.23303771, 0.7705032], [0.74698466, 0.2205954, 0.16990918], [0.86427146, 0.2018278, 0.43504176]], "sig_wind_adj" => [[1.127125, 0.052893594, 1.2472699], [1.5036446, 0.021982145, 3.6960568], [1.4205347, 0.16423136, 4.1276884], [1.5778238, 0.50511247, 7.107699], [0.1278618, 0.78106964, 0.9255426]], "tornado" => [[0.8548191, 0.2597601, 0.8644513], [0.78474545, 0.12277814, -0.4418093], [1.0832391, 0.038129725, 0.5669815], [0.75186867, 0.07035734, -0.6170576], [1.0223982, 0.12714103, 0.5787479]], "wind_adj" => [[0.9797254, 0.16798192, 0.8348953], [1.0740479, 0.15907931, 1.2374448], [1.0872117, 0.10037302, 1.0073562], [0.9246941, 0.19532852, 0.81610787], [0.73403955, 0.29090548, 0.62176204]], "sig_tornado" => [[0.7318563, 0.3246817, 0.37168285], [0.9523567, 0.26052013, 1.2165213], [1.473916, 0.17237528, 3.4237669], [1.3563457, 0.091417536, 2.4898052], [0.8635645, -0.0004715729, 0.056863528]], "wind" => [[0.9376985, 0.2066142, 0.70834965], [0.9932054, 0.15726718, 0.71593946], [1.0039783, 0.12171379, 0.6053313], [0.9382497, 0.13403365, 0.4595898], [0.74286795, 0.21366222, 0.2578835]])

# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado      1-2 -1.0          0.004710326  569208379 22345.0   5.24565e8   3.9211784e-5   6.456604e-5   3.973716e-5   0.00031916535 0.0003422021  626.2121     0.0017128479  3.9737166e-5    0.0003167851     0.0027803837   Float32[0.8548191,  0.2597601,     0.8644513]0.78474545,    0.12277814,    -0.4418093]]]]
# tornado      2-3 0.0013122079  0.011888903  5781338   22177.0   5.4210245e6 0.0038389629   0.0038622334  0.003845271   0.024633836   0.026100308   0.006912483  0.005551187   0.003845271     0.024592128      0.0071537965   Float32[0.78474545, 0.12277814,    -0.4418093]1.0832391,    0.038129725,   0.5669815]]]2]
# tornado      3-4 0.004710326   0.022864908  2222363   22155.0   2.0968936e6 0.00995732     0.00783764    0.009940845   0.054473914   0.058511913   0.016806537  0.012086184   0.009940845     0.054453935      0.017166965    Float32[1.0832391,  0.038129725,   0.5669815]0.75186867,    0.07035734,    -0.6170576]]
# tornado      4-5 0.011888903   0.04753879   999229    22155.0   948425.6    0.022077415    0.014752411   0.021978687   0.104568295   0.112321176   0.03025382   0.02623725    0.02197869      0.10446822       0.03129944     Float32[0.75186867, 0.07035734,    -0.6170576]1.0223982,    0.12714103,    0.5787479]]]
# tornado      5-6 0.022864908   1.0          499578    21984.0   476384.62   0.042412676    0.025556246   0.04374988    0.17315565    0.187718      0.085585535  0.06508306    0.04374988      0.17282005       0.086502574    Float32[1.0223982,  0.12714103,    0.5787479]0.9376985,     0.2066142,     0.70834965]6]8]5]]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind         1-2 -1.0          0.021842286  566591005 174683.0  5.221692e8  0.00033449524  0.00048726308 0.00031137464 0.0019269972  0.0020963203  0.013521824  0.009107277   0.0003113746    0.0019158684     0.014616285    Float32[0.9376985,  0.2066142,     0.70834965]0.9932054,    0.15726718,    0.71593946]]]]
# wind         2-3 0.008744916   0.04064058   8803587   174950.0  8.2099655e6 0.018622948    0.016239984   0.019803917   0.09495567    0.10050158    0.03273339   0.02709696    0.019803915     0.09469265       0.03376097     Float32[0.9932054,  0.15726718,    0.71593946]1.0039783,    0.12171379,    0.6053313]]1]
# wind         3-4 0.021842286   0.06938802   4101450   174955.0  3.8202078e6 0.03757928     0.026926853   0.042560276   0.17345326    0.18705434    0.061311703  0.05200345    0.042560272     0.17288563       0.062932536    Float32[1.0039783,  0.12171379,    0.6053313]0.9382497,     0.13403365,    0.4595898]3]]
# wind         4-5 0.04064058    0.122678265  2248953   174718.0  2.0916174e6 0.06618799     0.04108688    0.077734      0.26984674    0.29702047    18.022213    0.09432323    0.077734        0.26842478       0.11014767     Float32[0.9382497,  0.13403365,    0.4595898]0.74286795,    0.21366222,    0.2578835]]1]
# wind         5-6 0.06938802    1.0          1237865   174485.0  1.148873e6  0.12363328     0.06805951    0.14151917    0.39492828    0.44385427    0.23323557   0.2199623     0.14151916      0.39200872       0.24133481     Float32[0.74286795, 0.21366222,    0.2578835]0.9797254,     0.16798192,    0.8348953]]]]]4]5]]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind_adj     1-2 -1.0          0.008563636  568404909 52647.34  5.2389936e8 0.000105606894 0.00015426897 9.27926e-5    0.0006803202  0.00073936256 -336.9143    0.0031073175  9.2792594e-5    0.0006765985     0.0051007713   Float32[0.9797254,  0.16798192,    0.8348953]1.0740479,     0.15907931,    1.2374448]]1]]
# wind_adj     2-3 0.0028606635  0.017410113  7566482   52775.227 6.990983e6  0.006718954    0.0060438905  0.00695387    0.040193256   0.042339977   0.013645499  0.010509981   0.0069538704    0.040073097      0.0141548915   Float32[1.0740479,  0.15907931,    1.2374448]1.0872117,     0.10037302,    1.0073562]]]]
# wind_adj     3-4 0.008563636   0.031802237  2839518   52860.727 2.6136482e6 0.015582897    0.0113053005  0.018600188   0.09106082    0.0979951     0.028901337  0.024454327   0.018600188     0.09067247       0.030298846    Float32[1.0872117,  0.10037302,    1.0073562]0.9246941,     0.19532852,    0.81610787]]
# wind_adj     4-5 0.017410113   0.060865648  1376524   52972.758 1.2615952e6 0.02985603     0.0185977     0.038533863   0.16157289    0.17552921    0.057814516  0.055107273   0.038533863     0.15990607       0.06342116     Float32[0.9246941,  0.19532852,    0.81610787]0.73403955,   0.29090548,    0.62176204]
# wind_adj     5-6 0.031802237   1.0          685893    53250.652 625262.1    0.058943056    0.030459194   0.077745065   0.26776332    0.29967275    0.13099499   0.1240383     0.07774505      0.2632397        0.13559605     Float32[0.73403955, 0.29090548,    0.62176204]0.8451918,    0.2678841,     0.7947575]27]08]6]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# hail         1-2 -1.0          0.009600723  566981560 79203.0   5.225545e8  0.00012913495  0.00020661621 0.00014060138 0.0009784885  0.0010443097  0.0059510986 0.004502483   0.00014060139   0.00097103784    0.006586433    Float32[0.8451918,  0.2678841,     0.7947575]0.82366365,    0.1894165,     0.23550989]2]]
# hail         2-3 0.003458654   0.020057917  8790935   78861.0   8.159157e6  0.008192031    0.007832335   0.009004928   0.050347976   0.052898232   0.015294408  0.013610849   0.009004928     0.050146703      0.01655904     Float32[0.82366365, 0.1894165,     0.23550989]0.9017725,    0.23303771,    0.7705032]]]
# hail         3-4 0.009600723   0.037324984  3944739   79079.0   3.6542175e6 0.018018974    0.013768457   0.020106135   0.09679822    0.101909146   0.030801788  0.028565135   0.020106131     0.09623319       0.033326298    Float32[0.9017725,  0.23303771,    0.7705032]0.74698466,    0.2205954,     0.16990918]]
# hail         4-5 0.020057917   0.076065265  1984190   79461.0   1.8359385e6 0.035702016    0.022753712   0.04001863    0.16577706    0.17767467    0.05758951   0.053603645   0.040018626     0.16485053       0.060501777    Float32[0.74698466, 0.2205954,     0.16990918]0.86427146,   0.2018278,     0.43504176]]4]
# hail         5-6 0.037324984   1.0          1004021   79498.0   929559.8    0.0721943      0.03907389    0.07903699    0.2649131     0.2933911     0.1583389    0.1301408     0.079036996     0.26372224       0.1625015      Float32[0.86427146, 0.2018278,     0.43504176]0.7318563,    0.3246817,     0.37168285]6]]7]6]]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_tornado  1-2 -1.0          0.0033980992 571415613 3157.0    5.2664586e8 6.4524893e-6   1.0411996e-5  5.663919e-6   5.0117e-5     5.368466e-5   -270.88776   0.0014564153  5.663916e-6     4.956414e-5      0.002010731    Float32[0.7318563,  0.3246817,     0.37168285]0.9523567,    0.26052013,    1.2165213]]]]]]
# sig_tornado  2-3 0.0007784463  0.007858597  1527517   3099.0    1.4483592e6 0.0023718947   0.0022824816  0.0020592783  0.014226993   0.014800945   0.004824943  0.004631331   0.0020592778    0.014103547      0.00553721     Float32[0.9523567,  0.26052013,    1.2165213]1.473916,      0.17237528,    3.4237669]]0596]
# sig_tornado  3-4 0.0033980992  0.013500211  434467    3092.0    415090.38   0.0062952642   0.00516174    0.0071848864  0.041422654   0.043839168   0.0152505385 0.009980408   0.007184887     0.041027498      0.015249469    Float32[1.473916,   0.17237528,    3.4237669]1.3563457,     0.091417536,   2.4898052]]9]
# sig_tornado  4-5 0.007858597   0.021512743  154873    3093.0    148833.81   0.012311208    0.009323702   0.020040588   0.09868011    0.10735906    -4.9521246   0.02154685    0.020040585     0.096375324      0.030057909    Float32[1.3563457,  0.091417536,   2.4898052]0.8635645,     -0.0004715729, 0.056863528]
# sig_tornado  5-6 0.013500211   1.0          80240     3106.0    77358.25    0.022298414    0.015005912   0.038522772   0.16609094    0.18613957    0.05745042   0.041383285   0.038522772     0.16103002       0.05737542     Float32[0.8635645,  -0.0004715729, 0.056863528]1.0237541,   0.07430771,    0.77363414]]]7]]]3]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind     1-2 -1.0          0.0025200176 568154559 17088.0   5.2363437e8 2.9773972e-5   4.973542e-5   3.03213e-5    0.00025720775 0.00028017446 0.0011859061 0.0011574271  3.0321298e-5    0.0002568317     0.0014549745   Float32[1.0237541,  0.07430771,    0.77363414]0.9801676,    0.22527456,    1.4178667]]1]3]
# sig_wind     2-3 0.00076082407 0.0051898635 8088333   17092.0   7.52384e6   0.0019395489   0.0018818339  0.0021102298  0.014713033   0.015254561   0.0043461253 0.0044309245  0.0021102296    0.014650906      0.00510347     Float32[0.9801676,  0.22527456,    1.4178667]1.1523683,     0.2936303,     2.783076]]]3]4]
# sig_wind     3-4 0.0025200176  0.008781022  2940741   17123.0   2.7315768e6 0.004526102    0.0033781028  0.005812564   0.03525174    0.036748037   0.008891232  0.009010606   0.005812564     0.034847397      0.010287465    Float32[1.1523683,  0.2936303,     2.783076]0.7023142,      0.22948036,    0.17595705]]]
# sig_wind     4-5 0.0051898635  0.016167704  1449093   17113.0   1.3438842e6 0.008566266    0.005229259   0.011815105   0.06429231    0.06883264    -25.194296   0.015967341   0.011815103     0.06351321       0.017294403    Float32[0.7023142,  0.22948036,    0.17595705]0.49640635,   0.20968291,    -0.77673304]
# sig_wind     5-6 0.008781022   1.0          835020    17165.0   772316.25   0.017334769    0.008378071   0.020554679   0.09959018    0.10863672    0.031550676  0.02929273    0.020554679     0.09849223       0.032775298    Float32[0.49640635, 0.20968291,    -0.77673304]1.127125,    0.052893594,   1.2472699]]5]]]]]]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind_adj 1-2 -1.0          0.0014582027 570073265 5976.3447 5.2544336e8 1.3260324e-5   1.8264205e-5  1.0511677e-5  9.663733e-5   0.00010482192 0.0006804231 0.00047342628 1.0511673e-5    9.598097e-5      0.0007029565   Float32[1.127125,   0.052893594,   1.2472699]1.5036446,     0.021982145,   3.6960568]6]]2]
# sig_wind_adj 2-3 0.0005631988  0.00254442   4354061   6021.371  4.0048805e6 0.0011351769   0.0009954472  0.0013791606  0.010227289   0.010770103   0.002372932  0.0019563052  0.0013791608    0.010168242      0.0027636199   Float32[1.5036446,  0.021982145,   3.6960568]1.4205347,     0.16423136,    4.1276884]53]]]
# sig_wind_adj 3-4 0.0014582027  0.0040013813 1536512   6043.708  1.4040864e6 0.0022987134   0.0018911178  0.003933975   0.025846425   0.02701207    0.0063189953 0.005915751   0.003933975     0.025277013      0.0077176937   Float32[1.4205347,  0.16423136,    4.1276884]1.5778238,     0.50511247,    7.107699]]]]]
# sig_wind_adj 4-5 0.00254442    0.0057724603 690467    6046.878  627908.7    0.0035993038   0.0028931797  0.008797402   0.052323934   0.05348964    -754.5257    0.017745601   0.008797403     0.04874606       0.020440323    Float32[1.5778238,  0.50511247,    7.107699]0.1278618,      0.78106964,    0.9255426]4]]
# sig_wind_adj 5-6 0.0040013813  1.0          320543    6072.8413 290812.44   0.005930609    0.0048314314  0.018990444   0.102262706   0.102556184   0.026798423  0.04416696    0.018990444     0.090314254      0.044231992    Float32[0.1278618,  0.78106964,    0.9255426]0.8787013,     0.3119461,     1.6104017]5]]7]]
# event_name   bin HREF_ŷ_min    HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ    mean_SREF_ŷ   mean_y        HREF_logloss  SREF_logloss  HREF_au_pr   SREF_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_hail     1-2 -1.0          0.0022382603 569596459 9278.0    5.2497238e8 1.4782402e-5   2.6558637e-5  1.6486398e-5  0.00013887089 0.00014903312 0.0013314564 0.00082229695 1.6486392e-5    0.00013720112    0.0015203905   Float32[0.8787013,  0.3119461,     1.6104017]0.7609628,     0.27349025,    0.5037077]]3]]]
# sig_hail     2-3 0.0007729447  0.0047318107 4232683   9301.0    3.9274808e6 0.001876877    0.0018059806  0.0022036943  0.015428084   0.016026685   -975.4603    0.004061431   0.0022036943    0.015295155      0.005062829    Float32[0.7609628,  0.27349025,    0.5037077]0.6508231,     0.3390255,     0.28728625]]]]]]
# sig_hail     3-4 0.0022382603  0.009228747  1869527   9358.0    1.7325788e6 0.004318088    0.0034092576  0.0049954024  0.031148706   0.032184917   0.0071835946 0.009115861   0.004995401     0.03078179       0.009413696    Float32[0.6508231,  0.3390255,     0.28728625]0.562786,     0.35293543,    -0.048666567]]]
# sig_hail     4-5 0.0047318107  0.020026933  955018    9334.0    886772.9    0.008787836    0.0060045337  0.009760632   0.054368727   0.05646796    0.014208983  0.016647581   0.009760633     0.05371758       0.01737455     Float32[0.562786,   0.35293543,    -0.048666567]0.63449466, 0.3219732,     0.13853967]5]
# sig_hail     5-6 0.009228747   1.0          464334    9249.0    433322.44   0.019193677    0.011113506   0.019969653   0.09513787    0.100518756   0.03933925   0.03493383    0.019969657     0.09425095       0.042098153    Float32[0.63449466, 0.3219732,     0.13853967]


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)




# CHECKING CALIBRATION

import Dates

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


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer_combined_with_sig_gated(); just_hours_near_storm_events = false);

length(combined_validation_forecasts) # 21318

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) # 17918

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])

# rm("combined_validation_forecasts_href_newer_with_sig_gated"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "combined_validation_forecasts_href_newer_with_sig_gated");


# Should be higher AU-PR compared to HREF-only

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = 1 + (feature_i - 1) % CombinedHREFSREF.models_with_gated_count
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(combined_validation_forecasts, X, Ys, weights)


# tornado (59115.0)                     feature 1 TORPROB:calculated:hour  fcst:calculated_prob:                 AU-PR-curve: 0.0352309693031683
# wind (496053.0)                       feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.12659473844432825
# hail (219394.0)                       feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.07890266947843802
# sig_tornado (8276.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.0302730314492093
# sig_wind (49971.0)                    feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.017744760586191517
# sig_hail (26492.0)                    feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.01715683030643232
# sig_tornado_gated_by_tornado (8276.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.03253630832265019
# sig_wind_gated_by_wind (49971.0)      feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.01775037517373651
# sig_hail_gated_by_hail (26492.0)      feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.017151095481579393


# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = 1 + (feature_i - 1) % CombinedHREFSREF.models_with_gated_count
    (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_count = 20
    per_bin_pos_weight = sum(Metrics.parallel_iterate(is -> sum(Float64.(view(y, is) .* view(weights, is))), length(y))) / bin_count

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

      println("$model_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(combined_validation_forecasts, X, Ys, weights)

# event_name                   mean_y                 mean_ŷ                 Σweight              bin_max
# tornado                      4.909524495630813e-6   5.191515395102615e-6   5.66145619557776e8   0.00010476272
# tornado                      0.00022469031038641833 0.00020099042848059528 1.236769066485542e7  0.00037851158
# tornado                      0.0005674733141041569  0.0005637955985021336  4.897452368723333e6  0.00083093136
# tornado                      0.0010339114103023036  0.0011079464905188305  2.6874630951256156e6 0.0014715802
# tornado                      0.0018007763606225193  0.0018067070480678542  1.5430840557836294e6 0.0022178409
# tornado                      0.0026662699028421736  0.0026251042335659488  1.0422987072908282e6 0.0031096356
# tornado                      0.0036644195076950227  0.003596269238020215   758449.9006308913    0.0041613844
# tornado                      0.004994389043670202   0.0047221119450420925  556472.1395524144    0.005374303
# tornado                      0.006046105994798436   0.006107017662622152   459616.75691252947   0.0069641164
# tornado                      0.007222832813996427   0.00793960311971317    384729.5133559704    0.00907007
# tornado                      0.009575136720431515   0.01019488141317389    290280.55604308844   0.011505889
# tornado                      0.01314788998274206    0.012825172163678709   211363.9715692401    0.014370477
# tornado                      0.016712132424945823   0.016042243137879462   166312.2046249509    0.018007137
# tornado                      0.02073197558506787    0.02024443627402786    134052.06174963713   0.022806687
# tornado                      0.025398914316501348   0.025528229674823166   109414.38995802402   0.02864382
# tornado                      0.03131543730204889    0.031994882107870055   88748.210755229      0.03581971
# tornado                      0.04013541823209907    0.039906192540074784   69236.57992362976    0.04479864
# tornado                      0.05228940224455693    0.050259619649040986   53153.04514402151    0.057085127
# tornado                      0.06579662494109705    0.06678279487709324    42233.66175991297    0.081110165
# tornado                      0.1132962527889193     0.11101718885796476    24443.49192237854    1.0
# wind                         4.088953468936942e-5   3.7374434323550545e-5  5.638919807811253e8  0.0012141336
# wind                         0.002060307043235168   0.0021439870453246208  1.1191455771811783e7 0.0036123027
# wind                         0.004743322906459139   0.004892487332639334   4.860998408443093e6  0.00654091
# wind                         0.007871034555283747   0.008076736375039471   2.929426821330011e6  0.009928602
# wind                         0.011498292343091354   0.011721546892808956   2.0052670575348735e6 0.0138297565
# wind                         0.01572113504735975    0.015944357635992352   1.4666691059827805e6 0.018396724
# wind                         0.02100007850403258    0.020826770430582568   1.0979641969723105e6 0.0235852
# wind                         0.026664370343634107   0.026332343894730156   864740.7443090081    0.029400615
# wind                         0.03304031402749679    0.03249701287726116    697856.5163927674    0.035946246
# wind                         0.03966916307860225    0.0395227856260389     581251.5709347129    0.043491296
# wind                         0.04717153683555902    0.04768103236487372    488802.3796027303    0.05236961
# wind                         0.05724855018820538    0.05716416244124216    402759.09756469727   0.062472206
# wind                         0.06879243350979725    0.06781654238095217    335176.99840939045   0.07365064
# wind                         0.0805492807497559     0.07969610259311684    286257.1162791252    0.08653024
# wind                         0.09639977738613353    0.0939557178915965     239182.65097749233   0.102372125
# wind                         0.11005506163858933    0.11196495422316873    209509.96524852514   0.12304668
# wind                         0.13318855236272686    0.13579569558320234    173121.14619898796   0.15103984
# wind                         0.16494968569948268    0.1690847014119584     139784.22123414278   0.19168633
# wind                         0.21804241737313282    0.2225099686774519     105747.1225142479    0.26699963
# wind                         0.3592259560337863     0.3475524349370022     64163.258621931076   1.0
# hail                         1.784833392294855e-5   1.7284255541299695e-5  5.677363417049105e8  0.00062108587
# hail                         0.0011066892439336356  0.0010609138095567213  9.156656073971331e6  0.001731631
# hail                         0.0023902634073214232  0.0023352386647921787  4.239196804862559e6  0.0031148507
# hail                         0.003955563516677259   0.0038464235474494606  2.56174464612633e6   0.004720867
# hail                         0.005525902243076726   0.005581591660339347   1.8337269915571809e6 0.006575822
# hail                         0.007506247762383691   0.007527177078251382   1.3499556241416335e6 0.0085918605
# hail                         0.009237152552849476   0.00965390763093254    1.0970247369663715e6 0.010842035
# hail                         0.011790877366011898   0.012107741452823569   859444.2391913533    0.013606852
# hail                         0.014909165722022492   0.015228473716722249   679686.6919781566    0.017115649
# hail                         0.01858521584360951    0.019171036518423168   545248.4603154063    0.021556474
# hail                         0.02417690052408013    0.023945080797254308   419143.3498419523    0.026622295
# hail                         0.030260462296422156   0.02926906125366914    334855.025233984     0.032190148
# hail                         0.03639607148470413    0.03514404722199651    278424.94663113356   0.038421202
# hail                         0.042802044482701265   0.041950645222632754   236745.43769025803   0.045988597
# hail                         0.04997596354376832    0.05064428054181894    202764.13558870554   0.0561735
# hail                         0.061091613929534355   0.0628210143710201     165873.0831478238    0.07101908
# hail                         0.07856662404408142    0.08092840376795538    128976.5599603653    0.0932154
# hail                         0.10442454227375592    0.10738546565910853    97040.91948205233    0.12529097
# hail                         0.15048834340756648    0.14716658773504213    67332.58431386948    0.17819661
# hail                         0.24139792418181497    0.23735632834047737    41932.916332006454   1.0
# sig_tornado                  6.807928518702428e-7   8.008436386317785e-7   5.840448544005358e8  3.7121936e-5
# sig_tornado                  9.624105786757293e-5   7.910610815693316e-5   4.1299648335719705e6 0.0001595747
# sig_tornado                  0.000213393697771958   0.0002710920176522494  1.8650008643648624e6 0.0004497414
# sig_tornado                  0.000508532952149081   0.0006078639277791341  782768.215783        0.00082816026
# sig_tornado                  0.0010221266095046448  0.0010439549649522524  389107.6992596388    0.0013458312
# sig_tornado                  0.0018415000392293357  0.0016616817030880877  216183.50383728743   0.00207863
# sig_tornado                  0.0026984328063015834  0.002535342040918522   147500.94541949034   0.0030903113
# sig_tornado                  0.0036278267950075735  0.003657455732641466   109655.57708358765   0.004319122
# sig_tornado                  0.004559346880970474   0.005043519286438838   87117.09662806988    0.0059117507
# sig_tornado                  0.0075190024512401706  0.006635893307448995   52898.63525056839    0.007462263
# sig_tornado                  0.009452533466449952   0.008317387541194977   42066.323425233364   0.009293684
# sig_tornado                  0.011007104114255147   0.010354646243126597   36117.72711759806    0.011533024
# sig_tornado                  0.012721084300392415   0.012804577475788838   31268.181021928787   0.014209412
# sig_tornado                  0.012886586770983704   0.015976488092028132   30897.496753275394   0.018063081
# sig_tornado                  0.01707692046417773    0.020308227299329885   23271.87638735771    0.0230411
# sig_tornado                  0.031069210248067098   0.025258120819381547   12784.5491502285     0.02789064
# sig_tornado                  0.034688599350273915   0.03142283636359688    11477.135940551758   0.035927918
# sig_tornado                  0.040260589274274124   0.04271335277286889    9885.582913160324    0.051557023
# sig_tornado                  0.06938058847789325    0.061163256875633426   5737.657340288162    0.07588489
# sig_tornado                  0.10854328681515067    0.116026603952553      3556.6319618225098   1.0
# sig_wind                     4.092532398828646e-6   3.7466167176191885e-6  5.654140980314947e8  9.686412e-5
# sig_wind                     0.00022301739501749714 0.00017049231024324075 1.037552037300241e7  0.0002877989
# sig_wind                     0.0003993922956961035  0.0004347330374413316  5.792542812906861e6  0.0006433771
# sig_wind                     0.000803582930506928   0.0008225005481373304  2.8799011543074846e6 0.0010435348
# sig_wind                     0.001289251512291442   0.001244911389015085   1.795137311993122e6  0.0014823369
# sig_wind                     0.001632167579231511   0.0017420505105459486  1.4178109238839746e6 0.0020469674
# sig_wind                     0.0022471540497130777  0.002352691404936583   1.029757000972569e6  0.0027147084
# sig_wind                     0.0028467922680133614  0.003130217998044419   812845.450830996     0.003644798
# sig_wind                     0.004314427487617052   0.0041587641185845394  536275.408403337     0.004783633
# sig_wind                     0.00540275566863462    0.005519073881905231   428316.2809947133    0.0063986047
# sig_wind                     0.0074765893353948435  0.007236739640090125   309547.6141542196    0.008155964
# sig_wind                     0.009275439738373019   0.009019682083931392   249451.77554541826   0.009944375
# sig_wind                     0.01117607284930777    0.010828415107917003   207071.58458650112   0.011780977
# sig_wind                     0.012542263928227331   0.012787637186756542   184464.19353044033   0.013897699
# sig_wind                     0.014522803496968049   0.015071113151600543   159357.67871630192   0.016380142
# sig_wind                     0.017020717110137855   0.0177909884145004     135967.9998921752    0.019395392
# sig_wind                     0.020032346986579443   0.021206187676619265   115491.52009189129   0.023335924
# sig_wind                     0.028609939014717894   0.02547111670679214    80864.79535990953    0.028111978
# sig_wind                     0.0390418522976499     0.03152531412401306    59260.28987890482    0.036329433
# sig_wind                     0.04757170209129554    0.04918902578071911    48432.73055690527    1.0
# sig_hail                     2.1184461373548715e-6  2.111554512923177e-6   5.815631749474444e8  0.00018269816
# sig_hail                     0.00035407967341619676 0.00028830781086434595 3.4816032499084473e6 0.00044366336
# sig_hail                     0.0006863437000655851  0.0005968701510109353  1.7962960997476578e6 0.00079615705
# sig_hail                     0.0010443201076317183  0.001003282602135149   1.1801299708802104e6 0.0012537035
# sig_hail                     0.0015499799612907852  0.0014832226841133785  794824.4544389248    0.0017434956
# sig_hail                     0.0019585545776370623  0.0019998836823327646  629213.1436849833    0.002283742
# sig_hail                     0.002328968841366939   0.0025766069936357123  529199.5968875289    0.002903462
# sig_hail                     0.002890756093091239   0.0032320814048040873  426231.62522023916   0.0036022607
# sig_hail                     0.0036388546545684953  0.003975163843463993   338626.4638938308    0.004397356
# sig_hail                     0.00528061692511125    0.004759615964849176   233299.1667084694    0.005159094
# sig_hail                     0.006098818792831802   0.0055776756177078855  202097.8886396289    0.0060394946
# sig_hail                     0.006851886809773846   0.006549088259826895   179865.73627787828   0.00711954
# sig_hail                     0.0074880201844427195  0.007780350960949351   164588.8226556182    0.008531243
# sig_hail                     0.009265344238045429   0.009326933201113687   132995.82367235422   0.010243598
# sig_hail                     0.011503866587506708   0.011257426012732974   107118.24134236574   0.01246264
# sig_hail                     0.013317657497552088   0.01397083985255275    92529.17651832104    0.015835665
# sig_hail                     0.017326957211522657   0.01804550265408542    71121.71249330044    0.020767273
# sig_hail                     0.023927410629695604   0.023838948340489385   51512.998446047306   0.027790405
# sig_hail                     0.03636306384038426    0.03225256377820693    33893.67565816641    0.038419217
# sig_hail                     0.0514520717027423     0.052959497497396466   23792.13910061121    1.0
# sig_tornado_gated_by_tornado 6.80752214418846e-7    7.984708279462003e-7   5.840797189134696e8  3.7121936e-5
# sig_tornado_gated_by_tornado 9.690487767823499e-5   7.915055787365186e-5   4.1016736624818444e6 0.0001595747
# sig_tornado_gated_by_tornado 0.00021396569295197056 0.0002711317850125941  1.860015151513338e6  0.0004497414
# sig_tornado_gated_by_tornado 0.0005092468206548616  0.000607894448972814   781670.9215949774    0.00082816026
# sig_tornado_gated_by_tornado 0.0010225153397958582  0.0010439905012083962  388959.77194416523   0.0013458312
# sig_tornado_gated_by_tornado 0.0018410370971024622  0.0016616980735043915  216237.864746809     0.00207863
# sig_tornado_gated_by_tornado 0.002695682535143632   0.0025353744154411003  147651.4333165884    0.0030903113
# sig_tornado_gated_by_tornado 0.003642082269045679   0.003654461576273891   109242.91847938299   0.0043129027
# sig_tornado_gated_by_tornado 0.004772927016253142   0.004996140633031925   83274.44155365229    0.0058075455
# sig_tornado_gated_by_tornado 0.007924401279962207   0.006466235399123103   50225.47098058462    0.0072081974
# sig_tornado_gated_by_tornado 0.009369570395461472   0.008017330015465535   42453.05718010664    0.008935298
# sig_tornado_gated_by_tornado 0.01086834581871548    0.009934699068537538   36597.71710562706    0.0110481
# sig_tornado_gated_by_tornado 0.01358842279585626    0.012129624823022767   29258.178030490875   0.013322137
# sig_tornado_gated_by_tornado 0.012583775953097367   0.01492178228895268    31637.540101647377   0.016752237
# sig_tornado_gated_by_tornado 0.01428270772886795    0.01905733611614729    27824.847390174866   0.021886041
# sig_tornado_gated_by_tornado 0.027911759651136462   0.0239779405184206     14239.913113296032   0.026487
# sig_tornado_gated_by_tornado 0.035089066949397583   0.02940164648514335    11343.90390264988    0.032996174
# sig_tornado_gated_by_tornado 0.03675299705712767    0.039052827927571046   10822.034985780716   0.04723703
# sig_tornado_gated_by_tornado 0.06610395046347117    0.05574938810259044    6015.441865742207    0.0685214
# sig_tornado_gated_by_tornado 0.11866669181269343    0.0941141141569582     3251.74998742342     1.0
# sig_wind_gated_by_wind       4.092053391209834e-6   3.6914296638530973e-6  5.65480284279534e8   9.686412e-5
# sig_wind_gated_by_wind       0.00022377326940816602 0.00017049336283508337 1.0340473335612476e7 0.0002877989
# sig_wind_gated_by_wind       0.00040071007982497435 0.0004347836745927817  5.773493327084124e6  0.0006433771
# sig_wind_gated_by_wind       0.0008053332243974976  0.0008225366780859049  2.873642039144814e6  0.0010435348
# sig_wind_gated_by_wind       0.0012907997836990707  0.001244933285482203   1.7929841044949293e6 0.0014823369
# sig_wind_gated_by_wind       0.0016334921231825157  0.0017420385530720161  1.4166612685803175e6 0.0020469674
# sig_wind_gated_by_wind       0.0022488892052048744  0.002352666412454072   1.0289624804993868e6 0.0027147084
# sig_wind_gated_by_wind       0.002848653370411195   0.003130204789854964   812314.3968834281    0.003644798
# sig_wind_gated_by_wind       0.004317260408986797   0.0041588450398372155  535923.5125433207    0.004783633
# sig_wind_gated_by_wind       0.005403786543143663   0.00551912463245403    428234.5715614557    0.0063986047
# sig_wind_gated_by_wind       0.0074781252496913244  0.0072367560546235075  309484.03690856695   0.008155964
# sig_wind_gated_by_wind       0.009281002924260073   0.009019707003804067   249302.2500460148    0.009944375
# sig_wind_gated_by_wind       0.011184808736313917   0.010828461955142962   206909.85147082806   0.011780977
# sig_wind_gated_by_wind       0.012551322215088      0.012787671697182638   184331.0657569766    0.013897699
# sig_wind_gated_by_wind       0.0145348453747516     0.015071063903490127   159225.65352845192   0.016380142
# sig_wind_gated_by_wind       0.017028046987239212   0.017791086006580054   135909.47123473883   0.019395392
# sig_wind_gated_by_wind       0.02003980355262268    0.021206192989088584   115448.54710841179   0.023335924
# sig_wind_gated_by_wind       0.02861765592628326    0.025471178137536445   80842.98971390724    0.028111978
# sig_wind_gated_by_wind       0.039045325215299576   0.03152535883624966    59255.01892507076    0.036329433
# sig_wind_gated_by_wind       0.04757170209129554    0.04918902578071911    48432.73055690527    1.0
# sig_hail_gated_by_hail       2.1183907525984074e-6  2.11154694840994e-6    5.815783797602363e8  0.00018269816
# sig_hail_gated_by_hail       0.00035444262034593283 0.0002882922621801058  3.4780381109054685e6 0.00044366336
# sig_hail_gated_by_hail       0.0006878480077930287  0.0005968718547488978  1.792367641610086e6  0.00079615705
# sig_hail_gated_by_hail       0.0010543476569257941  0.0010012039523897382  1.1685366286997795e6 0.0012486598
# sig_hail_gated_by_hail       0.0015579990950454706  0.0014761385441546288  791072.013310194     0.00173401
# sig_hail_gated_by_hail       0.0019746855283529425  0.001986917686328419   624009.7126327753    0.0022666713
# sig_hail_gated_by_hail       0.0023348558751944658  0.0025566107960778065  528051.7322846055    0.002880079
# sig_hail_gated_by_hail       0.002867164562617147   0.0032084198630481834  429725.4627920985    0.0035781637
# sig_hail_gated_by_hail       0.0035442099033917017  0.003958714509927273   347803.93252539635   0.0043907687
# sig_hail_gated_by_hail       0.005246242870832698   0.004755293453050818   234852.52580720186   0.005157499
# sig_hail_gated_by_hail       0.0061099027689653345  0.00557570428272982    201730.54388415813   0.006036837
# sig_hail_gated_by_hail       0.006832860313712971   0.00654786092494984    180345.55162775517   0.0071198633
# sig_hail_gated_by_hail       0.007476725439603289   0.007781976414077951   164839.16725814342   0.008534303
# sig_hail_gated_by_hail       0.00929324355537165    0.009328794003199066   132595.27121341228   0.010244046
# sig_hail_gated_by_hail       0.01150484019506094    0.011259206351416919   107118.80533397198   0.012465945
# sig_hail_gated_by_hail       0.013340835951931047   0.013972895930609991   92366.73931872845    0.015836215
# sig_hail_gated_by_hail       0.01733249477473337    0.018045832417394554   71092.70247608423    0.020767454
# sig_hail_gated_by_hail       0.02391365699480623    0.023841974755769883   51545.55974382162    0.027797807
# sig_hail_gated_by_hail       0.036387170846917005   0.03226146896355261    33871.604806125164   0.038429897
# sig_hail_gated_by_hail       0.05145753785009017    0.0529708532603202     23771.467127621174   1.0
