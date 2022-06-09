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
cutoff = Dates.DateTime(2022, 1, 1, 0)
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
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts, X, Ys, weights)

# tornado (59115.0)    feature 1 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.034704086268813064
# wind (496053.0)      feature 2 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.1241132253371296
# hail (219394.0)      feature 3 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.07606227745207711
# sig_tornado (8276.0) feature 4 STORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.030114357169682002
# sig_wind (49971.0)   feature 5 SWINDPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.017052198209466424
# sig_hail (26492.0)   feature 6 SHAILPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.01617295057987654
# tornado (59115.0)    feature 7 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.01676863287496511
# wind (496053.0)      feature 8 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.08701657639742678
# hail (219394.0)      feature 9 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.05173021678813705
# sig_tornado (8276.0) feature 10 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.010822770316928862
# sig_wind (49971.0)   feature 11 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.01190220119314299
# sig_hail (26492.0)   feature 12 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011592162809614752


# 3. bin HREF predictions into 6 bins of equal weight of positive labels
# (we will combine adjacent pairs of bins into 5 pairwise overlapping bins)

const bin_count = 6

function find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Float64.(y .* weights))
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

# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.5853845780849815e-5   1.4665829473152134e-5   5.842390048706219e8     0.0010191497
# tornado 0.0018758275077599695   0.002025201220248366    4.937834509230912e6     0.0040240395
# tornado 0.005600519006725456    0.006230061873508629    1.6538671399421692e6    0.009904478
# tornado 0.012712171252649868    0.014207834860984668    728607.9864167571       0.021118658
# tornado 0.027945641776707757    0.02912316890709706     331437.49752676487      0.04198461
# tornado 0.065504799747229       0.06368800909819429     141362.93397641182      1.0
# event_to_bins["tornado"] = Float32[0.0010191497, 0.0040240395, 0.009904478, 0.021118658, 0.04198461, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# wind    0.00013230315029579736  0.00016404450269311287  5.809175382802093e8     0.008220518
# wind    0.012637976789159223    0.013127873973669527    6.081484948420346e6     0.020898983
# wind    0.02994618218602642     0.028377467957785684    2.5665137857894897e6    0.039187722
# wind    0.05715574285076989     0.0509478486499908      1.344698019165635e6     0.067865305
# wind    0.1022050791847526      0.08860500789525311     751988.3415951729       0.1211984
# wind    0.20777837652386383     0.18371465748544782     369891.5583719611       1.0
# event_to_bins["wind"] = Float32[0.008220518, 0.020898983, 0.039187722, 0.067865305, 0.1211984, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# hail    5.8038285303576936e-5   5.481763241640796e-5    5.819686003189316e8     0.0033015092
# hail    0.00597859441566105     0.005550000339188726    5.649484517095208e6     0.00926606
# hail    0.014070325470540466    0.013228083800377637    2.400510007613957e6     0.019275624
# hail    0.02962727645482236     0.025932565350328854    1.1400262690879107e6    0.03602357
# hail    0.05444529116877569     0.049844054982542865    620376.261177957        0.07374218
# hail    0.13343220377277326     0.11505125263661221     253117.56104546785      1.0
# event_to_bins["hail"] = Float32[0.0033015092, 0.00926606, 0.019275624, 0.03602357, 0.07374218, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_tornado     2.2430040254628904e-6   2.363536737318448e-6    5.902880814281807e8     0.0006729238
# sig_tornado     0.001098651026937036    0.0013275839656400072   1.205945251418829e6     0.0026572358
# sig_tornado     0.004112196768859794    0.0038333969485659794   322130.7866101861       0.0057637123
# sig_tornado     0.010449253368173208    0.007730984302431663    126792.0179285407       0.010840383
# sig_tornado     0.020034660205244498    0.015037121095208644    66114.8152358532        0.02260619
# sig_tornado     0.05729313311718383     0.03645366498055018     23050.63596433401       1.0
# event_to_bins["sig_tornado"] = Float32[0.0006729238, 0.0026572358, 0.0057637123, 0.010840383, 0.02260619, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_wind        1.3237516197725251e-5   1.4948682319164419e-5   5.826146524773085e8     0.0007965934
# sig_wind        0.0013361292654010322   0.0014097659895180557   5.771563741550624e6     0.0025051977
# sig_wind        0.0037129240471194657   0.0036000411716636963   2.0771503483120203e6    0.0053047705
# sig_wind        0.009402917108808178    0.006778861843898213    820219.21224159 0.008843536
# sig_wind        0.015238186713475012    0.011556568501712744    506082.10583239794      0.015871514
# sig_wind        0.03179596516477986     0.025459276971537846    242447.04850304127      1.0
# event_to_bins["sig_wind"] = Float32[0.0007965934, 0.0025051977, 0.0053047705, 0.008843536, 0.015871514, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_hail        6.997020070755674e-6    7.663191920893168e-6    5.870142398132646e8     0.00077555457
# sig_hail        0.001508387513392702    0.0012871377944565622   2.722967291804552e6     0.002129781
# sig_hail        0.003507390927005381    0.003013697774036426    1.1708713273137212e6    0.004338203
# sig_hail        0.006392607983348745    0.005974296223819121    642522.9687299132       0.00847202
# sig_hail        0.011882942322179277    0.01201845846421457     345628.62186038494      0.018383306
# sig_hail        0.030195515768886273    0.030352783251670346    135884.91165059805      1.0
# event_to_bins["sig_hail"] = Float32[0.00077555457, 0.002129781, 0.004338203, 0.00847202, 0.018383306, 1.0]

event_to_bins = Dict{String,Vector{Float32}}()
for prediction_i in 1:event_types_count
  (event_name, _) = CombinedHREFSREF.models[prediction_i]

  event_to_bins[event_name] = find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)

  println("event_to_bins[\"$event_name\"] = $(event_to_bins[event_name])")
end

println(event_to_bins)
# href_newer_event_to_bins = Dict{String, Vector{Float32}}(
#   "tornado"     => [0.0010191497,  0.0040240395, 0.009904478,  0.021118658, 0.04198461,  1.0],
#   "wind"        => [0.008220518,   0.020898983,  0.039187722,  0.067865305, 0.1211984,   1.0],
#   "hail"        => [0.0033015092,  0.00926606,   0.019275624,  0.03602357,  0.07374218,  1.0],
#   "sig_tornado" => [0.0006729238,  0.0026572358, 0.0057637123, 0.010840383, 0.02260619,  1.0],
#   "sig_wind"    => [0.0007965934,  0.0025051977, 0.0053047705, 0.008843536, 0.015871514, 1.0],
#   "sig_hail"    => [0.00077555457, 0.002129781,  0.004338203,  0.00847202,  0.018383306, 1.0],
# )



# 4. combine bin-pairs (overlapping, 9 bins total)
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
      ("HREF_au_pr", Metrics.area_under_pr_curve(bin_href_x, bin_y, bin_weights)),
      ("SREF_au_pr", Metrics.area_under_pr_curve(bin_sref_x, bin_y, bin_weights)),
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

event_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:event_types_count
  event_name, _ = CombinedHREFSREF.models[prediction_i]

  event_to_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end
print("event_to_bins_logistic_coeffs = ")
println(event_to_bins_logistic_coeffs)

# href_newer_event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[0.87852937, 0.39478606, 2.16777], [0.76197445, 0.28871304, 0.59921116], [0.64860153, 0.26866677, -0.18315962], [0.63177335, 0.26839426, -0.25605413], [0.7029047, 0.28590807, 0.18754157]], "hail" => [[0.85908127, 0.22126952, 0.5908619], [0.89740646, 0.122179836, 0.22032435], [0.9773174, 0.15612426, 0.7210259], [0.7939177, 0.19984028, 0.23948279], [0.8816676, 0.26308796, 0.7197715]], "tornado" => [[0.6823556, 0.30390468, -0.09062567], [0.7804226, 0.20210582, -0.1086965], [0.9002674, 0.099383384, -0.052697014], [1.1373994, -0.0431747, 0.25296646], [1.1550109, -0.12750162, -0.030436214]], "sig_tornado" => [[0.71876323, 0.196306, -0.7678675], [1.0591443, 0.3063447, 2.2627401], [1.0977775, 0.2782542, 2.271161], [0.9274584, 0.23464419, 1.1744276], [1.1490777, 0.07541906, 1.2876196]], "sig_wind" => [[1.0125877, 0.09472051, 0.75418925], [0.9509606, 0.161822, 0.7489338], [1.219375, 0.25611165, 2.849027], [0.73210114, 0.29001468, 0.62654376], [0.605263, 0.2646362, -0.023003729]], "wind" => [[0.9084373, 0.23359907, 0.64916927], [0.95582205, 0.22434467, 0.813807], [0.9773524, 0.17435656, 0.68770856], [0.9580874, 0.12913947, 0.4690096], [0.85546756, 0.19877374, 0.44007877]])

# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count           pos_count       weight          mean_HREF_ŷ     mean_SREF_ŷ     mean_y          HREF_logloss    SREF_logloss    HREF_au_pr              SREF_au_pr              mean_logistic_ŷ  logistic_logloss  logistic_au_pr          logistic_coeffs
# tornado         1-2     -1.0            0.0040240395    643998077       19963.0         5.891768e8      3.151593e-5     5.5821667e-5    3.144211e-5     0.00026834625   0.00028614033   0.0018994257647184996   0.0013447936306566946   3.144211e-5      0.00026586952     0.0020437700863172206   Float32[0.6823556, 0.30390468, -0.09062567]
# tornado         2-3     0.0010191497    0.009904478     7052733         19776.0         6.5917015e6     0.0030802065    0.0032242509    0.002810358     0.018801307     0.019725308     0.005503036252784898    0.004500134832751219    0.0028103583     0.018717172       0.0057082723867126805   Float32[0.7804226, 0.20210582, -0.1086965]
# tornado         3-4     0.0040240395    0.021118658     2527402         19666.0         2.382475e6      0.008669823     0.006936116     0.0077754036    0.044705484     0.047223624     0.013198359570861117    0.009654524977606013    0.007775403      0.044621266       0.013009264222904746    Float32[0.9002674, 0.099383384, -0.052697014]
# tornado         4-5     0.009904478     0.04198461      1114816         19641.0         1.0600455e6     0.018871315     0.01237922      0.017475119     0.08629286      0.09297325      0.028137002688555665    0.019206517744441223    0.017475119      0.08621253        0.028409693628088577    Float32[1.1373994, -0.0431747, 0.25296646]
# tornado         5-6     0.021118658     1.0             493501          19486.0         472800.44       0.03945773      0.022138977     0.03917548      0.16014427      0.17667471      0.07384424455790504     0.04788957257048494     0.03917548       0.15997282        0.07512312752622156     Float32[1.1550109, -0.12750162, -0.030436214]
# wind            1-2     -1.0            0.020898983     641607434       165377.0        5.86999e8       0.00029835367   0.00044199397   0.00026186567   0.0016617663    0.0018099623    0.01234092935091646     0.008257848852985417    0.0002618657     0.0016492376      0.013441895041435233    Float32[0.9084373, 0.23359907, 0.64916927]
# wind            2-3     0.008220518     0.039187722     9287858         165635.0        8.647998e6      0.01765358      0.01510229      0.017774627     0.08703719      0.091080055     0.029972083572609217    0.02585954831375807     0.017774627      0.08664817        0.0316671298365058      Float32[0.95582205, 0.22434467, 0.813807]
# wind            3-4     0.020898983     0.067865305     4204348         165816.0        3.9112118e6     0.036137298     0.02487627      0.039300993     0.16304605      0.17443597      0.05796090719702702     0.04949872500292691     0.039301         0.16248578        0.05918285883230057     Float32[0.9773524, 0.17435656, 0.68770856]
# wind            4-5     0.039187722     0.1211984       2256028         165547.0        2.0966862e6     0.0644538       0.0377879       0.073312946     0.2583128       0.2850548       0.10335454568235197     0.08834046636466983     0.073312946      0.25734508        0.10496823698996803     Float32[0.9580874, 0.12913947, 0.4690096]
# wind            5-6     0.067865305     1.0             1207198         164860.0        1.1218799e6     0.119963326     0.062302932     0.13701333      0.38446698      0.43712497      0.2427347035309433      0.21733627368274624     0.13701333       0.38199314        0.24861761211957256     Float32[0.85546756, 0.19877374, 0.44007877]
# hail            1-2     -1.0            0.00926606      642245609       73191.0         5.876181e8      0.000107649474  0.00017543387   0.00011495976   0.00081366155   0.0008794475    0.005806473148823822    0.003911011911598805    0.00011495976    0.00080926676     0.006035025760521689    Float32[0.85908127, 0.22126952, 0.5908619]
# hail            2-3     0.0033015092    0.019275624     8689352         72771.0         8.049995e6      0.007839606     0.0074039474    0.008391549     0.0474396       0.0505257       0.014409157958231049    0.012144250781692673    0.00839155       0.047350965       0.015089767937298118    Float32[0.89740646, 0.122179836, 0.22032435]
# hail            3-4     0.00926606      0.03602357      3828140         72875.0         3.5405362e6     0.017318832     0.013243247     0.019079547     0.09278079      0.098914534     0.029825938520212755    0.025934569238406487    0.019079547      0.09247384        0.031159219037881308    Float32[0.9773174, 0.15612426, 0.7210259]
# hail            4-5     0.019275624     0.07374218      1906354         73298.0         1.7604025e6     0.034359116     0.021750756     0.038373295     0.16042192      0.17255662      0.05614250049163809     0.05047980725326656     0.03837329       0.15964882        0.059053271456118314    Float32[0.7939177, 0.19984028, 0.23948279]
# hail            5-6     0.03602357      1.0             945231          73328.0         873493.75       0.06873954      0.036233604     0.077333815     0.2603757       0.28820503      0.15993880824010007     0.13521883674419635     0.077333815      0.25836176        0.1674334504860149      Float32[0.8816676, 0.26308796, 0.7197715]
# sig_tornado     1-2     -1.0            0.0026572358    646459114       2827.0          5.91494e8       5.0654126e-6    9.787601e-6     4.478374e-6     4.2476546e-5    4.7251506e-5    0.001390857989917192    0.0007006668591461548   4.478371e-6      4.223787e-5       0.001413204121156483    Float32[0.71876323, 0.196306, -0.7678675]
# sig_tornado     2-3     0.0006729238    0.0057637123    1610300         2758.0          1.528076e6      0.0018558296    0.0019180593    0.0017339309    0.012248282     0.012661073     0.004848176068225184    0.003605447156013654    0.0017339309     0.012107566       0.005146425590905232    Float32[1.0591443, 0.3063447, 2.2627401]
# sig_tornado     3-4     0.0026572358    0.010840383     467930          2735.0          448922.8        0.0049342164    0.004054436     0.0059020105    0.035455637     0.03674795      0.010471666395553985    0.008812155736206381    0.005902011      0.035104904       0.010863048231550973    Float32[1.0977775, 0.2782542, 2.271161]
# sig_tornado     4-5     0.0057637123    0.02260619      199799          2726.0          192906.83       0.010235012     0.0068527316    0.013734452     0.07187739      0.076171055     0.023221844966128686    0.01868492612584677     0.0137344515     0.071091264       0.02350625945750735     Float32[0.9274584, 0.23464419, 1.1744276]
# sig_tornado     5-6     0.010840383     1.0             91936           2714.0          89165.45        0.020573625     0.009884753     0.029666549     0.12928595      0.14859094      0.06736539163620504     0.036542642563452334    0.02966655       0.12724549        0.06686577916380419     Float32[1.1490777, 0.07541906, 1.2876196]
# sig_wind        1-2     -1.0            0.0025051977    643089605       16654.0         5.883862e8      2.8630644e-5    4.561592e-5     2.621395e-5     0.0002260067    0.00024754542   0.0013533315763700532   0.0009280472587145814   2.6213946e-5     0.00022543456     0.0013928504945990583   Float32[1.0125877, 0.09472051, 0.75418925]
# sig_wind        2-3     0.0007965934    0.0053047705    8445178         16604.0         7.848714e6      0.001989419     0.0018002244    0.0019651444    0.013882952     0.014484747     0.0037498490641232492   0.0035939856633062147   0.001965144      0.01385598        0.004103035487697642    Float32[0.9509606, 0.161822, 0.7489338]
# sig_wind        3-4     0.0025051977    0.008843536     3119611         16643.0         2.8973695e6     0.004499937     0.0031839565    0.00532371      0.032624483     0.034090415     0.00929273149324396     0.008609796879122177    0.0053237104     0.03236891        0.010189279478531767    Float32[1.219375, 0.25611165, 2.849027]
# sig_wind        4-5     0.0053047705    0.015871514     1430128         16661.0         1.3263012e6     0.00860191      0.004990342     0.011629505     0.06334448      0.06740536      0.015469076591543823    0.017507269258887924    0.011629505      0.06256348        0.01823391773498205     Float32[0.73210114, 0.29001468, 0.62654376]
# sig_wind        5-6     0.008843536     1.0             809764          16674.0         748529.2        0.016059628     0.0077364068    0.020601215     0.099445686     0.10848617      0.03237356471304926     0.03087136471337906     0.020601217      0.09835428        0.0340394059009364      Float32[0.605263, 0.2646362, -0.023003729]
# sig_hail        1-2     -1.0            0.002129781     644540729       8839.0          5.8973715e8     1.3570855e-5    2.3609186e-5    1.3929325e-5    0.00011739412   0.00012469696   0.0015535922375803084   0.0011578536548521685   1.3929323e-5     0.00011489496     0.0018063724211076868   Float32[0.87852937, 0.39478606, 2.16777]
# sig_hail        2-3     0.00077555457   0.004338203     4206333         8836.0          3.8938388e6     0.0018063117    0.0017452716    0.002109485     0.014874626     0.015372732     0.0036503947899569696   0.003625311188441475    0.0021094845     0.014737985       0.00412343288143834     Float32[0.76197445, 0.28871304, 0.59921116]
# sig_hail        3-4     0.002129781     0.00847202      1960613         8870.0          1.8133942e6     0.004062699     0.0031648867    0.004529683     0.028728474     0.029853879     0.006420026918117442    0.006440766444222495    0.0045296834     0.02853113        0.0070736421213559685   Float32[0.64860153, 0.26866677, -0.18315962]
# sig_hail        4-5     0.004338203     0.018383306     1067644         8849.0          988151.6        0.00808838      0.0052774907    0.008312978     0.047621794     0.049718276     0.012064163900399652    0.01177565085754454     0.008312978      0.047354605       0.013214184548086882    Float32[0.63177335, 0.26839426, -0.25605413]
# sig_hail        5-6     0.00847202      1.0             517638          8783.0          481513.53       0.017192475     0.009289718     0.017050818     0.08388308      0.08858308      0.03314034375319347     0.029780518305499937    0.01705082       0.08343667        0.03534250681913907     Float32[0.7029047, 0.28590807, 0.18754157]


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


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer_combined(); just_hours_near_storm_events = false);

length(combined_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) # 17918

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])

# rm("combined_validation_forecasts_href_newer"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "combined_validation_forecasts_href_newer");

event_types_count = length(CombinedHREFSREF.models)


# Should be higher AU-PR compared to HREF-only

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = 1 + (feature_i - 1) % event_types_count
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(combined_validation_forecasts, X, Ys, weights)


# tornado (59115.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.0352309693031683
# wind (496053.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.12659473844432825
# hail (219394.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07890266947843802
# sig_tornado (8276.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.0302730314492093
# sig_wind (49971.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.017744760586191517
# sig_hail (26492.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.01715683030643232


# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = 1 + (feature_i - 1) % event_types_count
    (event_name, _) = CombinedHREFSREF.models[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

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

      println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(combined_validation_forecasts, X, Ys, weights)

# event_name  mean_y                 mean_ŷ                 Σweight              bin_max
# tornado     4.909524495630813e-6   5.191515395102615e-6   5.66145619557776e8   0.00010476272
# tornado     0.00022469031038641833 0.00020099042848059528 1.236769066485542e7  0.00037851158
# tornado     0.0005674733141041569  0.0005637955985021336  4.897452368723333e6  0.00083093136
# tornado     0.0010339114103023036  0.0011079464905188305  2.6874630951256156e6 0.0014715802
# tornado     0.0018007763606225193  0.0018067070480678542  1.5430840557836294e6 0.0022178409
# tornado     0.0026662699028421736  0.0026251042335659488  1.0422987072908282e6 0.0031096356
# tornado     0.0036644195076950227  0.003596269238020215   758449.9006308913    0.0041613844
# tornado     0.004994389043670202   0.0047221119450420925  556472.1395524144    0.005374303
# tornado     0.006046105994798436   0.006107017662622152   459616.75691252947   0.0069641164
# tornado     0.007222832813996427   0.00793960311971317    384729.5133559704    0.00907007
# tornado     0.009575136720431515   0.01019488141317389    290280.55604308844   0.011505889
# tornado     0.01314788998274206    0.012825172163678709   211363.9715692401    0.014370477
# tornado     0.016712132424945823   0.016042243137879462   166312.2046249509    0.018007137
# tornado     0.02073197558506787    0.02024443627402786    134052.06174963713   0.022806687
# tornado     0.025398914316501348   0.025528229674823166   109414.38995802402   0.02864382
# tornado     0.03131543730204889    0.031994882107870055   88748.210755229      0.03581971
# tornado     0.04013541823209907    0.039906192540074784   69236.57992362976    0.04479864
# tornado     0.05228940224455693    0.050259619649040986   53153.04514402151    0.057085127
# tornado     0.06579662494109705    0.06678279487709324    42233.66175991297    0.081110165
# tornado     0.1132962527889193     0.11101718885796476    24443.49192237854    1.0
# wind        4.088953468936942e-5   3.7374434323550545e-5  5.638919807811253e8  0.0012141336
# wind        0.002060307043235168   0.0021439870453246208  1.1191455771811783e7 0.0036123027
# wind        0.004743322906459139   0.004892487332639334   4.860998408443093e6  0.00654091
# wind        0.007871034555283747   0.008076736375039471   2.929426821330011e6  0.009928602
# wind        0.011498292343091354   0.011721546892808956   2.0052670575348735e6 0.0138297565
# wind        0.01572113504735975    0.015944357635992352   1.4666691059827805e6 0.018396724
# wind        0.02100007850403258    0.020826770430582568   1.0979641969723105e6 0.0235852
# wind        0.026664370343634107   0.026332343894730156   864740.7443090081    0.029400615
# wind        0.03304031402749679    0.03249701287726116    697856.5163927674    0.035946246
# wind        0.03966916307860225    0.0395227856260389     581251.5709347129    0.043491296
# wind        0.04717153683555902    0.04768103236487372    488802.3796027303    0.05236961
# wind        0.05724855018820538    0.05716416244124216    402759.09756469727   0.062472206
# wind        0.06879243350979725    0.06781654238095217    335176.99840939045   0.07365064
# wind        0.0805492807497559     0.07969610259311684    286257.1162791252    0.08653024
# wind        0.09639977738613353    0.0939557178915965     239182.65097749233   0.102372125
# wind        0.11005506163858933    0.11196495422316873    209509.96524852514   0.12304668
# wind        0.13318855236272686    0.13579569558320234    173121.14619898796   0.15103984
# wind        0.16494968569948268    0.1690847014119584     139784.22123414278   0.19168633
# wind        0.21804241737313282    0.2225099686774519     105747.1225142479    0.26699963
# wind        0.3592259560337863     0.3475524349370022     64163.258621931076   1.0
# hail        1.784833392294855e-5   1.7284255541299695e-5  5.677363417049105e8  0.00062108587
# hail        0.0011066892439336356  0.0010609138095567213  9.156656073971331e6  0.001731631
# hail        0.0023902634073214232  0.0023352386647921787  4.239196804862559e6  0.0031148507
# hail        0.003955563516677259   0.0038464235474494606  2.56174464612633e6   0.004720867
# hail        0.005525902243076726   0.005581591660339347   1.8337269915571809e6 0.006575822
# hail        0.007506247762383691   0.007527177078251382   1.3499556241416335e6 0.0085918605
# hail        0.009237152552849476   0.00965390763093254    1.0970247369663715e6 0.010842035
# hail        0.011790877366011898   0.012107741452823569   859444.2391913533    0.013606852
# hail        0.014909165722022492   0.015228473716722249   679686.6919781566    0.017115649
# hail        0.01858521584360951    0.019171036518423168   545248.4603154063    0.021556474
# hail        0.02417690052408013    0.023945080797254308   419143.3498419523    0.026622295
# hail        0.030260462296422156   0.02926906125366914    334855.025233984     0.032190148
# hail        0.03639607148470413    0.03514404722199651    278424.94663113356   0.038421202
# hail        0.042802044482701265   0.041950645222632754   236745.43769025803   0.045988597
# hail        0.04997596354376832    0.05064428054181894    202764.13558870554   0.0561735
# hail        0.061091613929534355   0.0628210143710201     165873.0831478238    0.07101908
# hail        0.07856662404408142    0.08092840376795538    128976.5599603653    0.0932154
# hail        0.10442454227375592    0.10738546565910853    97040.91948205233    0.12529097
# hail        0.15048834340756648    0.14716658773504213    67332.58431386948    0.17819661
# hail        0.24139792418181497    0.23735632834047737    41932.916332006454   1.0
# sig_tornado 6.807928518702428e-7   8.008436386317785e-7   5.840448544005358e8  3.7121936e-5
# sig_tornado 9.624105786757293e-5   7.910610815693316e-5   4.1299648335719705e6 0.0001595747
# sig_tornado 0.000213393697771958   0.0002710920176522494  1.8650008643648624e6 0.0004497414
# sig_tornado 0.000508532952149081   0.0006078639277791341  782768.215783        0.00082816026
# sig_tornado 0.0010221266095046448  0.0010439549649522524  389107.6992596388    0.0013458312
# sig_tornado 0.0018415000392293357  0.0016616817030880877  216183.50383728743   0.00207863
# sig_tornado 0.0026984328063015834  0.002535342040918522   147500.94541949034   0.0030903113
# sig_tornado 0.0036278267950075735  0.003657455732641466   109655.57708358765   0.004319122
# sig_tornado 0.004559346880970474   0.005043519286438838   87117.09662806988    0.0059117507
# sig_tornado 0.0075190024512401706  0.006635893307448995   52898.63525056839    0.007462263
# sig_tornado 0.009452533466449952   0.008317387541194977   42066.323425233364   0.009293684
# sig_tornado 0.011007104114255147   0.010354646243126597   36117.72711759806    0.011533024
# sig_tornado 0.012721084300392415   0.012804577475788838   31268.181021928787   0.014209412
# sig_tornado 0.012886586770983704   0.015976488092028132   30897.496753275394   0.018063081
# sig_tornado 0.01707692046417773    0.020308227299329885   23271.87638735771    0.0230411
# sig_tornado 0.031069210248067098   0.025258120819381547   12784.5491502285     0.02789064
# sig_tornado 0.034688599350273915   0.03142283636359688    11477.135940551758   0.035927918
# sig_tornado 0.040260589274274124   0.04271335277286889    9885.582913160324    0.051557023
# sig_tornado 0.06938058847789325    0.061163256875633426   5737.657340288162    0.07588489
# sig_tornado 0.10854328681515067    0.116026603952553      3556.6319618225098   1.0
# sig_wind    4.092532398828646e-6   3.7466167176191885e-6  5.654140980314947e8  9.686412e-5
# sig_wind    0.00022301739501749714 0.00017049231024324075 1.037552037300241e7  0.0002877989
# sig_wind    0.0003993922956961035  0.0004347330374413316  5.792542812906861e6  0.0006433771
# sig_wind    0.000803582930506928   0.0008225005481373304  2.8799011543074846e6 0.0010435348
# sig_wind    0.001289251512291442   0.001244911389015085   1.795137311993122e6  0.0014823369
# sig_wind    0.001632167579231511   0.0017420505105459486  1.4178109238839746e6 0.0020469674
# sig_wind    0.0022471540497130777  0.002352691404936583   1.029757000972569e6  0.0027147084
# sig_wind    0.0028467922680133614  0.003130217998044419   812845.450830996     0.003644798
# sig_wind    0.004314427487617052   0.0041587641185845394  536275.408403337     0.004783633
# sig_wind    0.00540275566863462    0.005519073881905231   428316.2809947133    0.0063986047
# sig_wind    0.0074765893353948435  0.007236739640090125   309547.6141542196    0.008155964
# sig_wind    0.009275439738373019   0.009019682083931392   249451.77554541826   0.009944375
# sig_wind    0.01117607284930777    0.010828415107917003   207071.58458650112   0.011780977
# sig_wind    0.012542263928227331   0.012787637186756542   184464.19353044033   0.013897699
# sig_wind    0.014522803496968049   0.015071113151600543   159357.67871630192   0.016380142
# sig_wind    0.017020717110137855   0.0177909884145004     135967.9998921752    0.019395392
# sig_wind    0.020032346986579443   0.021206187676619265   115491.52009189129   0.023335924
# sig_wind    0.028609939014717894   0.02547111670679214    80864.79535990953    0.028111978
# sig_wind    0.0390418522976499     0.03152531412401306    59260.28987890482    0.036329433
# sig_wind    0.04757170209129554    0.04918902578071911    48432.73055690527    1.0
# sig_hail    2.1184461373548715e-6  2.111554512923177e-6   5.815631749474444e8  0.00018269816
# sig_hail    0.00035407967341619676 0.00028830781086434595 3.4816032499084473e6 0.00044366336
# sig_hail    0.0006863437000655851  0.0005968701510109353  1.7962960997476578e6 0.00079615705
# sig_hail    0.0010443201076317183  0.001003282602135149   1.1801299708802104e6 0.0012537035
# sig_hail    0.0015499799612907852  0.0014832226841133785  794824.4544389248    0.0017434956
# sig_hail    0.0019585545776370623  0.0019998836823327646  629213.1436849833    0.002283742
# sig_hail    0.002328968841366939   0.0025766069936357123  529199.5968875289    0.002903462
# sig_hail    0.002890756093091239   0.0032320814048040873  426231.62522023916   0.0036022607
# sig_hail    0.0036388546545684953  0.003975163843463993   338626.4638938308    0.004397356
# sig_hail    0.00528061692511125    0.004759615964849176   233299.1667084694    0.005159094
# sig_hail    0.006098818792831802   0.0055776756177078855  202097.8886396289    0.0060394946
# sig_hail    0.006851886809773846   0.006549088259826895   179865.73627787828   0.00711954
# sig_hail    0.0074880201844427195  0.007780350960949351   164588.8226556182    0.008531243
# sig_hail    0.009265344238045429   0.009326933201113687   132995.82367235422   0.010243598
# sig_hail    0.011503866587506708   0.011257426012732974   107118.24134236574   0.01246264
# sig_hail    0.013317657497552088   0.01397083985255275    92529.17651832104    0.015835665
# sig_hail    0.017326957211522657   0.01804550265408542    71121.71249330044    0.020767273
# sig_hail    0.023927410629695604   0.023838948340489385   51512.998446047306   0.027790405
# sig_hail    0.03636306384038426    0.03225256377820693    33893.67565816641    0.038419217
# sig_hail    0.0514520717027423     0.052959497497396466   23792.13910061121    1.0
