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

# tornado (59115.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.03546320685848304
# wind (496053.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.12622947359618547
# hail (219394.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07911189803221164
# sig_tornado (8276.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.031137889628780557
# sig_wind (49971.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.017785269909580678
# sig_hail (26492.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.01722716065254542


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

# event_name mean_y  mean_ŷ  Σweight bin_max
# tornado 4.909508673360088e-6    5.2336281690012724e-6   5.661311780489135e8     0.00010444016
# tornado 0.0002246678747338286   0.00019985272575460787  1.2368364105844617e7    0.00037566735
# tornado 0.0005703856964837563   0.0005574764095981105   4.872218566868663e6     0.00081882975
# tornado 0.0010318511527676348   0.0010911640437522765   2.6934920852622986e6    0.0014483026
# tornado 0.0017573022051952131   0.0017872458961313816   1.5812958945351243e6    0.0022057467
# tornado 0.0026281825294492533   0.002620468880824658    1.0575274634937644e6    0.0031169564
# tornado 0.00371392524348461     0.0036070438359030585   748318.6862217784       0.004177889
# tornado 0.005103927650611061    0.00474013280168609     544547.9929887056       0.0053932713
# tornado 0.006066594812638938    0.0061383666162111265   458052.2352371812       0.007005804
# tornado 0.0072239372143538105   0.007985479842395261    384686.3557930589       0.009108649
# tornado 0.009609619658142275    0.010213173879710111    289246.8598806262       0.011496777
# tornado 0.012921957826566762    0.01281618246656254     215034.2785077095       0.014362674
# tornado 0.016903743941954003    0.015990850332977827    164393.2752315998       0.017909829
# tornado 0.020388337049027573    0.020153838152070776    136300.70659780502      0.02269329
# tornado 0.026062950501473097    0.025278984227651725    106624.57057756186      0.028246038
# tornado 0.030374532698837404    0.03165345464631695     91508.78749078512       0.03558544
# tornado 0.03933216918920527     0.03982951970155387     70646.75243771076       0.044904735
# tornado 0.05290680637489835     0.05044720269042284     52532.162455797195      0.05738194
# tornado 0.06609077842576092     0.06737791880181056     42054.47014594078       0.08193591
# tornado 0.11493241256961831     0.11231599075498651     24091.63514727354       1.0
# wind    4.089542636706456e-5    3.7334529697000826e-5   5.63815053598428e8      0.0012082919
# wind    0.0020474744024357662   0.0021414255252892008   1.12616432189278e7      0.0036203987
# wind    0.004768365979311863    0.004898948581936374    4.835511594444752e6     0.0065414086
# wind    0.007941112369793897    0.008062588538951318    2.903607750895202e6     0.009889796
# wind    0.011397746471179719    0.011687095364314821    2.0229720231651664e6    0.013796491
# wind    0.015661236123797353    0.015883401809925462    1.4722469817139506e6    0.01829903
# wind    0.020750189491322627    0.020694702166231274    1.1112128586747646e6    0.023417838
# wind    0.026654319129809364    0.026100313833381617    865057.463750422        0.029113764
# wind    0.032843840307751664    0.03217826804141913     702034.6122245789       0.035613894
# wind    0.039292737053825195    0.039201239259360775    586813.2968890667       0.043214396
# wind    0.04681983343673512     0.04743907457295072     492480.2193476558       0.05216293
# wind    0.056934560637730544    0.05697893290179126     404977.8725990057       0.062285066
# wind    0.06868498549445301     0.06758104379697413     335702.99893701077      0.07334798
# wind    0.08066404077899515     0.07928042336575859     285851.0530347228       0.085990936
# wind    0.09559445420564451     0.09336369745596157     241200.7081951499       0.10172222
# wind    0.10883152939939415     0.11132043935058168     211864.71264117956      0.12244778
# wind    0.13259673562050406     0.13513610465806353     173892.11025846004      0.1502794
# wind    0.16713876979936595     0.1678796678578154      137955.2942302823       0.1899374
# wind    0.2162721291027469      0.22034155072833583     106611.5131379962       0.26388475
# wind    0.3522832546618264      0.34567878816126296     65425.04993611574       1.0
# hail    1.785346499182222e-5    1.7216331012464277e-5   5.675543717417934e8     0.000614193
# hail    0.0010902527680397359   0.0010523314452947853   9.29415695785606e6      0.0017225901
# hail    0.002363680577793649    0.0023268218603904626   4.286960402484238e6     0.0031091487
# hail    0.003950510553702029    0.003837727478567447    2.5651226577196717e6    0.004708258
# hail    0.005552532707645579    0.005560526840208284    1.8249040530560613e6    0.0065459087
# hail    0.007431656390126382    0.007507072557124522    1.3635549558832645e6    0.00858612
# hail    0.00921192061348823     0.009662092248803       1.0999923950878382e6    0.0108668925
# hail    0.011887066161731944    0.012138639045553722    852420.268047452        0.0136419125
# hail    0.014973657098500712    0.015273584448842048    676733.225140512        0.017169056
# hail    0.01873653943542186     0.019223459573896404    540835.829415977        0.021606132
# hail    0.024274927340314244    0.02399306026967015     417431.89138400555      0.026673349
# hail    0.030180052910147557    0.029319416855716932    335770.41272979975      0.032241385
# hail    0.03624551276204096     0.035196375355071426    279576.16907566786      0.03847061
# hail    0.04280938442208645     0.041965048738567684    236707.72773605585      0.045962103
# hail    0.04978505453319226     0.05060491706762454     203547.06775176525      0.056138206
# hail    0.0610812431841245      0.06279301682454272     165891.34158921242      0.07103089
# hail    0.07893657559511884     0.08098305270559146     128373.5009983778       0.09331748
# hail    0.10488362812392746     0.10752516117235328     96612.79625821114       0.12548476
# hail    0.15050641529684533     0.14756300932848895     67328.98230147362       0.17890006
# hail    0.24207225805490357     0.23864060452188726     41822.556034982204      1.0
# sig_tornado     6.806589461540568e-7    8.077043709918153e-7    5.841936880227618e8     3.7876714e-5
# sig_tornado     0.00010034983780655522  7.901418858126417e-5    3.963007770701468e6     0.00015656097
# sig_tornado     0.00020787099476198833  0.00027153940943750013  1.9120596334523559e6    0.00045916034
# sig_tornado     0.0005162773185999985   0.0006228018589952051   770691.5932626724       0.00085169065
# sig_tornado     0.0010610911894292075   0.0010675534630443094   375063.8114683032       0.0013645699
# sig_tornado     0.0018528370355426198   0.0016748949350922256   214624.4137350917       0.0020787085
# sig_tornado     0.0026960767645117637   0.0025216392327461045   147345.56681615114      0.0030551774
# sig_tornado     0.003607136020971087    0.0036133631476701206   110129.35502845049      0.0042697703
# sig_tornado     0.004633823704132394    0.004987393477963741    85917.07234799862       0.0058583817
# sig_tornado     0.007185732583843244    0.006659868742367659    55297.90495431423       0.0075879786
# sig_tornado     0.00984071281907092     0.008483416204306206    40369.56103348732       0.009502505
# sig_tornado     0.011155500185544513    0.010598628162884274    35653.33030194044       0.011811135
# sig_tornado     0.013184722350622943    0.013062034436241916    30166.0882743001        0.01442637
# sig_tornado     0.012582623318510676    0.016173029980271383    31570.558930575848      0.01820717
# sig_tornado     0.01816668817336865     0.020213352638238556    21876.063664972782      0.022651391
# sig_tornado     0.027992425919935737    0.025004181010273234    14218.141995191574      0.027881097
# sig_tornado     0.03554356248269728     0.03132648280171558     11197.542072951794      0.035704646
# sig_tornado     0.03825412447181251     0.04292112320670663     10402.310981750488      0.0527467
# sig_tornado     0.07026434906270337     0.06333813293801707     5665.211390912533       0.08056488
# sig_tornado     0.12236825500209796     0.1242460811409981      3170.980716228485       1.0
# sig_wind        4.093101799699436e-6    3.76491273117919e-6     5.653348295964116e8     9.6466e-5
# sig_wind        0.00022292022698300573  0.00016926273145608637  1.0379923975225449e7    0.00028498573
# sig_wind        0.00039855112851917965  0.0004299116488525444   5.80631694604373e6      0.0006352278
# sig_wind        0.0007950957089567412   0.0008131307453547058   2.910289165122211e6     0.0010327026
# sig_wind        0.0012740740722366835   0.0012342057997645408   1.8164337103256583e6    0.0014726059
# sig_wind        0.0016231799959237952   0.0017333449985840177   1.4255911833578348e6    0.0020407664
# sig_wind        0.0022223352152179157   0.002353497286441794    1.041205824581027e6     0.0027252333
# sig_wind        0.0028855981966146565   0.003142811367220506    801946.2915562391       0.003657357
# sig_wind        0.004336181001156264    0.004173554617194752    533610.8107416034       0.0047996407
# sig_wind        0.005424393307263525    0.005534684869479031    426613.8388929963       0.0064110085
# sig_wind        0.007541270786148612    0.007236945265191325    306840.1025774479       0.008138361
# sig_wind        0.009136753859322727    0.008997601909742452    253279.28343766928      0.009916732
# sig_wind        0.011103618331683452    0.010781612183091217    208353.63405650854      0.011710908
# sig_wind        0.012490291245462781    0.01269145055037114     185259.77551972866      0.013770169
# sig_wind        0.014304640015331832    0.014927083582023923    161789.28832399845      0.016226955
# sig_wind        0.016703575656264617    0.017657844678462934    138548.12072974443      0.019298919
# sig_wind        0.02012151704241398     0.021147038735956532    114983.82618129253      0.02333618
# sig_wind        0.028998727419648023    0.025528337565769346    79796.60013580322       0.028228598
# sig_wind        0.03955233310657378     0.03173091476560856     58503.07209199667       0.036637895
# sig_wind        0.0479880549754988      0.04983505325857353     47999.88598680496       1.0
# sig_hail        2.1196140618664013e-6   2.118571070180832e-6    5.816066919460453e8     0.0001848242
# sig_hail        0.0003566881718831543   0.00029137393763972334  3.454949631743908e6     0.00044785772
# sig_hail        0.0006941806793753607   0.0006009528802935747   1.775865816712916e6     0.00079958583
# sig_hail        0.001050996103124554    0.0010059657820231035   1.1726795455532074e6    0.0012546381
# sig_hail        0.00153169409468655     0.0014861601654033545   804793.0441725254       0.0017489267
# sig_hail        0.0019600158343093038   0.0020041759127180816   628804.2994360328       0.002285662
# sig_hail        0.002338930609350936    0.0025748761409921652   526948.6693937182       0.0028973122
# sig_hail        0.0029738982919583316   0.0032129455840046147   414450.3345935941       0.0035675233
# sig_hail        0.0034834672817721803   0.003952602602241154    353806.34349632263      0.004388927
# sig_hail        0.005279415394432615    0.004750394525167371    233456.88042610884      0.0051477696
# sig_hail        0.006205997613799235    0.005555065329758508    198618.2455305457       0.00600444
# sig_hail        0.006723843077294642    0.0065201746073158045   183354.8628909588       0.0071002156
# sig_hail        0.007615001278114909    0.007747711168162009    161780.44095534086      0.00848144
# sig_hail        0.00898721349987067     0.009297382909995773    137161.49506992102      0.010244141
# sig_hail        0.011839364435346045    0.011231406848602134    104092.55694550276      0.012400541
# sig_hail        0.013099056585345865    0.013922882280647647    94085.91249847412       0.01580819
# sig_hail        0.017085088293117483    0.01804258677162628     72125.52615255117       0.020797487
# sig_hail        0.02384834608076249     0.02391359009508504     51688.02511918545       0.02792928
# sig_hail        0.036770845484622984    0.03242767192637591     33516.32247233391       0.038694687
# sig_hail        0.05252987219787254     0.05355743703212033     23245.034498512745      1.0




