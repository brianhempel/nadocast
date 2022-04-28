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

# tornado (59115.0)    feature 1 TORPROB:calculated:hour    fcst:calculated_prob:blurred AU-PR-curve: 0.03499245922799354
# wind (496053.0)      feature 2 WINDPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.12382509696722048
# hail (219394.0)      feature 3 HAILPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.07635143069561279
# sig_tornado (8276.0) feature 4 STORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.031039683645313478
# sig_wind (49971.0)   feature 5 SWINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.017095480910378852
# sig_hail (26492.0)   feature 6 SHAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.01615547723934251
# tornado (59115.0)    feature 7 TORPROB:calculated:hour    fcst:calculated_prob:blurred AU-PR-curve: 0.01689145458378656
# wind (496053.0)      feature 8 WINDPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.08634946063569207
# hail (219394.0)      feature 9 HAILPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.051793667419186964
# sig_tornado (8276.0) feature 10 STORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.011523426169386659
# sig_wind (49971.0)   feature 11 SWINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011921143809491044
# sig_hail (26492.0)   feature 12 SHAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011681957729273851


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
# tornado 1.585627412719183e-5    1.5019125814091417e-5   5.841745838562961e8     0.0010219938
# tornado 0.0018442217259359015   0.002043351990263991    5.022179054546893e6     0.0040875063
# tornado 0.005677242293629276    0.006287977844174323    1.6314292145838141e6    0.00994226
# tornado 0.01271037441135178     0.014243580027931364    728735.0714734793       0.021133944
# tornado 0.02774871572294746     0.029169667083868162    333814.6034759879       0.042196225
# tornado 0.06549733600255886     0.06408604921822572     141373.13695544004      1.0
# event_to_bins["tornado"] = Float32[0.0010219938, 0.0040875063, 0.00994226, 0.021133944, 0.042196225, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# wind    0.00013231454071412768  0.00017320543761344177  5.808662757147758e8     0.008519708
# wind    0.012620625649047978    0.013532360481596287    6.0898617787950635e6    0.02144979
# wind    0.029560487388183062    0.029147533597453923    2.600012702262759e6     0.0403009
# wind    0.0570060645100827      0.05237697659648654     1.3482250527927876e6    0.0697437
# wind    0.10139155849458151     0.09121680817537046     758029.036770463        0.1250951
# wind    0.20787762723595077     0.1900328040852332      369710.6485797167       1.0
# event_to_bins["wind"] = Float32[0.008519708, 0.02144979, 0.0403009, 0.0697437, 0.1250951, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# hail    5.803764845083891e-5    5.648334104765748e-5    5.819789218923901e8     0.0033594805
# hail    0.005977448802913405    0.005624904375841729    5.650642658495367e6     0.009371412
# hail    0.014113686392710384    0.013360194329065028    2.3931470448597074e6    0.019450566
# hail    0.02976456211422143     0.026131477482661626    1.1347935315653682e6    0.036249824
# hail    0.05425490862039188     0.05023967433426355     622541.8256918192       0.074465364
# hail    0.13398386470863086     0.11588867485820345     252067.98182481527      1.0
# event_to_bins["hail"] = Float32[0.0033594805, 0.009371412, 0.019450566, 0.036249824, 0.074465364, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_tornado     2.243617223661975e-6    2.5111888881673344e-6   5.903153645486257e8     0.00070907583
# sig_tornado     0.0011275292899232076   0.0013797770428459425   1.1743939378501177e6    0.0027267952
# sig_tornado     0.004002549652098015    0.003991943624923849    330972.9277769923       0.006104077
# sig_tornado     0.010940756360980082    0.008126094307290591    121049.90981531143      0.011227695
# sig_tornado     0.01978916887653774     0.015541215881312074    66925.18020999432       0.023158755
# sig_tornado     0.05645818485996368     0.03709223891683685     23408.430941283703      1.0
# event_to_bins["sig_tornado"] = Float32[0.00070907583, 0.0027267952, 0.006104077, 0.011227695, 0.023158755, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_wind        1.323793558257507e-5    1.4931844985985406e-5   5.825447651738605e8     0.0007859762
# sig_wind        0.0013192507031081489   0.0013968328867966855   5.845750879078865e6     0.0024957801
# sig_wind        0.0037240467417506873   0.00358368519718438     2.0709835027632713e6    0.0052766525
# sig_wind        0.009433654517574771    0.00673985939864944     817515.2443544269       0.008785932
# sig_wind        0.015050016278023264    0.011525693385675191    512402.14104676247      0.015906833
# sig_wind        0.03202907914034734     0.02553244146781565     240697.99254828691      1.0
# event_to_bins["sig_wind"] = Float32[0.0007859762, 0.0024957801, 0.0052766525, 0.008785932, 0.015906833, 1.0]
# event_name      mean_y  mean_ŷ  Σweight bin_max
# sig_hail        6.996044184644087e-6    7.550323627800809e-6    5.87012376821348e8      0.00076262257
# sig_hail        0.0015150597675747512   0.001260221216050088    2.710856175330937e6     0.0020769562
# sig_hail        0.00348000107281502     0.00294361674162715     1.1801541500321627e6    0.004243912
# sig_hail        0.006355033234289453    0.005856178748939895    646304.4220900536       0.008325059
# sig_hail        0.011870985143387837    0.01179220396188794     345964.92858684063      0.017990546
# sig_hail        0.030074353009379738    0.02985484437148022     136458.43718630075      1.0
# event_to_bins["sig_hail"] = Float32[0.00076262257, 0.0020769562, 0.004243912, 0.008325059, 0.017990546, 1.0]

event_to_bins = Dict{String,Vector{Float32}}()
for prediction_i in 1:event_types_count
  (event_name, _) = CombinedHREFSREF.models[prediction_i]

  event_to_bins[event_name] = find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)

  println("event_to_bins[\"$event_name\"] = $(event_to_bins[event_name])")
end

println(event_to_bins)
# event_to_bins = Dict{String, Vector{Float32}}(
#   "tornado"     => [0.0010219938,  0.0040875063, 0.00994226,   0.021133944, 0.042196225, 1.0],
#   "wind"        => [0.008519708,   0.02144979,   0.0403009,    0.0697437,   0.1250951,   1.0],
#   "hail"        => [0.0033594805,  0.009371412,  0.019450566,  0.036249824, 0.074465364, 1.0],
#   "sig_tornado" => [0.00070907583, 0.0027267952, 0.006104077,  0.011227695, 0.023158755, 1.0],
#   "sig_wind"    => [0.0007859762,  0.0024957801, 0.0052766525, 0.008785932, 0.015906833, 1.0],
#   "sig_hail"    => [0.00076262257, 0.0020769562, 0.004243912,  0.008325059, 0.017990546, 1.0],
# )


# 4. combine bin-pairs (overlapping, 9 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + b)
# For the 2020 models, adding more terms resulted in dangerously large coefficients
# There's more data this year...try interaction terms this time?

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

# event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[0.88661367, 0.38997424, 2.2285037], [0.7429764, 0.2915021, 0.5264995], [0.6462555, 0.269669, -0.1717195], [0.65471405, 0.26225966, -0.16004708], [0.69204414, 0.29494107, 0.20776647]], "hail" => [[0.8667113, 0.21558651, 0.59692353], [0.90613884, 0.11804946, 0.23723379], [0.9837282, 0.15166068, 0.7282904], [0.7897085, 0.1952087, 0.2095446], [0.8960403, 0.2555574, 0.7325582]], "tornado" => [[0.68840486, 0.30125454, -0.08057284], [0.79298836, 0.2025301, -0.043320764], [0.87938726, 0.102382414, -0.13207436], [1.1286471, -0.042436406, 0.21654046], [1.162479, -0.12306028, 0.0014464383]], "sig_tornado" => [[0.70995516, 0.21704136, -0.6977991], [0.9985171, 0.32341582, 1.961638], [1.173152, 0.27435014, 2.5912948], [0.86561954, 0.26091522, 1.0081335], [1.1592792, 0.14715739, 1.6190162]], "sig_wind" => [[1.0086884, 0.10093948, 0.7754164], [0.9602558, 0.16216339, 0.8132331], [1.2342786, 0.24149285, 2.857296], [0.7123779, 0.2825677, 0.49686798], [0.6136547, 0.2704343, 0.039514035]], "wind" => [[0.9175508, 0.23498222, 0.6797868], [0.9369312, 0.23018824, 0.74334383], [0.97935945, 0.178703, 0.6830242], [0.94077724, 0.13174792, 0.40266746], [0.8477966, 0.192575, 0.37751102]])

# event_name      bin     HREF_ŷ_min      HREF_ŷ_max      count           pos_count       weight          mean_HREF_ŷ     mean_SREF_ŷ     mean_y          HREF_logloss    SREF_logloss    HREF_au_pr              SREF_au_pr              mean_logistic_ŷ logistic_logloss   logistic_au_pr          logistic_coeffs
# tornado         1-2     -1.0            0.0040875063    644019197       19963.0         5.891968e8      3.230817e-5     5.5548295e-5    3.1440843e-5    0.00026850903   0.00028634697   0.0018715903131512307   0.0013381063722501527   3.1440846e-5    0.00026608192      0.0020257205695958292   Float32[0.68840486, 0.30125454, -0.08057284]
# tornado         2-3     0.0010219938    0.00994226      7119763         19771.0         6.653608e6      0.0030841117    0.0031930539    0.002784058     0.018638004     0.019562287     0.005499835187763222    0.004483249438174986    0.0027840578    0.018551134        0.005717716477198319    Float32[0.79298836, 0.2025301, -0.043320764]
# tornado         3-4     0.0040875063    0.021133944     2503795         19664.0         2.3601645e6     0.008744385     0.0069750403    0.007848824     0.045096267     0.04765271      0.013243846930153366    0.009690594181229126    0.007848824     0.045008503        0.013003144560197612    Float32[0.87938726, 0.102382414, -0.13207436]
# tornado         4-5     0.00994226      0.042196225     1117496         19651.0         1.0625498e6     0.018932814     0.012481724     0.017434875     0.08616932      0.09292681      0.02789575345432789     0.019188749311691365    0.017434875     0.08608338         0.028179976129142997    Float32[1.1286471, -0.042436406, 0.21654046]
# tornado         5-6     0.021133944     1.0             495988          19488.0         475187.75       0.039557643     0.022317661     0.038979307     0.1593822       0.1759305       0.07477972641846233     0.04807290119083375     0.038979307     0.15920947         0.07587937619109285     Float32[1.162479, -0.12306028, 0.0014464383]
# wind            1-2     -1.0            0.02144979      641563634       165409.0        5.8695616e8     0.00031181102   0.00041765586   0.00026188482   0.0016646235    0.0018007546    0.012292847691764214    0.008354582425751898    0.00026188485   0.0016493445       0.013494734287446925    Float32[0.9175508, 0.23498222, 0.6797868]
# wind            2-3     0.008519708     0.0403009       9329149         165644.0        8.689874e6      0.018204425     0.014457053     0.017689036     0.08676839      0.09090827      0.02953172912170414     0.025918382738197552    0.017689038     0.08634487         0.03134325284756263     Float32[0.9369312, 0.23018824, 0.74334383]
# wind            3-4     0.02144979      0.0697437       4242288         165790.0        3.9482378e6     0.03707981      0.023979176     0.038932472     0.16174611      0.1737049       0.05775139578021173     0.04933141801378189     0.038932465     0.16123949         0.0590175344281342      Float32[0.97935945, 0.178703, 0.6830242]
# wind            4-5     0.0403009       0.1250951       2265419         165535.0        2.106254e6      0.06635521      0.036842573     0.07298016      0.2572716       0.2853647       0.10240282754514828     0.08805912846626071     0.072980165     0.25656432         0.1042215848015461      Float32[0.94077724, 0.13174792, 0.40266746]
# wind            5-6     0.0697437       1.0             1213058         164854.0        1.1277398e6     0.12361198      0.061248653     0.13630123      0.38255668      0.43768913      0.24257330837315685     0.2147188420849527      0.13630123      0.38071573         0.24810032780726798     Float32[0.8477966, 0.192575, 0.37751102]
# hail            1-2     -1.0            0.009371412     642258142       73198.0         5.876296e8      0.00011002924   0.00016634044   0.00011495866   0.0008140448    0.00087804714   0.005780121024352347    0.003879754063847663    0.00011495866   0.00080989196      0.006044946003297224    Float32[0.8667113, 0.21558651, 0.59692353]
# hail            2-3     0.0033594805    0.019450566     8680623         72766.0         8.04379e6       0.007926268     0.0070696026    0.0083981       0.04745978      0.050686143     0.014444863207148426    0.012138662559223776    0.008398099     0.04737983         0.015085062743419744    Float32[0.90613884, 0.11804946, 0.23723379]
# hail            3-4     0.009371412     0.036249824     3814289         72870.0         3.5279405e6     0.017468192     0.012716272     0.01914793      0.0930345       0.09960331      0.030087169509465578    0.025961393455167646    0.019147927     0.09274242         0.03129441447380458     Float32[0.9837282, 0.15166068, 0.7282904]
# hail            4-5     0.019450566     0.074465364     1903132         73296.0         1.7573354e6     0.034671884     0.02100555      0.03844035      0.16063517      0.1736584       0.05623625356158145     0.05047036140387735     0.038440347     0.15990426         0.05920379859585187     Float32[0.7897085, 0.1952087, 0.2095446]
# hail            5-6     0.036249824     1.0             946549          73326.0         874609.8        0.069160126     0.03508513      0.07723329      0.2598821       0.2895252       0.16074329238300952     0.13540067799403427     0.07723329      0.25796974         0.16802848535796258     Float32[0.8960403, 0.2555574, 0.7325582]
# sig_tornado     1-2     -1.0            0.0027267952    646454680       2826.0          5.914898e8      5.245729e-6     9.655999e-6     4.4778544e-6    4.2492007e-5    4.6929737e-5    0.0013797933289160474   0.0007310429454156993   4.477852e-6     4.219184e-5        0.001456871297447093    Float32[0.70995516, 0.21704136, -0.6977991]
# sig_tornado     2-3     0.00070907583   0.006104077     1586147         2756.0          1.5053668e6     0.001954093     0.0019215917    0.0017596371    0.012434961     0.0127947       0.0046784670976840765   0.0035464274816887067   0.0017596368    0.012288448        0.004949564545720826    Float32[0.9985171, 0.32341582, 1.961638]
# sig_tornado     3-4     0.0027267952    0.011227695     471174          2735.0          452022.8        0.0050990526    0.004033863     0.005860574     0.035132088     0.036447555     0.010898621524524702    0.008870196173143368    0.0058605727    0.03480286         0.011297957065245523    Float32[1.173152, 0.27435014, 2.5912948]
# sig_tornado     4-5     0.006104077     0.023158755     194650          2726.0          187975.08       0.010766118     0.006967447     0.014091078     0.073454954     0.07778949      0.022569175205536317    0.019918079705897564    0.014091076     0.07268114         0.023852890135584885    Float32[0.86561954, 0.26091522, 1.0081335]
# sig_tornado     5-6     0.011227695     1.0             93126           2715.0          90333.61        0.0211258       0.009777105     0.029291326     0.12738384      0.14625995      0.07003128211941761     0.03809884659261749     0.029291332     0.1255219          0.06930554459060641     Float32[1.1592792, 0.14715739, 1.6190162]
# sig_wind        1-2     -1.0            0.0024957801    643094271       16655.0         5.883905e8      2.8661243e-5    4.4492383e-5    2.6213374e-5    0.00022620204   0.00024705025   0.0013420768377580936   0.0009483212657347008   2.621337e-5     0.00022559414      0.001398093947437378    Float32[1.0086884, 0.10093948, 0.7754164]
# sig_wind        2-3     0.0007859762    0.0052766525    8517679         16604.0         7.916734e6      0.0019689042    0.0017556988    0.001948335     0.013772167     0.014377848     0.00376769337826039     0.0035534910816237756   0.0019483346    0.0137445945       0.004089402921135417    Float32[0.9602558, 0.16216339, 0.8132331]
# sig_wind        3-4     0.0024957801    0.008785932     3110029         16643.0         2.888499e6      0.0044769584    0.0031271225    0.005340004     0.032714304     0.034285918     0.009350093617459102    0.008486313325137957    0.005340003     0.03246235         0.010153034060007187    Float32[1.2342786, 0.24149285, 2.857296]
# sig_wind        4-5     0.0052766525    0.015906833     1434018         16662.0         1.3299174e6     0.008583788     0.004922084     0.011597575     0.06322082      0.067400396     0.01535557596564791     0.017460243457016595    0.011597574     0.062454045        0.01816696208265735     Float32[0.7123779, 0.2825677, 0.49686798]
# sig_wind        5-6     0.008785932     1.0             814680          16673.0         753100.1        0.016002383     0.00763637      0.020476686     0.09887696      0.10794477      0.032537395554224344    0.03099334659742581     0.02047669      0.09779582         0.034235658088816755    Float32[0.6136547, 0.2704343, 0.039514035]
# sig_hail        1-2     -1.0            0.0020769562    644525223       8837.0          5.8972326e8     1.3308636e-5    2.2646715e-5    1.3928354e-5    0.00011736053   0.00012436039   0.0015673274346260522   0.0011524686889429476   1.392835e-5     0.00011485079      0.0018032222022940506   Float32[0.88661367, 0.38997424, 2.2285037]
# sig_hail        2-3     0.00076262257   0.004243912     4203389         8834.0          3.8910102e6     0.0017707999    0.0016843328    0.002111032     0.014899518     0.015400251     0.0035349596803433466   0.0036159173491627406   0.0021110321    0.014752658        0.004071518132129405    Float32[0.7429764, 0.2915021, 0.5264995]
# sig_hail        3-4     0.0020769562    0.008325059     1975090         8872.0          1.8264585e6     0.0039742463    0.0030502854    0.00449735      0.0285625       0.029720012     0.006456449380413078    0.00633896214732802     0.0044973497    0.028355334        0.007016049879183336    Float32[0.6462555, 0.269669, -0.1717195]
# sig_hail        4-5     0.004243912     0.017990546     1072369         8850.0          992269.4        0.0079258345    0.005087355     0.0082782265    0.04743585      0.049682137     0.012126154954159479    0.011643616653725708    0.008278226     0.047176138        0.013226382183503055    Float32[0.65471405, 0.26225966, -0.16004708]
# sig_hail        5-6     0.008325059     1.0             518667          8783.0          482423.38       0.016901407     0.008992845     0.017019996     0.08375704      0.08861472      0.033123324512542504    0.030251523183823598    0.017019996     0.083278134        0.03560211600277626     Float32[0.69204414, 0.29494107, 0.20776647]



# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)


# CHECKING

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


(_, combined_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_href_newer_combined(); just_hours_near_storm_events = false);

length(combined_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2020, 11, 1, 0)
combined_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, combined_validation_forecasts);

length(combined_validation_forecasts) #

# Make sure a forecast loads
Forecasts.data(combined_validation_forecasts[100])


X, y, weights = TrainingShared.get_data_labels_weights(combined_validation_forecasts; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "combined_validation_forecasts_href_newer");

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



# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
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

      println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
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
end
test_calibration(combined_validation_forecasts, X, Ys, weights)

# event_name mean_y  mean_ŷ  Σweight bin_max







