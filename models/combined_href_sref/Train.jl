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

# tornado (59115.0)    feature 1 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.03499245922799354
# wind (496053.0)      feature 2 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.12382509696722048
# hail (219394.0)      feature 3 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.07635143069561279
# sig_tornado (8276.0) feature 4 STORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.031039683645313478
# sig_wind (49971.0)   feature 5 SWINDPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.017095480910378852
# sig_hail (26492.0)   feature 6 SHAILPRO:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.01615547723934251
# tornado (59115.0)    feature 7 TORPROB:calculated:hour   fcst:calculated_prob:blurred AU-PR-curve: 0.01689145458378656
# wind (496053.0)      feature 8 WINDPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.08634946063569207
# hail (219394.0)      feature 9 HAILPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.051793667419186964
# sig_tornado (8276.0) feature 10 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011523426169386659
# sig_wind (49971.0)   feature 11 SWINDPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011921143809491044
# sig_hail (26492.0)   feature 12 SHAILPRO:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.011681957729273851


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




