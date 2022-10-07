import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grid130
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 716

validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_forecasts);
length(validation_forecasts_0z_12z) # 358

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

dec11 = validation_forecasts_0z_12z[309]; Forecasts.time_title(dec11) # "2021-12-11 00Z +35"
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  prediction_i = div(i - 1, 2) + 1
  event_name, _, _, _, _ = HREFPrediction.models[prediction_i]
  PlotMap.plot_debug_map("dec11_0z_day_accs_$(i)_$event_name", dec11.grid, dec11_data[:,i]);
end
for (event_name, labeler) in TrainingShared.event_name_to_day_labeler
  dec11_labels = labeler(dec11);
  PlotMap.plot_debug_map("dec11_0z_day_$event_name", dec11.grid, dec11_labels);
end
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_1_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_2_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_3_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_4_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_5_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_6_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_7_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_8_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_9_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_10_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_11_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_12_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_13_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_14_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_15_sig_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_accs_16_sig_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_hail.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_sig_tornado.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_sig_wind.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_sig_wind_adj.pdf ./
# scp nadocaster:/home/brian/nadocast_dev/models/href_prediction/dec11_0z_day_sig_hail.pdf ./

# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    event_name, _ = HREFPrediction.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z_12z, X, Ys, weights)

# tornado (20606.0)       feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.15398282
# tornado (20606.0)       feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.14738984
# wind (148934.0)         feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.40272897
# wind (148934.0)         feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.3743683
# wind_adj (53051.016)    feature 5 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.24277368
# wind_adj (53051.016)    feature 6 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.22707964
# hail (67838.0)          feature 7 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.2654488
# hail (67838.0)          feature 8 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.24634801
# sig_tornado (2681.0)    feature 9 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.08502455
# sig_tornado (2681.0)    feature 10 highest hourly STORPROB:calculated:day           fcst:: AU-PR-curve: 0.09089609 (exception)
# sig_wind (17640.0)      feature 11 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.08987321
# sig_wind (17640.0)      feature 12 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.08219389
# sig_wind_adj (6668.761) feature 13 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.09385775
# sig_wind_adj (6668.761) feature 14 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.099324994 (exception)
# sig_hail (9334.0)       feature 15 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.08321426
# sig_hail (9334.0)       feature 16 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.07902908

accs_event_names = map(i -> HREFPrediction.models[div(i - 1, 2) + 1][1], 1:size(X,2))
Metrics.reliability_curves_midpoints(20, X, Ys, accs_event_names, weights, map(i -> accs_event_names[i] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2)))
# ŷ_tornado_tot,y_tornado_tot,ŷ_tornado_max,y_tornado_max,ŷ_wind_tot,y_wind_tot,ŷ_wind_max,y_wind_max,ŷ_wind_adj_tot,y_wind_adj_tot,ŷ_wind_adj_max,y_wind_adj_max,ŷ_hail_tot,y_hail_tot,ŷ_hail_max,y_hail_max,ŷ_sig_tornado_tot,y_sig_tornado_tot,ŷ_sig_tornado_max,y_sig_tornado_max,ŷ_sig_wind_tot,y_sig_wind_tot,ŷ_sig_wind_max,y_sig_wind_max,ŷ_sig_wind_adj_tot,y_sig_wind_adj_tot,ŷ_sig_wind_adj_max,y_sig_wind_adj_max,ŷ_sig_hail_tot,y_sig_hail_tot,ŷ_sig_hail_max,y_sig_hail_max,
# 0.00014385104,0.00011868292,2.4220833e-5,0.00011878335,0.0011431412,0.00087078236,0.00023239148,0.0008736091,0.0003626865,0.00030150518,7.363922e-5,0.00030179956,0.0005521953,0.00038311855,0.000114642084,0.0003835346,1.3997788e-5,1.467714e-5,2.0934851e-6,1.4788282e-5,0.00010182712,0.00010140307,1.9537485e-5,0.00010119399,4.3060078e-5,3.6887155e-5,7.335474e-6,3.6960217e-5,6.471027e-5,5.083026e-5,1.2484874e-5,5.1080136e-5,
# 0.003032044,0.0029303133,0.0006838064,0.002877402,0.029512031,0.022580715,0.0066744853,0.02236152,0.009259769,0.008240907,0.0021139528,0.008677032,0.017259995,0.01387913,0.0038320096,0.014695416,0.00054738147,0.0010956898,0.00012477508,0.0012453337,0.002540408,0.0026084806,0.0006060828,0.0027337421,0.0013224183,0.0008776949,0.000249588,0.0009950559,0.0037667723,0.002645468,0.00085008674,0.0026333542,
# 0.007449875,0.005211285,0.0017348687,0.005909616,0.05403744,0.042001188,0.012493321,0.04063519,0.017760139,0.015123122,0.0040342077,0.014623858,0.030817775,0.024084361,0.0067538135,0.024019098,0.0018691572,0.0012316017,0.00047315852,0.0009837942,0.0054336297,0.004941008,0.0012967243,0.0047374414,0.0030628582,0.0022915925,0.0005391962,0.0021137437,0.007338304,0.0063138683,0.0016966239,0.0052717314,
# 0.012997985,0.0117155295,0.0030777645,0.01015733,0.07935571,0.06120232,0.018722156,0.06095617,0.027380023,0.024510076,0.00636403,0.024225913,0.044571526,0.035672765,0.009622827,0.03422622,0.0047190264,0.004302531,0.0014025613,0.0033143123,0.009432041,0.005902956,0.0021264253,0.007536569,0.0047880867,0.005673834,0.0009949572,0.0033871257,0.010609511,0.00920245,0.0024493476,0.009030799,
# 0.018202595,0.017957337,0.004373065,0.015643895,0.10656581,0.080679275,0.025487896,0.08330558,0.038209025,0.033250455,0.009170005,0.03237726,0.058636103,0.046434868,0.012660346,0.044679537,0.007854758,0.005768503,0.002268715,0.008986004,0.014770069,0.00983114,0.003083208,0.010302522,0.006911538,0.005217033,0.0017453799,0.005400936,0.014080669,0.012139247,0.003241611,0.0116140405,
# 0.024484172,0.020132925,0.005865778,0.020169087,0.13547227,0.10488641,0.032665987,0.10434384,0.04999328,0.047111817,0.01243743,0.043601763,0.0739088,0.056443166,0.016228655,0.055301793,0.012158077,0.0077683674,0.0031596,0.008204309,0.020170433,0.01667944,0.004415901,0.013306048,0.010205528,0.00779188,0.0025519833,0.009858705,0.018210392,0.013539668,0.0042062006,0.012684711,
# 0.032620717,0.026062839,0.007866154,0.028508145,0.16422422,0.13948585,0.040131673,0.13227254,0.06258061,0.05623223,0.016049335,0.05536546,0.090915024,0.06948795,0.02044969,0.07081661,0.017229019,0.012457474,0.0046513802,0.0097577,0.026040375,0.018810147,0.006465458,0.016107375,0.013538228,0.014716393,0.0032994247,0.013242982,0.023076769,0.01709373,0.005351745,0.015946735,
# 0.042019937,0.038459085,0.010403755,0.036606018,0.19331698,0.16497932,0.0480924,0.15809081,0.07734982,0.06628151,0.020134559,0.06665977,0.10909011,0.087729275,0.025050864,0.088557646,0.022503745,0.020818071,0.006927967,0.018500162,0.03315395,0.022884136,0.008868758,0.022140035,0.016259473,0.020043321,0.0040056203,0.017033216,0.028296517,0.02242896,0.0066592586,0.020671954,
# 0.053018544,0.043945048,0.013496416,0.048868727,0.2233986,0.19433828,0.056790236,0.18498115,0.09367285,0.08083235,0.024944784,0.077570885,0.12774304,0.106429465,0.029899785,0.10245252,0.028455816,0.026385143,0.010051699,0.024214841,0.04051659,0.0317201,0.010685223,0.03371081,0.01947256,0.016802795,0.004718835,0.019567518,0.034005947,0.026428467,0.008153965,0.024874445,
# 0.06642444,0.052665576,0.016637186,0.05934798,0.25438398,0.22872691,0.066161156,0.2105407,0.11205295,0.09141305,0.03041901,0.089378364,0.14694498,0.12696111,0.035110887,0.115275666,0.039102986,0.021248445,0.014708219,0.026204854,0.04750633,0.041003585,0.012101981,0.042072978,0.024561368,0.016390039,0.005887699,0.016311675,0.040516313,0.032657593,0.009974224,0.027374465,
# 0.08107264,0.070676826,0.019927874,0.064522885,0.28593892,0.25437933,0.076089956,0.24425058,0.1319608,0.11143023,0.03637095,0.107556105,0.16746707,0.13990355,0.040480293,0.13690169,0.058757324,0.028508892,0.019788759,0.044098567,0.05442676,0.048267417,0.013788228,0.047830697,0.03224053,0.020512832,0.008235083,0.019682605,0.04766673,0.041638285,0.012211361,0.03249471,
# 0.096126005,0.09132529,0.023415359,0.06965363,0.31890655,0.27729702,0.08709317,0.27129093,0.15427414,0.12128391,0.043223098,0.12338645,0.19013132,0.16622524,0.04654984,0.14642423,0.07992037,0.054648496,0.024749996,0.05948301,0.061552696,0.057707787,0.015458878,0.05459968,0.041008953,0.034037054,0.011238105,0.031556,0.055336032,0.048843615,0.01469795,0.041970696,
# 0.1127624,0.10586217,0.02733126,0.08638261,0.35289368,0.31031945,0.09903879,0.29872602,0.17994142,0.14728677,0.051613472,0.13424271,0.21497642,0.18051401,0.054032452,0.16099097,0.09954657,0.08237758,0.030076178,0.082754046,0.06870038,0.065489806,0.017090624,0.061786506,0.050467957,0.033161674,0.01365423,0.03953301,0.0640403,0.05701708,0.017206572,0.060070466,
# 0.13335794,0.11975571,0.03282819,0.09087089,0.38798422,0.3503935,0.11215011,0.33047897,0.20753147,0.18071158,0.061659735,0.15887487,0.24418594,0.20024118,0.06298922,0.1873912,0.117867455,0.114247926,0.03439155,0.11710762,0.07684215,0.06621377,0.01862837,0.07351302,0.060947612,0.05217801,0.015717512,0.040051233,0.07385866,0.0708729,0.019547567,0.07507124,
# 0.1576756,0.15105228,0.042469647,0.09731387,0.42528516,0.38500524,0.12642226,0.35791254,0.23812784,0.20621435,0.0728577,0.18522173,0.27856606,0.22248586,0.07426375,0.21168967,0.13552763,0.1408635,0.03843454,0.099087216,0.086611204,0.073226936,0.020171322,0.0817014,0.070557475,0.06722099,0.017842513,0.04792126,0.084343575,0.08692378,0.022191046,0.076482065,
# 0.18533225,0.18059108,0.05621599,0.15946102,0.4684915,0.40410587,0.1432915,0.38660827,0.2745887,0.23428473,0.084589645,0.22876146,0.3188595,0.25637585,0.08828152,0.24862266,0.15351008,0.19091009,0.04315375,0.11074266,0.09726111,0.09079604,0.022220355,0.0759096,0.081417575,0.06552036,0.02069368,0.060347937,0.09571229,0.10127909,0.025418421,0.09607578,
# 0.22203691,0.18012376,0.06950047,0.21052974,0.5195661,0.4687057,0.16516481,0.4081054,0.31599724,0.2812677,0.098374575,0.27395743,0.36679864,0.30269524,0.10618553,0.27730292,0.16960952,0.19480833,0.051321074,0.09963866,0.108143754,0.11391547,0.025343074,0.07696771,0.09498344,0.08311889,0.024087813,0.11030839,0.10843645,0.11600182,0.029312253,0.10196538,
# 0.2796896,0.19657871,0.084017456,0.2872454,0.5813466,0.52235323,0.19593577,0.47666824,0.36061487,0.3450483,0.11668137,0.2985278,0.42432866,0.37652564,0.13114405,0.3286453,0.18390888,0.26269704,0.06356711,0.13990042,0.12153991,0.12208677,0.029933292,0.10535359,0.11202165,0.107416674,0.028629486,0.118682325,0.1247891,0.1248803,0.0346735,0.12016721,
# 0.3760596,0.31925818,0.106317304,0.26658607,0.66341776,0.58163965,0.24559304,0.54797065,0.4231542,0.4022551,0.14486608,0.36135763,0.49888816,0.42677727,0.16736478,0.40831068,0.20451546,0.16393551,0.077114426,0.24938191,0.14191979,0.14165722,0.036589257,0.15538429,0.13656668,0.15002558,0.037185457,0.17624232,0.14874814,0.139695,0.042848796,0.12539768,
# 0.54714584,0.3338828,0.18405898,0.31421718,0.80217695,0.7468226,0.36190954,0.6889906,0.52647233,0.5648305,0.20580831,0.52842176,0.6451874,0.56037027,0.25789347,0.51332194,0.2895049,0.09274198,0.10601104,0.17944488,0.18833996,0.19381674,0.0554533,0.19230871,0.17503642,0.32216388,0.05215524,0.44692773,0.20440929,0.15064074,0.06699826,0.15185784,


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

    println("$event_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
  end

  bins_max
end

event_types_count = length(HREFPrediction.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:event_types_count
  (event_name, _, model_name) = HREFPrediction.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# event_name   mean_y        mean_ŷ        Σweight    bin_max
# tornado      0.000549145   0.0006369488  8.804865e6 0.021043906
# tornado      0.0320742     0.038657922   150732.98  0.074019335
# tornado      0.100802325   0.10940487    47960.055  0.17095083
# tornado      0.22403128    0.29084346    21573.916  1.0
# wind         0.0040149437  0.0052385866  8.622984e6 0.12138814
# wind         0.1548656     0.18282828    223552.19  0.27036425
# wind         0.3084194     0.3468556     112252.664 0.44575575
# wind         0.52182925    0.5839583     66342.76   1.0
# wind_adj     0.0013985921  0.0016133083  8.715982e6 0.04442204
# wind_adj     0.06458513    0.073951654   188741.5   0.12223551
# wind_adj     0.14532952    0.17385516    83878.29   0.25552708
# wind_adj     0.33364823    0.35534853    36530.12   1.0
# hail         0.0018027672  0.0023350087  8.702008e6 0.066225864
# hail         0.08223741    0.10230367    190762.23  0.15690458
# hail         0.17731425    0.21291392    88476.51   0.29827937
# hail         0.35741645    0.4218812     43885.312  1.0
# sig_tornado  7.119834e-5   7.736269e-5   8.967004e6 0.009904385
# sig_tornado  0.01459437    0.020232938   43714.035  0.047619365
# sig_tornado  0.06064399    0.08218383    10517.94   0.14518347
# sig_tornado  0.16341351    0.21502864    3896.0593  1.0
# sig_wind     0.0004664578  0.00057608244 8.751253e6 0.017750502
# sig_wind     0.02346163    0.03042175    173973.69  0.050954822
# sig_wind     0.060928304   0.06799452    66989.875  0.09204348
# sig_wind     0.123951025   0.12387162    32915.754  1.0
# sig_wind_adj 0.0001721023  0.00021295004 8.855584e6 0.00855192
# sig_wind_adj 0.013621317   0.015525776   111892.25  0.028172063
# sig_wind_adj 0.03506404    0.04570783    43474.57   0.07532653
# sig_wind_adj 0.107353225   0.10500419    14180.898  1.0
# sig_hail     0.00024502972 0.00030266377 8.862825e6 0.016018612
# sig_hail     0.020367835   0.026400018   106617.805 0.044114113
# sig_hail     0.056877602   0.061670527   38183.52   0.08977107
# sig_hail     0.12399248    0.13110583    17505.314  1.0

println("event_to_day_bins = $event_to_day_bins")
# event_to_day_bins = Dict{String, Vector{Float32}}("sig_wind" => [0.017750502, 0.050954822, 0.09204348, 1.0], "sig_hail" => [0.016018612, 0.044114113, 0.08977107, 1.0], "hail" => [0.066225864, 0.15690458, 0.29827937, 1.0], "sig_wind_adj" => [0.00855192, 0.028172063, 0.07532653, 1.0], "tornado" => [0.021043906, 0.074019335, 0.17095083, 1.0], "wind_adj" => [0.04442204, 0.12223551, 0.25552708, 1.0], "sig_tornado" => [0.009904385, 0.047619365, 0.14518347, 1.0], "wind" => [0.12138814, 0.27036425, 0.44575575, 1.0])










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
      ("total_prob_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_total_prob_x, bin_y, bin_weights))),
      ("max_hourly_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_max_hourly_x, bin_y, bin_weights))),
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

event_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:event_types_count
  event_name, _ = HREFPrediction.models[prediction_i]

  event_to_day_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado      1-2 -1.0             0.074019335      9718800 10390.0   8.955598e6 0.0012768853      0.00033077784     0.0010797478  0.0058558607       0.0064523746       0.03274472       0.028818477      0.0010797479    0.0058391923     0.032105613    Float32[0.95958227, 0.04161413,   -0.10651286]1.2272763,   -0.15624464,  -0.18067063]]
# tornado      2-3 0.021043906      0.17095083       211661  10323.0   198693.03  0.055734657       0.015497775       0.048663635   0.1845327          0.2118171          0.10162046       0.089558855      0.048663624     0.18389481       0.10101143     Float32[1.2272763,  -0.15624464,  -0.18067063]0.5964124,   0.17200926,   -0.3083448]]
# tornado      3-4 0.074019335      1.0              73216   10216.0   69533.97   0.1656988         0.04697025        0.13903588    0.39089894         0.45915875         0.2386106        0.23309015       0.13903588      0.38513187       0.24188775     Float32[0.5964124,  0.17200926,   -0.3083448]1.0424639,    -0.005972234, -0.1651274]]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind         1-2 -1.0             0.27036425       9599708 74517.0   8.846536e6 0.009726283       0.0025577322      0.007826943   0.028457472        0.032392044        0.15124032       0.13482153       0.007826943     0.028225765      0.15131083     Float32[1.0424639,  -0.005972234, -0.1651274]1.1494,       -0.107195236, -0.3110462]33]]
# wind         2-3 0.12138814       0.44575575       361168  74509.0   335804.88  0.23765922        0.06770699        0.20619547    0.4901267          0.60512197         0.30748412       0.2719515        0.20619547      0.4870105        0.30837902     Float32[1.1494,     -0.107195236, -0.3110462]0.92132,      0.0022699288, -0.22744325]94]
# wind         3-4 0.27036425       1.0              192308  74417.0   178595.42  0.43493205        0.13861363        0.38769466    0.63685286         0.8504553          0.55524194       0.51678663       0.38769466      0.6316496        0.555251       Float32[0.92132,    0.0022699288, -0.22744325]0.99742454,  0.02784912,   -0.01899123]]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# wind_adj     1-2 -1.0             0.12223551       9660343 26377.705 8.904724e6 0.0031465671      0.00083671947     0.0027378725  0.01251359         0.014017867        0.06161958       0.05519649       0.0027378723    0.012482578      0.06138815     Float32[0.99742454, 0.02784912,   -0.01899123]1.0577022,   -0.1336032,   -0.51860476]]]
# wind_adj     2-3 0.04442204       0.25552708       296069  26534.912 272619.78  0.104689464       0.030789096       0.089428164   0.29192677         0.335548           0.14887409       0.13270415       0.089428164     0.2904334        0.14910421     Float32[1.0577022,  -0.1336032,   -0.51860476]1.2023934,   -0.058322832, -0.087366514]
# wind_adj     3-4 0.12223551       1.0              131673  26673.31  120408.414 0.22891754        0.07201162        0.2024626     0.47390556         0.5727649          0.36890876       0.34496242       0.2024626       0.4711759        0.3685007      Float32[1.2023934,  -0.058322832, -0.087366514]1.0235776,  0.02489767,   -0.070147164]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# hail         1-2 -1.0             0.15690458       9648557 33931.0   8.89277e6  0.004479475       0.0010905664      0.0035282015  0.015019934        0.016858088        0.08252732       0.07416679       0.0035282015    0.014897022      0.082630605    Float32[1.0235776,  0.02489767,   -0.070147164]1.1081746,  -0.088940814, -0.34076628]
# hail         2-3 0.066225864      0.29827937       302339  33919.0   279238.75  0.1373504         0.03518375        0.11236241    0.34156653         0.40259886         0.17611699       0.16087049       0.11236241      0.33865112       0.17745832     Float32[1.1081746,  -0.088940814, -0.34076628]1.1614795,   -0.24988303,  -0.7304756]]
# hail         3-4 0.15690458       1.0              143459  33907.0   132361.81  0.2821982         0.082033046       0.23702818    0.5251295          0.6478943          0.38524213       0.35740376       0.23702817      0.5188467        0.38724697     Float32[1.1614795,  -0.24988303,  -0.7304756]0.3738464,    0.5298857,    -0.016889505]]]]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_tornado  1-2 -1.0             0.047619365      9776970 1353.0    9.010718e6 0.00017514417     5.156889e-5       0.00014165514 0.0009077408       0.00096063246      -0.15514319      0.01544859       0.00014165512   0.0008998909     0.015278328    Float32[0.3738464,  0.5298857,    -0.016889505]0.53315455, 0.50932276,   0.41353387]]
# sig_tornado  2-3 0.009904385      0.14518347       56971   1332.0    54231.977  0.032247912       0.010516065       0.023525394   0.10436762         0.10903808         0.08063426       0.08205603       0.023525394     0.10241765       0.09009691     Float32[0.53315455, 0.50932276,   0.41353387]0.41091824,   0.60895973,   0.510827]]3]
# sig_tornado  3-4 0.047619365      1.0              15046   1328.0    14414.0    0.11809137        0.03646364        0.088422276   0.29228613         0.31435373         0.13337168       0.14487167       0.08842227      0.28464696       0.14249226     Float32[0.41091824, 0.60895973,   0.510827]0.60896295,     0.32489055,   -0.06371247]]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind     1-2 -1.0             0.050954822      9684039 8826.0    8.925226e6 0.0011578449      0.00029291        0.00091468793 0.005241813        0.005667949        0.02300804       0.021486476      0.000914688     0.005205673      0.02345116     Float32[0.60896295, 0.32489055,   -0.06371247]0.99172807,  0.15444452,   0.46648186]]
# sig_wind     2-3 0.017750502      0.09204348       259699  8825.0    240963.56  0.04086729        0.010679467       0.033877674   0.14376263         0.1606643          0.05855743       0.05737881       0.03387768      0.14298609       0.0596917      Float32[0.99172807, 0.15444452,   0.46648186]0.8760332,    0.31676358,   0.86151475]]]
# sig_wind     3-4 0.050954822      1.0              107977  8814.0    99905.625  0.08640426        0.021111885       0.0816923     0.27550456         0.32947657         0.13525863       0.12320096       0.081692316     0.2748943        0.13360213     Float32[0.8760332,  0.31676358,   0.86151475]0.98283756,   0.02864707,   -0.070753366]6]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_wind_adj 1-2 -1.0             0.028172063      9728631 3323.5562 8.967476e6 0.00040401678     9.4150906e-5      0.0003399157  0.0021760792       0.0023787252       0.08148481       0.011081328      0.0003399157    0.0021705371     0.011994057    Float32[0.98283756, 0.02864707,   -0.070753366]0.74302465, 0.17016491,   -0.2746076]]
# sig_wind_adj 2-3 0.00855192       0.07532653       170006  3353.0269 155366.81  0.023971286       0.0062256325      0.019621396   0.093790375        0.10304016         -0.19568825      0.03913906       0.019621395     0.093309216      0.041019       Float32[0.74302465, 0.17016491,   -0.2746076]0.37014917,   1.2757467,    3.3445797]]]
# sig_wind_adj 3-4 0.028172063      1.0              63385   3345.205  57655.47   0.060292322       0.015411966       0.052844238   0.19441897         0.22185005         0.16079274       0.17230234       0.052844245     0.19071871       0.17257723     Float32[0.37014917, 1.2757467,    3.3445797]1.2635667,     -0.21535042,  -0.32895777]]]
# event_name   bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# sig_hail     1-2 -1.0             0.044114113      9731954 4673.0    8.969443e6 0.0006128773      0.00016209374     0.0004842251  0.0027594366       0.0029945907       0.020729106      0.021594504      0.00048422528   0.0027425932     0.020822903    Float32[1.2635667,  -0.21535042,  -0.32895777]1.5278528,   -0.32414392,  0.0025766783]]
# sig_hail     2-3 0.016018612      0.08977107       156725  4683.0    144801.31  0.035700712       0.009499065       0.029995317   0.13012302         0.14597766         -0.6169255       0.05282002       0.029995324     0.12934238       0.058437664    Float32[1.5278528,  -0.32414392,  0.0025766783]1.1042255,  -0.16546442,  -0.4495338]]
# sig_hail     3-4 0.044114113      1.0              60062   4661.0    55688.836  0.08349693        0.023061702       0.077974595   0.2663782          0.31153002         0.12403267       0.11937379       0.07797461      0.26606894       0.124048114    Float32[1.1042255,  -0.16546442,  -0.4495338]


println("event_to_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[0.60896295, 0.32489055, -0.06371247], [0.99172807, 0.15444452, 0.46648186], [0.8760332, 0.31676358, 0.86151475]], "sig_hail" => [[1.2635667, -0.21535042, -0.32895777], [1.5278528, -0.32414392, 0.0025766783], [1.1042255, -0.16546442, -0.4495338]], "hail" => [[1.0235776, 0.02489767, -0.070147164], [1.1081746, -0.088940814, -0.34076628], [1.1614795, -0.24988303, -0.7304756]], "sig_wind_adj" => [[0.98283756, 0.02864707, -0.070753366], [0.74302465, 0.17016491, -0.2746076], [0.37014917, 1.2757467, 3.3445797]], "tornado" => [[0.95958227, 0.04161413, -0.10651286], [1.2272763, -0.15624464, -0.18067063], [0.5964124, 0.17200926, -0.3083448]], "wind_adj" => [[0.99742454, 0.02784912, -0.01899123], [1.0577022, -0.1336032, -0.51860476], [1.2023934, -0.058322832, -0.087366514]], "sig_tornado" => [[0.3738464, 0.5298857, -0.016889505], [0.53315455, 0.50932276, 0.41353387], [0.41091824, 0.60895973, 0.510827]], "wind" => [[1.0424639, -0.005972234, -0.1651274], [1.1494, -0.107195236, -0.3110462], [0.92132, 0.0022699288, -0.22744325]])



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
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected: 358
# 358

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
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (20606.0)                         feature 1 TORPROB:calculated:hour fcst:calculated_prob:                    AU-PR-curve: 0.1562636
# wind (148934.0)                           feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.40284145
# wind_adj (53051.016)                      feature 3 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.24262798 (lol trivially worse)
# hail (67838.0)                            feature 4 HAILPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.26647338
# sig_tornado (2681.0)                      feature 5 STORPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.090377256 (slightly worse)
# sig_wind (17640.0)                        feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.08896147 (slightly worse)
# sig_wind_adj (6668.761)                   feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.09988912
# sig_hail (9334.0)                         feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.08331046
# sig_tornado_gated_by_tornado (2681.0)     feature 9 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado   AU-PR-curve: 0.086182915 (worse)
# sig_wind_gated_by_wind (17640.0)          feature 10 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind     AU-PR-curve: 0.088962585
# sig_wind_adj_gated_by_wind_adj (6668.761) feature 11 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.099981256
# sig_hail_gated_by_hail (9334.0)           feature 12 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail     AU-PR-curve: 0.083310924


function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, HREFPrediction.models_with_gated))

  for event_name in event_names
    y = Ys[event_name]
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(sum(y))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (20606.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.1562636 ***best tor***
# tornado (20606.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.043532312
# tornado (20606.0)     feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.02481927
# tornado (20606.0)     feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.035571136
# tornado (20606.0)     feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.12339083
# tornado (20606.0)     feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.04694929
# tornado (20606.0)     feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.02226341
# tornado (20606.0)     feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.03144813
# tornado (20606.0)     feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.12253535
# tornado (20606.0)     feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.046951354
# tornado (20606.0)     feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.022343898
# tornado (20606.0)     feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.0314522
# wind (148934.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.21884723
# wind (148934.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.40284145 ***best wind***
# wind (148934.0)       feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.2584484
# wind (148934.0)       feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.16652994
# wind (148934.0)       feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.20128852
# wind (148934.0)       feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.3097344
# wind (148934.0)       feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.21550208
# wind (148934.0)       feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.1342677
# wind (148934.0)       feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.20174475
# wind (148934.0)       feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.30975288
# wind (148934.0)       feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.21603344
# wind (148934.0)       feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.13430235
# wind_adj (53051.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.058999192
# wind_adj (53051.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.10225016
# wind_adj (53051.0)    feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.24262798 ***best wind_adj***
# wind_adj (53051.0)    feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.085828975
# wind_adj (53051.0)    feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.05999237
# wind_adj (53051.0)    feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.16412897
# wind_adj (53051.0)    feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.2267024
# wind_adj (53051.0)    feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.079858676
# wind_adj (53051.0)    feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.060000226
# wind_adj (53051.0)    feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.16413383
# wind_adj (53051.0)    feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.22707203
# wind_adj (53051.0)    feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.07986895
# hail (67838.0)        feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.10647426
# hail (67838.0)        feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.10877037
# hail (67838.0)        feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.10454446
# hail (67838.0)        feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.26647338 ***best hail***
# hail (67838.0)        feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.09519305
# hail (67838.0)        feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.11641565
# hail (67838.0)        feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.089664355
# hail (67838.0)        feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.2156373
# hail (67838.0)        feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.094946325
# hail (67838.0)        feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.11642344
# hail (67838.0)        feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.08992765
# hail (67838.0)        feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.21564318
# sig_tornado (2681.0)  feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.08047178
# sig_tornado (2681.0)  feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.0105957985
# sig_tornado (2681.0)  feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0038604557
# sig_tornado (2681.0)  feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.006307488
# sig_tornado (2681.0)  feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.090377256 *** best sigtor***
# sig_tornado (2681.0)  feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.01017822
# sig_tornado (2681.0)  feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.0035990349
# sig_tornado (2681.0)  feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0047108326
# sig_tornado (2681.0)  feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.086182915 (not best sigtor)
# sig_tornado (2681.0)  feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.010178301
# sig_tornado (2681.0)  feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.0036122631
# sig_tornado (2681.0)  feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.004711013
# sig_wind (17640.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.03967122
# sig_wind (17640.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.055467352
# sig_wind (17640.0)    feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.08614578
# sig_wind (17640.0)    feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.034115303
# sig_wind (17640.0)    feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.03795299
# sig_wind (17640.0)    feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.08896147
# sig_wind (17640.0)    feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.09034605
# sig_wind (17640.0)    feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.030208504
# sig_wind (17640.0)    feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.038092896
# sig_wind (17640.0)    feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.088962585 (not best sigwind)
# sig_wind (17640.0)    feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.09047847 ***best sigwind***
# sig_wind (17640.0)    feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.030211626
# sig_wind_adj (6669.0) feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.008668457
# sig_wind_adj (6669.0) feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.016446035
# sig_wind_adj (6669.0) feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.07514209
# sig_wind_adj (6669.0) feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.01625834
# sig_wind_adj (6669.0) feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.009581744
# sig_wind_adj (6669.0) feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.04095978
# sig_wind_adj (6669.0) feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.09988912
# sig_wind_adj (6669.0) feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.015817167
# sig_wind_adj (6669.0) feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.009581198
# sig_wind_adj (6669.0) feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.040960137
# sig_wind_adj (6669.0) feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.099981256 ***best sigwind_adj***
# sig_wind_adj (6669.0) feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.015818216
# sig_hail (9334.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.021061353
# sig_hail (9334.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.016394684
# sig_hail (9334.0)     feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.019656738
# sig_hail (9334.0)     feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.07716082
# sig_hail (9334.0)     feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.018596077
# sig_hail (9334.0)     feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.026459418
# sig_hail (9334.0)     feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.01787866
# sig_hail (9334.0)     feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.08331046
# sig_hail (9334.0)     feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.018399393
# sig_hail (9334.0)     feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.026460137
# sig_hail (9334.0)     feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.017912706
# sig_hail (9334.0)     feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.083310924 ***best sighail***


# test y vs ŷ
event_names = map(m -> m[1], HREFPrediction.models_with_gated)
model_names = map(m -> m[3], HREFPrediction.models_with_gated)
Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)
# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,ŷ_sig_wind_gated_by_wind,y_sig_wind_gated_by_wind,ŷ_sig_wind_adj_gated_by_wind_adj,y_sig_wind_adj_gated_by_wind_adj,ŷ_sig_hail_gated_by_hail,y_sig_hail_gated_by_hail,
# 0.00011889208,0.00011867229,0.0008131319,0.00087078015,0.0002889879,0.00030152575,0.00037449508,0.0003831146,1.2998527e-5,1.4771842e-5,9.50439e-5,0.00010120271,3.4510555e-5,3.6910937e-5,4.5306107e-5,5.0902956e-5,1.2996232e-5,1.477153e-5,9.478991e-5,0.000101197395,3.440451e-5,3.6891775e-5,4.525614e-5,5.0902745e-5,
# 0.0025425232,0.0029419851,0.02241872,0.022558777,0.007767146,0.008228998,0.01282629,0.013905334,0.0005095983,0.0011977841,0.0022598708,0.002730501,0.0010920194,0.0008828687,0.0027855157,0.002711769,0.0005095808,0.0011972922,0.0022599676,0.002733507,0.001092299,0.0008895582,0.0027854447,0.00271207,
# 0.006280865,0.0051744217,0.0421905,0.042060122,0.015162765,0.015129829,0.023605391,0.024025073,0.0016429224,0.0010543535,0.0045290333,0.004942742,0.0025406713,0.002304841,0.005627138,0.0060957097,0.0016427174,0.0010561689,0.004529033,0.004944459,0.0025408585,0.0023162046,0.0056271534,0.006096304,
# 0.011003303,0.011740553,0.063175865,0.06121251,0.023668904,0.024482608,0.034828246,0.03586223,0.0041858004,0.00389026,0.007562285,0.0059392103,0.0040040184,0.0055436753,0.008375404,0.0093152765,0.0041861087,0.0038918417,0.0075624026,0.0059402627,0.004003963,0.005565349,0.008375471,0.009316067,
# 0.015415692,0.017903777,0.0862857,0.08062688,0.033387594,0.033224992,0.046554014,0.046197318,0.006386066,0.0076766275,0.011197518,0.011531105,0.005852346,0.0051616086,0.011333972,0.011903674,0.0063860207,0.0076763593,0.011197474,0.011531711,0.005852511,0.005188265,0.011333939,0.011903998,
# 0.020695614,0.020229997,0.111288995,0.104979165,0.04414147,0.047070723,0.05947211,0.05650269,0.008907415,0.0075256405,0.014975402,0.014900326,0.008722808,0.007916866,0.014832607,0.013400663,0.0089072855,0.0075231525,0.014975468,0.014902139,0.008722804,0.007954687,0.014832543,0.013401072,
# 0.02752036,0.026117474,0.13662523,0.13938235,0.05555065,0.05623249,0.07378093,0.069163494,0.012388628,0.011451016,0.01989036,0.017735135,0.011577186,0.014872413,0.018724523,0.018274529,0.012388699,0.011444545,0.019890405,0.01773646,0.011578408,0.014939325,0.018724542,0.018277574,
# 0.035524398,0.03829976,0.16283424,0.16516429,0.068407536,0.06643024,0.08903005,0.08741153,0.016763192,0.019254943,0.025997434,0.022330683,0.013886181,0.019358344,0.022798512,0.02381495,0.016764838,0.0192366,0.025997426,0.022330683,0.01388581,0.019462002,0.022798644,0.023814946,
# 0.045094907,0.04435019,0.19052139,0.19439262,0.08202841,0.0792736,0.10437473,0.10729696,0.022311522,0.026577353,0.032626506,0.03250867,0.01641229,0.017747588,0.027637914,0.026307857,0.022309026,0.026534533,0.032626506,0.03250867,0.016412508,0.017781403,0.027638001,0.026309436,
# 0.057168424,0.053338945,0.21955998,0.22973585,0.09631717,0.09195609,0.12011737,0.1251812,0.031543516,0.024622161,0.03954217,0.041032963,0.020200819,0.015515409,0.033947866,0.032053493,0.031531814,0.024457267,0.03954217,0.041032963,0.020200804,0.015545636,0.033947866,0.032053493,
# 0.07256332,0.06717904,0.25059497,0.25078857,0.11126653,0.11244005,0.13763468,0.14219716,0.045757595,0.033072922,0.04666395,0.0471961,0.025288496,0.02082508,0.042148232,0.038434736,0.045373693,0.03405337,0.04666395,0.0471961,0.025288712,0.020878078,0.042148232,0.038434736,
# 0.08858901,0.09274736,0.28282484,0.27689162,0.12844913,0.120925196,0.15773332,0.1630101,0.061199002,0.057137042,0.054052092,0.055491604,0.031314157,0.034457814,0.051082037,0.052602403,0.059323248,0.06310514,0.054052092,0.055491604,0.03131388,0.034551755,0.051082022,0.052602403,
# 0.10420568,0.10806544,0.3145093,0.3134728,0.14968893,0.15004152,0.17927237,0.18475817,0.07605492,0.074114695,0.061624505,0.06574845,0.039085634,0.034767285,0.060060542,0.05909109,0.07240866,0.07668635,0.061624505,0.06574845,0.039120834,0.034608956,0.060060542,0.05909109,
# 0.12102106,0.116944864,0.34563428,0.3520341,0.1753036,0.17692542,0.20370498,0.198593,0.09061565,0.10950221,0.070276044,0.06753287,0.0485961,0.04838453,0.069849335,0.07071624,0.08605136,0.1024641,0.070276044,0.06753287,0.04865595,0.04891964,0.069849335,0.07071624,
# 0.13717742,0.14863971,0.37697667,0.3844238,0.20726584,0.21075964,0.23179689,0.21961693,0.10403954,0.13000391,0.08053277,0.08011808,0.059703775,0.05533395,0.07951998,0.08549333,0.09893788,0.14085607,0.08053277,0.08011808,0.059738368,0.05540278,0.07951998,0.08549333,
# 0.15673207,0.17233202,0.41393113,0.40401843,0.24682249,0.23460913,0.26593435,0.2570302,0.11527834,0.15493634,0.091891035,0.09318738,0.07649539,0.057900287,0.089045234,0.10612989,0.11031457,0.15078272,0.091891035,0.09318738,0.07654044,0.057872884,0.089045234,0.10612989,
# 0.1803576,0.19112791,0.4603577,0.468584,0.29203388,0.27901834,0.30709186,0.30743837,0.12577787,0.17912082,0.10428671,0.10322924,0.09859661,0.104420014,0.10036062,0.10987321,0.120829545,0.15991935,0.10428671,0.10322924,0.09862872,0.1049091,0.10036062,0.10987321,
# 0.21512035,0.2046197,0.51799875,0.5224018,0.34211114,0.34451497,0.35592264,0.378745,0.13941431,0.20499559,0.120035976,0.1238735,0.12749632,0.11876565,0.11489148,0.12838237,0.13368374,0.20913896,0.120035976,0.1238735,0.12758304,0.11876325,0.11489148,0.12838237,
# 0.27166343,0.3318058,0.597714,0.5817142,0.4128675,0.40471217,0.42256165,0.42708427,0.15993898,0.19850257,0.14353691,0.14725763,0.18346079,0.1727036,0.13573161,0.14228083,0.15210858,0.17331935,0.14353691,0.14725763,0.18380527,0.17298263,0.13573161,0.14228083,
# 0.3821813,0.3331822,0.74548006,0.7467022,0.53086144,0.5622761,0.5678906,0.5656451,0.22346582,0.121664494,0.20223112,0.19897026,0.281509,0.4406937,0.18268217,0.14665508,0.21655422,0.109833695,0.20223112,0.19897026,0.28208202,0.44411442,0.18268217,0.14665508,

# fewer bins
Metrics.reliability_curves_midpoints(10, X, Ys, event_names, weights, model_names)
# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,ŷ_sig_wind_gated_by_wind,y_sig_wind_gated_by_wind,ŷ_sig_wind_adj_gated_by_wind_adj,y_sig_wind_adj_gated_by_wind_adj,ŷ_sig_hail_gated_by_hail,y_sig_hail_gated_by_hail,
# 0.000212819,0.00022808871,0.0016160766,0.0016767154,0.00055331265,0.0005817321,0.00070827105,0.0007455945,1.9023551e-5,2.9031735e-5,0.0001723924,0.00019508279,7.697451e-5,7.088012e-5,9.558262e-5,9.97904e-5,1.9023308e-5,2.9030987e-5,0.00017206496,0.00019508056,7.6556214e-5,7.086619e-5,9.552656e-5,9.9790195e-5,
# 0.007724952,0.0071809376,0.050734397,0.049859982,0.018411703,0.018702144,0.02810571,0.028768713,0.002183052,0.0016544504,0.0059034307,0.005394079,0.0029700806,0.0032516515,0.006700035,0.007356499,0.0021835053,0.0016568248,0.005903623,0.0053955186,0.0029705039,0.00326671,0.006700104,0.007357117,
# 0.017891267,0.018991545,0.097139984,0.09119262,0.037837874,0.038954776,0.05236246,0.050828375,0.007616932,0.0076284963,0.012836787,0.012991045,0.006980982,0.006242102,0.012958078,0.012571867,0.007616961,0.0076272828,0.012836751,0.012992219,0.0069813556,0.0062733516,0.01295807,0.012572408,
# 0.030762509,0.031044213,0.14861451,0.15115777,0.06144444,0.060907293,0.08051315,0.07722022,0.013917707,0.014114622,0.022583285,0.019757139,0.012550589,0.016918711,0.020470597,0.02062593,0.013918388,0.014104766,0.022583418,0.01975796,0.012550671,0.017001241,0.020470828,0.02062787,
# 0.050557744,0.04843638,0.20381348,0.21066783,0.08864402,0.08514551,0.11163223,0.11556834,0.026689876,0.02585529,0.03566327,0.03627603,0.018343916,0.01661045,0.030444216,0.028868642,0.026697557,0.025736611,0.03566327,0.03627603,0.018344274,0.016642218,0.030444358,0.028869594,
# 0.07925569,0.07781037,0.2658873,0.26309666,0.11954545,0.116508745,0.14698377,0.15186115,0.050699048,0.04051676,0.050017815,0.051044647,0.027431434,0.02555415,0.045850445,0.044330683,0.04954391,0.042582244,0.050017815,0.051044647,0.027430134,0.025620025,0.045850437,0.044330683,
# 0.11224538,0.11235403,0.3291589,0.33158588,0.16142577,0.16244726,0.19103469,0.19136085,0.08149914,0.08766459,0.06583241,0.06651965,0.04299184,0.040226012,0.064425796,0.06432289,0.07778833,0.08684738,0.06583241,0.06651965,0.042991288,0.04028316,0.064425796,0.06432289,
# 0.14612648,0.15980539,0.39498445,0.39398584,0.22595643,0.22191888,0.24751441,0.23686694,0.10872359,0.14110546,0.08573913,0.08588299,0.06782309,0.056608316,0.083652735,0.094560474,0.10392964,0.1460952,0.08573913,0.08588299,0.067827746,0.056646165,0.083652735,0.094560474,
# 0.1969181,0.19722623,0.48758325,0.49407983,0.31443346,0.3083107,0.32894972,0.33934787,0.13146496,0.18976097,0.11141054,0.11262602,0.1117947,0.11093978,0.10684263,0.1182667,0.12567131,0.18081753,0.11141054,0.11262602,0.1117833,0.11110065,0.10684263,0.1182667,
# 0.32620057,0.33200672,0.662326,0.65382564,0.46216524,0.47059765,0.4849712,0.4867182,0.19693235,0.15367857,0.16823643,0.16921006,0.20994672,0.24572918,0.15820777,0.1443478,0.18901423,0.13621844,0.16823643,0.16921006,0.20999658,0.24599668,0.15820777,0.1443478,



# Calibrate to SPC
# The targets below are computing in and copied from models/spc_outlooks/Stats.jl

target_warning_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.025348036),
    (0.05, 0.007337035),
    (0.1,  0.0014824796),
    (0.15, 0.00025343563),
    (0.3,  3.7446924e-5),
    (0.45, 3.261123e-6),
  ],
  "wind" => [
    (0.05, 0.07039726),
    (0.15, 0.021633422),
    (0.3,  0.0036298542),
    (0.45, 0.0004162882),
  ],
  "hail" => [
    (0.05, 0.052633155),
    (0.15, 0.015418012),
    (0.3,  0.0015550428),
    (0.45, 8.9432746e-5),
  ],
  "sig_tornado" => [(0.1, 0.0009527993)],
  "sig_wind"    => [(0.1, 0.0014686467)],
  "sig_hail"    => [(0.1, 0.002794325)],
)

# Assumes weights are proportional to gridpoint areas
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
  event_name, _ = HREFPrediction.models[prediction_i]
  spc_event_name = replace(event_name, "_adj" => "")
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  thresholds_to_match_warning_ratio =
    map(target_warning_ratios[spc_event_name]) do (nominal_prob, target_warning_ratio)
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
  for i in 1:length(target_warning_ratios[spc_event_name])
    nominal_prob, _ = target_warning_ratios[spc_event_name][i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, threshold_to_match_warning_ratio))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, threshold_to_match_warning_ratio))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, threshold_to_match_warning_ratio))
    println("$event_name\t$nominal_prob\t$threshold_to_match_warning_ratio\t$sr\t$pod\t$wr")
    push!(wr_thresholds, (Float32(nominal_prob), Float32(threshold_to_match_warning_ratio)))
  end

  wr_thresholds
end

println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _, _, _, _ = HREFPrediction.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end

# event_name   nominal_prob threshold_to_match_warning_ratio SR          POD           WR
# tornado      0.02         0.016950607                      0.064026326 0.7574176     0.025346832
# tornado      0.05         0.06830406                       0.14281912  0.48904803    0.0073368903
# tornado      0.1          0.17817497                       0.257954    0.17845055    0.0014822535
# tornado      0.15         0.3255825                        0.32377487  0.03830062    0.00025346005
# tornado      0.3          0.4591999                        0.48384824  0.008458492   3.7456797e-5
# tornado      0.45         0.60011864                       0.49274874  0.00073567254 3.1989384e-6
# wind         0.05         0.0479908                        0.1879388   0.8622612     0.07039821
# wind         0.15         0.21794319                       0.37482038  0.52845407    0.021633325
# wind         0.3          0.4940548                        0.60867655  0.14399344    0.0036299056
# wind         0.45         0.7420559                        0.84867144  0.023029312   0.00041637113
# wind_adj     0.05         0.011899948                      0.068940096 0.8982724     0.07039313
# wind_adj     0.15         0.06845665                       0.15628837  0.62584263    0.02163379
# wind_adj     0.3          0.24048424                       0.34531587  0.23201579    0.0036299035
# wind_adj     0.45         0.471941                         0.57944137  0.04463978    0.0004162044
# hail         0.05         0.029951096                      0.112065494 0.8483541     0.052633014
# hail         0.15         0.123464584                      0.2321128   0.5147346     0.015418328
# hail         0.3          0.3763752                        0.47983125  0.10731995    0.0015550521
# hail         0.45         0.6608143                        0.7090504   0.00912425    8.9469104e-5
# sig_tornado  0.1          0.06202507                       0.12471959  0.4204412     0.0009528316
# sig_wind     0.1          0.118608475                      0.15741116  0.1277735     0.0014682951
# sig_wind_adj 0.1          0.06686592                       0.114711285 0.2494503     0.0014685683
# sig_hail     0.1          0.070280075                      0.11138754  0.32341462    0.002794258


println(calibrations_wr)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_wind" => [(0.1, 0.118608475)], "sig_hail" => [(0.1, 0.070280075)], "hail" => [(0.05, 0.029951096), (0.15, 0.123464584), (0.3, 0.3763752), (0.45, 0.6608143)], "sig_wind_adj" => [(0.1, 0.06686592)], "tornado" => [(0.02, 0.016950607), (0.05, 0.06830406), (0.1, 0.17817497), (0.15, 0.3255825), (0.3, 0.4591999), (0.45, 0.60011864)], "wind_adj" => [(0.05, 0.011899948), (0.15, 0.06845665), (0.3, 0.24048424), (0.45, 0.471941)], "sig_tornado" => [(0.1, 0.06202507)], "wind" => [(0.05, 0.0479908), (0.15, 0.21794319), (0.3, 0.4940548), (0.45, 0.7420559)])
