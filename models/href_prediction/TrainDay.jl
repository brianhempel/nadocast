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
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (9446.0)                      feature 1 TORPROB:calculated:hour fcst:calculated_prob:                  AU-PR-curve: 0.12701213629264377
# wind (72111.0)                        feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.3834656173824258
# hail (31894.0)                        feature 3 HAILPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.23064103438576106
# sig_tornado (1268.0)                  feature 4 STORPROB:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.09451181604245633
# sig_wind (8732.0)                     feature 5 SWINDPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.07854469228892626
# sig_hail (4478.0)                     feature 6 SHAILPRO:calculated:hour fcst:calculated_prob:                 AU-PR-curve: 0.0691045922967536
# sig_tornado_gated_by_tornado (1268.0) feature 7 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.09363142784066525
# sig_wind_gated_by_wind (8732.0)       feature 8 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind    AU-PR-curve: 0.07855136828846834
# sig_hail_gated_by_hail (4478.0)       feature 9 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail    AU-PR-curve: 0.06912157017181443


function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, HREFPrediction.models_with_gated))

  # Feature order is all HREF severe probs then all SREF severe probs
  for event_name in event_names
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      y = Ys[event_name]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(round(sum(y)))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (9446.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.12701213629264377
# tornado (9446.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.04168648434556345
# tornado (9446.0)     feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.030765254579900772
# tornado (9446.0)     feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.11531809758773186
# tornado (9446.0)     feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.039822720343631504
# tornado (9446.0)     feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.032309275402615086
# tornado (9446.0)     feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.11624433417007461
# tornado (9446.0)     feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.03983155452623281
# tornado (9446.0)     feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.03231108262426286
# wind (72111.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.20397678719531137
# wind (72111.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.3834656173824258
# wind (72111.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.15227293177135814
# wind (72111.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.18899000884190237
# wind (72111.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.2871189699820319
# wind (72111.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.1232035272300651
# wind (72111.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.19067429713597928
# wind (72111.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.28719210932407674
# wind (72111.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.12326955300059134
# hail (31894.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.09099255929593081
# hail (31894.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.09452327656751952
# hail (31894.0)       feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.23064103438576106
# hail (31894.0)       feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.08234931731140827
# hail (31894.0)       feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.09955065202449667
# hail (31894.0)       feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.188595511108089
# hail (31894.0)       feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.0824502686306494
# hail (31894.0)       feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.09958118472086727
# hail (31894.0)       feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.1886802022992225
# sig_tornado (1268.0) feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.05728594701652062
# sig_tornado (1268.0) feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.011385789618738
# sig_tornado (1268.0) feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.005588672134719318
# sig_tornado (1268.0) feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.09451181604245633
# sig_tornado (1268.0) feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.008881777469212735
# sig_tornado (1268.0) feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.005321406702197539
# sig_tornado (1268.0) feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.09363142784066525
# sig_tornado (1268.0) feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.008882564887699068
# sig_tornado (1268.0) feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.005324294230821452
# sig_wind (8732.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.03572679857698384
# sig_wind (8732.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.05250853163761901
# sig_wind (8732.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.03161747330844581
# sig_wind (8732.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.034007477644701585
# sig_wind (8732.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.07854469228892626
# sig_wind (8732.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.028783730365721183
# sig_wind (8732.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.03429306032063456
# sig_wind (8732.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.07855136828846834
# sig_wind (8732.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.02879889359659011
# sig_hail (4478.0)    feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                       AU-PR-curve: 0.01798515458351701
# sig_hail (4478.0)    feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.013479598274345111
# sig_hail (4478.0)    feature 3 hail HAILPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.06589495461034388
# sig_hail (4478.0)    feature 4 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                  AU-PR-curve: 0.014544889120801473
# sig_hail (4478.0)    feature 5 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.01836176630607075
# sig_hail (4478.0)    feature 6 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.0691045922967536
# sig_hail (4478.0)    feature 7 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado AU-PR-curve: 0.01464675890085453
# sig_hail (4478.0)    feature 8 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind          AU-PR-curve: 0.018365716755777285
# sig_hail (4478.0)    feature 9 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail          AU-PR-curve: 0.06912157017181443



# test y vs ŷ

function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  total_weight = sum(Float64.(weights))

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPrediction.models_with_gated[prediction_i]
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
# tornado                      9.700865e-5   9.9049845e-5  4.573599e6  0.0017090585  1.0         0.0008555745
# tornado                      0.0018642666  0.00142832    237875.36   0.013719558   0.9499555   0.0023576757
# tornado                      0.0031376549  0.0036643934  141532.73   0.02121991    0.89993536  0.005619943
# tornado                      0.006996247   0.007203153   63402.914   0.032135308   0.84984547  0.009180412
# tornado                      0.00960002    0.011391198   46223.64    0.04145316    0.79981184  0.014106784
# tornado                      0.016310018   0.01626344    27224.174   0.053247765   0.74975955  0.018732127
# tornado                      0.02485353    0.020887442   17840.715   0.06355       0.6996758   0.023426436
# tornado                      0.028516073   0.026392281   15566.109   0.072204635   0.64966226  0.029872384
# tornado                      0.031154005   0.034372345   14257.465   0.08279699    0.59959453  0.039933395
# tornado                      0.041785702   0.046009205   10617.009   0.097539      0.54949385  0.053307083
# tornado                      0.04763488    0.0617777     9316.975    0.112589985   0.4994538   0.07096115
# tornado                      0.06391539    0.07893289    6942.6855   0.13275504    0.4493942   0.08757907
# tornado                      0.11442482    0.09269282    3880.694    0.15347265    0.3993423   0.09807844
# tornado                      0.14126927    0.10239867    3138.6887   0.16136983    0.3492562   0.10644376
# tornado                      0.16568106    0.10938814    2675.5706   0.16530076    0.2992431   0.11252215
# tornado                      0.1327265     0.11989863    3344.8777   0.16522467    0.24924229  0.12869819
# tornado                      0.11034642    0.14242381    4017.9119   0.17606342    0.19916676  0.15996142
# tornado                      0.13868733    0.18497059    3196.6716   0.21998936    0.14915797  0.21840203
# tornado                      0.32516035    0.23600286    1364.0076   0.31233206    0.099151924 0.2564116
# tornado                      0.3002684     0.3199752     1450.4642   0.3002684     0.049125206 1.0
# wind                         0.0007333039  0.0006652036  4.5677065e6 0.012911272   1.0         0.011141602
# wind                         0.017665606   0.017918272   189598.22   0.10266422    0.9499899   0.027255114
# wind                         0.034650616   0.03490376    96670.13    0.14012814    0.89998204  0.044160966
# wind                         0.05201806    0.05276521    64395.637   0.17070308    0.8499695   0.06265394
# wind                         0.066318914   0.07287459    50508.055   0.19910471    0.79995614  0.0843732
# wind                         0.08921028    0.09491389    37541.48    0.22978672    0.7499442   0.10652433
# wind                         0.11898993    0.116981484   28143.969   0.25893623    0.6999405   0.12846802
# wind                         0.14411765    0.13994569    23240.582   0.28469524    0.6499404   0.15250224
# wind                         0.1720042     0.16488889    19471.506   0.30989215    0.5999324   0.17824909
# wind                         0.19195645    0.19248405    17449.746   0.33425775    0.5499273   0.20799732
# wind                         0.21295361    0.22392222    15728.158   0.36103234    0.49991605  0.24049737
# wind                         0.2602031     0.25579062    12870.99    0.39127383    0.44990817  0.27177548
# wind                         0.2981486     0.2870136     11232.168   0.41757473    0.39990473  0.30342957
# wind                         0.3278786     0.31958458    10215.239   0.44292727    0.3499045   0.33644634
# wind                         0.3542443     0.35344136    9454.063    0.47045377    0.29989678  0.3715729
# wind                         0.3776399     0.39261442    8868.662    0.50350475    0.24989367  0.41578835
# wind                         0.43711454    0.44158578    7661.592    0.5493044     0.19988889  0.47099006
# wind                         0.52115136    0.50405425    6426.139    0.6007411     0.14988661  0.54199266
# wind                         0.57601786    0.58847904    5814.9453   0.65047044    0.099884346 0.64767545
# wind                         0.7473285     0.7519671     4469.818    0.7473285     0.049874317 1.0
# hail                         0.00031078092 0.00030401986 4.7322955e6 0.005669666   1.0         0.006891266
# hail                         0.01044888    0.011080333   140805.23   0.061384488   0.94999504  0.016829325
# hail                         0.021978628   0.021263007   66916.24    0.0841986     0.8999713   0.026533825
# hail                         0.03235512    0.031273644   45454.387   0.10102429    0.84996563  0.03663534
# hail                         0.045308188   0.041578356   32474.467   0.11647665    0.7999615   0.04720469
# hail                         0.055931125   0.052971583   26297.406   0.13011006    0.7499344   0.05915104
# hail                         0.06446846    0.06542572    22810.768   0.14373004    0.6999247   0.07218698
# hail                         0.078798346   0.07871479    18669.86    0.15874511    0.6499242   0.08571534
# hail                         0.08890267    0.09247879    16542.453   0.17341526    0.59990406  0.09952751
# hail                         0.10120925    0.10641503    14538.252   0.18982401    0.5499004   0.11399117
# hail                         0.11987173    0.12220042    12273.574   0.20805569    0.49987164  0.13112558
# hail                         0.13827227    0.14001511    10638.314   0.22659214    0.44984806  0.14941467
# hail                         0.15829067    0.15864913    9296.101    0.24626866    0.39983365  0.16864169
# hail                         0.18365733    0.17857906    8008.6064   0.26753646    0.34980217  0.18962103
# hail                         0.19555932    0.20265509    7520.521    0.2896        0.29979268  0.2174353
# hail                         0.2370742     0.23240577    6207.0747   0.32044885    0.24978767  0.24881782
# hail                         0.28289422    0.26606965    5198.3096   0.35140282    0.19975445  0.2847625
# hail                         0.3352303     0.30533865    4388.1494   0.3823156     0.14975406  0.328701
# hail                         0.38238993    0.35775426    3847.2227   0.41128483    0.09973774  0.39531642
# hail                         0.44512427    0.4873659     3285.0786   0.44512427    0.04971806  1.0
# sig_tornado                  1.2047162e-5  1.4437429e-5  5.0279775e6 0.00023340616 1.0         0.00046008435
# sig_tornado                  0.0015489991  0.000620892   39503.88    0.0072118156  0.94997233  0.0008245053
# sig_tornado                  0.0012606728  0.0012867345  48650.09    0.009076222   0.89943373  0.0019798223
# sig_tornado                  0.001833995   0.003183769   33270.344   0.014406291   0.8487792   0.0052594747
# sig_tornado                  0.00690796    0.006452647   8801.481    0.025394725   0.79838413  0.0078367125
# sig_tornado                  0.00995231    0.00908278    6115.261    0.03095475    0.74816865  0.010524806
# sig_tornado                  0.01157955    0.012097383   5262.185    0.03650292    0.69790304  0.013926878
# sig_tornado                  0.01346963    0.01593964    4508.503    0.04383515    0.6475773   0.018244676
# sig_tornado                  0.024001416   0.019918984   2547.455    0.054068234   0.5974216   0.021780202
# sig_tornado                  0.025120918   0.024143767   2422.7295   0.06113995    0.5469234   0.026980141
# sig_tornado                  0.03003723    0.03010828    2039.1814   0.07151833    0.49665758  0.033948936
# sig_tornado                  0.033580653   0.038780004   1817.3179   0.08479924    0.44606954  0.045137547
# sig_tornado                  0.05013078    0.05185326    1221.1023   0.10524846    0.395667    0.059880484
# sig_tornado                  0.080563776   0.06687405    760.7198    0.12545581    0.3451091   0.07437804
# sig_tornado                  0.09334492    0.08271323    656.9288    0.13874404    0.29449207  0.09200536
# sig_tornado                  0.11114166    0.10164373    552.5639    0.15433393    0.24384652  0.11262035
# sig_tornado                  0.12330799    0.12747169    498.82028   0.17187674    0.19312507  0.14483048
# sig_tornado                  0.14542684    0.1704235     417.73834   0.1999938     0.14232461  0.19986959
# sig_tornado                  0.30896956    0.21962947    196.64925   0.25134343    0.09215033  0.23971567
# sig_tornado                  0.20551312    0.27465805    247.26294   0.20551312    0.041969217 1.0
# sig_wind                     8.762574e-5   7.956794e-5   4.615673e6  0.0015565632  1.0         0.0010795709
# sig_wind                     0.0020825795  0.001814825   193981.45   0.013414203   0.94991076  0.0028963166
# sig_wind                     0.0039179027  0.0039368668  103105.52   0.019232223   0.8998797   0.005224411
# sig_wind                     0.005834378   0.006467304   69213.65    0.02498012    0.84985167  0.007945357
# sig_wind                     0.008835517   0.009320285   45742.48    0.03142871    0.7998408   0.01091732
# sig_wind                     0.011483245   0.012639817   35174.387   0.037897937   0.74978787  0.014694147
# sig_wind                     0.01744931    0.01655319    23148.463   0.045356132   0.69976497  0.01871097
# sig_wind                     0.020649865   0.021098858   19558.607   0.05172514    0.649741    0.02387907
# sig_wind                     0.025003763   0.027136791   16180.071   0.059148967   0.5997222   0.031004418
# sig_wind                     0.033851463   0.03475053    11945.531   0.06755926    0.5496192   0.038893435
# sig_wind                     0.039434146   0.043058835   10251.556   0.07505133    0.4995396   0.047464754
# sig_wind                     0.04473416    0.05181563    9045.808    0.08344653    0.44947395  0.056373745
# sig_wind                     0.06492861    0.0598693     6231.9253   0.093612395   0.39935932  0.063569665
# sig_wind                     0.065945596   0.06775919    6133.274    0.099947825   0.34924793  0.072085015
# sig_wind                     0.08764464    0.07611215    4606.5415   0.10939199    0.29915738  0.08034126
# sig_wind                     0.090064      0.08515918    4488.718    0.115124635   0.24915643  0.09031043
# sig_wind                     0.09548842    0.096013896   4236.5854   0.12378663    0.19908945  0.102228664
# sig_wind                     0.111371435   0.109113194   3625.3943   0.13748801    0.14898866  0.117370516
# sig_wind                     0.1494803     0.12685496    2704.816    0.15596397    0.09898441  0.13964514
# sig_wind                     0.16321121    0.18001986    2419.8394   0.16321121    0.04891188  1.0
# sig_hail                     4.2421325e-5  3.220337e-5   4.924094e6  0.0008022721  1.0         0.001318197
# sig_hail                     0.0025382582  0.0022416795  82272.734   0.015008614   0.9498081   0.0036093877
# sig_hail                     0.0043337564  0.005106818   48057.76    0.020673798   0.89962995  0.006999807
# sig_hail                     0.0067134313  0.008847184   31044.45    0.026576137   0.84958607  0.011068663
# sig_hail                     0.010588703   0.012986053   19688.49    0.032621574   0.79950756  0.015182176
# sig_hail                     0.017071232   0.016824745   12210.837   0.0378918     0.74941444  0.018555325
# sig_hail                     0.021248048   0.02004428    9799.877    0.041518603   0.6993265   0.02158984
# sig_hail                     0.026669838   0.022903282   7809.4185   0.04481297    0.6492928   0.02426124
# sig_hail                     0.025347624   0.025871573   8213.022    0.047512285   0.5992477   0.027540982
# sig_hail                     0.027341636   0.029312754   7630.691    0.051623642   0.54922545  0.031174533
# sig_hail                     0.030267406   0.033157837   6887.0146   0.05667976    0.49909386  0.035307348
# sig_hail                     0.037927397   0.03742463    5501.4717   0.0627922     0.4490064   0.039753683
# sig_hail                     0.048469674   0.041902665   4302.0317   0.068431295   0.3988698   0.044243563
# sig_hail                     0.0492872     0.04702346    4225.762    0.072734565   0.34876648  0.050134685
# sig_hail                     0.056120433   0.0533297     3716.3848   0.0790335     0.2987213   0.056848884
# sig_hail                     0.059499297   0.06103613    3501.9128   0.08612154    0.24860668  0.0658436
# sig_hail                     0.085206375   0.069975354   2444.3845   0.09707439    0.198541    0.07495049
# sig_hail                     0.095850974   0.08108813    2172.8608   0.10185565    0.14849555  0.088182054
# sig_hail                     0.11374573    0.09691916    1830.0184   0.10520578    0.09845164  0.1077964
# sig_hail                     0.09763601    0.1416834     2064.5598   0.09763601    0.04843512  1.0
# sig_tornado_gated_by_tornado 1.2044688e-5  1.4464373e-5  5.02901e6   0.00023340616 1.0         0.00046008435
# sig_tornado_gated_by_tornado 0.0015538402  0.00062055915 39380.8     0.0072588217  0.94997233  0.0008245053
# sig_tornado_gated_by_tornado 0.0012766533  0.0012856277  48041.113   0.0091455635  0.89943373  0.0019798223
# sig_tornado_gated_by_tornado 0.0018461995  0.0031905377  33050.406   0.014467288   0.8487792   0.0052594747
# sig_tornado_gated_by_tornado 0.0069257417  0.006443854   8780.715    0.025448764   0.79838413  0.007811773
# sig_tornado_gated_by_tornado 0.01077961    0.008947689   5649.973    0.03101798    0.74815816  0.010257343
# sig_tornado_gated_by_tornado 0.01272586    0.011599156   4794.479    0.035872545   0.6978566   0.013152388
# sig_tornado_gated_by_tornado 0.013809002   0.0149549795  4414.7456   0.041788157   0.64746463  0.017007226
# sig_tornado_gated_by_tornado 0.022520622   0.018557316   2726.78     0.050398786   0.59711456  0.020294942
# sig_tornado_gated_by_tornado 0.029579157   0.021911496   2063.1477   0.056941662   0.5463965   0.023793506
# sig_tornado_gated_by_tornado 0.030604053   0.026178267   2002.1079   0.06284972    0.49599442  0.028877527
# sig_tornado_gated_by_tornado 0.031029046   0.032508727   1955.7277   0.071397096   0.4453888   0.036711503
# sig_tornado_gated_by_tornado 0.04090295    0.04133125    1495.6295   0.08550168    0.39526904  0.047611304
# sig_tornado_gated_by_tornado 0.05159111    0.055509657   1186.4545   0.10176375    0.3447435   0.06523254
# sig_tornado_gated_by_tornado 0.079230666   0.073610626   772.46344   0.1221827     0.29418918  0.083278306
# sig_tornado_gated_by_tornado 0.08917256    0.09419476    687.0461    0.13766626    0.24364123  0.10639991
# sig_tornado_gated_by_tornado 0.092989005   0.12223298    660.36676   0.16055223    0.19304135  0.14342135
# sig_tornado_gated_by_tornado 0.15301146    0.16837253    396.83765   0.21664305    0.14232488  0.1984245
# sig_tornado_gated_by_tornado 0.34620395    0.21400294    176.84094   0.279994      0.092175096 0.22993922
# sig_tornado_gated_by_tornado 0.22719409    0.2585972     221.7547    0.22719409    0.041610427 1.0
# sig_wind_gated_by_wind       8.759892e-5   7.887138e-5   4.617086e6  0.0015565632  1.0         0.0010795709
# sig_wind_gated_by_wind       0.0020938707  0.0018146777  192935.42   0.013447442   0.94991076  0.0028963166
# sig_wind_gated_by_wind       0.003926863   0.003937269   102870.26   0.019250939   0.8998797   0.005224411
# sig_wind_gated_by_wind       0.0058410047  0.00646799    69135.125   0.024992133   0.84985167  0.007945357
# sig_wind_gated_by_wind       0.008835756   0.009320174   45741.242   0.031436898   0.7998408   0.01091732
# sig_wind_gated_by_wind       0.0114916805  0.012639401   35148.566   0.037910346   0.74978787  0.014694147
# sig_wind_gated_by_wind       0.017467922   0.016553186   23123.799   0.04536577    0.69976497  0.01871097
# sig_wind_gated_by_wind       0.020651773   0.021099042   19556.8     0.05172606    0.649741    0.02387907
# sig_wind_gated_by_wind       0.024994126   0.027136952   16186.31    0.059148967   0.5997222   0.031004418
# sig_wind_gated_by_wind       0.033828873   0.03475125    11953.508   0.06756567    0.5496192   0.038893435
# sig_wind_gated_by_wind       0.039471738   0.04305987    10241.793   0.075071186   0.4995396   0.047464754
# sig_wind_gated_by_wind       0.044756185   0.0518161     9041.356    0.08345507    0.44947395  0.056373745
# sig_wind_gated_by_wind       0.06492861    0.0598693     6231.9253   0.093612395   0.39935932  0.063569665
# sig_wind_gated_by_wind       0.065945596   0.06775919    6133.274    0.099947825   0.34924793  0.072085015
# sig_wind_gated_by_wind       0.08764464    0.07611215    4606.5415   0.10939199    0.29915738  0.08034126
# sig_wind_gated_by_wind       0.090064      0.08515918    4488.718    0.115124635   0.24915643  0.09031043
# sig_wind_gated_by_wind       0.09548842    0.096013896   4236.5854   0.12378663    0.19908945  0.102228664
# sig_wind_gated_by_wind       0.111371435   0.109113194   3625.3943   0.13748801    0.14898866  0.117370516
# sig_wind_gated_by_wind       0.1494803     0.12685496    2704.816    0.15596397    0.09898441  0.13964514
# sig_wind_gated_by_wind       0.16321121    0.18001986    2419.8394   0.16321121    0.04891188  1.0
# sig_hail_gated_by_hail       4.241907e-5   3.2193075e-5  4.924356e6  0.0008022721  1.0         0.001318197
# sig_hail_gated_by_hail       0.002543777   0.0022429482  82094.24    0.01502357    0.9498081   0.0036093877
# sig_hail_gated_by_hail       0.0043354956  0.005106204   48038.484   0.020683357   0.89962995  0.006999807
# sig_hail_gated_by_hail       0.0067161745  0.008847436   31031.768   0.02658901    0.84958607  0.011068663
# sig_hail_gated_by_hail       0.010594521   0.012986352   19677.68    0.032638125   0.79950756  0.015182176
# sig_hail_gated_by_hail       0.017215528   0.016809182   12091.6045  0.037910648   0.74941444  0.018515944
# sig_hail_gated_by_hail       0.021046346   0.020027013   9907.607    0.041476414   0.6993964   0.02158984
# sig_hail_gated_by_hail       0.026670078   0.022903148   7809.348    0.044834845   0.6492928   0.02426124
# sig_hail_gated_by_hail       0.02535061    0.025872072   8212.055    0.047538865   0.5992477   0.027540982
# sig_hail_gated_by_hail       0.027338756   0.029313533   7631.4946   0.051656753   0.54922545  0.031174533
# sig_hail_gated_by_hail       0.030258834   0.033157654   6888.9653   0.056724932   0.49909386  0.035307348
# sig_hail_gated_by_hail       0.037934076   0.03742634    5500.503    0.062857956   0.4490064   0.039753683
# sig_hail_gated_by_hail       0.048634436   0.04190139    4287.4575   0.0685165     0.3988698   0.044243563
# sig_hail_gated_by_hail       0.049389396   0.04702226    4217.0186   0.07279144    0.34876648  0.050134685
# sig_hail_gated_by_hail       0.056194518   0.053331286   3711.485    0.07906792    0.2987213   0.056848884
# sig_hail_gated_by_hail       0.059532415   0.061038014   3499.9646   0.08613551    0.24860668  0.0658436
# sig_hail_gated_by_hail       0.085206375   0.069975354   2444.3845   0.09707439    0.198541    0.07495049
# sig_hail_gated_by_hail       0.095850974   0.08108813    2172.8608   0.10185565    0.14849555  0.088182054
# sig_hail_gated_by_hail       0.11374573    0.09691916    1830.0184   0.10520578    0.09845164  0.1077964
# sig_hail_gated_by_hail       0.09763601    0.1416834     2064.5598   0.09763601    0.04843512  1.0




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
# (here they are because we do not do any fancy subsetting)
function spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
  event_name, _ = HREFPrediction.models[prediction_i]
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
  event_name, _ = HREFPrediction.models[prediction_i]
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
  event_name, _ = HREFPrediction.models[prediction_i]
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
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_sr_pod[event_name] = spc_calibrate_sr_pod(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end
println("event_name\tnominal_prob\tthreshold_to_match_success_ratio\tthreshold_to_match_POD\tthreshold_to_match_warning_ratio\tmean_threshold\tSR\tPOD\tWR")
calibrations_all = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPrediction.models)
  event_name, _ = HREFPrediction.models[prediction_i]
  calibrations_all[event_name] = spc_calibrate_all(prediction_i, X, Ys, weights)
end

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD mean_threshold SR          POD          WR
# tornado     0.02         0.01297058                       0.019601822            0.016286202    0.05811968  0.7255002    0.02133395
# tornado     0.05         0.05831237                       0.079683304            0.06899784     0.13066258  0.4559308    0.0059635467
# tornado     0.1          0.16381283                       0.15509605             0.15945444     0.21934602  0.14991193   0.0011680551
# tornado     0.15         0.21046689                       0.26961708             0.24004199     0.30903193  0.06797499   0.00037592635
# tornado     0.3          0.3783576                        0.3603382              0.3693479      0.2993795   0.008200346  4.681306e-5
# tornado     0.45         0.63601375                       0.5975361              0.6167749      0.54525286  0.0006632183 2.0788132e-6
# wind        0.05         0.025938107                      0.0877018              0.056819953    0.19041657  0.8146683    0.05523891
# wind        0.15         0.11240294                       0.25773048             0.18506671     0.34040433  0.5379147    0.020402689
# wind        0.3          0.31343934                       0.521204               0.41732168     0.5510731   0.19818887   0.004643432
# wind        0.45         0.5346906                        0.8000431              0.66736686     0.769832    0.042777162  0.0007174391
# hail        0.05         0.015346069                      0.04394722             0.029646644    0.106049754 0.8351591    0.044649545
# hail        0.15         0.07387878                       0.13708305             0.10548092     0.19760264  0.52892953   0.015176184
# hail        0.3          0.20529476                       0.3468647              0.2760797      0.37571222  0.16215828   0.0024470412
# hail        0.45         0.5594928                        0.59866524             0.57907903     0.58626467  0.009759113  9.437872e-5
# sig_tornado 0.1          0.040588986                      0.07995033             0.06026966     0.12599954  0.3443003    0.0006377945
# sig_wind    0.1          0.098119505                      0.106687546            0.10240352     0.13775808  0.14839154   0.0016767136
# sig_hail    0.1          0.059057                         0.051927567            0.055492282    0.08464337  0.25743744   0.0024400598

# this is the one we are using
# event_name  nominal_prob threshold_to_match_warning_ratio SR          POD          WR
# tornado     0.02         0.017892838                      0.06162817  0.7073454    0.019615944
# tornado     0.05         0.07787514                       0.14183877  0.4307434    0.0051901583
# tornado     0.1          0.17152214                       0.2464791   0.14091049   0.0009770577
# tornado     0.15         0.2814541                        0.26955867  0.026589876  0.0001685854
# tornado     0.3          0.3905239                        0.34836945  0.007654115  3.7550166e-5
# tornado     0.45         0.6009083                        0.47030112  0.0008843078 3.213545e-6
# wind        0.05         0.051660538                      0.18246952  0.8284546    0.058620222
# wind        0.15         0.21513557                       0.3675667   0.48940307   0.017190937
# wind        0.3          0.49578285                       0.6181463   0.13029508   0.0027214838
# wind        0.45         0.78172493                       0.8917094   0.020278241  0.0002936135
# hail        0.05         0.030927658                      0.108049504 0.8288017    0.04348959
# hail        0.15         0.12172127                       0.21620022  0.47577965   0.012476915
# hail        0.3          0.33656883                       0.41593274  0.09215542   0.0012561898
# hail        0.45         0.61953926                       0.6029355   0.007252866  6.820187e-5
# sig_tornado 0.1          0.063589096                      0.129504    0.33463305   0.00060311204
# sig_wind    0.1          0.11205864                       0.15010695  0.11518542   0.0011944375
# sig_hail    0.1          0.057775497                      0.08727195  0.242912     0.0022330375

# event_name  nominal_prob threshold_to_match_success_ratio threshold_to_match_POD threshold_to_match_warning_ratio mean_threshold SR         POD          WR
# tornado     0.02         0.01297058                       0.019601822            0.017892838                      0.016821748    0.05935366 0.7197565    0.020725023
# tornado     0.05         0.05831237                       0.079683304            0.07787514                       0.07195694     0.13404815 0.44696563   0.0056986273
# tornado     0.1          0.16381283                       0.15509605             0.17152214                       0.163477       0.22779244 0.1466783    0.0011004834
# tornado     0.15         0.21046689                       0.26961708             0.2814541                        0.25384602     0.30016658 0.051623434  0.00029392834
# tornado     0.3          0.3783576                        0.3603382              0.3905239                        0.37640658     0.31507045 0.007982044  4.3297554e-5
# tornado     0.45         0.63601375                       0.5975361              0.6009083                        0.611486       0.4998252  0.0006632183 2.2677507e-6
# wind        0.05         0.025938107                      0.0877018              0.051660538                      0.055100147    0.1878509  0.8194134    0.056319505
# wind        0.15         0.11240294                       0.25773048             0.21513557                       0.19508965     0.349676   0.52065736   0.019224508
# wind        0.3          0.31343934                       0.521204               0.49578285                       0.4434754      0.5752322  0.17310232   0.0038853372
# wind        0.45         0.5346906                        0.8000431              0.78172493                       0.70548624     0.8269018  0.03264143   0.00050966436
# hail        0.05         0.015346069                      0.04394722             0.030927658                      0.030073648    0.10670167 0.8331348    0.04426918
# hail        0.15         0.07387878                       0.13708305             0.12172127                       0.11089436     0.20442617 0.509736     0.014137293
# hail        0.3          0.20529476                       0.3468647              0.33656883                       0.29624274     0.38949108 0.13536102   0.0019703961
# hail        0.45         0.5594928                        0.59866524             0.61953926                       0.5925658      0.58640826 0.008812057  8.519903e-5
# sig_tornado 0.1          0.040588986                      0.07995033             0.063589096                      0.06137614     0.1267123  0.3394527    0.0006252775
# sig_wind    0.1          0.098119505                      0.106687546            0.11205864                       0.1056219      0.14019963 0.13421097   0.0014900743
# sig_hail    0.1          0.059057                         0.051927567            0.057775497                      0.056253355    0.0856354  0.25276944   0.002368061


println(calibrations_sr_pod)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.055492282)], "hail" => [(0.05, 0.029646644), (0.15, 0.10548092), (0.3, 0.2760797), (0.45, 0.57907903)], "tornado" => [(0.02, 0.016286202), (0.05, 0.06899784), (0.1, 0.15945444), (0.15, 0.24004199), (0.3, 0.3693479), (0.45, 0.6167749)], "sig_tornado" => [(0.1, 0.06026966)], "sig_wind" => [(0.1, 0.10240352)], "wind" => [(0.05, 0.056819953), (0.15, 0.18506671), (0.3, 0.41732168), (0.45, 0.66736686)])

println(calibrations_wr)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.057775497)], "hail" => [(0.05, 0.030927658), (0.15, 0.12172127), (0.3, 0.33656883), (0.45, 0.61953926)], "tornado" => [(0.02, 0.017892838), (0.05, 0.07787514), (0.1, 0.17152214), (0.15, 0.2814541), (0.3, 0.3905239), (0.45, 0.6009083)], "sig_tornado" => [(0.1, 0.063589096)], "sig_wind" => [(0.1, 0.11205864)], "wind" => [(0.05, 0.051660538), (0.15, 0.21513557), (0.3, 0.49578285), (0.45, 0.78172493)])

println(calibrations_all)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_hail" => [(0.1, 0.056253355)], "hail" => [(0.05, 0.030073648), (0.15, 0.11089436), (0.3, 0.29624274), (0.45, 0.5925658)], "tornado" => [(0.02, 0.016821748), (0.05, 0.07195694), (0.1, 0.163477), (0.15, 0.25384602), (0.3, 0.37640658), (0.45, 0.611486)], "sig_tornado" => [(0.1, 0.06137614)], "sig_wind" => [(0.1, 0.1056219)], "wind" => [(0.05, 0.055100147), (0.15, 0.19508965), (0.3, 0.4434754), (0.45, 0.70548624)])


# using the warning ratio calibrations rn
