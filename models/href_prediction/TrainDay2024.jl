import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grid130
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction2024.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2024, 2, 28, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 1010

validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_forecasts);
length(validation_forecasts_0z_12z) # 505

@time Forecasts.data(validation_forecasts[10]); # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

# rm("day_accumulators_validation_forecasts_0z_12z_2024"; recursive = true)
ENV["FORECAST_DISK_PREFETCH"] = "false"

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts_0z_12z;
    event_name_to_labeler = TrainingShared.event_name_to_day_labeler(1),
    save_dir = "day_accumulators_validation_forecasts_0z_12z_2024",
  );

# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    event_name, _ = HREFPrediction2024.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z_12z, X, Ys, weights)

# tornado (26090.0)             feature 1 independent events total TORPROB:calculated:day   fcst:: AU-PR-curve: 0.15364023
# tornado (26090.0)             feature 2 highest hourly TORPROB:calculated:day             fcst:: AU-PR-curve: 0.14157175
# wind (210196.0)               feature 3 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.42069083
# wind (210196.0)               feature 4 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.39926273
# wind_adj (76921.27)           feature 5 independent events total WINDPROB:calculated:day  fcst:: AU-PR-curve: 0.26403555
# wind_adj (76921.27)           feature 6 highest hourly WINDPROB:calculated:day            fcst:: AU-PR-curve: 0.25709826
# hail (97059.0)                feature 7 independent events total HAILPROB:calculated:day  fcst:: AU-PR-curve: 0.29317638
# hail (97059.0)                feature 8 highest hourly HAILPROB:calculated:day            fcst:: AU-PR-curve: 0.27437192
# sig_tornado (3437.0)          feature 9 independent events total STORPROB:calculated:day  fcst:: AU-PR-curve: 0.08517615
# sig_tornado (3437.0)          feature 10 highest hourly STORPROB:calculated:day           fcst:: AU-PR-curve: 0.08757988  **not for sigtor**
# sig_wind (23922.0)            feature 11 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.09506313
# sig_wind (23922.0)            feature 12 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.090158425
# sig_wind_adj (9466.223)       feature 13 independent events total SWINDPRO:calculated:day fcst:: AU-PR-curve: 0.07771641
# sig_wind_adj (9466.223)       feature 14 highest hourly SWINDPRO:calculated:day           fcst:: AU-PR-curve: 0.080509216 **not for sigwind_adj**
# sig_hail (14244.0)            feature 15 independent events total SHAILPRO:calculated:day fcst:: AU-PR-curve: 0.09916906
# sig_hail (14244.0)            feature 16 highest hourly SHAILPRO:calculated:day           fcst:: AU-PR-curve: 0.09328117
# tornado_life_risk (990.11566) feature 17 independent events total TORPROB:calculated:day  fcst:: AU-PR-curve: 0.022747247
# tornado_life_risk (990.11566) feature 18 highest hourly TORPROB:calculated:day            fcst:: AU-PR-curve: 0.026257461 **not for tor_life_risk**


accs_event_names = map(i -> HREFPrediction2024.models[div(i - 1, 2) + 1][1], 1:size(X,2))
Metrics.reliability_curves_midpoints(20, X, Ys, accs_event_names, weights, map(i -> accs_event_names[i] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2)))

# ŷ_tornado_tot,y_tornado_tot,ŷ_tornado_max,y_tornado_max,ŷ_wind_tot,y_wind_tot,ŷ_wind_max,y_wind_max,ŷ_wind_adj_tot,y_wind_adj_tot,ŷ_wind_adj_max,y_wind_adj_max,ŷ_hail_tot,y_hail_tot,ŷ_hail_max,y_hail_max,ŷ_sig_tornado_tot,y_sig_tornado_tot,ŷ_sig_tornado_max,y_sig_tornado_max,ŷ_sig_wind_tot,y_sig_wind_tot,ŷ_sig_wind_max,y_sig_wind_max,ŷ_sig_wind_adj_tot,y_sig_wind_adj_tot,ŷ_sig_wind_adj_max,y_sig_wind_adj_max,ŷ_sig_hail_tot,y_sig_hail_tot,ŷ_sig_hail_max,y_sig_hail_max,ŷ_tornado_life_risk_tot,y_tornado_life_risk_tot,ŷ_tornado_life_risk_max,y_tornado_life_risk_max,
# 0.00013071201,0.000105973515,2.6413798e-5,0.000106287946,0.0011933012,0.0008706149,0.00026807308,0.0008721359,0.00037108394,0.00030911202,8.603522e-5,0.00030934627,0.000593545,0.00038677963,0.0001301142,0.0003878069,1.8206321e-5,1.3231731e-5,4.3571463e-6,1.3223712e-5,0.00010897409,9.6873315e-5,2.3256522e-5,9.6544674e-5,4.5361256e-5,3.6883346e-5,8.175823e-6,3.710795e-5,8.375501e-5,5.483908e-5,1.8875277e-5,5.49677e-5,4.5000256e-6,3.7870495e-6,8.3041414e-7,3.84383e-6,
# 0.0034398965,0.0022950731,0.00075324805,0.0024203553,0.030836979,0.023130938,0.007345092,0.022982396,0.009905607,0.008707941,0.0024982744,0.008453472,0.019028323,0.014894701,0.0043439474,0.014076489,0.0012364653,0.0014087878,0.000346016,0.0013287495,0.0027575556,0.0028212203,0.0006677814,0.002861465,0.0013959896,0.0011138907,0.00029330543,0.0010054172,0.004641731,0.003323312,0.0010827899,0.003468994,0.00039497294,0.0011211361,8.723687e-5,0.00060640386,
# 0.007874757,0.0062128357,0.0017398694,0.005407214,0.054471314,0.042907584,0.013115944,0.042865016,0.018763993,0.015486136,0.004693484,0.016778177,0.03300595,0.025318174,0.0075618783,0.026488276,0.0022890721,0.0027739084,0.0006546686,0.0025601475,0.005612323,0.0039627356,0.0012887637,0.0048553757,0.0028973934,0.002487441,0.00065595855,0.0021551445,0.008313145,0.007924371,0.0019007972,0.0070337765,0.00073115627,0.00077519484,0.00019321927,0.00063188653,
# 0.012869635,0.010937977,0.003049521,0.00995858,0.07860969,0.062419094,0.01916756,0.064616464,0.029469743,0.02456989,0.0074116727,0.02475319,0.047605217,0.037926044,0.010921077,0.03821808,0.0038550205,0.0028593028,0.0010527926,0.003116909,0.0097731585,0.0068987673,0.0021623548,0.0070537804,0.004641693,0.0043900684,0.0011387763,0.0045684017,0.012197766,0.00915104,0.0028881629,0.00924493,0.0013028318,0.0018704502,0.00037339109,0.0013003997,
# 0.018895153,0.014974584,0.0046446514,0.013820899,0.10446469,0.08419996,0.025814496,0.08489553,0.041641995,0.03604751,0.010865899,0.035998117,0.06324161,0.048786547,0.014542563,0.050897464,0.0063014464,0.0050019957,0.0017246696,0.0036783193,0.014559366,0.011690625,0.0033670512,0.010537206,0.006868502,0.005433538,0.0017648364,0.005604504,0.017209942,0.012824128,0.004212888,0.011878614,0.0019037123,0.0023804419,0.0006139345,0.0016209031,
# 0.025733469,0.024040643,0.006550437,0.02066039,0.13198334,0.106434666,0.03287458,0.110838085,0.054405347,0.049779028,0.014562155,0.04907612,0.08028658,0.06367055,0.018462451,0.0637917,0.00917395,0.007839186,0.0026289958,0.006832724,0.019807601,0.016316894,0.0048428583,0.014640373,0.009938073,0.0073637115,0.0026842963,0.006477686,0.022510055,0.018298406,0.0056282333,0.01683281,0.002725259,0.0029599487,0.0009109646,0.003598886,
# 0.033515234,0.027966281,0.008779072,0.029298618,0.16052741,0.13427238,0.04023952,0.13236272,0.068245694,0.059327964,0.01842116,0.059046153,0.098013766,0.08200476,0.022666253,0.08124341,0.012622686,0.009707933,0.003782825,0.00873021,0.025511682,0.021797886,0.0066199577,0.01952981,0.013557615,0.011396195,0.0039914767,0.010933789,0.028029231,0.022215411,0.0069244807,0.021953467,0.0037117284,0.004544098,0.0012249151,0.004592406,
# 0.043415662,0.036649365,0.01147128,0.037749693,0.18998775,0.16120477,0.04850637,0.15366499,0.084338985,0.07274471,0.02278049,0.06913192,0.11630628,0.09991746,0.027196068,0.0941238,0.017004212,0.011915207,0.005829417,0.010590713,0.031723548,0.025141994,0.008588717,0.02612975,0.017531356,0.014787089,0.0054283123,0.014928653,0.033580553,0.033173744,0.008320962,0.026270768,0.005029176,0.0041523324,0.0016435361,0.004762982,
# 0.054636564,0.046994347,0.014706811,0.04679715,0.2207198,0.18617651,0.057589207,0.17895034,0.10231962,0.08568902,0.028014597,0.082982615,0.13596807,0.115353785,0.03240182,0.105965175,0.022959156,0.015422772,0.0086701345,0.01866384,0.038165648,0.035154138,0.010299523,0.03533699,0.02208613,0.017480198,0.006721831,0.023180492,0.03938943,0.03487544,0.009986512,0.030997492,0.006563869,0.007693057,0.0022144034,0.00572185,
# 0.06682123,0.05854597,0.018048663,0.06115403,0.25222933,0.22347955,0.06712022,0.20836392,0.12135502,0.1076369,0.033817824,0.10262151,0.15741757,0.1328504,0.03847533,0.12422891,0.030478511,0.022156255,0.011494208,0.025157118,0.04418931,0.04288436,0.011609818,0.042534474,0.027249541,0.022503108,0.0077878204,0.025328167,0.04680266,0.03747644,0.012005306,0.037095875,0.007754174,0.011019658,0.0028456773,0.008202303,
# 0.079937115,0.07254469,0.02150665,0.06353087,0.28467327,0.25183487,0.077042006,0.23903023,0.14136435,0.12642543,0.040004432,0.123924956,0.18057072,0.15548779,0.04540984,0.14365478,0.038629994,0.031931862,0.015029683,0.02910596,0.050575152,0.046252355,0.012771878,0.049089108,0.03260516,0.028979698,0.0088523505,0.030115012,0.055742733,0.045571946,0.014625907,0.040894236,0.008892076,0.013508087,0.0034279106,0.009810271,
# 0.094362915,0.08281355,0.025152752,0.07489564,0.31842157,0.28637195,0.08829701,0.26580426,0.16321658,0.14419422,0.046876386,0.1412328,0.20551325,0.18076888,0.053517807,0.15899509,0.050270986,0.029504554,0.020093972,0.03622357,0.057864998,0.051409673,0.01412849,0.05811425,0.03818489,0.035122138,0.010101735,0.03543988,0.06577007,0.058678932,0.017822374,0.054333035,0.010141489,0.013641441,0.004030835,0.010925618,
# 0.11090447,0.09890911,0.029648462,0.08286222,0.3538138,0.32426164,0.10148051,0.29154295,0.18831344,0.16467251,0.055134006,0.15485397,0.23465064,0.18974227,0.06309662,0.17513561,0.06669609,0.04557049,0.026566798,0.049203645,0.06622844,0.058136884,0.015760787,0.05966995,0.044897012,0.03608706,0.011661432,0.0337521,0.07649431,0.07432564,0.021180479,0.065811925,0.011920225,0.011800447,0.004758424,0.013774947,
# 0.1301335,0.11576474,0.035377022,0.095020585,0.39245734,0.3567607,0.116882265,0.32846418,0.21749295,0.18719819,0.06541225,0.1742664,0.2693404,0.21090387,0.07504652,0.197491,0.083911225,0.09093597,0.03250091,0.076997034,0.075647086,0.070307836,0.01790244,0.06573107,0.052542984,0.05058342,0.013701443,0.043590408,0.0872755,0.08944394,0.024658471,0.07542989,0.014608451,0.012210987,0.005765751,0.014971146,
# 0.15210804,0.1388956,0.042810842,0.11637836,0.43797967,0.38082585,0.1354684,0.3634665,0.2506319,0.22703624,0.078119315,0.20031676,0.30924204,0.24772413,0.09049057,0.22115241,0.101369254,0.09633962,0.03697887,0.10945904,0.086476795,0.07447277,0.020810183,0.072735325,0.06082842,0.06148475,0.016293326,0.049129855,0.098739035,0.10617108,0.02818303,0.09168808,0.01823743,0.015980761,0.006930852,0.023166003,
# 0.1781461,0.14764228,0.052300908,0.13803208,0.49047902,0.43756333,0.15868221,0.4056845,0.28723162,0.26758605,0.09362913,0.2438496,0.35615167,0.27909186,0.11008994,0.25885156,0.12166169,0.107393496,0.04107911,0.14032876,0.09955829,0.09936273,0.02500206,0.07614243,0.06960657,0.08949725,0.019962719,0.06505992,0.111971915,0.11469013,0.032514088,0.09548058,0.022366054,0.030438676,0.0079952795,0.02596369,
# 0.21355756,0.17092665,0.06342409,0.17965086,0.54963136,0.4976388,0.18830085,0.46876037,0.3301365,0.3185745,0.112703174,0.29107046,0.41225338,0.3276267,0.13407081,0.31111562,0.15267001,0.09805571,0.04641249,0.13776793,0.114697665,0.11477974,0.03145068,0.08369637,0.08095851,0.082898766,0.024711147,0.09012246,0.12802981,0.13130532,0.038256668,0.11248792,0.027424423,0.027376672,0.0096053,0.02642898,
# 0.26478228,0.18886343,0.078341655,0.21330294,0.6192362,0.5746606,0.22847356,0.5433541,0.38412544,0.3519748,0.14004238,0.34542647,0.47930038,0.4038802,0.16605625,0.36960846,0.19069499,0.15078636,0.055454634,0.114561155,0.13628344,0.120389715,0.041846376,0.12076718,0.09989021,0.091598764,0.032615844,0.1089524,0.14896841,0.14173195,0.046236526,0.12838778,0.040150873,0.017710341,0.011579149,0.050626118,
# 0.35058835,0.28082317,0.10412107,0.24916835,0.7059387,0.6557058,0.29126474,0.6368425,0.4703104,0.4276527,0.1861615,0.43685278,0.5658299,0.4959459,0.21785931,0.45231918,0.2278801,0.20463267,0.07309562,0.12023377,0.18022108,0.12608752,0.059095304,0.157223,0.13530964,0.12668933,0.047874987,0.16987482,0.18176153,0.14552975,0.05857791,0.15521245,0.064962626,0.027251951,0.013856659,0.04666784,
# 0.53479123,0.38427362,0.16697188,0.34419817,0.8391513,0.7778187,0.43431267,0.7469428,0.6415819,0.62137777,0.3046588,0.6183982,0.72780436,0.6350179,0.3461906,0.6069767,0.30602497,0.14084196,0.11014444,0.20801184,0.27369377,0.25170532,0.09946808,0.24613824,0.19682068,0.22557011,0.076438166,0.26716304,0.25974038,0.18043515,0.09180409,0.21173047,0.09962888,0.06316842,0.022722468,0.066423215,



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

event_types_count = length(HREFPrediction2024.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:event_types_count
  (event_name, _, model_name) = HREFPrediction2024.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# event_name        mean_y        mean_ŷ        Σweight     bin_max
# tornado           0.0004903907  0.0006316218  1.2463283e7 0.022570612
# tornado           0.034961343   0.040241364   174838.86   0.07342326
# tornado           0.09643957    0.10764161    63381.816   0.16399916
# tornado           0.20727663    0.26939303    29478.082   1.0
# wind              0.0040222933  0.005207318   1.2166085e7 0.11850963
# wind              0.1519959     0.18022498    321953.9    0.26850948
# wind              0.3129526     0.349497      156368.45   0.46391508
# wind              0.5652367     0.6170349     86574.22    1.0
# wind_adj          0.0014353079  0.0016948527  1.2320715e7 0.0483164
# wind_adj          0.06976881    0.07988042    253464.0    0.13139023
# wind_adj          0.16303475    0.18454023    108465.53   0.26873532
# wind_adj          0.36581174    0.3901795     48337.367   1.0
# hail              0.0018252683  0.0024339498  1.2313709e7 0.0720079
# hail              0.092403986   0.11058817    243226.16   0.16906372
# hail              0.19221757    0.2329425     116928.55   0.33150575
# hail              0.39344996    0.47285366    57118.066   1.0
# sig_tornado       6.4511805e-5  7.350612e-5   1.2637329e7 0.007751094
# sig_tornado       0.011721554   0.015853921   69592.36    0.03453768
# sig_tornado       0.04584312    0.057997108   17799.207   0.11010684
# sig_tornado       0.12990344    0.18684725    6261.6133   1.0
# sig_wind          0.0004479206  0.00055357994 1.2372047e7 0.017204525
# sig_wind          0.025139851   0.028981635   220424.05   0.04728317
# sig_wind          0.05815138    0.06510016    95301.46    0.093053006
# sig_wind          0.12821203    0.14482208    43209.223   1.0
# sig_wind_adj      0.00017337847 0.00020699644 1.248539e7  0.008334742
# sig_wind_adj      0.012707194   0.01575545    170398.77   0.029957727
# sig_wind_adj      0.039365016   0.04320339    55009.785   0.06545183
# sig_wind_adj      0.107183434   0.10260313    20183.412   1.0
# sig_hail          0.0002651922  0.00035253924 1.2535268e7 0.019996995
# sig_hail          0.026968027   0.03174356    123284.016  0.05117618
# sig_hail          0.06840354    0.07219607    48600.63    0.10483167
# sig_hail          0.13939083    0.15851603    23829.717   1.0
# tornado_life_risk 1.8646539e-5  1.4896649e-5  1.2656379e7 0.0022878477
# tornado_life_risk 0.004899044   0.004401102   48176.97    0.008275934
# tornado_life_risk 0.013228325   0.012501984   17877.422   0.020291256
# tornado_life_risk 0.027496696   0.043680158   8548.56     1.0


println("event_to_day_bins = $event_to_day_bins")
event_to_day_bins = Dict{String, Vector{Float32}}("sig_wind" => [0.017204525, 0.04728317, 0.093053006, 1.0], "sig_hail" => [0.019996995, 0.05117618, 0.10483167, 1.0], "hail" => [0.0720079, 0.16906372, 0.33150575, 1.0], "sig_wind_adj" => [0.008334742, 0.029957727, 0.06545183, 1.0], "tornado_life_risk" => [0.0022878477, 0.008275934, 0.020291256, 1.0], "tornado" => [0.022570612, 0.07342326, 0.16399916, 1.0], "wind_adj" => [0.0483164, 0.13139023, 0.26873532, 1.0], "sig_tornado" => [0.007751094, 0.03453768, 0.11010684, 1.0], "wind" => [0.11850963, 0.26850948, 0.46391508, 1.0])


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

    Threads.@threads :static for i in 1:length(bin_y)
      logit_total_prob = logit(bin_total_prob_x[i])
      logit_max_hourly = logit(bin_max_hourly_x[i])

      bin_X_features[i,1] = logit_total_prob
      bin_X_features[i,2] = logit_max_hourly
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

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
  event_name, _ = HREFPrediction2024.models[prediction_i]

  event_to_day_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
end

# event_name        bin total_prob_ŷ_min total_prob_ŷ_max count    pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado           1-2 -1.0             0.07342326       13714495 13139.0   1.2638122e7 0.0011795927      0.00032357432     0.0009672702  0.0052419323       0.0057224673       0.034310866      0.030124178      0.00096727035   0.0052199434     0.034458082    Float32[1.1039704,   -0.07336604,  -0.1759734]
# tornado           2-3 0.022570612      0.16399916       254093   13064.0   238220.67   0.05817411        0.017037176       0.051318455   0.19390126         0.219888           1.0412586        0.08888261       0.051318455     0.1934424        0.09758978     Float32[0.980029,    0.02896403,   -0.07270332]
# tornado           3-4 0.07342326       1.0              98265    12951.0   92859.9     0.15898909        0.04583053        0.13162445    0.37570423         0.43668777         0.23870525       0.2210763        0.13162443      0.371172         0.24002679     Float32[0.68161184,  0.16218366,   -0.23822184]
# wind              1-2 -1.0             0.26850948       13551673 105118.0  1.2488039e7 0.009719445       0.0026224114      0.007837199   0.028448831        0.032177273        0.14904602       0.132905         0.007837199     0.028217126      0.14843512     Float32[0.9594039,   0.081496894,  -0.023889013]
# wind              2-3 0.11850963       0.46391508       513311   105019.0  478322.38   0.23556171        0.068605445       0.2046143     0.4870253          0.59682643         0.30879617       0.28369257       0.2046143       0.4841071        0.30866596     Float32[1.064942,    -0.029093888, -0.19427404]
# wind              3-4 0.26850948       1.0              261087   105078.0  242942.66   0.4448359         0.15029909        0.4028557     0.63432956         0.8456853          0.5868932        0.56242687       0.40285575      0.62972486       0.5878054      Float32[0.8065231,   0.14930426,   0.047687516]
# wind_adj          1-2 -1.0             0.13139023       13641640 38278.812 1.2574179e7 0.003270878       0.00091538904     0.0028127404  0.0126735885       0.01402982         0.06767361       0.062905505      0.0028127402    0.012629514      0.06801478     Float32[0.8024022,   0.21128672,   0.17552774]
# wind_adj          2-3 0.0483164        0.26873532       392451   38429.992 361929.56   0.11124557        0.033394683       0.09771938    0.3082543          0.35537562         0.16483887       0.1551465        0.09771938      0.30724257       0.1655415      Float32[0.92636955,  0.059434433,  -0.096047774]
# wind_adj          3-4 0.13139023       1.0              171120   38642.465 156802.9    0.24793229        0.08292798        0.22554447    0.5004063          0.6055132          0.39981365       0.3928999        0.22554444      0.4985609        0.40172863     Float32[0.8363115,   0.19665669,   0.17679109]
# hail              1-2 -1.0             0.16906372       13624453 48620.0   1.2556936e7 0.004528883       0.0011598516      0.0035797658  0.014942365        0.016732348        0.112831734      0.078529455      0.0035797665    0.014816889      0.09058503     Float32[1.0999736,   -0.036068548, -0.11684635]
# hail              2-3 0.0720079        0.33150575       389752   48567.0   360154.7    0.15031199        0.041157532       0.124809675   0.36642092         0.43309677         0.19213499       0.17095426       0.124809675     0.36331648       0.19445087     Float32[1.1508383,   -0.20880625,  -0.63325006]
# hail              3-4 0.16906372       1.0              188307   48439.0   174046.62   0.3116758         0.10054249        0.25825736    0.5454031          0.65963995         0.42741445       0.40301725       0.25825736      0.5371891        0.42780825     Float32[1.0397497,   -0.15485394,  -0.61086625]
# sig_tornado       1-2 -1.0             0.03453768       13787545 1739.0    1.2706921e7 0.00015993117     5.154383e-5       0.00012835427 0.0008206348       0.00086264         0.011853862      0.011924904      0.00012835427   0.000816321      0.012620237    Float32[0.71196026,  0.28164828,   0.08433733]
# sig_tornado       2-3 0.007751094      0.11010684       92183    1717.0    87391.57    0.024437303       0.008624038       0.018671157   0.08792095         0.090660736        0.093059644      0.06838807       0.018671157     0.08612125       0.06827668     Float32[0.17950201,  0.808518,     0.56035143]
# sig_tornado       3-4 0.03453768       1.0              25215    1698.0    24060.82    0.0915292         0.030761085       0.067719065   0.23847069         0.24936514         0.1408722        0.14529932       0.067719065     0.23130949       0.14684357     Float32[0.2779992,   0.7922331,    0.76261055]
# sig_wind          1-2 -1.0             0.04728317       13663342 11975.0   1.2592472e7 0.0010511968      0.0002681011      0.0008801388  0.0049687484       0.005408804        0.025094489      0.02283702       0.0008801388    0.004946576      0.024869526    Float32[0.63392025,  0.35701632,   0.2873989]
# sig_wind          2-3 0.017204525      0.093053006      339966   11963.0   315725.5    0.03988398        0.010546717       0.035104353   0.14789812         0.16642801         0.05809444       0.05732523       0.03510435      0.14740734       0.05951839     Float32[0.7724101,   0.3031301,    0.52364856]
# sig_wind          3-4 0.04728317       1.0              149418   11947.0   138510.69   0.08996988        0.024306877       0.08000722    0.2701287          0.31330404         0.14461458       0.13709478       0.08000721      0.26902694       0.14370498     Float32[0.61689895,  0.30869296,   0.15061612]
# sig_wind_adj      1-2 -1.0             0.029957727      13730457 4717.7437 1.2655788e7 0.00041634237     0.000111179965    0.00034213497 0.002178409        0.002327409        0.012739849      0.013333386      0.00034213494   0.0021659562     0.013525564    Float32[0.45143408,  0.5053786,    0.26962078]
# sig_wind_adj      2-3 0.008334742      0.06545183       245371   4731.1606 225408.55   0.022453977       0.006319634       0.019212896   0.09117215         0.09947667         0.04071039       0.04016245       0.019212894     0.09060983       0.04247328     Float32[0.59369177,  0.5571225,    1.1116248]
# sig_wind_adj      3-4 0.029957727      1.0              82303    4748.4795 75193.2     0.059147503       0.016641133       0.05756888    0.21001509         0.2401911          0.12937997       0.13331944       0.05756889      0.20822781       0.13279304     Float32[0.20004301,  0.8840309,    1.3765225]
# sig_hail          1-2 -1.0             0.05117618       13734945 7141.0    1.2658552e7 0.0006582623      0.00018154182     0.0005252561  0.0028963282       0.0031453804       0.025923109      0.02117475       0.0005252562    0.0028768552     0.026978144    Float32[1.3696966,   -0.2898184,   -0.28453603]
# sig_hail          2-3 0.019996995      0.10483167       185345   7118.0    171884.64   0.04318157        0.012320596       0.03868398    0.15814202         0.179341           0.07240149       0.06382895       0.038683977     0.15751258       0.07204154     Float32[1.5516882,   -0.38941807,  -0.16205323]
# sig_hail          3-4 0.05117618       1.0              77815    7103.0    72430.34    0.1005955         0.029737236       0.09175849    0.2984543          0.3450185          0.14727515       0.14119159       0.09175848      0.29776028       0.14727046     Float32[0.81827897,  0.048856616,  -0.317173]
# tornado_life_risk 1-2 -1.0             0.008275934      13785136 497.85663 1.2704556e7 3.1529587e-5      1.0794626e-5      3.7153502e-5  0.00026915042      0.00029230973      0.0054189446     0.0053258627     3.7153502e-5    0.00026849384    0.005131794    Float32[1.2006066,   -0.14701405,  0.33870423]
# tornado_life_risk 2-3 0.0022878477     0.020291256      69660    497.37744 66054.39    0.0065935818      0.0022335188      0.007153339   0.041378643        0.044208005        0.012835101      0.019535178      0.0071533388    0.04078029       0.019290978    Float32[-0.07807352, 1.0566423,    1.1250377]
# tornado_life_risk 3-4 0.008275934      1.0              27624    492.259   26425.982   0.022587832       0.0059062634      0.017844012   0.08981608         0.09327492         0.03539305       0.042413056      0.017844012     0.08457397       0.043830648    Float32[-0.65504843, 1.7922586,    2.599002]


println("event_to_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_wind" => [[0.63392025, 0.35701632, 0.2873989], [0.7724101, 0.3031301, 0.52364856], [0.61689895, 0.30869296, 0.15061612]], "sig_hail" => [[1.3696966, -0.2898184, -0.28453603], [1.5516882, -0.38941807, -0.16205323], [0.81827897, 0.048856616, -0.317173]], "hail" => [[1.0999736, -0.036068548, -0.11684635], [1.1508383, -0.20880625, -0.63325006], [1.0397497, -0.15485394, -0.61086625]], "sig_wind_adj" => [[0.45143408, 0.5053786, 0.26962078], [0.59369177, 0.5571225, 1.1116248], [0.20004301, 0.8840309, 1.3765225]], "tornado_life_risk" => [[1.2006066, -0.14701405, 0.33870423], [-0.07807352, 1.0566423, 1.1250377], [-0.65504843, 1.7922586, 2.599002]], "tornado" => [[1.1039704, -0.07336604, -0.1759734], [0.980029, 0.02896403, -0.07270332], [0.68161184, 0.16218366, -0.23822184]], "wind_adj" => [[0.8024022, 0.21128672, 0.17552774], [0.92636955, 0.059434433, -0.096047774], [0.8363115, 0.19665669, 0.17679109]], "sig_tornado" => [[0.71196026, 0.28164828, 0.08433733], [0.17950201, 0.808518, 0.56035143], [0.2779992, 0.7922331, 0.76261055]], "wind" => [[0.9594039, 0.081496894, -0.023889013], [1.064942, -0.029093888, -0.19427404], [0.8065231, 0.14930426, 0.047687516]])



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
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPrediction2024.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2024, 2, 28, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

length(day_validation_forecasts)

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected: 505
#

# rm("day_validation_forecasts_0z_12z_with_sig_gated_2024"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_12z;
    event_name_to_labeler = TrainingShared.event_name_to_day_labeler(1),
    save_dir = "day_validation_forecasts_0z_12z_with_sig_gated_2024",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPrediction2024.models_with_gated[prediction_i]
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

  event_names = unique(map(first, HREFPrediction2024.models_with_gated))

  for event_name in event_names
    y = Ys[event_name]
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = HREFPrediction2024.models_with_gated[prediction_i]
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
event_names = map(m -> m[1], HREFPrediction2024.models_with_gated)
model_names = map(m -> m[3], HREFPrediction2024.models_with_gated)
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
  event_name, _ = HREFPrediction2024.models[prediction_i]
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
for prediction_i in 1:length(HREFPrediction2024.models)
  event_name, _, _, _, _ = HREFPrediction2024.models[prediction_i]
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
