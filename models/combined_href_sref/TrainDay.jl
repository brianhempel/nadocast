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

accs_event_names = map(i -> CombinedHREFSREF.models[div(i - 1, 2) + 1][1], 1:size(X,2))
Metrics.reliability_curves_midpoints(20, X, Ys, accs_event_names, weights, map(i -> accs_event_names[i] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2)))
# ŷ_tornado_tot,y_tornado_tot,ŷ_tornado_max,y_tornado_max,ŷ_wind_tot,y_wind_tot,ŷ_wind_max,y_wind_max,ŷ_wind_adj_tot,y_wind_adj_tot,ŷ_wind_adj_max,y_wind_adj_max,ŷ_hail_tot,y_hail_tot,ŷ_hail_max,y_hail_max,ŷ_sig_tornado_tot,y_sig_tornado_tot,ŷ_sig_tornado_max,y_sig_tornado_max,ŷ_sig_wind_tot,y_sig_wind_tot,ŷ_sig_wind_max,y_sig_wind_max,ŷ_sig_wind_adj_tot,y_sig_wind_adj_tot,ŷ_sig_wind_adj_max,y_sig_wind_adj_max,ŷ_sig_hail_tot,y_sig_hail_tot,ŷ_sig_hail_max,y_sig_hail_max,
# 0.00012840932,0.000123116,2.2358412e-5,0.00012306387,0.0010836109,0.00089441607,0.00022367963,0.0008962146,0.00031483866,0.0003042497,6.699472e-5,0.00030367333,0.0005317298,0.0003992221,0.00010732838,0.00039939917,1.4138236e-5,1.5716161e-5,1.8965037e-6,1.5753116e-5,9.320341e-5,0.000102982376,1.8274763e-5,0.000102500715,3.1560245e-5,3.7122994e-5,5.8772935e-6,3.7152204e-5,7.430586e-5,5.171544e-5,1.3689685e-5,5.1928608e-5,
# 0.0034208605,0.0029255094,0.00076074514,0.0031123755,0.030598253,0.024061922,0.006932474,0.023894783,0.008852782,0.007374642,0.0020317992,0.008413356,0.017245525,0.014520264,0.0037457065,0.0151365325,0.00041075578,0.0014571181,8.445213e-5,0.0018871152,0.00234489,0.0026708792,0.00058192736,0.0027194503,0.0012289854,0.0009570861,0.0002460072,0.0010013639,0.0043658367,0.0037859262,0.00096778275,0.0029619047,
# 0.0085112415,0.006971065,0.0019492664,0.0072431625,0.057397857,0.043926544,0.013343781,0.043492854,0.017888848,0.016237834,0.0040225335,0.015148265,0.031314515,0.025100423,0.006753481,0.025763765,0.0013736609,0.0009889985,0.00032285368,0.00080666936,0.0050428333,0.004642231,0.001209506,0.0050097,0.0029720236,0.002282644,0.00058111333,0.0021453826,0.0077414867,0.006604606,0.0018170062,0.0069028805,
# 0.013939744,0.013806425,0.0032562881,0.012717114,0.08557987,0.06500294,0.020193268,0.06673216,0.027643805,0.025154546,0.0064753336,0.02452211,0.045916688,0.035810538,0.00969703,0.035275295,0.0044877417,0.003268472,0.0012570613,0.0030097277,0.008868032,0.00605713,0.0019563276,0.0076166303,0.004910041,0.004916691,0.0010870454,0.003367554,0.011041985,0.010708537,0.0024931058,0.010911254,
# 0.01954445,0.017566212,0.0045338687,0.016389685,0.11505711,0.08923783,0.027236808,0.093325034,0.038659003,0.035016287,0.009449604,0.033403736,0.060596563,0.048330504,0.012787063,0.04477558,0.00856611,0.0061653536,0.0022884049,0.008003983,0.01386639,0.010323043,0.0029546376,0.010313399,0.0072884266,0.0056083607,0.0018691239,0.006242039,0.014247545,0.013848821,0.0031209548,0.014461029,
# 0.02645245,0.022078533,0.0061278734,0.020987429,0.14498083,0.11881268,0.034585554,0.11594058,0.05046242,0.04781661,0.012796645,0.04764412,0.076607816,0.055989135,0.016340775,0.05938256,0.012987717,0.010112051,0.0032180082,0.011244028,0.019535327,0.014152791,0.0044072135,0.012208568,0.0104396595,0.009646081,0.0026545168,0.009810942,0.017872907,0.015711416,0.0038166207,0.016562019,
# 0.034893602,0.028028063,0.008267139,0.03101433,0.17446211,0.15192452,0.0424502,0.140006,0.063187115,0.059118766,0.016332388,0.060587566,0.09417956,0.07607425,0.020417629,0.072651125,0.01743901,0.0144152,0.004524178,0.010599296,0.02617379,0.01941641,0.0066458834,0.016325155,0.013318403,0.016616175,0.0033521997,0.016777668,0.022182083,0.018923607,0.004820392,0.014411723,
# 0.045052167,0.03935258,0.010960864,0.038574103,0.2038532,0.1798727,0.050776884,0.1704748,0.07751547,0.06952895,0.020256137,0.06715825,0.112228744,0.09170151,0.024969427,0.09302004,0.022287603,0.020323027,0.006619204,0.017008733,0.03338472,0.024624586,0.009106806,0.024294075,0.016041312,0.017330399,0.0039500683,0.018934261,0.027364144,0.020230578,0.0064152884,0.014689241,
# 0.056493986,0.046658836,0.014171915,0.04867064,0.23405088,0.20702808,0.05982673,0.19946098,0.09403557,0.078863785,0.024686676,0.08102534,0.13108565,0.11433387,0.029723173,0.105761796,0.028709969,0.022680843,0.00951038,0.025119174,0.041176394,0.03032566,0.0111020105,0.032806497,0.019804718,0.014828761,0.004768013,0.014307129,0.033871662,0.023147,0.00832995,0.023060862,
# 0.0692967,0.058774292,0.017223468,0.06361896,0.2652646,0.23716031,0.06921359,0.22873859,0.11264272,0.093820654,0.029626738,0.09229957,0.15070409,0.12736706,0.034646165,0.12916653,0.03957729,0.02481019,0.013594402,0.027385704,0.049222536,0.039532736,0.012787074,0.044971406,0.025882864,0.014561671,0.006068442,0.017812138,0.04127287,0.030423716,0.010214112,0.031801566,
# 0.08311519,0.0702705,0.020100385,0.06823126,0.29727203,0.26411057,0.07903188,0.2581047,0.13266037,0.11273678,0.035246823,0.109355606,0.17232765,0.14048284,0.039673794,0.14576283,0.056707866,0.029785454,0.018452518,0.037957177,0.05725493,0.047117252,0.014395134,0.048890933,0.033561967,0.025042467,0.007978085,0.023460232,0.04944921,0.03602103,0.01209909,0.043142796,
# 0.09742007,0.09314639,0.02320693,0.07066351,0.33007187,0.28980267,0.08977279,0.2854537,0.15508935,0.12945783,0.04205127,0.11971715,0.19529422,0.16658358,0.045482706,0.14560273,0.076990254,0.047205888,0.023490286,0.054849047,0.06552671,0.056876875,0.016164027,0.05365675,0.04238364,0.031524353,0.010371154,0.032081272,0.058474414,0.05232721,0.014124583,0.048729412,
# 0.11300705,0.104963265,0.027016522,0.09071687,0.36435607,0.31824306,0.10136073,0.31983814,0.18146446,0.14197887,0.05060894,0.13649045,0.21985586,0.18860102,0.05273449,0.16069384,0.09537407,0.07803811,0.028592732,0.08145353,0.073421724,0.07034654,0.017973782,0.06302113,0.05305262,0.04037241,0.013492925,0.031425722,0.06700298,0.06956121,0.016456867,0.059246093,
# 0.13252534,0.11920456,0.03238558,0.088272616,0.4000273,0.35473627,0.11436899,0.33607706,0.2125142,0.1692329,0.060712244,0.15677452,0.24844903,0.2007099,0.06155805,0.18333021,0.11210467,0.11563029,0.03267676,0.112088226,0.08195752,0.07190071,0.019710831,0.07907983,0.065705515,0.0476678,0.016968943,0.053829588,0.07596131,0.07912994,0.019171217,0.06832472,
# 0.15520945,0.15139146,0.04154517,0.09472858,0.43954974,0.37504002,0.12914775,0.37543225,0.24657181,0.21305119,0.07235785,0.1955755,0.28294873,0.21831018,0.07283296,0.21198207,0.12879968,0.12537554,0.035938818,0.1025449,0.09166274,0.08570154,0.021467693,0.09078565,0.07822883,0.08703402,0.02009384,0.07731593,0.08684262,0.087083384,0.022749314,0.070295,
# 0.17937222,0.18569003,0.053079672,0.16648073,0.48582014,0.4291243,0.14714815,0.3865536,0.28411505,0.25533837,0.08528552,0.23873657,0.32380036,0.25446874,0.08711145,0.23997924,0.14803414,0.1547204,0.03884477,0.10991911,0.10226239,0.10142709,0.023652324,0.08695518,0.09174606,0.08394174,0.023832997,0.08941548,0.09917767,0.11297941,0.02688791,0.10742076,
# 0.21150486,0.17297174,0.06291148,0.22737926,0.53986794,0.48901695,0.17083444,0.42839733,0.32786605,0.28329137,0.09994622,0.2896528,0.37501946,0.3072564,0.10587295,0.28065002,0.16880186,0.14328644,0.043471318,0.10882144,0.11363893,0.117262885,0.026538996,0.09920489,0.11019821,0.13880825,0.02799951,0.13784964,0.113983035,0.11802522,0.031338006,0.123593055,
# 0.26404116,0.18038715,0.07477671,0.24997064,0.60473806,0.54868394,0.20415415,0.49509788,0.37875503,0.33124548,0.119066596,0.30998918,0.43409917,0.3970033,0.131196,0.33616772,0.18712175,0.22423403,0.050860763,0.11068682,0.12882729,0.12746371,0.030284418,0.13026811,0.12865245,0.1835387,0.032509767,0.18801874,0.13366799,0.13786195,0.037194777,0.13285068,
# 0.35348994,0.28821528,0.09312593,0.27430168,0.6864221,0.628531,0.2567791,0.57739586,0.44096822,0.43865225,0.14792591,0.37484223,0.5083193,0.4260122,0.16688412,0.41812858,0.21171084,0.14170215,0.061370913,0.16015735,0.15311958,0.14572704,0.036350984,0.1479292,0.1604113,0.1970058,0.040563054,0.20746435,0.16085622,0.16556078,0.04664997,0.14095706,
# 0.5106103,0.32362208,0.14751808,0.2768219,0.8302569,0.76425946,0.38287085,0.70745826,0.55053705,0.6003347,0.20856333,0.54460156,0.6523597,0.5546805,0.2530648,0.5056397,0.2844225,0.10411548,0.08451147,0.15021236,0.20090753,0.20118411,0.05713672,0.19096056,0.2253754,0.33678108,0.059254985,0.29372057,0.22606945,0.16638465,0.073354416,0.16376886,



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

    Threads.@threads :static for i in 1:length(bin_y)
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
length(day_validation_forecasts_0z_12z) # Expected: 308
# 308

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

# tornado (18464.0)                       feature 1 TORPROB:calculated:hour fcst:calculated_prob:                    AU-PR-curve: 0.14826901
# wind (132107.0)                         feature 2 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.41996258
# wind_adj (45948.0)                      feature 3 WINDPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.25080815 (trivially worse)
# hail (60620.0)                          feature 4 HAILPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.2707911
# sig_tornado (2453.0)                    feature 5 STORPROB:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.08256114
# sig_wind (15350.0)                      feature 6 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.09190887 (worse)
# sig_wind_adj (5766.0)                   feature 7 SWINDPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.106514186
# sig_hail (8202.0)                       feature 8 SHAILPRO:calculated:hour fcst:calculated_prob:                   AU-PR-curve: 0.092866085
# sig_tornado_gated_by_tornado (2453.0)   feature 9 STORPROB:calculated:hour fcst:calculated_prob:gated by tornado   AU-PR-curve: 0.07968105 (worse)
# sig_wind_gated_by_wind (15350.0)        feature 10 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind     AU-PR-curve: 0.09191168
# sig_wind_adj_gated_by_wind_adj (5766.0) feature 11 SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.10672214
# sig_hail_gated_by_hail (8202.0)         feature 12 SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail     AU-PR-curve: 0.092866875


function test_predictive_power_all(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  event_names = unique(map(first, CombinedHREFSREF.models_with_gated))

  # Feature order is all HREF severe probs then all SREF severe probs
  for event_name in event_names
    y = Ys[event_name]
    for feature_i in 1:length(inventory)
      prediction_i = feature_i
      (_, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
      x = @view X[:,feature_i]
      au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
      println("$event_name ($(sum(y))) feature $feature_i $model_name $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
    end
  end
end
test_predictive_power_all(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado (18464.0)        feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.14826901 ***best tor**
# tornado (18464.0)        feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.048111282
# tornado (18464.0)        feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.026417494
# tornado (18464.0)        feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.03769853
# tornado (18464.0)        feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.11919093
# tornado (18464.0)        feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.05203152
# tornado (18464.0)        feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.023538161
# tornado (18464.0)        feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.034951087
# tornado (18464.0)        feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.11847384
# tornado (18464.0)        feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.0520366
# tornado (18464.0)        feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.023641393
# tornado (18464.0)        feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.034952097
# wind (132107.0)          feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.23750569
# wind (132107.0)          feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.41996258 ***best wind***
# wind (132107.0)          feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.26382613
# wind (132107.0)          feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.16925876
# wind (132107.0)          feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.21166264
# wind (132107.0)          feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.32219085
# wind (132107.0)          feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.21695094
# wind (132107.0)          feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.13536471
# wind (132107.0)          feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.21206793
# wind (132107.0)          feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.32223076
# wind (132107.0)          feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.21760708
# wind (132107.0)          feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.13537118
# wind_adj (45947.96)      feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.060670942
# wind_adj (45947.96)      feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.10455627
# wind_adj (45947.96)      feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.25080815 ***best wind_adj***
# wind_adj (45947.96)      feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.085725784
# wind_adj (45947.96)      feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.05909475
# wind_adj (45947.96)      feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.16671911
# wind_adj (45947.96)      feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.22884703
# wind_adj (45947.96)      feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.08012605
# wind_adj (45947.96)      feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.05912331
# wind_adj (45947.96)      feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.16673052
# wind_adj (45947.96)      feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.22927126
# wind_adj (45947.96)      feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.08012925
# hail (60620.0)           feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.10694888
# hail (60620.0)           feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.106387354
# hail (60620.0)           feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.103704095
# hail (60620.0)           feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.2707911 ***best hail***
# hail (60620.0)           feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.093489416
# hail (60620.0)           feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.111419596
# hail (60620.0)           feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.088233426
# hail (60620.0)           feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.2174997
# hail (60620.0)           feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.09331971
# hail (60620.0)           feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.111437894
# hail (60620.0)           feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.08856822
# hail (60620.0)           feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.21750417
# sig_tornado (2453.0)     feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.0700202
# sig_tornado (2453.0)     feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.01205708
# sig_tornado (2453.0)     feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.004126008
# sig_tornado (2453.0)     feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.005973841
# sig_tornado (2453.0)     feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.08256114 ***best sigtor***
# sig_tornado (2453.0)     feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.011346154
# sig_tornado (2453.0)     feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.0038076392
# sig_tornado (2453.0)     feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.0051615
# sig_tornado (2453.0)     feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.07968105 (not best sigtor)
# sig_tornado (2453.0)     feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.011346459
# sig_tornado (2453.0)     feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.003824785
# sig_tornado (2453.0)     feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.00516167
# sig_wind (15350.0)       feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.041251373
# sig_wind (15350.0)       feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.056009118
# sig_wind (15350.0)       feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.08744482
# sig_wind (15350.0)       feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.03178346
# sig_wind (15350.0)       feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.03849379
# sig_wind (15350.0)       feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.09190887 (not best sigwind)
# sig_wind (15350.0)       feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.092256226
# sig_wind (15350.0)       feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.028716838
# sig_wind (15350.0)       feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.038608886
# sig_wind (15350.0)       feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.09191168 (not best sigwind)
# sig_wind (15350.0)       feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.0924767 ***best sigwind***
# sig_wind (15350.0)       feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.028717887
# sig_wind_adj (5766.4995) feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.008761741
# sig_wind_adj (5766.4995) feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.016984629
# sig_wind_adj (5766.4995) feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.079091184
# sig_wind_adj (5766.4995) feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.013985491
# sig_wind_adj (5766.4995) feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.009180126
# sig_wind_adj (5766.4995) feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.043251134
# sig_wind_adj (5766.4995) feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.106514186
# sig_wind_adj (5766.4995) feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.014008616
# sig_wind_adj (5766.4995) feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.009184345
# sig_wind_adj (5766.4995) feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.04325205
# sig_wind_adj (5766.4995) feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.10672214 ***best sigwind_adj***
# sig_wind_adj (5766.4995) feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.014009072
# sig_hail (8202.0)        feature 1 tornado TORPROB:calculated:hour fcst:calculated_prob:                                           AU-PR-curve: 0.022044
# sig_hail (8202.0)        feature 2 wind WINDPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.014758687
# sig_hail (8202.0)        feature 3 wind_adj WINDPROB:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.018963654
# sig_hail (8202.0)        feature 4 hail HAILPROB:calculated:hour fcst:calculated_prob:                                             AU-PR-curve: 0.08206327
# sig_hail (8202.0)        feature 5 sig_tornado STORPROB:calculated:hour fcst:calculated_prob:                                      AU-PR-curve: 0.018859537
# sig_hail (8202.0)        feature 6 sig_wind SWINDPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.022376891
# sig_hail (8202.0)        feature 7 sig_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:                                     AU-PR-curve: 0.01814246
# sig_hail (8202.0)        feature 8 sig_hail SHAILPRO:calculated:hour fcst:calculated_prob:                                         AU-PR-curve: 0.092866085
# sig_hail (8202.0)        feature 9 sig_tornado_gated_by_tornado STORPROB:calculated:hour fcst:calculated_prob:gated by tornado     AU-PR-curve: 0.018698893
# sig_hail (8202.0)        feature 10 sig_wind_gated_by_wind SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind             AU-PR-curve: 0.022378672
# sig_hail (8202.0)        feature 11 sig_wind_adj_gated_by_wind_adj SWINDPRO:calculated:hour fcst:calculated_prob:gated by wind_adj AU-PR-curve: 0.01819592
# sig_hail (8202.0)        feature 12 sig_hail_gated_by_hail SHAILPRO:calculated:hour fcst:calculated_prob:gated by hail             AU-PR-curve: 0.092866875 ***best sighail***


# test y vs ŷ
event_names = map(m -> m[1], CombinedHREFSREF.models_with_gated)
model_names = map(m -> m[3], CombinedHREFSREF.models_with_gated)
Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)
# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,ŷ_sig_wind_gated_by_wind,y_sig_wind_gated_by_wind,ŷ_sig_wind_adj_gated_by_wind_adj,y_sig_wind_adj_gated_by_wind_adj,ŷ_sig_hail_gated_by_hail,y_sig_hail_gated_by_hail,
# 0.00011439203,0.00012310079,0.00081953296,0.00089423097,0.00027773227,0.00030394463,0.00038151306,0.00039911259,1.5322561e-5,1.580418e-5,9.872865e-5,0.00010250751,3.2085158e-5,3.70803e-5,5.4565608e-5,5.1644383e-5,1.5290738e-5,1.5804057e-5,9.8090655e-5,0.000102495964,3.197012e-5,3.706051e-5,5.456094e-5,5.1644343e-5,
# 0.003034773,0.0029581804,0.024223967,0.024143195,0.0079072025,0.007593231,0.013247244,0.014639257,0.0004214944,0.0015164607,0.0023701275,0.0027325186,0.0011352561,0.0009693138,0.0033615276,0.0038947451,0.00042152737,0.0015155277,0.0023703314,0.0027389354,0.0011355821,0.0009764964,0.0033614067,0.0038942741,
# 0.0074839354,0.0069393683,0.046261273,0.04390103,0.015891291,0.015955541,0.024535332,0.025177572,0.0013553582,0.0008839501,0.004515026,0.0051529757,0.0025572376,0.0022956068,0.006081522,0.006575688,0.0013549135,0.00088459096,0.004515229,0.00515744,0.0025576134,0.0023080443,0.0060815755,0.0065762457,
# 0.012151344,0.014149223,0.07004471,0.06459001,0.0246825,0.025650883,0.036489736,0.035857588,0.0041722134,0.0035998814,0.00706493,0.006785758,0.004236331,0.004250471,0.008899831,0.010554015,0.0041724974,0.0036009392,0.0070650377,0.0067878575,0.0042368,0.004273976,0.008899779,0.01055632,
# 0.016828189,0.017726915,0.095221296,0.09042606,0.03466237,0.034211583,0.04882666,0.047561996,0.007118136,0.006394494,0.010332507,0.010614689,0.006531874,0.0055869166,0.011766628,0.0131343985,0.007118142,0.0063941483,0.010332498,0.010615513,0.0065317703,0.0056240126,0.011766686,0.013132168,
# 0.022638444,0.021617547,0.121495396,0.1182944,0.045631737,0.04890285,0.06245704,0.056128178,0.009920524,0.01208098,0.01442322,0.0127883,0.009255238,0.010243716,0.014866797,0.0163169,0.009920801,0.01208,0.014423214,0.012788886,0.009255323,0.010318857,0.014866916,0.016318958,
# 0.0296873,0.028602252,0.14813542,0.15186122,0.05709,0.059386905,0.07710908,0.07628928,0.012982222,0.011262619,0.020010322,0.017838528,0.01145002,0.018991707,0.018239107,0.020435229,0.012982779,0.01126072,0.02001044,0.017837822,0.011450112,0.019098857,0.018239178,0.020436332,
# 0.038330723,0.038284004,0.17462799,0.18025464,0.069566235,0.06883829,0.091854826,0.092254795,0.017412884,0.018309735,0.02639645,0.023313614,0.013364243,0.018592212,0.022373166,0.020322593,0.017413683,0.018304478,0.026396023,0.02332198,0.013364152,0.018660735,0.022373239,0.020327939,
# 0.04851824,0.04676548,0.2018156,0.20598413,0.08306256,0.07887398,0.10713681,0.11121328,0.02288811,0.027036889,0.033502806,0.0313347,0.016328996,0.013478501,0.027658733,0.02493692,0.022889277,0.026955582,0.033502147,0.031337366,0.016329689,0.013527868,0.02765882,0.024935303,
# 0.06039561,0.05861901,0.22963423,0.2404826,0.097034305,0.09347788,0.122679316,0.13122505,0.031584136,0.025341569,0.041330338,0.03922341,0.020998253,0.016584013,0.03445751,0.027373757,0.03158481,0.02517676,0.041330058,0.039225485,0.020999484,0.01664099,0.034457944,0.027377643,
# 0.07473179,0.070190676,0.25931945,0.2588614,0.11132848,0.11210368,0.14052218,0.13645962,0.044640996,0.034912083,0.04942185,0.048977584,0.026480375,0.025203517,0.043273035,0.035716068,0.04445102,0.035039805,0.04942185,0.048977584,0.026477614,0.025270011,0.043273035,0.035716068,
# 0.08952953,0.093841285,0.28976464,0.28966358,0.12784275,0.13016511,0.16052514,0.1649172,0.058337655,0.048903033,0.0578504,0.053721793,0.033055376,0.030209346,0.053164743,0.049557634,0.05744606,0.050889596,0.0578504,0.053721793,0.03305743,0.030279707,0.053164743,0.049557634,
# 0.10423421,0.105226085,0.3207853,0.31919575,0.14887503,0.14307928,0.18087047,0.19724366,0.07272293,0.06416561,0.06649776,0.072250985,0.042733442,0.0356017,0.06282177,0.06762369,0.07011594,0.073572956,0.06649776,0.072250985,0.042730246,0.03572836,0.06282177,0.06762369,
# 0.12032348,0.11602743,0.35258535,0.35535574,0.17679292,0.16844808,0.20403364,0.19546416,0.088370964,0.08783313,0.07529654,0.08197254,0.057715613,0.049583253,0.072197266,0.085749246,0.08305039,0.09472144,0.07529654,0.08197254,0.057718582,0.049622923,0.072197266,0.085749246,
# 0.13596456,0.146656,0.38756573,0.37568393,0.21207128,0.2141206,0.23242252,0.21582012,0.10176133,0.13629381,0.08489837,0.08753594,0.07586559,0.08831547,0.082088545,0.09308073,0.09628913,0.11932844,0.08489837,0.08753594,0.0758639,0.08828435,0.082088545,0.09308073,
# 0.15309666,0.1757119,0.42870733,0.43236035,0.25345173,0.2540478,0.2675052,0.26044354,0.110806346,0.18889168,0.096417606,0.093757555,0.09766588,0.08585528,0.09356525,0.114998825,0.10742663,0.15326926,0.096417606,0.093757555,0.09758729,0.08589625,0.09356525,0.114998825,
# 0.17361972,0.1862082,0.47861475,0.48710817,0.3008287,0.2837437,0.31090122,0.30943885,0.1181781,0.21655016,0.11042335,0.11036481,0.12719257,0.14509761,0.108026996,0.114009336,0.115928516,0.21530949,0.11042335,0.11036481,0.12704849,0.14373673,0.108026996,0.114009336,
# 0.20434313,0.19139843,0.54055214,0.54839003,0.35701367,0.3298026,0.3613009,0.39696681,0.12639625,0.19231522,0.12728976,0.14870209,0.15745796,0.1859903,0.12846206,0.1298336,0.12357865,0.19988371,0.12728976,0.14870209,0.15728648,0.18770535,0.12846206,0.1298336,
# 0.2557364,0.29954046,0.6216102,0.6305038,0.42719218,0.442935,0.42842576,0.42683762,0.13765357,0.21207555,0.15194036,0.14856929,0.2136373,0.19904412,0.1550266,0.1808889,0.13452981,0.1857295,0.15194036,0.14856929,0.21337953,0.20001237,0.1550266,0.1808889,
# 0.3533377,0.3152353,0.7794176,0.76185143,0.5515064,0.5954248,0.57483107,0.5705769,0.18729448,0.082768016,0.22015385,0.20741571,0.33426335,0.3390779,0.21583815,0.16654947,0.18450335,0.07839314,0.22015385,0.20741571,0.33392593,0.33862507,0.21583815,0.16654947,

Metrics.reliability_curves_midpoints(10, X, Ys, event_names, weights, model_names)
# ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_wind_adj,y_wind_adj,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,ŷ_sig_wind,y_sig_wind,ŷ_sig_wind_adj,y_sig_wind_adj,ŷ_sig_hail,y_sig_hail,ŷ_sig_tornado_gated_by_tornado,y_sig_tornado_gated_by_tornado,ŷ_sig_wind_gated_by_wind,y_sig_wind_gated_by_wind,ŷ_sig_wind_adj_gated_by_wind_adj,y_sig_wind_adj_gated_by_wind_adj,ŷ_sig_hail_gated_by_hail,y_sig_hail_gated_by_hail,
# 0.00023099118,0.0002362167,0.0016554861,0.0017246311,0.0005713599,0.0005844799,0.00072295207,0.0007770267,1.946929e-5,3.1098883e-5,0.00018038909,0.00019753017,7.267462e-5,7.138039e-5,9.769846e-5,0.00010181856,1.9440784e-5,3.109844e-5,0.00017958744,0.00019752543,7.226732e-5,7.136299e-5,9.7697404e-5,0.00010181832,
# 0.009015698,0.009309097,0.055885006,0.05227262,0.019261377,0.019672658,0.029466942,0.029583452,0.0019057646,0.0014166408,0.005605061,0.005836043,0.0031443734,0.0029769964,0.007157439,0.008076456,0.0019056678,0.001417562,0.005605572,0.0058397073,0.0031447064,0.0029932894,0.0071573616,0.0080775535,
# 0.019440765,0.019457882,0.10660444,0.1024999,0.039173353,0.040249158,0.055076487,0.0514901,0.008073858,0.00837298,0.012181521,0.011588899,0.0074753333,0.007232646,0.013141787,0.014552561,0.008073976,0.008372449,0.012181545,0.0115896305,0.007474884,0.0072819004,0.013141645,0.014552008,
# 0.033363283,0.032805663,0.1602486,0.16484353,0.0628643,0.06375849,0.083776705,0.08349805,0.014607088,0.013924421,0.022773499,0.020212393,0.01238617,0.018641695,0.02029671,0.020355595,0.014607876,0.013921437,0.022772755,0.020215085,0.012387257,0.018730953,0.020296564,0.020358825,
# 0.053745266,0.051869974,0.21464907,0.22189133,0.089452356,0.085550085,0.11425681,0.120393135,0.027178368,0.025954599,0.036969714,0.034826897,0.018382682,0.01488201,0.030851386,0.026166782,0.027185481,0.025836015,0.036969285,0.03482936,0.018383585,0.01493548,0.030851288,0.02616767,
# 0.081023514,0.08036515,0.2736834,0.2733947,0.1189594,0.120474085,0.14956142,0.14933477,0.050176878,0.040312763,0.053427733,0.051224634,0.029391412,0.027283277,0.04726992,0.041415207,0.04951213,0.041353226,0.053427733,0.051224634,0.029391851,0.02734891,0.04726992,0.041415207,
# 0.11184175,0.11016757,0.3358247,0.33629116,0.16167024,0.15468983,0.19248328,0.19626598,0.07891548,0.07466909,0.07054384,0.07714398,0.04887683,0.041291017,0.06679998,0.07487629,0.075441815,0.08224224,0.07054384,0.07714398,0.048888054,0.04139415,0.06679998,0.07487629,
# 0.14371535,0.1599744,0.40667215,0.40206203,0.23093405,0.23234156,0.24827628,0.23612955,0.10515358,0.15281864,0.09037395,0.08987458,0.086638056,0.08705223,0.087084144,0.102774486,0.100894906,0.13307229,0.09037395,0.08987458,0.086590566,0.08706236,0.087084144,0.102774486,
# 0.18870021,0.18854396,0.5077065,0.5158731,0.3266769,0.3050467,0.3328532,0.34780952,0.12229934,0.20568305,0.11756743,0.1268399,0.13978918,0.16255863,0.11739198,0.12089121,0.11957053,0.20927837,0.11756743,0.1268399,0.13959913,0.16218513,0.11739198,0.12089121,
# 0.30303723,0.3073541,0.6929684,0.6898671,0.47989482,0.5073153,0.49078405,0.48769665,0.17196785,0.122763336,0.18011624,0.1729101,0.25594062,0.24838707,0.1860674,0.1743001,0.16811742,0.11223902,0.18011624,0.1729101,0.2562552,0.2497461,0.1860674,0.1743001,




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
  event_name, _ = CombinedHREFSREF.models[prediction_i]
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
for prediction_i in 1:length(CombinedHREFSREF.models)
  event_name, _ = CombinedHREFSREF.models[prediction_i]
  calibrations_wr[event_name] = spc_calibrate_warning_ratio(prediction_i, X, Ys, weights)
end


# event_name   nominal_prob threshold_to_match_warning_ratio SR          POD           WR
# tornado      0.02         0.018671036                      0.06670111  0.75727624    0.02534866
# tornado      0.05         0.07438469                       0.14525408  0.4773293     0.007337084
# tornado      0.1          0.17809486                       0.24449195  0.16234364    0.001482533
# tornado      0.15         0.31424522                       0.2893897   0.032852195   0.00025346328
# tornado      0.3          0.4208889                        0.34888482  0.0058636772  3.752509e-5
# tornado      0.45         0.5354595                        0.11523869  0.00016964582 3.28684e-6
# wind         0.05         0.04947853                       0.19556674  0.8698387     0.070396446
# wind         0.15         0.22776604                       0.38672954  0.52859944    0.021633476
# wind         0.3          0.5126095                        0.6385383   0.14644314    0.0036298535
# wind         0.45         0.7822857                        0.8633927   0.022703465   0.00041618917
# wind_adj     0.05         0.011903763                      0.069717295 0.90240324    0.070401795
# wind_adj     0.15         0.069215775                      0.15769151  0.6271789     0.021632504
# wind_adj     0.3          0.24395943                       0.35368696  0.23603159    0.0036297336
# wind_adj     0.45         0.48122978                       0.6051523   0.04632854    0.000416397
# hail         0.05         0.032194138                      0.116115816 0.84436774    0.052632086
# hail         0.15         0.12929726                       0.2369699   0.5047792     0.015417665
# hail         0.3          0.38471794                       0.4838382   0.10396624    0.0015552583
# hail         0.45         0.6738186                        0.74441147  0.009198359   8.943503e-5
# sig_tornado  0.1          0.06749153                       0.12375108  0.39132497    0.0009528099
# sig_wind     0.1          0.1238575                        0.16619988  0.13317949    0.0014687241
# sig_wind_adj 0.1          0.067705154                      0.13839334  0.29883972    0.0014685602
# sig_hail     0.1          0.06931114                       0.12050349  0.34222764    0.0027944276

println(calibrations_wr)
# Dict{String, Vector{Tuple{Float32, Float32}}}("sig_wind" => [(0.1, 0.1238575)], "sig_hail" => [(0.1, 0.06931114)], "hail" => [(0.05, 0.032194138), (0.15, 0.12929726), (0.3, 0.38471794), (0.45, 0.6738186)], "sig_wind_adj" => [(0.1, 0.067705154)], "tornado" => [(0.02, 0.018671036), (0.05, 0.07438469), (0.1, 0.17809486), (0.15, 0.31424522), (0.3, 0.4208889), (0.45, 0.5354595)], "wind_adj" => [(0.05, 0.011903763), (0.15, 0.069215775), (0.3, 0.24395943), (0.45, 0.48122978)], "sig_tornado" => [(0.1, 0.06749153)], "wind" => [(0.05, 0.04947853), (0.15, 0.22776604), (0.3, 0.5126095), (0.45, 0.7822857)])








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



