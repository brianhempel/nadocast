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

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_fourhourly_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 1, 1, 0)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) #

@time Forecasts.data(validation_forecasts[10]) # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


compute_fourhourly_labels(events, forecast) = begin
  # The original hourlies are ±30min, so four consecutive forecasts is -3:30 to +0:30 from the last valid time.
  end_seconds   = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  start_seconds = end_seconds - 4*HOUR
  # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
  # utc_datetime   = Dates.unix2datetime(end_seconds)
  # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
  # println(Forecasts.valid_yyyymmdd_hhz(forecast))
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_fourhourly_labeler = Dict(
  "tornado"     => (forecast -> compute_fourhourly_labels(StormEvents.conus_tornado_events(),     forecast)),
  "wind"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_wind_events(), forecast)),
  "hail"        => (forecast -> compute_fourhourly_labels(StormEvents.conus_severe_hail_events(), forecast)),
  "sig_tornado" => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_tornado_events(), forecast)),
  "sig_wind"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_wind_events(),    forecast)),
  "sig_hail"    => (forecast -> compute_fourhourly_labels(StormEvents.conus_sig_hail_events(),    forecast)),
)

# rm("four-hourly_accumulators_validation_forecasts"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts;
    event_name_to_labeler = event_name_to_fourhourly_labeler,
    save_dir = "four-hourly_accumulators_validation_forecasts",
  );





# # should do some checks here.
# import PlotMap

# aug29 = validation_forecasts_0z[85]; Forecasts.time_title(aug29) # "2020-08-29 00Z +35"
# aug29_data = Forecasts.data(aug29);
# for i in 1:size(aug29_data,2)
#   PlotMap.plot_debug_map("aug29_0z_day_accs_$(i)_recalib", aug29.grid, aug29_data[:,i]);
# end
# for (event_name, labeler) in event_name_to_day_labeler
#   aug29_labels = event_name_to_day_labeler[event_name](aug29);
#   PlotMap.plot_debug_map("aug29_0z_day_$(event_name)_recalib", aug29.grid, aug29_labels);
# end
# # scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_accs_1.pdf ./
# # scp nadocaster2:/home/brian/nadocast_dev/models/combined_href_sref/aug29_0z_day_tornadoes.pdf ./

# july11 = validation_forecasts_0z[78]; Forecasts.time_title(july11) # "2020-07-11 00Z +35"
# july11_data = Forecasts.data(july11);
# for i in 1:size(july11_data,2)
#   PlotMap.plot_debug_map("july11_0z_day_accs_$i", july11.grid, july11_data[:,i]);
# end
# for (event_name, labeler) in event_name_to_day_labeler
#   july11_labels = event_name_to_day_labeler[event_name](july11);
#   PlotMap.plot_debug_map("july11_0z_day_$event_name", july11.grid, july11_labels);
# end
# # scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_accs_1.pdf ./
# # scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/july11_0z_day_tornado.pdf ./

# dec11 = validation_forecasts_0z[130]; Forecasts.time_title(dec11) # "2021-12-11 00Z +35"
# dec11_data = Forecasts.data(dec11);
# for i in 1:size(dec11_data,2)
#   PlotMap.plot_debug_map("dec11_0z_day_accs_$i", dec11.grid, dec11_data[:,i]);
# end
# for (event_name, labeler) in event_name_to_day_labeler
#   dec11_labels = event_name_to_day_labeler[event_name](dec11);
#   PlotMap.plot_debug_map("dec11_0z_day_$event_name", dec11.grid, dec11_labels);
# end
# # scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_accs_1.pdf ./
# # scp nadocaster:/home/brian/nadocast_dev/models/combined_href_sref/dec11_0z_day_tornado.pdf ./


# # Confirm that the accs are better than the maxes
# function test_predictive_power(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   # Feature order is all HREF severe probs then all SREF severe probs
#   for feature_i in 1:length(inventory)
#     prediction_i = div(feature_i - 1, 2) + 1
#     (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]
#     y = Ys[event_name]
#     x = @view X[:,feature_i]
#     au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
#     println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
#   end
# end
# test_predictive_power(validation_forecasts_0z, X, Ys, weights)





# # 3. bin predictions into 4 bins of equal weight of positive labels

# const bin_count = 4

# function find_ŷ_bin_splits(event_name, ŷ, Ys, weights)
#   y = Ys[event_name]

#   total_positive_weight = sum(Float64.(y .* weights))
#   per_bin_pos_weight = total_positive_weight / bin_count

#   sort_perm      = Metrics.parallel_sort_perm(ŷ);
#   y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
#   ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
#   weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

#   bins_Σŷ      = zeros(Float64, bin_count)
#   bins_Σy      = zeros(Float64, bin_count)
#   bins_Σweight = zeros(Float64, bin_count)
#   bins_max     = ones(Float32, bin_count)

#   bin_i = 1
#   for i in 1:length(y_sorted)
#     if ŷ_sorted[i] > bins_max[bin_i]
#       bin_i += 1
#     end

#     bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
#     bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
#     bins_Σweight[bin_i] += Float64(weights_sorted[i])

#     if bins_Σy[bin_i] >= per_bin_pos_weight
#       bins_max[bin_i] = ŷ_sorted[i]
#     end
#   end

#   for bin_i in 1:bin_count
#     Σŷ      = bins_Σŷ[bin_i]
#     Σy      = bins_Σy[bin_i]
#     Σweight = bins_Σweight[bin_i]

#     mean_ŷ = Σŷ / Σweight
#     mean_y = Σy / Σweight

#     println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
#   end

#   bins_max
# end

# event_types_count = length(CombinedHREFSREF.models)
# event_to_day_bins = Dict{String,Vector{Float32}}()
# println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
# for prediction_i in 1:event_types_count
#   (event_name, _, model_name) = CombinedHREFSREF.models[prediction_i]

#   ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

#   event_to_day_bins[event_name] = find_ŷ_bin_splits(event_name, ŷ, Ys, weights)

#   # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
# end

# # event_name  mean_y                 mean_ŷ                 Σweight              bin_max


# # event_name   mean_y                  mean_ŷ                  Σweight               bin_max


# println(event_to_day_bins)


# # 4. combine bin-pairs (overlapping, 3 bins total)
# # 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + b)


# function find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
#   y = Ys[event_name]
#   ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

#   bins_max = event_to_day_bins[event_name]
#   bins_logistic_coeffs = []

#   # Paired, overlapping bins
#   for bin_i in 1:(bin_count - 1)
#     bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
#     bin_max = bins_max[bin_i+1]

#     bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

#     bin_total_prob_x  = X[bin_members, prediction_i*2 - 1]
#     bin_max_hourly_x  = X[bin_members, prediction_i*2]
#     # bin_ŷ       = ŷ[bin_members]
#     bin_y       = y[bin_members]
#     bin_weights = weights[bin_members]
#     bin_weight  = sum(bin_weights)

#     # logit(HREF), logit(SREF)
#     bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

#     Threads.@threads for i in 1:length(bin_y)
#       logit_total_prob = logit(bin_total_prob_x[i])
#       logit_max_hourly = logit(bin_max_hourly_x[i])

#       bin_X_features[i,1] = logit_total_prob
#       bin_X_features[i,2] = logit_max_hourly
#     end

#     coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

#     # println("Fit logistic coefficients: $(coeffs)")

#     logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

#     stuff = [
#       ("event_name", event_name),
#       ("bin", "$bin_i-$(bin_i+1)"),
#       ("total_prob_ŷ_min", bin_min),
#       ("total_prob_ŷ_max", bin_max),
#       ("count", length(bin_y)),
#       ("pos_count", sum(bin_y)),
#       ("weight", bin_weight),
#       ("mean_total_prob_ŷ", sum(bin_total_prob_x .* bin_weights) / bin_weight),
#       ("mean_max_hourly_ŷ", sum(bin_max_hourly_x .* bin_weights) / bin_weight),
#       ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
#       ("total_prob_logloss", sum(logloss.(bin_y, bin_total_prob_x) .* bin_weights) / bin_weight),
#       ("max_hourly_logloss", sum(logloss.(bin_y, bin_max_hourly_x) .* bin_weights) / bin_weight),
#       ("total_prob_au_pr", Metrics.area_under_pr_curve(bin_total_prob_x, bin_y, bin_weights)),
#       ("max_hourly_au_pr", Metrics.area_under_pr_curve(bin_max_hourly_x, bin_y, bin_weights)),
#       ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
#       ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
#       ("logistic_au_pr", Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights)),
#       ("logistic_coeffs", coeffs)
#     ]

#     headers = map(first, stuff)
#     row     = map(last, stuff)

#     bin_i == 1 && println(join(headers, "\t"))
#     println(join(row, "\t"))

#     push!(bins_logistic_coeffs, coeffs)
#   end

#   bins_logistic_coeffs
# end

# event_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
# for prediction_i in 1:event_types_count
#   event_name, _ = CombinedHREFSREF.models[prediction_i]

#   event_to_day_bins_logistic_coeffs[event_name] = find_logistic_coeffs(event_name, prediction_i, X, Ys, weights)
# end

# # event_name  bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight      mean_total_prob_ŷ mean_max_hourly_ŷ mean_y        total_prob_logloss max_hourly_logloss total_prob_au_pr     max_hourly_au_pr     mean_logistic_ŷ logistic_logloss logistic_au_pr       logistic_coeffs
# # tornado     1-2 -1.0             0.061602164      4731273 4209.0    4.327817e6  0.0011134022      0.00024953068     0.0009035082  0.0051579303       0.0057157436       0.026692282420898487 0.02314910262149825  0.0009035081    0.005135906      0.026116179553460395 Float32[0.915529,    0.075384595, -0.13371886]
# # tornado     2-3 0.018898962      0.13501288       99627   4161.0    93511.52    0.046163313       0.011292203       0.041814737   0.16413356         0.19283397         0.0962565752711827   0.07540747000717556  0.041814737     0.1635127        0.09683596772813384  Float32[1.3713769,   -0.14764006, 0.3025914]
# # tornado     3-4 0.061602164      1.0              35247   4117.0    33620.68    0.14025263        0.03499804        0.116238475   0.35077775         0.41896638         0.19964915178928266  0.18000191591109055  0.116238475     0.34602785       0.1996282709172069   Float32[0.7308742,   -0.01088467, -0.7200494]
# # wind        1-2 -1.0             0.2604162        4684085 31709.0   4.2848795e6 0.0085914815      0.0020453343      0.0068676714  0.025646808        0.029395983        0.14005624222334634  0.12718100010983155  0.006867672     0.025428742      0.13970361586617938  Float32[0.8752906,   0.14241132,  0.019422961]
# # wind        2-3 0.11713328       0.43730876       158996  31762.0   147526.94   0.2293581         0.05869442        0.1994681     0.48277742         0.60856056         0.2921690467592043   0.26438822915217985  0.19946808      0.47999385       0.29267077150132276  Float32[1.0762116,   -0.07859125, -0.31812784]
# # wind        3-4 0.2604162        1.0              82435   31627.0   76558.44    0.42556888        0.12491704        0.38434294    0.6271495          0.8677727          0.5716532810041381   0.5425964559868411   0.38434297      0.6231382        0.5720873029568192   Float32[0.8967682,   0.0795651,   -0.053071257]
# # hail        1-2 -1.0             0.14193675       4700656 14108.0   4.3005735e6 0.003746903       0.0008246585      0.0030246011  0.013220488        0.014967729        0.07183032811646992  0.07133696842462632  0.0030246011    0.013116469      0.07254503320323895  Float32[0.69415,     0.3440887,   0.4457177]
# # hail        2-3 0.058389254      0.26998883       138852  14074.0   128298.055  0.12361407        0.028034544       0.101386614   0.32032704         0.38457015         0.15735357202747363  0.13849845010973116  0.1013866       0.31769595       0.1630493631922233   Float32[1.1330315,   -0.17600663, -0.6018588]
# # hail        3-4 0.14193675       1.0              65864   14044.0   60864.168   0.25895888        0.06771119        0.21368471    0.50153005         0.62741184         0.3465903799789047   0.31890086992964656  0.21368471      0.4924614        0.354143863850937    Float32[1.4685977,   -0.6171552,  -1.4658885]
# # sig_tornado 1-2 -1.0             0.027291382      4755736 579.0     4.3510115e6 0.0001431049      3.4930188e-5      0.00012531038 0.0008154274       0.0008881768       0.018449631807736816 0.019159588090648404 0.00012531038   0.00081216113    0.02114136905891041  Float32[0.9246038,   0.14572951,  0.4728381]
# # sig_tornado 2-3 0.010414468      0.11116843       24091   564.0     23124.406   0.02954026        0.007996856       0.02358327    0.11005916         0.11708307         0.04899531336795075  0.057967113876265686 0.02358327      0.10669346       0.05813556935374382  Float32[-0.27371246, 1.0642519,   0.48847285]
# # sig_tornado 3-4 0.027291382      1.0              10784   559.0     10425.912   0.072314985       0.018098857       0.05221923    0.1902982          0.21011928         0.14452761953831217  0.11803562342468982  0.05221923      0.18464838       0.1344448561028822   Float32[0.77029693,  0.587577,    1.354599]
# # sig_wind    1-2 -1.0             0.047133457      4718220 3784.0    4.316604e6  0.0010036379      0.0002244146      0.00081077305 0.004767808        0.00519899         0.02233480092482336  0.019438854132948973 0.0008107731    0.0047234986     0.020428984492083635 Float32[0.28025302,  0.6347386,   0.37337035]
# # sig_wind    2-3 0.01371814       0.08599372       132745  3772.0    123285.555  0.03464007        0.00806825        0.028384628   0.122832276        0.13845            0.06525466466811923  0.068942286831688    0.028384631     0.12159428       0.07037209534773976  Float32[1.0196984,   0.3275573,   1.3824697]
# # sig_wind    3-4 0.047133457      1.0              48300   3771.0    44833.906   0.08155861        0.018383339       0.07801821    0.26706594         0.32243758         0.12108428956826832  0.1252189924049733   0.0780182       0.26607645       0.12502188609240958  Float32[0.59162444,  0.5482808,   1.1521151]
# # sig_hail    1-2 -1.0             0.03298726       4727984 1949.0    4.3256485e6 0.00048047877     0.00011195232     0.0004181904  0.0023050203       0.0025967043       0.023283310014022587 0.018172292746630257 0.00041819026   0.0022790162     0.02666639382695081  Float32[1.9733405,   -0.7847382,  -0.51065785]
# # sig_hail    2-3 0.01797507       0.07266835       65826   1955.0    60899.133   0.03445054        0.00779719        0.029705029   0.13395527         0.15440375         0.03825078042948839  0.03258042902226003  0.029705029     0.13243617       0.03940314191209964  Float32[1.3154331,   -0.7666408,  -2.8496652]
# # sig_hail    3-4 0.03298726       1.0              38536   1938.0    35788.918   0.064266294       0.015569708       0.050489526   0.19443694         0.22026618         0.10390857002319569  0.0896882406063745   0.050489534     0.19257256       0.10569746828816608  Float32[1.3292868,   -0.31144178, -0.6903727]



# print("event_to_0z_day_bins_logistic_coeffs = ")
# println(event_to_day_bins_logistic_coeffs)
# # event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("sig_hail" => [[1.9733405, -0.7847382, -0.51065785], [1.3154331, -0.7666408, -2.8496652], [1.3292868, -0.31144178, -0.6903727]], "hail" => [[0.69415, 0.3440887, 0.4457177], [1.1330315, -0.17600663, -0.6018588], [1.4685977, -0.6171552, -1.4658885]], "tornado" => [[0.915529, 0.075384595, -0.13371886], [1.3713769, -0.14764006, 0.3025914], [0.7308742, -0.01088467, -0.7200494]], "sig_tornado" => [[0.9246038, 0.14572951, 0.4728381], [-0.27371246, 1.0642519, 0.48847285], [0.77029693, 0.587577, 1.354599]], "sig_wind" => [[0.28025302, 0.6347386, 0.37337035], [1.0196984, 0.3275573, 1.3824697], [0.59162444, 0.5482808, 1.1521151]], "wind" => [[0.8752906, 0.14241132, 0.019422961], [1.0762116, -0.07859125, -0.31812784], [0.8967682, 0.0795651, -0.053071257]])





# # 6. prediction is weighted mean of the two overlapping logistic models
# # 7. predictions should thereby be calibrated (check)



# import Dates
# import Printf

# push!(LOAD_PATH, (@__DIR__) * "/../shared")
# # import TrainGBDTShared
# import TrainingShared
# import LogisticRegression
# using Metrics

# push!(LOAD_PATH, @__DIR__)
# import CombinedHREFSREF

# push!(LOAD_PATH, (@__DIR__) * "/../../lib")
# import Forecasts
# import Inventories
# import StormEvents

# MINUTE = 60 # seconds
# HOUR   = 60*MINUTE

# (_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(CombinedHREFSREF.forecasts_day_with_sig_gated(); just_hours_near_storm_events = false);

# length(day_validation_forecasts)

# # We don't have storm events past this time.
# cutoff = Dates.DateTime(2022, 1, 1, 0)
# day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

# length(day_validation_forecasts)

# # Make sure a forecast loads
# @time Forecasts.data(day_validation_forecasts[10])

# day_validation_forecasts_0z = filter(forecast -> forecast.run_hour == 0, day_validation_forecasts);
# length(day_validation_forecasts_0z) # Expected: 132
# # 132

# compute_day_labels(events, forecast) = begin
#   # Annoying that we have to recalculate this.
#   # The end_seconds will always be the last hour of the convective day
#   # start_seconds depends on whether the run started during the day or not
#   # I suppose for 0Z the answer is always "no" but whatev here's the right math
#   start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + 2*HOUR) - 30*MINUTE
#   end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
#   # println(Forecasts.yyyymmdd_thhz_fhh(forecast))
#   # utc_datetime = Dates.unix2datetime(start_seconds)
#   # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
#   # println(Forecasts.valid_yyyymmdd_hhz(forecast))
#   window_half_size = (end_seconds - start_seconds) ÷ 2
#   window_mid_time  = (end_seconds + start_seconds) ÷ 2
#   StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
# end

# event_name_to_day_labeler = Dict(
#   "tornado"     => (forecast -> compute_day_labels(StormEvents.conus_tornado_events(),     forecast)),
#   "wind"        => (forecast -> compute_day_labels(StormEvents.conus_severe_wind_events(), forecast)),
#   "hail"        => (forecast -> compute_day_labels(StormEvents.conus_severe_hail_events(), forecast)),
#   "sig_tornado" => (forecast -> compute_day_labels(StormEvents.conus_sig_tornado_events(), forecast)),
#   "sig_wind"    => (forecast -> compute_day_labels(StormEvents.conus_sig_wind_events(),    forecast)),
#   "sig_hail"    => (forecast -> compute_day_labels(StormEvents.conus_sig_hail_events(),    forecast)),
# )

# # rm("day_validation_forecasts_0z"; recursive = true)

# X, Ys, weights =
#   TrainingShared.get_data_labels_weights(
#     day_validation_forecasts_0z;
#     event_name_to_labeler = event_name_to_day_labeler,
#     save_dir = "day_validation_forecasts_0z_with_sig_gated",
#   );

# # Confirm that the combined is better than the accs
# function test_predictive_power(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   # Feature order is all HREF severe probs then all SREF severe probs
#   for feature_i in 1:length(inventory)
#     prediction_i = feature_i
#     (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
#     y = Ys[event_name]
#     x = @view X[:,feature_i]
#     au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
#     println("$model_name ($(round(sum(y)))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
#   end
# end
# test_predictive_power(day_validation_forecasts_0z, X, Ys, weights)



# # rm("day_accumulators_validation_forecasts_0z"; recursive = true)

# # test y vs ŷ

# function test_calibration(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   total_weight = sum(Float64.(weights))

#   println("event_name\tmean_y\tmean_ŷ\tΣweight\tSR\tPOD\tbin_max")
#   for feature_i in 1:length(inventory)
#     prediction_i = feature_i
#     (event_name, _, model_name) = CombinedHREFSREF.models_with_gated[prediction_i]
#     y = Ys[event_name]
#     ŷ = @view X[:, feature_i]

#     total_pos_weight = sum(Float64.(y .* weights))

#     sort_perm      = Metrics.parallel_sort_perm(ŷ);
#     y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
#     ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
#     weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

#     bin_count = 20
#     per_bin_pos_weight = Float64(sum(y .* weights)) / bin_count

#     # bins = map(_ -> Int64[], 1:bin_count)
#     bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
#     bins_Σy      = map(_ -> 0.0, 1:bin_count)
#     bins_Σweight = map(_ -> 0.0, 1:bin_count)
#     bins_max     = map(_ -> 1.0f0, 1:bin_count)

#     bin_i = 1
#     for i in 1:length(y_sorted)
#       if ŷ_sorted[i] > bins_max[bin_i]
#         bin_i += 1
#       end

#       bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
#       bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
#       bins_Σweight[bin_i] += Float64(weights_sorted[i])

#       if bins_Σy[bin_i] >= per_bin_pos_weight
#         bins_max[bin_i] = ŷ_sorted[i]
#       end
#     end

#     for bin_i in 1:bin_count
#       Σŷ      = bins_Σŷ[bin_i]
#       Σy      = bins_Σy[bin_i]
#       Σweight = Float32(bins_Σweight[bin_i])

#       mean_ŷ = Float32(Σŷ / Σweight)
#       mean_y = Float32(Σy / Σweight)

#       pos_weight_in_and_after = sum(bins_Σy[bin_i:bin_count])
#       weight_in_and_after     = sum(bins_Σweight[bin_i:bin_count])

#       sr  = Float32(pos_weight_in_and_after / weight_in_and_after)
#       pod = Float32(pos_weight_in_and_after / total_pos_weight)

#       println("$model_name\t$mean_y\t$mean_ŷ\t$Σweight\t$sr\t$pod\t$(bins_max[bin_i])")
#     end
#   end
# end
# test_calibration(day_validation_forecasts_0z, X, Ys, weights)

# # event_name                   mean_y        mean_ŷ        Σweight     SR            POD         bin_max