import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics
import PredictionForecasts

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations2

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import Grids
import LogisticRegression
import PlotMap


σ(x)     = 1f0 / (1f0 + exp(-x))
logit(p) = log(p / (1f0 - p))

const ε = eps(Float32) # 1.1920929f-7
const logloss_ε_correction = log(1f0 + ε) # make logloss(0,0) and logloss(1,1) be zero instead of slightly negative
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε) + logloss_ε_correction


# (_, validation_forecasts, test_forecasts) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations2.forecasts(); just_hours_near_storm_events = false);

# length(validation_forecasts) # 24373

# forecast_i = 1
# start_time = time_ns()
# for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(validation_forecasts)
#   elapsed = (time_ns() - start_time) / 1.0e9
#   print("\r$forecast_i/~$(length(validation_forecasts)) forecasts loaded.  $(Float32(elapsed / forecast_i))s each.  ~$(Float32((elapsed / forecast_i) * (length(validation_forecasts) - forecast_i) / 60 / 60)) hours left.            ")
#   forecast_i += 1
# end

# # We don't have storm events past this time.
# cutoff = Dates.DateTime(2022, 6, 1, 12)
# test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
# length(test_forecasts) # 24662

# forecast_i = 1
# start_time = time_ns()
# for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(test_forecasts)
#   elapsed = (time_ns() - start_time) / 1.0e9
#   print("\r$forecast_i/~$(length(test_forecasts)) forecasts loaded.  $(Float32(elapsed / forecast_i))s each.  ~$(Float32((elapsed / forecast_i) * (length(test_forecasts) - forecast_i) / 60 / 60)) hours left.            ")
#   forecast_i += 1
# end


function inspect_predictive_power(forecasts, X, Ys, weights, model_names, event_names)
  inventory = Forecasts.inventory(forecasts[1])

  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    # au_pr_curve = Metrics.area_under_pr_curve_fast(x, y, weights)
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)

    println("$model_name ($(Float32(sum(Ys[event_name])))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(au_pr_curve)")
  end

  ()
end


function find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
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

  println("model_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for bin_i in 1:bin_count
    Σŷ      = bins_Σŷ[bin_i]
    Σy      = bins_Σy[bin_i]
    Σweight = bins_Σweight[bin_i]

    mean_ŷ = Σŷ / Σweight
    mean_y = Σy / Σweight

    println("$model_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
  end

  bins_max
end

function find_single_predictor_logistic_coeffs(model_name, ŷ, y, weights, bins_max)

  bin_count = length(bins_max)
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    bin_X_features = Array{Float32}(undef, (length(bin_y), 1))

    Threads.@threads :static for i in 1:length(bin_y)
      logit_href = logit(bin_ŷ[i])

      bin_X_features[i,1] = logit_href
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
      ("model_name", model_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("input_ŷ_min", bin_min),
      ("input_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_input_ŷ", sum(bin_ŷ .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("input_logloss", sum(logloss.(bin_y, bin_ŷ) .* bin_weights) / bin_weight),
      ("input_au_pr", Metrics.area_under_pr_curve_fast(bin_ŷ, bin_y, bin_weights)),
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

# ŷ1 is used as the first guess
function find_two_predictor_logistic_coeffs(model_name, ŷ1, ŷ2, y, weights, bins_max)

  bin_count = length(bins_max)
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ1 .> bin_min) .* (ŷ1 .<= bin_max)

    bin_ŷ1      = ŷ1[bin_members]
    bin_ŷ2      = ŷ2[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(total), logit(max)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

    Threads.@threads :static for i in 1:length(bin_y)
      logit_total_prob = logit(bin_ŷ1[i])
      logit_max_hourly = logit(bin_ŷ2[i])

      bin_X_features[i,1] = logit_total_prob
      bin_X_features[i,2] = logit_max_hourly
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("model_name", model_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("ŷ1_min", bin_min),
      ("ŷ1_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_ŷ1", sum(bin_ŷ1 .* bin_weights) / bin_weight),
      ("mean_ŷ2", sum(bin_ŷ2 .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("ŷ1_logloss", sum(logloss.(bin_y, bin_ŷ1) .* bin_weights) / bin_weight),
      ("ŷ2_logloss", sum(logloss.(bin_y, bin_ŷ2) .* bin_weights) / bin_weight),
      ("ŷ1_au_pr", Metrics.area_under_pr_curve_fast(bin_ŷ1, bin_y, bin_weights)),
      ("ŷ2_au_pr", Metrics.area_under_pr_curve_fast(bin_ŷ2, bin_y, bin_weights)),
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


# Assumes weights are proportional to gridpoint areas
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_warning_ratio(event_name, model_name, ŷ, y, weights)
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
  for i in 1:length(target_warning_ratios[event_name])
    nominal_prob, _ = target_warning_ratios[event_name][i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, threshold_to_match_warning_ratio))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, threshold_to_match_warning_ratio))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, threshold_to_match_warning_ratio))
    println("$model_name\t$nominal_prob\t$threshold_to_match_warning_ratio\t$sr\t$pod\t$wr")
    push!(wr_thresholds, (Float32(nominal_prob), Float32(threshold_to_match_warning_ratio)))
  end

  wr_thresholds
end


# rm("validation_forecasts_blurred"; recursive=true)
# rm("validation_forecasts_calibrated"; recursive=true)
# rm("validation_day_accumulators_forecasts_0z_12z"; recursive=true)
# rm("validation_day_forecasts_0z_12z"; recursive=true)
function do_it_all(forecasts, forecast_hour_range, model_names, event_names, make_calibrated_hourly_models)
  @assert length(model_names) == length(event_names)
  nmodels = length(model_names)

  forecasts = filter(f -> f.forecast_hour in forecast_hour_range, forecasts)

  _, validation_forecasts, _ = TrainingShared.forecasts_train_validation_test(forecasts; just_hours_near_storm_events = false)

  # for testing
  if get(ENV, "ONLY_N_FORECASTS", "") != ""
    validation_forecasts = validation_forecasts[1:parse(Int64, ENV["ONLY_N_FORECASTS"])]
  end

  valid_times = Forecasts.valid_utc_datetime.(validation_forecasts)
  println("$(length(validation_forecasts)) validation forecasts from $(minimum(valid_times)) to $(maximum(valid_times))")

  grid = validation_forecasts[1].grid

  # blur_0mi_grid_is  = Grids.radius_grid_is(grid, 0.0)
  blur_15mi_grid_is = Grids.radius_grid_is(grid, 15.0)
  blur_25mi_grid_is = Grids.radius_grid_is(grid, 25.0)
  # blur_35mi_grid_is = Grids.radius_grid_is(grid, 35.0)
  # blur_50mi_grid_is = Grids.radius_grid_is(grid, 50.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_blurs_910
    (blur_25mi_grid_is, blur_25mi_grid_is), # wind_mean_prob_computed_climatology_blurs_910
    (blur_15mi_grid_is, blur_25mi_grid_is), # hail_mean_prob_computed_climatology_blurs_910
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_blurs_910_before_20200523
    (blur_25mi_grid_is, blur_25mi_grid_is), # wind_mean_prob_computed_climatology_blurs_910_before_20200523
    (blur_15mi_grid_is, blur_25mi_grid_is), # hail_mean_prob_computed_climatology_blurs_910_before_20200523
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_full_13831
    (blur_25mi_grid_is, blur_25mi_grid_is), # wind_full_13831
    (blur_15mi_grid_is, blur_25mi_grid_is), # hail_full_13831
  ]

  validation_forecasts_blurred = PredictionForecasts.blurred(validation_forecasts, forecast_hour_range, blur_grid_is)

  # Make sure a forecast loads
  @assert nmodels <= size(Forecasts.data(validation_forecasts_blurred[1]), 2)

  println("\nLoading blurred hourly forecasts...")

  X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_blurred");

  println("\nChecking the hourly blurred forecast performance...")

  inspect_predictive_power(validation_forecasts_blurred, X, Ys, weights, model_names, event_names)

  # tornado_mean_prob_computed_climatology_blurs_910                 (75293.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.04281302
  # wind_mean_prob_computed_climatology_blurs_910                    (588576.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.10547267
  # hail_mean_prob_computed_climatology_blurs_910                    (264451.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.07189761
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 (75293.0)  feature 4 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.040235322
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    (588576.0) feature 5 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.10547267
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    (264451.0) feature 6 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.07189761
  # tornado_full_13831                                               (75293.0)  feature 7 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.047563255
  # wind_full_13831                                                  (588576.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.1163383
  # hail_full_13831                                                  (264451.0) feature 9 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.0753613


  bin_count = 6
  println("\nFinding $bin_count bins of equal positive weight, for calibrating the hourlies....")

  model_name_to_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]
    model_name_to_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end

  # model_name                                                       mean_y                 mean_ŷ                 Σweight              bin_max
  # tornado_mean_prob_computed_climatology_blurs_910                 1.9443193384695918e-5  2.056610038730587e-5   6.065345226027207e8  0.001259957
  # tornado_mean_prob_computed_climatology_blurs_910                 0.002472666316966537   0.002345205792965128   4.76918044762063e6   0.004403745
  # tornado_mean_prob_computed_climatology_blurs_910                 0.00657427976029817    0.006821190774966018   1.7937452619232535e6 0.010849744
  # tornado_mean_prob_computed_climatology_blurs_910                 0.015878771893361742   0.015195404705541685   742643.8353065848    0.022011435
  # tornado_mean_prob_computed_climatology_blurs_910                 0.0320217103577533     0.03062640920572916    368281.1955651045    0.044833306
  # tornado_mean_prob_computed_climatology_blurs_910                 0.07555310753219195    0.06903765604997823    156048.87840247154   1.0
  # wind_mean_prob_computed_climatology_blurs_910                    0.000152011683025612   0.00017360932885535863 6.003491102158413e8  0.007349782
  # wind_mean_prob_computed_climatology_blurs_910                    0.01197672844699669    0.011902110035814217   7.619720927109599e6  0.019179698
  # wind_mean_prob_computed_climatology_blurs_910                    0.028510196162986008   0.026074819343726248   3.2009669710893035e6 0.035851445
  # wind_mean_prob_computed_climatology_blurs_910                    0.05317124352744193    0.046317541634079924   1.7163344164347053e6 0.06126755
  # wind_mean_prob_computed_climatology_blurs_910                    0.0950349406648781     0.07948392053909535    960281.0552627444    0.107910745
  # wind_mean_prob_computed_climatology_blurs_910                    0.1761673568570009     0.16590597067497206    518008.6314294934    1.0
  # hail_mean_prob_computed_climatology_blurs_910                    6.766553284966636e-5   6.65920894115126e-5    6.025214282447281e8  0.0034476055
  # hail_mean_prob_computed_climatology_blurs_910                    0.006119813988567221   0.0058618730759041875  6.661932288779795e6  0.009904056
  # hail_mean_prob_computed_climatology_blurs_910                    0.015001760554223791   0.01401252592376581    2.7176941820005774e6 0.020123107
  # hail_mean_prob_computed_climatology_blurs_910                    0.028995820148646108   0.027156024776271024   1.406055902730763e6  0.0378312
  # hail_mean_prob_computed_climatology_blurs_910                    0.054912504986799046   0.05196074927548999    742463.0205870867    0.07595588
  # hail_mean_prob_computed_climatology_blurs_910                    0.12948328249727095    0.11940326369650435    314848.5797548294    1.0
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 1.943986564512464e-5   1.99918318840342e-5    6.066183877947023e8  0.001294466
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.0024895416564732647  0.002428613059598415   4.73673078578186e6   0.004569119
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.006871627658182113   0.006888658988190258   1.7162042612421513e6 0.010611137
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.015146156102804674   0.0148929353367278     778593.3880943656    0.021681689
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.03288613640864503    0.02975443318807044    358587.5554342866    0.043206826
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.07561738912104882    0.0675478864394674     155918.43635493517   1.0
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.000152011683025612   0.00017360932885535863 6.003491102158413e8  0.007349782
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.01197672844699669    0.011902110035814217   7.619720927109599e6  0.019179698
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.028510196162986008   0.026074819343726248   3.2009669710893035e6 0.035851445
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.05317124352744193    0.046317541634079924   1.7163344164347053e6 0.06126755
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.0950349406648781     0.07948392053909535    960281.0552627444    0.107910745
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.1761673568570009     0.16590597067497206    518008.6314294934    1.0
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    6.766553284966636e-5   6.65920894115126e-5    6.025214282447281e8  0.0034476055
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.006119813988567221   0.0058618730759041875  6.661932288779795e6  0.009904056
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.015001760554223791   0.01401252592376581    2.7176941820005774e6 0.020123107
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.028995820148646108   0.027156024776271024   1.406055902730763e6  0.0378312
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.054912504986799046   0.05196074927548999    742463.0205870867    0.07595588
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.12948328249727095    0.11940326369650435    314848.5797548294    1.0
  # tornado_full_13831                                               1.9433767699185833e-5  2.0002769098688164e-5  6.06803381293335e8   0.0012153604
  # tornado_full_13831                                               0.0025180521817733475  0.0023336320664977006  4.683162616125584e6  0.004538168
  # tornado_full_13831                                               0.006885780923916211   0.007216856966735946   1.7126045786351562e6 0.011742671
  # tornado_full_13831                                               0.017711000447631936   0.0162420525903571     665834.9625293612    0.023059275
  # tornado_full_13831                                               0.03148117549867956    0.03282356196542077    374601.9258365035    0.049979284
  # tornado_full_13831                                               0.09444591873851253    0.0749646351119608     124836.84526234865   1.0
  # wind_full_13831                                                  0.00015177235421138296 0.0001759040880477857  6.012909149650973e8  0.007898662
  # wind_full_13831                                                  0.012873070292758028   0.012634500069020822   7.089167727771163e6  0.020139106
  # wind_full_13831                                                  0.03034147485134284    0.027377279772583935   3.0077603277353644e6 0.037745386
  # wind_full_13831                                                  0.05700243569052713    0.048938992965258846   1.600978193070054e6  0.06497206
  # wind_full_13831                                                  0.10047507571321915    0.08491628121110167    908279.1504637003    0.115874514
  # wind_full_13831                                                  0.1952788061033616     0.1780347020156526     467321.8534272909    1.0
  # hail_full_13831                                                  6.764960230567528e-5   6.531116938039389e-5   6.026699636046774e8  0.0033965781
  # hail_full_13831                                                  0.006278763760885456   0.005701827717092226   6.493403477720737e6  0.00951148
  # hail_full_13831                                                  0.014420953862893071   0.013691235395506226   2.8271335290228724e6 0.0200999
  # hail_full_13831                                                  0.03036611534729695    0.027114783444007525   1.3426075132088661e6 0.037743744
  # hail_full_13831                                                  0.0563960274097572     0.05205358097325824    722922.2855994105    0.07645232
  # hail_full_13831                                                  0.1321928161649276     0.12157954443463385    308391.8084717989    1.0

  println("model_name_to_bins = $model_name_to_bins")
  # model_name_to_bins = Dict{String, Vector{Float32}}("wind_mean_prob_computed_climatology_blurs_910" => [0.007349782, 0.019179698, 0.035851445, 0.06126755, 0.107910745, 1.0], "hail_mean_prob_computed_climatology_blurs_910" => [0.0034476055, 0.009904056, 0.020123107, 0.0378312, 0.07595588, 1.0], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.007349782, 0.019179698, 0.035851445, 0.06126755, 0.107910745, 1.0], "tornado_mean_prob_computed_climatology_blurs_910" => [0.001259957, 0.004403745, 0.010849744, 0.022011435, 0.044833306, 1.0], "tornado_full_13831" => [0.0012153604, 0.004538168, 0.011742671, 0.023059275, 0.049979284, 1.0], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.0034476055, 0.009904056, 0.020123107, 0.0378312, 0.07595588, 1.0], "wind_full_13831" => [0.007898662, 0.020139106, 0.037745386, 0.06497206, 0.115874514, 1.0], "hail_full_13831" => [0.0033965781, 0.00951148, 0.0200999, 0.037743744, 0.07645232, 1.0], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.001294466, 0.004569119, 0.010611137, 0.021681689, 0.043206826, 1.0])


  println("\nFinding logistic coefficients for the $(bin_count-1) overlapping bin pairs, for calibrating the hourlies....")

  model_name_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]

    model_name_to_bins_logistic_coeffs[model_name] = find_single_predictor_logistic_coeffs(model_name, ŷ, y, weights, model_name_to_bins[model_name])
  end

  # model_name                                                       bin input_ŷ_min  input_ŷ_max count     pos_count weight      mean_input_ŷ  mean_y        input_logloss input_au_pr  mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
  # tornado_mean_prob_computed_climatology_blurs_910                 1-2 -1.0         0.004403745 663322993 25309.0   6.113037e8  3.870214e-5   3.858239e-5   0.0003139103  0.002298217  3.8582388e-5    0.00031364762    0.002182196    Float32[1.065771,   0.47239247]
  # tornado_mean_prob_computed_climatology_blurs_910                 2-3 0.001259957  0.010849744 7010378   25103.0   6.562926e6  0.0035685587  0.003593698   0.023236444   0.0064523756 0.0035936977    0.023232993      0.0066154865   Float32[0.9286899,  -0.38221243]
  # tornado_mean_prob_computed_climatology_blurs_910                 3-4 0.004403745  0.022011435 2694066   25050.0   2.536389e6  0.009273125   0.009298596   0.051659387   0.015761869  0.009298593     0.051656187      0.015739227    Float32[1.0567164,  0.26145768]
  # tornado_mean_prob_computed_climatology_blurs_910                 4-5 0.010849744  0.044833306 1173587   25037.0   1.110925e6  0.020310916   0.021230295   0.10108617    0.032052144  0.021230299     0.10106504       0.032177724    Float32[0.9913568,  0.01256458]
  # tornado_mean_prob_computed_climatology_blurs_910                 5-6 0.022011435  1.0         551181    24934.0   524330.1    0.042058196   0.044977333   0.17678712    0.09231928   0.044977333     0.17663798       0.09232958     Float32[1.0834966,  0.3218238]
  # wind_mean_prob_computed_climatology_blurs_910                    1-2 -1.0         0.019179698 659693732 196061.0  6.0796883e8 0.00032060355 0.00030021177 0.001899069   0.011736585  0.00030021174   0.0018959689     0.01160573     Float32[1.0957842,  0.44560102]
  # wind_mean_prob_computed_climatology_blurs_910                    2-3 0.007349782  0.035851445 11614151  196224.0  1.0820688e7 0.01609467    0.016867645   0.08350764    344.049      0.016867643     0.08347073       0.028150115    Float32[1.1037341,  0.462977]
  # wind_mean_prob_computed_climatology_blurs_910                    3-4 0.019179698  0.06126755  5280824   196380.0  4.9173015e6 0.03314034    0.037117884   0.15648143    0.05354868   0.037117884     0.15622857       0.053443216    Float32[1.0819606,  0.3897019]
  # wind_mean_prob_computed_climatology_blurs_910                    4-5 0.035851445  0.107910745 2878229   196265.0  2.6766155e6 0.058216535   0.068190545   0.24574691    53.775078    0.068190545     0.24485439       0.095833786    Float32[1.0832871,  0.39767647]
  # wind_mean_prob_computed_climatology_blurs_910                    5-6 0.06126755   1.0         1593684   196135.0  1.4782896e6 0.10976714    0.12346462    0.36465365    0.19677661   0.12346462      0.36336994       0.19677867     Float32[0.85362035, -0.15877303]
  # hail_mean_prob_computed_climatology_blurs_910                    1-2 -1.0         0.009904056 660957331 88050.0   6.0918336e8 0.00012996836 0.00013385086 0.0009393033  0.0056318296 0.00013385092   0.00093891355    0.005665922    Float32[1.0478816,  0.3216053]
  # hail_mean_prob_computed_climatology_blurs_910                    2-3 0.0034476055 0.020123107 10132742  87776.0   9.379627e6  0.008223479   0.008693308   0.048801612   74.197815    0.008693306     0.0487882        0.014693217    Float32[1.0098401,  0.10208285]
  # hail_mean_prob_computed_climatology_blurs_910                    3-4 0.009904056  0.0378312   4464969   87955.0   4.12375e6   0.018494006   0.019773249   0.095792115   134.8446     0.019773249     0.095747456      149.45749      Float32[0.98269755, 0.0008943792]
  # hail_mean_prob_computed_climatology_blurs_910                    4-5 0.020123107  0.07595588  2329210   88322.0   2.1485188e6 0.03572779    0.037951846   0.15869334    0.056688458  0.03795184      0.15862256       0.05683977     Float32[1.0002121,  0.06374693]
  # hail_mean_prob_computed_climatology_blurs_910                    5-6 0.0378312    1.0         1145940   88446.0   1.0573115e6 0.07204394    0.07711836    0.26085076    0.14733681   0.077118374     0.26065543       0.14734122     Float32[1.0119866,  0.10483526]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 1-2 -1.0         0.004569119 663378317 25311.0   6.1135514e8 3.8653638e-5  3.857802e-5   0.00031336662 0.0022558437 3.8578015e-5    0.00031326117    0.0021809454   Float32[1.039819,   0.28398132]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 2-3 0.001294466  0.010611137 6891276   25098.0   6.452935e6  0.003614794   0.0036549887  0.023552638   0.0067084613 0.0036549896    0.02355158       0.006773679    Float32[0.9637524,  -0.18635054]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 3-4 0.004569119  0.021681689 2649370   25055.0   2.4947975e6 0.0093866885  0.009454      0.052519795   0.015030104  0.009453997     0.05251938       7.3728514      Float32[1.0136148,  0.06931454]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 4-5 0.010611137  0.043206826 1201340   25054.0   1.1371809e6 0.019579215   0.020740109   0.09889399    0.03425656   0.02074011      0.09883718       0.03411985     Float32[1.114865,   0.49855673]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 5-6 0.021681689  1.0         540553    24927.0   514506.0    0.04120755    0.045835625   0.18034296    0.084556356  0.045835625     0.18007728       0.084564865    Float32[1.0013322,  0.116965175]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    1-2 -1.0         0.019179698 659693732 196061.0  6.0796883e8 0.00032060355 0.00030021177 0.001899069   0.011736585  0.00030021174   0.0018959689     0.01160573     Float32[1.0957842,  0.44560102]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    2-3 0.007349782  0.035851445 11614151  196224.0  1.0820688e7 0.01609467    0.016867645   0.08350764    344.049      0.016867643     0.08347073       0.028150115    Float32[1.1037341,  0.462977]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    3-4 0.019179698  0.06126755  5280824   196380.0  4.9173015e6 0.03314034    0.037117884   0.15648143    0.05354868   0.037117884     0.15622857       0.053443216    Float32[1.0819606,  0.3897019]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    4-5 0.035851445  0.107910745 2878229   196265.0  2.6766155e6 0.058216535   0.068190545   0.24574691    53.775078    0.068190545     0.24485439       0.095833786    Float32[1.0832871,  0.39767647]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    5-6 0.06126755   1.0         1593684   196135.0  1.4782896e6 0.10976714    0.12346462    0.36465365    0.19677661   0.12346462      0.36336994       0.19677867     Float32[0.85362035, -0.15877303]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    1-2 -1.0         0.009904056 660957331 88050.0   6.0918336e8 0.00012996836 0.00013385086 0.0009393033  0.0056318296 0.00013385092   0.00093891355    0.005665922    Float32[1.0478816,  0.3216053]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    2-3 0.0034476055 0.020123107 10132742  87776.0   9.379627e6  0.008223479   0.008693308   0.048801612   74.197815    0.008693306     0.0487882        0.014693217    Float32[1.0098401,  0.10208285]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    3-4 0.009904056  0.0378312   4464969   87955.0   4.12375e6   0.018494006   0.019773249   0.095792115   134.8446     0.019773249     0.095747456      149.45749      Float32[0.98269755, 0.0008943792]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    4-5 0.020123107  0.07595588  2329210   88322.0   2.1485188e6 0.03572779    0.037951846   0.15869334    0.056688458  0.03795184      0.15862256       0.05683977     Float32[1.0002121,  0.06374693]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    5-6 0.0378312    1.0         1145940   88446.0   1.0573115e6 0.07204394    0.07711836    0.26085076    0.14733681   0.077118374     0.26065543       0.14734122     Float32[1.0119866,  0.10483526]
  # tornado_full_13831                                               1-2 -1.0         0.004538168 663522521 25328.0   6.114865e8  3.7722053e-5  3.8569822e-5  0.00031415743 0.0023053437 3.8569815e-5    0.0003138219     0.0023922462   Float32[1.0710595,  0.5394862]
  # tornado_full_13831                                               2-3 0.0012153604 0.011742671 6826598   25086.0   6.395767e6  0.003641221   0.003687606   0.023732848   0.0068456633 0.003687606     0.02372406       0.0067909104   Float32[0.89414734, -0.5616081]
  # tornado_full_13831                                               3-4 0.004538168  0.023059275 2521603   25053.0   2.3784395e6 0.009743426   0.009916259   0.05431361    0.017359346  0.009916258     0.054299563      0.017177863    Float32[1.108447,   0.50613326]
  # tornado_full_13831                                               4-5 0.011742671  0.049979284 1096364   25059.0   1.0404369e6 0.022212109   0.022668853   0.10695673    0.032490294  0.022668855     0.10690148       0.032723896    Float32[0.83939207, -0.5758451]
  # tornado_full_13831                                               5-6 0.023059275  1.0         524116    24912.0   499438.75   0.043356903   0.047219485   0.1817415     0.10419383   0.04721949      0.18125294       0.10419369     Float32[1.2171175,  0.7334359]
  # wind_full_13831                                                  1-2 -1.0         0.020139106 660132696 196107.0  6.0838e8    0.0003210783  0.0003000077  0.0018850672  0.01253512   0.0003000077    0.0018816878     0.012375654    Float32[1.0976788,  0.44982857]
  # wind_full_13831                                                  2-3 0.007898662  0.037745386 10837713  196423.0  1.0096928e7 0.017026208   0.01807671    0.08825989    0.030129705  0.018076705     0.08820334       0.030392418    Float32[1.1166184,  0.5214092]
  # wind_full_13831                                                  3-4 0.020139106  0.06497206  4952477   196504.0  4.6087385e6 0.034867365   0.039602928   0.16435848    0.05650685   0.03960292      0.16402282       0.05666814     Float32[1.0803565,  0.39499816]
  # wind_full_13831                                                  4-5 0.037745386  0.115874514 2700227   196268.0  2.5092575e6 0.06196173    0.072738275   0.25753838    0.10074633   0.07273827      0.2565792        0.100803725    Float32[1.0272967,  0.24607334]
  # wind_full_13831                                                  5-6 0.06497206   1.0         1483067   195965.0  1.375601e6  0.11655065    0.13268198    0.37929082    0.22102256   0.132682        0.37793967       0.22102539     Float32[0.9264786,  0.009931214]
  # hail_full_13831                                                  1-2 -1.0         0.00951148  660934254 87998.0   6.091634e8  0.00012539385 0.00013385723 0.00093786436 0.0058163637 0.00013385725   0.0009370643     0.0059634214   Float32[1.0600259,  0.43388337]
  # hail_full_13831                                                  2-3 0.0033965781 0.0200999   10070263  87700.0   9.320537e6  0.008125199   0.008748478   0.049155217   0.014804893  0.0087484745    0.049130168      0.014807932    Float32[0.96319646, -0.097637914]
  # hail_full_13831                                                  3-4 0.00951148   0.037743744 4516233   87979.0   4.169741e6  0.018013459   0.019555109   0.094599314   0.030076755  0.019555109     0.09452413       0.030354485    Float32[1.0792756,  0.39443073]
  # hail_full_13831                                                  4-5 0.0200999    0.07645232  2240170   88414.0   2.0655298e6 0.03584321    0.03947642    0.16378525    0.05771606   0.03947642      0.16359507       0.05759762     Float32[0.9629313,  -0.01864577]
  # hail_full_13831                                                  5-6 0.037743744  1.0         1117753   88474.0   1.0313141e6 0.07284379    0.07906139    0.26499486    0.15608536   0.07906139      0.26470453       0.15608558     Float32[1.0167043,  0.13172606]

  println("model_name_to_bins_logistic_coeffs = $model_name_to_bins_logistic_coeffs")
  # model_name_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [[1.0957842, 0.44560102], [1.1037341, 0.462977], [1.0819606, 0.3897019], [1.0832871, 0.39767647], [0.85362035, -0.15877303]], "hail_mean_prob_computed_climatology_blurs_910" => [[1.0478816, 0.3216053], [1.0098401, 0.10208285], [0.98269755, 0.0008943792], [1.0002121, 0.06374693], [1.0119866, 0.10483526]], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0957842, 0.44560102], [1.1037341, 0.462977], [1.0819606, 0.3897019], [1.0832871, 0.39767647], [0.85362035, -0.15877303]], "tornado_mean_prob_computed_climatology_blurs_910" => [[1.065771, 0.47239247], [0.9286899, -0.38221243], [1.0567164, 0.26145768], [0.9913568, 0.01256458], [1.0834966, 0.3218238]], "tornado_full_13831" => [[1.0710595, 0.5394862], [0.89414734, -0.5616081], [1.108447, 0.50613326], [0.83939207, -0.5758451], [1.2171175, 0.7334359]], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0478816, 0.3216053], [1.0098401, 0.10208285], [0.98269755, 0.0008943792], [1.0002121, 0.06374693], [1.0119866, 0.10483526]], "wind_full_13831" => [[1.0976788, 0.44982857], [1.1166184, 0.5214092], [1.0803565, 0.39499816], [1.0272967, 0.24607334], [0.9264786, 0.009931214]], "hail_full_13831" => [[1.0600259, 0.43388337], [0.96319646, -0.097637914], [1.0792756, 0.39443073], [0.9629313, -0.01864577], [1.0167043, 0.13172606]], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.039819, 0.28398132], [0.9637524, -0.18635054], [1.0136148, 0.06931454], [1.114865, 0.49855673], [1.0013322, 0.116965175]])

  println("\nCopy model_name_to_bins and model_name_to_bins_logistic_coeffs into the prediction model.")
  # println("\nYou can stop the process now if you want.")

  println("\nChecking the calibration...")

  hour_models = make_calibrated_hourly_models(model_name_to_bins, model_name_to_bins_logistic_coeffs)
  validation_forecasts_calibrated = PredictionForecasts.simple_prediction_forecasts(validation_forecasts_blurred, hour_models)

  X, Ys, weights = nothing, nothing, nothing # free
  X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_calibrated; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_calibrated");

  println("\nChecking the hourly calibrated blurred forecast performance (should be the same if calibration was monotonic)...")
  inspect_predictive_power(validation_forecasts_calibrated, X, Ys, weights, model_names, event_names)
  # tornado_mean_prob_computed_climatology_blurs_910                 (75293.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.04281302
  # wind_mean_prob_computed_climatology_blurs_910                    (588576.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.10547266
  # hail_mean_prob_computed_climatology_blurs_910                    (264451.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07189761
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 (75293.0)  feature 4 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.04023532
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    (588576.0) feature 5 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.10547266
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    (264451.0) feature 6 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07189761
  # tornado_full_13831                                               (75293.0)  feature 7 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.047563255
  # wind_full_13831                                                  (588576.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.11633829
  # hail_full_13831                                                  (264451.0) feature 9 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07536129

  println("\nPlotting reliability...")
  Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)
  # ŷ_tornado_mean_prob_computed_climatology_blurs_910,y_tornado_mean_prob_computed_climatology_blurs_910,ŷ_wind_mean_prob_computed_climatology_blurs_910,y_wind_mean_prob_computed_climatology_blurs_910,ŷ_hail_mean_prob_computed_climatology_blurs_910,y_hail_mean_prob_computed_climatology_blurs_910,ŷ_tornado_mean_prob_computed_climatology_blurs_910_before_20200523,y_tornado_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_wind_mean_prob_computed_climatology_blurs_910_before_20200523,y_wind_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_hail_mean_prob_computed_climatology_blurs_910_before_20200523,y_hail_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_tornado_full_13831,y_tornado_full_13831,ŷ_wind_full_13831,y_wind_full_13831,ŷ_hail_full_13831,y_hail_full_13831,
  # 6.0495126e-6,6.004684e-6,4.6187684e-5,4.7099868e-5,2.0822368e-5,2.0877113e-5,6.2185604e-6,5.9962063e-6,4.6187684e-5,4.7099868e-5,2.0822368e-5,2.0877113e-5,6.1531127e-6,6.0045345e-6,4.4799577e-5,4.6982437e-5,2.0477783e-5,2.0870466e-5,
  # 0.00028455627,0.00030199366,0.0022196237,0.0022967476,0.0011086342,0.0011489157,0.00029244594,0.00032168234,0.0022196237,0.0022967476,0.0011086342,0.0011489157,0.00026546858,0.00029663646,0.0022928612,0.0023896291,0.0011119856,0.001162719,
  # 0.0007443073,0.00077677146,0.0046575447,0.0047695325,0.002368193,0.00252727,0.0007550663,0.0007617509,0.0046575447,0.0047695325,0.002368193,0.00252727,0.0007241435,0.0007362149,0.004952928,0.0047245077,0.0023980946,0.0025161966,
  # 0.001406874,0.001372048,0.0076008053,0.007426497,0.0038500822,0.0039977334,0.0014118422,0.001485591,0.0076008053,0.007426497,0.0038500822,0.0039977334,0.0014086519,0.0014777686,0.008171466,0.008041755,0.003946606,0.0039206226,
  # 0.0022536025,0.0023284957,0.011060749,0.010771918,0.0056386786,0.0053963666,0.0022333506,0.0022249315,0.011060749,0.010771918,0.0056386786,0.0053963666,0.0022743954,0.0023369447,0.0118081225,0.011538035,0.005770756,0.0057041226,
  # 0.003212026,0.003129776,0.014996874,0.014889702,0.0077293064,0.007518339,0.0032476534,0.0030731529,0.014996874,0.014889702,0.0077293064,0.007518339,0.0032991148,0.0031599442,0.015952142,0.015929786,0.0077474723,0.007746222,
  # 0.004253359,0.0043680132,0.019382972,0.01970002,0.010036042,0.009916169,0.0044043697,0.0043696673,0.019382972,0.01970002,0.010036042,0.009916169,0.004397003,0.0044003646,0.020619202,0.021144513,0.0098469565,0.009742342,
  # 0.005507147,0.005525653,0.024382453,0.024792003,0.01272798,0.013114877,0.005750796,0.0058354884,0.024382453,0.024792003,0.01272798,0.013114877,0.0057230773,0.0057523674,0.02600502,0.025979558,0.012369373,0.012067936,
  # 0.007167853,0.0070996718,0.030061856,0.0301095,0.015819162,0.015917048,0.0073799896,0.0074800844,0.030061856,0.0301095,0.015819162,0.015917048,0.0075928513,0.0072833244,0.03221223,0.03161772,0.015530437,0.015330491,
  # 0.009428681,0.00881761,0.036563985,0.03572255,0.019433325,0.018998021,0.009429831,0.009169049,0.036563985,0.03572255,0.019433325,0.018998021,0.01026361,0.009662776,0.03914892,0.039424997,0.019474972,0.01945359,
  # 0.012277016,0.012680451,0.043964975,0.043701384,0.023582537,0.023669435,0.011917669,0.011935267,0.043964975,0.043701384,0.023582537,0.023669435,0.013689927,0.013830343,0.046941694,0.04801962,0.024301428,0.024507402,
  # 0.015432273,0.015928734,0.052428287,0.052150417,0.028347956,0.029082334,0.014870291,0.01517907,0.052428287,0.052150417,0.028347956,0.029082334,0.017167425,0.018398501,0.055963457,0.055633806,0.029772168,0.03001216,
  # 0.019217266,0.018839393,0.06221131,0.06309241,0.03414831,0.033839498,0.018725637,0.018328968,0.06221131,0.06309241,0.03414831,0.033839498,0.020688688,0.021057948,0.06637986,0.066011526,0.035898324,0.036441717,
  # 0.02370082,0.02407947,0.07394913,0.07433568,0.041541785,0.03995694,0.023815615,0.023559948,0.07394913,0.07433568,0.041541785,0.03995694,0.02426165,0.025226573,0.07843019,0.078780584,0.042901717,0.04263499,
  # 0.029050767,0.029027393,0.0880882,0.08833129,0.050874542,0.050351053,0.030067062,0.02968811,0.0880882,0.08833129,0.050874542,0.050351053,0.028935073,0.028913405,0.09286435,0.0933398,0.051738426,0.052434433,
  # 0.036066838,0.036490936,0.10364184,0.10508745,0.06269969,0.06302556,0.037671886,0.036350504,0.10364184,0.10508745,0.06269969,0.06302556,0.036926188,0.03251899,0.10963969,0.11118981,0.06336136,0.06260348,
  # 0.04576804,0.04344247,0.12044023,0.12302868,0.07806904,0.08047625,0.04654805,0.050908495,0.12044023,0.12302868,0.07806904,0.08047625,0.050845355,0.048398882,0.12949318,0.12960927,0.07952274,0.07946183,
  # 0.05997108,0.055935655,0.14274588,0.14746752,0.09876565,0.10359786,0.058451675,0.058184117,0.14274588,0.14746752,0.09876565,0.10359786,0.06879845,0.073836155,0.15662718,0.15697022,0.102680996,0.10188601,
  # 0.081727594,0.08649645,0.1779675,0.17748934,0.13193563,0.13206011,0.07752831,0.08003294,0.1779675,0.17748934,0.13193563,0.13206011,0.093195826,0.10680757,0.20079821,0.19912603,0.14015123,0.13652179,
  # 0.13379784,0.1353726,0.2616117,0.25352815,0.2149755,0.20887364,0.124385,0.121273816,0.2616117,0.25352815,0.2149755,0.20887364,0.15691693,0.15076944,0.30707332,0.3066239,0.22729546,0.23269866,

  X, Ys, weights = nothing, nothing, nothing # free

  println("\nLoading 0Z and 12Z daily accumulators...")

  # vec of (model_name, var_name)
  models = map(m -> (m[1], m[2]), hour_models)

  validation_forecasts_day_accumulators, validation_forecasts_day2_accumulators, validation_forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(validation_forecasts_calibrated, models; module_name = "daily_accs")

  # ensure we don't accidentally use these
  validation_forecasts = nothing
  validation_forecasts_blurred = nothing
  validation_forecasts_calibrated = nothing

  _, validation_day_acc_forecasts, _ = TrainingShared.forecasts_train_validation_test(validation_forecasts_day_accumulators; just_hours_near_storm_events = false)

  validation_day_acc_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_day_acc_forecasts)

  @time Forecasts.data(validation_day_acc_forecasts_0z_12z[1]); # Check if a forecast loads

  # Two features per model, total and max, have to double the column names
  acc_model_names = map(i -> model_names[div(i - 1, 2) + 1] * (isodd(i) ? "_tot" : "_max"), 1:2*length(model_names))
  acc_event_names = vcat(map(name -> [name, name], event_names)...)

  println("Drawing Dec 11 to check...")

  dec11 = filter(f -> Forecasts.time_title(f) == "2021-12-11 00Z +35", validation_day_acc_forecasts_0z_12z)[1];
  dec11_data = Forecasts.data(dec11);
  for i in 1:size(dec11_data,2)
    model_name = acc_model_names[i]
    PlotMap.plot_debug_map("dec11_0z_12z_day_accs_$(i)_$model_name", dec11.grid, dec11_data[:,i]);
  end
  for event_name in unique(acc_event_names)
    labeler = TrainingShared.event_name_to_day_labeler[event_name]
    PlotMap.plot_debug_map("dec11_0z_12z_day_$event_name", dec11.grid, labeler(dec11));
  end

  # scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_accs_1.pdf ./
  # scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_tornado.pdf ./

  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_day_acc_forecasts_0z_12z;
      event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
      save_dir = "validation_day_accumulators_forecasts_0z_12z",
    );


  println("\nChecking daily accumulator performance...")

  inspect_predictive_power(validation_day_acc_forecasts_0z_12z, X, Ys, weights, acc_model_names, acc_event_names)

  # tornado_mean_prob_computed_climatology_blurs_910_tot (20606.0)                 feature 1 independent events total TORPROB:calculated:day   fcst::	AU-PR-curve: 0.14386787
  # tornado_mean_prob_computed_climatology_blurs_910_max (20606.0)                 feature 2 highest hourly TORPROB:calculated:day             fcst::	AU-PR-curve: 0.13589665
  # wind_mean_prob_computed_climatology_blurs_910_tot (148934.0)                   feature 3 independent events total WINDPROB:calculated:day  fcst::	AU-PR-curve: 0.38566333
  # wind_mean_prob_computed_climatology_blurs_910_max (148934.0)                   feature 4 highest hourly WINDPROB:calculated:day            fcst::	AU-PR-curve: 0.36165154
  # hail_mean_prob_computed_climatology_blurs_910_tot (67838.0)                    feature 5 independent events total HAILPROB:calculated:day  fcst::	AU-PR-curve: 0.2565145
  # hail_mean_prob_computed_climatology_blurs_910_max (67838.0)                    feature 6 highest hourly HAILPROB:calculated:day            fcst::	AU-PR-curve: 0.2389621
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523_tot (20606.0) feature 7 independent events total TORPROB:calculated:day   fcst::	AU-PR-curve: 0.1433445
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523_max (20606.0) feature 8 highest hourly TORPROB:calculated:day             fcst::	AU-PR-curve: 0.1342907
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523_tot (148934.0)   feature 9 independent events total WINDPROB:calculated:day  fcst::	AU-PR-curve: 0.38566333
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523_max (148934.0)   feature 10 highest hourly WINDPROB:calculated:day           fcst::	AU-PR-curve: 0.36165154
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523_tot (67838.0)    feature 11 independent events total HAILPROB:calculated:day fcst::	AU-PR-curve: 0.2565145
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523_max (67838.0)    feature 12 highest hourly HAILPROB:calculated:day           fcst::	AU-PR-curve: 0.2389621
  # tornado_full_13831_tot (20606.0)                                               feature 13 independent events total TORPROB:calculated:day  fcst::	AU-PR-curve: 0.15398282
  # tornado_full_13831_max (20606.0)                                               feature 14 highest hourly TORPROB:calculated:day            fcst::	AU-PR-curve: 0.14738984
  # wind_full_13831_tot (148934.0)                                                 feature 15 independent events total WINDPROB:calculated:day fcst::	AU-PR-curve: 0.4025323
  # wind_full_13831_max (148934.0)                                                 feature 16 highest hourly WINDPROB:calculated:day           fcst::	AU-PR-curve: 0.3749552
  # hail_full_13831_tot (67838.0)                                                  feature 17 independent events total HAILPROB:calculated:day fcst::	AU-PR-curve: 0.2654488
  # hail_full_13831_max (67838.0)                                                  feature 18 highest hourly HAILPROB:calculated:day           fcst::	AU-PR-curve: 0.24634801

  println("\nPlotting daily accumulators reliability...")

  Metrics.reliability_curves_midpoints(20, X, Ys, acc_event_names, weights, acc_model_names)
  # ŷ_tornado_mean_prob_computed_climatology_blurs_910_tot,y_tornado_mean_prob_computed_climatology_blurs_910_tot,ŷ_tornado_mean_prob_computed_climatology_blurs_910_max,y_tornado_mean_prob_computed_climatology_blurs_910_max,ŷ_wind_mean_prob_computed_climatology_blurs_910_tot,y_wind_mean_prob_computed_climatology_blurs_910_tot,ŷ_wind_mean_prob_computed_climatology_blurs_910_max,y_wind_mean_prob_computed_climatology_blurs_910_max,ŷ_hail_mean_prob_computed_climatology_blurs_910_tot,y_hail_mean_prob_computed_climatology_blurs_910_tot,ŷ_hail_mean_prob_computed_climatology_blurs_910_max,y_hail_mean_prob_computed_climatology_blurs_910_max,ŷ_tornado_mean_prob_computed_climatology_blurs_910_before_20200523_tot,y_tornado_mean_prob_computed_climatology_blurs_910_before_20200523_tot,ŷ_tornado_mean_prob_computed_climatology_blurs_910_before_20200523_max,y_tornado_mean_prob_computed_climatology_blurs_910_before_20200523_max,ŷ_wind_mean_prob_computed_climatology_blurs_910_before_20200523_tot,y_wind_mean_prob_computed_climatology_blurs_910_before_20200523_tot,ŷ_wind_mean_prob_computed_climatology_blurs_910_before_20200523_max,y_wind_mean_prob_computed_climatology_blurs_910_before_20200523_max,ŷ_hail_mean_prob_computed_climatology_blurs_910_before_20200523_tot,y_hail_mean_prob_computed_climatology_blurs_910_before_20200523_tot,ŷ_hail_mean_prob_computed_climatology_blurs_910_before_20200523_max,y_hail_mean_prob_computed_climatology_blurs_910_before_20200523_max,ŷ_tornado_full_13831_tot,y_tornado_full_13831_tot,ŷ_tornado_full_13831_max,y_tornado_full_13831_max,ŷ_wind_full_13831_tot,y_wind_full_13831_tot,ŷ_wind_full_13831_max,y_wind_full_13831_max,ŷ_hail_full_13831_tot,y_hail_full_13831_tot,ŷ_hail_full_13831_max,y_hail_full_13831_max,
  # 0.00014371765,0.00011854989,2.5485231e-5,0.00011913049,0.0012297512,0.0008701467,0.000262593,0.0008738003,0.00059601717,0.00038282646,0.00012925918,0.00038321514,0.00013798143,0.00011862929,2.4733847e-5,0.00011934237,0.0012297512,0.0008701467,0.000262593,0.0008738003,0.00059601717,0.00038282646,0.00012925918,0.00038321514,0.00014385104,0.00011868292,2.4220833e-5,0.00011878335,0.0011387498,0.000870939,0.00023141767,0.00087375636,0.0005521953,0.00038311855,0.000114642084,0.0003835346,
  # 0.0032212206,0.0028949024,0.0007328072,0.0026828013,0.030645637,0.024132034,0.007122821,0.023416836,0.01712104,0.0141131375,0.0039314227,0.01506661,0.0032391956,0.0028590066,0.00074113486,0.0025757223,0.030645637,0.024132034,0.007122821,0.023416836,0.01712104,0.0141131375,0.0039314227,0.01506661,0.003032044,0.0029303133,0.0006838064,0.002877402,0.029380055,0.022555685,0.0066226143,0.022356458,0.017259995,0.01387913,0.0038320096,0.014695416,
  # 0.0077111647,0.005475896,0.0018568172,0.005817962,0.053482648,0.043110237,0.012870458,0.039859537,0.030108089,0.02486942,0.0067826007,0.024684837,0.007660165,0.0058515305,0.0018591956,0.0063672774,0.053482648,0.043110237,0.012870458,0.039859537,0.030108089,0.02486942,0.0067826007,0.024684837,0.007449875,0.005211285,0.0017348687,0.005909616,0.053788126,0.041794978,0.012363922,0.040577896,0.030817775,0.024084361,0.0067538135,0.024019098,
  # 0.013373986,0.011416487,0.0032142824,0.010724233,0.076623976,0.06381911,0.019035097,0.062158406,0.04314916,0.03799304,0.009883217,0.033739045,0.013172845,0.010705428,0.0032145288,0.009520844,0.076623976,0.06381911,0.019035097,0.062158406,0.04314916,0.03799304,0.009883217,0.033739045,0.012997985,0.0117155295,0.0030777645,0.01015733,0.079117365,0.060828898,0.018541895,0.060956534,0.044571526,0.035672765,0.009622827,0.03422622,
  # 0.018792802,0.017973358,0.004466764,0.016642686,0.101444796,0.078618966,0.025421342,0.084072955,0.05724687,0.043834325,0.013413917,0.04522354,0.019258246,0.015927715,0.004707115,0.016255789,0.101444796,0.078618966,0.025421342,0.084072955,0.05724687,0.043834325,0.013413917,0.04522354,0.018202595,0.017957337,0.004373065,0.015643895,0.106388144,0.08066211,0.025288291,0.08250904,0.058636103,0.046434868,0.012660346,0.044679537,
  # 0.024892326,0.021170935,0.0059697037,0.019250356,0.12850013,0.10172512,0.032129552,0.106377736,0.07362275,0.05483741,0.01727769,0.05755578,0.025921049,0.021816587,0.0062950132,0.020818993,0.12850013,0.10172512,0.032129552,0.106377736,0.07362275,0.05483741,0.01727769,0.05755578,0.024484172,0.020132925,0.005865778,0.020169087,0.13519184,0.1058878,0.032479934,0.10394624,0.0739088,0.056443166,0.016228655,0.055301793,
  # 0.032443617,0.026713712,0.007961658,0.025637694,0.15625584,0.1277942,0.0392317,0.1255596,0.09081931,0.07348394,0.021410188,0.070775956,0.033293914,0.029611915,0.008237798,0.025963817,0.15625584,0.1277942,0.0392317,0.1255596,0.09081931,0.07348394,0.021410188,0.070775956,0.032620717,0.026062839,0.007866154,0.028508145,0.16395418,0.13810724,0.039910346,0.1335301,0.090915024,0.06948795,0.02044969,0.07081661,
  # 0.040920567,0.039634444,0.010382925,0.03680284,0.18436654,0.15355548,0.04695955,0.14597875,0.10829716,0.08665808,0.025697414,0.08715733,0.041930422,0.03551649,0.010438342,0.036984917,0.18436654,0.15355548,0.04695955,0.14597875,0.10829716,0.08665808,0.025697414,0.08715733,0.042019937,0.038459085,0.010403755,0.036606018,0.19329305,0.16507989,0.047834132,0.15689865,0.10909011,0.087729275,0.025050864,0.088557646,
  # 0.050288744,0.048154525,0.013005414,0.048522636,0.2135787,0.18068554,0.055302653,0.17440021,0.12692535,0.10292174,0.030328501,0.09892081,0.05184906,0.04810046,0.012720646,0.046039872,0.2135787,0.18068554,0.055302653,0.17440021,0.12692535,0.10292174,0.030328501,0.09892081,0.053018544,0.043945048,0.013496416,0.048868727,0.2232924,0.19603154,0.056594066,0.18426885,0.12774304,0.106429465,0.029899785,0.10245252,
  # 0.061468825,0.054680757,0.015799599,0.057108276,0.24384232,0.21493553,0.0641413,0.20440973,0.14638488,0.12440862,0.03562318,0.11006684,0.062401365,0.05992028,0.015582615,0.048594456,0.24384232,0.21493553,0.0641413,0.20440973,0.14638488,0.12440862,0.03562318,0.11006684,0.06642444,0.052665576,0.016637186,0.05934798,0.25424004,0.22925441,0.065935746,0.21357286,0.14694498,0.12696111,0.035110887,0.115275666,
  # 0.075385734,0.061245397,0.019083753,0.063516416,0.27409223,0.24789403,0.073744245,0.23321488,0.1672602,0.1387689,0.041772585,0.12659584,0.07527941,0.062734306,0.019285394,0.063829966,0.27409223,0.24789403,0.073744245,0.23321488,0.1672602,0.1387689,0.041772585,0.12659584,0.08107264,0.070676826,0.019927874,0.064522885,0.2858035,0.25447226,0.0756966,0.2454128,0.16746707,0.13990355,0.040480293,0.13690169,
  # 0.0919406,0.076363705,0.022989828,0.07343763,0.3050717,0.27508408,0.08446677,0.2609437,0.19005838,0.160872,0.048830897,0.1459488,0.091148145,0.07663603,0.023716912,0.07440813,0.3050717,0.27508408,0.08446677,0.2609437,0.19005838,0.160872,0.048830897,0.1459488,0.096126005,0.09132529,0.023415359,0.06965363,0.31891453,0.27674177,0.086596936,0.27173054,0.19013132,0.16622524,0.04654984,0.14642423,
  # 0.11079618,0.09383401,0.02756095,0.08445129,0.33705336,0.31090727,0.095787905,0.29406026,0.21502936,0.17794403,0.056696557,0.1799489,0.10958255,0.090101264,0.028870486,0.09006824,0.33705336,0.31090727,0.095787905,0.29406026,0.21502936,0.17794403,0.056696557,0.1799489,0.1127624,0.10586217,0.02733126,0.08638261,0.35279945,0.3135548,0.09846824,0.29877138,0.21497642,0.18051401,0.054032452,0.16099097,
  # 0.13126391,0.1301481,0.033945978,0.08529338,0.37166423,0.3282471,0.107868016,0.3145146,0.24374811,0.19066976,0.06559966,0.1920556,0.13048755,0.11972601,0.035019867,0.09829473,0.37166423,0.3282471,0.107868016,0.3145146,0.24374811,0.19066976,0.06559966,0.1920556,0.13335794,0.11975571,0.03282819,0.09087089,0.38781336,0.3464552,0.11160649,0.32734466,0.24418594,0.20024118,0.06298922,0.1873912,
  # 0.15298903,0.14467295,0.042055145,0.12449045,0.4104013,0.35525218,0.12045677,0.33969894,0.2769115,0.22288981,0.07649675,0.21466385,0.15251422,0.15912239,0.042104386,0.12033973,0.4104013,0.35525218,0.12045677,0.33969894,0.2769115,0.22288981,0.07649675,0.21466385,0.1576756,0.15105228,0.042469647,0.09731387,0.4255537,0.38424,0.12592366,0.35953596,0.27856606,0.22248586,0.07426375,0.21168967,
  # 0.1797352,0.1731783,0.05083011,0.1610262,0.45462686,0.3868238,0.13581239,0.36846274,0.3148967,0.26059428,0.08995785,0.24500646,0.17807546,0.17186634,0.04996351,0.1508149,0.45462686,0.3868238,0.13581239,0.36846274,0.3148967,0.26059428,0.08995785,0.24500646,0.18533225,0.18059108,0.05621599,0.15946102,0.46896106,0.40391076,0.14265758,0.3883551,0.3188595,0.25637585,0.08828152,0.24862266,
  # 0.21614125,0.18266566,0.060906503,0.18306604,0.50790733,0.43873346,0.15543692,0.41167563,0.35906512,0.32449996,0.10701924,0.28394043,0.21007343,0.19565172,0.059323102,0.18671848,0.50790733,0.43873346,0.15543692,0.41167563,0.35906512,0.32449996,0.10701924,0.28394043,0.22203691,0.18012376,0.06950047,0.21052974,0.5202321,0.4688886,0.16425996,0.41074365,0.36679864,0.30269524,0.10618553,0.27730292,
  # 0.2689792,0.20031914,0.07410705,0.22378543,0.5714288,0.51357776,0.18122993,0.47512513,0.41264138,0.36745465,0.12968004,0.32897413,0.25573993,0.20976104,0.07104428,0.2310118,0.5714288,0.51357776,0.18122993,0.47512513,0.41264138,0.36745465,0.12968004,0.32897413,0.2796896,0.19657871,0.084017456,0.2872454,0.5821669,0.521709,0.19490711,0.47533125,0.42432866,0.37652564,0.13114405,0.3286453,
  # 0.34296295,0.2776258,0.09374712,0.26049155,0.65302,0.55564857,0.22010973,0.5351367,0.4859298,0.41500193,0.16481319,0.39508668,0.3267192,0.24819547,0.087434195,0.25942492,0.65302,0.55564857,0.22010973,0.5351367,0.4859298,0.41500193,0.16481319,0.39508668,0.3760596,0.31925818,0.106317304,0.26658607,0.6646871,0.58040583,0.24445422,0.5481529,0.49888816,0.42677727,0.16736478,0.40831068,
  # 0.52142614,0.2960098,0.15566358,0.26858437,0.79193634,0.7189492,0.31165493,0.643699,0.62677944,0.49726826,0.25148132,0.45710906,0.51289064,0.2960403,0.14049271,0.2572229,0.79193634,0.7189492,0.31165493,0.643699,0.62677944,0.49726826,0.25148132,0.45710906,0.54714584,0.3338828,0.18405898,0.31421718,0.80317277,0.7463828,0.3592062,0.68864423,0.6451874,0.56037027,0.25789347,0.51332194,


  bin_count = 4
  println("\nFinding $bin_count bins of equal positive weight, to bin the total_prob accs for calibrating the dailies....")

  model_name_to_day_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc
    y = Ys[event_name]
    model_name_to_day_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end

  # model_name                                                       mean_y                mean_ŷ                Σweight             bin_max
  # tornado_mean_prob_computed_climatology_blurs_910                 0.0005490906696204292 0.0006534959256070859 8.805662290875912e6 0.02166239
  # tornado_mean_prob_computed_climatology_blurs_910                 0.03356390752877112   0.03751259242483352   144040.49411451817  0.06790343
  # tornado_mean_prob_computed_climatology_blurs_910                 0.09127118677384351   0.1038085683828254    52967.86109626293   0.16562855
  # tornado_mean_prob_computed_climatology_blurs_910                 0.21518865136850374   0.2820103313987108    22461.202476978302  1.0
  # wind_mean_prob_computed_climatology_blurs_910                    0.0040247006818123635 0.005163470099723012  8.602000178638816e6 0.115368165
  # wind_mean_prob_computed_climatology_blurs_910                    0.1455312349557188    0.17472390557773823   237893.63209080696  0.25932494
  # wind_mean_prob_computed_climatology_blurs_910                    0.2985294990979474    0.33353633283732503   115971.76984763145  0.43146545
  # wind_mean_prob_computed_climatology_blurs_910                    0.49980180453117223   0.5720345219384813    69266.26798641682   1.0
  # hail_mean_prob_computed_climatology_blurs_910                    0.001803390632154808  0.0023165098751245523 8.698875173536897e6 0.06547251
  # hail_mean_prob_computed_climatology_blurs_910                    0.08177830971962885   0.1020199331819231    191831.1800661087   0.15664904
  # hail_mean_prob_computed_climatology_blurs_910                    0.17373140542853072   0.21251068196506565   90297.25231820345   0.29551116
  # hail_mean_prob_computed_climatology_blurs_910                    0.3554729435714935    0.417258032223628     44128.24264246225   1.0
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.000548789486725683  0.0006636681394457174 8.809325107719958e6 0.022665959
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.03442825392985085   0.03861256292794679   140445.29216206074  0.06817255
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.09120689287253764   0.10320479897123526   53013.07734787464   0.1642723
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.21623924697288283   0.27559180541300704   22348.371333777905  1.0
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.0040247006818123635 0.005163470099723012  8.602000178638816e6 0.115368165
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.1455312349557188    0.17472390557773823   237893.63209080696  0.25932494
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.2985294990979474    0.33353633283732503   115971.76984763145  0.43146545
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.49980180453117223   0.5720345219384813    69266.26798641682   1.0
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.001803390632154808  0.0023165098751245523 8.698875173536897e6 0.06547251
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.08177830971962885   0.1020199331819231    191831.1800661087   0.15664904
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.17373140542853072   0.21251068196506565   90297.25231820345   0.29551116
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.3554729435714935    0.417258032223628     44128.24264246225   1.0
  # tornado_full_13831                                               0.0005491449770002533 0.0006369487877066247 8.804864899285913e6 0.021043906
  # tornado_full_13831                                               0.03207420128115769   0.03865792287987463   150732.980168283    0.074019335
  # tornado_full_13831                                               0.10080232163872288   0.10940486581450679   47960.05282700062   0.17095083
  # tornado_full_13831                                               0.2240312898444039    0.2908434622530887    21573.916282474995  1.0
  # wind_full_13831                                                  0.004014741997566956  0.005232740004959892  8.623402529861748e6 0.12129908
  # wind_full_13831                                                  0.15525518233282076   0.18264840613941974   222991.96906524897  0.27009565
  # wind_full_13831                                                  0.3081771998950197    0.3468846013817595    112340.71190792322  0.4460996
  # wind_full_13831                                                  0.5214057546400773    0.584787968213315     66396.6377287507    1.0
  # hail_full_13831                                                  0.001802767135236088  0.002335008691200905  8.702007792500556e6 0.066225864
  # hail_full_13831                                                  0.08223740886956366   0.10230366652032856   190762.23891943693  0.15690458
  # hail_full_13831                                                  0.17731425714857116   0.2129139200988159    88476.50530314445   0.29827937
  # hail_full_13831                                                  0.3574164599806298    0.4218812134258124    43885.31184053421   1.0

  println("model_name_to_day_bins = $model_name_to_day_bins")
  # model_name_to_day_bins = Dict{String, Vector{Float32}}("wind_mean_prob_computed_climatology_blurs_910" => [0.115368165, 0.25932494, 0.43146545, 1.0], "hail_mean_prob_computed_climatology_blurs_910" => [0.06547251, 0.15664904, 0.29551116, 1.0], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.115368165, 0.25932494, 0.43146545, 1.0], "tornado_mean_prob_computed_climatology_blurs_910" => [0.02166239, 0.06790343, 0.16562855, 1.0], "tornado_full_13831" => [0.021043906, 0.074019335, 0.17095083, 1.0], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.06547251, 0.15664904, 0.29551116, 1.0], "wind_full_13831" => [0.12129908, 0.27009565, 0.4460996, 1.0], "hail_full_13831" => [0.066225864, 0.15690458, 0.29827937, 1.0], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.022665959, 0.06817255, 0.1642723, 1.0])


  model_name_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))

    ŷ1 = @view X[:, (prediction_i*2) - 1] # total prob acc
    ŷ2 = @view X[:, (prediction_i*2)]     # max prob acc
    y  = Ys[event_name]

    model_name_to_day_bins_logistic_coeffs[model_name] = find_two_predictor_logistic_coeffs(model_name, ŷ1, ŷ2, y, weights, model_name_to_day_bins[model_name])
  end

  # model_name                                                       bin ŷ1_min      ŷ1_max      count   pos_count weight     mean_ŷ1      mean_ŷ2       mean_y       ŷ1_logloss   ŷ2_logloss   ŷ1_au_pr    ŷ2_au_pr    mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
  # tornado_mean_prob_computed_climatology_blurs_910                 1-2 -1.0        0.06790343  9712335 10378.0   8.949703e6 0.0012467225 0.00032986148 0.0010804458 0.005826766  0.006429697  0.033406913 0.031298634 0.0010804457    0.0058139195     0.033594888    Float32[0.98352575, 0.03981221,    0.010008344]
  # tornado_mean_prob_computed_climatology_blurs_910                 2-3 0.02166239  0.16562855  210172  10314.0   197008.36  0.055336993  0.015332372   0.04907914   0.18692976   0.21396728   0.097479895 0.08947249  0.049079135     0.18652107       0.098801054    Float32[0.9775675,  0.045042068,   -0.0055927183]
  # tornado_mean_prob_computed_climatology_blurs_910                 3-4 0.06790343  1.0         79681   10228.0   75429.06   0.15687333   0.043077637   0.12817124   0.37201178   0.4326398    0.22009405  0.2092169   0.12817124      0.3655498        0.22126897     Float32[0.57631963, 0.20392773,    -0.28303716]
  # wind_mean_prob_computed_climatology_blurs_910                    1-2 -1.0        0.25932494  9592628 74550.0   8.839894e6 0.009726573  0.0026224314  0.0078328345 0.028680576  0.032380696  0.14344735  0.13037716  0.0078328345    0.028444752      0.14349008     Float32[0.9183711,  0.11854008,    0.020747136]
  # wind_mean_prob_computed_climatology_blurs_910                    2-3 0.115368165 0.43146545  380567  74498.0   353865.4   0.22677125   0.06481917    0.19567314   0.47651067   0.5806634    0.29206303  0.26643395  0.19567311      0.47345075       0.29207438     Float32[1.0090979,  0.052492317,   -0.043933667]
  # wind_mean_prob_computed_climatology_blurs_910                    3-4 0.25932494  1.0         199388  74384.0   185238.03  0.42271823   0.12863548    0.3737915    0.635032     0.8430241    0.5310278   0.49922827  0.3737915       0.6285987        0.53222036     Float32[0.769292,   0.13937852,    -0.011131755]
  # hail_mean_prob_computed_climatology_blurs_910                    1-2 -1.0        0.15664904  9646362 33950.0   8.890706e6 0.0044677705 0.0011337367  0.003528977  0.015025933  0.016732877  0.08170354  0.0770017   0.0035289775    0.014898867      0.08250876     Float32[0.9237591,  0.13382848,    0.12147692]
  # hail_mean_prob_computed_climatology_blurs_910                    2-3 0.06547251  0.29551116  305507  33922.0   282128.44  0.1373833    0.036513306   0.11120856   0.340039     0.39533988   0.173863    0.16208322  0.11120856      0.3368729        0.17392835     Float32[0.9564363,  0.047680803,   -0.16887067]
  # hail_mean_prob_computed_climatology_blurs_910                    3-4 0.15664904  1.0         145654  33888.0   134425.48  0.27972367   0.08309209    0.23339224   0.5236517    0.6382745    0.36902794  0.34237817  0.23339224      0.5174033        0.3695708      Float32[0.98554933, -0.08279941,   -0.4769262]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 1-2 -1.0        0.06817255  9712464 10379.0   8.94977e6  0.0012591855 0.00033697707 0.001080447  0.005827387  0.006419156  0.034747098 0.030802712 0.0010804468    0.005813511      0.034724437    Float32[1.0136324,  -0.0008003565, -0.1033998]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 2-3 0.022665959 0.1642723   206311  10317.0   193458.38  0.05631266   0.015748546   0.049987208  0.1898066    0.21723233   0.10022544  0.08815341  0.049987223     0.18939486       0.101178944    Float32[1.0090433,  0.026480297,   0.0030092064]
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 3-4 0.06817255  1.0         79552   10227.0   75361.45   0.154326     0.042071007   0.12828512   0.37210512   0.4348556    0.21873944  0.20629326  0.12828512      0.36616552       0.21977983     Float32[0.5625795,  0.22864038,    -0.21183096]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    1-2 -1.0        0.25932494  9592628 74550.0   8.839894e6 0.009726573  0.0026224314  0.0078328345 0.028680576  0.032380696  0.14344735  0.13037716  0.0078328345    0.028444752      0.14349008     Float32[0.9183711,  0.11854008,    0.020747136]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    2-3 0.115368165 0.43146545  380567  74498.0   353865.4   0.22677125   0.06481917    0.19567314   0.47651067   0.5806634    0.29206303  0.26643395  0.19567311      0.47345075       0.29207438     Float32[1.0090979,  0.052492317,   -0.043933667]
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    3-4 0.25932494  1.0         199388  74384.0   185238.03  0.42271823   0.12863548    0.3737915    0.635032     0.8430241    0.5310278   0.49922827  0.3737915       0.6285987        0.53222036     Float32[0.769292,   0.13937852,    -0.011131755]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    1-2 -1.0        0.15664904  9646362 33950.0   8.890706e6 0.0044677705 0.0011337367  0.003528977  0.015025933  0.016732877  0.08170354  0.0770017   0.0035289775    0.014898867      0.08250876     Float32[0.9237591,  0.13382848,    0.12147692]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    2-3 0.06547251  0.29551116  305507  33922.0   282128.44  0.1373833    0.036513306   0.11120856   0.340039     0.39533988   0.173863    0.16208322  0.11120856      0.3368729        0.17392835     Float32[0.9564363,  0.047680803,   -0.16887067]
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    3-4 0.15664904  1.0         145654  33888.0   134425.48  0.27972367   0.08309209    0.23339224   0.5236517    0.6382745    0.36902794  0.34237817  0.23339224      0.5174033        0.3695708      Float32[0.98554933, -0.08279941,   -0.4769262]
  # tornado_full_13831                                               1-2 -1.0        0.074019335 9718800 10390.0   8.955598e6 0.0012768853 0.00033077784 0.0010797478 0.0058559645 0.0064523434 0.03274472  0.028818477 0.0010797479    0.005839293      0.04356679     Float32[0.95958227, 0.04161413,    -0.10651286]
  # tornado_full_13831                                               2-3 0.021043906 0.17095083  211661  10323.0   198693.03  0.055734657  0.015497775   0.048663635  0.1845328    0.21181716   0.10162046  0.089558855 0.048663624     0.1838949        0.10141609     Float32[1.2272763,  -0.15624464,   -0.18067063]
  # tornado_full_13831                                               3-4 0.074019335 1.0         73216   10216.0   69533.97   0.1656988    0.04697025    0.13903588   0.39089903   0.45915884   0.2386106   0.23309015  0.13903588      0.38513196       0.24188775     Float32[0.5964124,  0.17200926,    -0.3083448]
  # wind_full_13831                                                  1-2 -1.0        0.27009565  9599552 74517.0   8.846394e6 0.009704876  0.0025361686  0.007827075  0.02845159   0.032430086  0.15157403  0.13541351  0.007827077     0.028224675      0.15166616     Float32[1.0433985,  -0.0069722184, -0.16421609]
  # wind_full_13831                                                  2-3 0.12129908  0.4460996   360669  74512.0   335332.7   0.23766962   0.06728358    0.206486     0.490576     0.6066902    0.3071339   0.27248484  0.20648599      0.4875199        0.30803534     Float32[1.1460801,  -0.10947033,   -0.3195157]
  # wind_full_13831                                                  3-4 0.27009565  1.0         192464  74417.0   178737.36  0.43525997   0.13793935    0.38738647   0.63694006   0.8510782    0.55482537  0.5173215   0.38738647      0.63157266       0.55479926     Float32[0.9224374,  -0.005079186,  -0.24423301]
  # hail_full_13831                                                  1-2 -1.0        0.15690458  9648557 33931.0   8.89277e6  0.004479475  0.0010905664  0.0035282015 0.015020049  0.01685818   0.08252732  0.07416679  0.0035282015    0.014897133      0.08661193     Float32[1.0235776,  0.02489767,    -0.070147164]
  # hail_full_13831                                                  2-3 0.066225864 0.29827937  302339  33919.0   279238.75  0.1373504    0.03518375    0.11236241   0.34156662   0.40259892   0.17611699  0.16087049  0.11236241      0.33865124       0.17745832     Float32[1.1081746,  -0.088940814,  -0.34076628]
  # hail_full_13831                                                  3-4 0.15690458  1.0         143459  33907.0   132361.81  0.2821982    0.082033046   0.23702818   0.5251296    0.6478943    0.38524213  0.35740376  0.23702817      0.51884675       0.38724697     Float32[1.1614795,  -0.24988303,   -0.7304756]

  println("model_name_to_day_bins_logistic_coeffs = $model_name_to_day_bins_logistic_coeffs")
  # model_name_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [[0.9183711, 0.11854008, 0.020747136], [1.0090979, 0.052492317, -0.043933667], [0.769292, 0.13937852, -0.011131755]], "hail_mean_prob_computed_climatology_blurs_910" => [[0.9237591, 0.13382848, 0.12147692], [0.9564363, 0.047680803, -0.16887067], [0.98554933, -0.08279941, -0.4769262]], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [[0.9183711, 0.11854008, 0.020747136], [1.0090979, 0.052492317, -0.043933667], [0.769292, 0.13937852, -0.011131755]], "tornado_mean_prob_computed_climatology_blurs_910" => [[0.98352575, 0.03981221, 0.010008344], [0.9775675, 0.045042068, -0.0055927183], [0.57631963, 0.20392773, -0.28303716]], "tornado_full_13831" => [[0.95958227, 0.04161413, -0.10651286], [1.2272763, -0.15624464, -0.18067063], [0.5964124, 0.17200926, -0.3083448]], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [[0.9237591, 0.13382848, 0.12147692], [0.9564363, 0.047680803, -0.16887067], [0.98554933, -0.08279941, -0.4769262]], "wind_full_13831" => [[1.0433985, -0.0069722184, -0.16421609], [1.1460801, -0.10947033, -0.3195157], [0.9224374, -0.005079186, -0.24423301]], "hail_full_13831" => [[1.0235776, 0.02489767, -0.070147164], [1.1081746, -0.088940814, -0.34076628], [1.1614795, -0.24988303, -0.7304756]], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0136324, -0.0008003565, -0.1033998], [1.0090433, 0.026480297, 0.0030092064], [0.5625795, 0.22864038, -0.21183096]])


  validation_day_forecasts_0z_12z = PredictionForecasts.period_forecasts_from_accumulators(validation_day_acc_forecasts_0z_12z, model_name_to_day_bins, model_name_to_day_bins_logistic_coeffs, models; module_name = "daily", period_name = "day")

  X, Ys, weights = nothing, nothing, nothing # free
  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_day_forecasts_0z_12z;
      event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
      save_dir = "validation_day_forecasts_0z_12z",
    );


  println("\nChecking daily performance...")

  inspect_predictive_power(validation_day_forecasts_0z_12z, X, Ys, weights, model_names, event_names)
  # tornado_mean_prob_computed_climatology_blurs_910                 (20606.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.14475623
  # wind_mean_prob_computed_climatology_blurs_910                    (148934.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.3863571
  # hail_mean_prob_computed_climatology_blurs_910                    (67838.0)  feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.2568212
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 (20606.0)  feature 4 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.14425442
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    (148934.0) feature 5 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.3863571
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    (67838.0)  feature 6 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.2568212
  # tornado_full_13831                                               (20606.0)  feature 7 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.1562636
  # wind_full_13831                                                  (148934.0) feature 8 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.40261915
  # hail_full_13831                                                  (67838.0)  feature 9 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.26647338


  println("\nPlotting daily accumulators reliability...")

  Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)
  # ŷ_tornado_mean_prob_computed_climatology_blurs_910,y_tornado_mean_prob_computed_climatology_blurs_910,ŷ_wind_mean_prob_computed_climatology_blurs_910,y_wind_mean_prob_computed_climatology_blurs_910,ŷ_hail_mean_prob_computed_climatology_blurs_910,y_hail_mean_prob_computed_climatology_blurs_910,ŷ_tornado_mean_prob_computed_climatology_blurs_910_before_20200523,y_tornado_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_wind_mean_prob_computed_climatology_blurs_910_before_20200523,y_wind_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_hail_mean_prob_computed_climatology_blurs_910_before_20200523,y_hail_mean_prob_computed_climatology_blurs_910_before_20200523,ŷ_tornado_full_13831,y_tornado_full_13831,ŷ_wind_full_13831,y_wind_full_13831,ŷ_hail_full_13831,y_hail_full_13831,
  # 0.0001134045,0.00011855099,0.0008761321,0.0008702636,0.00039552778,0.0003828304,0.000112993745,0.00011862761,0.0008761321,0.0008702636,0.00039552778,0.0003828304,0.00011889208,0.00011867229,0.00081215386,0.00087095215,0.00037449508,0.0003831146,
  # 0.0026995738,0.002883395,0.023414072,0.024103547,0.012724508,0.014131579,0.002721142,0.0028592178,0.023414072,0.024103547,0.012724508,0.014131579,0.0025425232,0.0029419851,0.022360653,0.02254095,0.01282629,0.013905334,
  # 0.006611966,0.0055141146,0.041945223,0.042848013,0.023101129,0.024997005,0.006507881,0.0058532814,0.041945223,0.042848013,0.023101129,0.024997005,0.006280865,0.0051744217,0.042070255,0.041827917,0.023605391,0.024025073,
  # 0.011585749,0.011448514,0.06127483,0.0636582,0.03380317,0.037933312,0.011275668,0.01070865,0.06127483,0.0636582,0.03380317,0.037933312,0.011003303,0.011740553,0.063111745,0.060791567,0.034828246,0.03586223,
  # 0.016415864,0.017854238,0.08243498,0.07848401,0.0457986,0.043418717,0.016579872,0.015920268,0.08243498,0.07848401,0.0457986,0.043418717,0.015415692,0.017903777,0.08631787,0.080632016,0.046554014,0.046197318,
  # 0.02187849,0.021140657,0.105708726,0.10272915,0.05993039,0.054887608,0.022395136,0.021892559,0.105708726,0.10272915,0.05993039,0.054887608,0.020695614,0.020229997,0.111274436,0.10614912,0.05947211,0.05650269,
  # 0.028676976,0.02642947,0.12956977,0.12866372,0.07432455,0.075061284,0.028867615,0.029448286,0.12956977,0.12866372,0.07432455,0.075061284,0.02752036,0.026117474,0.13666223,0.13795479,0.07378093,0.069163494,
  # 0.036338367,0.04005221,0.15448336,0.15298569,0.08859911,0.08664865,0.03655914,0.035528123,0.15448336,0.15298569,0.08859911,0.08664865,0.035524398,0.03829976,0.1631872,0.1644673,0.08903005,0.08741153,
  # 0.0446643,0.0490739,0.18144765,0.178357,0.10365863,0.10203979,0.04549802,0.048259795,0.18144765,0.178357,0.10365863,0.10203979,0.045094907,0.04435019,0.19095488,0.19477257,0.10437473,0.10729696,
  # 0.05459126,0.05435662,0.20998073,0.21694721,0.11900043,0.12383108,0.055135105,0.060363423,0.20998073,0.21694721,0.11900043,0.12383108,0.057168424,0.053338945,0.21987148,0.23189346,0.12011737,0.1251812,
  # 0.06740698,0.06307182,0.2395083,0.24798724,0.13616683,0.14006455,0.067434624,0.06360918,0.2395083,0.24798724,0.13616683,0.14006455,0.07256332,0.06717904,0.25061944,0.25177547,0.13763468,0.14219716,
  # 0.08287006,0.075163364,0.27001777,0.2739428,0.15607761,0.1608302,0.08266256,0.07634118,0.27001777,0.2739428,0.15607761,0.1608302,0.08858901,0.09274736,0.2826716,0.2776592,0.15773332,0.1630101,
  # 0.099365845,0.0924961,0.30011153,0.3108468,0.17766756,0.17745508,0.0992417,0.08907371,0.30011153,0.3108468,0.17766756,0.17745508,0.10420568,0.10806544,0.31423613,0.3123641,0.17927237,0.18475817,
  # 0.11478873,0.13078278,0.33094203,0.32831654,0.20186988,0.19125886,0.116243474,0.11396742,0.33094203,0.32831654,0.20186988,0.19125886,0.12102106,0.116944864,0.34522346,0.35031825,0.20370498,0.198593,
  # 0.1296558,0.13811475,0.36258224,0.3563359,0.22880352,0.2222693,0.1321703,0.15324041,0.36258224,0.3563359,0.22880352,0.2222693,0.13717742,0.14863971,0.37661222,0.38453656,0.23179689,0.21961693,
  # 0.14758159,0.16531236,0.39881828,0.38213938,0.25981605,0.25923675,0.14900757,0.1764327,0.39881828,0.38213938,0.25981605,0.25923675,0.15673207,0.17233202,0.41344002,0.40402344,0.26593435,0.2570302,
  # 0.1705173,0.1919997,0.44386277,0.43772304,0.29598224,0.32808515,0.16982228,0.19420336,0.44386277,0.43772304,0.29598224,0.32808515,0.1803576,0.19112791,0.45980436,0.46903685,0.30709186,0.30743837,
  # 0.20010602,0.21687044,0.49787465,0.5216293,0.34058133,0.36725804,0.19628426,0.2260117,0.49787465,0.5216293,0.34058133,0.36725804,0.21512035,0.2046197,0.5173739,0.5208262,0.35592264,0.378745,
  # 0.24316965,0.27065727,0.5714386,0.5558848,0.40370944,0.4121197,0.23663928,0.25642827,0.5714386,0.5558848,0.40370944,0.4121197,0.27166343,0.3318058,0.5973254,0.5805249,0.42256165,0.42708427,
  # 0.35289577,0.29500008,0.7103429,0.72298694,0.5375405,0.5013366,0.34944484,0.29213113,0.7103429,0.72298694,0.5375405,0.5013366,0.3821813,0.3331822,0.74489653,0.74609774,0.5678906,0.5656451,


  println("Determining SPC-like calibration curve...")

  println("model_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
  spc_calibrations = Dict{String,Vector{Tuple{Float32,Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]
    spc_calibrations[model_name] = spc_calibrate_warning_ratio(event_name, model_name, ŷ, y, weights)
  end

  # model_name                                                       nominal_prob threshold_to_match_warning_ratio SR          POD           WR
  # tornado_mean_prob_computed_climatology_blurs_910                 0.02         0.01799202                       0.06419382  0.75942755    0.025347784
  # tornado_mean_prob_computed_climatology_blurs_910                 0.05         0.06854057                       0.1377441   0.47169042    0.0073372093
  # tornado_mean_prob_computed_climatology_blurs_910                 0.1          0.17239952                       0.24644762  0.1705114     0.0014824349
  # tornado_mean_prob_computed_climatology_blurs_910                 0.15         0.29992485                       0.29077744  0.034393243   0.00025343074
  # tornado_mean_prob_computed_climatology_blurs_910                 0.3          0.46686745                       0.32967794  0.0057634856  3.7457794e-5
  # tornado_mean_prob_computed_climatology_blurs_910                 0.45         0.6732578                        0.4334652   0.00063932553 3.1602005e-6
  # wind_mean_prob_computed_climatology_blurs_910                    0.05         0.04979515                       0.18666871  0.85642475    0.070397444
  # wind_mean_prob_computed_climatology_blurs_910                    0.15         0.21516609                       0.36619323  0.5162885     0.02163323
  # wind_mean_prob_computed_climatology_blurs_910                    0.3          0.4797535                        0.595859    0.14096119    0.0036299052
  # wind_mean_prob_computed_climatology_blurs_910                    0.45         0.70975685                       0.78855383  0.021391029   0.00041623594
  # hail_mean_prob_computed_climatology_blurs_910                    0.05         0.02939415                       0.111877434 0.8469571     0.05263467
  # hail_mean_prob_computed_climatology_blurs_910                    0.15         0.12362099                       0.2301538   0.5104074     0.015418843
  # hail_mean_prob_computed_climatology_blurs_910                    0.3          0.367239                         0.45130342  0.1009303     0.0015549124
  # hail_mean_prob_computed_climatology_blurs_910                    0.45         0.6290493                        0.6397271   0.008234327   8.9492445e-5
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.02         0.018140793                      0.06446767  0.762628      0.02534648
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.05         0.06867409                       0.13740186  0.47050112    0.0073369388
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.1          0.17149925                       0.24736677  0.17116243    0.0014825655
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.15         0.29029655                       0.27952138  0.03306936    0.00025348816
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.3          0.47507286                       0.34711012  0.0060625267  3.7422542e-5
  # tornado_mean_prob_computed_climatology_blurs_910_before_20200523 0.45         0.70194054                       0.4835026   0.00074192084 3.2878017e-6
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.05         0.04979515                       0.18666871  0.85642475    0.070397444
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.15         0.21516609                       0.36619323  0.5162885     0.02163323
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.3          0.4797535                        0.595859    0.14096119    0.0036299052
  # wind_mean_prob_computed_climatology_blurs_910_before_20200523    0.45         0.70975685                       0.78855383  0.021391029   0.00041623594
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.05         0.02939415                       0.111877434 0.8469571     0.05263467
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.15         0.12362099                       0.2301538   0.5104074     0.015418843
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.3          0.367239                         0.45130342  0.1009303     0.0015549124
  # hail_mean_prob_computed_climatology_blurs_910_before_20200523    0.45         0.6290493                        0.6397271   0.008234327   8.9492445e-5
  # tornado_full_13831                                               0.02         0.016950607                      0.064026326 0.7574176     0.025346832
  # tornado_full_13831                                               0.05         0.06830406                       0.14281912  0.48904803    0.0073368903
  # tornado_full_13831                                               0.1          0.17817497                       0.257954    0.17845055    0.0014822535
  # tornado_full_13831                                               0.15         0.3255825                        0.32377487  0.03830062    0.00025346005
  # tornado_full_13831                                               0.3          0.4591999                        0.48384824  0.008458492   3.7456797e-5
  # tornado_full_13831                                               0.45         0.60011864                       0.49274874  0.00073567254 3.1989384e-6
  # wind_full_13831                                                  0.05         0.047945023                      0.18788931  0.8620227     0.07039727
  # wind_full_13831                                                  0.15         0.21813774                       0.37468866  0.52826184    0.021633059
  # wind_full_13831                                                  0.3          0.49361992                       0.6080255   0.14383546    0.0036298062
  # wind_full_13831                                                  0.45         0.7422848                        0.8450049   0.022920607   0.00041620387
  # hail_full_13831                                                  0.05         0.029951096                      0.112065494 0.8483541     0.052633014
  # hail_full_13831                                                  0.15         0.123464584                      0.2321128   0.5147346     0.015418328
  # hail_full_13831                                                  0.3          0.3763752                        0.47983125  0.10731995    0.0015550521
  # hail_full_13831                                                  0.45         0.6608143                        0.7090504   0.00912425    8.9469104e-5

  println("spc_calibrations = $spc_calibrations")
  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [(0.05, 0.04979515), (0.15, 0.21516609), (0.3, 0.4797535), (0.45, 0.70975685)], "hail_mean_prob_computed_climatology_blurs_910" => [(0.05, 0.02939415), (0.15, 0.12362099), (0.3, 0.367239), (0.45, 0.6290493)], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.05, 0.04979515), (0.15, 0.21516609), (0.3, 0.4797535), (0.45, 0.70975685)], "tornado_mean_prob_computed_climatology_blurs_910" => [(0.02, 0.01799202), (0.05, 0.06854057), (0.1, 0.17239952), (0.15, 0.29992485), (0.3, 0.46686745), (0.45, 0.6732578)], "tornado_full_13831" => [(0.02, 0.016950607), (0.05, 0.06830406), (0.1, 0.17817497), (0.15, 0.3255825), (0.3, 0.4591999), (0.45, 0.60011864)], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.05, 0.02939415), (0.15, 0.12362099), (0.3, 0.367239), (0.45, 0.6290493)], "wind_full_13831" => [(0.05, 0.047945023), (0.15, 0.21813774), (0.3, 0.49361992), (0.45, 0.7422848)], "hail_full_13831" => [(0.05, 0.029951096), (0.15, 0.123464584), (0.3, 0.3763752), (0.45, 0.6608143)], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.02, 0.018140793), (0.05, 0.06867409), (0.1, 0.17149925), (0.15, 0.29029655), (0.3, 0.47507286), (0.45, 0.70194054)])

  println("Now copy-paste the appropriate bits from the above into the Prediction.jl forecast files")

  ()
end

model_names = first.(HREFPredictionAblations2.models)
event_names = map(name -> split(name, "_")[1], model_names)
println("model_names = $model_names")
println("event_names = $event_names")

do_it_all(
  HREFPredictionAblations2.forecasts(),
  2:35,
  model_names,
  event_names,
  HREFPredictionAblations2.make_calibrated_hourly_models
)
