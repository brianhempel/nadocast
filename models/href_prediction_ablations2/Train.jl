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

    bin_href_x  = ŷ[bin_members]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    bin_X_features = Array{Float32}(undef, (length(bin_y), 1))

    Threads.@threads :static for i in 1:length(bin_y)
      logit_href = logit(bin_href_x[i])

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
      ("mean_input_ŷ", sum(bin_href_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("input_logloss", sum(logloss.(bin_y, bin_href_x) .* bin_weights) / bin_weight),
      ("input_au_pr", Metrics.area_under_pr_curve_fast(bin_href_x, bin_y, bin_weights)),
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

    bin_ŷ1  = ŷ1[bin_members, prediction_i*2 - 1]
    bin_ŷ2  = ŷ2[bin_members, prediction_i*2]
    # bin_ŷ       = ŷ[bin_members]
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
      ("ŷ1_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_ŷ1, bin_y, bin_weights))),
      ("ŷ2_au_pr", Float32(Metrics.area_under_pr_curve_fast(bin_ŷ2, bin_y, bin_weights))),
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

  bin_count = 6
  println("\nFinding $bin_count bins of equal positive weight, for calibrating the hourlies....")

  model_name_to_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]
    model_name_to_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end
  println("model_name_to_bins = $model_name_to_bins")

  println("\nFinding logistic coefficients for the $(bin_count-1) overlapping bin pairs, for calibrating the hourlies....")

  model_name_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]

    model_name_to_bins_logistic_coeffs[model_name] = find_single_predictor_logistic_coeffs(model_name, ŷ, y, weights, model_name_to_bins[model_name])
  end
  println("model_name_to_bins_logistic_coeffs = $model_name_to_bins_logistic_coeffs")

  println("\nCopy model_name_to_bins and model_name_to_bins_logistic_coeffs into the prediction model.")
  # println("\nYou can stop the process now if you want.")

  println("\nChecking the calibration...")

  hour_models = make_calibrated_hourly_models(model_name_to_bins, model_name_to_bins_logistic_coeffs)
  validation_forecasts_calibrated = PredictionForecasts.simple_prediction_forecasts(validation_forecasts_blurred, hour_models)

  X, Ys, weights = nothing, nothing, nothing # free
  X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_calibrated; event_name_to_labeler = TrainingShared.event_name_to_labeler, save_dir = "validation_forecasts_calibrated");

  println("\nChecking the hourly calibrated blurred forecast performance (should be the same if calibration was monotonic)...")
  inspect_predictive_power(validation_forecasts_calibrated, X, Ys, weights, model_names, event_names)

  println("\nPlotting reliability...")
  Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)

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
  acc_model_names = map(i -> model_names[div(i - 1, 2) + 1] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2))
  acc_event_names = vcat(map(name -> [name, name], event_names)...)

  println("Drawing Dec 11 to check...")

  dec11 = filter(f -> Forecasts.time_title(f) == "2021-12-11 00Z +35", validation_day_acc_forecasts_0z_12z)[1];
  dec11_data = Forecasts.data(dec11);
  for i in 1:size(dec11_data,2)
    model_name = acc_model_names[i]
    PlotMap.plot_debug_map("dec11_0z_12z_day_accs_$(i)_$model_name", dec11.grid, dec11_data[:,i]);
  end
  for event_name in unique(acc_event_names)
    labeler = TraingShared.event_name_to_day_labeler[event_name]
    PlotMap.plot_debug_map("dec11_0z_12z_day_$event_name", dec11.grid, labeler(dec11));
  end

  # scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_accs_1.pdf ./
  # scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_tornado.pdf ./

  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_day_acc_forecasts_0z_12z;
      event_name_to_labeler = TraingShared.event_name_to_day_labeler,
      save_dir = "validation_day_accumulators_forecasts_0z_12z",
    );


  println("\nChecking daily accumulator performance...")

  inspect_predictive_power(validation_day_acc_forecasts_0z_12z, X, Ys, weights, acc_model_names, acc_event_names)

  println("\nPlotting daily accumulators reliability...")

  Metrics.reliability_curves_midpoints(20, X, Ys, acc_event_names, weights, acc_model_names)


  bin_count = 4
  println("\nFinding $bin_count bins of equal positive weight, to bin the total_prob accs for calibrating the dailies....")

  model_name_to_day_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc
    y = Ys[event_name]
    model_name_to_day_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end
  println("model_name_to_day_bins = $model_name_to_day_bins")


  model_name_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))

    ŷ1 = @view X[:, (prediction_i*2) - 1] # total prob acc
    ŷ2 = @view X[:, (prediction_i*2)]     # max prob acc
    y  = Ys[event_name]

    model_name_to_day_bins_logistic_coeffs[model_name] = find_two_predictor_logistic_coeffs(model_name, ŷ1, ŷ2, y, weights, model_name_to_day_bins[model_name])
  end
  println("model_name_to_day_bins_logistic_coeffs = $model_name_to_day_bins_logistic_coeffs")

  validation_day_forecasts_0z_12z = PredictionForecasts.period_forecasts_from_accumulators(validation_day_acc_forecasts_0z_12z, model_name_to_day_bins, model_name_to_day_bins_logistic_coeffs, models; module_name = "daily", period_name = "day")

  X, Ys, weights = nothing, nothing, nothing # free
  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_day_forecasts_0z_12z;
      event_name_to_labeler = event_name_to_day_labeler,
      save_dir = "validation_day_forecasts_0z_12z",
    );


  println("\nChecking daily performance...")

  inspect_predictive_power(validation_day_forecasts_0z_12z, X, Ys, weights, model_names, event_names)

  println("\nPlotting daily accumulators reliability...")

  Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)

  println("Determining SPC-like calibration curve...")

  println("model_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
  spc_calibrations = Dict{String,Vector{Tuple{Float32,Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]
    spc_calibrations[model_name] = spc_calibrate_warning_ratio(event_name, model_name, ŷ, y, weights)
  end
  println("spc_calibrations = $spc_calibrations")

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
