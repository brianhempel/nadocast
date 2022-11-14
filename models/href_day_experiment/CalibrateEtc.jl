import Dates
import Serialization

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics
import PredictionForecasts

push!(LOAD_PATH, @__DIR__)
import HREFDayExperiment

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


# rm("validation_forecasts_with_blurs"; recursive=true)
# rm("validation_forecasts_blurred"; recursive=true)
# rm("validation_day_forecasts_0z_12z_calibrated"; recursive=true)
function do_it_all(forecasts, model_names, event_names, make_calibrated_hourly_models)
  @assert length(model_names) == length(event_names)
  nmodels = length(model_names)

  forecasts = filter(f -> f.forecast_hour in [23, 35], forecasts)

  _, validation_forecasts, _ = TrainingShared.forecasts_train_validation_test(forecasts; just_hours_near_storm_events = false)

  # for testing
  if get(ENV, "ONLY_N_FORECASTS", "") != ""
    validation_forecasts = validation_forecasts[1:parse(Int64, ENV["ONLY_N_FORECASTS"])]
  end

  valid_times = Forecasts.valid_utc_datetime.(validation_forecasts)
  println("$(length(validation_forecasts)) validation forecasts from $(minimum(valid_times)) to $(maximum(valid_times))")

  grid = validation_forecasts[1].grid

  blurrers = [
    (0,   Grids.radius_grid_is(grid, 0.0)),
    (15,  Grids.radius_grid_is(grid, 15.0)),
    (25,  Grids.radius_grid_is(grid, 25.0)),
    (35,  Grids.radius_grid_is(grid, 35.0)),
    (50,  Grids.radius_grid_is(grid, 50.0)),
    (70,  Grids.radius_grid_is(grid, 70.0)),
    (100, Grids.radius_grid_is(grid, 100.0)),
  ]
  blur_radii = [15, 25, 35, 50, 70, 100]

  validation_forecasts_with_blurs = PredictionForecasts.with_blurs_and_forecast_hour(validation_forecasts, blur_radii)

  X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_with_blurs; event_name_to_labeler = TrainingShared.event_name_to_day_labeler, save_dir = "validation_forecasts_with_blurs");

  println("0Z 12Z pooled performance")
  inspect_predictive_power(validation_forecasts_with_blurs, (@view X[:,1:end-1]), Ys, weights, map(i -> model_names[i ÷ length(blurrers) + 1], 0:size(X,2)-2), map(i -> event_names[i ÷ length(blurrers) + 1], 0:size(X,2)-2))

  run_times = Serialization.deserialize("validation_forecasts_with_blurs/run_times.serialized")

  is_0z  = findall(t -> Dates.hour(t) == 0,  run_times)
  is_12z = findall(t -> Dates.hour(t) == 12, run_times)

  run_times = nothing # free

  best_blur_radius_0z  = map(_ -> 0,   model_names)
  best_blur_au_pr_0z   = map(_ -> 0f0, model_names)
  best_blur_radius_12z = map(_ -> 0,   model_names)
  best_blur_au_pr_12z  = map(_ -> 0f0, model_names)
  nradii = length([0; blur_radii])
  println("feature_i model_name run_time radius_mi au_pr")
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    for (radius_i, radius_mi) in enumerate([0; blur_radii])
      col_i       = (prediction_i - 1) * nradii + radius_i
      y_0z        = view(Ys[event_name], is_0z)
      x_0z        = @view X[is_0z, col_i]
      weights_0z  = @view weights[is_0z]
      au_pr_0z    = Metrics.area_under_pr_curve(x_0z, y_0z, weights_0z)
      println("$col_i $model_name 0Z $radius_mi $au_pr_0z")
      if au_pr_0z > best_blur_au_pr_0z[prediction_i]
        best_blur_au_pr_0z[prediction_i]  = au_pr_0z
        best_blur_radius_0z[prediction_i] = radius_mi
      end
      y_12z       = view(Ys[event_name], is_12z)
      x_12z       = @view X[is_12z, col_i]
      weights_12z = @view weights[is_12z]
      au_pr_12z   = Metrics.area_under_pr_curve(x_12z, y_12z, weights_12z)
      if au_pr_12z > best_blur_au_pr_12z[prediction_i]
        best_blur_au_pr_12z[prediction_i]  = au_pr_12z
        best_blur_radius_12z[prediction_i] = radius_mi
      end
      println("$col_i $model_name 12Z $radius_mi $au_pr_12z")
    end
  end

  println("best_blur_radius_0z  = $best_blur_radius_0z")
  println("best_blur_radius_12z = $best_blur_radius_12z")

  # Needs to be the same order as models
  blur_grid_is = map(enumerate(model_names)) do (prediction_i, _)
    burrer_0z_i  = findfirst(blurrer -> blurrer[1] == best_blur_radius_0z[prediction_i],  blurrers)
    burrer_12z_i = findfirst(blurrer -> blurrer[1] == best_blur_radius_12z[prediction_i], blurrers)
    _, blur_0z_grid_is  = blurrers[burrer_0z_i]
    _, blur_12z_grid_is = blurrers[burrer_12z_i]

    (blur_12z_grid_is, blur_0z_grid_is)
  end

  validation_forecasts_blurred = PredictionForecasts.blurred(validation_forecasts, 23:35, blur_grid_is)

  # Make sure a forecast loads
  @assert nmodels <= size(Forecasts.data(validation_forecasts_blurred[1]), 2)

  println("\nLoading blurred daily forecasts...")

  X_blurs = X
  X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = TrainingShared.event_name_to_day_labeler, save_dir = "validation_forecasts_blurred");

  for (prediction_i, _) in enumerate(model_names)
    burrer_0z_i  = findfirst(blurrer -> blurrer[1] == best_blur_radius_0z[prediction_i],  blurrers)
    burrer_12z_i = findfirst(blurrer -> blurrer[1] == best_blur_radius_12z[prediction_i], blurrers)

    blurs_col_0Z_i  = (prediction_i - 1) * nradii + burrer_0z_i
    blurs_col_12Z_i = (prediction_i - 1) * nradii + burrer_12z_i

    println("Blurred $prediction_i should match blurs $blurs_col_0Z_i for 0Z")
    @assert X_blurs[is_0z, blurs_col_0Z_i]   = X[is_0z, prediction_i]
    println("Blurred $prediction_i should match blurs $blurs_col_12Z_i for 12Z")
    @assert X_blurs[is_12z, blurs_col_12Z_i] = X[is_12z, prediction_i]
  end

  println("\nChecking the daily blurred forecast performance...")

  inspect_predictive_power(validation_forecasts_blurred, X, Ys, weights, model_names, event_names)


  bin_count = 4
  println("\nFinding $bin_count bins of equal positive weight, to bin the probs for calibrating dailies....")

  model_name_to_day_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,prediction_i]
    y = Ys[event_name]
    model_name_to_day_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end
  println("model_name_to_day_bins = $model_name_to_day_bins")


  model_name_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,prediction_i]
    y = Ys[event_name]
    model_name_to_day_bins_logistic_coeffs[model_name] = find_single_predictor_logistic_coeffs(model_name, ŷ, y, weights, model_name_to_day_bins[model_name])
  end
  println("model_name_to_day_bins_logistic_coeffs = $model_name_to_day_bins_logistic_coeffs")

  calib_day_models = HREFDayExperiment.make_calibrated_models(model_name_to_day_bins, model_name_to_day_bins_logistic_coeffs)

  validation_day_forecasts_0z_12z_calibrated = PredictionForecasts.simple_prediction_forecasts(validation_forecasts_blurred, calib_day_models; model_name = "HREFDayExperiment_calibrated")

  X, Ys, weights = nothing, nothing, nothing # free
  X, Ys, weights =
    TrainingShared.get_data_labels_weights(
      validation_day_forecasts_0z_12z_calibrated;
      event_name_to_labeler = TrainingShared.event_name_to_day_labeler,
      save_dir = "validation_day_forecasts_0z_12z_calibrated",
    );


  println("\nChecking calibrated performance...")

  inspect_predictive_power(validation_day_forecasts_0z_12z_calibrated, X, Ys, weights, model_names, event_names)

  println("\nPlotting calibrated reliability...")

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

model_names = first.(HREFDayExperiment.models)
event_names = model_names
println("model_names = $model_names")
println("event_names = $event_names")

do_it_all(
  HREFDayExperiment.uncalibrated_day_prediction_forecasts(),
  model_names,
  event_names,
  (_ -> ())
)
