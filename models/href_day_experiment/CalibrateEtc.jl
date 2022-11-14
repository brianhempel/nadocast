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
      ("input_au_pr", Metrics.area_under_pr_curve(bin_href_x, bin_y, bin_weights)),
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

  # feature_i model_name  run_time radius_mi au_pr
  # 1         tornado     0Z       0         0.12786487
  # 1         tornado     12Z      0         0.1544998
  # 2         tornado     0Z       15        0.1294705
  # 2         tornado     12Z      15        0.15537526
  # 3         tornado     0Z       25        0.13067819
  # 3         tornado     12Z      25        0.15519707
  # 4         tornado     0Z       35        0.13217057
  # 4         tornado     12Z      35        0.15416709
  # 5         tornado     0Z       50        0.1333855
  # 5         tornado     12Z      50        0.15124844
  # 6         tornado     0Z       70        0.13233377
  # 6         tornado     12Z      70        0.14585833
  # 7         tornado     0Z       100       0.12627704
  # 7         tornado     12Z      100       0.13463569
  # 8         wind        0Z       0         0.35702384
  # 8         wind        12Z      0         0.38482037
  # 9         wind        0Z       15        0.35877413
  # 9         wind        12Z      15        0.38711163
  # 10        wind        0Z       25        0.35916543
  # 10        wind        12Z      25        0.3875034
  # 11        wind        0Z       35        0.35859838
  # 11        wind        12Z      35        0.3863866
  # 12        wind        0Z       50        0.35683402
  # 12        wind        12Z      50        0.38295773
  # 13        wind        0Z       70        0.35336804
  # 13        wind        12Z      70        0.37633815
  # 14        wind        0Z       100       0.34662548
  # 14        wind        12Z      100       0.36276436
  # 15        hail        0Z       0         0.21235014
  # 15        hail        12Z      0         0.2620384
  # 16        hail        0Z       15        0.2131384
  # 16        hail        12Z      15        0.26411107
  # 17        hail        0Z       25        0.21267323
  # 17        hail        12Z      25        0.26393855
  # 18        hail        0Z       35        0.21038839
  # 18        hail        12Z      35        0.2612414
  # 19        hail        0Z       50        0.20564729
  # 19        hail        12Z      50        0.25439882
  # 20        hail        0Z       70        0.19790818
  # 20        hail        12Z      70        0.24344759
  # 21        hail        0Z       100       0.18567652
  # 21        hail        12Z      100       0.2217629
  # 22        sig_tornado 0Z       0         0.054310996
  # 22        sig_tornado 12Z      0         0.09374113
  # 23        sig_tornado 0Z       15        0.055438574
  # 23        sig_tornado 12Z      15        0.09401861
  # 24        sig_tornado 0Z       25        0.056356583
  # 24        sig_tornado 12Z      25        0.09252274
  # 25        sig_tornado 0Z       35        0.057655033
  # 25        sig_tornado 12Z      35        0.090497285
  # 26        sig_tornado 0Z       50        0.05928553
  # 26        sig_tornado 12Z      50        0.08868729
  # 27        sig_tornado 0Z       70        0.061797563
  # 27        sig_tornado 12Z      70        0.09670891
  # 28        sig_tornado 0Z       100       0.07501501
  # 28        sig_tornado 12Z      100       0.090316735

  println("best_blur_radius_0z  = $best_blur_radius_0z")
  println("best_blur_radius_12z = $best_blur_radius_12z")

  # best_blur_radius_0z  = [50, 25, 15, 100]
  # best_blur_radius_12z = [15, 25, 15, 70]


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
    @assert X_blurs[is_0z, blurs_col_0Z_i]   == X[is_0z, prediction_i]
    println("Blurred $prediction_i should match blurs $blurs_col_12Z_i for 12Z")
    @assert X_blurs[is_12z, blurs_col_12Z_i] == X[is_12z, prediction_i]
  end
  # Blurred 1 should match blurs 5 for 0Z
  # Blurred 1 should match blurs 2 for 12Z
  # Blurred 2 should match blurs 10 for 0Z
  # Blurred 2 should match blurs 10 for 12Z
  # Blurred 3 should match blurs 16 for 0Z
  # Blurred 3 should match blurs 16 for 12Z
  # Blurred 4 should match blurs 28 for 0Z
  # Blurred 4 should match blurs 27 for 12Z

  println("\nChecking the daily blurred forecast performance...")

  inspect_predictive_power(validation_forecasts_blurred, X, Ys, weights, model_names, event_names)
  # tornado     (20606.0)  feature 1 TORPROB:calculated:hour  fcst:calculated_prob:blurred AU-PR-curve: 0.14098175
  # wind        (148934.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.3729874
  # hail        (67838.0)  feature 3 HAILPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.2386961
  # sig_tornado (2681.0)   feature 4 STORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.077359244

  bin_count = 4
  println("\nFinding $bin_count bins of equal positive weight, to bin the probs for calibrating dailies....")

  # model_name  mean_y                mean_ŷ                Σweight             bin_max
  # tornado     0.0005500241804774725 0.0005369805288177626 8.790871373489559e6 0.017779902
  # tornado     0.03145649235376229   0.03161041028395532   153710.28380143642  0.056085516
  # tornado     0.08921990419281871   0.07882750970527863   54185.09851181507   0.11559871
  # tornado     0.18330057841333006   0.18273992318938512   26365.09276086092   1.0
  # wind        0.0040284560090752455 0.004039647483909972  8.594173634453237e6 0.08978425
  # wind        0.1453945610920795    0.13457977141449937   238114.38466775417  0.19696076
  # wind        0.28618142588908096   0.2562145123893205    120974.60549122095  0.335077
  # wind        0.4816993460701437    0.4501644666340919    71869.22395145893   1.0
  # hail        0.0018050343646954354 0.001640066169903731  8.691274674825251e6 0.048839394
  # hail        0.08309066550617629   0.07607029885958908   188799.26752007008  0.11616666
  # hail        0.1606428085651753    0.15851887998951678   97656.18107324839   0.22025906
  # hail        0.330909793222472     0.30596047094156564   47401.72514510155   1.0
  # sig_tornado 7.141125176418434e-5  8.10414766964188e-5   8.939867256696284e6 0.007377854
  # sig_tornado 0.010848778877185968  0.012951453213153974  58865.149409770966  0.022638915
  # sig_tornado 0.03244699148122466   0.03145710372269796   19661.17999511957   0.047667596
  # sig_tornado 0.0943811467809371    0.06957126364446617   6738.2624624967575  1.0

  model_name_to_day_bins = Dict{String,Vector{Float32}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,prediction_i]
    y = Ys[event_name]
    model_name_to_day_bins[model_name] = find_ŷ_bin_splits(model_name, ŷ, y, weights, bin_count)
  end
  println("model_name_to_day_bins = $model_name_to_day_bins")

  # model_name_to_day_bins = Dict{String, Vector{Float32}}("hail" => [0.048839394, 0.11616666, 0.22025906, 1.0], "tornado" => [0.017779902, 0.056085516, 0.11559871, 1.0], "sig_tornado" => [0.007377854, 0.022638915, 0.047667596, 1.0], "wind" => [0.08978425, 0.19696076, 0.335077, 1.0])


  model_name_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:,prediction_i]
    y = Ys[event_name]
    model_name_to_day_bins_logistic_coeffs[model_name] = find_single_predictor_logistic_coeffs(model_name, ŷ, y, weights, model_name_to_day_bins[model_name])
  end

  # model_name  bin input_ŷ_min input_ŷ_max count   pos_count weight     mean_input_ŷ mean_y        input_logloss input_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
  # tornado     1-2 -1.0        0.056085516 9707078 10413.0   8.944582e6 0.0010709693 0.0010811437  0.005866548   0.03158136  0.0010811436    0.0058664964     0.031581346    Float32[1.0015056,  0.016357854]
  # tornado     2-3 0.017779902 0.11559871  221744  10319.0   207895.38  0.043916903  0.04651174    0.17983219    0.088792324 0.046511732     0.17964691       0.08879232     Float32[1.1276554,  0.43491325]
  # tornado     3-4 0.056085516 1.0         84938   10193.0   80550.19   0.112839356  0.1200137     0.35172924    0.21909066  0.1200137       0.35137722       0.21909064     Float32[0.93115807, -0.062422168]
  # wind        1-2 -1.0        0.19696076  9584620 74571.0   8.832288e6 0.007558949  0.00783962    0.028688313   0.14068365  0.007839621     0.028673481      0.14068362     Float32[1.0421747,  0.15582056]
  # wind        2-3 0.08978425  0.335077    385802  74488.0   359088.97  0.17555769   0.19282469    0.47333708    0.2806192   0.19282469      0.47223735       0.28061917     Float32[1.0635164,  0.21312958]
  # wind        3-4 0.19696076  1.0         207396  74363.0   192843.83  0.32849598   0.35904726    0.6246096     0.51263994  0.35904726      0.6223894        0.5126399      Float32[1.0056635,  0.14828795]
  # hail        1-2 -1.0        0.11616666  9635114 33963.0   8.880074e6 0.0032225274 0.0035332483  0.014940207   0.07859565  0.0035332488    0.014923536      0.078595735    Float32[1.0227455,  0.17348957]
  # hail        2-3 0.048839394 0.22025906  309916  33903.0   286455.44  0.104178034  0.109529145   0.33623746    0.1660837   0.10952915      0.33602694       0.16608356     Float32[0.92801446, -0.09182161]
  # hail        3-4 0.11616666  1.0         156902  33875.0   145057.9   0.20669954   0.2162823     0.4983232     0.34015056  0.2162823       0.49790654       0.34015054     Float32[1.0785028,  0.16131271]
  # sig_tornado 1-2 -1.0        0.022638915 9764598 1356.0    8.998732e6 0.0001652332 0.00014191134 0.00092006614 0.011236978 0.00014191133   0.0009180771     0.011236974    Float32[1.0351814,  0.035491165]
  # sig_tornado 2-3 0.007377854 0.047667596 82679   1340.0    78526.33   0.017584842  0.016256474   0.079973266   0.03764813  0.016256472     0.079801075      0.037648115    Float32[1.2383188,  0.8401045]
  # sig_tornado 3-4 0.022638915 1.0         27418   1325.0    26399.441  0.04118546   0.04825523    0.18335064    0.13141018  0.04825523      0.18136714       0.13141017     Float32[1.4964094,  1.6581587]

  println("model_name_to_day_bins_logistic_coeffs = $model_name_to_day_bins_logistic_coeffs")

  # model_name_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("hail" => [[1.0227455, 0.17348957], [0.92801446, -0.09182161], [1.0785028, 0.16131271]], "tornado" => [[1.0015056, 0.016357854], [1.1276554, 0.43491325], [0.93115807, -0.062422168]], "sig_tornado" => [[1.0351814, 0.035491165], [1.2383188, 0.8401045], [1.4964094, 1.6581587]], "wind" => [[1.0421747, 0.15582056], [1.0635164, 0.21312958], [1.0056635, 0.14828795]])

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
  # tornado (20606.0) feature 1 TORPROB:calculated:hour fcst:calculated_prob:	AU-PR-curve: 0.14098173
  # wind (148934.0) feature 2 WINDPROB:calculated:hour fcst:calculated_prob:	AU-PR-curve: 0.37298736
  # hail (67838.0) feature 3 HAILPROB:calculated:hour fcst:calculated_prob:	AU-PR-curve: 0.23869608
  # sig_tornado (2681.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob:	AU-PR-curve: 0.07735924

  println("\nPlotting calibrated reliability...")

  Metrics.reliability_curves_midpoints(20, X, Ys, event_names, weights, model_names)
  # ŷ_tornado,y_tornado,ŷ_wind,y_wind,ŷ_hail,y_hail,ŷ_sig_tornado,y_sig_tornado,
  # 0.00013134995,0.00011843994,0.000818266,0.00087598583,0.00038657084,0.00038366142,1.4345453e-5,1.4710603e-5,
  # 0.0027048204,0.003187759,0.022334035,0.02145594,0.012939246,0.013542846,0.00086865656,0.0011092217,
  # 0.0057206415,0.006294886,0.04208846,0.042164184,0.023673184,0.024223622,0.002080959,0.0019258133,
  # 0.009810605,0.009814919,0.06195004,0.059982035,0.035156403,0.036733776,0.0038074732,0.003786738,
  # 0.014941979,0.0141764525,0.08313135,0.07951022,0.047281053,0.04699608,0.005575609,0.0067118937,
  # 0.020897264,0.01924385,0.10541627,0.10438638,0.06060659,0.05843306,0.0076126023,0.0061904085,
  # 0.027446182,0.027920142,0.1284696,0.12732504,0.073981665,0.076691754,0.009832607,0.012335022,
  # 0.034723256,0.036330532,0.15270855,0.15037367,0.08672153,0.09371921,0.0118676815,0.014308246,
  # 0.043405697,0.042489573,0.1776858,0.18387905,0.09993577,0.100554995,0.014494701,0.014190837,
  # 0.053925585,0.04993584,0.20357351,0.2055122,0.11429911,0.10622752,0.018629482,0.013384952,
  # 0.06585663,0.065290116,0.23028484,0.2413093,0.13000187,0.12049526,0.023288963,0.02356218,
  # 0.07769836,0.08230266,0.25791505,0.26947764,0.14760502,0.14141887,0.027746085,0.025068555,
  # 0.08999903,0.08934192,0.28701305,0.29204297,0.16716824,0.1696269,0.034989152,0.02618188,
  # 0.10283678,0.10877207,0.31779006,0.30529642,0.18921308,0.19202818,0.04311108,0.055242956,
  # 0.11636499,0.12191817,0.35051364,0.3429442,0.21562018,0.2155577,0.051687784,0.06818374,
  # 0.13422783,0.12399165,0.38741848,0.36651364,0.24612023,0.25238943,0.06573757,0.05042573,
  # 0.15889733,0.14902608,0.43135116,0.41420168,0.2804574,0.30427223,0.086884215,0.08361463,
  # 0.1906426,0.178466,0.48734456,0.48547402,0.3215229,0.33664262,0.112039365,0.113936454,
  # 0.23545957,0.24517012,0.5569721,0.580798,0.37718174,0.36918792,0.15571903,0.13341472,
  # 0.3444305,0.36027357,0.6713607,0.6873694,0.48697594,0.45890474,0.20019063,0.26925153,


  println("Determining SPC-like calibration curve...")

  println("model_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
  spc_calibrations = Dict{String,Vector{Tuple{Float32,Float32}}}()
  for (prediction_i, (model_name, event_name)) in enumerate(zip(model_names, event_names))
    ŷ = @view X[:, prediction_i]
    y = Ys[event_name]
    spc_calibrations[model_name] = spc_calibrate_warning_ratio(event_name, model_name, ŷ, y, weights)
  end
  # model_name  nominal_prob threshold_to_match_warning_ratio SR          POD          WR
  # tornado     0.02         0.018529892                      0.06301025  0.74537677   0.025346119
  # tornado     0.05         0.071611404                      0.13199824  0.45200795   0.0073371064
  # tornado     0.1          0.16732597                       0.23433603  0.16214712   0.0014825762
  # tornado     0.15         0.27986336                       0.36170173  0.042768326  0.0002533486
  # tornado     0.3          0.41645622                       0.52410316  0.009176224  3.7514063e-5
  # tornado     0.45         0.5810032                        0.7332181   0.0011121558 3.24997e-6
  # wind        0.05         0.052114487                      0.18543418  0.8507414    0.07039584
  # wind        0.15         0.2152462                        0.35734016  0.5038089    0.021633327
  # wind        0.3          0.4713688                        0.58470845  0.13831793   0.0036297638
  # wind        0.45         0.67717934                       0.72756493  0.019737555  0.00041625634
  # hail        0.05         0.03102684                       0.111510806 0.84417415   0.05263421
  # hail        0.15         0.12537575                       0.22053955  0.4890614    0.015418065
  # hail        0.3          0.35396385                       0.41604617  0.09305421   0.0015550613
  # hail        0.45         0.56297874                       0.5660734   0.007285849  8.94871e-5
  # sig_tornado 0.1          0.046430588                      0.08917182  0.3005327    0.00095259794

  println("spc_calibrations = $spc_calibrations")
  # spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}("hail" => [(0.05, 0.03102684), (0.15, 0.12537575), (0.3, 0.35396385), (0.45, 0.56297874)], "tornado" => [(0.02, 0.018529892), (0.05, 0.071611404), (0.1, 0.16732597), (0.15, 0.27986336), (0.3, 0.41645622), (0.45, 0.5810032)], "sig_tornado" => [(0.1, 0.046430588)], "wind" => [(0.05, 0.052114487), (0.15, 0.2152462), (0.3, 0.4713688), (0.45, 0.67717934)])


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
