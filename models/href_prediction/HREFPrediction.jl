module HREFPrediction

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts



push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF


_forecasts = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred = [] # For downstream combination with other forecasts
_forecasts_calibrated = []
_forecasts_calibrated_with_sig_gated = []

_forecasts_day_accumulators                   = []
_forecasts_day2_accumulators                  = []
_forecasts_fourhourly_accumulators            = []
_forecasts_day                                = []
_forecasts_day_with_sig_gated                 = []
_forecasts_day2                               = []
_forecasts_day2_with_sig_gated                = []
_forecasts_fourhourly                         = []
_forecasts_fourhourly_with_sig_gated          = []
_forecasts_day_spc_calibrated                 = []
_forecasts_day_spc_calibrated_with_sig_gated  = []
_forecasts_day2_spc_calibrated                = []
_forecasts_day2_spc_calibrated_with_sig_gated = []



σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

blur_radii = [15, 25, 35, 50, 70, 100]


# we only trained out to f35, but with some assumptions we can try to go out to f47 even though I don't have an HREF archive out to f48 to train against
regular_forecasts(forecasts)  = filter(f -> f.forecast_hour in 2:35,  forecasts)
extended_forecasts(forecasts) = filter(f -> f.forecast_hour in 36:47,  forecasts)

function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
    _forecasts
  else
    _forecasts
  end
end

function example_forecast()
  forecasts()[1]
end

function grid()
  HREF.grid()
end

function forecasts_with_blurs_and_forecast_hour()
  if isempty(_forecasts_with_blurs_and_forecast_hour)
    reload_forecasts()
    _forecasts_with_blurs_and_forecast_hour
  else
    _forecasts_with_blurs_and_forecast_hour
  end
end

function forecasts_blurred()
  if isempty(_forecasts_blurred)
    reload_forecasts()
    _forecasts_blurred
  else
    _forecasts_blurred
  end
end

function forecasts_calibrated()
  if isempty(_forecasts_calibrated)
    reload_forecasts()
    _forecasts_calibrated
  else
    _forecasts_calibrated
  end
end

function forecasts_calibrated_with_sig_gated()
  if isempty(_forecasts_calibrated_with_sig_gated)
    reload_forecasts()
    _forecasts_calibrated_with_sig_gated
  else
    _forecasts_calibrated_with_sig_gated
  end
end

function forecasts_day_accumulators()
  if isempty(_forecasts_day_accumulators)
    reload_forecasts()
    _forecasts_day_accumulators
  else
    _forecasts_day_accumulators
  end
end

function forecasts_day2_accumulators()
  if isempty(_forecasts_day2_accumulators)
    reload_forecasts()
    _forecasts_day2_accumulators
  else
    _forecasts_day2_accumulators
  end
end

function forecasts_fourhourly_accumulators()
  if isempty(_forecasts_fourhourly_accumulators)
    reload_forecasts()
    _forecasts_fourhourly_accumulators
  else
    _forecasts_fourhourly_accumulators
  end
end

function forecasts_day()
  if isempty(_forecasts_day)
    reload_forecasts()
    _forecasts_day
  else
    _forecasts_day
  end
end

function forecasts_day_with_sig_gated()
  if isempty(_forecasts_day_with_sig_gated)
    reload_forecasts()
    _forecasts_day_with_sig_gated
  else
    _forecasts_day_with_sig_gated
  end
end

function forecasts_day2()
  if isempty(_forecasts_day2)
    reload_forecasts()
    _forecasts_day2
  else
    _forecasts_day2
  end
end

function forecasts_day2_with_sig_gated()
  if isempty(_forecasts_day2_with_sig_gated)
    reload_forecasts()
    _forecasts_day2_with_sig_gated
  else
    _forecasts_day2_with_sig_gated
  end
end

function forecasts_fourhourly()
  if isempty(_forecasts_fourhourly)
    reload_forecasts()
    _forecasts_fourhourly
  else
    _forecasts_fourhourly
  end
end

function forecasts_fourhourly_with_sig_gated()
  if isempty(_forecasts_fourhourly_with_sig_gated)
    reload_forecasts()
    _forecasts_fourhourly_with_sig_gated
  else
    _forecasts_fourhourly_with_sig_gated
  end
end

function forecasts_day_spc_calibrated()
  if isempty(_forecasts_day_spc_calibrated)
    reload_forecasts()
    _forecasts_day_spc_calibrated
  else
    _forecasts_day_spc_calibrated
  end
end

function forecasts_day_spc_calibrated_with_sig_gated()
  if isempty(_forecasts_day_spc_calibrated_with_sig_gated)
    reload_forecasts()
    _forecasts_day_spc_calibrated_with_sig_gated
  else
    _forecasts_day_spc_calibrated_with_sig_gated
  end
end

function forecasts_day2_spc_calibrated()
  if isempty(_forecasts_day2_spc_calibrated)
    reload_forecasts()
    _forecasts_day2_spc_calibrated
  else
    _forecasts_day2_spc_calibrated
  end
end

function forecasts_day2_spc_calibrated_with_sig_gated()
  if isempty(_forecasts_day2_spc_calibrated_with_sig_gated)
    reload_forecasts()
    _forecasts_day2_spc_calibrated_with_sig_gated
  else
    _forecasts_day2_spc_calibrated_with_sig_gated
  end
end


# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
models = [
  ("tornado", "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-19T10.19.24.875_tornado_climatology_all/440_trees_loss_0.0011318683.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_tornado_climatology_all/676_trees_loss_0.0012007512.model",
                         "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_tornado_climatology_all/538_trees_loss_0.0012588982.model"
  ),
  ("wind", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-21T20.38.44.560_wind_climatology_all/1189_trees_loss_0.006638963.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_wind_climatology_all/1251_trees_loss_0.007068191.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_wind_climatology_all/877_trees_loss_0.0074603.model"
  ),
  ("wind_adj", "WINDPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T17.52.21.064_wind_adj_climatology_all/1251_trees_loss_0.0024581964.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-21T10.19.53.046_wind_adj_climatology_all/877_trees_loss_0.002577666.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.35.163_wind_adj_climatology_all/943_trees_loss_0.002697009.model"
  ),
  ("hail", "HAILPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-17T18.47.33.015_hail_climatology_all/1163_trees_loss_0.0033772641.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-20T18.33.36.099_hail_climatology_all/1057_trees_loss_0.0036187447.model",
                       "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.35.163_hail_climatology_all/1249_trees_loss_0.0038463715.model"
  ),
  ("sig_tornado", "STORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-20T10.55.18.222_sig_tornado_climatology_all/941_trees_loss_0.00017207165.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-22T02.14.27_sig_tornado_climatology_all/1046_trees_loss_0.00018272862.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T15.45.33.554_sig_tornado_climatology_all/298_trees_loss_0.00019245308.model"
  ),
  ("sig_wind", "SWINDPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-20T10.55.18.222_sig_wind_climatology_all/459_trees_loss_0.0009965667.model",
                          "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-22T02.14.27_sig_wind_climatology_all/466_trees_loss_0.0010346768.model",
                          "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T15.45.33.554_sig_wind_climatology_all/698_trees_loss_0.001081094.model"
  ),
  ("sig_wind_adj", "SWINDPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-17T18.47.33.015_sig_wind_adj_climatology_all/312_trees_loss_0.00038774137.model",
                               "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-20T18.33.36.099_sig_wind_adj_climatology_all/349_trees_loss_0.00039779497.model",
                               "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-26T17.12.11.352_sig_wind_adj_climatology_all/338_trees_loss_0.00041165648.model"
  ),
  ("sig_hail", "SHAILPRO", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-15T17.52.21.064_sig_hail_climatology_all/562_trees_loss_0.000527038.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-26T08.42.12.448_sig_hail_climatology_all/1146_trees_loss_0.000567788.model",
                           "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.35.163_sig_hail_climatology_all/574_trees_loss_0.00061303715.model"
  ),
]

# (gated_event_name, original_event_name, gate_event_name)
gated_models =
  [
    ("sig_tornado_gated_by_tornado",   "sig_tornado",  "tornado"),
    ("sig_wind_gated_by_wind",         "sig_wind",     "wind"),
    ("sig_wind_adj_gated_by_wind_adj", "sig_wind_adj", "wind_adj"),
    ("sig_hail_gated_by_hail",         "sig_hail",     "hail"),
  ]

# (event_name, grib2_var_name, model_name)
# I don't think the middle value here is ever used.
models_with_gated =
  [ ("tornado",      "TORPROB",  "tornado")
  , ("wind",         "WINDPROB", "wind")
  , ("wind_adj",     "WINDPROB", "wind_adj")
  , ("hail",         "HAILPROB", "hail")
  , ("sig_tornado",  "STORPRO",  "sig_tornado")
  , ("sig_wind",     "SWINDPRO", "sig_wind")
  , ("sig_wind_adj", "SWINDPRO", "sig_wind_adj")
  , ("sig_hail",     "SHAILPRO", "sig_hail")
  , ("sig_tornado",  "STORPRO",  "sig_tornado_gated_by_tornado")
  , ("sig_wind",     "SWINDPRO", "sig_wind_gated_by_wind")
  , ("sig_wind_adj", "SWINDPRO", "sig_wind_adj_gated_by_wind_adj")
  , ("sig_hail",     "SHAILPRO", "sig_hail_gated_by_hail")
  ]


function reload_forecasts()
  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred
  global _forecasts_calibrated
  global _forecasts_calibrated_with_sig_gated

  global _forecasts_day_accumulators
  global _forecasts_day2_accumulators
  global _forecasts_fourhourly_accumulators
  global _forecasts_day
  global _forecasts_day_with_sig_gated
  global _forecasts_day2
  global _forecasts_day2_with_sig_gated
  global _forecasts_fourhourly
  global _forecasts_fourhourly_with_sig_gated
  global _forecasts_day_spc_calibrated
  global _forecasts_day_spc_calibrated_with_sig_gated
  global _forecasts_day2_spc_calibrated
  global _forecasts_day2_spc_calibrated_with_sig_gated

  _forecasts = []

  href_forecasts = HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predictors = map(models) do (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
    predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f2_to_f13)
    predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f13_to_f24)
    predict_f24_to_f35 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f24_to_f35)

    predict(forecast, data) = begin
      if forecast.forecast_hour in 25:47 # For f36-f47, use the f24-f35 GBDT, but we will blur more below
        predict_f24_to_f35(data)
      elseif forecast.forecast_hour == 24
        0.5f0 .* (predict_f24_to_f35(data) .+ predict_f13_to_f24(data))
      elseif forecast.forecast_hour in 14:23
        predict_f13_to_f24(data)
      elseif forecast.forecast_hour == 13
        0.5f0 .* (predict_f13_to_f24(data) .+ predict_f2_to_f13(data))
      elseif forecast.forecast_hour in 2:12
        predict_f2_to_f13(data)
      else
        error("HREF forecast hour $(forecast.forecast_hour) not in 2:47")
      end
    end

    (event_name, grib2_var_name, predict)
  end

  # Don't forget to clear the cache during development.
  # rm -r lib/computation_cache/cached_forecasts/href_prediction_raw_2021_models
  _forecasts =
    ForecastCombinators.disk_cache_forecasts(
      PredictionForecasts.simple_prediction_forecasts(href_forecasts, predictors),
      "href_prediction_raw_2022_models_$(hash(models))"
    )

  # Only used incidentally to determine best blur radii
  # _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(regular_forecasts(_forecasts), blur_radii)

  grid = _forecasts[1].grid

  # Determined in Train.jl
  # event_name   best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado      15                  15                   0.047551293
  # wind         15                  25                   0.1164659
  # wind_adj     15                  25                   0.06341821
  # hail         15                  25                   0.07535621
  # sig_tornado  15                  15                   0.033054337
  # sig_wind     15                  35                   0.016219445
  # sig_wind_adj 100                 100                  0.012251711
  # sig_hail     0                   15                   0.018009394

  blur_0mi_grid_is   = Grids.radius_grid_is(grid, 0.0)
  blur_15mi_grid_is  = Grids.radius_grid_is(grid, 15.0)
  blur_25mi_grid_is  = Grids.radius_grid_is(grid, 25.0)
  blur_35mi_grid_is  = Grids.radius_grid_is(grid, 35.0)
  blur_50mi_grid_is  = Grids.radius_grid_is(grid, 50.0)
  blur_100mi_grid_is = Grids.radius_grid_is(grid, 100.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_15mi_grid_is,  blur_15mi_grid_is),  # tornado
    (blur_15mi_grid_is,  blur_25mi_grid_is),  # wind
    (blur_15mi_grid_is,  blur_25mi_grid_is),  # wind_adj
    (blur_15mi_grid_is,  blur_25mi_grid_is),  # hail
    (blur_15mi_grid_is,  blur_15mi_grid_is),  # sig_tornado
    (blur_15mi_grid_is,  blur_35mi_grid_is),  # sig_wind
    (blur_100mi_grid_is, blur_100mi_grid_is), # sig_wind_adj
    (blur_0mi_grid_is,   blur_15mi_grid_is),  # sig_hail
  ]

  extended_forecasts_blur_grid_is = [
    (blur_15mi_grid_is,  blur_50mi_grid_is),  # tornado
    (blur_25mi_grid_is,  blur_50mi_grid_is),  # wind
    (blur_25mi_grid_is,  blur_50mi_grid_is),  # wind_adj
    (blur_25mi_grid_is,  blur_50mi_grid_is),  # hail
    (blur_15mi_grid_is,  blur_50mi_grid_is),  # sig_tornado
    (blur_35mi_grid_is,  blur_50mi_grid_is),  # sig_wind
    (blur_100mi_grid_is, blur_100mi_grid_is), # sig_wind_adj
    (blur_15mi_grid_is,  blur_50mi_grid_is),  # sig_hail
  ]

  _forecasts_blurred =
    vcat(
      PredictionForecasts.blurred(regular_forecasts(_forecasts),  2:35,  blur_grid_is),
      PredictionForecasts.blurred(extended_forecasts(_forecasts), 35:47, extended_forecasts_blur_grid_is), # Yes, 35:47 is correct so that f36 uses a bit of the larger radii blur
    )



  # Calibrating hourly predictions to validation data so we can make HREF-only day 2 forecasts.

  event_to_bins = Dict{String, Vector{Float32}}(
    "tornado"      => [0.0012153604,  0.004538168,  0.011742671,  0.023059275,  0.049979284,  1.0],
    "wind"         => [0.0078376075,  0.020091131,  0.037815828,  0.065400586,  0.11733462,   1.0],
    "wind_adj"     => [0.0025728762,  0.008021789,  0.016496103,  0.030200316,  0.058084033,  1.0],
    "hail"         => [0.0033965781,  0.00951148,   0.0200999,    0.037743744,  0.07645232,   1.0],
    "sig_tornado"  => [0.00080238597, 0.0032272756, 0.007747574,  0.0136314705, 0.022335978,  1.0],
    "sig_wind"     => [0.0006871624,  0.0022018508, 0.0047563836, 0.008192127,  0.015201166,  1.0],
    "sig_wind_adj" => [0.00042257016, 0.0012706288, 0.002259613,  0.003675646,  0.0054352577, 1.0],
    "sig_hail"     => [0.00072055974, 0.0021597594, 0.00463198,   0.009187652,  0.01903059,   1.0],
  )
  event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"      => [[1.0710595, 0.5394862],  [0.89414734, -0.5616081],   [1.108447,  0.50613326],  [0.83939207, -0.5758451],  [1.2171175,  0.7334359]],
    "wind"         => [[1.0940193, 0.44082266], [1.1078424,  0.48961264],   [1.0705312, 0.36004978],  [1.0221666,  0.22469698],  [0.9163128,  -0.022937609]],
    "wind_adj"     => [[1.1026261, 0.554337],   [1.1907648,  0.9671046],    [1.1658177, 0.862035],    [1.0740721,  0.5209931],   [0.9590977,  0.1968526]],
    "hail"         => [[1.0600259, 0.43388337], [0.96319646, -0.097637914], [1.0792756, 0.39443073],  [0.9629313,  -0.01864577], [1.0167043,  0.13172606]],
    "sig_tornado"  => [[1.0426707, 0.21755064], [1.1516033,  0.7787872],    [1.6196959, 3.2147853],   [1.4218588,  2.3287997],   [1.032921,   0.7553395]],
    "sig_wind"     => [[1.0894886, 0.7778136],  [1.0595194,  0.49330923],   [1.3312615, 2.0373042],   [0.87163,    -0.28045976], [0.66988313, -1.1392391]],
    "sig_wind_adj" => [[1.0503622, 0.2618211],  [1.5049535,  3.5953336],    [1.3739139, 2.7983332],   [2.022486,   6.61417],     [0.99839044, 1.0990105]],
    "sig_hail"     => [[1.0996535, 0.9160373],  [1.0066487,  0.20982021],   [0.9350299, -0.20348155], [0.9644414,  -0.02964372], [0.8333078,  -0.5743816]],
  )

  # Returns array of (event_name, var_name, predict)
  function make_models(event_to_bins, event_to_bins_logistic_coeffs)
    ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

    map(1:length(models)) do model_i
      event_name, var_name, _, _, _ = models[model_i] # event_name == model_name here

      predict(forecasts, data) = begin
        href_ŷs = @view data[:,model_i]

        out = Array{Float32}(undef, length(href_ŷs))

        bin_maxes            = event_to_bins[event_name]
        bins_logistic_coeffs = event_to_bins_logistic_coeffs[event_name]

        @assert length(bin_maxes) == length(bins_logistic_coeffs) + 1

        predict_one(coeffs, href_ŷ) = σ(coeffs[1]*logit(href_ŷ) + coeffs[2])

        Threads.@threads for i in 1:length(href_ŷs)
          href_ŷ = href_ŷs[i]
          if href_ŷ <= bin_maxes[1]
            # Bin 1-2 predictor only
            ŷ = predict_one(bins_logistic_coeffs[1], href_ŷ)
          elseif href_ŷ > bin_maxes[length(bin_maxes) - 1]
            # Bin 5-6 predictor only
            ŷ = predict_one(bins_logistic_coeffs[length(bins_logistic_coeffs)], href_ŷ)
          else
            # Overlapping bins
            higher_bin_i = findfirst(bin_max -> href_ŷ <= bin_max, bin_maxes)
            lower_bin_i  = higher_bin_i - 1
            coeffs_higher_bin = bins_logistic_coeffs[higher_bin_i]
            coeffs_lower_bin  = bins_logistic_coeffs[lower_bin_i]

            # Bin 1-2 and 2-3 predictors
            ratio = ratio_between(href_ŷ, bin_maxes[lower_bin_i], bin_maxes[higher_bin_i])
            ŷ = ratio*predict_one(coeffs_higher_bin, href_ŷ) + (1f0 - ratio)*predict_one(coeffs_lower_bin, href_ŷ)
          end
          out[i] = ŷ
        end

        out
      end

      (event_name, var_name, predict)
    end
  end

  hour_models = make_models(event_to_bins, event_to_bins_logistic_coeffs)

  _forecasts_calibrated                = PredictionForecasts.simple_prediction_forecasts(_forecasts_blurred, hour_models; model_name = "HREF_hour_severe_probabilities")
  _forecasts_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_calibrated, models, gated_models; model_name = "HREF_hour_severe_probabilities_with_sig_gated")


  # Day & Four-hourly forecasts

  # 1. Try both independent events total prob and max hourly prob as the main descriminator
  # 2. bin predictions into 10 bins of equal weight of positive labels
  # 3. combine bin-pairs (overlapping, 9 bins total)
  # 4. train a logistic regression for each bin,
  #   σ(a1*logit(independent events total prob) +
  #     a2*logit(max hourly prob) +
  #     b)
  # 5. prediction is weighted mean of the two overlapping logistic models
  # 6. should thereby be absolutely calibrated (check)
  # 7. calibrate to SPC thresholds (linear interpolation)

  _forecasts_day_accumulators, _forecasts_day2_accumulators, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(_forecasts_calibrated, models; module_name = "HREFPrediction")

  # The following was computed in TrainDay.jl

  event_to_0z_day_bins = Dict{String, Vector{Float32}}(
    "tornado"     => [0.017401028, 0.057005595, 0.13199422,  1.0],
    "wind"        => [0.105925485, 0.24237353,  0.41793627,  1.0],
    "wind_adj"        => [0.105925485, 0.24237353,  0.41793627,  1.0],
    "hail"        => [0.057583164, 0.13694991,  0.26336262,  1.0],
    "sig_tornado" => [0.00944276,  0.03155332,  0.1166033,   1.0],
    "sig_wind"    => [0.014537211, 0.044868514, 0.080929644, 1.0],
    "sig_wind_adj"    => [0.014537211, 0.044868514, 0.080929644, 1.0],
    "sig_hail"    => [0.017137118, 0.032750417, 0.06910927,  1.0],
  )
  event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[0.93318164, 0.06823707, -0.10806717],  [1.3238057,    -0.114339165, 0.34774518],  [0.6581387, 0.06536053,  -0.60667735]],
    "wind"        => [[0.9533327,  0.08198542, -0.049861502], [1.1315389,    -0.06399821,  -0.22480214], [0.9192797, 0.07381143,  -0.066281155]],
    "wind_adj"        => [[0.9533327,  0.08198542, -0.049861502], [1.1315389,    -0.06399821,  -0.22480214], [0.9192797, 0.07381143,  -0.066281155]],
    "hail"        => [[0.78442734, 0.27793196, 0.407107],     [1.0553415,    -0.1081846,   -0.47820166], [1.2830826, -0.47052482, -1.2408078]],
    "sig_tornado" => [[0.5930697,  0.4203289,  0.43342784],   [-0.032218266, 0.8780866,    0.3627351],   [0.6223907, 0.6945181,   1.4135333]],
    "sig_wind"    => [[0.53879964, 0.4059111,  0.116683125],  [1.0535268,    0.2508744,    1.1575938],   [0.619009,  0.59856415,  1.4483397]],
    "sig_wind_adj"    => [[0.53879964, 0.4059111,  0.116683125],  [1.0535268,    0.2508744,    1.1575938],   [0.619009,  0.59856415,  1.4483397]],
    "sig_hail"    => [[1.7509274,  -0.6264722, -0.59064865],  [1.5742372,    -0.7514552,   -1.8797148],  [1.4796587, -0.48270363, -0.9502378]],
  )

  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_0z_day_bins, event_to_0z_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day")
  _forecasts_day_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_with_sig_gated")

  _forecasts_day2 = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day2_accumulators, event_to_0z_day_bins, event_to_0z_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day2")
  _forecasts_day2_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_with_sig_gated")

  _forecasts_fourhourly = [] # PredictionForecasts.period_forecasts_from_accumulators(_forecasts_fourhourly_accumulators, event_to_fourhourly_bins, event_to_fourhourly_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "four-hourly")
  _forecasts_fourhourly_with_sig_gated = [] # PredictionForecasts.added_gated_predictions(_forecasts_fourhourly, models, gated_models; model_name = "HREFPrediction_four-hourly_severe_probabilities_with_sig_gated")

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado" => [
      (0.02, 0.017892838),
      (0.05, 0.07787514),
      (0.1,  0.17152214),
      (0.15, 0.2814541),
      (0.3,  0.3905239),
      (0.45, 0.6009083)
    ],
    "wind" => [
      (0.05, 0.051660538),
      (0.15, 0.21513557),
      (0.3,  0.49578285),
      (0.45, 0.78172493)
    ],
    "wind_adj" => [
      (0.05, 0.051660538),
      (0.15, 0.21513557),
      (0.3,  0.49578285),
      (0.45, 0.78172493)
    ],
    "hail" => [
      (0.05, 0.030927658),
      (0.15, 0.12172127),
      (0.3,  0.33656883),
      (0.45, 0.61953926)
    ],
    "sig_tornado"  => [(0.1, 0.063589096)],
    "sig_wind"     => [(0.1, 0.11205864)],
    "sig_wind_adj" => [(0.1, 0.11205864)],
    "sig_hail"     => [(0.1, 0.057775497)],
  )

  # ensure ordered the same as the features in the data
  calibrations = map(m -> spc_calibrations[m[1]], models)

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds")
  _forecasts_day_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  _forecasts_day2_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day2, calibrations; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds")
  _forecasts_day2_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  ()
end

end # module HREFPrediction