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

  # event_name   best_blur_radius_f2 best_blur_radius_f35 AU_PR
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

        Threads.@threads :static for i in 1:length(href_ŷs)
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

  event_to_day_bins = Dict{String, Vector{Float32}}(
    "tornado"      => [0.021043906, 0.074019335, 0.17095083, 1.0],
    "wind"         => [0.12138814,  0.27036425,  0.44575575, 1.0],
    "wind_adj"     => [0.04442204,  0.12223551,  0.25552708, 1.0],
    "hail"         => [0.066225864, 0.15690458,  0.29827937, 1.0],
    "sig_tornado"  => [0.009904385, 0.047619365, 0.14518347, 1.0],
    "sig_wind"     => [0.017750502, 0.050954822, 0.09204348, 1.0],
    "sig_wind_adj" => [0.00855192,  0.028172063, 0.07532653, 1.0],
    "sig_hail"     => [0.016018612, 0.044114113, 0.08977107, 1.0],
  )
  event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"      => [[0.95958227, 0.04161413,   -0.10651286],  [1.2272763,  -0.15624464,  -0.18067063],  [0.5964124,  0.17200926,   -0.3083448]],
    "wind"         => [[1.0424639,  -0.005972234, -0.1651274],   [1.1494,     -0.107195236, -0.3110462],   [0.92132,    0.0022699288, -0.22744325]],
    "wind_adj"     => [[0.99742454, 0.02784912,   -0.01899123],  [1.0577022,  -0.1336032,   -0.51860476],  [1.2023934,  -0.058322832, -0.087366514]],
    "hail"         => [[1.0235776,  0.02489767,   -0.070147164], [1.1081746,  -0.088940814, -0.34076628],  [1.1614795,  -0.24988303,  -0.7304756]],
    "sig_tornado"  => [[0.3738464,  0.5298857,    -0.016889505], [0.53315455, 0.50932276,   0.41353387],   [0.41091824, 0.60895973,   0.510827]],
    "sig_wind"     => [[0.60896295, 0.32489055,   -0.06371247],  [0.99172807, 0.15444452,   0.46648186],   [0.8760332,  0.31676358,   0.86151475]],
    "sig_wind_adj" => [[0.98283756, 0.02864707,   -0.070753366], [0.74302465, 0.17016491,   -0.2746076],   [0.37014917, 1.2757467,    3.3445797]],
    "sig_hail"     => [[1.2635667,  -0.21535042,  -0.32895777],  [1.5278528,  -0.32414392,  0.0025766783], [1.1042255,  -0.16546442,  -0.4495338]],
  )

  event_to_fourhourly_bins = Dict{String, Vector{Float32}}(
    "tornado"      => [0.009209944,  0.035263043, 0.09849834,  1.0],
    "wind"         => [0.04380678,   0.1264081,   0.26765373,  1.0],
    "wind_adj"     => [0.016413603,  0.058604192, 0.15199223,  1.0],
    "hail"         => [0.021138927,  0.06349602,  0.15349576,  1.0],
    "sig_tornado"  => [0.005631322,  0.026526544, 0.092573315, 1.0],
    "sig_wind"     => [0.0057922173, 0.0215623,   0.047722477, 1.0],
    "sig_wind_adj" => [0.0030460916, 0.012603851, 0.040286843, 1.0],
    "sig_hail"     => [0.005561535,  0.017045472, 0.04543988,  1.0],
  )

  event_to_fourhourly_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"      => [[1.0933163,  -0.07491289,   -0.09395848],  [1.2857381,  -0.32322577, -0.54111695], [0.89995205, 0.025575645, -0.32821795]],
    "wind"         => [[0.97546655, 0.025674382,   -0.12313984],  [1.0660465,  -0.06211259, -0.19784704], [1.0678099,  -0.11768198, -0.33153704]],
    "wind_adj"     => [[1.0981627,  -0.08259544,   -0.08323573],  [1.1117252,  -0.15500262, -0.3613525],  [1.1775827,  -0.12373863, -0.11497526]],
    "hail"         => [[0.96931684, 0.03862384,    -0.061203808], [1.1384751,  -0.16171022, -0.3580413],  [1.1030476,  -0.20756821, -0.5726693]],
    "sig_tornado"  => [[1.2125412,  -0.21225019,   -0.29702586],  [1.0641434,  -0.11383827, -0.45060942], [0.5815711,  0.39582604,  -0.006172086]],
    "sig_wind"     => [[0.9948475,  0.00020837433, -0.15796766],  [0.99380416, 0.082129076, 0.25521365],  [0.9400184,  0.20437491,  0.5816906]],
    "sig_wind_adj" => [[1.359399,   -0.3299467,    -0.18067853],  [0.9419031,  0.005614741, -0.3110229],  [0.6525776,  0.65669703,  1.5379776]],
    "sig_hail"     => [[1.1378537,  -0.13094603,   -0.17552955],  [1.2316782,  -0.27330196, -0.5414381],  [1.1437855,  -0.1293063,  -0.20190638]],
  )

  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day")
  _forecasts_day_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_with_sig_gated")

  _forecasts_day2 = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day2_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day2")
  _forecasts_day2_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_with_sig_gated")

  _forecasts_fourhourly = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_fourhourly_accumulators, event_to_fourhourly_bins, event_to_fourhourly_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "four-hourly")
  _forecasts_fourhourly_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_fourhourly, models, gated_models; model_name = "HREFPrediction_four-hourly_severe_probabilities_with_sig_gated")

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado" => [
      (0.02, 0.016950607),
      (0.05, 0.06830406),
      (0.1,  0.17817497),
      (0.15, 0.3255825),
      (0.3,  0.4591999),
      (0.45, 0.60011864)
    ],
    "wind_adj" => [
      (0.05, 0.011899948),
      (0.15, 0.06845665),
      (0.3,  0.24048424),
      (0.45, 0.471941)
    ],
    "wind" => [
      (0.05, 0.0479908),
      (0.15, 0.21794319),
      (0.3,  0.4940548),
      (0.45, 0.7420559)
    ],
    "hail" => [
      (0.05, 0.029951096),
      (0.15, 0.123464584),
      (0.3,  0.3763752),
      (0.45, 0.6608143)
    ],
    "sig_tornado"  => [(0.1, 0.06202507)],
    "sig_wind"     => [(0.1, 0.118608475)],
    "sig_wind_adj" => [(0.1, 0.06686592)],
    "sig_hail"     => [(0.1, 0.070280075)],
  )

  # spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
  #   "tornado" => [
  #     (0.02, 0.017892838),
  #     (0.05, 0.07787514),
  #     (0.1,  0.17152214),
  #     (0.15, 0.2814541),
  #     (0.3,  0.3905239),
  #     (0.45, 0.6009083)
  #   ],
  #   "wind" => [
  #     (0.05, 0.051660538),
  #     (0.15, 0.21513557),
  #     (0.3,  0.49578285),
  #     (0.45, 0.78172493)
  #   ],
  #   "wind_adj" => [
  #     (0.05, 0.051660538),
  #     (0.15, 0.21513557),
  #     (0.3,  0.49578285),
  #     (0.45, 0.78172493)
  #   ],
  #   "hail" => [
  #     (0.05, 0.030927658),
  #     (0.15, 0.12172127),
  #     (0.3,  0.33656883),
  #     (0.45, 0.61953926)
  #   ],
  #   "sig_tornado"  => [(0.1, 0.063589096)],
  #   "sig_wind"     => [(0.1, 0.11205864)],
  #   "sig_wind_adj" => [(0.1, 0.11205864)],
  #   "sig_hail"     => [(0.1, 0.057775497)],
  # )

  # ensure ordered the same as the features in the data
  calibrations = map(m -> spc_calibrations[m[1]], models)

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds")
  _forecasts_day_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  _forecasts_day2_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day2, calibrations; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds")
  _forecasts_day2_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  ()
end

end # module HREFPrediction