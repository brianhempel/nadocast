module HREFPredictionAblations

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

_forecasts_day_accumulators                   = []
_forecasts_day                                = []


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

function forecasts_day_accumulators()
  if isempty(_forecasts_day_accumulators)
    reload_forecasts()
    _forecasts_day_accumulators
  else
    _forecasts_day_accumulators
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


# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
models = [
  ("tornadao_mean_58",                                           "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.584_tornado_excluding_13773_features_4410986610349231978/700_trees_loss_0.0012174041.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.26.38.345_tornado_excluding_13773_features_4410986610349231978/707_trees_loss_0.0012815231.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.26.10.688_tornado_excluding_13773_features_4410986610349231978/705_trees_loss_0.0013271595.model"),
  ("tornadao_prob_80",                                           "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.550_tornado_excluding_13751_features_16850198170780752800/812_trees_loss_0.0012124014.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T01.27.28.031_tornado_excluding_13751_features_16850198170780752800/526_trees_loss_0.0012738121.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T02.44.38.100_tornado_excluding_13751_features_16850198170780752800/566_trees_loss_0.0013254478.model"),
  ("tornadao_mean_prob_138",                                     "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.590_tornado_excluding_13693_features_7893827997600368035/1207_trees_loss_0.0011827772.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.18.52.783_tornado_excluding_13693_features_7893827997600368035/732_trees_loss_0.001246924.model",    "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.19.01.750_tornado_excluding_13693_features_7893827997600368035/390_trees_loss_0.0012985576.model"),
  ("tornadao_mean_prob_computed_no_sv_219",                      "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.728_tornado_excluding_13612_features_12247408129229835884/779_trees_loss_0.0011766684.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T02.48.31.809_tornado_excluding_13612_features_12247408129229835884/971_trees_loss_0.0012399479.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T03.09.01.898_tornado_excluding_13612_features_12247408129229835884/832_trees_loss_0.0012905387.model"),
  ("tornadao_mean_prob_computed_220",                            "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T11.29.03.470_tornado_excluding_13611_features_10698383838099131809/1041_trees_loss_0.0011743465.model", "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T04.29.00.505_tornado_excluding_13611_features_10698383838099131809/882_trees_loss_0.0012400731.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T05.37.18.512_tornado_excluding_13611_features_10698383838099131809/964_trees_loss_0.0012917896.model"),
  ("tornadao_mean_prob_computed_partial_climatology_227",        "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T11.53.56.474_tornado_excluding_13604_features_16220645629827272623/1158_trees_loss_0.0011694039.model", "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T06.17.31.380_tornado_excluding_13604_features_16220645629827272623/963_trees_loss_0.0012260334.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T07.10.18.386_tornado_excluding_13604_features_16220645629827272623/994_trees_loss_0.0012774036.model"),
  ("tornadao_mean_prob_computed_climatology_253",                "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.03.56.368_tornado_excluding_13578_features_1255749354977721151/1135_trees_loss_0.0011634461.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T07.18.00.712_tornado_excluding_13578_features_1255749354977721151/1207_trees_loss_0.0012247592.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T08.44.56.699_tornado_excluding_13578_features_1255749354977721151/866_trees_loss_0.0012709117.model"),
  ("tornadao_mean_prob_computed_climatology_blurs_910",          "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.20.32.745_tornado_excluding_12921_features_7409514120157424229/868_trees_loss_0.0011486912.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T10.52.42.948_tornado_excluding_12921_features_7409514120157424229/859_trees_loss_0.0012130085.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T11.04.58.206_tornado_excluding_12921_features_7409514120157424229/852_trees_loss_0.0012569018.model"),
  ("tornadao_mean_prob_computed_climatology_grads_1348",         "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.51.679_tornado_excluding_12483_features_2140312678839727709/793_trees_loss_0.0011474881.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T11.24.59.952_tornado_excluding_12483_features_2140312678839727709/855_trees_loss_0.0012191372.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T12.47.02.062_tornado_excluding_12483_features_2140312678839727709/735_trees_loss_0.00126751.model"),
  ("tornadao_mean_prob_computed_climatology_blurs_grads_2005",   "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.46.15.004_tornado_excluding_11826_features_14576562183048218741/523_trees_loss_0.0011400548.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T16.13.41.561_tornado_excluding_11826_features_14576562183048218741/558_trees_loss_0.0012086717.model",  "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T16.13.15.348_tornado_excluding_11826_features_14576562183048218741/578_trees_loss_0.0012580294.model"),
  ("tornadao_mean_prob_computed_climatology_prior_next_hrs_691", "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.58.07.439_tornado_excluding_13140_features_3720177259145965027/836_trees_loss_0.0011478734.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T14.24.13.566_tornado_excluding_13140_features_3720177259145965027/903_trees_loss_0.0012148821.model",   "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T15.07.29.700_tornado_excluding_13140_features_3720177259145965027/1175_trees_loss_0.001263166.model"),
  ("tornadao_mean_prob_computed_climatology_3hr_1567",           "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T09.10.44.152_tornado_excluding_12264_features_11093869335794226923/1246_trees_loss_0.0011406931.model", "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.41.12.505_tornado_excluding_12264_features_11093869335794226923/1144_trees_loss_0.0012080203.model", "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.41.12.050_tornado_excluding_12264_features_11093869335794226923/1106_trees_loss_0.0012586837.model"),
  ("tornado_full_13831",                                         "TORPROB", "gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-19T10.19.24.875_tornado_climatology_all/440_trees_loss_0.0011318683.model",                                "gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_tornado_climatology_all/676_trees_loss_0.0012007512.model",                                "gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_tornado_climatology_all/538_trees_loss_0.0012588982.model"),
]


function reload_forecasts()
  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred
  global _forecasts_calibrated

  global _forecasts_day_accumulators
  global _forecasts_day

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
      "href_prediction_raw_2022_ablations_models_$(hash(models))"
    )

  # Only used incidentally to determine best blur radii
  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(regular_forecasts(_forecasts), blur_radii)

  grid = _forecasts[1].grid

  # Determined in Train.jl
  # event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado     15                  15                   0.038589306
  # wind        25                  25                   0.115706086
  # hail        15                  25                   0.07425974
  # sig_tornado 25                  35                   0.03381651
  # sig_wind    15                  35                   0.016237844
  # sig_hail    15                  25                   0.015587974

  blur_0mi_grid_is  = Grids.radius_grid_is(grid, 0.0)
  # blur_15mi_grid_is = Grids.radius_grid_is(grid, 15.0)
  # blur_25mi_grid_is = Grids.radius_grid_is(grid, 25.0)
  # blur_35mi_grid_is = Grids.radius_grid_is(grid, 35.0)
  # blur_50mi_grid_is = Grids.radius_grid_is(grid, 50.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_0mi_grid_is, blur_0mi_grid_is), # tornado
  ]

  _forecasts_blurred = PredictionForecasts.blurred(regular_forecasts(_forecasts),  2:35,  blur_grid_is)


  # Calibrating hourly predictions to validation data

  event_to_bins = Dict{String, Vector{Float32}}(
    "tornado_full_13831"     => [0.0009693373,  0.003943406,  0.009779687,  0.021067958, 0.04314823,  1.0],
  )
  event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado_full_13831"     => [[0.9410416,  -0.43989772], [0.9696831, -0.2419776],   [0.9994333,  -0.073430814], [1.094306,   0.3302174],  [1.1247456, 0.4527394]],
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

  _forecasts_calibrated = PredictionForecasts.simple_prediction_forecasts(_forecasts_blurred, hour_models; model_name = "HREF_hour_ablations_severe_probabilities")


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

  _forecasts_day_accumulators, _forecasts_day2_accumulators, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(_forecasts_calibrated, models; module_name = "HREFPredictionAblations")

  # The following was computed in TrainDay.jl

  event_to_0z_day_bins = Dict{String, Vector{Float32}}(
    "tornado_full_13831"     => [0.017401028, 0.057005595, 0.13199422,  1.0],
  )
  event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado_full_13831"     => [[0.93318164, 0.06823707, -0.10806717],  [1.3238057,    -0.114339165, 0.34774518],  [0.6581387, 0.06536053,  -0.60667735]],
  )

  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_0z_day_bins, event_to_0z_day_bins_logistic_coeffs, models; module_name = "HREFPredictionAblations", period_name = "day")


  ()
end

end # module HREFPredictionAblations