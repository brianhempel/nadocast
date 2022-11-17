module HREFPredictionAblations2

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


_forecasts                              = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred                      = [] # For downstream combination with other forecasts
_forecasts_calibrated                   = []

_forecasts_day_accumulators   = []
_forecasts_day                = []
_forecasts_day_spc_calibrated = []



blur_radii = []


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

function forecasts_day_spc_calibrated()
  if isempty(_forecasts_day_spc_calibrated)
    reload_forecasts()
    _forecasts_day_spc_calibrated
  else
    _forecasts_day_spc_calibrated
  end
end

# best hyperparameters:
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 46400.0)  gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.20.32.745_tornado_excluding_12921_features_7409514120157424229/868_trees_loss_0.0011486912.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 10000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 1000.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T10.52.42.948_tornado_excluding_12921_features_7409514120157424229/859_trees_loss_0.0012130085.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 1.8,  :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 10000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 2150.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T11.04.58.206_tornado_excluding_12921_features_7409514120157424229/852_trees_loss_0.0012569018.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-10-27T23.22.17.325_wind_only_910_features_136173177384287588/1251_trees_loss_0.006788157.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 100.0,   :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-10-28T16.20.28.372_wind_only_910_features_136173177384287588/1041_trees_loss_0.0071827183.model
# Dict{Symbol, Real}(:max_depth => 7, :max_delta_score => 1.8,  :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 100.0,   :normalize_second_opinion => false, :feature_fraction => 0.18,  :second_opinion_weight => 0.1,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 2150.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-10-31T15.49.28.012_wind_only_910_features_136173177384287588/841_trees_loss_0.007530943.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 1.8,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 4640.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-10-29T16.43.14.158_hail_only_910_features_136173177384287588/1056_trees_loss_0.00339138.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 10000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 4640.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-10-30T03.32.11.507_hail_only_910_features_136173177384287588/1248_trees_loss_0.003630137.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-10-31T15.50.17.900_hail_only_910_features_136173177384287588/1235_trees_loss_0.003855499.model
# trained on data before 20200523:
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_tornado_only_910_features_136173177384287588/780_trees_loss_0.001148528.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 15, :l2_regularization => 0.001,   :normalize_second_opinion => false, :feature_fraction => 0.18,  :second_opinion_weight => 0.33, :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100.0)    gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_tornado_only_910_features_136173177384287588/443_trees_loss_0.0012150184.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 10000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 4640.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_tornado_only_910_features_136173177384287588/814_trees_loss_0.0012562973.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_wind_only_910_features_136173177384287588/1251_trees_loss_0.006788157.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 100.0,   :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_wind_only_910_features_136173177384287588/1041_trees_loss_0.0071827183.model
# Dict{Symbol, Real}(:max_depth => 7, :max_delta_score => 1.8,  :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 100.0,   :normalize_second_opinion => false, :feature_fraction => 0.18,  :second_opinion_weight => 0.1,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 2150.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_wind_only_910_features_136173177384287588/841_trees_loss_0.007530943.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 1.8,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 4640.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_hail_only_910_features_136173177384287588/1056_trees_loss_0.00339138.model
# Dict{Symbol, Real}(:max_depth => 6, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 10000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 4640.0)   gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_hail_only_910_features_136173177384287588/1248_trees_loss_0.003630137.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0,  :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,  :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215000.0) gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_hail_only_910_features_136173177384287588/1235_trees_loss_0.003855499.model


# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
models = [
  ("tornado_mean_prob_computed_climatology_blurs_910",                 "TORPROB",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.20.32.745_tornado_excluding_12921_features_7409514120157424229/868_trees_loss_0.0011486912.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T10.52.42.948_tornado_excluding_12921_features_7409514120157424229/859_trees_loss_0.0012130085.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T11.04.58.206_tornado_excluding_12921_features_7409514120157424229/852_trees_loss_0.0012569018.model"),
  ("wind_mean_prob_computed_climatology_blurs_910",                    "WINDPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-10-27T23.22.17.325_wind_only_910_features_136173177384287588/1251_trees_loss_0.006788157.model",            (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-10-28T16.20.28.372_wind_only_910_features_136173177384287588/1041_trees_loss_0.0071827183.model",           (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-10-31T15.49.28.012_wind_only_910_features_136173177384287588/841_trees_loss_0.007530943.model"),
  ("hail_mean_prob_computed_climatology_blurs_910",                    "HAILPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-10-29T16.43.14.158_hail_only_910_features_136173177384287588/1056_trees_loss_0.00339138.model",             (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-10-30T03.32.11.507_hail_only_910_features_136173177384287588/1248_trees_loss_0.003630137.model",            (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-10-31T15.50.17.900_hail_only_910_features_136173177384287588/1235_trees_loss_0.003855499.model"),
  ("tornado_mean_prob_computed_climatology_blurs_910_before_20200523", "TORPROB",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_tornado_only_910_features_136173177384287588/780_trees_loss_0.001148528.model",          (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_tornado_only_910_features_136173177384287588/443_trees_loss_0.0012150184.model",         (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_tornado_only_910_features_136173177384287588/814_trees_loss_0.0012562973.model"),
  ("wind_mean_prob_computed_climatology_blurs_910_before_20200523",    "WINDPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_wind_only_910_features_136173177384287588/1251_trees_loss_0.006788157.model",            (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_wind_only_910_features_136173177384287588/1041_trees_loss_0.0071827183.model",           (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_wind_only_910_features_136173177384287588/841_trees_loss_0.007530943.model"),
  ("hail_mean_prob_computed_climatology_blurs_910_before_20200523",    "HAILPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-11-01T18.52.55.342_hail_only_910_features_136173177384287588/1056_trees_loss_0.00339138.model",             (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-11-01T18.55.14.777_hail_only_910_features_136173177384287588/1248_trees_loss_0.003630137.model",            (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-11-03T17.32.21.260_hail_only_910_features_136173177384287588/1235_trees_loss_0.003855499.model"),
  ("tornado_full_13831",                                               "TORPROB",  (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-19T10.19.24.875_tornado_climatology_all/440_trees_loss_0.0011318683.model",     (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_tornado_climatology_all/676_trees_loss_0.0012007512.model",     (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_tornado_climatology_all/538_trees_loss_0.0012588982.model"),
  ("wind_full_13831",                                                  "WINDPROB", (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-21T20.38.44.560_wind_climatology_all/1189_trees_loss_0.006638963.model",        (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_wind_climatology_all/1251_trees_loss_0.007068191.model",        (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_wind_climatology_all/877_trees_loss_0.0074603.model"),
  ("hail_full_13831",                                                  "HAILPROB", (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-17T18.47.33.015_hail_climatology_all/1163_trees_loss_0.0033772641.model",       (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-20T18.33.36.099_hail_climatology_all/1057_trees_loss_0.0036187447.model",       (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.35.163_hail_climatology_all/1249_trees_loss_0.0038463715.model"),
]

# Returns array of (model_name, var_name, predict)
# similar code is duplicated across various models but i don't want to introduce bugs by refactoring rn
function make_calibrated_hourly_models(model_name_to_bins, model_name_to_bins_logistic_coeffs)
  σ(x) = 1.0f0 / (1.0f0 + exp(-x))
  logit(p) = log(p / (1f0 - p))
  ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

  map(1:length(models)) do model_i
    model_name, var_name, _, _, _ = models[model_i]

    predict(_forecast, data) = begin
      href_ŷs = @view data[:,model_i]

      out = Array{Float32}(undef, length(href_ŷs))

      bin_maxes            = model_name_to_bins[model_name]
      bins_logistic_coeffs = model_name_to_bins_logistic_coeffs[model_name]

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

    (model_name, var_name, predict)
  end
end


function reload_forecasts()
  global _forecasts
  global _forecasts_with_blurs_and_forecast_hour
  global _forecasts_blurred
  global _forecasts_calibrated

  global _forecasts_day_accumulators
  global _forecasts_day
  global _forecasts_day_spc_calibrated

  _forecasts = []

  href_forecasts = HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()

  predictors = map(models) do (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
    predict_f2_to_f13  = MemoryConstrainedTreeBoosting.load_unbinned_predictor(gbdt_f2_to_f13)
    predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor(gbdt_f13_to_f24)
    predict_f24_to_f35 = MemoryConstrainedTreeBoosting.load_unbinned_predictor(gbdt_f24_to_f35)

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
      "href_prediction_raw_2022_ablations2_models_$(hash(models))"
    )

  # Only used incidentally to determine best blur radii
  # _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(regular_forecasts(_forecasts), blur_radii)

  grid = _forecasts[1].grid

  # Determined in HREF Train.jl
  # event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado     15                  15                   0.038589306
  # wind        25                  25                   0.115706086
  # hail        15                  25                   0.07425974
  # sig_tornado 25                  35                   0.03381651
  # sig_wind    15                  35                   0.016237844
  # sig_hail    15                  25                   0.015587974

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

  _forecasts_blurred = PredictionForecasts.blurred(regular_forecasts(_forecasts),  2:35,  blur_grid_is)


  # Calibrating hourly predictions to validation data

  model_name_to_bins = Dict{String, Vector{Float32}}("wind_mean_prob_computed_climatology_blurs_910" => [0.007349782, 0.019179698, 0.035851445, 0.06126755, 0.107910745, 1.0], "hail_mean_prob_computed_climatology_blurs_910" => [0.0034476055, 0.009904056, 0.020123107, 0.0378312, 0.07595588, 1.0], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.007349782, 0.019179698, 0.035851445, 0.06126755, 0.107910745, 1.0], "tornado_mean_prob_computed_climatology_blurs_910" => [0.001259957, 0.004403745, 0.010849744, 0.022011435, 0.044833306, 1.0], "tornado_full_13831" => [0.0012153604, 0.004538168, 0.011742671, 0.023059275, 0.049979284, 1.0], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.0034476055, 0.009904056, 0.020123107, 0.0378312, 0.07595588, 1.0], "wind_full_13831" => [0.007898662, 0.020139106, 0.037745386, 0.06497206, 0.115874514, 1.0], "hail_full_13831" => [0.0033965781, 0.00951148, 0.0200999, 0.037743744, 0.07645232, 1.0], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.001294466, 0.004569119, 0.010611137, 0.021681689, 0.043206826, 1.0])

  model_name_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [[1.0957842, 0.44560102], [1.1037341, 0.462977], [1.0819606, 0.3897019], [1.0832871, 0.39767647], [0.85362035, -0.15877303]], "hail_mean_prob_computed_climatology_blurs_910" => [[1.0478816, 0.3216053], [1.0098401, 0.10208285], [0.98269755, 0.0008943792], [1.0002121, 0.06374693], [1.0119866, 0.10483526]], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0957842, 0.44560102], [1.1037341, 0.462977], [1.0819606, 0.3897019], [1.0832871, 0.39767647], [0.85362035, -0.15877303]], "tornado_mean_prob_computed_climatology_blurs_910" => [[1.065771, 0.47239247], [0.9286899, -0.38221243], [1.0567164, 0.26145768], [0.9913568, 0.01256458], [1.0834966, 0.3218238]], "tornado_full_13831" => [[1.0710595, 0.5394862], [0.89414734, -0.5616081], [1.108447, 0.50613326], [0.83939207, -0.5758451], [1.2171175, 0.7334359]], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0478816, 0.3216053], [1.0098401, 0.10208285], [0.98269755, 0.0008943792], [1.0002121, 0.06374693], [1.0119866, 0.10483526]], "wind_full_13831" => [[1.0976788, 0.44982857], [1.1166184, 0.5214092], [1.0803565, 0.39499816], [1.0272967, 0.24607334], [0.9264786, 0.009931214]], "hail_full_13831" => [[1.0600259, 0.43388337], [0.96319646, -0.097637914], [1.0792756, 0.39443073], [0.9629313, -0.01864577], [1.0167043, 0.13172606]], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.039819, 0.28398132], [0.9637524, -0.18635054], [1.0136148, 0.06931454], [1.114865, 0.49855673], [1.0013322, 0.116965175]])

  hour_models = make_calibrated_hourly_models(model_name_to_bins, model_name_to_bins_logistic_coeffs)

  _forecasts_calibrated = PredictionForecasts.simple_prediction_forecasts(_forecasts_blurred, hour_models; model_name = "HREF_hour_ablations2_severe_probabilities")


  # # Day & Four-hourly forecasts

  # # 1. Try both independent events total prob and max hourly prob as the main descriminator
  # # 2. bin predictions into 4 bins of equal weight of positive labels
  # # 3. combine bin-pairs (overlapping, 3 bins total)
  # # 4. train a logistic regression for each bin,
  # #   σ(a1*logit(independent events total prob) +
  # #     a2*logit(max hourly prob) +
  # #     b)
  # # 5. prediction is weighted mean of the two overlapping logistic models
  # # 6. should thereby be absolutely calibrated (check)
  # # 7. calibrate to SPC thresholds (linear interpolation)

  _forecasts_day_accumulators, _forecasts_day2_accumulators, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(_forecasts_calibrated, models; module_name = "HREFPredictionAblations2")

  # # The following was computed in TrainDay.jl
  model_name_to_day_bins = Dict{String, Vector{Float32}}("wind_mean_prob_computed_climatology_blurs_910" => [0.115368165, 0.25932494, 0.43146545, 1.0], "hail_mean_prob_computed_climatology_blurs_910" => [0.06547251, 0.15664904, 0.29551116, 1.0], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.115368165, 0.25932494, 0.43146545, 1.0], "tornado_mean_prob_computed_climatology_blurs_910" => [0.02166239, 0.06790343, 0.16562855, 1.0], "tornado_full_13831" => [0.021043906, 0.074019335, 0.17095083, 1.0], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.06547251, 0.15664904, 0.29551116, 1.0], "wind_full_13831" => [0.12129908, 0.27009565, 0.4460996, 1.0], "hail_full_13831" => [0.066225864, 0.15690458, 0.29827937, 1.0], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [0.022665959, 0.06817255, 0.1642723, 1.0])

  model_name_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [[0.9183711, 0.11854008, 0.020747136], [1.0090979, 0.052492317, -0.043933667], [0.769292, 0.13937852, -0.011131755]], "hail_mean_prob_computed_climatology_blurs_910" => [[0.9237591, 0.13382848, 0.12147692], [0.9564363, 0.047680803, -0.16887067], [0.98554933, -0.08279941, -0.4769262]], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [[0.9183711, 0.11854008, 0.020747136], [1.0090979, 0.052492317, -0.043933667], [0.769292, 0.13937852, -0.011131755]], "tornado_mean_prob_computed_climatology_blurs_910" => [[0.98352575, 0.03981221, 0.010008344], [0.9775675, 0.045042068, -0.0055927183], [0.57631963, 0.20392773, -0.28303716]], "tornado_full_13831" => [[0.95958227, 0.04161413, -0.10651286], [1.2272763, -0.15624464, -0.18067063], [0.5964124, 0.17200926, -0.3083448]], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [[0.9237591, 0.13382848, 0.12147692], [0.9564363, 0.047680803, -0.16887067], [0.98554933, -0.08279941, -0.4769262]], "wind_full_13831" => [[1.0433985, -0.0069722184, -0.16421609], [1.1460801, -0.10947033, -0.3195157], [0.9224374, -0.005079186, -0.24423301]], "hail_full_13831" => [[1.0235776, 0.02489767, -0.070147164], [1.1081746, -0.088940814, -0.34076628], [1.1614795, -0.24988303, -0.7304756]], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [[1.0136324, -0.0008003565, -0.1033998], [1.0090433, 0.026480297, 0.0030092064], [0.5625795, 0.22864038, -0.21183096]])

  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, model_name_to_day_bins, model_name_to_day_bins_logistic_coeffs, models; module_name = "HREFPredictionAblations2", period_name = "day")

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}("wind_mean_prob_computed_climatology_blurs_910" => [(0.05, 0.04979515), (0.15, 0.21516609), (0.3, 0.4797535), (0.45, 0.70975685)], "hail_mean_prob_computed_climatology_blurs_910" => [(0.05, 0.02939415), (0.15, 0.12362099), (0.3, 0.367239), (0.45, 0.6290493)], "wind_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.05, 0.04979515), (0.15, 0.21516609), (0.3, 0.4797535), (0.45, 0.70975685)], "tornado_mean_prob_computed_climatology_blurs_910" => [(0.02, 0.01799202), (0.05, 0.06854057), (0.1, 0.17239952), (0.15, 0.29992485), (0.3, 0.46686745), (0.45, 0.6732578)], "tornado_full_13831" => [(0.02, 0.016950607), (0.05, 0.06830406), (0.1, 0.17817497), (0.15, 0.3255825), (0.3, 0.4591999), (0.45, 0.60011864)], "hail_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.05, 0.02939415), (0.15, 0.12362099), (0.3, 0.367239), (0.45, 0.6290493)], "wind_full_13831" => [(0.05, 0.047945023), (0.15, 0.21813774), (0.3, 0.49361992), (0.45, 0.7422848)], "hail_full_13831" => [(0.05, 0.029951096), (0.15, 0.123464584), (0.3, 0.3763752), (0.45, 0.6608143)], "tornado_mean_prob_computed_climatology_blurs_910_before_20200523" => [(0.02, 0.018140793), (0.05, 0.06867409), (0.1, 0.17149925), (0.15, 0.29029655), (0.3, 0.47507286), (0.45, 0.70194054)])

  # ensure ordered the same as the features in the data
  calibrations =
    map(models) do (model_name, _, _)
      spc_calibrations[model_name]
    end

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "HREFPredictionAblations2_day_severe_probabilities_calibrated_to_SPC_thresholds")

  ()
end

end # module HREFPredictionAblations2