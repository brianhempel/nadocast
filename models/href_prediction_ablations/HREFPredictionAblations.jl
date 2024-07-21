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


_forecasts                              = [] # Raw, unblurred predictions
_forecasts_with_blurs_and_forecast_hour = [] # For Train.jl
_forecasts_blurred                      = [] # For downstream combination with other forecasts
_forecasts_calibrated                   = []

_forecasts_day_accumulators   = []
_forecasts_day                = []
_forecasts_day_spc_calibrated = []


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

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


# (event_name, grib2_var_name, gbdt_f2_to_f13, gbdt_f13_to_f24, gbdt_f24_to_f35)
models = [
  ("tornado_mean_58",                                           "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.584_tornado_excluding_13773_features_4410986610349231978/700_trees_loss_0.0012174041.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.26.38.345_tornado_excluding_13773_features_4410986610349231978/707_trees_loss_0.0012815231.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.26.10.688_tornado_excluding_13773_features_4410986610349231978/705_trees_loss_0.0013271595.model"),
  ("tornado_prob_80",                                           "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.550_tornado_excluding_13751_features_16850198170780752800/812_trees_loss_0.0012124014.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T01.27.28.031_tornado_excluding_13751_features_16850198170780752800/526_trees_loss_0.0012738121.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T02.44.38.100_tornado_excluding_13751_features_16850198170780752800/566_trees_loss_0.0013254478.model"),
  ("tornado_mean_prob_138",                                     "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.590_tornado_excluding_13693_features_7893827997600368035/1207_trees_loss_0.0011827772.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.18.52.783_tornado_excluding_13693_features_7893827997600368035/732_trees_loss_0.001246924.model",    (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.19.01.750_tornado_excluding_13693_features_7893827997600368035/390_trees_loss_0.0012985576.model"),
  ("tornado_mean_prob_computed_no_sv_219",                      "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.27.728_tornado_excluding_13612_features_12247408129229835884/779_trees_loss_0.0011766684.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T02.48.31.809_tornado_excluding_13612_features_12247408129229835884/971_trees_loss_0.0012399479.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T03.09.01.898_tornado_excluding_13612_features_12247408129229835884/832_trees_loss_0.0012905387.model"),
  ("tornado_mean_prob_computed_220",                            "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T11.29.03.470_tornado_excluding_13611_features_10698383838099131809/1041_trees_loss_0.0011743465.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T04.29.00.505_tornado_excluding_13611_features_10698383838099131809/882_trees_loss_0.0012400731.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T05.37.18.512_tornado_excluding_13611_features_10698383838099131809/964_trees_loss_0.0012917896.model"),
  ("tornado_mean_prob_computed_partial_climatology_227",        "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T11.53.56.474_tornado_excluding_13604_features_16220645629827272623/1158_trees_loss_0.0011694039.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T06.17.31.380_tornado_excluding_13604_features_16220645629827272623/963_trees_loss_0.0012260334.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T07.10.18.386_tornado_excluding_13604_features_16220645629827272623/994_trees_loss_0.0012774036.model"),
  ("tornado_mean_prob_computed_climatology_253",                "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.03.56.368_tornado_excluding_13578_features_1255749354977721151/1135_trees_loss_0.0011634461.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T07.18.00.712_tornado_excluding_13578_features_1255749354977721151/1207_trees_loss_0.0012247592.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T08.44.56.699_tornado_excluding_13578_features_1255749354977721151/866_trees_loss_0.0012709117.model"),
  ("tornado_mean_prob_computed_climatology_blurs_910",          "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.20.32.745_tornado_excluding_12921_features_7409514120157424229/868_trees_loss_0.0011486912.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T10.52.42.948_tornado_excluding_12921_features_7409514120157424229/859_trees_loss_0.0012130085.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T11.04.58.206_tornado_excluding_12921_features_7409514120157424229/852_trees_loss_0.0012569018.model"),
  ("tornado_mean_prob_computed_climatology_grads_1348",         "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.43.51.679_tornado_excluding_12483_features_2140312678839727709/793_trees_loss_0.0011474881.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T11.24.59.952_tornado_excluding_12483_features_2140312678839727709/855_trees_loss_0.0012191372.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T12.47.02.062_tornado_excluding_12483_features_2140312678839727709/735_trees_loss_0.00126751.model"),
  ("tornado_mean_prob_computed_climatology_blurs_grads_2005",   "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T08.46.15.004_tornado_excluding_11826_features_14576562183048218741/523_trees_loss_0.0011400548.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T16.13.41.561_tornado_excluding_11826_features_14576562183048218741/558_trees_loss_0.0012086717.model",  (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T16.13.15.348_tornado_excluding_11826_features_14576562183048218741/578_trees_loss_0.0012580294.model"),
  ("tornado_mean_prob_computed_climatology_prior_next_hrs_691", "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T12.58.07.439_tornado_excluding_13140_features_3720177259145965027/836_trees_loss_0.0011478734.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-29T14.24.13.566_tornado_excluding_13140_features_3720177259145965027/903_trees_loss_0.0012148821.model",   (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-29T15.07.29.700_tornado_excluding_13140_features_3720177259145965027/1175_trees_loss_0.001263166.model"),
  ("tornado_mean_prob_computed_climatology_3hr_1567",           "TORPROB", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-30T09.10.44.152_tornado_excluding_12264_features_11093869335794226923/1246_trees_loss_0.0011406931.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-28T23.41.12.505_tornado_excluding_12264_features_11093869335794226923/1144_trees_loss_0.0012080203.model", (@__DIR__) * "/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-28T23.41.12.050_tornado_excluding_12264_features_11093869335794226923/1106_trees_loss_0.0012586837.model"),
  ("tornado_full_13831",                                        "TORPROB", (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-09-19T10.19.24.875_tornado_climatology_all/440_trees_loss_0.0011318683.model",       (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_tornado_climatology_all/676_trees_loss_0.0012007512.model",       (@__DIR__) * "/../href_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-09-24T18.25.33.579_tornado_climatology_all/538_trees_loss_0.0012588982.model"),
]


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
      "href_prediction_raw_2022_ablations_models_$(hash(models))"
    )

  # Only used incidentally to determine best blur radii
  # _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(regular_forecasts(_forecasts), blur_radii)

  grid = _forecasts[1].grid

  # Determined in HREF Train.jl
  # event_name  best_blur_radius_f2 best_blur_radius_f38 AU_PR
  # tornado     15                  15                   0.038589306

  # blur_0mi_grid_is  = Grids.radius_grid_is(grid, 0.0)
  blur_15mi_grid_is = Grids.radius_grid_is(grid, 15.0)
  # blur_25mi_grid_is = Grids.radius_grid_is(grid, 25.0)
  # blur_35mi_grid_is = Grids.radius_grid_is(grid, 35.0)
  # blur_50mi_grid_is = Grids.radius_grid_is(grid, 50.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_58
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_prob_80
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_138
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_no_sv_219
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_220
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_partial_climatology_227
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_253
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_blurs_910
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_grads_1348
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_blurs_grads_2005
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_prior_next_hrs_691
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_mean_prob_computed_climatology_3hr_1567
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado_full_13831
  ]

  _forecasts_blurred = PredictionForecasts.blurred(regular_forecasts(_forecasts),  2:35,  blur_grid_is)


  # # Calibrating hourly predictions to validation data

  event_to_bins = Dict{String, Vector{Float32}}(
    "tornado_mean_58"                                           => [0.000958296,  0.0034187553, 0.008049908,  0.016707344, 0.035788268, 1.0],
    "tornado_prob_80"                                           => [0.0010611907, 0.0038441075, 0.009374168,  0.018635446, 0.037465766, 1.0],
    "tornado_mean_prob_138"                                     => [0.0011344727, 0.0040516052, 0.009403912,  0.019440345, 0.04088651,  1.0],
    "tornado_mean_prob_computed_no_sv_219"                      => [0.001112896,  0.0040969327, 0.009968531,  0.020722248, 0.042520937, 1.0],
    "tornado_mean_prob_computed_220"                            => [0.0011188483, 0.0041876957, 0.009798439,  0.019887649, 0.04154461,  1.0],
    "tornado_mean_prob_computed_partial_climatology_227"        => [0.0011116201, 0.004092531,  0.010467994,  0.023395522, 0.047907665, 1.0],
    "tornado_mean_prob_computed_climatology_253"                => [0.001142257,  0.004164435,  0.0104533415, 0.02289575,  0.0471496,   1.0],
    "tornado_mean_prob_computed_climatology_blurs_910"          => [0.001259957,  0.004403745,  0.010849744,  0.022011435, 0.044833306, 1.0],
    "tornado_mean_prob_computed_climatology_grads_1348"         => [0.0012080052, 0.0043216683, 0.010642204,  0.021209253, 0.044587743, 1.0],
    "tornado_mean_prob_computed_climatology_blurs_grads_2005"   => [0.0012621598, 0.004405628,  0.01059343,   0.021830112, 0.046009377, 1.0],
    "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [0.0011541534, 0.004339112,  0.010959722,  0.023348574, 0.04785411,  1.0],
    "tornado_mean_prob_computed_climatology_3hr_1567"           => [0.001206508,  0.004330907,  0.0109639205, 0.023913587, 0.051235575, 1.0],
    "tornado_full_13831"                                        => [0.0012153604, 0.004538168,  0.011742671,  0.023059275, 0.049979284, 1.0],
  )
  event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado_mean_58"                                           => [[1.0608453, 0.34233692], [1.0113722,  0.013388185], [1.0653579, 0.29739413],  [1.0437187,  0.20861574], [1.0588217, 0.27578637]],
    "tornado_prob_80"                                           => [[1.0096253, -0.0469623], [0.9850988,  -0.17768726], [1.0664655, 0.24821338],  [0.9513827,  -0.2362276], [1.0854785, 0.2633485]],
    "tornado_mean_prob_138"                                     => [[1.0230768, 0.09093034], [1.0292343,  0.12141731],  [1.0123438, 0.037362378], [1.0350122,  0.1281267],  [1.0963296, 0.3398288]],
    "tornado_mean_prob_computed_no_sv_219"                      => [[1.0354019, 0.1911897],  [1.0219095,  0.07485228],  [1.0493485, 0.2075882],   [1.0940257,  0.38835412], [1.0213909, 0.12248391]],
    "tornado_mean_prob_computed_220"                            => [[1.0302787, 0.14280733], [1.0681692,  0.34719107],  [1.0482181, 0.24779604],  [1.0291348,  0.15958983], [0.9713004, -0.044102136]],
    "tornado_mean_prob_computed_partial_climatology_227"        => [[1.0562521, 0.38805947], [0.92319447, -0.45271772], [0.9888554, -0.13014041], [1.1344755,  0.48079926], [1.0812244, 0.30014265]],
    "tornado_mean_prob_computed_climatology_253"                => [[1.0602976, 0.41283026], [0.92125416, -0.4563326],  [0.9710596, -0.20777129], [1.1646894,  0.60834074], [1.1520658, 0.5673795]],
    "tornado_mean_prob_computed_climatology_blurs_910"          => [[1.065771,  0.47239247], [0.9286899,  -0.38221243], [1.0567164, 0.26145768],  [0.9913568,  0.01256458], [1.0834966, 0.3218238]],
    "tornado_mean_prob_computed_climatology_grads_1348"         => [[1.0203846, 0.15487157], [0.93945754, -0.35369343], [1.1307863, 0.61256605],  [1.0428475,  0.24978055], [1.0871354, 0.40001607]],
    "tornado_mean_prob_computed_climatology_blurs_grads_2005"   => [[1.0343003, 0.2688026],  [0.95909494, -0.18902206], [1.0664421, 0.33880964],  [1.0681071,  0.33059332], [1.078756,  0.37049353]],
    "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [[1.0457013, 0.29932234], [0.91751766, -0.48559842], [1.0170314, 0.0401797],   [1.0982827,  0.39915916], [1.1686069, 0.6304107]],
    "tornado_mean_prob_computed_climatology_3hr_1567"           => [[1.0639268, 0.45781243], [0.9226182,  -0.4165027],  [0.9799438, -0.11478417], [1.0525097,  0.19820513], [1.2263151, 0.7710571]],
    "tornado_full_13831"                                        => [[1.0710595, 0.5394862],  [0.89414734, -0.5616081],  [1.108447,  0.50613326],  [0.83939207, -0.5758451], [1.2171175, 0.7334359]],
  )

  # Returns array of (model_name, var_name, predict)
  function make_models(event_to_bins, event_to_bins_logistic_coeffs)
    ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

    map(1:length(models)) do model_i
      model_name, var_name, _, _, _ = models[model_i]

      predict(forecasts, data) = begin
        href_ŷs = @view data[:,model_i]

        out = Array{Float32}(undef, length(href_ŷs))

        bin_maxes            = event_to_bins[model_name]
        bins_logistic_coeffs = event_to_bins_logistic_coeffs[model_name]

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

  hour_models = make_models(event_to_bins, event_to_bins_logistic_coeffs)

  _forecasts_calibrated = PredictionForecasts.simple_prediction_forecasts(_forecasts_blurred, hour_models; model_name = "HREF_hour_ablations_severe_probabilities")


  # Day & Four-hourly forecasts

  # 1. Try both independent events total prob and max hourly prob as the main descriminator
  # 2. bin predictions into 4 bins of equal weight of positive labels
  # 3. combine bin-pairs (overlapping, 3 bins total)
  # 4. train a logistic regression for each bin,
  #   σ(a1*logit(independent events total prob) +
  #     a2*logit(max hourly prob) +
  #     b)
  # 5. prediction is weighted mean of the two overlapping logistic models
  # 6. should thereby be absolutely calibrated (check)
  # 7. calibrate to SPC thresholds (linear interpolation)

  _forecasts_day_accumulators, _forecasts_day2_accumulators, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(_forecasts_calibrated, models, 2; module_name = "HREFPredictionAblations")

  # The following was computed in TrainDay.jl
  event_to_day_bins = Dict{String, Vector{Float32}}(
    "tornado_mean_58"                                           => [0.017304773, 0.0553223,   0.13471735, 1.0],
    "tornado_prob_80"                                           => [0.019274753, 0.059863195, 0.13051617, 1.0],
    "tornado_mean_prob_138"                                     => [0.019996958, 0.06293694,  0.14385402, 1.0],
    "tornado_mean_prob_computed_no_sv_219"                      => [0.019619932, 0.06317351,  0.14767814, 1.0],
    "tornado_mean_prob_computed_220"                            => [0.020045973, 0.06332173,  0.14715679, 1.0],
    "tornado_mean_prob_computed_partial_climatology_227"        => [0.019988786, 0.061916392, 0.16038308, 1.0],
    "tornado_mean_prob_computed_climatology_253"                => [0.02144064,  0.063346006, 0.16949469, 1.0],
    "tornado_mean_prob_computed_climatology_blurs_910"          => [0.02166239,  0.06790343,  0.16562855, 1.0],
    "tornado_mean_prob_computed_climatology_grads_1348"         => [0.020726241, 0.06985498,  0.16851029, 1.0],
    "tornado_mean_prob_computed_climatology_blurs_grads_2005"   => [0.021495968, 0.071746476, 0.16894186, 1.0],
    "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [0.020654399, 0.06460478,  0.16610569, 1.0],
    "tornado_mean_prob_computed_climatology_3hr_1567"           => [0.021308538, 0.067999676, 0.17814146, 1.0],
    "tornado_full_13831"                                        => [0.021043906, 0.074019335, 0.17095083, 1.0],
  )
  event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado_mean_58"                                           => [[0.6559432,  0.33183938,  0.2690549],    [0.6666154,  0.36046487,   0.4614278],     [0.42497587, 0.44417486, 0.20855717]],
    "tornado_prob_80"                                           => [[0.8796285,  0.16921686,  0.22449368],   [0.8071765,  0.3629977,    0.88375133],    [0.51591927, 0.29846162, -0.097147524]],
    "tornado_mean_prob_138"                                     => [[0.93436664, 0.07926717,  -0.026851058], [0.9042307,  0.17261904,   0.32986057],    [0.46952277, 0.3558928,  0.015089741]],
    "tornado_mean_prob_computed_no_sv_219"                      => [[0.8970471,  0.12673378,  0.10686361],   [0.77730256, 0.24956441,   0.30919433],    [0.38325134, 0.41652393, 0.074524485]],
    "tornado_mean_prob_computed_220"                            => [[0.9081216,  0.107832894, 0.050864797],  [0.800826,   0.23256853,   0.31022668],    [0.40040252, 0.4055984,  0.067884356]],
    "tornado_mean_prob_computed_partial_climatology_227"        => [[0.9746325,  0.05395953,  0.059440296],  [0.8418499,  0.16475138,   0.1517096],     [0.48570332, 0.34678283, -0.001083027]],
    "tornado_mean_prob_computed_climatology_253"                => [[0.982298,   0.05099982,  0.06179544],   [0.80724055, 0.13905986,   -0.0778435],    [0.58624,    0.255771,   -0.13458002]],
    "tornado_mean_prob_computed_climatology_blurs_910"          => [[0.98352575, 0.03981221,  0.010008344],  [0.9775675,  0.045042068,  -0.0055927183], [0.57631963, 0.20392773, -0.28303716]],
    "tornado_mean_prob_computed_climatology_grads_1348"         => [[0.94020444, 0.0850467,   0.051697694],  [1.1303655,  -0.08044045,  -0.13138889],   [0.63580066, 0.1437665,  -0.37667832]],
    "tornado_mean_prob_computed_climatology_blurs_grads_2005"   => [[0.95560277, 0.05098174,  -0.06451891],  [1.1089742,  -0.043572877, -0.017801635],  [0.64432305, 0.10341309, -0.47638172]],
    "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [[0.81858176, 0.18513452,  0.15438604],   [0.6977342,  0.27534917,   0.21286853],    [0.5196121,  0.31482816, -0.040216677]],
    "tornado_mean_prob_computed_climatology_3hr_1567"           => [[0.96216094, 0.034555722, -0.12364115],  [1.0485787,  -0.011567661, -0.047783367],  [0.6019041,  0.21778609, -0.17962365]],
    "tornado_full_13831"                                        => [[0.95958227, 0.04161413,  -0.10651286],  [1.2272763,  -0.15624464,  -0.18067063],   [0.5964124,  0.17200926, -0.3083448]],
    "tornado"                                                   => [[0.95958227, 0.04161413,  -0.10651286],  [1.2272763,  -0.15624464,  -0.18067063],   [0.5964124,  0.17200926, -0.3083448]],

  )

  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "HREFPredictionAblations", period_name = "day")

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado_mean_58"                                           => [(0.02, 0.018461227), (0.05, 0.06462288),  (0.1, 0.15732765), (0.15, 0.29961205), (0.3, 0.42596245), (0.45, 0.5749607)],
    "tornado_prob_80"                                           => [(0.02, 0.017969131), (0.05, 0.072660446), (0.1, 0.16031837), (0.15, 0.27256584), (0.3, 0.35550117), (0.45, 0.43262672)],
    "tornado_mean_prob_138"                                     => [(0.02, 0.018060684), (0.05, 0.068006516), (0.1, 0.16544151), (0.15, 0.30457115), (0.3, 0.4305973),  (0.45, 0.54815865)],
    "tornado_mean_prob_computed_no_sv_219"                      => [(0.02, 0.017709732), (0.05, 0.06767845),  (0.1, 0.16807747), (0.15, 0.29662895), (0.3, 0.40353203), (0.45, 0.47664833)],
    "tornado_mean_prob_computed_220"                            => [(0.02, 0.017793655), (0.05, 0.068590164), (0.1, 0.16698647), (0.15, 0.29432106), (0.3, 0.40807152), (0.45, 0.5101414)],
    "tornado_mean_prob_computed_partial_climatology_227"        => [(0.02, 0.017900467), (0.05, 0.065675735), (0.1, 0.17772484), (0.15, 0.3134899),  (0.3, 0.45659447), (0.45, 0.56458473)],
    "tornado_mean_prob_computed_climatology_253"                => [(0.02, 0.018064499), (0.05, 0.06471443),  (0.1, 0.17611122), (0.15, 0.32598686), (0.3, 0.47203255), (0.45, 0.60515404)],
    "tornado_mean_prob_computed_climatology_blurs_910"          => [(0.02, 0.01799202),  (0.05, 0.06854057),  (0.1, 0.17239952), (0.15, 0.29992485), (0.3, 0.46686745), (0.45, 0.6732578)],
    "tornado_mean_prob_computed_climatology_grads_1348"         => [(0.02, 0.01742363),  (0.05, 0.06785774),  (0.1, 0.17230797), (0.15, 0.3162861),  (0.3, 0.45212746), (0.45, 0.59908485)],
    "tornado_mean_prob_computed_climatology_blurs_grads_2005"   => [(0.02, 0.017370224), (0.05, 0.06944466),  (0.1, 0.17526436), (0.15, 0.31707573), (0.3, 0.45667458), (0.45, 0.6287174)],
    "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [(0.02, 0.017599106), (0.05, 0.06641579),  (0.1, 0.17908287), (0.15, 0.3228054),  (0.3, 0.49178123), (0.45, 0.6537876)],
    "tornado_mean_prob_computed_climatology_3hr_1567"           => [(0.02, 0.017438889), (0.05, 0.0646534),   (0.1, 0.18551826), (0.15, 0.3344822),  (0.3, 0.50450325), (0.45, 0.6276798)],
    "tornado_full_13831"                                        => [(0.02, 0.016950607), (0.05, 0.06830406),  (0.1, 0.17817497), (0.15, 0.3255825),  (0.3, 0.4591999),  (0.45, 0.60011864)],
  )

  # ensure ordered the same as the features in the data
  calibrations =
    map(models) do (model_name, _, _)
      spc_calibrations[model_name]
    end

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "HREFPredictionAblations_day_severe_probabilities_calibrated_to_SPC_thresholds")

  ()
end

end # module HREFPredictionAblations