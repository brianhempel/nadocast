module HREFPrediction2024

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


# we only trained out to f36, but with some assumptions we can try to go out to f48 even though I don't have an HREF archive out to f48 to train against
regular_forecasts(forecasts)  = filter(f -> f.forecast_hour in 1:36,  forecasts)
extended_forecasts(forecasts) = filter(f -> f.forecast_hour in 37:48,  forecasts)

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


# (event_name, grib2_var_name, gbdt_f1_to_f12, gbdt_f13_to_f24, gbdt_f25_to_f36)
models = [
  ("tornado", "TORPROB", "gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_tornado/886_trees_loss_0.000570307.model",
                         "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_tornado/986_trees_loss_0.00061401847.model",
                         "gbdt_2024-2005_features_f25-36_2024-07-02T04.25.59.068_tornado/1217_trees_loss_0.0006426961.model"
  ),
  ("wind", "WINDPROB", "gbdt_2024-2005_features_f1-12_2024-06-25T14.35.47.253_wind/1210_trees_loss_0.0036398584.model",
                       "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_wind/1226_trees_loss_0.0039208783.model",
                       "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.48.183_wind/1251_trees_loss_0.0041529043.model"
  ),
  ("wind_adj", "WINDPROB", "gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_wind_adj/1142_trees_loss_0.0013586894.model",
                           "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_wind_adj/1250_trees_loss_0.00144605.model",
                           "gbdt_2024-2005_features_f25-36_2024-07-09T08.13.32.361_wind_adj/1008_trees_loss_0.0015145774.model" # not gbdt_2024-2005_features_f25-36_2024-07-03T07.38.48.183_wind_adj/989_trees_loss_0.0015146342.model
  ),
  ("hail", "HAILPROB", "gbdt_2024-2005_features_f1-12_2024-06-13T07.09.03.411_hail/1245_trees_loss_0.0018700788.model",
                       "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_hail/1250_trees_loss_0.0020206557.model",
                       "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_hail/1249_trees_loss_0.0021328537.model"
  ),
  ("sig_tornado", "STORPROB", "gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_sig_tornado/864_trees_loss_8.5906206e-5.model",
                              "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_tornado/656_trees_loss_9.254449e-5.model",
                              "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_sig_tornado/639_trees_loss_9.4197974e-5.model"
  ),
  ("sig_wind", "SWINDPRO", "gbdt_2024-2005_features_f1-12_2024-06-25T14.35.47.253_sig_wind/534_trees_loss_0.00051885995.model",
                           "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_wind/879_trees_loss_0.0005461219.model",
                           "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.48.183_sig_wind/891_trees_loss_0.0005720487.model"
  ),
  ("sig_wind_adj", "SWINDPRO", "gbdt_2024-2005_features_f1-12_2024-06-25T14.35.47.253_sig_wind_adj/571_trees_loss_0.00021140957.model",
                               "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_wind_adj/607_trees_loss_0.0002206263.model",
                               "gbdt_2024-2005_features_f25-36_2024-07-07T21.16.47.443_sig_wind_adj/369_trees_loss_0.00022711804.model" # not gbdt_2024-2005_features_f25-36_2024-07-03T07.38.48.183_sig_wind_adj/480_trees_loss_0.00022736519.model
  ),
  ("sig_hail", "SHAILPRO", "gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_sig_hail/953_trees_loss_0.0003080716.model",
                           "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_hail/422_trees_loss_0.00033432615.model",
                           "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_sig_hail/419_trees_loss_0.0003533024.model" # not gbdt_2024-2005_features_f25-36_2024-07-09T08.13.32.361_sig_hail/443_trees_loss_0.0003533743.model
  ),
  ("tornado_life_risk", "TORPROB", "gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_tornado_life_risk/700_trees_loss_2.3796445e-5.model",
                                   "gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_tornado_life_risk/444_trees_loss_2.5661777e-5.model",
                                   "gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_tornado_life_risk/773_trees_loss_2.4938929e-5.model"
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
  [ ("tornado",           "TORPROB",  "tornado")
  , ("wind",              "WINDPROB", "wind")
  , ("wind_adj",          "WINDPROB", "wind_adj")
  , ("hail",              "HAILPROB", "hail")
  , ("sig_tornado",       "STORPRO",  "sig_tornado")
  , ("sig_wind",          "SWINDPRO", "sig_wind")
  , ("sig_wind_adj",      "SWINDPRO", "sig_wind_adj")
  , ("sig_hail",          "SHAILPRO", "sig_hail")
  , ("sig_tornado",       "STORPRO",  "sig_tornado_gated_by_tornado")
  , ("sig_wind",          "SWINDPRO", "sig_wind_gated_by_wind")
  , ("sig_wind_adj",      "SWINDPRO", "sig_wind_adj_gated_by_wind_adj")
  , ("sig_hail",          "SHAILPRO", "sig_hail_gated_by_hail")
  , ("tornado_life_risk", "TORPROB",  "tornado_life_risk")
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

  href_forecasts = HREF.feature_engineered_forecasts_with_climatology()

  predictors = map(models) do (event_name, grib2_var_name, gbdt_f1_to_f12, gbdt_f13_to_f24, gbdt_f25_to_f36)
    predict_f1_to_f12  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f1_to_f12)
    predict_f13_to_f24 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f13_to_f24)
    predict_f25_to_f36 = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/../href_mid_2018_forward/" * gbdt_f25_to_f36)

    predict(forecast, data) = begin
      if forecast.forecast_hour in 25:48 # For f37-f48, use the f25-f36 GBDT, but we will blur more below
        predict_f25_to_f36(data)
      elseif forecast.forecast_hour in 13:24
        predict_f13_to_f24(data)
      elseif forecast.forecast_hour in 1:12
        predict_f1_to_f12(data)
      else
        error("HREF forecast hour $(forecast.forecast_hour) not in 1:48")
      end
    end

    (event_name, grib2_var_name, predict)
  end

  # Don't forget to clear the cache during development.
  # rm -r lib/computation_cache/cached_forecasts/href_prediction_raw_2021_models
  _forecasts =
    ForecastCombinators.disk_cache_forecasts(
      PredictionForecasts.simple_prediction_forecasts(href_forecasts, predictors),
      "href_prediction_raw_2024_models_$(hash(models))"
    )

  # Only used incidentally to determine best blur radii
  _forecasts_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(regular_forecasts(_forecasts), blur_radii)

  grid = _forecasts[1].grid

  # Determined in Train2024.jl

  # event_name        best_blur_radius_f1 best_blur_radius_f36 AU_PR
  # tornado           15                  15                   0.044846594
  # wind              15                  35                   0.12939164
  # wind_adj          25                  25                   0.08271261
  # hail              15                  25                   0.09276263
  # sig_tornado       15                  15                   0.031650614
  # sig_wind          15                  70                   0.026028465
  # sig_wind_adj      35                  50                   0.017637556
  # sig_hail          15                  25                   0.023982298
  # tornado_life_risk 0                   15                   0.006764876

  blur_0mi_grid_is   = Grids.radius_grid_is(grid, 0.0)
  blur_15mi_grid_is  = Grids.radius_grid_is(grid, 15.0)
  blur_25mi_grid_is  = Grids.radius_grid_is(grid, 25.0)
  blur_35mi_grid_is  = Grids.radius_grid_is(grid, 35.0)
  blur_50mi_grid_is  = Grids.radius_grid_is(grid, 50.0)
  blur_70mi_grid_is  = Grids.radius_grid_is(grid, 70.0)
  blur_100mi_grid_is = Grids.radius_grid_is(grid, 100.0)

  # Needs to be the same order as models
  blur_grid_is = [
    (blur_15mi_grid_is, blur_15mi_grid_is), # tornado
    (blur_15mi_grid_is, blur_35mi_grid_is), # wind
    (blur_25mi_grid_is, blur_25mi_grid_is), # wind_adj
    (blur_15mi_grid_is, blur_25mi_grid_is), # hail
    (blur_15mi_grid_is, blur_15mi_grid_is), # sig_tornado
    (blur_15mi_grid_is, blur_70mi_grid_is), # sig_wind
    (blur_35mi_grid_is, blur_50mi_grid_is), # sig_wind_adj
    (blur_15mi_grid_is, blur_25mi_grid_is), # sig_hail
    (blur_0mi_grid_is,  blur_15mi_grid_is), # tornado_life_risk
  ]

  extended_forecasts_blur_grid_is = [
    (blur_15mi_grid_is, blur_50mi_grid_is),  # tornado
    (blur_35mi_grid_is, blur_50mi_grid_is),  # wind
    (blur_25mi_grid_is, blur_50mi_grid_is),  # wind_adj
    (blur_25mi_grid_is, blur_50mi_grid_is),  # hail
    (blur_15mi_grid_is, blur_50mi_grid_is),  # sig_tornado
    (blur_70mi_grid_is, blur_100mi_grid_is), # sig_wind
    (blur_50mi_grid_is, blur_100mi_grid_is), # sig_wind_adj
    (blur_25mi_grid_is, blur_50mi_grid_is),  # sig_hail
    (blur_15mi_grid_is, blur_50mi_grid_is),  # tornado_life_risk
  ]

  _forecasts_blurred =
    vcat(
      PredictionForecasts.blurred(regular_forecasts(_forecasts),  1:36,  blur_grid_is),
      PredictionForecasts.blurred(extended_forecasts(_forecasts), 36:48, extended_forecasts_blur_grid_is), # Yes, 36:48 is correct so that f37 uses a bit of the larger radii blur
    )


  # Calibrating hourly predictions to validation data

  event_to_bins = Dict{String, Vector{Float32}}(
    "tornado"           => [0.0011201472,  0.0040470827,  0.009924359,  0.020150831,  0.04450299,   1.0],
    "wind"              => [0.0075269686,  0.019717596,   0.037927717,  0.067318514,  0.12851062,   1.0],
    "wind_adj"          => [0.0028242674,  0.008482344,   0.017187402,  0.032382138,  0.06685648,   1.0],
    "hail"              => [0.003987495,   0.011505866,   0.023705853,  0.04564259,   0.09055661,   1.0],
    "sig_tornado"       => [0.00048080692, 0.0023937228,  0.006148654,  0.012262353,  0.022267453,  1.0],
    "sig_wind"          => [0.00078459026, 0.0025756117,  0.005378738,  0.009706564,  0.020927433,  1.0],
    "sig_wind_adj"      => [0.00041078764, 0.0013922893,  0.002989863,  0.0053869793, 0.010809073,  1.0],
    "sig_hail"          => [0.00091282465, 0.0031095394,  0.00634054,   0.012477984,  0.02483106,   1.0],
    "tornado_life_risk" => [0.00011407318, 0.00061691855, 0.0014705167, 0.0029623061, 0.0065326905, 1.0],
  )

  event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"           => [[1.0151445,  0.2071434],   [0.990645,  0.08936819], [1.0937685, 0.61661005],  [0.86215496, -0.3573002],  [0.9471853,  -0.049062375]],
    "wind"              => [[1.052107,   0.26756564],  [1.0934643, 0.45261195], [1.0603855, 0.33706936],  [0.97554934, 0.09084179],  [0.98142743, 0.10940261]],
    "wind_adj"          => [[1.0665799,  0.36155185],  [1.2188917, 1.1445187],  [1.0872297, 0.5649324],   [1.0266879,  0.34076434],  [1.036534,   0.3751357]],
    "hail"              => [[0.98223704, -0.05874168], [1.0158885, 0.1087229],  [1.0228801, 0.14139068],  [1.0846623,  0.34556273],  [1.127743,   0.46917012]],
    "sig_tornado"       => [[1.003107,   0.06779992],  [1.0754979, 0.41287646], [1.3813931, 2.1319306],   [1.3965261,  2.2143273],   [0.91918075, 0.24742183]],
    "sig_wind"          => [[1.030558,   0.19321586],  [1.1623001, 1.0385767],  [1.1844709, 1.1859512],   [0.7777289,  -0.807949],   [0.9798549,  0.035456475]],
    "sig_wind_adj"      => [[0.9896531,  -0.11605113], [1.1852539, 1.3334796],  [1.3313962, 2.2548943],   [0.9650048,  0.245746],    [1.0000306,  0.3686924]],
    "sig_hail"          => [[0.98084414, -0.12066075], [1.166308,  0.9986842],  [0.9872798, 0.031519376], [1.0985931,  0.55506366],  [0.87546086, -0.3075612]],
    "tornado_life_risk" => [[0.97598267, 0.19340746],  [1.0696578, 0.8193858],  [1.0577359, 0.78586924],  [0.8839506,  -0.27480733], [0.64814734, -1.5873486]],
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


  # # Day & Four-hourly forecasts

  # # 1. Try both independent events total prob and max hourly prob as the main descriminator
  # # 2. bin predictions into 10 bins of equal weight of positive labels
  # # 3. combine bin-pairs (overlapping, 9 bins total)
  # # 4. train a logistic regression for each bin,
  # #   σ(a1*logit(independent events total prob) +
  # #     a2*logit(max hourly prob) +
  # #     b)
  # # 5. prediction is weighted mean of the two overlapping logistic models
  # # 6. should thereby be absolutely calibrated (check)
  # # 7. calibrate to SPC thresholds (linear interpolation)

  # _forecasts_day_accumulators, _forecasts_day2_accumulators, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(_forecasts_calibrated, models; module_name = "HREFPrediction")

  # # The following was computed in TrainDay.jl

  # event_to_day_bins = Dict{String, Vector{Float32}}(
  #   "tornado"      => [0.021043906, 0.074019335, 0.17095083, 1.0],
  #   "wind"         => [0.12138814,  0.27036425,  0.44575575, 1.0],
  #   "wind_adj"     => [0.04442204,  0.12223551,  0.25552708, 1.0],
  #   "hail"         => [0.066225864, 0.15690458,  0.29827937, 1.0],
  #   "sig_tornado"  => [0.009904385, 0.047619365, 0.14518347, 1.0],
  #   "sig_wind"     => [0.017750502, 0.050954822, 0.09204348, 1.0],
  #   "sig_wind_adj" => [0.00855192,  0.028172063, 0.07532653, 1.0],
  #   "sig_hail"     => [0.016018612, 0.044114113, 0.08977107, 1.0],
  # )
  # event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
  #   "tornado"      => [[0.95958227, 0.04161413,   -0.10651286],  [1.2272763,  -0.15624464,  -0.18067063],  [0.5964124,  0.17200926,   -0.3083448]],
  #   "wind"         => [[1.0424639,  -0.005972234, -0.1651274],   [1.1494,     -0.107195236, -0.3110462],   [0.92132,    0.0022699288, -0.22744325]],
  #   "wind_adj"     => [[0.99742454, 0.02784912,   -0.01899123],  [1.0577022,  -0.1336032,   -0.51860476],  [1.2023934,  -0.058322832, -0.087366514]],
  #   "hail"         => [[1.0235776,  0.02489767,   -0.070147164], [1.1081746,  -0.088940814, -0.34076628],  [1.1614795,  -0.24988303,  -0.7304756]],
  #   "sig_tornado"  => [[0.3738464,  0.5298857,    -0.016889505], [0.53315455, 0.50932276,   0.41353387],   [0.41091824, 0.60895973,   0.510827]],
  #   "sig_wind"     => [[0.60896295, 0.32489055,   -0.06371247],  [0.99172807, 0.15444452,   0.46648186],   [0.8760332,  0.31676358,   0.86151475]],
  #   "sig_wind_adj" => [[0.98283756, 0.02864707,   -0.070753366], [0.74302465, 0.17016491,   -0.2746076],   [0.37014917, 1.2757467,    3.3445797]],
  #   "sig_hail"     => [[1.2635667,  -0.21535042,  -0.32895777],  [1.5278528,  -0.32414392,  0.0025766783], [1.1042255,  -0.16546442,  -0.4495338]],
  # )

  # event_to_fourhourly_bins = Dict{String, Vector{Float32}}(
  #   "tornado"      => [0.009209944,  0.035263043, 0.09849834,  1.0],
  #   "wind"         => [0.04380678,   0.1264081,   0.26765373,  1.0],
  #   "wind_adj"     => [0.016413603,  0.058604192, 0.15199223,  1.0],
  #   "hail"         => [0.021138927,  0.06349602,  0.15349576,  1.0],
  #   "sig_tornado"  => [0.005631322,  0.026526544, 0.092573315, 1.0],
  #   "sig_wind"     => [0.0057922173, 0.0215623,   0.047722477, 1.0],
  #   "sig_wind_adj" => [0.0030460916, 0.012603851, 0.040286843, 1.0],
  #   "sig_hail"     => [0.005561535,  0.017045472, 0.04543988,  1.0],
  # )

  # event_to_fourhourly_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
  #   "tornado"      => [[1.0933163,  -0.07491289,   -0.09395848],  [1.2857381,  -0.32322577, -0.54111695], [0.89995205, 0.025575645, -0.32821795]],
  #   "wind"         => [[0.97546655, 0.025674382,   -0.12313984],  [1.0660465,  -0.06211259, -0.19784704], [1.0678099,  -0.11768198, -0.33153704]],
  #   "wind_adj"     => [[1.0981627,  -0.08259544,   -0.08323573],  [1.1117252,  -0.15500262, -0.3613525],  [1.1775827,  -0.12373863, -0.11497526]],
  #   "hail"         => [[0.96931684, 0.03862384,    -0.061203808], [1.1384751,  -0.16171022, -0.3580413],  [1.1030476,  -0.20756821, -0.5726693]],
  #   "sig_tornado"  => [[1.2125412,  -0.21225019,   -0.29702586],  [1.0641434,  -0.11383827, -0.45060942], [0.5815711,  0.39582604,  -0.006172086]],
  #   "sig_wind"     => [[0.9948475,  0.00020837433, -0.15796766],  [0.99380416, 0.082129076, 0.25521365],  [0.9400184,  0.20437491,  0.5816906]],
  #   "sig_wind_adj" => [[1.359399,   -0.3299467,    -0.18067853],  [0.9419031,  0.005614741, -0.3110229],  [0.6525776,  0.65669703,  1.5379776]],
  #   "sig_hail"     => [[1.1378537,  -0.13094603,   -0.17552955],  [1.2316782,  -0.27330196, -0.5414381],  [1.1437855,  -0.1293063,  -0.20190638]],
  # )

  # _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day")
  # _forecasts_day_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_with_sig_gated")

  # _forecasts_day2 = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day2_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "day2")
  # _forecasts_day2_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_with_sig_gated")

  # _forecasts_fourhourly = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_fourhourly_accumulators, event_to_fourhourly_bins, event_to_fourhourly_bins_logistic_coeffs, models; module_name = "HREFPrediction", period_name = "four-hourly")
  # _forecasts_fourhourly_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_fourhourly, models, gated_models; model_name = "HREFPrediction_four-hourly_severe_probabilities_with_sig_gated")

  # spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
  #   "tornado" => [
  #     (0.02, 0.016950607),
  #     (0.05, 0.06830406),
  #     (0.1,  0.17817497),
  #     (0.15, 0.3255825),
  #     (0.3,  0.4591999),
  #     (0.45, 0.60011864)
  #   ],
  #   "wind_adj" => [
  #     (0.05, 0.011899948),
  #     (0.15, 0.06845665),
  #     (0.3,  0.24048424),
  #     (0.45, 0.471941)
  #   ],
  #   "wind" => [
  #     (0.05, 0.0479908),
  #     (0.15, 0.21794319),
  #     (0.3,  0.4940548),
  #     (0.45, 0.7420559)
  #   ],
  #   "hail" => [
  #     (0.05, 0.029951096),
  #     (0.15, 0.123464584),
  #     (0.3,  0.3763752),
  #     (0.45, 0.6608143)
  #   ],
  #   "sig_tornado"  => [(0.1, 0.06202507)],
  #   "sig_wind"     => [(0.1, 0.118608475)],
  #   "sig_wind_adj" => [(0.1, 0.06686592)],
  #   "sig_hail"     => [(0.1, 0.070280075)],
  # )

  # # spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
  # #   "tornado" => [
  # #     (0.02, 0.017892838),
  # #     (0.05, 0.07787514),
  # #     (0.1,  0.17152214),
  # #     (0.15, 0.2814541),
  # #     (0.3,  0.3905239),
  # #     (0.45, 0.6009083)
  # #   ],
  # #   "wind" => [
  # #     (0.05, 0.051660538),
  # #     (0.15, 0.21513557),
  # #     (0.3,  0.49578285),
  # #     (0.45, 0.78172493)
  # #   ],
  # #   "wind_adj" => [
  # #     (0.05, 0.051660538),
  # #     (0.15, 0.21513557),
  # #     (0.3,  0.49578285),
  # #     (0.45, 0.78172493)
  # #   ],
  # #   "hail" => [
  # #     (0.05, 0.030927658),
  # #     (0.15, 0.12172127),
  # #     (0.3,  0.33656883),
  # #     (0.45, 0.61953926)
  # #   ],
  # #   "sig_tornado"  => [(0.1, 0.063589096)],
  # #   "sig_wind"     => [(0.1, 0.11205864)],
  # #   "sig_wind_adj" => [(0.1, 0.11205864)],
  # #   "sig_hail"     => [(0.1, 0.057775497)],
  # # )

  # # ensure ordered the same as the features in the data
  # calibrations = map(m -> spc_calibrations[m[1]], models)

  # _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds")
  # _forecasts_day_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  # _forecasts_day2_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day2, calibrations; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds")
  # _forecasts_day2_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day2_spc_calibrated, models, gated_models; model_name = "HREFPrediction_day2_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  ()
end

end # module HREFPrediction