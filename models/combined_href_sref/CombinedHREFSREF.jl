module CombinedHREFSREF

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids
import Inventories

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts
import Climatology
import FeatureEngineeringShared

push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF

push!(LOAD_PATH, (@__DIR__) * "/../href_prediction")
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../sref_prediction")
import SREFPrediction


MINUTE = 60
HOUR   = 60*MINUTE

# Forecast run time is always the newer forecast.

_forecasts_href_newer = [] # Output is a 2-feature forecast: layer 1 is the HREF-based prediction, layer 2 is the SREF-based prediction
_forecasts_sref_newer = [] # Output is a 2-feature forecast: layer 1 is the HREF-based prediction, layer 2 is the SREF-based prediction
_forecasts_href_newer_combined = []
_forecasts_sref_newer_combined = []
_forecasts_href_newer_combined_with_sig_gated = []
_forecasts_sref_newer_combined_with_sig_gated = []

# For day, allow 0Z to 21Z runs
_forecasts_day_accumulators                   = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21
_forecasts_fourhourly_accumulators            = []
_forecasts_day                                = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
_forecasts_day_with_sig_gated                 = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
_forecasts_fourhourly                         = []
_forecasts_fourhourly_with_sig_gated          = []
# _forecasts_day_with_blurs_and_forecast_hour = [] # For Train.jl
# _forecasts_day_blurred                      = []
_forecasts_day_spc_calibrated                 = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
_forecasts_day_spc_calibrated_with_sig_gated  = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z

# SREF 3 hours behind HREF
function forecasts_href_newer()
  if isempty(_forecasts_href_newer)
    reload_forecasts()
    _forecasts_href_newer
  else
    _forecasts_href_newer
  end
end

# HREF 3 hours behind SREF
function forecasts_sref_newer()
  if isempty(_forecasts_sref_newer)
    reload_forecasts()
    _forecasts_sref_newer
  else
    _forecasts_sref_newer
  end
end

# SREF 3 hours behind HREF
function forecasts_href_newer_combined()
  if isempty(_forecasts_href_newer_combined)
    reload_forecasts()
    _forecasts_href_newer_combined
  else
    _forecasts_href_newer_combined
  end
end

# HREF 3 hours behind SREF
function forecasts_sref_newer_combined()
  if isempty(_forecasts_sref_newer_combined)
    reload_forecasts()
    _forecasts_sref_newer_combined
  else
    _forecasts_sref_newer_combined
  end
end

# SREF 3 hours behind HREF
function forecasts_href_newer_combined_with_sig_gated()
  if isempty(_forecasts_href_newer_combined_with_sig_gated)
    reload_forecasts()
    _forecasts_href_newer_combined_with_sig_gated
  else
    _forecasts_href_newer_combined_with_sig_gated
  end
end

# HREF 3 hours behind SREF
function forecasts_sref_newer_combined_with_sig_gated()
  if isempty(_forecasts_sref_newer_combined_with_sig_gated)
    reload_forecasts()
    _forecasts_sref_newer_combined_with_sig_gated
  else
    _forecasts_sref_newer_combined_with_sig_gated
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

# function forecasts_day_with_blurs_and_forecast_hour()
#   if isempty(_forecasts_day_with_blurs_and_forecast_hour)
#     reload_forecasts()
#     _forecasts_day_with_blurs_and_forecast_hour
#   else
#     _forecasts_day_with_blurs_and_forecast_hour
#   end
# end

# function forecasts_day_blurred()
#   if isempty(_forecasts_day_blurred)
#     reload_forecasts()
#     _forecasts_day_blurred
#   else
#     _forecasts_day_blurred
#   end
# end

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

function example_forecast()
  forecasts()[1]
end

function grid()
  HREF.grid()
end

@assert length(HREFPrediction.models)     == length(SREFPrediction.models)
@assert map(first, HREFPrediction.models) == map(first, SREFPrediction.models) # Same event names
# array of (event_name, grib2_var_name, model_name)
models = map(((event_name, grib2_var_name, _, _, _),) -> (event_name, grib2_var_name, event_name), HREFPrediction.models)
event_types_count = length(models)

# (gated_event_name, original_event_name, gate_event_name)
gated_models =
  [
    ("sig_tornado_gated_by_tornado", "sig_tornado", "tornado"),
    ("sig_wind_gated_by_wind",       "sig_wind",    "wind"),
    ("sig_hail_gated_by_hail",       "sig_hail",    "hail"),
  ]

# (event_name, grib2_var_name, model_name)
# I don't think the grib2_var_name is ever used.
models_with_gated =
  [ models;
    [ ("sig_tornado", "STORPRO",  "sig_tornado_gated_by_tornado")
    , ("sig_wind",    "SWINDPRO", "sig_wind_gated_by_wind")
    , ("sig_hail",    "SHAILPRO", "sig_hail_gated_by_hail")
    ]
  ]
models_with_gated_count = length(models_with_gated)


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

blur_radii = HREFPrediction.blur_radii


function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_href_newer
  global _forecasts_sref_newer
  global _forecasts_href_newer_combined
  global _forecasts_sref_newer_combined
  global _forecasts_href_newer_combined_with_sig_gated
  global _forecasts_sref_newer_combined_with_sig_gated
  global _forecasts_day_accumulators
  global _forecasts_fourhourly_accumulators
  global _forecasts_day
  global _forecasts_day_with_sig_gated
  global _forecasts_fourhourly
  global _forecasts_fourhourly_with_sig_gated
  # global _forecasts_day_with_blurs_and_forecast_hour
  # global _forecasts_day_blurred
  global _forecasts_day_spc_calibrated
  global _forecasts_day_spc_calibrated_with_sig_gated

  _forecasts_href_newer = []
  _forecasts_sref_newer = []

  href_prediction_forecasts = HREFPrediction.forecasts_blurred()
  sref_prediction_forecasts = SREFPrediction.forecasts_blurred()

  sref_upsampled_prediction_forecasts =
    ForecastCombinators.resample_forecasts(
      sref_prediction_forecasts,
      Grids.get_interpolating_upsampler,
      HREF.grid()
    )

  # Index to avoid O(n^2)

  run_time_seconds_to_href_forecasts = Forecasts.run_time_seconds_to_forecasts(href_prediction_forecasts)
  run_time_seconds_to_sref_forecasts = Forecasts.run_time_seconds_to_forecasts(sref_upsampled_prediction_forecasts)

  paired_href_newer = []
  paired_sref_newer = []

  run_date = Dates.Date(2019, 1, 9)
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in 0:3:21
      run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      hrefs_for_run_time = get(run_time_seconds_to_href_forecasts, run_time_seconds, Forecasts.Forecast[])
      srefs_for_run_time = get(run_time_seconds_to_sref_forecasts, run_time_seconds, Forecasts.Forecast[])
      hrefs_3hrs_earlier = get(run_time_seconds_to_href_forecasts, run_time_seconds - 3*HOUR, Forecasts.Forecast[])
      srefs_3hrs_earlier = get(run_time_seconds_to_sref_forecasts, run_time_seconds - 3*HOUR, Forecasts.Forecast[])

      for forecast_hour in 0:39 # 2:35 or 2:32 in practice
        valid_time_seconds = run_time_seconds + forecast_hour*HOUR

        perhaps_href_forecast              = filter(href_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast), hrefs_for_run_time)
        perhaps_sref_forecast              = filter(sref_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast), srefs_for_run_time)
        perhaps_href_forecast_3hrs_earlier = filter(href_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast), hrefs_3hrs_earlier)
        perhaps_sref_forecast_3hrs_earlier = filter(sref_forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast), srefs_3hrs_earlier)

        if length(perhaps_href_forecast) >= 2
          error("shouldn't have two matching href forecasts!")
        elseif length(perhaps_sref_forecast) >= 2
          error("shouldn't have two matching sref forecasts!")
        elseif length(perhaps_href_forecast) == 1 && length(perhaps_sref_forecast_3hrs_earlier) == 1
          push!(paired_href_newer, (perhaps_href_forecast[1], perhaps_sref_forecast_3hrs_earlier[1]))
        elseif length(perhaps_sref_forecast) == 1 && length(perhaps_href_forecast_3hrs_earlier) == 1
          push!(paired_sref_newer, (perhaps_href_forecast_3hrs_earlier[1], perhaps_sref_forecast[1]))
        end
      end
    end

    run_date += Dates.Day(1)
  end

  _forecasts_href_newer = ForecastCombinators.concat_forecasts(paired_href_newer; model_name = "Paired_HREF_and_SREF_hour_severe_probabilities_href_newer")
  _forecasts_sref_newer = ForecastCombinators.concat_forecasts(paired_sref_newer; model_name = "Paired_HREF_and_SREF_hour_severe_probabilities_sref_newer")


  href_newer_event_to_bins = Dict{String, Vector{Float32}}(
    "tornado"     => [0.0010191497,  0.0040240395, 0.009904478,  0.021118658, 0.04198461,  1.0],
    "wind"        => [0.008220518,   0.020898983,  0.039187722,  0.067865305, 0.1211984,   1.0],
    "hail"        => [0.0033015092,  0.00926606,   0.019275624,  0.03602357,  0.07374218,  1.0],
    "sig_tornado" => [0.0006729238,  0.0026572358, 0.0057637123, 0.010840383, 0.02260619,  1.0],
    "sig_wind"    => [0.0007965934,  0.0025051977, 0.0053047705, 0.008843536, 0.015871514, 1.0],
    "sig_hail"    => [0.00077555457, 0.002129781,  0.004338203,  0.00847202,  0.018383306, 1.0],
  )
  href_newer_event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[0.6823556,  0.30390468, -0.09062567], [0.7804226,  0.20210582,  -0.1086965],  [0.9002674,  0.099383384, -0.052697014], [1.1373994,  -0.0431747,   0.25296646],   [1.1550109, -0.12750162, -0.030436214]],
    "wind"        => [[0.9084373,  0.23359907,  0.64916927], [0.95582205, 0.22434467,   0.813807],   [0.9773524,  0.17435656,   0.68770856],  [0.9580874,   0.12913947,  0.4690096],   [0.85546756, 0.19877374,   0.44007877]],
    "hail"        => [[0.85908127, 0.22126952,  0.5908619],  [0.89740646, 0.122179836,  0.22032435], [0.9773174,  0.15612426,   0.7210259],   [0.7939177,   0.19984028,  0.23948279],  [0.8816676,  0.26308796,   0.7197715]],
    "sig_tornado" => [[0.71876323, 0.196306,   -0.7678675],  [1.0591443,  0.3063447,    2.2627401],  [1.0977775,  0.2782542,    2.271161],    [0.9274584,   0.23464419,  1.1744276],   [1.1490777,  0.07541906,   1.2876196]],
    "sig_wind"    => [[1.0125877,  0.09472051,  0.75418925], [0.9509606,  0.161822,     0.7489338],  [1.219375,   0.25611165,   2.849027],    [0.73210114,  0.29001468,  0.62654376],  [0.605263,   0.2646362,   -0.023003729]],
    "sig_hail"    => [[0.87852937, 0.39478606,  2.16777],    [0.76197445, 0.28871304,   0.59921116], [0.64860153, 0.26866677,  -0.18315962],  [0.63177335,  0.26839426, -0.25605413], [0.7029047,  0.28590807,    0.18754157]],
  )

  # Returns array of (event_name, var_name, predict)
  function make_models(event_to_bins, event_to_bins_logistic_coeffs)
    ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

    map(1:event_types_count) do model_i
      event_name, var_name, model_name = models[model_i] # event_name == model_name here

      predict(forecasts, data) = begin
        href_ŷs = @view data[:,model_i]
        sref_ŷs = @view data[:,model_i + event_types_count]

        out = Array{Float32}(undef, length(href_ŷs))

        bin_maxes            = event_to_bins[model_name]
        bins_logistic_coeffs = event_to_bins_logistic_coeffs[model_name]

        @assert length(bin_maxes) == length(bins_logistic_coeffs) + 1

        # # Fit logistic coefficients: Float32[0.77540565, 0.19299681, -0.21271989]
        # href_newer_bin_1_2_predict(href_ŷ, sref_ŷ) = σ(0.77540565f0*logit(href_ŷ) + 0.19299681f0*logit(sref_ŷ) + -0.21271989f0)

        predict_one(coeffs, href_ŷ, sref_ŷ) = σ(coeffs[1]*logit(href_ŷ) + coeffs[2]*logit(sref_ŷ) + coeffs[3])

        Threads.@threads for i in 1:length(href_ŷs)
          href_ŷ = href_ŷs[i]
          sref_ŷ = sref_ŷs[i]
          if href_ŷ <= bin_maxes[1]
            # Bin 1-2 predictor only
            ŷ = predict_one(bins_logistic_coeffs[1], href_ŷ, sref_ŷ)
          elseif href_ŷ > bin_maxes[length(bin_maxes) - 1]
            # Bin 5-6 predictor only
            ŷ = predict_one(bins_logistic_coeffs[length(bins_logistic_coeffs)], href_ŷ, sref_ŷ)
          else
            # Overlapping bins
            higher_bin_i = findfirst(bin_max -> href_ŷ <= bin_max, bin_maxes)
            lower_bin_i  = higher_bin_i - 1
            coeffs_higher_bin = bins_logistic_coeffs[higher_bin_i]
            coeffs_lower_bin  = bins_logistic_coeffs[lower_bin_i]

            # Bin 1-2 and 2-3 predictors
            ratio = ratio_between(href_ŷ, bin_maxes[lower_bin_i], bin_maxes[higher_bin_i])
            ŷ = ratio*predict_one(coeffs_higher_bin, href_ŷ, sref_ŷ) + (1f0 - ratio)*predict_one(coeffs_lower_bin, href_ŷ, sref_ŷ)
          end
          out[i] = ŷ
        end

        out
      end

      (event_name, var_name, predict)
    end
  end

  href_newer_hour_models = make_models(href_newer_event_to_bins, href_newer_event_to_bins_logistic_coeffs)

  _forecasts_href_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_href_newer, href_newer_hour_models; model_name = "CombinedHREFSREF_hour_severe_probabilities_href_newer")
  _forecasts_href_newer_combined_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_href_newer_combined, models, gated_models; model_name = "CombinedHREFSREF_hour_severe_probabilities_href_newer_with_sig_gated")

  # _forecasts_sref_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_sref_newer, sref_newer_hour_models; model_name = "CombinedHREFSREF_hour_severe_probabilities_sref_newer")
  _forecasts_sref_newer_combined = Forecasts.Forecast[]
  _forecasts_sref_newer_combined_with_sig_gated = Forecasts.Forecast[]


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

  hourly_prediction_forecasts = vcat(_forecasts_href_newer_combined,_forecasts_sref_newer_combined)

  _forecasts_day_accumulators, _forecasts_day2_accumulators_unused, _forecasts_fourhourly_accumulators = PredictionForecasts.daily_and_fourhourly_accumulators(hourly_prediction_forecasts, models; module_name = "CombinedHREFSREF")


  event_to_0z_day_bins = Dict{String, Vector{Float32}}(
    "tornado"     => [0.018898962, 0.061602164, 0.13501288, 1.0],
    "wind"        => [0.11713328,  0.2604162,   0.43730876, 1.0],
    "hail"        => [0.058389254, 0.14193675,  0.26998883, 1.0],
    "sig_tornado" => [0.010414468, 0.027291382, 0.11116843, 1.0],
    "sig_wind"    => [0.01371814,  0.047133457, 0.08599372, 1.0],
    "sig_hail"    => [0.01797507,  0.03298726,  0.07266835, 1.0],
  )
  event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[0.915529,   0.075384595, -0.13371886], [1.3713769,   -0.14764006, 0.3025914],   [0.7308742,  -0.01088467, -0.7200494]],
    "wind"        => [[0.8752906,  0.14241132,  0.019422961], [1.0762116,   -0.07859125, -0.31812784], [0.8967682,  0.0795651,   -0.053071257]],
    "hail"        => [[0.69415,    0.3440887,   0.4457177],   [1.1330315,   -0.17600663, -0.6018588],  [1.4685977,  -0.6171552,  -1.4658885]],
    "sig_tornado" => [[0.9246038,  0.14572951,  0.4728381],   [-0.27371246, 1.0642519,   0.48847285],  [0.77029693, 0.587577,    1.354599]],
    "sig_wind"    => [[0.28025302, 0.6347386,   0.37337035],  [1.0196984,   0.3275573,   1.3824697],   [0.59162444, 0.5482808,   1.1521151]],
    "sig_hail"    => [[1.9733405,  -0.7847382,  -0.51065785], [1.3154331,   -0.7666408,  -2.8496652],  [1.3292868,  -0.31144178, -0.6903727]],
  )

  event_to_fourhourly_bins = Dict{String, Vector{Float32}}(
    "tornado"     => [0.00753908,  0.030358605, 0.08693349, 1.0],
    "wind"        => [0.043117322, 0.12710203,  0.2744185,  1.0],
    "hail"        => [0.020449053, 0.0625404,   0.15308933, 1.0],
    "sig_tornado" => [0.00470224,  0.022806743, 0.08246323, 1.0],
    "sig_wind"    => [0.00543831,  0.021594528, 0.05260824, 1.0],
    "sig_hail"    => [0.006030224, 0.016691525, 0.04071429, 1.0],
  )
  event_to_fourhourly_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[1.1338079,  -0.12365579, -0.21844162], [1.2712703,  -0.23105474,  -0.21188106], [0.88705343, 0.104377,     -0.08031504]],
    "wind"        => [[0.94788873, 0.04973387,  -0.11124777], [1.0852667,  -0.08753359,  -0.23767674], [1.080012,   -0.108289875, -0.3081989]],
    "hail"        => [[0.9889517,  0.015257805, -0.10187187], [1.1896296,  -0.2049748,   -0.39164874], [1.1730652,  -0.2797999,   -0.66866827]],
    "sig_tornado" => [[1.5224317,  -0.49362254, -0.39044833], [0.96165395, -0.110244565, -0.8577433],  [0.94144565, 0.13474676,   0.0427606]],
    "sig_wind"    => [[0.9259525,  0.06745399,  -0.08790273], [0.97327083, 0.05183811,   0.023566378], [0.9228902,  0.17639892,   0.3989572]],
    "sig_hail"    => [[1.326802,   -0.31069145, -0.27217078], [1.4527221,  -0.5298689,   -0.9338967],  [1.1630019,  -0.154732,    -0.25065106]],
  )

  # We only ever use the 0Z forecasts (normally) but here we are using the 0Z calibration on non-0Z runs too
  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_0z_day_bins, event_to_0z_day_bins_logistic_coeffs, models; module_name = "CombinedHREFSREF", period_name = "day")
  _forecasts_day_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day, models, gated_models; model_name = "CombinedHREFSREF_day_severe_probabilities_with_sig_gated")

  _forecasts_fourhourly = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_fourhourly_accumulators, event_to_fourhourly_bins, event_to_fourhourly_bins_logistic_coeffs, models; module_name = "CombinedHREFSREF", period_name = "four-hourly")
  _forecasts_fourhourly_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_fourhourly, models, gated_models; model_name = "CombinedHREFSREF_four-hourly_severe_probabilities_with_sig_gated")

  # _forecasts_day_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts_day, blur_radii)

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado" => [
      (0.02, 0.019361496),
      (0.05, 0.08399773),
      (0.1,  0.17687798),
      (0.15, 0.2711239),
      (0.3,  0.37184715),
      (0.45, 0.57912636),
    ],
    "wind" => [
      (0.05, 0.05350685),
      (0.15, 0.23026466),
      (0.3,  0.52155495),
      (0.45, 0.8279896)
    ],
    "hail" => [
      (0.05, 0.033460617),
      (0.15, 0.1253109),
      (0.3,  0.35450554),
      (0.45, 0.647604)
    ],
    "sig_tornado" => [(0.1, 0.069143295)],
    "sig_wind"    => [(0.1, 0.11427498)],
    "sig_hail"    => [(0.1, 0.05564308)],
  )

  # ensure ordered the same as the features in the data
  calibrations =
    map(models) do (_, _, model_name)  # event_name == model_name here
      spc_calibrations[model_name]
    end

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "CombinedHREFSREF_day_severe_probabilities_calibrated_to_SPC_thresholds")
  _forecasts_day_spc_calibrated_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day_spc_calibrated, models, gated_models; model_name = "CombinedHREFSREF_day_severe_probabilities_calibrated_to_SPC_thresholds_with_sig_gated")

  ()
end

end # module StackedHREFSREF
