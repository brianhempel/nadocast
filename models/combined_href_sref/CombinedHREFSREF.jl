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
    ("sig_tornado_gated_by_tornado", "sig_tornado",  "tornado"),
    ("sig_wind_gated_by_wind",       "sig_wind",     "wind"),
    ("sig_wind_gated_by_wind_adj",   "sig_wind_adj", "wind_adj"),
    ("sig_hail_gated_by_hail",       "sig_hail",     "hail"),
  ]

# (event_name, grib2_var_name, model_name)
# I don't think the grib2_var_name is ever used.
models_with_gated =
  [ models;
    [ ("sig_tornado",  "STORPRO",  "sig_tornado_gated_by_tornado")
    , ("sig_wind",     "SWINDPRO", "sig_wind_gated_by_wind")
    , ("sig_wind_adj", "SWINDPRO", "sig_wind_adj_gated_by_wind_adj")
    , ("sig_hail",     "SHAILPRO", "sig_hail_gated_by_hail")
    ]
  ]
models_with_gated_count = length(models_with_gated)


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))


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
    "tornado"      => [0.0013122079,  0.004710326,  0.011888903,  0.022864908,  0.04753879,   1.0],
    "wind"         => [0.008744916,   0.021842286,  0.04064058,   0.06938802,   0.122678265,  1.0],
    "wind_adj"     => [0.0028606635,  0.008563636,  0.017410113,  0.031802237,  0.060865648,  1.0],
    "hail"         => [0.003458654,   0.009600723,  0.020057917,  0.037324984,  0.076065265,  1.0],
    "sig_tornado"  => [0.0007784463,  0.0033980992, 0.007858597,  0.013500211,  0.021512743,  1.0],
    "sig_wind"     => [0.00076082407, 0.0025200176, 0.0051898635, 0.008781022,  0.016167704,  1.0],
    "sig_wind_adj" => [0.0005631988,  0.0014582027, 0.00254442,   0.0040013813, 0.0057724603, 1.0],
    "sig_hail"     => [0.0007729447,  0.0022382603, 0.0047318107, 0.009228747,  0.020026933,  1.0],
  )
  href_newer_event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"      => [[0.8548191, 0.2597601,   0.8644513],  [0.78474545, 0.12277814,  -0.4418093], [1.0832391, 0.038129725, 0.5669815],  [0.75186867, 0.07035734,  -0.6170576],   [1.0223982,  0.12714103,    0.5787479]],
    "wind"         => [[0.9376985, 0.2066142,   0.70834965], [0.9932054,  0.15726718,  0.71593946], [1.0039783, 0.12171379,  0.6053313],  [0.9382497,  0.13403365,  0.4595898],    [0.74286795, 0.21366222,    0.2578835]],
    "wind_adj"     => [[0.9797254, 0.16798192,  0.8348953],  [1.0740479,  0.15907931,  1.2374448],  [1.0872117, 0.10037302,  1.0073562],  [0.9246941,  0.19532852,  0.81610787],   [0.73403955, 0.29090548,    0.62176204]],
    "hail"         => [[0.8451918, 0.2678841,   0.7947575],  [0.82366365, 0.1894165,   0.23550989], [0.9017725, 0.23303771,  0.7705032],  [0.74698466, 0.2205954,   0.16990918],   [0.86427146, 0.2018278,     0.43504176]],
    "sig_tornado"  => [[0.7318563, 0.3246817,   0.37168285], [0.9523567,  0.26052013,  1.2165213],  [1.473916,  0.17237528,  3.4237669],  [1.3563457,  0.091417536, 2.4898052],    [0.8635645,  -0.0004715729, 0.056863528]],
    "sig_wind"     => [[1.0237541, 0.07430771,  0.77363414], [0.9801676,  0.22527456,  1.4178667],  [1.1523683, 0.2936303,   2.783076],   [0.7023142,  0.22948036,  0.17595705],   [0.49640635, 0.20968291,    -0.77673304]],
    "sig_wind_adj" => [[1.127125,  0.052893594, 1.2472699],  [1.5036446,  0.021982145, 3.6960568],  [1.4205347, 0.16423136,  4.1276884],  [1.5778238,  0.50511247,  7.107699],     [0.1278618,  0.78106964,    0.9255426]],
    "sig_hail"     => [[0.8787013, 0.3119461,   1.6104017],  [0.7609628,  0.27349025,  0.5037077],  [0.6508231, 0.3390255,   0.28728625], [0.562786,   0.35293543,  -0.048666567], [0.63449466, 0.3219732,     0.13853967]],
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
  # _forecasts_sref_newer_combined_with_sig_gated = Forecasts.Forecast[]


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

  event_to_day_bins = Dict{String, Vector{Float32}}(
    "tornado"      => [0.022898283, 0.07626015,  0.16708241, 1.0],
    "wind"         => [0.13070628,  0.2815187,   0.46197507, 1.0],
    "wind_adj"     => [0.044802103, 0.122619815, 0.26531604, 1.0],
    "hail"         => [0.06832412,  0.16137148,  0.302417,   1.0],
    "sig_tornado"  => [0.010960604, 0.047013957, 0.13788173, 1.0],
    "sig_wind"     => [0.016674783, 0.053257223, 0.09693365, 1.0],
    "sig_wind_adj" => [0.008988211, 0.02997692,  0.08362562, 1.0],
    "sig_hail"     => [0.015938511, 0.04517084,  0.09306535, 1.0],
  )
  event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"      => [[0.89827985, 0.08686848,  -0.081391975], [1.148995,   -0.06917223, -0.03546825], [0.60322016, 0.1676681,    -0.30535954]],
    "wind"         => [[0.9534275,  0.06817708,  -0.06704924],  [1.0759317,  -0.11009742, -0.3944708],  [0.95407456, -0.031302005, -0.28859687]],
    "wind_adj"     => [[0.8554493,  0.14431913,  0.08940227],   [0.9335435,  -0.04865307, -0.47948343], [1.1919755,  -0.05395548,  -0.10846542]],
    "hail"         => [[0.9860878,  0.049258705, -0.05329424],  [1.136249,   -0.14193086, -0.47682077], [1.2677307,  -0.3622067,   -0.94742197]],
    "sig_tornado"  => [[0.48711854, 0.40669206,  -0.15710305],  [0.4327529,  0.54704756,  0.31344774],  [0.52736497, 0.50139576,   0.3873906]],
    "sig_wind"     => [[0.28422165, 0.6058833,   0.1703358],    [0.78268194, 0.4231073,   1.0043924],   [0.8006246,  0.42102247,   1.0746965]],
    "sig_wind_adj" => [[0.4749642,  0.4533547,   0.14967395],   [0.46408254, 0.43720487,  0.07941379],  [1.194624,   0.28879985,   1.5873092]],
    "sig_hail"     => [[1.2652406,  -0.22227533, -0.34498304],  [1.5420237,  -0.37746048, -0.21447094], [1.409416,   -0.404029,    -0.62075996]],
  )

  # event_to_fourhourly_bins = Dict{String, Vector{Float32}}(
  #   "tornado"     => [0.00753908,  0.030358605, 0.08693349, 1.0],
  #   "wind"        => [0.043117322, 0.12710203,  0.2744185,  1.0],
  #   "hail"        => [0.020449053, 0.0625404,   0.15308933, 1.0],
  #   "sig_tornado" => [0.00470224,  0.022806743, 0.08246323, 1.0],
  #   "sig_wind"    => [0.00543831,  0.021594528, 0.05260824, 1.0],
  #   "sig_hail"    => [0.006030224, 0.016691525, 0.04071429, 1.0],
  # )
  # event_to_fourhourly_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
  #   "tornado"     => [[1.1338079,  -0.12365579, -0.21844162], [1.2712703,  -0.23105474,  -0.21188106], [0.88705343, 0.104377,     -0.08031504]],
  #   "wind"        => [[0.94788873, 0.04973387,  -0.11124777], [1.0852667,  -0.08753359,  -0.23767674], [1.080012,   -0.108289875, -0.3081989]],
  #   "hail"        => [[0.9889517,  0.015257805, -0.10187187], [1.1896296,  -0.2049748,   -0.39164874], [1.1730652,  -0.2797999,   -0.66866827]],
  #   "sig_tornado" => [[1.5224317,  -0.49362254, -0.39044833], [0.96165395, -0.110244565, -0.8577433],  [0.94144565, 0.13474676,   0.0427606]],
  #   "sig_wind"    => [[0.9259525,  0.06745399,  -0.08790273], [0.97327083, 0.05183811,   0.023566378], [0.9228902,  0.17639892,   0.3989572]],
  #   "sig_hail"    => [[1.326802,   -0.31069145, -0.27217078], [1.4527221,  -0.5298689,   -0.9338967],  [1.1630019,  -0.154732,    -0.25065106]],
  # )

  # # We only ever use the 0Z forecasts (normally) but here we are using the 0Z calibration on non-0Z runs too
  _forecasts_day = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_day_accumulators, event_to_day_bins, event_to_day_bins_logistic_coeffs, models; module_name = "CombinedHREFSREF", period_name = "day")
  _forecasts_day_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_day, models, gated_models; model_name = "CombinedHREFSREF_day_severe_probabilities_with_sig_gated")

  # _forecasts_fourhourly = PredictionForecasts.period_forecasts_from_accumulators(_forecasts_fourhourly_accumulators, event_to_fourhourly_bins, event_to_fourhourly_bins_logistic_coeffs, models; module_name = "CombinedHREFSREF", period_name = "four-hourly")
  # _forecasts_fourhourly_with_sig_gated = PredictionForecasts.added_gated_predictions(_forecasts_fourhourly, models, gated_models; model_name = "CombinedHREFSREF_four-hourly_severe_probabilities_with_sig_gated")

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado" => [
      (0.02, 0.018671036),
      (0.05, 0.07438469),
      (0.1,  0.17809486),
      (0.15, 0.31424522),
      (0.3,  0.4208889),
      (0.45, 0.5354595)
    ],
    "wind" => [
      (0.05, 0.04947853),
      (0.15, 0.22776604),
      (0.3,  0.5126095),
      (0.45, 0.7822857)
    ],
    "wind_adj" => [
      (0.05, 0.011903763),
      (0.15, 0.069215775),
      (0.3,  0.24395943),
      (0.45, 0.48122978)
    ],
    "hail" => [
      (0.05, 0.032194138),
      (0.15, 0.12929726),
      (0.3,  0.38471794),
      (0.45, 0.6738186)
    ],
    "sig_tornado"  => [(0.1, 0.06749153)],
    "sig_wind"     => [(0.1, 0.1238575)],
    "sig_wind_adj" => [(0.1, 0.067705154)],
    "sig_hail"     => [(0.1, 0.06931114)],
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
