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

# For day, allow 0Z to 21Z runs
_forecasts_day_accumulators   = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
_forecasts_day                = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
# _forecasts_day_with_blurs_and_forecast_hour = [] # For Train.jl
# _forecasts_day_blurred = []
_forecasts_day_spc_calibrated = [] # HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z

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

function example_forecast()
  forecasts()[1]
end

function grid()
  HREF.grid()
end

@assert length(HREFPrediction.models)     == length(SREFPrediction.models)
@assert map(first, HREFPrediction.models) == map(first, SREFPrediction.models) # Same event names
# array of (event_name, grib2_var_name)
models = map(((event_name, grib2_var_name, _, _, _),) -> (event_name, grib2_var_name), HREFPrediction.models)
event_types_count = length(models)


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

blur_radii = HREFPrediction.blur_radii


function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_href_newer
  global _forecasts_sref_newer
  global _forecasts_href_newer_combined
  global _forecasts_sref_newer_combined
  global _forecasts_day_accumulators
  global _forecasts_day
  # global _forecasts_day_with_blurs_and_forecast_hour
  # global _forecasts_day_blurred
  global _forecasts_day_spc_calibrated

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
      event_name, var_name = models[model_i]

      predict(forecasts, data) = begin
        href_ŷs = @view data[:,model_i]
        sref_ŷs = @view data[:,model_i + event_types_count]

        out = Array{Float32}(undef, length(href_ŷs))

        bin_maxes            = event_to_bins[event_name]
        bins_logistic_coeffs = event_to_bins_logistic_coeffs[event_name]

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
  # _forecasts_sref_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_sref_newer, sref_newer_hour_models; model_name = "CombinedHREFSREF_hour_severe_probabilities_sref_newer")
  _forecasts_sref_newer_combined = Forecasts.Forecast[]


  # Day forecasts

  run_time_seconds_to_hourly_prediction_forecasts = Forecasts.run_time_seconds_to_forecasts(vcat(_forecasts_href_newer_combined,_forecasts_sref_newer_combined))

  run_date = Dates.Date(2019, 1, 9)
  associated_forecasts = []
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in 0:3:21
      run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      forecasts_for_run_time = get(run_time_seconds_to_hourly_prediction_forecasts, run_time_seconds, Forecasts.Forecast[])

      forecast_hours_in_convective_day = max(12-run_hour,2):clamp(23+12-run_hour,2,35)

      forecasts_for_convective_day = filter(forecast -> forecast.forecast_hour in forecast_hours_in_convective_day, forecasts_for_run_time)

      if (length(forecast_hours_in_convective_day) == length(forecasts_for_convective_day))
        push!(associated_forecasts, forecasts_for_convective_day)
      end

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
    end

    run_date += Dates.Day(1)
  end

  # Which run time and forecast hour to use for the set.
  # Namely: latest run time, then longest forecast hour
  choose_canonical_forecast(day_hourlies) = begin
    canonical = day_hourlies[1]
    for forecast in day_hourlies
      if (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)) > (Forecasts.run_time_in_seconds_since_epoch_utc(canonical), Forecasts.valid_time_in_seconds_since_epoch_utc(canonical))
        canonical = forecast
      end
    end
    canonical
  end

  day_hourly_predictions = ForecastCombinators.concat_forecasts(associated_forecasts, forecasts_tuple_to_canonical_forecast = choose_canonical_forecast)

  day_inventory_transformer(base_forecast, base_inventory) = begin
    out = Inventories.InventoryLine[]
    for model_i in 1:event_types_count
      event_name, var_name = models[model_i]
      push!(out, Inventories.InventoryLine("", "", base_inventory[1].date_str, "independent events total $(var_name)", "calculated", "day fcst", "", ""))
      push!(out, Inventories.InventoryLine("", "", base_inventory[1].date_str, "highest hourly $(var_name)", "calculated", "day fcst", "", ""))
    end
    out
  end

  day_data_transformer(base_forecast, base_data) = begin
    point_count, base_feature_count = size(base_data)
    hours_count = div(base_feature_count, event_types_count)

    out = Array{Float32}(undef, (point_count, 2 * event_types_count))

    Threads.@threads for i in 1:point_count
      for event_i in 1:event_types_count
        prob_no_tor = 1.0
        for hour_i in 1:hours_count
          prob_no_tor *= 1.0 - Float64((@view base_data[i, event_i:event_types_count:base_feature_count])[hour_i])
        end
        out[i, event_i*2 - 1] = Float32(1.0 - prob_no_tor)
        out[i, event_i*2    ] = maximum(@view base_data[i, event_i:event_types_count:base_feature_count])

        # sorted_probs = sort((@view base_data[i, event_i:event_types_count:base_feature_count]); rev = true)
        # out[i,2] = sorted_probs[1]
        # out[i,3] = sorted_probs[2]
        # out[i,4] = sorted_probs[3]
        # out[i,5] = sorted_probs[4]
        # out[i,6] = sorted_probs[5]
        # out[i,7] = sorted_probs[6]
      end
    end
    out
  end

  # Caching barely helps load times, so we don't do it

  _forecasts_day_accumulators = ForecastCombinators.map_forecasts(day_hourly_predictions; inventory_transformer = day_inventory_transformer, data_transformer = day_data_transformer, model_name = "Day_severe_probability_accumulators_from_CombinedHREFSREF_hours")

  event_to_0z_day_bins = Dict{String, Vector{Float32}}(
    "tornado"     => [0.009306263,  0.028845591, 0.061602164, 0.106333174, 0.19435626, 1.0],
    "wind"        => [0.07021518,   0.16140154,  0.2604162,   0.36931401,  0.52724814, 1.0],
    "hail"        => [0.03750208,   0.082046196, 0.14193675,  0.21595338,  0.34294724, 1.0],
    "sig_tornado" => [0.0046208645, 0.015867244, 0.027446639, 0.07221879,  0.14377913, 1.0],
    "sig_wind"    => [0.0071317935, 0.025168268, 0.047133457, 0.07531253,  0.10004484, 1.0],
    "sig_hail"    => [0.012925774,  0.02218357,  0.033009935, 0.053475916, 0.09119451, 1.0],
  )
  event_to_0z_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[0.7936844,     0.19060811,  0.025705637], [1.1202875, -0.06516523, -0.13911094], [1.5175567,  -0.2130138,   0.37245846], [0.9057919, -0.1376394,   -0.7896346],  [0.5802065,  0.06708273, -0.6733654]],
    "wind"        => [[0.6955889,     0.29533672,  0.15469939],  [1.1739702, -0.03139503, -0.04425332], [1.0905153,  -0.07237125, -0.26282614], [0.8646443, -0.028500073, -0.3300517],  [0.9196351,  0.12227182,  0.0019935018]],
    "hail"        => [[0.67516387,    0.39498883,  0.6715193],   [0.78884,    0.11251911, -0.2715415],  [1.1522207,  -0.12645392, -0.39178276], [1.5031779, -0.6665933,   -1.5814598],  [1.5379245, -0.69794226, -1.6085442]],
    "sig_tornado" => [[1.1938772,    -0.13410978, -0.09725317],  [0.8071592,  0.551987,    2.28532],    [-1.0015389,  1.529142,    0.14288497], [0.6298735,  0.917871,     2.3416257],  [1.6475935, -0.2897538,  -0.1320373]],
    "sig_wind"    => [[-0.060351126,  0.9043533,   0.35478225],  [1.0681612,  0.08553178,  0.41871232], [0.46654662,  0.55500215,  0.7106551],  [0.8931556,  1.0609696,    4.039079],   [0.3618952,  0.17861357, -0.61968464]],
    "sig_hail"    => [[1.9470923,    -0.7360531,  -0.30602866],  [2.0595398, -1.1718963,  -2.2438788],  [1.2635527,  -0.6335322,  -2.3711135],  [2.0416996, -0.59624666,   0.07782969], [1.308884,  -0.365359,   -0.92940474]],
  )

  # Returns array of (event_name, var_name, predict)
  function make_day_models(event_to_day_bins, event_to_day_bins_logistic_coeffs)
    ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

    map(1:event_types_count) do model_i
      event_name, var_name = models[model_i]

      predict(forecasts, data) = begin
        total_prob_ŷs = @view data[:, model_i*2 - 1]
        max_hourly_ŷs = @view data[:, model_i*2]

        out = Array{Float32}(undef, length(total_prob_ŷs))

        bin_maxes            = event_to_day_bins[event_name]
        bins_logistic_coeffs = event_to_day_bins_logistic_coeffs[event_name]

        @assert length(bin_maxes) == length(bins_logistic_coeffs) + 1

        predict_one(coeffs, total_prob_ŷ, max_hourly_ŷ) = σ(coeffs[1]*logit(total_prob_ŷ) + coeffs[2]*logit(max_hourly_ŷ) + coeffs[3])

        Threads.@threads for i in 1:length(total_prob_ŷs)
          total_prob_ŷ = total_prob_ŷs[i]
          max_hourly_ŷ = max_hourly_ŷs[i]
          if total_prob_ŷ <= bin_maxes[1]
            # Bin 1-2 predictor only
            ŷ = predict_one(bins_logistic_coeffs[1], total_prob_ŷ, max_hourly_ŷ)
          elseif total_prob_ŷ > bin_maxes[length(bin_maxes) - 1]
            # Bin 5-6 predictor only
            ŷ = predict_one(bins_logistic_coeffs[length(bins_logistic_coeffs)], total_prob_ŷ, max_hourly_ŷ)
          else
            # Overlapping bins
            higher_bin_i = findfirst(bin_max -> total_prob_ŷ <= bin_max, bin_maxes)
            lower_bin_i  = higher_bin_i - 1
            coeffs_higher_bin = bins_logistic_coeffs[higher_bin_i]
            coeffs_lower_bin  = bins_logistic_coeffs[lower_bin_i]

            # Bin 1-2 and 2-3 predictors
            ratio = ratio_between(total_prob_ŷ, bin_maxes[lower_bin_i], bin_maxes[higher_bin_i])
            ŷ = ratio*predict_one(coeffs_higher_bin, total_prob_ŷ, max_hourly_ŷ) + (1f0 - ratio)*predict_one(coeffs_lower_bin, total_prob_ŷ, max_hourly_ŷ)
          end
          out[i] = ŷ
        end

        out
      end

      (event_name, var_name, predict)
    end
  end

  # We only ever use the 0Z forecasts (normally) but here we are using the 0Z calibration non-0Z runs too
  day_models = make_day_models(event_to_0z_day_bins, event_to_0z_day_bins_logistic_coeffs)
  _forecasts_day = PredictionForecasts.simple_prediction_forecasts(_forecasts_day_accumulators, day_models; model_name = "CombinedHREFSREF_day_severe_probabilities")

  # _forecasts_day_with_blurs_and_forecast_hour = PredictionForecasts.with_blurs_and_forecast_hour(_forecasts_day, blur_radii)

  spc_calibrations = Dict{String, Vector{Tuple{Float32, Float32}}}(
    "tornado" => [
      (0.02, 0.01799427),
      (0.05, 0.07155262),
      (0.1,  0.17533639),
      (0.15, 0.22911775),
      (0.3,  0.25342447)
    ],
    "wind" => [
      (0.05, 0.06466381),
      (0.15, 0.19475871),
      (0.3,  0.42799288),
      (0.45, 0.66319275)
    ],
    "hail" => [
      (0.05, 0.03239231),
      (0.15, 0.10845302),
      (0.3,  0.28606683),
      (0.45, 0.5663775)
    ],
    "sig_tornado" => [(0.1, 0.08093022)],
    "sig_wind"    => [(0.1, 0.1095853)],
    "sig_hail"    => [(0.1, 0.057069816)],
  )

  # ensure ordered the same as the features in the data
  calibrations =
    map(models) do (event_name, _)
      spc_calibrations[event_name]
    end

  _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, calibrations; model_name = "CombinedHREFSREF_day_severe_probabilities_calibrated_to_SPC_thresholds")

  ()
end

end # module StackedHREFSREF
