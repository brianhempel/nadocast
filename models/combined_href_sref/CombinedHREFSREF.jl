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


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))



function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_href_newer
  global _forecasts_sref_newer
  global _forecasts_href_newer_combined
  global _forecasts_sref_newer_combined
  global _forecasts_day_accumulators
  global _forecasts_day
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
    "tornado"     => [0.0010219938,  0.0040875063, 0.00994226,   0.021133944, 0.042196225, 1.0],
    "wind"        => [0.008519708,   0.02144979,   0.0403009,    0.0697437,   0.1250951,   1.0],
    "hail"        => [0.0033594805,  0.009371412,  0.019450566,  0.036249824, 0.074465364, 1.0],
    "sig_tornado" => [0.00070907583, 0.0027267952, 0.006104077,  0.011227695, 0.023158755, 1.0],
    "sig_wind"    => [0.0007859762,  0.0024957801, 0.0052766525, 0.008785932, 0.015906833, 1.0],
    "sig_hail"    => [0.00076262257, 0.0020769562, 0.004243912,  0.008325059, 0.017990546, 1.0],
  )
  href_newer_event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}(
    "tornado"     => [[0.68840486, 0.30125454, -0.08057284], [0.79298836, 0.2025301,  -0.043320764], [0.87938726, 0.102382414, -0.13207436], [1.1286471,  -0.042436406,  0.21654046], [1.162479,   -0.12306028, 0.0014464383]],
    "wind"        => [[0.9175508,  0.23498222,  0.6797868],  [0.9369312,  0.23018824,  0.74334383],  [0.97935945, 0.178703,     0.6830242],  [0.94077724,  0.13174792,   0.40266746], [0.8477966,   0.192575,   0.37751102]],
    "hail"        => [[0.8667113,  0.21558651,  0.59692353], [0.90613884, 0.11804946,  0.23723379],  [0.9837282,  0.15166068,   0.7282904],  [0.7897085,   0.1952087,    0.2095446],  [0.8960403,   0.2555574,  0.7325582]],
    "sig_tornado" => [[0.70995516, 0.21704136, -0.6977991],  [0.9985171,  0.32341582,  1.961638],    [1.173152,   0.27435014,   2.5912948],  [0.86561954,  0.26091522,   1.0081335],  [1.1592792,   0.14715739, 1.6190162]],
    "sig_wind"    => [[1.0086884,  0.10093948,  0.7754164],  [0.9602558,  0.16216339,  0.8132331],   [1.2342786,  0.24149285,   2.857296],   [0.7123779,   0.2825677,    0.49686798], [0.6136547,   0.2704343,  0.039514035]],
    "sig_hail"    => [[0.88661367, 0.38997424,  2.2285037],  [0.7429764,  0.2915021,   0.5264995],   [0.6462555,  0.269669,    -0.1717195],  [0.65471405,  0.26225966,  -0.16004708], [0.69204414,  0.29494107, 0.20776647]],
  )

  # Returns array of (event_name, var_name, predict)
  function make_models(event_to_bins, event_to_bins_logistic_coeffs)
    event_types_count = length(models)

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
    event_types_count = length(models)

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

  # day_predict(forecast, data) = begin
  #   indep_events_ŷs  = @view data[:,1]
  #   max_hourly_probs = @view data[:,2]

  #   out = Array{Float32}(undef, length(indep_events_ŷs))

  #   bin_maxes = Float32[0.008833055, 0.025307992, 0.06799701, 0.11479675, 0.18474162, 1.0]

  #   Threads.@threads for i in 1:length(indep_events_ŷs)
  #     indep_events_ŷ   = indep_events_ŷs[i]
  #     max_hourly_prob  = max_hourly_probs[i]
  #     if indep_events_ŷ <= bin_maxes[1]
  #       # Bin 1-2 predictor only
  #       ŷ = day_bin_1_2_predict(indep_events_ŷ, max_hourly_prob)
  #     elseif indep_events_ŷ <= bin_maxes[2]
  #       # Bin 1-2 and 2-3 predictors
  #       ratio = ratio_between(indep_events_ŷ, bin_maxes[1], bin_maxes[2])
  #       ŷ = ratio*day_bin_2_3_predict(indep_events_ŷ, max_hourly_prob) + (1f0 - ratio)*day_bin_1_2_predict(indep_events_ŷ, max_hourly_prob)
  #     elseif indep_events_ŷ <= bin_maxes[3]
  #       # Bin 2-3 and 3-4 predictors
  #       ratio = ratio_between(indep_events_ŷ, bin_maxes[2], bin_maxes[3])
  #       ŷ = ratio*day_bin_3_4_predict(indep_events_ŷ, max_hourly_prob) + (1f0 - ratio)*day_bin_2_3_predict(indep_events_ŷ, max_hourly_prob)
  #     elseif indep_events_ŷ <= bin_maxes[4]
  #       # Bin 3-4 and 4-5 predictors
  #       ratio = ratio_between(indep_events_ŷ, bin_maxes[3], bin_maxes[4])
  #       ŷ = ratio*day_bin_4_5_predict(indep_events_ŷ, max_hourly_prob) + (1f0 - ratio)*day_bin_3_4_predict(indep_events_ŷ, max_hourly_prob)
  #     elseif indep_events_ŷ <= bin_maxes[5]
  #       # Bin 4-5 and 5-6 predictors
  #       ratio = ratio_between(indep_events_ŷ, bin_maxes[4], bin_maxes[5])
  #       ŷ = ratio*day_bin_5_6_predict(indep_events_ŷ, max_hourly_prob) + (1f0 - ratio)*day_bin_4_5_predict(indep_events_ŷ, max_hourly_prob)
  #     else
  #       # Bin 5-6 predictor only
  #       ŷ = day_bin_5_6_predict(indep_events_ŷ, max_hourly_prob)
  #     end
  #     out[i] = ŷ
  #   end

  #   out
  # end

  # _forecasts_day = PredictionForecasts.simple_prediction_forecasts(_forecasts_day_accumulators, day_predict; model_name = "CombinedHREFSREF_day_severe_probabilities")

  # spc_calibration = [
  #   (0.02, 0.016253397),
  #   (0.05, 0.0649308),
  #   (0.1,  0.18771306),
  #   (0.15, 0.28330332),
  #   (0.3,  0.32384455),
  # ]

  # _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, spc_calibration; model_name = "CombinedHREFSREF_day_severe_probabilities_calibrated_to_SPC_thresholds")

  ()
end

end # module StackedHREFSREF
