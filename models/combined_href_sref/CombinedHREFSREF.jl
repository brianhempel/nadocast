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
models = map(((event_name, grib2_var_name, _, _, _)) -> (event_name, grib2_var_name), HREFPrediction.models)


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

# # Bin 1-2 --------
# # -1.0 < HREF_ŷ <= 0.0040673064
# # Fit logistic coefficients: Float32[0.77540565, 0.19299681, -0.21271989]
# href_newer_bin_1_2_predict(href_ŷ, sref_ŷ) = σ(0.77540565f0*logit(href_ŷ) + 0.19299681f0*logit(sref_ŷ) + -0.21271989f0)
# # Bin 2-3 --------
# # 0.000962529 < HREF_ŷ <= 0.009957244
# # Fit logistic coefficients: Float32[0.84564245, 0.14841641, -0.06817224]
# href_newer_bin_2_3_predict(href_ŷ, sref_ŷ) = σ(0.84564245f0*logit(href_ŷ) + 0.14841641f0*logit(sref_ŷ) + -0.06817224f0)
# # Bin 3-4 --------
# # 0.0040673064 < HREF_ŷ <= 0.020302918
# # Fit logistic coefficients: Float32[0.9977281, 0.14388186, 0.64254296]
# href_newer_bin_3_4_predict(href_ŷ, sref_ŷ) = σ(0.9977281f0*logit(href_ŷ) + 0.14388186f0*logit(sref_ŷ) + 0.64254296f0)
# # Bin 4-5 --------
# # 0.009957244 < HREF_ŷ <= 0.037081156
# # Fit logistic coefficients: Float32[1.3795987, 0.091625534, 1.9759048]
# href_newer_bin_4_5_predict(href_ŷ, sref_ŷ) = σ(1.3795987f0*logit(href_ŷ) + 0.091625534f0*logit(sref_ŷ) + 1.9759048f0)
# # Bin 5-6 --------
# # 0.020302918 < HREF_ŷ <= 1.0
# # Fit logistic coefficients: Float32[0.9358031, 0.1812378, 0.836498]
# href_newer_bin_5_6_predict(href_ŷ, sref_ŷ) = σ(0.9358031f0*logit(href_ŷ) + 0.1812378f0*logit(sref_ŷ) + 0.836498f0)


# # Bin 1-2 --------
# # -1.0 < HREF_ŷ <= 0.0038515618
# # Fit logistic coefficients: Float32[0.68609446, 0.29776412, -0.06935765]
# sref_newer_bin_1_2_predict(href_ŷ, sref_ŷ) = σ(0.68609446f0*logit(href_ŷ) + 0.29776412f0*logit(sref_ŷ) + -0.06935765f0)
# # Bin 2-3 --------
# # 0.0009233353 < HREF_ŷ <= 0.00954726
# # Fit logistic coefficients: Float32[0.73588675, 0.26439694, 0.021931686]
# sref_newer_bin_2_3_predict(href_ŷ, sref_ŷ) = σ(0.73588675f0*logit(href_ŷ) + 0.26439694f0*logit(sref_ŷ) + 0.021931686f0)
# # Bin 3-4 --------
# # 0.0038515618 < HREF_ŷ <= 0.01923272
# # Fit logistic coefficients: Float32[0.8676892, 0.3169111, 0.9392679]
# sref_newer_bin_3_4_predict(href_ŷ, sref_ŷ) = σ(0.8676892f0*logit(href_ŷ) + 0.3169111f0*logit(sref_ŷ) + 0.9392679f0)
# # Bin 4-5 --------
# # 0.00954726 < HREF_ŷ <= 0.035036117
# # Fit logistic coefficients: Float32[1.2143207, 0.26236853, 2.1367016]
# sref_newer_bin_4_5_predict(href_ŷ, sref_ŷ) = σ(1.2143207f0*logit(href_ŷ) + 0.26236853f0*logit(sref_ŷ) + 2.1367016f0)
# # Bin 5-6 --------
# # 0.01923272 < HREF_ŷ <= 1.0
# # Fit logistic coefficients: Float32[0.851503, 0.31797588, 1.0932944]
# sref_newer_bin_5_6_predict(href_ŷ, sref_ŷ) = σ(0.851503f0*logit(href_ŷ) + 0.31797588f0*logit(sref_ŷ) + 1.0932944f0)


# day_bins_logistic_coeffs = Vector{Float32}[Float32[0.8790791, 0.17466258, 0.42071092], Float32[0.5856237, 0.17571865, -0.84646785], Float32[1.4197825, 0.00979548, 0.91951996], Float32[1.5459903, -0.15001805, 0.5937435], Float32[1.1762913, -0.5233394, -1.4415414]]

# day_bin_1_2_predict(indep_events_ŷ, max_hourly_prob) = σ(day_bins_logistic_coeffs[1][1]*logit(indep_events_ŷ) + day_bins_logistic_coeffs[1][2]*logit(max_hourly_prob) + day_bins_logistic_coeffs[1][3])
# day_bin_2_3_predict(indep_events_ŷ, max_hourly_prob) = σ(day_bins_logistic_coeffs[2][1]*logit(indep_events_ŷ) + day_bins_logistic_coeffs[2][2]*logit(max_hourly_prob) + day_bins_logistic_coeffs[2][3])
# day_bin_3_4_predict(indep_events_ŷ, max_hourly_prob) = σ(day_bins_logistic_coeffs[3][1]*logit(indep_events_ŷ) + day_bins_logistic_coeffs[3][2]*logit(max_hourly_prob) + day_bins_logistic_coeffs[3][3])
# day_bin_4_5_predict(indep_events_ŷ, max_hourly_prob) = σ(day_bins_logistic_coeffs[4][1]*logit(indep_events_ŷ) + day_bins_logistic_coeffs[4][2]*logit(max_hourly_prob) + day_bins_logistic_coeffs[4][3])
# day_bin_5_6_predict(indep_events_ŷ, max_hourly_prob) = σ(day_bins_logistic_coeffs[5][1]*logit(indep_events_ŷ) + day_bins_logistic_coeffs[5][2]*logit(max_hourly_prob) + day_bins_logistic_coeffs[5][3])


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

  # ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

  # href_newer_predict(forecasts, data) = begin
  #   href_ŷs = @view data[:,1]
  #   sref_ŷs = @view data[:,2]

  #   out = Array{Float32}(undef, length(href_ŷs))

  #   bin_maxes = Float32[0.000962529, 0.0040673064, 0.009957244, 0.020302918, 0.037081156, 1.0]

  #   Threads.@threads for i in 1:length(href_ŷs)
  #     href_ŷ = href_ŷs[i]
  #     sref_ŷ = sref_ŷs[i]
  #     if href_ŷ <= bin_maxes[1]
  #       # Bin 1-2 predictor only
  #       ŷ = href_newer_bin_1_2_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[2]
  #       # Bin 1-2 and 2-3 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[1], bin_maxes[2])
  #       ŷ = ratio*href_newer_bin_2_3_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*href_newer_bin_1_2_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[3]
  #       # Bin 2-3 and 3-4 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[2], bin_maxes[3])
  #       ŷ = ratio*href_newer_bin_3_4_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*href_newer_bin_2_3_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[4]
  #       # Bin 3-4 and 4-5 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[3], bin_maxes[4])
  #       ŷ = ratio*href_newer_bin_4_5_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*href_newer_bin_3_4_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[5]
  #       # Bin 4-5 and 5-6 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[4], bin_maxes[5])
  #       ŷ = ratio*href_newer_bin_5_6_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*href_newer_bin_4_5_predict(href_ŷ, sref_ŷ)
  #     else
  #       # Bin 5-6 predictor only
  #       ŷ = href_newer_bin_5_6_predict(href_ŷ, sref_ŷ)
  #     end
  #     out[i] = ŷ
  #   end

  #   out
  # end

  # sref_newer_predict(forecast, data) = begin
  #   href_ŷs = @view data[:,1]
  #   sref_ŷs = @view data[:,2]

  #   out = Array{Float32}(undef, length(href_ŷs))

  #   bin_maxes = Float32[0.0009233353, 0.0038515618, 0.00954726, 0.01923272, 0.035036117, 1.0]

  #   Threads.@threads for i in 1:length(href_ŷs)
  #     href_ŷ = href_ŷs[i]
  #     sref_ŷ = sref_ŷs[i]
  #     if href_ŷ <= bin_maxes[1]
  #       # Bin 1-2 predictor only
  #       ŷ = sref_newer_bin_1_2_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[2]
  #       # Bin 1-2 and 2-3 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[1], bin_maxes[2])
  #       ŷ = ratio*sref_newer_bin_2_3_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*sref_newer_bin_1_2_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[3]
  #       # Bin 2-3 and 3-4 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[2], bin_maxes[3])
  #       ŷ = ratio*sref_newer_bin_3_4_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*sref_newer_bin_2_3_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[4]
  #       # Bin 3-4 and 4-5 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[3], bin_maxes[4])
  #       ŷ = ratio*sref_newer_bin_4_5_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*sref_newer_bin_3_4_predict(href_ŷ, sref_ŷ)
  #     elseif href_ŷ <= bin_maxes[5]
  #       # Bin 4-5 and 5-6 predictors
  #       ratio = ratio_between(href_ŷ, bin_maxes[4], bin_maxes[5])
  #       ŷ = ratio*sref_newer_bin_5_6_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*sref_newer_bin_4_5_predict(href_ŷ, sref_ŷ)
  #     else
  #       # Bin 5-6 predictor only
  #       ŷ = sref_newer_bin_5_6_predict(href_ŷ, sref_ŷ)
  #     end
  #     out[i] = ŷ
  #   end

  #   out
  # end

  # _forecasts_href_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_href_newer, href_newer_predict; model_name = "CombinedHREFSREF_hour_severe_probabilities_href_newer")
  # _forecasts_sref_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_sref_newer, sref_newer_predict; model_name = "CombinedHREFSREF_hour_severe_probabilities_sref_newer")


  # # Day forecasts

  # run_time_seconds_to_hourly_prediction_forecasts = Forecasts.run_time_seconds_to_forecasts(vcat(_forecasts_href_newer_combined,_forecasts_sref_newer_combined))

  # run_date = Dates.Date(2019, 1, 9)
  # associated_forecasts = []
  # while run_date <= Dates.Date(Dates.now(Dates.UTC))
  #   run_year  = Dates.year(run_date)
  #   run_month = Dates.month(run_date)
  #   run_day   = Dates.day(run_date)

  #   for run_hour in 0:3:21
  #     run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

  #     forecasts_for_run_time = get(run_time_seconds_to_hourly_prediction_forecasts, run_time_seconds, Forecasts.Forecast[])

  #     forecast_hours_in_convective_day = max(12-run_hour,2):clamp(23+12-run_hour,2,35)

  #     forecasts_for_convective_day = filter(forecast -> forecast.forecast_hour in forecast_hours_in_convective_day, forecasts_for_run_time)

  #     if (length(forecast_hours_in_convective_day) == length(forecasts_for_convective_day))
  #       push!(associated_forecasts, forecasts_for_convective_day)
  #     end

  #     # 1. Try both independent events total prob and max hourly prob as the main descriminator
  #     # 2. bin predictions into 10 bins of equal weight of positive labels
  #     # 3. combine bin-pairs (overlapping, 9 bins total)
  #     # 4. train a logistic regression for each bin,
  #     #   σ(a1*logit(independent events total prob) +
  #     #     a2*logit(max hourly prob) +
  #     #     a3*logit(2nd highest hourly prob) +
  #     #     a4*logit(3rd highest hourly prob) +
  #     #     a5*logit(4th highest hourly prob) +
  #     #     a6*logit(5th highest hourly prob) +
  #     #     a7*logit(6th highest hourly prob) +
  #     #     a8*logit(tornado day climatological prob) +
  #     #     a9*logit(tornado day given severe day climatological prob) +
  #     #     a10*logit(geomean(above two)) +
  #     #     a11*logit(tornado prob for given month) +
  #     #     a12*logit(tornado prob given severe day for given month) +
  #     #     a13*logit(geomean(above two)) +
  #     #     b)
  #     #   Check & eliminate terms via 3-fold cross-validation.
  #     # 5. prediction is weighted mean of the two overlapping logistic models
  #     # 6. should thereby be absolutely calibrated (check)
  #     # 7. calibrate to SPC thresholds (linear interpolation)
  #   end

  #   run_date += Dates.Day(1)
  # end

  # # Which run time and forecast hour to use for the set.
  # # Namely: latest run time, then longest forecast hour
  # choose_canonical_forecast(day_hourlies) = begin
  #   canonical = day_hourlies[1]
  #   for forecast in day_hourlies
  #     if (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)) > (Forecasts.run_time_in_seconds_since_epoch_utc(canonical), Forecasts.valid_time_in_seconds_since_epoch_utc(canonical))
  #       canonical = forecast
  #     end
  #   end
  #   canonical
  # end

  # day_hourly_predictions = ForecastCombinators.concat_forecasts(associated_forecasts, forecasts_tuple_to_canonical_forecast = choose_canonical_forecast)

  # day_inventory_transformer(base_forecast, base_inventory) = begin
  #   [ Inventories.InventoryLine("", "", base_inventory[1].date_str, "independent events total tornado probability", "calculated", "day fcst", "", "")
  #   , Inventories.InventoryLine("", "", base_inventory[1].date_str, "highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "2nd highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "3rd highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "4th highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "5th highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "6th highest hourly tornado probability", "calculated", "day fcst", "", "")
  #   ]
  # end

  # day_data_transformer(base_forecast, base_data) = begin
  #   point_count, hours_count = size(base_data)
  #   out = Array{Float32}(undef, (point_count, 2))
  #   Threads.@threads for i in 1:point_count
  #     # sorted_probs = sort((@view base_data[i,:]); rev = true)
  #     prob_no_tor = 1.0
  #     for hour_i in 1:hours_count
  #       prob_no_tor *= 1.0 - Float64(base_data[i,hour_i])
  #     end
  #     out[i,1] = Float32(1.0 - prob_no_tor)
  #     out[i,2] = maximum(@view base_data[i,:])
  #     # out[i,2] = sorted_probs[1]
  #     # out[i,3] = sorted_probs[2]
  #     # out[i,4] = sorted_probs[3]
  #     # out[i,5] = sorted_probs[4]
  #     # out[i,6] = sorted_probs[5]
  #     # out[i,7] = sorted_probs[6]
  #   end
  #   out
  # end

  # # Caching barely helps load times, so we don't do it

  # _forecasts_day_accumulators = ForecastCombinators.map_forecasts(day_hourly_predictions; inventory_transformer = day_inventory_transformer, data_transformer = day_data_transformer, model_name = "Day_severe_probability_accumulators_from_CombinedHREFSREF_hours")

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
