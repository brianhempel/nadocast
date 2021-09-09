module CombinedHRRRRAPHREFSREF

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

push!(LOAD_PATH, (@__DIR__) * "/../hrrr_prediction")
import HRRRPrediction

push!(LOAD_PATH, (@__DIR__) * "/../rap_prediction")
import RAPPrediction

push!(LOAD_PATH, (@__DIR__) * "/../href_prediction")
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../sref_prediction")
import SREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../combined_href_sref")
import CombinedHREFSREF


MINUTE = 60
HOUR   = 60*MINUTE

# Forecast run time is always the newest forecast (HRRR/RAP).

_forecasts_separate = [] # Output is a 8-feature forecast: 3 HRRRs, 3 RAPs, HREF, SREF
_forecasts = [] # Combined to single prediction

# For day, allow 0Z to 23Z runs
_forecasts_day_accumulators   = []
_forecasts_day                = []
_forecasts_day_spc_calibrated = []

function forecasts_separate()
  if isempty(_forecasts_separate)
    reload_forecasts()
    _forecasts_separate
  else
    _forecasts_separate
  end
end

# HREF 3 hours behind SREF
function forecasts()
  if isempty(_forecasts)
    reload_forecasts()
    _forecasts
  else
    _forecasts
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

function latest_within_6_hours(run_time_seconds_to_forecasts, run_time_seconds)
  for hour in 0:5
    latest_forecasts = get(run_time_seconds_to_forecasts, run_time_seconds - hour*HOUR, Forecasts.Forecast[])
    if length(latest_forecasts) > 0
      return latest_forecasts
    end
  end
  Forecasts.Forecast[]
end

ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

function predict_one(ŷs_i, coeffs)
  # coeffs = bins_logistic_coeffs[low_bin_i]
  logit_out = last(coeffs) # Constant term
  for coeff_i in 1:length(ŷs_i)
    logit_out += coeffs[coeff_i] * logit(ŷs_i[coeff_i])
  end
  σ(logit_out)
end

function make_combined_prediction(data; first_guess_feature_i, bin_maxes, bins_logistic_coeffs)
  first_guess_ŷs = @view data[:, first_guess_feature_i]

  out = Array{Float32}(undef, length(first_guess_ŷs))

  # bin_maxes = Float32[0.000962529, 0.0040673064, 0.009957244, 0.020302918, 0.037081156, 1.0]

  # bins_logistic_coeffs = Vector{Float32}[[0.2958114, 0.093472, 0.014974893, 0.25723657, -0.08649167, -0.03651152, 0.51546174, -0.17759001, -0.9377826], [0.55761576, -0.08032898, 0.029375209, -0.043275304, 0.2413465, -0.029677542, 0.5895104, -0.27158892, -0.28064802], [0.5240835, -0.15477853, 0.30952054, -0.16907471, 0.26656932, 0.04274331, 0.7430586, -0.30433056, 0.9741227], [0.4446033, -0.31768143, 0.402504, 0.055888213, 0.10179166, 0.26728168, 0.8905479, -0.28445807, 2.323705], [0.16820262, -0.698822, 0.5922206, 0.16202259, 0.031716842, 0.39105946, 0.7149798, -0.2696285, 0.8772054]]

  bcs = bins_logistic_coeffs

  Threads.@threads for i in 1:length(first_guess_ŷs)
    first_guess_ŷ = first_guess_ŷs[i]
    ŷs_i = @view data[i,:]
    if first_guess_ŷ <= bin_maxes[1]
      # Bin 1-2 predictor only
      ŷ = predict_one(ŷs_i, bcs[1])
    elseif first_guess_ŷ <= bin_maxes[2]
      # Bin 1-2 and 2-3 predictors
      ratio = ratio_between(first_guess_ŷ, bin_maxes[1], bin_maxes[2])
      ŷ = ratio*predict_one(ŷs_i, bcs[2]) + (1f0 - ratio)*predict_one(ŷs_i, bcs[1])
    elseif first_guess_ŷ <= bin_maxes[3]
      # Bin 2-3 and 3-4 predictors
      ratio = ratio_between(first_guess_ŷ, bin_maxes[2], bin_maxes[3])
      ŷ = ratio*predict_one(ŷs_i, bcs[3]) + (1f0 - ratio)*predict_one(ŷs_i, bcs[2])
    elseif first_guess_ŷ <= bin_maxes[4]
      # Bin 3-4 and 4-5 predictors
      ratio = ratio_between(first_guess_ŷ, bin_maxes[3], bin_maxes[4])
      ŷ = ratio*predict_one(ŷs_i, bcs[4]) + (1f0 - ratio)*predict_one(ŷs_i, bcs[3])
    elseif first_guess_ŷ <= bin_maxes[5]
      # Bin 4-5 and 5-6 predictors
      ratio = ratio_between(first_guess_ŷ, bin_maxes[4], bin_maxes[5])
      ŷ = ratio*predict_one(ŷs_i, bcs[5]) + (1f0 - ratio)*predict_one(ŷs_i, bcs[4])
    else
      # Bin 5-6 predictor only
      ŷ = predict_one(ŷs_i, bcs[5])
    end
    out[i] = ŷ
  end

  out
end

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_separate
  global _forecasts
  global _forecasts_day_accumulators
  global _forecasts_day
  global _forecasts_day_spc_calibrated

  _forecasts_separate = []

  hrrr_prediction_forecasts = ForecastCombinators.resample_forecasts(HRRRPrediction.forecasts_blurred(), Grids.get_upsampler, grid())
  rap_prediction_forecasts  = ForecastCombinators.resample_forecasts(RAPPrediction.forecasts_blurred(),  Grids.get_upsampler, grid())
  href_prediction_forecasts = HREFPrediction.forecasts_blurred()
  sref_prediction_forecasts = ForecastCombinators.resample_forecasts(SREFPrediction.forecasts_blurred(), Grids.get_interpolating_upsampler, grid())

  # Index to avoid O(n^2)

  run_time_seconds_to_hrrr_forecasts = Forecasts.run_time_seconds_to_forecasts(hrrr_prediction_forecasts)
  run_time_seconds_to_rap_forecasts  = Forecasts.run_time_seconds_to_forecasts(rap_prediction_forecasts)
  run_time_seconds_to_href_forecasts = Forecasts.run_time_seconds_to_forecasts(href_prediction_forecasts)
  run_time_seconds_to_sref_forecasts = Forecasts.run_time_seconds_to_forecasts(sref_prediction_forecasts)

  associated_forecasts = []

  run_date = Dates.Date(2019, 1, 9)
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in 0:23
      run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      # Finish times:
      # https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat_new/
      # HRRR is up to 1:45min after run time (b/c 0,6,12,18Z are longer)
      # RAP  up to 1:45
      # SREF up to 3:50
      # HREF up to 3:20
      # So the delta relative to HRRR is:
      # RAP -0
      # SREF -2:05
      # HREF -1:35

      hrrrs_for_run_time_minus0 = get(run_time_seconds_to_hrrr_forecasts, run_time_seconds - 0*HOUR, Forecasts.Forecast[])
      hrrrs_for_run_time_minus1 = get(run_time_seconds_to_hrrr_forecasts, run_time_seconds - 1*HOUR, Forecasts.Forecast[])
      hrrrs_for_run_time_minus2 = get(run_time_seconds_to_hrrr_forecasts, run_time_seconds - 2*HOUR, Forecasts.Forecast[])
      raps_for_run_time_minus0  = get(run_time_seconds_to_rap_forecasts,  run_time_seconds - 0*HOUR, Forecasts.Forecast[])
      raps_for_run_time_minus1  = get(run_time_seconds_to_rap_forecasts,  run_time_seconds - 1*HOUR, Forecasts.Forecast[])
      raps_for_run_time_minus2  = get(run_time_seconds_to_rap_forecasts,  run_time_seconds - 2*HOUR, Forecasts.Forecast[])
      hrefs_for_run_time = latest_within_6_hours(run_time_seconds_to_href_forecasts, run_time_seconds - 2*HOUR)
      srefs_for_run_time = latest_within_6_hours(run_time_seconds_to_sref_forecasts, run_time_seconds - 3*HOUR)


      for forecast_hour in 0:18 # 2:15 in practice
        valid_time_seconds = run_time_seconds + forecast_hour*HOUR

        function perhaps_forecast_valid_at(forecasts_for_run_time, valid_time_seconds)
          perhaps_forecasts = filter(forecast -> valid_time_seconds == Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), forecasts_for_run_time)
          if length(perhaps_forecasts) >= 2
            error("shouldn't have two matching forecasts!")
          end
          perhaps_forecasts
        end

        perhaps_hrrr_minus0_forecast = perhaps_forecast_valid_at(hrrrs_for_run_time_minus0, valid_time_seconds)
        perhaps_hrrr_minus1_forecast = perhaps_forecast_valid_at(hrrrs_for_run_time_minus1, valid_time_seconds)
        perhaps_hrrr_minus2_forecast = perhaps_forecast_valid_at(hrrrs_for_run_time_minus2, valid_time_seconds)
        perhaps_rap_minus0_forecast  = perhaps_forecast_valid_at(raps_for_run_time_minus0, valid_time_seconds)
        perhaps_rap_minus1_forecast  = perhaps_forecast_valid_at(raps_for_run_time_minus1, valid_time_seconds)
        perhaps_rap_minus2_forecast  = perhaps_forecast_valid_at(raps_for_run_time_minus2, valid_time_seconds)
        perhaps_href_forecast        = perhaps_forecast_valid_at(hrefs_for_run_time, valid_time_seconds)
        perhaps_sref_forecast        = perhaps_forecast_valid_at(srefs_for_run_time, valid_time_seconds)

        tuple = (perhaps_hrrr_minus0_forecast, perhaps_hrrr_minus1_forecast, perhaps_hrrr_minus2_forecast, perhaps_rap_minus0_forecast, perhaps_rap_minus1_forecast, perhaps_rap_minus2_forecast, perhaps_href_forecast, perhaps_sref_forecast)

        # If all perhapses are length 1
        if all(length.(tuple) .== 1)
          push!(associated_forecasts, first.(tuple))
        end
      end
    end

    run_date += Dates.Day(1)
  end

  _forecasts_separate = ForecastCombinators.concat_forecasts(associated_forecasts)

  # See Train.jl for where all these numbers come from
  predict(forecasts, data) = begin
    make_combined_prediction(
      data;
      first_guess_feature_i = 1,
      bin_maxes             = Float32[0.000962529, 0.0040673064, 0.009957244, 0.020302918, 0.037081156, 1.0],
      bins_logistic_coeffs  = Vector{Float32}[[0.2958114, 0.093472, 0.014974893, 0.25723657, -0.08649167, -0.03651152, 0.51546174, -0.17759001, -0.9377826], [0.55761576, -0.08032898, 0.029375209, -0.043275304, 0.2413465, -0.029677542, 0.5895104, -0.27158892, -0.28064802], [0.5240835, -0.15477853, 0.30952054, -0.16907471, 0.26656932, 0.04274331, 0.7430586, -0.30433056, 0.9741227], [0.4446033, -0.31768143, 0.402504, 0.055888213, 0.10179166, 0.26728168, 0.8905479, -0.28445807, 2.323705], [0.16820262, -0.698822, 0.5922206, 0.16202259, 0.031716842, 0.39105946, 0.7149798, -0.2696285, 0.8772054]]
    )
  end

  _forecasts = PredictionForecasts.simple_prediction_forecasts(_forecasts_separate, predict)

  run_time_seconds_to_hourly_prediction_forecasts =
    Forecasts.run_time_seconds_to_forecasts(_forecasts)

  run_time_seconds_to_hourly_href_sref_only_prediction_forecasts_href_newer =
    Forecasts.run_time_seconds_to_forecasts(CombinedHREFSREF.forecasts_href_newer_combined())

  run_time_seconds_to_hourly_href_sref_only_prediction_forecasts_sref_newer =
    Forecasts.run_time_seconds_to_forecasts(CombinedHREFSREF.forecasts_sref_newer_combined())

  run_date = Dates.Date(2019, 1, 9)
  associated_forecasts = []
  while run_date <= Dates.Date(Dates.now(Dates.UTC))
    run_year  = Dates.year(run_date)
    run_month = Dates.month(run_date)
    run_day   = Dates.day(run_date)

    for run_hour in 0:23
      run_time_seconds = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)

      # Finish times:
      # https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat_new/
      # HRRR is up to 1:45min after run time (b/c 0,6,12,18Z are longer)
      # RAP  up to 1:45
      # SREF up to 3:50
      # HREF up to 3:20
      # So the delta relative to HRRR is:
      # RAP -0
      # SREF -2:05
      # HREF -1:35

      forecasts_for_run_time = get(run_time_seconds_to_hourly_prediction_forecasts, run_time_seconds, Forecasts.Forecast[])

      combined_href_sref_href_newer_for_run_time =
        latest_within_6_hours(run_time_seconds_to_hourly_href_sref_only_prediction_forecasts_href_newer, run_time_seconds - 2*HOUR)
      combined_href_sref_sref_newer_for_run_time =
        latest_within_6_hours(run_time_seconds_to_hourly_href_sref_only_prediction_forecasts_sref_newer, run_time_seconds - 3*HOUR)

      forecast_hours_in_convective_day = max(12-run_hour,2):clamp(23+12-run_hour,2,35)

      forecasts_for_convective_day = Forecasts.Forecast[]

      for forecast_hour in forecast_hours_in_convective_day
        if forecast_hour <= 15
          forecast_i = findfirst(forecast -> forecast.forecast_hour == forecast_hour, forecasts_for_run_time)
          if isnothing(forecast_i)
            break # abort
          else
            push!(forecasts_for_convective_day, forecasts_for_run_time[forecast_i])
          end
        else
          valid_time_seconds = run_time_seconds + forecast_hour*HOUR
          href_newer_forecast_i = findfirst(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) == valid_time_seconds, combined_href_sref_href_newer_for_run_time)
          sref_newer_forecast_i = findfirst(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) == valid_time_seconds, combined_href_sref_sref_newer_for_run_time)
          if isnothing(href_newer_forecast_i) && isnothing(sref_newer_forecast_i)
            break # abort
          elseif isnothing(sref_newer_forecast_i)
            push!(forecasts_for_convective_day, combined_href_sref_href_newer_for_run_time[href_newer_forecast_i])
          elseif isnothing(href_newer_forecast_i)
            push!(forecasts_for_convective_day, combined_href_sref_sref_newer_for_run_time[sref_newer_forecast_i])
          elseif Forecasts.run_time_in_seconds_since_epoch_utc(combined_href_sref_href_newer_for_run_time[href_newer_forecast_i]) > Forecasts.run_time_in_seconds_since_epoch_utc(combined_href_sref_sref_newer_for_run_time[sref_newer_forecast_i])
            push!(forecasts_for_convective_day, combined_href_sref_href_newer_for_run_time[href_newer_forecast_i])
          else
            push!(forecasts_for_convective_day, combined_href_sref_sref_newer_for_run_time[sref_newer_forecast_i])
          end
        end
      end

      if (length(forecast_hours_in_convective_day) == length(forecasts_for_convective_day))
        push!(associated_forecasts, forecasts_for_convective_day)
      end
    end

    run_date += Dates.Day(1)
  end

  # # Which run time and forecast hour to use for the set.
  # # Namely: latest run time (from HRRR), then forecast hour corresponding to longest HREF/SREF
  make_canonical_forecast(day_hourlies) = begin
    latest_run_time_forecast   = day_hourlies[1]
    latest_valid_time_forecast = day_hourlies[1]
    for forecast in day_hourlies
      if (Forecasts.run_time_in_seconds_since_epoch_utc(forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)) > (Forecasts.run_time_in_seconds_since_epoch_utc(latest_run_time_forecast), Forecasts.valid_time_in_seconds_since_epoch_utc(latest_run_time_forecast))
        latest_run_time_forecast = forecast
      end
      if (Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), Forecasts.run_time_in_seconds_since_epoch_utc(forecast)) > (Forecasts.valid_time_in_seconds_since_epoch_utc(latest_valid_time_forecast), Forecasts.run_time_in_seconds_since_epoch_utc(latest_valid_time_forecast))
        latest_valid_time_forecast = forecast
      end
    end
    forecast_hour = (Forecasts.valid_time_in_seconds_since_epoch_utc(latest_valid_time_forecast) - Forecasts.run_time_in_seconds_since_epoch_utc(latest_run_time_forecast)) ÷ HOUR

    Forecasts.Forecast("", latest_run_time_forecast.run_year, latest_run_time_forecast.run_month, latest_run_time_forecast.run_day, latest_run_time_forecast.run_hour, forecast_hour, [], latest_run_time_forecast.grid, () -> [], () -> [], [])
  end

  day_hourly_predictions = ForecastCombinators.concat_forecasts(associated_forecasts, forecasts_tuple_to_canonical_forecast = make_canonical_forecast)

  day_inventory_transformer(base_forecast, base_inventory) = begin
    [ Inventories.InventoryLine("", "", base_inventory[1].date_str, "independent events total tornado probability", "calculated", "day fcst", "", "")
    , Inventories.InventoryLine("", "", base_inventory[1].date_str, "highest hourly tornado probability", "calculated", "day fcst", "", "")
    # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "2nd highest hourly tornado probability", "calculated", "day fcst", "", "")
    # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "3rd highest hourly tornado probability", "calculated", "day fcst", "", "")
    # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "4th highest hourly tornado probability", "calculated", "day fcst", "", "")
    # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "5th highest hourly tornado probability", "calculated", "day fcst", "", "")
    # , Inventories.InventoryLine("", "", base_inventory[1].date_str, "6th highest hourly tornado probability", "calculated", "day fcst", "", "")
    ]
  end

  day_data_transformer(base_forecast, base_data) = begin
    point_count, hours_count = size(base_data)
    out = Array{Float32}(undef, (point_count, 2))
    Threads.@threads for i in 1:point_count
      # sorted_probs = sort((@view base_data[i,:]); rev = true)
      prob_no_tor = 1.0
      for hour_i in 1:hours_count
        prob_no_tor *= 1.0 - Float64(base_data[i,hour_i])
      end
      out[i,1] = Float32(1.0 - prob_no_tor)
      out[i,2] = maximum(@view base_data[i,:])
      # out[i,2] = sorted_probs[1]
      # out[i,3] = sorted_probs[2]
      # out[i,4] = sorted_probs[3]
      # out[i,5] = sorted_probs[4]
      # out[i,6] = sorted_probs[5]
      # out[i,7] = sorted_probs[6]
    end
    out
  end

  _forecasts_day_accumulators = ForecastCombinators.map_forecasts(day_hourly_predictions; inventory_transformer = day_inventory_transformer, data_transformer = day_data_transformer)

  # See TrainDay.jl for where all these numbers come from
  day_predict(forecasts, data) = begin
    make_combined_prediction(
      data;
      first_guess_feature_i = 1,
      bin_maxes             = Float32[0.008764716, 0.028725764, 0.097140715, 0.18266037, 0.2817919, 1.0],
      bins_logistic_coeffs  = Vector{Float32}[[0.9605998, 0.06898633, -0.13992947], [0.29916266, 0.38762733, -1.1065903], [0.91976273, 0.3145933, 0.35971346], [1.017148, 0.17090763, 0.1382908], [0.6610863, 0.0075369733, -0.6788495]]
    )
  end

  _forecasts_day = PredictionForecasts.simple_prediction_forecasts(_forecasts_day_accumulators, day_predict)

  # spc_calibration = [
  #   (0.02, 0.016253397),
  #   (0.05, 0.0649308),
  #   (0.1,  0.18771306),
  #   (0.15, 0.28330332),
  #   (0.3,  0.32384455),
  # ]

  # _forecasts_day_spc_calibrated = PredictionForecasts.calibrated_forecasts(_forecasts_day, spc_calibration)

  ()
end

end # module CombinedHRRRRAPHREFSREF
