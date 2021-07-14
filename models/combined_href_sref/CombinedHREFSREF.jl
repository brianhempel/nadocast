# Output is a 2-feature forecast: layer 1 is the HREF-based prediction, layer 2 is the SREF-based prediction
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

_forecasts_href_newer = []
_forecasts_sref_newer = []
_forecasts_href_newer_combined = []

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


function example_forecast()
  forecasts()[1]
end

function grid()
  HREF.grid()
end


σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

# Bin 1-2 --------
# -1.0 < HREF_ŷ <= 0.0040673064
# Fit logistic coefficients: Float32[0.77540565, 0.19299681, -0.21271989]
bin_1_2_predict(href_ŷ, sref_ŷ) = σ(0.77540565*logit(href_ŷ) + 0.19299681*logit(sref_ŷ) + -0.21271989f0)
# Bin 2-3 --------
# 0.000962529 < HREF_ŷ <= 0.009957244
# Fit logistic coefficients: Float32[0.84564245, 0.14841641, -0.06817224]
bin_2_3_predict(href_ŷ, sref_ŷ) = σ(0.84564245*logit(href_ŷ) + 0.14841641*logit(sref_ŷ) + -0.06817224)
# Bin 3-4 --------
# 0.0040673064 < HREF_ŷ <= 0.020302918
# Fit logistic coefficients: Float32[0.9977281, 0.14388186, 0.64254296]
bin_3_4_predict(href_ŷ, sref_ŷ) = σ(0.9977281*logit(href_ŷ) + 0.14388186*logit(sref_ŷ) + 0.64254296)
# Bin 4-5 --------
# 0.009957244 < HREF_ŷ <= 0.037081156
# Fit logistic coefficients: Float32[1.3795987, 0.091625534, 1.9759048]
bin_4_5_predict(href_ŷ, sref_ŷ) = σ(1.3795987*logit(href_ŷ) + 0.091625534*logit(sref_ŷ) + 1.9759048)
# Bin 5-6 --------
# 0.020302918 < HREF_ŷ <= 1.0
# Fit logistic coefficients: Float32[0.9358031, 0.1812378, 0.836498]
bin_5_6_predict(href_ŷ, sref_ŷ) = σ(0.9358031*logit(href_ŷ) + 0.1812378*logit(sref_ŷ) + 0.836498)

function reload_forecasts()
  # href_paths = Grib2.all_grib2_file_paths_in("/Volumes/SREF_HREF_1/href")

  global _forecasts_href_newer
  global _forecasts_sref_newer
  global _forecasts_href_newer_combined

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

  _forecasts_href_newer = ForecastCombinators.concat_forecasts(paired_href_newer)
  _forecasts_sref_newer = ForecastCombinators.concat_forecasts(paired_sref_newer)

  ratio_between(x, lo, hi) = (x - lo) / (hi - lo)

  predict(forecasts, data) = begin
    href_ŷs = @view data[:,1]
    sref_ŷs = @view data[:,2]

    out = Array{Float32}(undef, length(href_ŷs))

    bin_maxes = Float32[0.000962529, 0.0040673064, 0.009957244, 0.020302918, 0.037081156, 1.0]

    Threads.@threads for i in 1:length(href_ŷs)
      href_ŷ = href_ŷs[i]
      sref_ŷ = sref_ŷs[i]
      if href_ŷ <= bin_maxes[1]
        # Bin 1-2 predictor only
        ŷ = bin_1_2_predict(href_ŷ, sref_ŷ)
      elseif href_ŷ <= bin_maxes[2]
        # Bin 1-2 and 2-3 predictors
        ratio = ratio_between(href_ŷ, bin_maxes[1], bin_maxes[2])
        ŷ = ratio*bin_2_3_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*bin_1_2_predict(href_ŷ, sref_ŷ)
      elseif href_ŷ <= bin_maxes[3]
        # Bin 2-3 and 3-4 predictors
        ratio = ratio_between(href_ŷ, bin_maxes[2], bin_maxes[3])
        ŷ = ratio*bin_3_4_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*bin_2_3_predict(href_ŷ, sref_ŷ)
      elseif href_ŷ <= bin_maxes[4]
        # Bin 3-4 and 4-5 predictors
        ratio = ratio_between(href_ŷ, bin_maxes[3], bin_maxes[4])
        ŷ = ratio*bin_4_5_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*bin_3_4_predict(href_ŷ, sref_ŷ)
      elseif href_ŷ <= bin_maxes[5]
        # Bin 4-5 and 5-6 predictors
        ratio = ratio_between(href_ŷ, bin_maxes[4], bin_maxes[5])
        ŷ = ratio*bin_5_6_predict(href_ŷ, sref_ŷ) + (1f0 - ratio)*bin_4_5_predict(href_ŷ, sref_ŷ)
      else
        # Bin 5-6 predictor only
        ŷ = bin_5_6_predict(href_ŷ, sref_ŷ)
      end
      out[i] = ŷ
    end

    out
  end

  _forecasts_href_newer_combined = PredictionForecasts.simple_prediction_forecasts(_forecasts_href_newer, predict)

  ()
end

end # module StackedHREFSREF