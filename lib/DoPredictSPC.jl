#  FORECASTS_ROOT_EXACT=... FORECASTS_OUTPUT_DIR=... RUN_HOUR=12 DAY1OR2=1 DRAW_PNG=true WGRIB2=/usr/bin/wgrib2 JULIA_NUM_THREADS=16 JULIA=/usr/local/julia/bin/julia ruby $HOME/nadocast/lib/forecast_only_spc.rb >> $HOME/nadocast/forecaster.log 2>&1
#
# To predict the past, also set RUN_DATE and RUN_HOUR in the environment.

import Dates
using Printf

push!(LOAD_PATH, @__DIR__)
import Forecasts
import ForecastCombinators
import Grids
import Grib2

grib2 = get(ENV, "OUTPUT_GRIB2", "true")  == "true"
draw  = get(ENV, "DRAW_PNG",     "false") == "true"

draw && import PlotMap # Was having trouble loading this on the SPC machine

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction")
import HREFPrediction

hourly_forecasts, fourhourly_forecasts, day1_forecasts, day1_absolutely_calibrated_forecasts, day2_forecasts, day2_absolutely_calibrated_forecasts =
  (
    HREFPrediction.forecasts_calibrated_with_sig_gated(),
    HREFPrediction.forecasts_fourhourly_with_sig_gated(),
    HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated(),
    HREFPrediction.forecasts_day_with_sig_gated(),
    HREFPrediction.forecasts_day2_spc_calibrated_with_sig_gated(),
    HREFPrediction.forecasts_day2_with_sig_gated()
  );

if get(ENV, "DAY1OR2", "1") == "1"
  forecasts, absolutely_calibrated_forecasts = day1_forecasts, day1_absolutely_calibrated_forecasts
elseif get(ENV, "DAY1OR2", "1") == "2"
  forecasts, absolutely_calibrated_forecasts = day2_forecasts, day2_absolutely_calibrated_forecasts
else
  println("DAY1OR2 should be 1 or 2!")
  exit(1)
end;

if haskey(ENV, "RUN_DATE")
  year, month, day = map(str -> parse(Int64, str), split(ENV["RUN_DATE"], "-"))
  filter!(forecasts) do forecast
    forecast.run_year == year && forecast.run_month == month && forecast.run_day == day
  end
end;
if haskey(ENV, "RUN_DATES")
  ymds =
    map(split(ENV["RUN_DATES"],",")) do date_str
      year, month, day = map(str -> parse(Int64, str), split(date_str, "-"))
      (year, month, day)
    end
  filter!(forecasts) do forecast
    any(ymds) do (year, month, day)
      forecast.run_year == year && forecast.run_month == month && forecast.run_day == day
    end
  end
end;
if haskey(ENV, "RUN_HOUR")
  run_hour = parse(Int64, ENV["RUN_HOUR"])
  filter!(forecasts) do forecast
    forecast.run_hour == run_hour
  end
end;
if haskey(ENV, "RUN_HOURS")
  run_hours = parse.(Int64, split(ENV["RUN_HOURS"],","))
  filter!(forecasts) do forecast
    forecast.run_hour in run_hours
  end
end;
sort!(forecasts, alg=MergeSort, by=Forecasts.run_utc_datetime);

if !haskey(ENV, "RUN_DATES")
  forecasts = [last(forecasts)]
end;


if forecasts == []
  exit(1)
end

this_file_dir = @__DIR__

function find(pred, arr)
  i = findfirst(pred, arr)
  isnothing(i) ? nothing : arr[i]
end

function do_forecast(forecast)

  run_year_month_day_hour               = Forecasts.run_year_month_day_hour(forecast)
  run_year_month_day_hour_forecast_hour = Forecasts.run_year_month_day_hour_forecast_hour(forecast)
  absolutely_calibrated_forecast = find(absolutely_calibrated_forecasts) do f
    Forecasts.run_year_month_day_hour_forecast_hour(f) == run_year_month_day_hour_forecast_hour
  end;

  forecasts_dir = get(ENV, "FORECASTS_OUTPUT_DIR", "$this_file_dir/../forecasts")
  # absolute file paths are too long for GMT, make them shorter by cd-ing
  cd(forecasts_dir)

  nadocast_run_time_utc = Forecasts.run_utc_datetime(forecast)
  nadocast_run_hour     = forecast.run_hour

  out_dir_daily      = "$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/"
  out_dir_hourly     = "$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/hourly/"
  out_dir_fourhourly = "$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/four-hourly/"

  non_sig_model_count = count(m -> !occursin("sig_", m[1]), HREFPrediction.models)

  function output_forecast(forecast; is_hourly, is_fourhourly, is_absolutely_calibrated = false)

    @assert !(is_hourly && is_fourhourly)

    predictions = Forecasts.data(forecast);

    period_stop_forecast_hour  = forecast.forecast_hour
    period_start_forecast_hour =
      if is_hourly
        forecast.forecast_hour
      elseif is_fourhourly
        forecast.forecast_hour - 3
      else
        max(2, period_stop_forecast_hour - 23)
      end
    f_str = is_hourly ? (@sprintf "%02d" forecast.forecast_hour) : "$((@sprintf "%02d" period_start_forecast_hour))-$((@sprintf "%02d" period_stop_forecast_hour))"

    out_dir =
      if is_hourly
        out_dir_hourly
      elseif is_fourhourly
        out_dir_fourhourly
      else
        out_dir_daily
      end
    mkpath(out_dir)

    for model_i in 1:non_sig_model_count
      event_name, _, _     = HREFPrediction.models[model_i]
      sig_model_i          = findfirst(m -> m[3] == "sig_$(event_name)_gated_by_$(event_name)", HREFPrediction.models_with_gated)
      sig_event_name, _, _ = HREFPrediction.models_with_gated[sig_model_i]
      calibration_blurb    = is_absolutely_calibrated ? "_abs_calib" : ""
      println("outputting$(grib2 ? " grib2" : "")$(draw ? " png" : "") for (sig_)$(event_name)$(calibration_blurb) f$(f_str)...")
      out_path_prefix      = out_dir *     "nadocast_2022_models_conus_$(event_name)$(calibration_blurb)_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"
      sig_out_path_prefix  = out_dir * "nadocast_2022_models_conus_$(sig_event_name)$(calibration_blurb)_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"
      period_path          = out_path_prefix     * "_f$(f_str)"
      sig_period_path      = sig_out_path_prefix * "_f$(f_str)"
      # write(period_path * ".float16.bin", Float16.(prediction))
      prediction           = @view predictions[:, model_i]
      sig_prediction       = @view predictions[:, sig_model_i]

      @sync begin
        if grib2
          @async Grib2.write_15km_HREF_probs_grib2(
            prediction;
            run_time = Forecasts.run_utc_datetime(forecast),
            forecast_hour = (is_hourly ? forecast.forecast_hour : (period_start_forecast_hour, period_stop_forecast_hour)),
            event_type = event_name,
            out_name = period_path * ".grib2",
          )
          @async Grib2.write_15km_HREF_probs_grib2(
            sig_prediction;
            run_time = Forecasts.run_utc_datetime(forecast),
            forecast_hour = (is_hourly ? forecast.forecast_hour : (period_start_forecast_hour, period_stop_forecast_hour)),
            event_type = sig_event_name,
            out_name = sig_period_path * ".grib2",
          )
        end

        if draw
          @async PlotMap.plot_fast(period_path,     forecast.grid, prediction;     val_to_color=PlotMap.event_name_to_colorer_more_sig_colors[event_name])
          @async PlotMap.plot_fast(sig_period_path, forecast.grid, sig_prediction; val_to_color=PlotMap.event_name_to_colorer_more_sig_colors[sig_event_name])
        end
      end
    end
  end

  ForecastCombinators.turn_forecast_caching_on()
  output_forecast(forecast;                       is_hourly = false, is_fourhourly = false)
  ForecastCombinators.clear_cached_forecasts()
  output_forecast(absolutely_calibrated_forecast; is_hourly = false, is_fourhourly = false, is_absolutely_calibrated = true)
  ForecastCombinators.clear_cached_forecasts()

  is_day1 = forecast.forecast_hour <= 35

  hourly_fourhourly_forecast_hour_range =
    if is_day1
      2:forecast.forecast_hour
    else
      (forecast.forecast_hour - 23):47
    end

  for fourhourly_forecast in fourhourly_forecasts
    if Forecasts.run_year_month_day_hour(fourhourly_forecast) == run_year_month_day_hour && fourhourly_forecast.forecast_hour in hourly_fourhourly_forecast_hour_range
      output_forecast(fourhourly_forecast; is_hourly = false, is_fourhourly = true, is_absolutely_calibrated = true)
    end
  end
  ForecastCombinators.clear_cached_forecasts()

  for hourly_forecast in hourly_forecasts
    if Forecasts.run_year_month_day_hour(hourly_forecast) == run_year_month_day_hour && hourly_forecast.forecast_hour in hourly_fourhourly_forecast_hour_range
      output_forecast(hourly_forecast; is_hourly = true, is_fourhourly = false, is_absolutely_calibrated = true)
    end
  end
  ForecastCombinators.clear_cached_forecasts()
end

for forecast in forecasts
  do_forecast(forecast)
end
