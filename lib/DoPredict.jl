# To predict the past, set RUN_DATE and RUN_HOUR in the environment:
#
# $ RUN_DATE=2019-4-7 RUN_HOUR=10 TWEET=true FORECASTS_ROOT=$(pwd)/test_grib2s make forecast
#
# Using data on remote machine:
# Make sure ~/.ssh/config doesn't use screen
# $ mkdir ~/nadocaster2
# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ RUN_DATE=2019-4-7 RUN_HOUR=0 HRRR_RAP=false make forecast

# FORECASTS_ROOT=/home/brian/nadocaster2/ RUN_DATE=2022-4-5 RUN_HOUR=0 HRRR_RAP=false JULIA_NUM_THREADS=$CORE_COUNT FORECAST_DISK_PREFETCH=false julia --project=. lib/DoPredict.jl

# FORECASTS_ROOT=~/nadocaster2 PUBLISH=true TWEET=false RUN_HOURS=0 DAY1OR2=1 HRRR_RAP=false RUN_DATES=2022-10-13,2022-10-14,2022-10-15,2022-10-16,2022-10-17,2022-10-18,2022-10-19 JULIA_NUM_THREADS=$CORE_COUNT julia --project lib/DoPredict.jl
# FORECASTS_ROOT=~/nadocaster2 PUBLISH=false TWEET=false RUN_HOURS=12 DAY1OR2=1 HRRR_RAP=false RUN_DATES=2022-10-13,2022-10-14,2022-10-15,2022-10-16,2022-10-17,2022-10-18,2022-10-19 JULIA_NUM_THREADS=$CORE_COUNT julia --project lib/DoPredict.jl

import Dates
using Printf

push!(LOAD_PATH, @__DIR__)
import Forecasts
import ForecastCombinators
import Grids
import Grib2
import PlotMap

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction")
import HREFPrediction
import HREFPrediction2024

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_hrrr_rap_href_sref")
import CombinedHRRRRAPHREFSREF

model_year = get(ENV, "NADOCAST_MODEL", "2022")

models, models_with_gated, day1_forecasts, day1_absolutely_calibrated_forecasts, day2_forecasts, day2_absolutely_calibrated_forecasts =
  if get(ENV, "HRRR_RAP", "false") == "false"
    if model_year == "2022"
      (CombinedHREFSREF.models, CombinedHREFSREF.models_with_gated, CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFPrediction.forecasts_day2_spc_calibrated_with_sig_gated(), HREFPrediction.forecasts_day2_with_sig_gated())
    elseif model_year == "2024_preliminary"
      # Only absolutely calibrated for the moment
      (HREFPrediction2024.models, HREFPrediction2024.models_with_gated, HREFPrediction2024.forecasts_day_with_sig_gated(), HREFPrediction2024.forecasts_day_with_sig_gated(), HREFPrediction2024.forecasts_day2_with_sig_gated(), HREFPrediction2024.forecasts_day2_with_sig_gated())
    else
      (Forecasts.Forecast[], Forecasts.Forecast[], Forecasts.Forecast[], Forecasts.Forecast[])
    end
  else
    (Forecasts.Forecast[], Forecasts.Forecast[], Forecasts.Forecast[], Forecasts.Forecast[])
    # [CombinedHREFSREF.forecasts_day_spc_calibrated(); CombinedHRRRRAPHREFSREF.forecasts_day_spc_calibrated()]
  end;

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

  # day2_forecast = find(day2_forecasts) do f
  #   Forecasts.run_year_month_day_hour(f) == run_year_month_day_hour && f.forecast_hour == forecast.forecast_hour + 24
  # end;
  # day2_absolutely_calibrated_forecast = find(day2_absolutely_calibrated_forecasts) do f
  #   Forecasts.run_year_month_day_hour(f) == run_year_month_day_hour && f.forecast_hour == forecast.forecast_hour + 24
  # end;

  plotting_paths = []
  daily_paths_to_perhaps_tweet = []
  rsync_dirs = []

  forecasts_dir = "$this_file_dir/../forecasts"
  # absolute file paths are too long for GMT, make them shorter by cd-ing
  cd(forecasts_dir)
  changelog_file = "CHANGELOG.txt"

  nadocast_run_time_utc      = Forecasts.run_utc_datetime(forecast)
  nadocast_run_hour          = forecast.run_hour

  out_dir_daily      = "$(Dates.format(nadocast_run_time_utc, "yyyymm"))/$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/t$(nadocast_run_hour)z/"
  out_dir_hourly     = "$(Dates.format(nadocast_run_time_utc, "yyyymm"))/$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/t$(nadocast_run_hour)z/hourly/"
  out_dir_fourhourly = "$(Dates.format(nadocast_run_time_utc, "yyyymm"))/$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/t$(nadocast_run_hour)z/four-hourly/"

  rsync_dir = "$(Dates.format(nadocast_run_time_utc, "yyyymm"))"
  push!(rsync_dirs, rsync_dir)

  non_sig_model_is = findall(m -> !occursin("sig_", m[1]), models)

  function plot_forecast(forecast; is_hourly, is_fourhourly, is_absolutely_calibrated = false, grib2 = true, draw = true, pdf = true)

    @assert !(is_hourly && is_fourhourly)

    # Follows Forecasts.based_on to return a list of forecasts with the given model_name
    function model_parts(forecast, model_name)
      if forecast.model_name == model_name
        [forecast]
      else
        vcat(map(forecast -> model_parts(forecast, model_name), forecast.based_on)...)
      end
    end

    hrrr_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(forecast, "HRRR")))
    rap_run_hours  = unique(map(forecast -> forecast.run_hour, model_parts(forecast, "RAP")))
    href_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(forecast, "HREF")))
    sref_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(forecast, "SREF")))

    ForecastCombinators.turn_forecast_caching_on()
    # ForecastCombinators.turn_forecast_gc_circumvention_on()
    predictions = Forecasts.data(forecast);
    ForecastCombinators.clear_cached_forecasts()

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

    is_day1 = period_start_forecast_hour <= 12

    out_dir =
      if is_hourly
        out_dir_hourly
      elseif is_fourhourly
        out_dir_fourhourly
      else
        out_dir_daily
      end
    mkpath(out_dir)

    for model_i in non_sig_model_is
      event_name, _, _     = models[model_i]
      calibration_blurb    = is_absolutely_calibrated ? "_abs_calib" : ""
      println("plotting$(grib2 ? " grib2" : "")$(draw ? " png" : "")$(draw && pdf ? " pdf" : "") for (sig_)$(event_name)$(calibration_blurb) f$(f_str)...")
      out_path_prefix      = out_dir * "nadocast_$(model_year)_models_conus_$(event_name)$(calibration_blurb)_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"
      period_path          = out_path_prefix     * "_f$(f_str)"
      # write(period_path * ".float16.bin", Float16.(prediction))
      prediction           = @view predictions[:, model_i]
      sig_model_i          = findfirst(m -> m[3] == "sig_$(event_name)_gated_by_$(event_name)", models_with_gated)
      if !isnothing(sig_model_i)
        sig_event_name, _, _ = models_with_gated[sig_model_i]
        sig_out_path_prefix  = out_dir * "nadocast_$(model_year)_models_conus_$(sig_event_name)$(calibration_blurb)_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"
        sig_period_path      = sig_out_path_prefix * "_f$(f_str)"
        sig_prediction       = @view predictions[:, sig_model_i]
      end

      if grib2
        Grib2.write_15km_HREF_probs_grib2(
          prediction;
          run_time = Forecasts.run_utc_datetime(forecast),
          forecast_hour = (is_hourly ? forecast.forecast_hour : (period_start_forecast_hour, period_stop_forecast_hour)),
          event_type = event_name,
          out_name = period_path * ".grib2",
        )
        !isnothing(sig_model_i) && Grib2.write_15km_HREF_probs_grib2(
          sig_prediction;
          run_time = Forecasts.run_utc_datetime(forecast),
          forecast_hour = (is_hourly ? forecast.forecast_hour : (period_start_forecast_hour, period_stop_forecast_hour)),
          event_type = sig_event_name,
          out_name = sig_period_path * ".grib2",
        )
      end

      if draw
        PlotMap.plot_map(
          period_path,
          forecast.grid,
          prediction;
          sig_vals = !isnothing(sig_model_i) ? sig_prediction : nothing,
          event_title = Dict("tornado" => "Tor", "wind" => "Wind", "wind_adj" => "Wind Adjusted", "hail" => "Hail", "tornado_life_risk" => "Tor Life Risk")[event_name],
          models_str = "$(model_year) Models$(is_absolutely_calibrated ? ", Absolutely Calibrated" : "")",
          run_time_utc = nadocast_run_time_utc,
          forecast_hour_range = period_start_forecast_hour:period_stop_forecast_hour,
          hrrr_run_hours = hrrr_run_hours,
          rap_run_hours  = rap_run_hours,
          href_run_hours = href_run_hours,
          sref_run_hours = sref_run_hours,
          pdf = pdf
        )
        push!(plotting_paths, period_path)
        !isnothing(sig_model_i) && PlotMap.plot_map(
          sig_period_path,
          forecast.grid,
          sig_prediction;
          event_title = Dict("sig_tornado" => "Sigtor", "sig_wind" => "Sigwind", "sig_wind_adj" => "Sigwind Adjusted", "sig_hail" => "Sighail")[sig_event_name],
          models_str = "$(model_year) Models$(is_absolutely_calibrated ? ", Absolutely Calibrated" : "")",
          run_time_utc = nadocast_run_time_utc,
          forecast_hour_range = period_start_forecast_hour:period_stop_forecast_hour,
          hrrr_run_hours = hrrr_run_hours,
          rap_run_hours  = rap_run_hours,
          href_run_hours = href_run_hours,
          sref_run_hours = sref_run_hours,
          pdf = pdf
        )
        !isnothing(sig_model_i) && push!(plotting_paths, sig_period_path)
      end

      if event_name == "tornado" && !is_hourly && !is_fourhourly && !is_absolutely_calibrated
        push!(daily_paths_to_perhaps_tweet, period_path)
      end
    end
  end

  if model_year != "2024_preliminary"
    plot_forecast(forecast; is_hourly = false, is_fourhourly = false)
  end
  plot_forecast(absolutely_calibrated_forecast; is_hourly = false, is_fourhourly = false, is_absolutely_calibrated = true, draw = true, pdf = false)

  # @sync doesn't seem to work; poll until subprocesses are done.
  for path in plotting_paths
    while !isfile(path * ".png") || isfile(path * ".sh")
      sleep(1)
    end
  end
  plotting_paths = []

  should_publish = get(ENV, "PUBLISH", "false") == "true"
  if should_publish
    rsync_processes = map(rsync_dir -> run(`rsync -r --update --perms --chmod=a+rx $changelog_file $rsync_dir web@data.nadocast.com:\~/forecasts/`; wait = false), unique(rsync_dirs))
  end

  is_day1 = forecast.forecast_hour <= 35

  if get(ENV, "TWEET", "false") == "true"
    tweet_script_path = (@__DIR__) * "/tweet.rb"

    valid_date = convert(Dates.Date, Forecasts.valid_utc_datetime(forecast))  - Dates.Day(1)

    tweet_str =
      if Dates.now(Dates.UTC) > Forecasts.valid_utc_datetime(forecast)
        "$(nadocast_run_hour)Z Day $(is_day1 ? "" : "2 ")Tornado Reforecast for $(Dates.format(valid_date, "yyyy-m-d")) (New New $(model_year) Models)"
      elseif is_day1
        "$(nadocast_run_hour)Z Day Tornado Forecast (New New $(model_year) Models)"
      else
        "$(nadocast_run_hour)Z Day 2 Tornado Forecast for $(Dates.format(valid_date, "yyyy-m-d")) (New New $(model_year) Models)"
      end

    for path in daily_paths_to_perhaps_tweet
      println("Tweeting daily $(path)...")
      run(`ruby $tweet_script_path "$(tweet_str)" $path.png`)
    end

    # if !isnothing(animation_glob_path)
    #   println("Tweeting hourlies $(hourlies_movie_path)...")
    #   run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
    # end
  end

  should_publish && map(wait, rsync_processes)

  # if is_day1
  #   # Make grib2s for the hourlies/four-hourlies but don't draw yet because that takes an extra ~9mins.

  #   if get(ENV, "HRRR_RAP", "true") == "false"
  #     hourly_forecasts = filter(CombinedHREFSREF.forecasts_href_newer_combined_with_sig_gated()) do hourly_forecast
  #       Forecasts.run_year_month_day_hour(hourly_forecast) == run_year_month_day_hour
  #     end
  #     hourly_forecasts = sort(hourly_forecasts; by=(f -> f.forecast_hour))
  #   else
  #     hourly_forecasts = []
  #   end

  #   for hourly_forecast in hourly_forecasts
  #     plot_forecast(hourly_forecast; is_hourly = true, is_fourhourly = false, is_absolutely_calibrated = true, grib2 = true, draw = false, pdf = false)
  #   end

  #   if get(ENV, "HRRR_RAP", "true") == "false"
  #     fourhourly_forecasts = filter(CombinedHREFSREF.forecasts_fourhourly_with_sig_gated()) do fourhourly_forecast
  #       Forecasts.run_year_month_day_hour(fourhourly_forecast) == run_year_month_day_hour
  #     end
  #     fourhourly_forecasts = sort(fourhourly_forecasts; by=(f -> f.forecast_hour))
  #   else
  #     fourhourly_forecasts = []
  #   end

  #   for fourhourly_forecast in fourhourly_forecasts
  #     plot_forecast(fourhourly_forecast; is_hourly = false, is_fourhourly = true, is_absolutely_calibrated = true, grib2 = true, draw = false, pdf = false)
  #   end

  #   if should_publish
  #     rsync_processes = map(rsync_dir -> run(`rsync -r --update --perms --chmod=a+rx $changelog_file $rsync_dir web@data.nadocast.com:\~/forecasts/`; wait = false), unique(rsync_dirs))
  #     map(wait, rsync_processes)
  #   end

  #   # Now draw the hourlies & four-hourlies

  #   for hourly_forecast in hourly_forecasts
  #     plot_forecast(hourly_forecast; is_hourly = true, is_fourhourly = false, is_absolutely_calibrated = true, grib2 = false, pdf = false)
  #   end

  #   if hourly_forecasts != []
  #     animation_glob_paths = map(CombinedHREFSREF.models) do (event_name, _, _)
  #       out_dir_hourly * "nadocast_conus_$(event_name)_abs_calib_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z_f%02d.png"
  #     end
  #   else
  #     animation_glob_paths = []
  #   end

  #   # @sync doesn't seem to work; poll until subprocesses are done.
  #   for path in plotting_paths
  #     while !isfile(path * ".png") || isfile(path * ".sh")
  #       sleep(1)
  #     end
  #   end
  #   plotting_paths = []

  #   for animation_glob_path in animation_glob_paths
  #     println("Making hourlies movie out of $(animation_glob_path)...")
  #     hourly_movie_path = replace(animation_glob_path, "_f%02d.png" => "_hourly.mp4", out_dir_hourly => out_dir_daily)
  #     run(`ffmpeg -y -framerate 2 -i "$(animation_glob_path)" -c:v libx264 -vf format=yuv420p,scale=1200:-1 $hourly_movie_path`)
  #   end

  #   # if get(ENV, "TWEET", "false") == "true"
  #   #   if !isnothing(animation_glob_path)
  #   #     println("Tweeting hourlies $(hourlies_movie_path)...")
  #   #     run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
  #   #   end
  #   # end

  #   for fourhourly_forecast in fourhourly_forecasts
  #     plot_forecast(fourhourly_forecast; is_hourly = false, is_fourhourly = true, is_absolutely_calibrated = true, grib2 = false, pdf = false)
  #   end

  #   if fourhourly_forecasts != []
  #     animation_glob_paths = map(CombinedHREFSREF.models) do (event_name, _, _)
  #       out_dir_fourhourly * "nadocast_conus_$(event_name)_abs_calib_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z_f*-*.png"
  #     end
  #   else
  #     animation_glob_paths = []
  #   end

  #   # @sync doesn't seem to work; poll until subprocesses are done.
  #   for path in plotting_paths
  #     while !isfile(path * ".png") || isfile(path * ".sh")
  #       sleep(1)
  #     end
  #   end
  #   plotting_paths = []

  #   for animation_glob_path in animation_glob_paths
  #     println("Making four-hourlies movie out of $(animation_glob_path)...")
  #     fourhourly_movie_path = replace(animation_glob_path, "_f*-*.png" => "_four-hourly.mp4", out_dir_fourhourly => out_dir_daily)
  #     run(`ffmpeg -y -framerate 2 -pattern_type glob -i "$(animation_glob_path)" -c:v libx264 -vf format=yuv420p,scale=1200:-1 $fourhourly_movie_path`)
  #   end

  #   if should_publish
  #     rsync_processes = map(rsync_dir -> run(`rsync -r --perms --update --chmod=a+rx $changelog_file $rsync_dir web@data.nadocast.com:\~/forecasts/`; wait = false), unique(rsync_dirs))
  #   end

  #   # if get(ENV, "TWEET", "false") == "true"
  #   #   if !isnothing(animation_glob_path)
  #   #     println("Tweeting hourlies $(hourlies_movie_path)...")
  #   #     run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
  #   #   end
  #   # end

  #   should_publish && map(wait, rsync_processes)
  # end
end

for forecast in forecasts
  do_forecast(forecast)
end
