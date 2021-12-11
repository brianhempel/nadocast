import MemoryConstrainedTreeBoosting

# To predict the past, set FORECAST_DATE=2019-4-7 in the environment:
#
# $ FORECAST_DATE=2019-4-7 TWEET=true FORECASTS_ROOT=$(pwd)/test_grib2s make forecast
#
# or
#
# $ SREF_RUN_TIME=2021-4-22t21 HREF_RUN_TIME=2021-4-23t0 USE_HRRR=false USE_RAP=false make forecast
#
# Using data on remote machine:
# Make sure ~/.ssh/config doesn't use screen
# $ mkdir ~/nadocaster2
# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ SREF_RUN_TIME=2021-4-22t21 HREF_RUN_TIME=2021-4-23t0 USE_HRRR=false USE_RAP=false make forecast


import Dates
using Printf

push!(LOAD_PATH, @__DIR__)
import Forecasts
import ForecastCombinators
import Grids
import PlotMap

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_hrrr_rap_href_sref")
import CombinedHRRRRAPHREFSREF

newest_href_sref          = last(CombinedHREFSREF.forecasts_day_spc_calibrated());
newest_hrrr_rap_href_sref = last(filter(forecast -> forecast.run_hour >= 10, CombinedHRRRRAPHREFSREF.forecasts_day_spc_calibrated()));

if Forecasts.run_utc_datetime(newest_hrrr_rap_href_sref) >= Forecasts.run_utc_datetime(newest_href_sref)
  newest_forecast = newest_hrrr_rap_href_sref
else
  newest_forecast = newest_href_sref
end

# Follows Forecasts.based_on to return a list of forecasts with the given model_name
function model_parts(forecast, model_name)
  if forecast.model_name == model_name
    [forecast]
  else
    vcat(map(forecast -> model_parts(forecast, model_name), forecast.based_on)...)
  end
end

hrrr_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(newest_forecast, "HRRR")))
rap_run_hours  = unique(map(forecast -> forecast.run_hour, model_parts(newest_forecast, "RAP")))
href_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(newest_forecast, "HREF")))
sref_run_hours = unique(map(forecast -> forecast.run_hour, model_parts(newest_forecast, "SREF")))

if Dates.now(Dates.UTC) > Forecasts.valid_utc_datetime(newest_forecast)
  println("Newest forecast is in the past $(Forecasts.time_title(newest_forecast))")
  exit(1)
end

ForecastCombinators.turn_forecast_caching_on()
# ForecastCombinators.turn_forecast_gc_circumvention_on()
prediction = Forecasts.data(newest_forecast)
ForecastCombinators.clear_cached_forecasts()


period_stop_forecast_hour  = newest_forecast.forecast_hour
period_start_forecast_hour = max(2, period_stop_forecast_hour - 23)
nadocast_run_time_utc      = Forecasts.run_utc_datetime(newest_forecast)
nadocast_run_hour          = newest_forecast.run_hour


paths = []
daily_paths_to_perhaps_tweet = []

out_dir = (@__DIR__) * "/../forecasts/$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/t$(nadocast_run_hour)z/"
mkpath(out_dir)
out_path_prefix = out_dir * "nadocast_conus_tor_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"
period_path = out_path_prefix * "_f$((@sprintf "%02d" period_start_forecast_hour))-$((@sprintf "%02d" period_stop_forecast_hour))"
write(period_path * ".float16.bin", Float16.(prediction))
PlotMap.plot_map(
  period_path,
  newest_forecast.grid,
  prediction;
  run_time_utc = nadocast_run_time_utc,
  forecast_hour_range = period_start_forecast_hour:period_stop_forecast_hour,
  hrrr_run_hours = hrrr_run_hours,
  rap_run_hours  = rap_run_hours,
  href_run_hours = href_run_hours,
  sref_run_hours = sref_run_hours
)
push!(paths, period_path)
push!(daily_paths_to_perhaps_tweet, period_path)

# @sync doesn't seem to work; poll until subprocesses are done.
for path in paths
  while !isfile(path * ".pdf") || isfile(path * ".sh")
    sleep(1)
  end
end

# if !isnothing(animation_glob_path)
#   println("Making hourlies movie out of $(animation_glob_path)...")
#   hourlies_movie_path = out_path_prefix * "_hourlies"
#   run(`ffmpeg -framerate 2 -i "$(animation_glob_path)" -c:v libx264 -vf format=yuv420p,scale=1200:-1 $hourlies_movie_path.mp4`)
# end

if get(ENV, "TWEET", "false") == "true"
  tweet_script_path = (@__DIR__) * "/tweet.rb"

  for path in daily_paths_to_perhaps_tweet
    println("Tweeting daily $(path)...")
    run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Day Tornado Forecast" $path.png`)
  end

  # if !isnothing(animation_glob_path)
  #   println("Tweeting hourlies $(hourlies_movie_path)...")
  #   run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
  # end
end
