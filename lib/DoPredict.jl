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
import Grids
import PlotMap

# println("Loading SREF module")
push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

newest_forecast = last(CombinedHREFSREF.forecasts_day_spc_calibrated())

if Dates.now(Dates.UTC) > Forecasts.valid_utc_datetime(newest_forecast)
  println("Newest forecast is in the past $(Forecasts.time_title(newest_forecast))")
  exit(1)
end

prediction = Forecasts.data(newest_forecast)

period_stop_forecast_hour  = newest_forecast.forecast_hour
period_start_forecast_hour = max(2, period_stop_forecast_hour - 23)
nadocast_run_time_utc      = Forecasts.run_utc_datetime(newest_forecast)
nadocast_run_hour          = newest_forecast.run_hour

# HREF newer for 0Z 6Z 12Z 18Z, SREF newer for 3Z 9Z 15Z 21Z
href_run_hours, sref_run_hours =
  if newest_forecast.run_hour in [0,6,12,18]
    ([newest_forecast.run_hour],[mod(newest_forecast.run_hour-3,24)])
  elseif newest_forecast.run_hour in [3,9,15,21]
    ([mod(newest_forecast.run_hour-3,24)],[newest_forecast.run_hour])
  else
    ([-1],[-1])
  end

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
  hrrr_run_hours = [],
  rap_run_hours  = [],
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
    run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Day Tornado Forecast (new new models, calibrated)" $path.png`)
  end

  # if !isnothing(animation_glob_path)
  #   println("Tweeting hourlies $(hourlies_movie_path)...")
  #   run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
  # end
end
