import MemoryConstrainedTreeBoosting

# To predict the past, set FORECAST_DATE=2019-4-7 in the environment:
#
# $ FORECAST_DATE=2019-4-7 TWEET=true FORECASTS_ROOT=$(pwd)/test_grib2s make forecast

import Dates
using Printf

push!(LOAD_PATH, @__DIR__)
import Forecasts
import Grids
import PlotMap

# println("Loading SREF module")
push!(LOAD_PATH, (@__DIR__) * "/../models/sref_mid_2018_forward")
import SREF

# println("Loading HREF module")
push!(LOAD_PATH, (@__DIR__) * "/../models/href_mid_2018_forward")
import HREF

# println("Loading RAP module")
push!(LOAD_PATH, (@__DIR__) * "/../models/rap_march_2014_forward")
import RAP

# println("Loading HRRR module")
push!(LOAD_PATH, (@__DIR__) * "/../models/hrrr_mid_july_2016_forward")
import HRRR

# println("BEGIN")
HOUR = 60*60

HREF_WEIGHT =
  if haskey(ENV, "HREF_WEIGHT")
    parse(Float32, ENV["HREF_WEIGHT"])
  else
    0.75
  end

RAP_VS_HREF_SREF_WEIGHT =
  if haskey(ENV, "RAP_VS_HREF_SREF_WEIGHT")
    parse(Float32, ENV["RAP_VS_HREF_SREF_WEIGHT"])
  else
    0.5
  end

HRRR_VS_OTHERS_WEIGHT =
  if haskey(ENV, "HRRR_VS_OTHERS_WEIGHT")
    parse(Float32, ENV["HRRR_VS_OTHERS_WEIGHT"])
  else
    0.4
  end


sref_model_path = (@__DIR__) * "/../models/sref_mid_2018_forward/gbdt_f1-39_2019-03-26T00.59.57.772/78_trees_loss_0.001402743.model"

# print("Gather SREF forecasts...")
all_sref_forecasts = SREF.feature_engineered_forecasts()
# println("done.")

sref_run_time_seconds =
  if haskey(ENV, "FORECAST_DATE")
    # Use 3z SREF and 6z HREF, both should be out before 12z
    year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
    Forecasts.time_in_seconds_since_epoch_utc(year, month, day, 3)
  else
    maximum(map(Forecasts.run_time_in_seconds_since_epoch_utc, all_sref_forecasts))
  end

sref_forecasts_to_plot = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) == sref_run_time_seconds, all_sref_forecasts)

# out_dir = (@__DIR__) * "/../forecasts/$(Forecasts.yyyymmdd(forecasts_to_plot[1]))/"
# mkpath(out_dir)

# print("Load SREF model...")
sref_predict = MemoryConstrainedTreeBoosting.load_unbinned_predictor(sref_model_path)
# println("done.")


# @sync begin
#   for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(sref_forecasts_to_plot)
#     path = out_dir * "sref_" * Forecasts.yyyymmdd_thhz_fhh(forecast) * "_probabilities"
#
#     println(path)
#
#     data = SREF.get_feature_engineered_data(forecast, data)
#
#     sref_predictions = MemoryConstrainedTreeBoosting.predict(data, sref_bin_splits, sref_trees)
#
#     PlotMap.plot_map(path, forecast.grid, sref_predictions)
#   end
# end



rap_model_path  = (@__DIR__) * "/../models/rap_march_2014_forward/gbdt_f12_2019-04-17T19.27.16.893/568_trees_loss_0.0012037802.model"
hrrr_model_path = (@__DIR__) * "/../models/hrrr_mid_july_2016_forward/gbdt_f12_2019-05-04T13.05.05.929/157_trees_loss_0.0011697214.model"

# print("Load RAP forecasts...")
all_rap_forecasts = RAP.feature_engineered_forecasts()
# println("done.")

# print("Load HRRR forecasts...")
all_hrrr_forecasts = HRRR.feature_engineered_forecasts()
# println("done.")

rap_forecast_candidates  = filter(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) >= sref_run_time_seconds, all_rap_forecasts)
hrrr_forecast_candidates = filter(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) >= sref_run_time_seconds, all_hrrr_forecasts)

if haskey(ENV, "FORECAST_DATE")
  # Use 10z RAP/HRRR at latest, should be out before 12z
  year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
  run_time_seconds_10z = Forecasts.time_in_seconds_since_epoch_utc(year, month, day, 10)

  rap_forecast_candidates  = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) <= run_time_seconds_10z, rap_forecast_candidates)
  hrrr_forecast_candidates = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) <= run_time_seconds_10z, hrrr_forecast_candidates)
end

# print("Load RAP model...")
rap_predict = MemoryConstrainedTreeBoosting.load_unbinned_predictor(rap_model_path)
# println("done.")

# print("Load HRRR model...")
hrrr_predict = MemoryConstrainedTreeBoosting.load_unbinned_predictor(hrrr_model_path)
# println("done.")




href_model_path = (@__DIR__) * "/../models/href_mid_2018_forward/gbdt_f1-36_2019-03-28T13.34.42.186/99_trees_annealing_round_1_loss_0.0012652115.model"

# print("Load HREF forecasts...")
all_href_forecasts = HREF.feature_engineered_forecasts()
# println("done.")

href_run_time_seconds =
  if haskey(ENV, "FORECAST_DATE")
    # Use 9z SREF and 6z HREF, both should be out before 12z
    year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
    Forecasts.time_in_seconds_since_epoch_utc(year, month, day, 6)
  else
    maximum(map(Forecasts.run_time_in_seconds_since_epoch_utc, all_href_forecasts))
  end

# print("Load HREF model...")
href_predict = MemoryConstrainedTreeBoosting.load_unbinned_predictor(href_model_path)
# println("done.")

nadocast_run_time_seconds = max(href_run_time_seconds, sref_run_time_seconds, map(Forecasts.run_time_in_seconds_since_epoch_utc, hrrr_forecast_candidates)...)
nadocast_run_time_utc     = Dates.unix2datetime(nadocast_run_time_seconds)
nadocast_run_hour         = Dates.hour(nadocast_run_time_utc)

href_forecasts_to_plot = filter(all_href_forecasts) do forecast
  Forecasts.valid_utc_datetime(forecast) > nadocast_run_time_utc &&
  Forecasts.run_time_in_seconds_since_epoch_utc(forecast) == href_run_time_seconds
end

out_dir = (@__DIR__) * "/../forecasts/$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))/t$(nadocast_run_hour)z/"
mkpath(out_dir)
out_path_prefix = out_dir * "nadocast_conus_tor_$(Dates.format(nadocast_run_time_utc, "yyyymmdd"))_t$((@sprintf "%02d" nadocast_run_hour))z"

paths                        = []
animation_glob_path          = nothing
daily_paths_to_perhaps_tweet = []

period_inverse_prediction              = nothing
period_convective_days_since_epoch_utc = nothing
period_start_forecast_hour             = nothing
period_stop_forecast_hour              = nothing
href_run_time_str                      = nothing
sref_run_time_str                      = nothing
period_hrrr_run_hours                  = Int64[]
period_rap_run_hours                   = Int64[]

# println("Compute upsamplers")
sref_to_href_layer_upsampler = Grids.get_interpolating_upsampler(SREF.grid(), HREF.grid())
rap_to_href_layer_upsampler  = Grids.get_upsampler(RAP.grid(), HREF.grid())
hrrr_to_href_layer_resampler = Grids.get_upsampler(HRRR.grid(), HREF.grid())

# for (run_hour, hrrr_run_hour, rap_run_hour, href_run_hour, sref_run_hour) in FORECAST_SCHEDULE
#   run_time_in_seconds_since_epoch_utc = Forecasts.time_in_seconds_since_epoch_utc(run_year, run_month, run_day, run_hour)
#
#   hrrr_delay_hours = hrrr_run_hour > run_hour ? (run_hour + 24) - hrrr_run_hour : run_hour - hrrr_run_hour # HRRR run hour == nadocast run hour so this line is superfulous.
#   rap_delay_hours  = rap_run_hour  > run_hour ? (run_hour + 24) - rap_run_hour  : run_hour - rap_run_hour
#   href_delay_hours = href_run_hour > run_hour ? (run_hour + 24) - href_run_hour : run_hour - href_run_hour
#   sref_delay_hours = sref_run_hour > run_hour ? (run_hour + 24) - sref_run_hour : run_hour - sref_run_hour
#
#   hrrr_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - hrrr_delay_hours*HOUR
#   rap_run_time_in_seconds_since_epoch_utc  = run_time_in_seconds_since_epoch_utc - rap_delay_hours*HOUR
#   href_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - href_delay_hours*HOUR
#   sref_run_time_in_seconds_since_epoch_utc = run_time_in_seconds_since_epoch_utc - sref_delay_hours*HOUR

# println("Start forecasts")
for (href_forecast, href_data) in Forecasts.iterate_data_of_uncorrupted_forecasts(href_forecasts_to_plot)

  valid_time_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast)

  perhaps_sref_forecast = filter(sref_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) == valid_time_seconds, sref_forecasts_to_plot)

  for (sref_forecast, sref_data) in Forecasts.iterate_data_of_uncorrupted_forecasts(perhaps_sref_forecast)
    sref_weight = 1.0 - HREF_WEIGHT

    nadocast_forecast_hour = fld(valid_time_seconds - nadocast_run_time_seconds, HOUR)

    # Take 3 time-lagged RAPs/HRRRs
    rap_forecasts  = collect(Iterators.take(reverse(sort(filter(rap_forecast  -> Forecasts.valid_time_in_seconds_since_epoch_utc(rap_forecast)  == valid_time_seconds, rap_forecast_candidates),  by=Forecasts.run_time_in_seconds_since_epoch_utc)), 3))
    hrrr_forecasts = collect(Iterators.take(reverse(sort(filter(hrrr_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(hrrr_forecast) == valid_time_seconds, hrrr_forecast_candidates), by=Forecasts.run_time_in_seconds_since_epoch_utc)), 3))

    if length(rap_forecasts) != 3 # Don't use RAP if less than 3 available.
      rap_forecasts = []
    end
    if length(hrrr_forecasts) != 3 # Don't use HRRR if less than 3 available.
      hrrr_forecasts = []
    end

    global animation_glob_path
    animation_glob_path = out_path_prefix * "_f%02d.png"
    path                = out_path_prefix * "_f$((@sprintf "%02d" nadocast_forecast_hour))"
    println(path)

    # print("Predicting SREF...")
    sref_predictions = sref_predict(sref_data)
    # println("done.")
    # print("Predicting HREF...")
    href_predictions = href_predict(href_data)
    # println("done.")

    sref_predictions_upsampled = sref_to_href_layer_upsampler(sref_predictions)

    mean_rap_predictions = nothing
    rap_count            = 0

    for (rap_forecast, rap_data) in Forecasts.iterate_data_of_uncorrupted_forecasts(rap_forecasts)
      # print("Predicting RAP...")
      rap_predictions = rap_predict(rap_data)
      # println("done.")
      if isnothing(mean_rap_predictions)
        mean_rap_predictions = rap_to_href_layer_upsampler(rap_predictions)
      else
        mean_rap_predictions .+= rap_to_href_layer_upsampler(rap_predictions)
      end
      rap_count += 1
    end

    if rap_count > 0
      mean_rap_predictions .*= Float32(1.0 / rap_count)
    end

    mean_hrrr_predictions = nothing
    hrrr_count            = 0

    for (hrrr_forecast, hrrr_data) in Forecasts.iterate_data_of_uncorrupted_forecasts(hrrr_forecasts)
      # print("Predicting with HRRR...")
      hrrr_predictions = hrrr_predict(hrrr_data)
      # println("done.")
      if isnothing(mean_hrrr_predictions)
        mean_hrrr_predictions = hrrr_to_href_layer_resampler(hrrr_predictions)
      else
        mean_hrrr_predictions .+= hrrr_to_href_layer_resampler(hrrr_predictions)
      end
      hrrr_count += 1
    end

    if hrrr_count > 0
      mean_hrrr_predictions .*= Float32(1.0 / hrrr_count)
    end

    mean_predictions = (href_predictions .* HREF_WEIGHT) .+ (sref_predictions_upsampled .* sref_weight)

    if rap_count > 0
      mean_predictions = (mean_rap_predictions .* RAP_VS_HREF_SREF_WEIGHT) .+ (mean_predictions .* (1.0 - RAP_VS_HREF_SREF_WEIGHT))
    end

    if hrrr_count > 0
      mean_predictions = (mean_hrrr_predictions .* HRRR_VS_OTHERS_WEIGHT) .+ (mean_predictions .* (1.0 - HRRR_VS_OTHERS_WEIGHT))
    end

    global period_inverse_prediction
    global period_convective_days_since_epoch_utc
    global period_start_forecast_hour
    global period_stop_forecast_hour
    global href_run_time_str
    global sref_run_time_str
    global period_hrrr_run_hours
    global period_rap_run_hours

    if isnothing(period_inverse_prediction) || period_convective_days_since_epoch_utc != Forecasts.valid_time_in_convective_days_since_epoch_utc(href_forecast)
      if !isnothing(period_inverse_prediction)
        period_path = out_path_prefix * "_f$((@sprintf "%02d" period_start_forecast_hour))-$((@sprintf "%02d" period_stop_forecast_hour))"
        period_prediction = 1.0 .- period_inverse_prediction
        write(period_path * ".float16.bin", Float16.(period_prediction))
        PlotMap.plot_map(
          period_path,
          href_forecast.grid,
          period_prediction;
          run_time_utc = nadocast_run_time_utc,
          forecast_hour_range = period_start_forecast_hour:period_stop_forecast_hour,
          hrrr_run_hours = period_hrrr_run_hours,
          rap_run_hours  = period_rap_run_hours,
          href_run_hours = [href_forecast.run_hour],
          sref_run_hours = [sref_forecast.run_hour]
        )
        push!(paths, period_path)
        if period_stop_forecast_hour - period_start_forecast_hour + 1 >= 14 && Dates.hour(nadocast_run_time_utc + Dates.Hour(period_stop_forecast_hour)) == 11
          push!(daily_paths_to_perhaps_tweet, period_path)
        end
      end
      period_inverse_prediction              = 1.0 .- Float64.(mean_predictions)
      href_run_time_str                      = Forecasts.yyyymmdd_thhz(href_forecast)
      sref_run_time_str                      = "$(sref_forecast.run_hour)z"
      period_convective_days_since_epoch_utc = Forecasts.valid_time_in_convective_days_since_epoch_utc(href_forecast)
      period_start_forecast_hour             = nadocast_forecast_hour
      period_stop_forecast_hour              = nadocast_forecast_hour
      period_hrrr_run_hours                  = Int64[]
      period_rap_run_hours                   = Int64[]
    else
      period_inverse_prediction .*= (1.0 .- Float64.(mean_predictions))
      period_stop_forecast_hour   = nadocast_forecast_hour
    end
    for rap_forecast in rap_forecasts
      if !(rap_forecast.run_hour in period_rap_run_hours)
        push!(period_rap_run_hours, rap_forecast.run_hour)
      end
    end
    for hrrr_forecast in hrrr_forecasts
      if !(hrrr_forecast.run_hour in period_hrrr_run_hours)
        push!(period_hrrr_run_hours, hrrr_forecast.run_hour)
      end
    end

    push!(paths, path)

    write(path * ".float16.bin", Float16.(mean_predictions))

    PlotMap.plot_map(
      path,
      href_forecast.grid,
      mean_predictions;
      run_time_utc = nadocast_run_time_utc,
      forecast_hour_range = nadocast_forecast_hour:nadocast_forecast_hour,
      hrrr_run_hours = map(forecast -> forecast.run_hour, hrrr_forecasts),
      rap_run_hours  = map(forecast -> forecast.run_hour, rap_forecasts),
      href_run_hours = [href_forecast.run_hour],
      sref_run_hours = [sref_forecast.run_hour]
    )
  end
end

if !isnothing(period_inverse_prediction)
  period_path = out_path_prefix * "_f$((@sprintf "%02d" period_start_forecast_hour))-$((@sprintf "%02d" period_stop_forecast_hour))"
  period_prediction = 1.0 .- period_inverse_prediction
  write(period_path * ".float16.bin", Float16.(period_prediction))
  PlotMap.plot_map(
    period_path,
    href_forecasts_to_plot[1].grid,
    period_prediction;
    run_time_utc = nadocast_run_time_utc,
    forecast_hour_range = period_start_forecast_hour:period_stop_forecast_hour,
    hrrr_run_hours = period_hrrr_run_hours,
    rap_run_hours  = period_rap_run_hours,
    href_run_hours = [href_forecasts_to_plot[1].run_hour],
    sref_run_hours = [sref_forecasts_to_plot[1].run_hour]
  )
  push!(paths, period_path)
  if period_stop_forecast_hour - period_start_forecast_hour + 1 >= 14 && Dates.hour(nadocast_run_time_utc + Dates.Hour(period_stop_forecast_hour)) == 11
    push!(daily_paths_to_perhaps_tweet, period_path)
  end
end


last_href_valid_time_seconds = isempty(href_forecasts_to_plot) ? nadocast_run_time_seconds : maximum(map(Forecasts.valid_time_in_seconds_since_epoch_utc, href_forecasts_to_plot))
longer_range_sref_forecasts = filter(sref_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) > last_href_valid_time_seconds, sref_forecasts_to_plot)

for (sref_forecast, sref_data) in Forecasts.iterate_data_of_uncorrupted_forecasts(longer_range_sref_forecasts)
  path = out_dir * "sref_" * Forecasts.yyyymmdd_thhz_fhh(sref_forecast) * ""
  println(path)

  # sref_data = SREF.get_feature_engineered_data(sref_forecast, sref_data)

  # sref_predictions = MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees)
  sref_predictions = sref_predict(sref_data)

  push!(paths, path)

  nadocast_forecast_hour = fld(Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) - nadocast_run_time_seconds, HOUR)

  write(path * ".float16.bin", Float16.(sref_predictions))
  PlotMap.plot_map(
    path,
    sref_forecast.grid,
    sref_predictions;
    run_time_utc = nadocast_run_time_utc,
    forecast_hour_range = nadocast_forecast_hour:nadocast_forecast_hour,
    hrrr_run_hours = [],
    rap_run_hours  = [],
    href_run_hours = [],
    sref_run_hours = [sref_forecast.run_hour]
  )
end

# @sync doesn't seem to work; poll until subprocesses are done.
for path in paths
  while !isfile(path * ".pdf") || isfile(path * ".sh")
    sleep(1)
  end
end

if !isnothing(animation_glob_path)
  println("Making hourlies movie out of $(animation_glob_path)...")
  hourlies_movie_path = out_path_prefix * "_hourlies"
  run(`ffmpeg -framerate 2 -i "$(animation_glob_path)" -c:v libx264 -vf format=yuv420p,scale=1200:-1 $hourlies_movie_path.mp4`)
end

if get(ENV, "TWEET", "false") == "true"
  tweet_script_path = (@__DIR__) * "/tweet.rb"

  for path in daily_paths_to_perhaps_tweet
    println("Tweeting daily $(path)...")
    run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Day Tornado Forecast" $path.png`)
  end

  if !isnothing(animation_glob_path)
    println("Tweeting hourlies $(hourlies_movie_path)...")
    run(`ruby $tweet_script_path "$(nadocast_run_hour)Z Hourly Tornado Forecasts" $hourlies_movie_path.mp4`)
  end
end
