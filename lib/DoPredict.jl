import MemoryConstrainedTreeBoosting

# To predict the past, set FORECAST_DATE=2019-4-7 in the environment:
#
# $ FORECAST_DATE=2019-4-7 make forecast

using Printf

push!(LOAD_PATH, @__DIR__)
import Forecasts
import Grids
import PlotMap

push!(LOAD_PATH, (@__DIR__) * "/../models/sref_mid_2018_forward")
import SREF

push!(LOAD_PATH, (@__DIR__) * "/../models/href_mid_2018_forward")
import HREF

push!(LOAD_PATH, (@__DIR__) * "/../models/rap_march_2014_forward")
import RAP

push!(LOAD_PATH, (@__DIR__) * "/../models/hrrr_mid_july_2016_forward")
import HRRR


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

all_sref_forecasts = SREF.forecasts()

sref_run_time_seconds =
  if haskey(ENV, "FORECAST_DATE")
    # Use 3z SREF and 6z HREF, both should be out before 12z
    year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
    Forecasts.run_time_in_seconds_since_epoch_utc(year, month, day, 3)
  else
    maximum(map(Forecasts.run_time_in_seconds_since_epoch_utc, all_sref_forecasts))
  end

sref_forecasts_to_plot = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) == sref_run_time_seconds, all_sref_forecasts)

# out_dir = (@__DIR__) * "/../forecasts/$(Forecasts.yyyymmdd(forecasts_to_plot[1]))/"
# mkpath(out_dir)

sref_bin_splits, sref_trees = MemoryConstrainedTreeBoosting.load(sref_model_path)

# @sync begin
#   for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(sref_forecasts_to_plot)
#     path = out_dir * "sref_" * Forecasts.yyyymmdd_thhz_fhh(forecast) * "_probabilities"
#
#     println(path)
#
#     data = SREF.get_feature_engineered_data(forecast, data)
#
#     sref_predictions = MemoryConstrainedTreeBoosting.predict(data, sref_bin_splits, sref_trees)
#
#     PlotMap.plot_map(path, Forecasts.grid(forecast), sref_predictions)
#   end
# end



rap_model_path  = (@__DIR__) * "/../models/rap_march_2014_forward/gbdt_f12_2019-04-17T19.27.16.893/568_trees_loss_0.0012037802.model"
hrrr_model_path = (@__DIR__) * "/../models/hrrr_mid_july_2016_forward/gbdt_f12_2019-05-04T13.05.05.929/157_trees_loss_0.0011697214.model"

all_rap_forecasts  = RAP.forecasts()
all_hrrr_forecasts = HRRR.forecasts()

rap_forecast_candidates  = filter(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) >= sref_run_time_seconds, all_rap_forecasts)
hrrr_forecast_candidates = filter(forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) >= sref_run_time_seconds, all_hrrr_forecasts)

if haskey(ENV, "FORECAST_DATE")
  # Use 10z RAP/HRRR at latest, should be out before 12z
  year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
  run_time_seconds_10z = Forecasts.run_time_in_seconds_since_epoch_utc(year, month, day, 10)

  rap_forecast_candidates  = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) <= run_time_seconds_10z, rap_forecast_candidates)
  hrrr_forecast_candidates = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) <= run_time_seconds_10z, hrrr_forecast_candidates)
end

rap_bin_splits,  rap_trees  = MemoryConstrainedTreeBoosting.load(rap_model_path)
hrrr_bin_splits, hrrr_trees = MemoryConstrainedTreeBoosting.load(hrrr_model_path)




href_model_path = (@__DIR__) * "/../models/href_mid_2018_forward/gbdt_f1-36_2019-03-28T13.34.42.186/99_trees_annealing_round_1_loss_0.0012652115.model"

all_href_forecasts = HREF.forecasts()

href_run_time_seconds =
  if haskey(ENV, "FORECAST_DATE")
    # Use 9z SREF and 6z HREF, both should be out before 12z
    year, month, day = map(num_str -> parse(Int64, num_str), split(ENV["FORECAST_DATE"], "-"))
    Forecasts.run_time_in_seconds_since_epoch_utc(year, month, day, 6)
  else
    maximum(map(Forecasts.run_time_in_seconds_since_epoch_utc, all_href_forecasts))
  end

href_forecasts_to_plot = filter(forecast -> Forecasts.run_time_in_seconds_since_epoch_utc(forecast) == href_run_time_seconds, all_href_forecasts)

href_bin_splits, href_trees = MemoryConstrainedTreeBoosting.load(href_model_path)

if href_run_time_seconds > sref_run_time_seconds
  out_dir = (@__DIR__) * "/../forecasts/$(Forecasts.yyyymmdd(href_forecasts_to_plot[1]))/"
else
  out_dir = (@__DIR__) * "/../forecasts/$(Forecasts.yyyymmdd(sref_forecasts_to_plot[1]))/"
end
mkpath(out_dir)

paths = []

period_inverse_prediction              = nothing
period_convective_days_since_epoch_utc = nothing
period_start_str                       = nothing
period_stop_str                        = nothing
href_run_time_str                      = nothing
sref_run_time_str                      = nothing
period_rap_strs                        = String[]

sref_to_href_layer_upsampler = Grids.get_interpolating_upsampler(SREF.grid(), HREF.grid())
rap_to_href_layer_upsampler  = Grids.get_upsampler(RAP.grid(), HREF.grid())
hrrr_to_href_layer_resampler = Grids.get_upsampler(HRRR.grid(), HREF.grid())

for (href_forecast, href_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(href_forecasts_to_plot)

  valid_time_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast)

  perhaps_sref_forecast = filter(sref_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) == valid_time_seconds, sref_forecasts_to_plot)

  for (sref_forecast, sref_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(perhaps_sref_forecast)
    sref_weight = 1.0 - HREF_WEIGHT

    # Take 3 time-lagged RAPs/HRRRs
    rap_forecasts  = collect(Iterators.take(reverse(sort(filter(rap_forecast  -> Forecasts.valid_time_in_seconds_since_epoch_utc(rap_forecast)  == valid_time_seconds, rap_forecast_candidates),  by=Forecasts.run_time_in_seconds_since_epoch_utc)), 3))
    hrrr_forecasts = collect(Iterators.take(reverse(sort(filter(hrrr_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(hrrr_forecast) == valid_time_seconds, hrrr_forecast_candidates), by=Forecasts.run_time_in_seconds_since_epoch_utc)), 3))

    if length(rap_forecasts) != 3 # Don't use RAP if less than 3 available.
      rap_forecasts = []
    end
    if length(hrrr_forecasts) != 3 # Don't use HRRR if less than 3 available.
      hrrr_forecasts = []
    end

    rap_strs = map(rap_forecast -> (@sprintf "t%02dz" rap_forecast.run_hour), rap_forecasts)
    raps_str =
      if isempty(rap_strs)
        ""
      elseif length(rap_strs) == 1
        "_rap_" * rap_strs[1] * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
      else
        "_rap_" * first(rap_strs) * "-" * last(rap_strs) * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
      end

    hrrr_str =
      if isempty(hrrr_forecasts)
        ""
      else
        "_hrrr_w$(HRRR_VS_OTHERS_WEIGHT)"
      end

    path = out_dir * "href_" * Forecasts.yyyymmdd_thhz_fhh(href_forecast) * "_w$(HREF_WEIGHT)_sref_t$(sref_forecast.run_hour)z" * raps_str * hrrr_str
    println(path)

    sref_data = SREF.get_feature_engineered_data(sref_forecast, sref_data)
    href_data = HREF.get_feature_engineered_data(href_forecast, href_data)

    sref_predictions = MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees)
    href_predictions = MemoryConstrainedTreeBoosting.predict(href_data, href_bin_splits, href_trees)

    sref_predictions_upsampled = sref_to_href_layer_upsampler(sref_predictions)

    mean_rap_predictions = nothing
    rap_count            = 0

    for (rap_forecast, rap_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(rap_forecasts)
      rap_data        = RAP.get_feature_engineered_data(rap_forecast, rap_data)
      rap_predictions = MemoryConstrainedTreeBoosting.predict(rap_data, rap_bin_splits, rap_trees)
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

    for (hrrr_forecast, hrrr_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(hrrr_forecasts)
      hrrr_data        = HRRR.get_feature_engineered_data(hrrr_forecast, hrrr_data)
      hrrr_predictions = MemoryConstrainedTreeBoosting.predict(hrrr_data, hrrr_bin_splits, hrrr_trees)
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
    global period_start_str
    global period_stop_str
    global href_run_time_str
    global sref_run_time_str
    global period_rap_strs
    global period_hrrr_str

    if isnothing(period_inverse_prediction) || period_convective_days_since_epoch_utc != Forecasts.valid_time_in_convective_days_since_epoch_utc(href_forecast)
      if !isnothing(period_inverse_prediction)
        period_raps_str =
          if isempty(period_rap_strs)
            ""
          elseif length(period_rap_strs) == 1
            "_rap_" * period_rap_strs[1] * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
          else
            "_rap_" * first(period_rap_strs) * "-" * last(period_rap_strs) * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
          end

        period_path = out_dir * "href_" * href_run_time_str * "_w$(HREF_WEIGHT)_sref_" * sref_run_time_str * "$(period_raps_str)$(period_hrrr_str)_$(period_start_str)_to_$(period_stop_str)"
        period_prediction = 1.0 .- period_inverse_prediction
        PlotMap.plot_map(period_path, Forecasts.grid(href_forecast), period_prediction)
        push!(paths, period_path)
      end
      period_inverse_prediction              = 1.0 .- Float64.(mean_predictions)
      href_run_time_str                      = Forecasts.yyyymmdd_thhz(href_forecast)
      sref_run_time_str                      = "t$(sref_forecast.run_hour)z"
      period_convective_days_since_epoch_utc = Forecasts.valid_time_in_convective_days_since_epoch_utc(href_forecast)
      period_start_str                       = Forecasts.valid_yyyymmdd_hhz(href_forecast)
      period_stop_str                        = Forecasts.valid_hhz(href_forecast)
      period_raps_str                        = Set{String}(Set(rap_strs))
      period_hrrr_str                        = hrrr_str
    else
      period_inverse_prediction .*= (1.0 .- Float64.(mean_predictions))
      period_stop_str             = Forecasts.valid_hhz(href_forecast)
      if !isempty(rap_strs)
        push!(period_rap_strs, rap_strs...)
      end
      if hrrr_str != ""
        period_hrrr_str = hrrr_str
      end
    end

    push!(paths, path)

    PlotMap.plot_map(path, Forecasts.grid(href_forecast), mean_predictions)
  end
end

period_raps_str =
  if isempty(period_rap_strs)
    ""
  elseif length(period_rap_strs) == 1
    "_rap_" * period_rap_strs[1] * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
  else
    "_rap_" * first(period_rap_strs) * "-" * last(period_rap_strs) * "_w$(RAP_VS_HREF_SREF_WEIGHT)"
  end

period_path = out_dir * "href_" * href_run_time_str * "_w$(HREF_WEIGHT)_sref_" * sref_run_time_str * "$(period_raps_str)$(period_hrrr_str)_$(period_start_str)_to_$(period_stop_str)"
period_prediction = 1.0 .- period_inverse_prediction
PlotMap.plot_map(period_path, Forecasts.grid(href_forecasts_to_plot[1]), period_prediction)
push!(paths, period_path)


last_href_valid_time_seconds = maximum(map(Forecasts.valid_time_in_seconds_since_epoch_utc, href_forecasts_to_plot))
longer_range_sref_forecasts = filter(sref_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) > last_href_valid_time_seconds, sref_forecasts_to_plot)

for (sref_forecast, sref_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(longer_range_sref_forecasts)
  path = out_dir * "sref_" * Forecasts.yyyymmdd_thhz_fhh(sref_forecast) * ""
  println(path)

  sref_data = SREF.get_feature_engineered_data(sref_forecast, sref_data)

  sref_predictions = MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees)

  push!(paths, path)

  PlotMap.plot_map(path, Forecasts.grid(sref_forecast), sref_predictions)
end

# @sync doesn't seem to work; poll until subprocesses are done.
for path in paths
  while !isfile(path * ".pdf") || isfile(path * ".sh")
    sleep(1)
  end
end
