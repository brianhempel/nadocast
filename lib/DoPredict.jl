import MemoryConstrainedTreeBoosting

# To predict the past, set FORECAST_DATE=2019-4-7 in the environment:
#
# $ FORECAST_DATE=2019-4-7 make forecast

push!(LOAD_PATH, @__DIR__)
import Forecasts
import Grids
import PlotMap

push!(LOAD_PATH, (@__DIR__) * "/../models/sref_mid_2018_forward")
import SREF

push!(LOAD_PATH, (@__DIR__) * "/../models/href_mid_2018_forward")
import HREF


HREF_WEIGHT =
  if haskey(ENV, "HREF_WEIGHT")
    parse(Int64, ENV["HREF_WEIGHT"])
  else
    0.5
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

for (href_forecast, href_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(href_forecasts_to_plot)

  valid_time_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(href_forecast)

  perhaps_sref_forecast = filter(sref_forecast -> Forecasts.valid_time_in_seconds_since_epoch_utc(sref_forecast) == valid_time_seconds, sref_forecasts_to_plot)

  for (sref_forecast, sref_data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(perhaps_sref_forecast)
    path = out_dir * "href_" * Forecasts.yyyymmdd_thhz_fhh(href_forecast) * "_sref_" * Forecasts.yyyymmdd_thhz_fhh(sref_forecast) * ""
    println(path)

    sref_data = SREF.get_feature_engineered_data(sref_forecast, sref_data)
    href_data = HREF.get_feature_engineered_data(href_forecast, href_data)

    sref_predictions = MemoryConstrainedTreeBoosting.predict(sref_data, sref_bin_splits, sref_trees)
    href_predictions = MemoryConstrainedTreeBoosting.predict(href_data, href_bin_splits, href_trees)

    sref_predictions_upsampled =
      map(Forecasts.grid(href_forecast).latlons) do latlon
        sref_grid_i = Grids.latlon_to_closest_grid_i(SREF.grid(), latlon)
        sref_predictions[sref_grid_i]
      end

    mean_predictions = (href_predictions .* HREF_WEIGHT) .+ (sref_predictions_upsampled .* (1.0 - HREF_WEIGHT))

    push!(paths, path)

    PlotMap.plot_map(path, Forecasts.grid(href_forecast), mean_predictions)
  end
end

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
  while !isfile(path * ".pdf")
    sleep(1)
  end
end
