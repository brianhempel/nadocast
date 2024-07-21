module TrainingShared

import Random
import Serialization
import Printf
import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
import Grids
using HREF15KMGrid
using Grid130
import Inventories
import StormEvents
# import PlotMap

push!(LOAD_PATH, @__DIR__)

import Climatology


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

EVENT_TIME_WINDOW_HALF_SIZE = 30*MINUTE
EVENT_SPATIAL_RADIUS_MILES  = 25.0

NEAR_EVENT_TIME_WINDOW_HALF_SIZE  = 90*MINUTE
NEAR_EVENT_RADIUS_MILES           = 100.0

function is_train(forecast :: Forecasts.Forecast) :: Bool
  !is_validation(forecast) && !is_test(forecast)
end

# 0 == Thursday  in convective days (just starts as 12Z of the UTC day).
# 1 == Friday    in convective days (just starts as 12Z of the UTC day).
# 2 == Saturday  in convective days (just starts as 12Z of the UTC day).
# 3 == Sunday    in convective days (just starts as 12Z of the UTC day).
# 4 == Monday    in convective days (just starts as 12Z of the UTC day).
# 5 == Tuesday   in convective days (just starts as 12Z of the UTC day).
# 6 == Wednesday in convective days (just starts as 12Z of the UTC day).

# Saturdays
function is_validation(forecast :: Forecasts.Forecast) :: Bool
  mod(Forecasts.valid_time_in_convective_days_since_epoch_utc(forecast), 7) == 2
end

# Sundays
function is_test(forecast :: Forecasts.Forecast) :: Bool
  mod(Forecasts.valid_time_in_convective_days_since_epoch_utc(forecast), 7) == 3
end

_conus_event_hours_in_seconds_from_epoch_utc = nothing
function conus_event_hours_in_seconds_from_epoch_utc()
  global _conus_event_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_event_hours_in_seconds_from_epoch_utc)
    _conus_event_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_event_hours_in_seconds_from_epoch_utc
end

_conus_tornado_hours_in_seconds_from_epoch_utc = nothing
function conus_tornado_hours_in_seconds_from_epoch_utc()
  global _conus_tornado_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_tornado_hours_in_seconds_from_epoch_utc)
    _conus_tornado_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_tornado_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_tornado_hours_in_seconds_from_epoch_utc
end

_conus_severe_wind_hours_in_seconds_from_epoch_utc = nothing
function conus_severe_wind_hours_in_seconds_from_epoch_utc()
  global _conus_severe_wind_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_severe_wind_hours_in_seconds_from_epoch_utc)
    _conus_severe_wind_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_severe_wind_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_severe_wind_hours_in_seconds_from_epoch_utc
end

_conus_severe_hail_hours_in_seconds_from_epoch_utc = nothing
function conus_severe_hail_hours_in_seconds_from_epoch_utc()
  global _conus_severe_hail_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_severe_hail_hours_in_seconds_from_epoch_utc)
    _conus_severe_hail_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_severe_hail_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_severe_hail_hours_in_seconds_from_epoch_utc
end

_conus_sig_tornado_hours_in_seconds_from_epoch_utc = nothing
function conus_sig_tornado_hours_in_seconds_from_epoch_utc()
  global _conus_sig_tornado_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_sig_tornado_hours_in_seconds_from_epoch_utc)
    _conus_sig_tornado_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_sig_tornado_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_sig_tornado_hours_in_seconds_from_epoch_utc
end

_conus_sig_wind_hours_in_seconds_from_epoch_utc = nothing
function conus_sig_wind_hours_in_seconds_from_epoch_utc()
  global _conus_sig_wind_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_sig_wind_hours_in_seconds_from_epoch_utc)
    _conus_sig_wind_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_sig_wind_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_sig_wind_hours_in_seconds_from_epoch_utc
end

_conus_sig_hail_hours_in_seconds_from_epoch_utc = nothing
function conus_sig_hail_hours_in_seconds_from_epoch_utc()
  global _conus_sig_hail_hours_in_seconds_from_epoch_utc
  if isnothing(_conus_sig_hail_hours_in_seconds_from_epoch_utc)
    _conus_sig_hail_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.conus_sig_hail_events(), EVENT_TIME_WINDOW_HALF_SIZE)
  end
  _conus_sig_hail_hours_in_seconds_from_epoch_utc
end

# Use all forecasts in which there is a tornado, wind, or hail event in that hour or ±2hours.
function is_relevant_forecast(forecast)
  valid_time = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)

  ((valid_time + 2*HOUR) in conus_event_hours_in_seconds_from_epoch_utc()) ||
  ((valid_time +   HOUR) in conus_event_hours_in_seconds_from_epoch_utc()) ||
  (valid_time            in conus_event_hours_in_seconds_from_epoch_utc()) ||
  ((valid_time -   HOUR) in conus_event_hours_in_seconds_from_epoch_utc()) ||
  ((valid_time - 2*HOUR) in conus_event_hours_in_seconds_from_epoch_utc())
end

function forecast_is_tornado_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_tornado_hours_in_seconds_from_epoch_utc()
end

function forecast_is_severe_wind_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_severe_wind_hours_in_seconds_from_epoch_utc()
end

function forecast_is_severe_hail_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_severe_hail_hours_in_seconds_from_epoch_utc()
end

function forecast_is_sig_tornado_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_sig_tornado_hours_in_seconds_from_epoch_utc()
end

function forecast_is_sig_wind_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_sig_wind_hours_in_seconds_from_epoch_utc()
end

function forecast_is_sig_hail_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_sig_hail_hours_in_seconds_from_epoch_utc()
end


# returns (train_forecasts, validation_forecasts, test_forecasts)
function forecasts_train_validation_test(all_forecasts; forecast_hour_range = 1:10000, just_hours_near_storm_events = true)
  # This filtering here is probably pretty slow.
  forecasts =
    if just_hours_near_storm_events
      filter(all_forecasts) do forecast
        (forecast.forecast_hour in forecast_hour_range) && is_relevant_forecast(forecast)
      end
    else
      filter(all_forecasts) do forecast
        forecast.forecast_hour in forecast_hour_range
      end
    end

  # Look for last storm event. Ignore any forecasts after that convective day.
  last_convective_day = maximum(StormEvents.start_time_in_convective_days_since_epoch_utc, StormEvents.conus_events())
  cutoff = Dates.unix2datetime(StormEvents.convective_days_since_epoch_to_seconds_utc(last_convective_day)) + Dates.Hour(24)
  filter!(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, forecasts);

  train_forecasts      = filter(is_train, forecasts)
  validation_forecasts = filter(is_validation, forecasts)
  test_forecasts       = filter(is_test, forecasts)

  (train_forecasts, validation_forecasts, test_forecasts)
end

function grid_to_labels(events, forecast)
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, EVENT_SPATIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
end

function grid_to_tor_life_risk_labels(tor_events, forecast)
  StormEvents.grid_to_tor_life_risk_neighborhoods(tor_events, forecast.grid, EVENT_SPATIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
end

_hour_estimated_wind_gridded_normalization = nothing
function hour_estimated_wind_gridded_normalization()
  global _hour_estimated_wind_gridded_normalization
  if isnothing(_hour_estimated_wind_gridded_normalization)
    _hour_estimated_wind_gridded_normalization = Climatology.read_float16_file(joinpath(Climatology.asos_climatology_data_dir, "hour_x1_normalization_grid_130_cropped.float16.bin"))
  end
  _hour_estimated_wind_gridded_normalization

end

_hour_estimated_sig_wind_gridded_normalization = nothing
function hour_estimated_sig_wind_gridded_normalization()
  global _hour_estimated_sig_wind_gridded_normalization
  if isnothing(_hour_estimated_sig_wind_gridded_normalization)
    _hour_estimated_sig_wind_gridded_normalization = Climatology.read_float16_file(joinpath(Climatology.asos_climatology_data_dir, "sig_hour_x1_normalization_grid_130_cropped.float16.bin"))
  end
  _hour_estimated_sig_wind_gridded_normalization
end

_day_estimated_wind_gridded_normalization = nothing
function day_estimated_wind_gridded_normalization()
  global _day_estimated_wind_gridded_normalization
  if isnothing(_day_estimated_wind_gridded_normalization)
    _day_estimated_wind_gridded_normalization = Climatology.read_float16_file(joinpath(Climatology.asos_climatology_data_dir, "day_x1_normalization_grid_130_cropped.float16.bin"))
  end
  _day_estimated_wind_gridded_normalization

end

_day_estimated_sig_wind_gridded_normalization = nothing
function day_estimated_sig_wind_gridded_normalization()
  global _day_estimated_sig_wind_gridded_normalization
  if isnothing(_day_estimated_sig_wind_gridded_normalization)
    _day_estimated_sig_wind_gridded_normalization = Climatology.read_float16_file(joinpath(Climatology.asos_climatology_data_dir, "sig_day_x1_normalization_grid_130_cropped.float16.bin"))
  end
  _day_estimated_sig_wind_gridded_normalization
end

function grid_to_adjusted_wind_labels(measured_events, estimated_events, gridded_normalization, forecast)
  measured_labels  = grid_to_labels(measured_events, forecast) # vals are 0 or 1
  estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, forecast.grid, GRID_130_CROPPED, gridded_normalization, EVENT_SPATIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
  max.(measured_labels, estimated_labels)
end

function compute_is_near_storm_event(forecast) :: Array{Float32,1}
  StormEvents.grid_to_event_neighborhoods(StormEvents.conus_events(), forecast.grid, NEAR_EVENT_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), NEAR_EVENT_TIME_WINDOW_HALF_SIZE)
end

# Dict of name to (forecast_has_event, forecast_to_gridpoint_labels)
event_name_to_forecast_predicate = Dict(
  "tornado"           => forecast_is_tornado_hour,
  "wind"              => forecast_is_severe_wind_hour,
  "wind_adj"          => forecast_is_severe_wind_hour,
  "hail"              => forecast_is_severe_hail_hour,
  "sig_tornado"       => forecast_is_sig_tornado_hour,
  "sig_wind"          => forecast_is_sig_wind_hour,
  "sig_wind_adj"      => forecast_is_sig_wind_hour,
  "sig_hail"          => forecast_is_sig_hail_hour,
  "tornado_life_risk" => forecast_is_tornado_hour,
)

event_name_to_labeler = Dict(
  "tornado"           => (forecast -> grid_to_labels(StormEvents.conus_tornado_events(),                                                                                                                           forecast)),
  "wind"              => (forecast -> grid_to_labels(StormEvents.conus_severe_wind_events(),                                                                                                                       forecast)),
  "wind_adj"          => (forecast -> grid_to_adjusted_wind_labels(StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), hour_estimated_wind_gridded_normalization(), forecast)),
  "hail"              => (forecast -> grid_to_labels(StormEvents.conus_severe_hail_events(),                                                                                                                       forecast)),
  "sig_tornado"       => (forecast -> grid_to_labels(StormEvents.conus_sig_tornado_events(),                                                                                                                       forecast)),
  "sig_wind"          => (forecast -> grid_to_labels(StormEvents.conus_sig_wind_events(),                                                                                                                          forecast)),
  "sig_wind_adj"      => (forecast -> grid_to_adjusted_wind_labels(StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), hour_estimated_sig_wind_gridded_normalization(),   forecast)),
  "sig_hail"          => (forecast -> grid_to_labels(StormEvents.conus_sig_hail_events(),                                                                                                                          forecast)),
  "tornado_life_risk" => (forecast -> grid_to_tor_life_risk_labels(StormEvents.conus_tornado_events(),                                                                                                             forecast)),
)


function grid_to_day_labels(events, forecast, f1_or_f2_is_soonest)
  # Annoying that we have to recalculate this.
  # The end_seconds will always be the last hour of the convective day
  # start_seconds depends on whether the run started during the day or not
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + f1_or_f2_is_soonest*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

function grid_to_tor_life_risk_day_labels(tor_events, forecast, f1_or_f2_is_soonest)
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + f1_or_f2_is_soonest*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_tor_life_risk_neighborhoods(tor_events, forecast.grid, EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
end

function grid_to_adjusted_wind_day_labels(measured_events, estimated_events, gridded_normalization, forecast, f1_or_f2_is_soonest)
  # Annoying that we have to recalculate this.
  # The end_seconds will always be the last hour of the convective day
  # start_seconds depends on whether the run started during the day or not
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + f1_or_f2_is_soonest*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  measured_labels  = StormEvents.grid_to_event_neighborhoods(measured_events, forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
  estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, forecast.grid, Grid130.GRID_130_CROPPED, gridded_normalization, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
  max.(measured_labels, estimated_labels)
end

function compute_is_near_day_storm_event(forecast, f1_or_f2_is_soonest) :: Array{Float32,1}
  start_seconds    = max(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) - 23*HOUR, Forecasts.run_time_in_seconds_since_epoch_utc(forecast) + f1_or_f2_is_soonest*HOUR) - 30*MINUTE
  end_seconds      = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) + 30*MINUTE
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  StormEvents.grid_to_event_neighborhoods(StormEvents.conus_events(), forecast.grid, NEAR_EVENT_RADIUS_MILES, window_mid_time, window_half_size)
end

event_name_to_day_labeler(f1_or_f2_is_soonest) = Dict(
  "tornado"           => (forecast -> grid_to_day_labels(StormEvents.conus_tornado_events(),     forecast, f1_or_f2_is_soonest)),
  "wind"              => (forecast -> grid_to_day_labels(StormEvents.conus_severe_wind_events(), forecast, f1_or_f2_is_soonest)),
  "wind_adj"          => (forecast -> grid_to_adjusted_wind_day_labels(StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), day_estimated_wind_gridded_normalization(), forecast, f1_or_f2_is_soonest)),
  "hail"              => (forecast -> grid_to_day_labels(StormEvents.conus_severe_hail_events(), forecast, f1_or_f2_is_soonest)),
  "sig_tornado"       => (forecast -> grid_to_day_labels(StormEvents.conus_sig_tornado_events(), forecast, f1_or_f2_is_soonest)),
  "sig_wind"          => (forecast -> grid_to_day_labels(StormEvents.conus_sig_wind_events(),    forecast, f1_or_f2_is_soonest)),
  "sig_wind_adj"      => (forecast -> grid_to_adjusted_wind_day_labels(StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), day_estimated_sig_wind_gridded_normalization(), forecast, f1_or_f2_is_soonest)),
  "sig_hail"          => (forecast -> grid_to_day_labels(StormEvents.conus_sig_hail_events(),    forecast, f1_or_f2_is_soonest)),
  "tornado_life_risk" => (forecast -> grid_to_tor_life_risk_day_labels(StormEvents.conus_tornado_events(), forecast, f1_or_f2_is_soonest)),
)


# Concatenating all the forecasts doubles peak memory usage if done in-RAM.
#
# We load each file to disk, after which we know the appropriate size needed
# and load back into RAM.
#
# If prior_predictor is provided, computes prior predictor's loss during loading.
# prior_predictor is fed the untransformed data.
#
# If save_dir already exists and loading appears to have happened, reads the data in from disk.
function finished_loading(save_dir)
  isdir(save_dir) && isfile(joinpath(save_dir, "weights.serialized"))
end

# labels are return as a dictionary of event_name_to_labels
function get_data_labels_weights(forecasts; save_dir = nothing, X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, event_name_to_labeler, only_features = nothing, only_before = Dates.DateTime(2099,1,1,12))
  if isnothing(save_dir)
    save_dir = "data_labels_weights_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  end
  if !finished_loading(save_dir)
    load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = X_transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, prior_predictor = prior_predictor, event_name_to_labeler = event_name_to_labeler)
  end
  read_data_labels_weights_from_disk(save_dir; only_features = only_features, only_before = only_before)
end

# Loads the data to disk but does not read it back.
# Returns (data_count, feature_count) if the data wasn't already saved to disk.
function prepare_data_labels_weights(forecasts; save_dir = nothing, X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, event_name_to_labeler)
  if isnothing(save_dir)
    save_dir = "data_labels_weights_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  end
  if finished_loading(save_dir)
    println("$save_dir appears to have finished loading. Skipping.")
    return (nothing, nothing)
  end
  load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = X_transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, prior_predictor = prior_predictor, event_name_to_labeler = event_name_to_labeler)
end


function mask_rows_threaded(data, mask; final_row_count=count(mask))
  out = Array{Float32}(undef, (final_row_count, size(data, 2)))

  Threads.@threads :static for col_i in 1:size(data, 2)
    out[:, col_i] = @view data[mask, col_i]
  end

  out
end

# Points within 25mi of 1+ tornado and 1+ severe wind and 1+ severe hail report in the climatological background
const verifiability_mask_path = joinpath((@__DIR__), "..", "..", "climatological_background_1998-2013", "verifiable_area_mask.bits")
const verifiability_mask_on_href_cropped_15km_grid = read!(verifiability_mask_path, BitVector(undef, length(HREF_CROPPED_15KM_GRID.latlons)))

function is_verifiable(latlon :: Tuple{Float64, Float64})
  flat_i = Grids.latlon_to_closest_grid_i(HREF_CROPPED_15KM_GRID, latlon)
  verifiability_mask_on_href_cropped_15km_grid[flat_i]
end

# We can keep weights and labels in memory at least. It's the features that really kill us.
# prior_predictor, if provided, has its logloss calculated against the first labeler
function load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, event_name_to_labeler)
  mkpath(save_dir)

  save_path(path) = joinpath(save_dir, path)
  serialize_async(path, value) = Threads.@spawn Serialization.serialize(path, value)

  event_names    = collect(keys(event_name_to_labeler))
  label_arrays   = map(_ -> Float32[], event_names)
  weights        = Float32[]
  lats           = Float32[]
  lons           = Float32[]
  run_times      = Dates.DateTime[]
  forecast_hours = UInt8[]

  prior_losses         = []
  prior_losses_weights = []
  prior_loss_forecasts = []
  ε             = eps(1.0f0)
  logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε) # Copied from Flux.jl

  grid = first(forecasts).grid

  verifiable_grid_bitmask = Conus.is_in_conus.(grid.latlons) .&& is_verifiable.(grid.latlons) :: BitVector
  verifiable_grid_weights = Float32.(grid.point_weights[verifiable_grid_bitmask])
  verifiable_lats         = Float32.(first.(grid.latlons)[verifiable_grid_bitmask])
  verifiable_lons         = Float32.(last.(grid.latlons)[verifiable_grid_bitmask])

  verifiable_point_count = count(verifiable_grid_bitmask)

  # Deterministic randomness for calc_inclusion_probabilities, presuming forecasts are given in the same order.
  rng = Random.MersenneTwister(12345)

  forecast_i = 1

  start_time = time_ns()

  serialization_task = nothing
  feature_count = 0

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(forecasts)
    forecast_label_layers_full_grid = map(labeler -> labeler(forecast), values(event_name_to_labeler))         :: Vector{Vector{Float32}}
    forecast_label_layers           = map(layer -> layer[verifiable_grid_bitmask], forecast_label_layers_full_grid) :: Vector{Vector{Float32}}

    # PlotMap.plot_debug_map("tornadoes_$(Forecasts.valid_yyyymmdd_hhz(forecast))", grid, compute_forecast_labels(forecast))
    # PlotMap.plot_debug_map("near_events_$(Forecasts.valid_yyyymmdd_hhz(forecast))", grid, compute_is_near_storm_event(forecast))

    if !isnothing(calc_inclusion_probabilities)
      probabilities       = Float32.(calc_inclusion_probabilities(forecast, forecast_label_layers_full_grid)[verifiable_grid_bitmask])
      probabilities       = clamp.(probabilities, 0f0, 1f0)
      mask                = map(p -> p > 0f0 && rand(rng, Float32) <= p, probabilities)
      probabilities       = probabilities[mask]

      forecast_weights = verifiable_grid_weights[mask] ./ probabilities
      forecast_lats    = @view verifiable_lats[mask]
      forecast_lons    = @view verifiable_lons[mask]

      forecast_label_layers = map(layer -> layer[mask], forecast_label_layers) :: Vector{Vector{Float32}}
      # Combine verifiable mask and mask
      data_mask = verifiable_grid_bitmask[:]
      data_mask[verifiable_grid_bitmask] = mask
      data_masked = mask_rows_threaded(data, data_mask)

      # print("$(count(mask) / length(mask))")
    else
      forecast_weights = verifiable_grid_weights
      forecast_lats    = verifiable_lats
      forecast_lons    = verifiable_lons
      data_masked      = mask_rows_threaded(data, verifiable_grid_bitmask; final_row_count=verifiable_point_count)
    end
    X_transformed = X_transformer(data_masked)

    if forecast_i == 1
      inventory_lines      = Forecasts.inventory(forecast)
      feature_descriptions = Inventories.inventory_line_description.(inventory_lines)
      feature_count        = size(X_transformed, 2)
      @assert feature_count == length(inventory_lines)
      write(save_path("features.txt"), join(feature_descriptions, "\n"))
    end

    if !isnothing(prior_predictor)
      predictions = prior_predictor(data_masked)
      push!(prior_losses, sum(logloss.(forecast_label_layers[1], predictions) .* forecast_weights))
      push!(prior_losses_weights, sum(forecast_weights))
      push!(prior_loss_forecasts, forecast)
    end

    data_file_name = Printf.@sprintf "data_%06d_%dx%d.serialized" forecast_i size(X_transformed,1) size(X_transformed,2)

    !isnothing(serialization_task) && wait(serialization_task) # Make sure we don't get ahead of disk writes.
    serialization_task = serialize_async(save_path(data_file_name), X_transformed)

    for label_layer_i in 1:length(forecast_label_layers)
      append!(label_arrays[label_layer_i], forecast_label_layers[label_layer_i])
    end
    append!(weights, forecast_weights)
    append!(lats, forecast_lats)
    append!(lons, forecast_lons)
    append!(run_times, fill(Forecasts.run_utc_datetime(forecast), length(forecast_weights)))
    append!(forecast_hours, fill(UInt8(forecast.forecast_hour), length(forecast_weights)))

    elapsed = (Base.time_ns() - start_time) / 1.0e9
    print("\r$forecast_i/~$(length(forecasts)) forecasts loaded.  $(Float32(elapsed / forecast_i))s each.  ~$(Float32((elapsed / forecast_i) * (length(forecasts) - forecast_i) / 60 / 60)) hours left.            ")

    forecast_i += 1
  end

  wait(serialization_task) # Synchronize

  for i in 1:length(event_names)
    @assert length(label_arrays[i]) == length(weights)
    Serialization.serialize(save_path("labels-$(event_names[i]).serialized"), label_arrays[i])
  end
  @assert length(lats)           == length(weights)
  @assert length(lons)           == length(weights)
  @assert length(run_times)      == length(weights)
  @assert length(forecast_hours) == length(weights)
  Serialization.serialize(save_path("weights.serialized"), weights)
  # In case we need them later...
  Serialization.serialize(save_path("lats.serialized"), lats)
  Serialization.serialize(save_path("lons.serialized"), lons)
  Serialization.serialize(save_path("run_times.serialized"), run_times)
  Serialization.serialize(save_path("forecast_hours.serialized"), forecast_hours)

  if !isnothing(prior_predictor)
    prior_loss = sum(prior_losses) / sum(prior_losses_weights)
    print("loss via prior predictor: $prior_loss ")
    worst_first_permutation = sortperm(prior_losses, rev=true)
    worst_losses    = prior_losses[worst_first_permutation][1:10]
    worst_forecasts = prior_loss_forecasts[worst_first_permutation][1:10]

    for (loss, forecast) in zip(worst_losses, worst_forecasts)
      print(Forecasts.time_title(forecast) * ": $loss ")
    end
  end

  println()

  (length(weights), feature_count)
end

function chunk_range(chunk_i, n_chunks, array_len)
  start = div((chunk_i-1) * array_len, n_chunks) + 1
  stop  = div( chunk_i    * array_len, n_chunks)
  start:stop
end

# If using MPI for data-parallel distributed learning, chunk_i = rank+1, chunk_count = rank_count
function read_data_labels_weights_from_disk(save_dir; chunk_i = 1, chunk_count = 1, only_features = nothing, only_before = Dates.DateTime(2099,1,1,12))
  save_path(path)   = joinpath(save_dir, path)
  deserialize(path) = Serialization.deserialize(save_path(path))

  valid_times = deserialize("run_times.serialized") .+ Dates.Hour.(deserialize("forecast_hours.serialized"))

  is_to_use = findall(t -> t < only_before, valid_times)

  my_is = is_to_use[chunk_range(chunk_i, chunk_count, length(is_to_use))]

  # free
  is_to_use    = nothing
  valid_times  = nothing

  data_count = length(my_is)

  data_file_names = sort(filter(file_name -> startswith(file_name, "data_"), readdir(save_dir)), by=(name -> parse(Int64, split(name, r"_|\.|-")[2])))

  forecast_data_1 = deserialize(data_file_names[1])

  raw_feature_count = size(forecast_data_1, 2)

  feature_names = readlines(save_path("features.txt"))
  @assert isnothing(only_features) || all(feat_name -> feat_name in feature_names, only_features) || all(feat_i -> feat_i in 1:raw_feature_count, only_features)
  only_feature_is =
    if isnothing(only_features)
      1:raw_feature_count
    elseif only_features[1] in feature_names
      map(feat_name -> findfirst(isequal(feat_name), feature_names), only_features)
    else
      only_features :: Vector{Int64}
    end

  feature_count = length(only_feature_is)
  feature_type  = eltype(forecast_data_1)

  forecast_data_1 = nothing # free

  data = Array{feature_type}(undef, (data_count, feature_count))

  full_i = 1
  rows_filled  = 0

  for data_file_name in data_file_names
    forecast_row_count, forecast_feature_count = parse.(Int64, match(r"data_\d+_(\d+)x(\d+).serialized", data_file_name))

    @assert raw_feature_count == forecast_feature_count

    file_full_range = full_i:(full_i + forecast_row_count - 1)

    my_part = my_is[searchsortedfirst(my_is, file_full_range.start):searchsortedlast(my_is, file_full_range.stop)]

    if length(my_part) > 0
      forecast_data = deserialize(data_file_name)
      @assert forecast_row_count == size(forecast_data, 1)
      @assert raw_feature_count  == size(forecast_data, 2)
      data[rows_filled .+ (1:length(my_part)), :] = forecast_data[my_part .- (full_i - 1), only_feature_is]
      rows_filled += length(my_part)
    end

    full_i += forecast_row_count
  end

  @assert rows_filled == data_count

  all_weights = deserialize("weights.serialized")

  if chunk_i == chunk_count
    @assert full_i - 1 == length(all_weights)
  end

  weights = all_weights[my_is]

  label_file_names = sort(filter(file_name -> startswith(file_name, "labels-"), readdir(save_dir)))

  Ys = Dict(
    map(label_file_names) do label_file_name
      event_name = split(label_file_name, r"-|\.")[2]
      Y = deserialize(label_file_name)[my_is]
      event_name => Y
    end
  )

  (data, Ys, weights)
end


# Provide a function that returns a loss and kwargs that contain arrays of values to try.
# *** Assumes each array of values is in order ***
function coordinate_descent_hyperparameter_search(f; print = print, random_start_count = 0, max_hyperparameter_coordinate_descent_iterations = 4, kwargs...)
  combos_tried = []

  last_iteration_best_loss = Inf32
  best_loss                = Inf32

  rng = Random.MersenneTwister(100) # deterministic randomness, in case of MPI

  hyperparameters_to_combo(pick_value) =
    Dict(
      map(collect(kwargs)) do (arg_name, values)
        arg_name => pick_value(values)
      end
    )

  initial_combo =
    hyperparameters_to_combo() do values
      values[1 + div(length(values)-1,2)] # middle_value
    end

  random_combos =
    map(1:random_start_count) do _
      hyperparameters_to_combo(options -> rand(rng, options))
    end

  best_combo = Dict(initial_combo)

  # Returns (is_best, loss)
  try_combo(combo) = begin
    loss = f(; combo...)
    push!(combos_tried, combo)

    if loss < best_loss
      best_combo = combo
      best_loss  = loss
      print("New best! Loss: $best_loss\n$(best_combo)\n")
      (true, loss)
    else
      (false, loss)
    end
  end

  for combo in [initial_combo; random_combos]
    print("Trying $(combo)\n")
    try_combo(combo)
  end

  # Returns true if combo is better than best, or false if not or already tried
  try_combo_modification(arg_name, value) = begin
    combo = merge(best_combo, Dict(arg_name => value))
    if !(combo in combos_tried)
      print("Trying $(arg_name) = $(value)\n")
      is_best, loss = try_combo(combo)
      print("Loss = $(loss) for $(arg_name) = $(value)\n")
      is_best
    else
      false
    end
  end

  for iteration_i in 1:max_hyperparameter_coordinate_descent_iterations
    print("Hyperparameter coordinate descent iteration $iteration_i\n")

    for (arg_name, values) in kwargs
      best_value_i = findfirst(isequal(best_combo[arg_name]), values)
      direction    = nothing
      if best_value_i + 1 <= length(values) && try_combo_modification(arg_name, values[best_value_i + 1])
        direction     = 1
        best_value_i += 1
      elseif best_value_i - 1 >= 1 && try_combo_modification(arg_name, values[best_value_i - 1])
        direction     = -1
        best_value_i -= 1
      end

      while !isnothing(direction) && ((best_value_i + direction) in 1:length(values)) && try_combo_modification(arg_name, values[best_value_i + direction])
        best_value_i += direction
      end
    end

    if last_iteration_best_loss == best_loss
      break
    end
    last_iteration_best_loss = best_loss
  end

  print("Best hyperparameters (loss = $best_loss):\n")
  print("$best_combo\n")
end

end # module TrainingShared