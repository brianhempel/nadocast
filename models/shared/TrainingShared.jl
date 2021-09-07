module TrainingShared

import Random
import Serialization
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
import Inventories
import StormEvents
# import PlotMap


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

EVENT_TIME_WINDOW_HALF_SIZE  = 30*MINUTE
TORNADO_SPACIAL_RADIUS_MILES = 25.0

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
    _conus_tornado_hours_in_seconds_from_epoch_utc = StormEvents.event_hours_set_in_seconds_from_epoch_utc(StormEvents.tornadoes(), EVENT_TIME_WINDOW_HALF_SIZE)
  end

  _conus_tornado_hours_in_seconds_from_epoch_utc
end


# Use all forecasts in which there is a tornado, wind, or hail event in that or an adjacent hour.
# (Though we are only looking for tornadoes for now.)
function is_relevant_forecast(forecast)
  valid_time = Forecasts.valid_time_in_seconds_since_epoch_utc(forecast)

  (valid_time in conus_event_hours_in_seconds_from_epoch_utc()) ||
    ((valid_time + HOUR) in conus_event_hours_in_seconds_from_epoch_utc()) ||
    ((valid_time - HOUR) in conus_event_hours_in_seconds_from_epoch_utc())
end

function forecast_is_tornado_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_tornado_hours_in_seconds_from_epoch_utc()
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
      all_forecasts
    end

  train_forecasts      = filter(is_train, forecasts)
  validation_forecasts = filter(is_validation, forecasts)
  test_forecasts       = filter(is_test, forecasts)

  (train_forecasts, validation_forecasts, test_forecasts)
end


function compute_forecast_labels(forecast) :: Array{Float32,1}
  StormEvents.grid_to_conus_tornado_neighborhoods(forecast.grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
end

function compute_is_near_storm_event(forecast) :: Array{Float32,1}
  StormEvents.grid_to_conus_event_neighborhoods(forecast.grid, NEAR_EVENT_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), NEAR_EVENT_TIME_WINDOW_HALF_SIZE)
end


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
  isdir(save_dir) && isfile(joinpath(save_dir, "labels.serialized")) && isfile(joinpath(save_dir, "weights.serialized"))
end

function get_data_labels_weights(forecasts; save_dir = nothing, X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, compute_forecast_labels = compute_forecast_labels)
  if isnothing(save_dir)
    save_dir = "data_labels_weights_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  end
  if !finished_loading(save_dir)
    load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = X_transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, prior_predictor = prior_predictor, compute_forecast_labels = compute_forecast_labels)
  end
  read_data_labels_weights_from_disk(save_dir)
end

# Loads the data to disk but does not read it back.
# Returns (data_count, feature_count) if the data wasn't already saved to disk.
function prepare_data_labels_weights(forecasts; save_dir = nothing, X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, compute_forecast_labels = compute_forecast_labels)
  if isnothing(save_dir)
    save_dir = "data_labels_weights_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  end
  if finished_loading(save_dir)
    println("$save_dir appears to have finished loading. Skipping.")
    return (nothing, nothing)
  end
  load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = X_transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, prior_predictor = prior_predictor, compute_forecast_labels = compute_forecast_labels)
end


function mask_rows_threaded(data, mask; final_row_count=count(mask))
  out = Array{Float32}(undef, (final_row_count, size(data, 2)))

  Threads.@threads for col_i in 1:size(data, 2)
    out[:, col_i] = @view data[mask, col_i]
  end

  out
end

# We can keep weights and labels in memory at least. It's the features that really kill us.
function load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = identity, calc_inclusion_probabilities = nothing, prior_predictor = nothing, compute_forecast_labels = compute_forecast_labels)
  mkpath(save_dir)

  save_path(path) = joinpath(save_dir, path)
  serialize_async(path, value) = Threads.@spawn Serialization.serialize(path, value)

  labels  = Float32[]
  weights = Float32[]

  prior_losses         = []
  prior_losses_weights = []
  prior_loss_forecasts = []
  ε             = eps(1.0f0)
  logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε) # Copied from Flux.jl

  grid = first(forecasts).grid

  conus_on_grid      = map(latlon -> Conus.is_in_conus(latlon) ? 1.0f0 : 0.0f0, grid.latlons)
  conus_grid_bitmask = (conus_on_grid .== 1.0f0)
  conus_grid_weights = Float32.(grid.point_weights[conus_grid_bitmask])

  conus_point_count  = count(conus_grid_bitmask)

  # Deterministic randomness for calc_inclusion_probabilities, presuming forecasts are given in the same order.
  rng = Random.MersenneTwister(12345)

  forecast_i = 1

  start_time = time_ns()

  serialization_task = nothing
  feature_count = 0

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(forecasts)
    forecast_labels = compute_forecast_labels(forecast)[conus_grid_bitmask]     :: Array{Float32,1}

    # PlotMap.plot_debug_map("tornadoes_$(Forecasts.valid_yyyymmdd_hhz(forecast))", grid, compute_forecast_labels(forecast))
    # PlotMap.plot_debug_map("near_events_$(Forecasts.valid_yyyymmdd_hhz(forecast))", grid, compute_is_near_storm_event(forecast))

    if !isnothing(calc_inclusion_probabilities)
      is_near_storm_event = compute_is_near_storm_event(forecast)[conus_grid_bitmask] :: Array{Float32,1}
      probabilities       = Float32.(calc_inclusion_probabilities(forecast_labels, is_near_storm_event))
      probabilities       = clamp.(probabilities, 0f0, 1f0)
      mask                = map(p -> p > 0f0 && rand(rng, Float32) <= p, probabilities)
      probabilities       = probabilities[mask]

      forecast_weights = conus_grid_weights[mask] ./ probabilities
      forecast_labels  = forecast_labels[mask]
      # Combine conus mask and mask
      data_mask = conus_grid_bitmask[:]
      data_mask[conus_grid_bitmask] = mask
      data_masked = mask_rows_threaded(data, data_mask)

      # print("$(count(mask) / length(mask))")
    else
      forecast_weights = conus_grid_weights
      data_masked      = mask_rows_threaded(data, conus_grid_bitmask; final_row_count=conus_point_count)
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
      push!(prior_losses, sum(logloss.(forecast_labels, predictions) .* forecast_weights))
      push!(prior_losses_weights, sum(forecast_weights))
      push!(prior_loss_forecasts, forecast)
    end

    data_file_name = Printf.@sprintf "data_%06d.serialized" forecast_i

    !isnothing(serialization_task) && wait(serialization_task) # Make sure we don't get ahead of disk writes.
    serialization_task = serialize_async(save_path(data_file_name), X_transformed)

    append!(labels, forecast_labels)
    append!(weights, forecast_weights)

    elapsed = (Base.time_ns() - start_time) / 1.0e9
    print("\r$forecast_i/~$(length(forecasts)) forecasts loaded.  $(elapsed / forecast_i)s each.  ~$((elapsed / forecast_i) * (length(forecasts) - forecast_i) / 60 / 60) hours left.            ")

    forecast_i += 1
  end

  wait(serialization_task) # Synchronize

  Serialization.serialize(save_path("labels.serialized"), labels)
  Serialization.serialize(save_path("weights.serialized"), weights)

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

  println("")

  (length(labels), feature_count)
end

function read_data_labels_weights_from_disk(save_dir)
  save_path(path) = joinpath(save_dir, path)

  labels  = Serialization.deserialize(save_path("labels.serialized"))
  weights = Serialization.deserialize(save_path("weights.serialized"))

  @assert length(labels) == length(weights)
  data_count = length(labels)

  data_file_names = sort(filter(file_name -> startswith(file_name, "data_"), readdir(save_dir)), by=(name -> parse(Int64, split(name, r"_|\.")[2])))

  forecast_data_1 = Serialization.deserialize(save_path(data_file_names[1]))

  feature_count = size(forecast_data_1, 2)
  feature_type  = eltype(forecast_data_1)

  forecast_data_1 = nothing # free

  data = Array{feature_type}(undef, (data_count, feature_count))

  row_i = 1

  for data_file_name in data_file_names
    forecast_data = Serialization.deserialize(save_path(data_file_name))

    forecast_row_count, forecast_feature_count = size(forecast_data)

    @assert feature_count == forecast_feature_count

    data[row_i:(row_i + forecast_row_count - 1), :] = forecast_data

    row_i += forecast_row_count
  end

  @assert row_i - 1 == data_count

  (data, labels, weights)
end


# Provide a function that returns a loss and kwargs that contain arrays of values to try.
# *** Assumes each array of values is in order ***
function coordinate_descent_hyperparameter_search(f; random_start_count = 0, kwargs...)
  combos_tried = []

  last_iteration_best_loss = Inf32
  best_loss                = Inf32
  iteration_i = 1

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
      hyperparameters_to_combo(rand)
    end

  best_combo = Dict(initial_combo)

  # Returns (is_best, loss)
  try_combo(combo) = begin
    loss = f(; combo...)
    push!(combos_tried, combo)

    if loss < best_loss
      best_combo = combo
      best_loss  = loss
      println("New best! Loss: $best_loss")
      println(best_combo)
      (true, loss)
    else
      (false, loss)
    end
  end

  for combo in [initial_combo; random_combos]
    print("Trying ")
    println(combo)
    try_combo(combo)
  end

  # Returns true if combo is better than best, or false if not or already tried
  try_combo_modification(arg_name, value) = begin
    combo = merge(best_combo, Dict(arg_name => value))
    if !(combo in combos_tried)
      println("Trying $(arg_name) = $(value)")
      is_best, loss = try_combo(combo)
      println("Loss = $(loss) for $(arg_name) = $(value)")
      is_best
    else
      false
    end
  end

  while true
    println("Hyperparameter coordinate descent iteration $iteration_i")

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

      while direction != nothing && ((best_value_i + direction) in 1:length(values)) && try_combo_modification(arg_name, values[best_value_i + direction])
        best_value_i += direction
      end
    end

    if last_iteration_best_loss == best_loss
      break
    end
    last_iteration_best_loss = best_loss
    iteration_i += 1
  end

  println("Best hyperparameters (loss = $best_loss):")
  println(best_combo)
end

end # module TrainingShared