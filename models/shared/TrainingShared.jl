module TrainingShared

import Random
import Serialization
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
import Inventories
import StormEvents


MINUTE = 60 # seconds
HOUR   = 60*MINUTE

EVENT_TIME_WINDOW_HALF_SIZE  = 30*MINUTE
TORNADO_SPACIAL_RADIUS_MILES = 25.0


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
function forecasts_train_validation_test(all_forecasts; forecast_hour_range = 1:10000)
  # This filtering here is probably pretty slow.
  forecasts =
    filter(all_forecasts) do forecast
      (forecast.forecast_hour in forecast_hour_range) && is_relevant_forecast(forecast)
    end

  train_forecasts      = filter(is_train, forecasts)
  validation_forecasts = filter(is_validation, forecasts)
  test_forecasts       = filter(is_test, forecasts)

  (train_forecasts, validation_forecasts, test_forecasts)
end


function compute_forecast_labels(forecast) :: Array{Float32,1}
  StormEvents.grid_to_tornado_neighborhoods(forecast.grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
end


# Concatenating all the forecasts doubles peak memory usage if done in-RAM.
#
# We load each file to disk, after which we know the appropriate size needed
# and load back into RAM.
#
# If prior_predictor is provided, computes prior predictor's loss during loading.
# prior_predictor is fed the untransformed data.
#
# If save_dir already exists, reads the data in from disk.
function get_data_labels_weights(forecasts; save_dir = nothing, X_transformer = identity, X_and_labels_to_inclusion_probabilities = nothing, prior_predictor = nothing)
  if isnothing(save_dir)
    save_dir = "data_labels_weights_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  end
  if !isdir(save_dir)
    load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = X_transformer, X_and_labels_to_inclusion_probabilities = X_and_labels_to_inclusion_probabilities, prior_predictor = prior_predictor)
  end
  read_data_labels_weights_from_disk(save_dir)
end

# We can keep weights and labels in memory at least. It's the features that really kill us.
function load_data_labels_weights_to_disk(save_dir, forecasts; X_transformer = identity, X_and_labels_to_inclusion_probabilities = nothing, prior_predictor = nothing)
  mkpath(save_dir)

  save_path(path) = joinpath(save_dir, path)

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

  # Deterministic randomness for X_and_labels_to_inclusion_probabilities, presuming forecasts are given in the same order.
  rng = Random.MersenneTwister(12345)

  forecast_i = 1

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(forecasts)
    data_in_conus = Array{Float32}(undef, (conus_point_count, size(data, 2)))

    Threads.@threads for feature_i in 1:size(data, 2)
      data_in_conus[:, feature_i] = @view data[conus_grid_bitmask, feature_i]
    end

    forecast_labels = compute_forecast_labels(forecast)[conus_grid_bitmask] :: Array{Float32,1}

    if X_and_labels_to_inclusion_probabilities != nothing
      probabilities = Float32.(X_and_labels_to_inclusion_probabilities(data_in_conus, forecast_labels))
      probabilities = clamp.(probabilities, 0f0, 1f0)
      mask          = map(p -> p > 0f0 && rand(rng, Float32) <= p, probabilities)
      probabilities = probabilities[mask]

      forecast_weights = conus_grid_weights[mask] ./ probabilities
      forecast_labels  = forecast_labels[mask]
      data_masked      = data_in_conus[mask, :]
      X_transformed    = X_transformer(data_masked)

      # print("$(count(mask) / length(mask))")
    else
      forecast_weights = conus_grid_weights
      data_masked      = data_in_conus
      X_transformed    = X_transformer(data_in_conus)
    end

    if forecast_i == 1
      inventory_lines      = Forecasts.inventory(forecast)
      feature_descriptions = Inventories.inventory_line_description.(inventory_lines)
      write(save_path("features.txt"), join(feature_descriptions, "\n"))
    end

    if !isnothing(prior_predictor)
      predictions = prior_predictor(data_masked)
      push!(prior_losses, sum(logloss.(forecast_labels, predictions) .* forecast_weights))
      push!(prior_losses_weights, sum(forecast_weights))
      push!(prior_loss_forecasts, forecast)
    end

    data_file_name = Printf.@sprintf "data_%06d.serialized" forecast_i
    Serialization.serialize(save_path(data_file_name), X_transformed)

    append!(labels, forecast_labels)
    append!(weights, forecast_weights)

    forecast_i += 1

    print(".")
  end

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

  ()
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
function coordinate_descent_hyperparameter_search(f; kwargs...)
  combos_tried = []

  last_iteration_best_loss = Inf32
  best_loss                = Inf32
  iteration_i = 1

  best_combo =
    map(collect(kwargs)) do (arg_name, values)
      middle_value = values[1 + div(length(values)-1,2)]
      arg_name => middle_value
    end

  best_combo = Dict(best_combo)

  try_combo(arg_name, value) = begin
    combo = merge(best_combo, Dict(arg_name => value))
    if !(combo in combos_tried)
      println("Trying $(arg_name) = $(value)")
      loss = f(; combo...)
      println("Loss = $(loss) for $(arg_name) = $(value)")
      push!(combos_tried, combo)

      if loss < best_loss
        best_combo = combo
        best_loss  = loss
        println("New best!")
        println(best_combo)
        true
      else
        false
      end
    else
      false
    end
  end

  try_combo(first(best_combo)[1], first(best_combo)[2]) # Ensure initial combo has a loss

  while true
    println("Hyperparameter coordinate descent iteration $iteration_i")

    for (arg_name, values) in kwargs
      best_value_i = findfirst(isequal(best_combo[arg_name]), values)
      direction    = nothing
      if best_value_i + 1 <= length(values) && try_combo(arg_name, values[best_value_i + 1])
        direction     = 1
        best_value_i += 1
      elseif best_value_i - 1 >= 1 && try_combo(arg_name, values[best_value_i - 1])
        direction     = -1
        best_value_i -= 1
      end

      while direction != nothing && ((best_value_i + direction) in 1:length(values)) && try_combo(arg_name, values[best_value_i + direction])
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