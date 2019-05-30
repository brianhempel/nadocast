module TrainingShared

import Random

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Conus
import Forecasts
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
    _conus_event_hours_in_seconds_from_epoch_utc = StormEvents.conus_event_hours_set_in_seconds_from_epoch_utc(EVENT_TIME_WINDOW_HALF_SIZE)
  end

  _conus_event_hours_in_seconds_from_epoch_utc
end


_conus_tornado_hours_in_seconds_from_epoch_utc = nothing

function conus_tornado_hours_in_seconds_from_epoch_utc()
  global _conus_tornado_hours_in_seconds_from_epoch_utc

  if isnothing(_conus_tornado_hours_in_seconds_from_epoch_utc)
    _conus_tornado_hours_in_seconds_from_epoch_utc = Set{Int64}()

    for event in StormEvents.tornadoes()
      event_time_range =
        (event.start_seconds_from_epoch_utc - EVENT_TIME_WINDOW_HALF_SIZE):(event.end_seconds_from_epoch_utc + EVENT_TIME_WINDOW_HALF_SIZE - 1)

      for hour_from_epoch in fld(event_time_range.start, HOUR):fld(event_time_range.stop, HOUR)
        hour_second = hour_from_epoch*HOUR
        if hour_second in event_time_range
          push!(_conus_tornado_hours_in_seconds_from_epoch_utc, hour_second)
        end
      end
    end
  end

  _conus_tornado_hours_in_seconds_from_epoch_utc
end


# Use all forecasts in which there is a tornado, wind, or hail event.
# (Though we are only looking for tornadoes for now.)
function is_relevant_forecast(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_event_hours_in_seconds_from_epoch_utc()
end

function forecast_is_tornado_hour(forecast)
  Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in conus_tornado_hours_in_seconds_from_epoch_utc()
end


# returns (grid, conus_on_grid, train_forecasts, validation_forecasts, test_forecasts)
function forecasts_grid_conus_grid_bitmask_train_validation_test(all_forecasts; forecast_hour_range = 1:10000)
  # This filtering here is probably pretty slow.
  forecasts =
    filter(all_forecasts) do forecast
      (forecast.forecast_hour in forecast_hour_range) && is_relevant_forecast(forecast)
    end

  grid = Forecasts.grid(forecasts[1])

  train_forecasts      = filter(is_train, forecasts)
  validation_forecasts = filter(is_validation, forecasts)
  test_forecasts       = filter(is_test, forecasts)

  conus_on_grid      = map(latlon -> Conus.is_in_conus(latlon) ? 1.0f0 : 0.0f0, grid.latlons)
  conus_grid_bitmask = (conus_on_grid .== 1.0f0)

  (grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts)
end


function forecast_labels(grid, forecast) :: Array{Float32,1}
  StormEvents.grid_to_tornado_neighborhoods(grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), EVENT_TIME_WINDOW_HALF_SIZE)
end


# get_feature_engineered_data should be a function that takes a forecast and the raw data and returns new data
# c.f. SREF.get_feature_engineered_data
#
# concatenating all the forecasts doubles peak memory usage if done in-RAM
# so we do the concatenation as an on-disk file append
function get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, forecasts; X_transformer = identity, X_and_labels_to_inclusion_probabilities = nothing)
  # Xs      = []
  # Ys      = []
  # weights = []

  data_count    = 0
  feature_count = nothing
  feature_type  = nothing
  feature_files = nothing
  labels_file   = nothing
  weights_file  = nothing

  loading_tmp_dir = "loading_tmp_$(Random.rand(Random.RandomDevice(), UInt64))" # ignore random seed, which we may have set elsewhere to ensure determinism
  mkpath(loading_tmp_dir)

  concat_path(name)                                = joinpath(loading_tmp_dir, name)
  open_concat_file(name)                           = open(concat_path(name), "w")
  read_and_remove_concat_file(name, data_count, T) = begin
    buffer = Array{T}(undef, data_count)
    read!(concat_path(name), buffer)
    # Remove as we go in case we are swapping and need the space.
    rm(concat_path(name))
    buffer
  end

  conus_grid_weights = Float32.(grid.point_weights[conus_grid_bitmask])

  # Deterministic randomness for X_and_labels_to_inclusion_probabilities, presuming forecasts are given in the same order.
  rng = Random.MersenneTwister(12345)

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data = get_feature_engineered_data(forecast, data)

    data_in_conus = data[conus_grid_bitmask, :]
    labels        = forecast_labels(grid, forecast)[conus_grid_bitmask] :: Array{Float32,1}

    if X_and_labels_to_inclusion_probabilities != nothing
      probabilities = Float32.(X_and_labels_to_inclusion_probabilities(data_in_conus, labels))
      probabilities = clamp.(probabilities, 0f0, 1f0)
      mask          = map(p -> p > 0f0 && rand(rng, Float32) <= p, probabilities)
      probabilities = probabilities[mask]

      forecast_weights = conus_grid_weights[mask] ./ probabilities
      labels           = labels[mask]
      X_transformed    = X_transformer(data_in_conus[mask, :])

      # print("$(count(mask) / length(mask))")
    else
      forecast_weights = conus_grid_weights
      X_transformed    = X_transformer(data_in_conus)
    end

    if isnothing(feature_files)
      feature_count = size(X_transformed,2)
      feature_type  = typeof(X_transformed[1,1])
      feature_files =
        map(1:feature_count) do feature_i
          open_concat_file("feature_$(feature_i)")
        end
      labels_file  = open_concat_file("labels")
      weights_file = open_concat_file("weights")
    end

    for feature_i in 1:feature_count
      write(feature_files[feature_i], X_transformed[:,feature_i])
    end
    write(labels_file,  labels)
    write(weights_file, forecast_weights)

    data_count += length(labels)

    # push!(Xs, X_transformed)
    # push!(Ys, labels)
    # push!(weights, forecast_weights)

    print(".")
  end

  for feature_i in 1:feature_count
    close(feature_files[feature_i])
  end
  close(labels_file)
  close(weights_file)

  X       = Array{feature_type}(undef, (data_count, feature_count))
  Y       = Array{Float32}(undef, data_count)
  weights = Array{Float32}(undef, data_count)

  for feature_i in 1:feature_count
    X[:, feature_i] = read_and_remove_concat_file("feature_$(feature_i)", data_count, feature_type)
  end
  Y       = read_and_remove_concat_file("labels", data_count, Float32)
  weights = read_and_remove_concat_file("weights", data_count, Float32)

  rm(loading_tmp_dir, recursive = true)

  # @assert vcat(Xs...) == X
  # @assert vcat(Ys...) == Y
  # @assert vcat(weights...) == weights2

  # (vcat(Xs...), vcat(Ys...), vcat(weights...))
  (X, Y, weights)
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