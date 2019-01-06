using Flux

import Random

push!(LOAD_PATH, (@__DIR__) * "/../..")

# println(LOAD_PATH)

import Conus
import Forecasts
import NNTrain
import StormEvents

push!(LOAD_PATH, @__DIR__)
import SREF

MINUTE = 60 # seconds

TORNADO_TIME_WINDOW_HALF_SIZE = 30*MINUTE
TORNADO_SPACIAL_RADIUS_MILES  = 25.0

function is_relevant_forecast(forecast)
  for tornado in StormEvents.tornadoes
    tornado_relevant_time_range =
      (tornado.start_seconds_from_epoch_utc - TORNADO_TIME_WINDOW_HALF_SIZE + 1):(tornado.end_seconds_from_epoch_utc + TORNADO_TIME_WINDOW_HALF_SIZE - 1)

    if Forecasts.valid_time_in_seconds_since_epoch_utc(forecast) in tornado_relevant_time_range
      return true
    end
  end
  false
end

all_sref_forecasts = SREF.forecasts()[1:1000]

println("$(length(all_sref_forecasts)) total SREF forecast hour snapshots")

forecasts = filter(is_relevant_forecast, all_sref_forecasts)

println("$(length(forecasts)) relevant forecast hour snapshots")

grid          = Forecasts.grid(forecasts[1])
feature_count = length(Forecasts.inventory(forecasts[1]))

train_forecasts      = filter(Forecasts.is_train, forecasts)
validation_forecasts = filter(Forecasts.is_validation, forecasts)
test_forecasts       = filter(Forecasts.is_test, forecasts)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")

forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)

forecasts_per_minibatch = 10

# Should return [(X1, Y1), (X2, Y2), ...] where X1,Y1 is minibatch 1 etc...
#
# For 2D data, X1 etc should be 4-dimensional (x, y, channels, image in batch) or (y, x, channels, image in batch)
#
# In our case, it is indeed (x, y, channels, image in batch), although y is south to north.
function get_next_chunk()
  print("Loading chunk")
  global forecasts_remaining_in_epoch

  X = Array{Float32,4}[]
  Y = Array{Float32,4}[]
  for forecast in Iterators.take(forecasts_remaining_in_epoch, forecasts_per_minibatch)
    data   = Forecasts.data(forecast)
    data   = reshape(data, (grid.width, grid.height, :))
    labels = StormEvents.grid_to_tornado_neighborhoods(grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), TORNADO_TIME_WINDOW_HALF_SIZE)
    labels = reshape(labels, (grid.width, grid.height))

    X = cat(X, data,   dims = 4)
    Y = cat(Y, labels, dims = 3)
    print(".")
  end
  forecasts_remaining_in_epoch = forecasts_remaining_in_epoch[(forecasts_per_minibatch+1):length(forecasts_remaining_in_epoch)]
  println("done.")

  # Right now, only one minibatch per loading cycle.
  [(X, Y)]
end

conus_on_grid = map(latlon -> Conus.is_in_conus(latlon) ? 1.0f0 : 0.0f0, grid.latlons)

logistic_model = Chain(
  Conv((1,1), feature_count => 1, σ),
  x -> reshape(x, (size(x, 1), size(x, 2), size(x, 4))) # Flatten away the single channel dimension
)

loss(x, y) = sum(Flux.binarycrossentropy(logistic_model(x), y) .* reshape(repeat(conus_on_grid, size(y, 3)), size(y)))

function test_loss(forecasts)
  test_loss      = 0.0f0
  point_count_on_conus = sum(conus_on_grid)
  for forecast in forecasts
    data   = Forecasts.data(forecast)
    data   = reshape(data, (grid.width, grid.height, :, 1))
    labels = StormEvents.grid_to_tornado_neighborhoods(grid, TORNADO_SPACIAL_RADIUS_MILES, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), TORNADO_TIME_WINDOW_HALF_SIZE)
    labels = reshape(labels, (grid.width, grid.height, 1))

    test_loss += Tracker.data(loss(data, labels)) / point_count_on_conus
    print(".")
  end

  test_loss
end

while true
  print("Calculating validation loss")
  println("Validation loss: $(test_loss(validation_forecasts))")
  NNTrain.train_one_epoch!(get_next_chunk, loss, params(logistic_model), Descent(0.1))
end
