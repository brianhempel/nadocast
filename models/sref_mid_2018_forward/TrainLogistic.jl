using Flux

import BSON
import Plots
import Random
import Statistics

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import NNTrain
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF
import SREFTrainShared

all_sref_forecasts = SREF.forecasts() # [1:33:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_sref_forecasts)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")

feature_normalization_forecast_sample_count = 100

forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)

forecasts_per_minibatch = 3

function reset_epoch()
  global forecasts_remaining_in_epoch
  forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)
end

# Should return [(X1, Y1), (X2, Y2), ...] where X1,Y1 is minibatch 1 etc...
#
# For 1D data, X1 should be (channels, item in batch)
function get_next_chunk()
  global forecasts_remaining_in_epoch

  minibatch_forecasts = collect(Iterators.take(forecasts_remaining_in_epoch, forecasts_per_minibatch))

  (X, y) = SREFTrainShared.get_data_and_labels(grid, conus_grid_bitmask, minibatch_forecasts)

  forecasts_remaining_in_epoch = forecasts_remaining_in_epoch[(forecasts_per_minibatch+1):length(forecasts_remaining_in_epoch)]

  if isempty(y)
    []
  else
    # Right now, only one minibatch per loading cycle.
    [(Float64.(X') :: Array{Float64,2}, y)]
  end
end


print("Normalizing features by sampling $feature_normalization_forecast_sample_count training forecasts")

(sample_X, _) = SREFTrainShared.get_data_and_labels(grid, conus_grid_bitmask, Iterators.take(Random.shuffle(train_forecasts), feature_normalization_forecast_sample_count))

sample_X = Float64.(sample_X')

ε = 1e-10

means         = Statistics.mean(sample_X, dims=2)
stddevs       = max.(ε, Statistics.std(sample_X, dims=2))
feature_count = length(means)

sample_X = nothing # freeeeeeee

println("done.")


minusone(dims...)      = zeros(dims...)   .- 1.0
minustwo(dims...)      = zeros(dims...)   .- 2.0
minusfive(dims...)     = zeros(dims...)   .- 5.0
minusten(dims...)      = zeros(dims...)   .- 10.0


dense_layer = Dense(feature_count, 1, σ, initb = minusfive)

logistic_model = Chain(
  x -> (x .- means) ./ stddevs,
  dense_layer,
  x -> x[1,:]
)

# Tracker.update!(dense_layer.W, -learning_rate * weight_decay * dense_layer.W)


losses(x, y) = Flux.binarycrossentropy.(logistic_model(x), y, ϵ = ε)
loss(x, y)   = sum(losses(x, y)) / length(y)

function test_loss(forecasts)
  test_loss            = 0.0
  forecast_count       = 0.0

  for forecast in forecasts
    (X, y) = SREFTrainShared.get_data_and_labels(grid, conus_grid_bitmask, [forecast])

    if isempty(y)
      continue
    end

    test_loss      += Tracker.data(loss(Float64.(X'), y))
    forecast_count += 1
  end

  test_loss / forecast_count
end

learning_rate        = 0.05
weight_decay         = 0.01 # As a fraction of learning rate. Affects L1 regularization.
last_validation_loss = nothing
epoch_n              = 1

while true
  global last_validation_loss
  global learning_rate
  global epoch_n

  print("Calculating validation loss")
  validation_loss = test_loss(validation_forecasts)
  println("done.")
  println("Validation loss: $validation_loss")
  if last_validation_loss != nothing && validation_loss > last_validation_loss
    learning_rate = learning_rate / 2.0
  else
    learning_rate = learning_rate * 1.1
  end
  println("New learning rate: $learning_rate")
  last_validation_loss = validation_loss
  BSON.@save "logistic_model_epoch_$(epoch_n-1)_validation_loss_$(validation_loss).bson" logistic_model


  println("===== Epoch $epoch_n =====")

  sgd_with_weight_decay = Flux.Optimise.optimiser(params(logistic_model), p -> Flux.Optimise.descentweightdecay(p, learning_rate, weight_decay))

  NNTrain.train_one_epoch!(get_next_chunk, loss, sgd_with_weight_decay)
  reset_epoch()

  # for forecast in validation_forecasts[[5,10,15,30,40,50]]
  #   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
  #   data     = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
  #   data     = Float64.(data')
  #   labels   = Float64.(TrainingShared.forecast_labels(grid, forecast))
  #   prefix   = "epoch_$(epoch_n)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
  #   Plots.png(Grib2.plot(grid, Float32.(Tracker.data(logistic_model(data)))), "$(prefix)_predictions.png")
  #   Plots.png(Grib2.plot(grid, Float32.(labels)), "$(prefix)_labels.png")
  #   Plots.png(Grib2.plot(grid, Float32.(Tracker.data(losses(data, labels)))), "$(prefix)_losses.png")
  #   println("done.")
  # end

  epoch_n += 1
end
