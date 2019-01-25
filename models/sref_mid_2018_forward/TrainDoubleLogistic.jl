using Flux

import BSON
import DelimitedFiles
import Plots
import Random

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import NNTrain
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF

all_sref_forecasts = SREF.forecasts() # [1:11:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_on_grid, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_on_grid_train_validation_test(all_sref_forecasts)

feature_count = size(SREF.get_feature_engineered_data(train_forecasts[1], Forecasts.get_data(train_forecasts[1])), 2)

conus_on_grid = Float64.(conus_on_grid)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")

normalizing_factors = DelimitedFiles.readdlm((@__DIR__) * "/normalizing_factors.txt", Float64)[:,1] :: Array{Float64,1}

forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)

forecasts_per_minibatch = 3

# Should return [(X1, Y1), (X2, Y2), ...] where X1,Y1 is minibatch 1 etc...
#
# For 1D data, X1 should be (channels, item in batch)
#
# For 2D data, X1 etc should be 4-dimensional (x, y, channels, image in batch) or (y, x, channels, image in batch)
function reset_epoch()
  global forecasts_remaining_in_epoch
  forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)
end

function get_next_chunk()
  print("Loading chunk")
  global forecasts_remaining_in_epoch

  X = nothing
  Y = Array{Float64,1}[]
  # X = Array{Float32,4}[]
  # Y = Array{Float32,3}[]
  minibatch_forecasts = collect(Iterators.take(forecasts_remaining_in_epoch, forecasts_per_minibatch))
  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(minibatch_forecasts)
    data = SREF.get_feature_engineered_data(forecast, data)

    transposed_normalized = Float64.(data') ./ normalizing_factors # dims: layer_count by grid_count

    # transposed = collect(data') # dims: layer_count by grid_count
    # data   = reshape(data, (grid.width, grid.height, :))
    labels = Float64.(TrainingShared.forecast_labels(grid, forecast))
    # labels = reshape(labels, (grid.width, grid.height))

    if X == nothing
      X = transposed_normalized
    else
      X = hcat(X, transposed_normalized)
    end
    Y = vcat(Y, labels)
    # X = cat(X, data,   dims = 4)
    # Y = cat(Y, labels, dims = 3)
    print(".")
  end
  forecasts_remaining_in_epoch = forecasts_remaining_in_epoch[(forecasts_per_minibatch+1):length(forecasts_remaining_in_epoch)]
  println("done.")

  if X != nothing
    # Right now, only one minibatch per loading cycle.
    [(X :: Array{Float64,2}, Y)]
  else
    []
  end
end



minusone(dims...)      = zeros(dims...)   .- 1.0
minustwo(dims...)      = zeros(dims...)   .- 2.0
minusfive(dims...)     = zeros(dims...)   .- 5.0
minusten(dims...)      = zeros(dims...)   .- 10.0

double_logistic_model = Chain(
  Dense(feature_count, 2, σ, initb = minusone),
  x -> x[1,:] .* x[2,:]
)


losses(x, y) = Flux.binarycrossentropy.(double_logistic_model(x), y) .* repeat(conus_on_grid, div(length(y), length(conus_on_grid)))
loss(x, y)   = sum(losses(x, y))

function test_loss(forecasts)
  test_loss            = 0.0
  forecast_count       = 0.0
  point_count_on_conus = sum(conus_on_grid)

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    # print("Loading $(Forecasts.time_title(forecast))...")
    data = SREF.get_feature_engineered_data(forecast, data)

    transposed_normalized = Float64.(data') ./ normalizing_factors # dims: layer_count by grid_count

    # print("transposing...")
    # transposed = collect(data') # dims: layer_count by grid_count
    # data   = reshape(data, (grid.width, grid.height, :, 1))
    # print("labeling...")
    labels = Float64.(TrainingShared.forecast_labels(grid, forecast))
    # println("done.")
    # labels_transposed = collect(labels')
    # labels = reshape(labels, (grid.width, grid.height, 1))


    test_loss += Tracker.data(loss(transposed_normalized, labels)) / point_count_on_conus
    forecast_count += 1
    print(".")
  end

  test_loss / forecast_count
end

learning_rate        = 0.005
weight_decay         = 0.4 # As a fraction of learning rate. Affects L1 regularization.
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
    println("New learning rate: $learning_rate")
  end
  last_validation_loss = validation_loss
  BSON.@save "double_logistic_model_epoch_$(epoch_n-1)_validation_loss_$(validation_loss).bson" double_logistic_model

  sgd_with_weight_decay = Flux.Optimise.optimiser(params(double_logistic_model), p -> Flux.Optimise.descentweightdecay(p, learning_rate, weight_decay))

  NNTrain.train_one_epoch!(get_next_chunk, loss, sgd_with_weight_decay)
  reset_epoch()

  # for forecast in validation_forecasts[[5,10,15,30,40,50]]
  #   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
  #   data     = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
  #   data     = Float64.(data') ./ normalizing_factors
  #   labels   = Float64.(TrainingShared.forecast_labels(grid, forecast))
  #   prefix   = "epoch_$(epoch_n)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
  #   Plots.png(Grib2.plot(grid, Float32.(Tracker.data(double_logistic_model(data)))), "$(prefix)_predictions.png")
  #   Plots.png(Grib2.plot(grid, Float32.(labels)), "$(prefix)_labels.png")
  #   Plots.png(Grib2.plot(grid, Float32.(Tracker.data(losses(data, labels)))), "$(prefix)_losses.png")
  #   println("done.")
  # end

  epoch_n += 1
end
