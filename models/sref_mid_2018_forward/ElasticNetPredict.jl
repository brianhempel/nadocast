import Dates
import Random

import BSON
import GLMNet
# import Plots
import Distributions
# using Lasso


push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF


model_path = "elastic_net_2019-02-07T01-47-09.475_alpha_0.99_loss_0.004674660862515808.model"

all_sref_forecasts = SREF.forecasts() # [1:2:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_sref_forecasts)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")


BSON.@load model_path path best_model_i means stddevs

X_transformer(X) = Float64.((X .- means) ./ stddevs)


println("Loading validation data")
validation_X, validation_y, validation_weights = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, validation_forecasts, X_transformer = X_transformer)

validation_y = [(1.0 .- validation_y) validation_y]
println("done.")


print("Predicting...")
# validation_losses = deviance(path, validation_X, Float32.(validation_y), wts = Float32.(validation_weights)) / 2.0 # Deviance is twice the loss.
validation_losses = GLMNet.loss(path, validation_X, validation_y, validation_weights) / 2.0 # Loss is logistic deviance, which is double the logloss.
println("Validation losses: $validation_losses")

println("Best model according to save file $(best_model_i)")

validation_loss, best_model_i = findmin(validation_losses)

println("Best model according to data $(best_model_i)\tLoss: $(validation_loss)")

println("Verifying loss")

σ(x) = 1.0 / (1.0 + exp(-x))

validation_ŷ = σ.(GLMNet.predict(path, validation_X, best_model_i))

const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0

# Copied from Flux.jl.
logloss(y, ŷ) = -y*log(ŷ + ε) - (1 - y)*log(1 - ŷ + ε)

loss = sum(logloss.(validation_y, validation_ŷ) .* validation_weights) ./ sum(validation_weights)

println("Loss: $loss")

println("Resaving")
BSON.@save replace(model_path, r"loss_[0-9\.]+\.model" => "loss_$loss.model") path best_model_i means stddevs

# Loading:
#
# import Distributions
#
# BSON.@load "$(model_prefix)_alpha_$(alpha)_loss_$(validation_loss).model" path best_model_i
#
#


# for forecast in validation_forecasts[[5,10,15,30,40,50]]
#   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
#   X = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
#   y = TrainingShared.compute_forecast_labels(grid, forecast)
#   ŷ = MemoryConstrainedTreeBoosting.predict(X, bin_splits, trees)
#   prefix = "$(model_prefix)/epoch_$(epoch_i)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
#   Plots.png(Grib2.plot(grid, Float32.(ŷ)), "$(prefix)_predictions.png")
#   Plots.png(Grib2.plot(grid, y), "$(prefix)_labels.png")
#   println("done.")
# end
