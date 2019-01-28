import Dates
import Plots

import BSON
import GLMNet

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF
import FeatureEngineering

model_prefix = "elastic_net_$(Dates.now())"

all_sref_forecasts = SREF.forecasts() # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_sref_forecasts)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")

# feature_importance_text = read("feature_importance.txt", String)
#
# feature_is = sort(unique(map(match -> parse(Int64, match[1]), eachmatch(r"^[0-9\.]+\t(\d+)\t"m, feature_importance_text))))
#
# println("Feature preselection: Using $(length(feature_is)) features used by the first 356 trees (best validation loss) in a GBDT run with the following parameters:
#       min_data_weight_in_leaf = 30000.0,
#       l2_regularization       = 1.0,
#       max_leaves              = 4,
#       max_depth               = 4,
#       max_delta_score         = 5.0,
#       learning_rate           = 0.03,
#       feature_fraction        = 0.2,")
#
# println("")
#
# inventory = Forecasts.inventory(SREF.example_forecast())
# for feature_i in feature_is
#   feature_name = FeatureEngineering.feature_i_to_name(inventory, feature_i)
#   println("$feature_i\t$feature_name")
# end
# println()
#
# X_transformer(X) = Float64.(@view X[:, feature_is])



print("Normalizing features by sampling $feature_normalization_forecast_sample_count training forecasts")

import Statistics

(sample_X, _) = TrainingShared.get_data_and_labels(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), feature_normalization_forecast_sample_count))

ε = 1f-10

means         = Statistics.mean(sample_X, dims=1)
stddevs       = max.(ε, Statistics.std(sample_X, dims=1))
feature_count = length(means)

sample_X = nothing # freeeeeeee

X_transformer(X) = Float64.((X .- means) ./ stddevs)

println("done.")



print("Loading")
X, y = TrainingShared.get_data_and_labels(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, train_forecasts, X_transformer = X_transformer)

y = [(1.0 .- y) y]
println("done.")


println("Fitting Lasso...")

alpha = 0.9

# glmnet! bang version doesn't make a second copy of the data.
path = GLMNet.glmnet!(X, Float64.(y), GLMNet.Binomial(), nlambda = 50, alpha = alpha, maxit = 100000, standardize = false)

X = nothing # freeeeeeeedom


println("Loading validation data")
validation_X, validation_y = TrainingShared.get_data_and_labels(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, validation_forecasts, X_transformer = X_transformer)

validation_y = [(1.0 .- validation_y) validation_y]
println("done.")

print("Predicting...")
validation_losses = GLMNet.loss(path, validation_X, validation_y) / 2.0 # Loss is logistic deviance, which is double the logloss.
println("Validation losses: $validation_losses")

(validation_loss, best_model_i) = findmin(validation_losses)

BSON.@save "$(model_prefix)_alpha_$(alpha)_loss_$(validation_loss).model" path best_model_i

# Loading:
#
# import Distributions
#
# BSON.@load "$(model_prefix)_alpha_$(alpha)_loss_$(validation_loss).model" path best_model_i
#
# ŷ = GLMNet.predict(path, X, best_model_i)


# for forecast in validation_forecasts[[5,10,15,30,40,50]]
#   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
#   X = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
#   y = TrainingShared.forecast_labels(grid, forecast)
#   ŷ = MagicTreeBoosting.predict(X, bin_splits, trees)
#   prefix = "$(model_prefix)/epoch_$(epoch_i)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
#   Plots.png(Grib2.plot(grid, Float32.(ŷ)), "$(prefix)_predictions.png")
#   Plots.png(Grib2.plot(grid, y), "$(prefix)_labels.png")
#   println("done.")
# end
