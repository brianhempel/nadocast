import Dates
import Random

import BSON
import GLMNet
# import Plots
# import Distributions



push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import HREF


model_prefix = "elastic_net_$(replace(repr(Dates.now()), ":" => "."))"

all_href_forecasts = HREF.forecasts() # [1:55:27856] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_href_forecasts)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")


feature_normalization_forecast_sample_count = 100

print("Normalizing features by sampling $feature_normalization_forecast_sample_count training forecasts")

import Statistics

sample_X, _, _ = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, HREF.get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), feature_normalization_forecast_sample_count))

ε = 1f-10

means         = Statistics.mean(sample_X, dims=1) :: Array{Float32, 2}
stddevs       = max.(ε, Statistics.std(sample_X, dims=1)) :: Array{Float32, 2}
feature_count = length(means)

sample_X = nothing # freeeeeeee

X_transformer(X) = Float64.((X .- means) ./ stddevs)

println("done.")


print("Preparing pre-predictor for data inclusion...")
import MemoryConstrainedTreeBoosting

gbdt_model_path = (@__DIR__) * "/gbdt_2019-02-01T00-23-46.807_rotation_invariant_winds_grid_weighted/265_trees_loss_0.004611843066900281.model"

bin_splits, trees = MemoryConstrainedTreeBoosting.load(gbdt_model_path)

X_and_labels_to_inclusion_probabilities(X, labels) = begin
  ŷ = MemoryConstrainedTreeBoosting.predict(X, bin_splits, trees)
  losses = MemoryConstrainedTreeBoosting.logloss.(labels, ŷ)
  map(1:size(X,1)) do i
    max(0.01, losses[i]*100, labels[i])
  end
end
println("done.")



print("Loading")
X, y, weights = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, HREF.get_feature_engineered_data, train_forecasts, X_transformer = X_transformer, X_and_labels_to_inclusion_probabilities = X_and_labels_to_inclusion_probabilities)

y = [(1.0 .- y) y]
println("done.")


println("Fitting Lasso...")

alpha = 0.99

# glmnet! bang version doesn't make a second copy of the data.
path = GLMNet.glmnet!(X, Float64.(y), GLMNet.Binomial(), weights = weights, nlambda = 50, alpha = alpha, maxit = 100000, standardize = false, algorithm = :modifiednewtonraphson)

X = nothing # freeeeeeeedom


println("Loading validation data")
validation_X, validation_y, validation_weights = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, HREF.get_feature_engineered_data, validation_forecasts, X_transformer = X_transformer)

validation_y = [(1.0 .- validation_y) validation_y]
println("done.")

print("Predicting...")

validation_losses = GLMNet.loss(path, validation_X, validation_y, validation_weights) / 2.0 # Loss is logistic deviance, which is double the logloss.
println("Validation losses: $validation_losses")

validation_loss, best_model_i = findmin(validation_losses)

BSON.@save "$(model_prefix)_alpha_$(alpha)_loss_$(validation_loss).model" path best_model_i means stddevs
