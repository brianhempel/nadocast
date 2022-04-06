import Dates
import Random

import BSON
import GLMNet
# import Plots
# import Distributions
# using Lasso


push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import FeatureEngineeringShared

push!(LOAD_PATH, @__DIR__)
import SREF


model_prefix = "elastic_net_$(replace(string(Dates.now()), ":" => "."))"

all_sref_forecasts = SREF.forecasts() # [1:99:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

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
# println()
#
# inventory = Forecasts.inventory(SREF.example_forecast())
# for feature_i in feature_is
#   feature_name = FeatureEngineering.feature_i_to_name(inventory, feature_i)
#   println("$feature_i\t$feature_name")
# end
# println()
#
# X_transformer(X) = Float64.(@view X[:, feature_is])

feature_normalization_forecast_sample_count = 200

print("Normalizing features by sampling $feature_normalization_forecast_sample_count training forecasts")

import Statistics

sample_X, _, _ = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), feature_normalization_forecast_sample_count))

ε = 1f-10

means         = Statistics.mean(sample_X, dims=1) :: Array{Float32, 2}
stddevs       = max.(ε, Statistics.std(sample_X, dims=1)) :: Array{Float32, 2}
feature_count = length(means)

sample_X = nothing # freeeeeeee


# raw_feature_count = size(Forecasts.get_data(SREF.example_forecast()), 2)
#
# features_is = vcat(
#   collect(FeatureEngineeringShared.feature_range(FeatureEngineeringShared.raw_features_block, raw_feature_count)),
#   collect(FeatureEngineeringShared.feature_range(FeatureEngineeringShared.fifty_mi_mean_block, raw_feature_count)),
#   collect(FeatureEngineeringShared.feature_range(FeatureEngineeringShared.fifty_mi_forward_gradient_block, raw_feature_count)),
#   collect(FeatureEngineeringShared.feature_range(FeatureEngineeringShared.fifty_mi_leftward_gradient_block, raw_feature_count)),
#   collect(FeatureEngineeringShared.feature_range(FeatureEngineeringShared.fifty_mi_linestraddling_gradient_block, raw_feature_count)),
#   collect(FeatureEngineeringShared.feature_range(last(SREF.layer_blocks_to_make), raw_feature_count).stop+1:feature_count)
# )
#
#
# X_transformer(X) = begin
#   Float64.(((X .- means) ./ stddevs)[:, features_is])
# end
X_transformer(X) = Float64.((X .- means) ./ stddevs)

println("done.")


# print("Preparing pre-predictor for data inclusion...")
# import MemoryConstrainedTreeBoosting
#
# gbdt_model_path = (@__DIR__) * "/gbdt_2019-02-10T02-13-53.803_50mi_features_slightly_better/100_trees_loss_0.00498328331702105.model"
# # gbdt_model_path = (@__DIR__) * "/gbdt_2019-02-03T23-25-46.315_25mi_50mi_100mi_0.2_features_weighted/175_trees_loss_0.0049027752388540475.model"
#
# bin_splits, trees = MemoryConstrainedTreeBoosting.load(gbdt_model_path)
#
# calc_inclusion_probabilities(labels, is_near_storm_event) = begin
#   ŷ = MemoryConstrainedTreeBoosting.predict(X, bin_splits, trees)
#   losses = MemoryConstrainedTreeBoosting.logloss.(labels, ŷ)
#   map(1:size(X,1)) do i
#     max(0.01, losses[i]*100, labels[i])
#   end
# end
# println("done.")

calc_inclusion_probabilities(labels, is_near_storm_event) = map(label -> max(0.05, label), labels)


print("Loading training data")
X, y, weights = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, train_forecasts, X_transformer = X_transformer, calc_inclusion_probabilities = calc_inclusion_probabilities)
y       = Float64.(y)
weights = Float64.(weights)

y = [(1.0 .- y) y]
println("done.")


println("Loading validation data")
validation_calc_inclusion_probabilities(labels, is_near_storm_event) = map(label -> max(0.05, label), labels)

validation_X, validation_y, validation_weights = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, validation_forecasts, X_transformer = X_transformer, calc_inclusion_probabilities = validation_calc_inclusion_probabilities)
validation_y       = Float64.(validation_y)
validation_weights = Float64.(validation_weights)

validation_y = [(1.0 .- validation_y) validation_y]
println("done.")


best_model_file_path = nothing
best_loss            = nothing

alphas = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]

function print_solution(path, model_i)
  println("$(path.a0[model_i])\tIntercept")
  predictors = []
  for feature_i in 1:feature_count
    coeff = GLMNet.getindex(path.betas, feature_i, model_i)
    if coeff != 0.0
      push!(predictors, (coeff, SREF.feature_i_to_name(feature_i)))
    end
  end
  for (coeff, feature_name) in sort(predictors, by=(coeff_and_feature_name -> -abs(coeff_and_feature_name[1])))
    println("$(coeff)\t$(feature_name)")
  end
  println("$(length(predictors)) predictors.")
end

for alpha in alphas
  global best_model_file_path
  global best_loss

  println("Fitting Elastic Net α=$alpha...")

  # path = fit(LassoPath, X, Float32.(y), Distributions.Binomial(), α = alpha, wts = Float32.(weights), standardize = false, irls_maxiter = 60)

  # glmnet! bang version doesn't make a second copy of the data.
  path = GLMNet.glmnet(X, Float64.(y), GLMNet.Binomial(), weights = weights, nlambda = 100, alpha = alpha, standardize = false)#, algorithm = :modifiednewtonraphson)

  # X = nothing # freeeeeeeedom

  print("Predicting...")
  # validation_losses = deviance(path, validation_X, Float32.(validation_y), wts = Float32.(validation_weights)) / 2.0 # Deviance is twice the loss.
  validation_losses = GLMNet.loss(path, validation_X, validation_y, validation_weights) / 2.0 # Loss is logistic deviance, which is double the logloss.
  println("Validation losses: $validation_losses")

  validation_loss, best_model_i = findmin(validation_losses)

  model_file_path = "$(model_prefix)_alpha_$(alpha)_loss_$(validation_loss).model"


  if (best_loss == nothing && !isnan(validation_loss)) || validation_loss < best_loss
    BSON.@save model_file_path path best_model_i means stddevs

    best_model_file_path = model_file_path
    best_loss            = validation_loss
    println("New best: $best_model_file_path")
    print_solution(path, best_model_i)
  end
end




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
#   y = TrainingShared.compute_forecast_labels(grid, forecast)
#   ŷ = MemoryConstrainedTreeBoosting.predict(X, bin_splits, trees)
#   prefix = "$(model_prefix)/epoch_$(epoch_i)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
#   Plots.png(Grib2.plot(grid, Float32.(ŷ)), "$(prefix)_predictions.png")
#   Plots.png(Grib2.plot(grid, y), "$(prefix)_labels.png")
#   println("done.")
# end
