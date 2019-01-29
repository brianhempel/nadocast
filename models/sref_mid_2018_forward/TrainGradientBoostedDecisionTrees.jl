import Dates
import Plots
import Random

import MagicTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF
import FeatureEngineering

model_prefix = "gbdt_$(Dates.now())"

all_sref_forecasts = SREF.forecasts() # [1:33:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_sref_forecasts)


println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")


forecasts_per_chunk = 350
bin_split_forecast_sample_count = 100


# Returns (X_binned, labels)
function get_data_and_labels_binned(forecasts, bin_splits)
  transformer(X) = MagicTreeBoosting.apply_bins(X, bin_splits)
  TrainingShared.get_data_and_labels(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, forecasts, X_transformer = transformer)
end


# booster_config = [
#   "eta"              => 0.1, # learning rate (aka shrinkage rate)
#   "min_child_weight" => 50,
#   "max_leaves"       => 10,
#   "reg_alpha"        => 0.1, # L1 regularization on term weights
#   "reg_lambda"       => 0.1, # L2 regularization on term weights
#   "tree_method"      => "hist",
#   "grow_policy"      => "lossguide",
#   "objective"        => "binary:logistic",
#   "eval_metric"      => "logloss"
# ]

function save(epoch_i, validation_loss, bin_splits, trees)
  try
    mkdir("$(model_prefix)")
  catch
  end
  MagicTreeBoosting.save("$(model_prefix)/epoch_$(epoch_i)_$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
end

epoch_i    = 1
bin_splits = nothing
trees      = MagicTreeBoosting.Tree[MagicTreeBoosting.Leaf(-6.5,nothing, nothing)]
validation_X_binned, validation_y = (nothing, nothing)


println("Preparing bin splits by sampling $bin_split_forecast_sample_count training forecasts")

(bin_sample_X, _) = TrainingShared.get_data_and_labels(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), bin_split_forecast_sample_count))
bin_splits        = MagicTreeBoosting.prepare_bin_splits(bin_sample_X, 255)
bin_sample_X      = nothing # freeeeeeee

println("done.")


println("Loading training data")
X_binned, y = get_data_and_labels_binned(train_forecasts, bin_splits)
println("done.")

learning_rate = 0.03

while length(trees) <= 15 / learning_rate
  global epoch_i
  global bin_splits
  global trees
  global validation_X_binned
  global validation_y

  println("===== Epoch $epoch_i =====")

  trees =
    MagicTreeBoosting.train_on_binned(
      X_binned, y,
      prior_trees             = trees,
      iteration_count         = 5,
      min_data_weight_in_leaf = 30000.0,
      l2_regularization       = 1.0,
      max_leaves              = 4,
      max_depth               = 4,
      max_delta_score         = 5.0,
      learning_rate           = learning_rate,
      feature_fraction        = 0.2,
      feature_i_to_name       = SREF.feature_i_to_name,
    )

  # for chunk_of_forecasts in Iterators.partition(Random.shuffle(train_forecasts), forecasts_per_chunk)
  #   X_binned, y = get_data_and_labels_binned(chunk_of_forecasts, bin_splits)
  #
  #   trees =
  #     MagicTreeBoosting.train_on_binned(
  #       X_binned, y,
  #       prior_trees             = trees,
  #       iteration_count         = 3,
  #       min_data_weight_in_leaf = 6000.0,
  #       l2_regularization       = 1.0,
  #       max_leaves              = 4,
  #       max_depth               = 4,
  #       max_delta_score         = 5.0,
  #       learning_rate           = 0.1,
  #       feature_fraction        = 0.3,
  #     )
  # end

  if validation_X_binned == nothing
    println("Loading validation data")
    validation_X_binned, validation_y = get_data_and_labels_binned(validation_forecasts, bin_splits)
    println("done.")
  end
  print("Predicting...")
  validation_ŷ    = MagicTreeBoosting.predict_on_binned(validation_X_binned, trees)
  validation_loss = sum(MagicTreeBoosting.logloss.(validation_y, validation_ŷ)) / length(validation_y)
  println("done.")

  println("Validation loss: $validation_loss")

  save(epoch_i, validation_loss, bin_splits, trees)

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

  epoch_i += 1
end

