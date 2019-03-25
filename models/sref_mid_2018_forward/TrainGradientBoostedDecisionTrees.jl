import Dates
# import Plots
# import Random

import MemoryConstrainedTreeBoosting

# push!(LOAD_PATH, (@__DIR__) * "/../../lib")
# import Forecasts
# import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainingShared
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import SREF


model_prefix = "gbdt_$(replace(repr(Dates.now()), ":" => "."))"

all_sref_forecasts  = SREF.forecasts()[1:33:21034]
forecast_hour_range = 1:87 # 4:39             # SREF files come out 3-4 hours after run time

# (grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
#   TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(all_sref_forecasts, forecast_hour_range = forecast_hour_range)
#
#
# println("$(length(train_forecasts)) for training.")
# println("$(length(validation_forecasts)) for validation.")
# println("$(length(test_forecasts)) for testing.")
#
#
# # forecasts_per_chunk = 350
# bin_split_forecast_sample_count = 100
#
#
# # Returns (X_binned, labels, weights)
# function get_data_labels_weights_binned(forecasts, bin_splits)
#   transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
#   TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, forecasts, X_transformer = transformer)
# end
#
# # Returns (X_binned_compressed, labels, weights)
# function get_data_labels_weights_binned_compressed(forecasts, bin_splits)
#   ys                = Vector{Float32}[]
#   weightss          = Vector{Float32}[]
#   binned_compressed = nothing
#   for forcast_chunk in Iterators.partition(forecasts, 10)
#     (X_chunk, labels, weights) = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, forcast_chunk)
#
#     binned_compressed = MemoryConstrainedTreeBoosting.bin_and_compress(X_chunk, bin_splits, prior_data = binned_compressed)
#
#     push!(ys, labels)
#     push!(weightss, weights)
#   end
#
#   (MemoryConstrainedTreeBoosting.finalize_loading(binned_compressed), vcat(ys...), vcat(weightss...))
# end
#
#
# # Best (loss = 0.00480496), for SREF tornado hours, with 50mi features
# # Dict{Symbol,Real}(:max_depth=>4,:max_delta_score=>5.0,:learning_rate=>0.03,:max_leaves=>6,:l2_regularization=>3.0,:feature_fraction=>0.5,:bagging_temperature=>0.25,:min_data_weight_in_leaf=>150000.0)
#
# function save(validation_loss, bin_splits, trees)
#   try
#     mkdir("$(model_prefix)")
#   catch
#   end
#   MemoryConstrainedTreeBoosting.save("$(model_prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
# end
#
# println("Preparing bin splits by sampling $bin_split_forecast_sample_count training forecasts")
#
# (bin_sample_X, _, _) = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, SREF.get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), bin_split_forecast_sample_count))
# bin_splits           = MemoryConstrainedTreeBoosting.prepare_bin_splits(bin_sample_X)
# bin_sample_X         = nothing # freeeeeeee
#
# println("done.")
#
#
# println("Loading training data")
# X_binned, y, weights = get_data_labels_weights_binned(train_forecasts, bin_splits)
# println("done.")
#
# println("Loading validation data")
# validation_X_binned, validation_y, validation_weights = get_data_labels_weights_binned(validation_forecasts, bin_splits)
# println("done.")

# Best (loss = 0.00480496), for SREF tornado hours, with 50mi features
# Dict{Symbol,Real}(:max_depth=>4,:max_delta_score=>5.0,:learning_rate=>0.03,:max_leaves=>6,:l2_regularization=>3.0,:feature_fraction=>0.5,:bagging_temperature=>0.25,:min_data_weight_in_leaf=>150000.0)

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    all_sref_forecasts;
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    get_feature_engineered_data = SREF.get_feature_engineered_data,
    bin_split_forecast_sample_count = 100,
    max_iterations_without_improvement = 20,

    min_data_weight_in_leaf = [7000.0, 10000.0, 15000.0, 20000.0, 35000.0, 50000.0, 70000.0, 100000.0, 150000.0, 200000.0, 350000.0, 500000.0],
    l2_regularization       = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    max_leaves              = [3, 4, 5, 6, 8, 10, 12, 15],
    max_depth               = [2, 3, 4, 5, 6, 7],
    max_delta_score         = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 1000.0],
    learning_rate           = [0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01],
    feature_fraction        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    bagging_temperature     = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  )

# @time MemoryConstrainedTreeBoosting.train_on_binned(
#   X_binned, y,
#   prior_trees             = trees,
#   weights                 = weights,
#   iteration_count         = Int64(15 / learning_rate),
#   min_data_weight_in_leaf = 30000.0,
#   l2_regularization       = 1.0,
#   max_leaves              = 4,
#   max_depth               = 4,
#   max_delta_score         = 5.0,
#   learning_rate           = learning_rate,
#   feature_fraction        = 0.8,
#   bagging_temperature     = 0.0,
#   feature_i_to_name       = SREF.feature_i_to_name,
#   iteration_callback      = iteration_callback
# )



# for chunk_of_forecasts in Iterators.partition(Random.shuffle(train_forecasts), forecasts_per_chunk)
#   X_binned, y, weights = get_data_labels_weights_binned(chunk_of_forecasts, bin_splits)
#
#   trees =
#     MemoryConstrainedTreeBoosting.train_on_binned(
#       X_binned, y,
#       prior_trees             = trees,
#       weights                 = weights,
#       iteration_count         = 5,
#       min_data_weight_in_leaf = 30000.0,
#       l2_regularization       = 1.0,
#       max_leaves              = 4,
#       max_depth               = 4,
#       max_delta_score         = 5.0,
#       learning_rate           = learning_rate,
#       feature_fraction        = 0.2,
#       bagging_temperature     = 0.0,
#       feature_i_to_name       = SREF.feature_i_to_name,
#     )
# end

# for forecast in validation_forecasts[[5,10,15,30,40,50]]
#   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
#   X = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
#   y = TrainingShared.forecast_labels(grid, forecast)
#   ŷ = MemoryConstrainedTreeBoosting.predict(X, bin_splits, trees)
#   prefix = "$(model_prefix)/forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
#   Plots.png(Grib2.plot(grid, Float32.(ŷ)), "$(prefix)_predictions.png")
#   Plots.png(Grib2.plot(grid, y), "$(prefix)_labels.png")
#   println("done.")
# end
