import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import SREF


forecast_hour_range = 2:38 # 1:87 # 4:39             # SREF files come out 3-4 hours after run time

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_f$(forecast_hour_range.start)-$(forecast_hour_range.stop)_$(replace(repr(Dates.now()), ":" => "."))"


# Annoyingly, because of world age issues, have to do this at the top level.
# prior_predictor = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/gbdt_f1-39_2019-03-26T00.59.57.772/78_trees_loss_0.001402743.model")


TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,

    training_X_and_labels_to_inclusion_probabilities   = (X, labels) -> max.(0.2f0, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels) -> max.(0.2f0, labels),

    # prior_predictor = prior_predictor, # To compare validation loss

    bin_split_forecast_sample_count    = 200,
    max_iterations_without_improvement = 20,

    min_data_weight_in_leaf = [10.0, 15.0, 20.0, 35.0, 50.0, 70.0, 100.0, 150.0, 200.0, 350.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3500.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 35000.0, 50000.0, 70000.0, 100000.0, 150000.0, 200000.0, 350000.0, 500000.0, 700000.0, 1000000.0, 1500000.0, 2000000.0, 3500000.0, 5000000.0, 7000000.0, 10000000.0],
    l2_regularization       = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    max_leaves              = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35],
    max_depth               = [2, 3, 4, 5, 6, 7, 8, 9],
    max_delta_score         = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 1000.0],
    learning_rate           = [0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02],
    feature_fraction        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    bagging_temperature     = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  )
