import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import SREF


forecast_hour_range = 2:38 # 1:87 # 4:39             # SREF files come out 3-4 hours after run time

model_prefix = "gbdt_3hr_window_f$(forecast_hour_range.start)-$(forecast_hour_range.stop)_$(replace(string(Dates.now()), ":" => "."))"


# validation loss via prior predictor: 0.0019582207
# 2019-05-24 21Z +33: 115.09951 (2019-05-26 0600)
# 2019-05-25 09Z +21: 113.697235 (2019-05-26 0600)
# 2019-05-25 03Z +27: 113.535645 (2019-05-26 0600)
# 2019-05-25 15Z +15: 97.07941 (2019-05-26 0600)
# 2019-05-24 15Z +37: 94.32985 (2019-05-26 0400)
# 2019-05-25 09Z +19: 93.44213 (2019-05-26 0400)
# 2019-05-24 21Z +31: 92.04001 (2019-05-26 0400)
# 2019-05-25 21Z +9: 88.49613 (2019-05-26 0600)
# 2019-05-25 03Z +25: 86.8154 (2019-05-26 0400)
# 2019-05-26 03Z +3: 76.79073 (2019-05-26 0600)

# Train with tornadoes up to 2019-8-21
# Training:   10985931 datapoints with 6555 features each = 72,012,777,705 bytes
# Validation:



# Annoyingly, because of world age issues, have to do this at the top level.
prior_predictor = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/gbdt_f1-39_2019-09-17T14.50.32.041/182_trees_loss_0.0016721075.model")


TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    SREF.three_hour_window_feature_engineered_forecasts();
    # SREF.three_hour_window_feature_engineered_forecasts_middle_hour_only(); # Debug
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,

    training_calc_inclusion_probabilities   = (labels, is_near_storm_event) -> max.(0.2f0, labels),
    validation_calc_inclusion_probabilities = (labels, is_near_storm_event) -> max.(0.2f0, labels),

    prior_predictor = prior_predictor, # To compare validation loss

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
