import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import HRRR

forecast_hour = parse(Int64, ENV["FORECAST_HOUR"])
forecast_hour_range = forecast_hour:forecast_hour

# 2
# 6
# 12
# 17

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.04"))

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_f$(forecast_hour)_$(replace(repr(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    HRRR.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "hrrr_f$(forecast_hour)_$(data_subset_ratio)",

    training_X_and_labels_to_inclusion_probabilities   = (X, labels) -> max.(data_subset_ratio, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels) -> max.(data_subset_ratio, labels),

    bin_split_forecast_sample_count    = 200,
    max_iterations_without_improvement = 20,

    # Start with middle value for each parameter, plus some number of random choices, before beginning coordinate descent.
    random_start_count = 20,

    # Roughly factors of 1.78 (4 steps per power of 10)
    min_data_weight_in_leaf = [100.0, 180.0, 320.0, 560.0, 1000.0, 1800.0, 3200.0, 5600.0, 10000.0, 18000.0, 32000.0, 56000.0, 100000.0, 180000.0, 320000.0, 560000.0, 1000000.0, 1800000.0, 3200000.0, 5600000.0, 10000000.0],
    l2_regularization       = [3.2],
    max_leaves              = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35],
    max_depth               = [3, 4, 5, 6, 7, 8],
    max_delta_score         = [0.56, 1.0, 1.8, 3.2, 5.6],
    learning_rate           = [0.063], # [0.025, 0.040, 0.063, 0.1, 0.16], # factors of 1.585 (5 steps per power of 10)
    feature_fraction        = [0.1, 0.25, 0.5, 0.75, 1.0],
    bagging_temperature     = [0.25]
  )
