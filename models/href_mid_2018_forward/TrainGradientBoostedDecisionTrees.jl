import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import HREF


# HREF files come out 2-3 hours after run time

forecast_hour_range =
  if occursin(r"^\d+:\d+$", get(ENV, "FORECAST_HOUR_RANGE", ""))
    start, stop = map(str -> parse(Int64, str), split(ENV["FORECAST_HOUR_RANGE"], ":"))
    start:stop
  else
    2:35
  end

# 2:13
# 13:24
# 24:35

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.025"))

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(repr(Dates.now()), ":" => "."))"

# $ FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.025 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 11165 for training. (1970 with tornadoes.)
# 2282 for validation.
# 2125 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 6990 positive and 71820 negative labels...computing bin splits...done.
# Loading training data
# done. 10129095 datapoints with 17758 features each.
# Loading validation data
# done. 2074205 datapoints with 17758 features each.
# 
# Middle config:
# New best! Loss: 0.0010878555
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0010728862
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 3.2,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 180.0)
#
# Best after coordinate descent:
# Best hyperparameters (loss = 0.0010708775):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 3.2,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320.0)
# 102:45:03 elapsed

# $ FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.025 make train_gradient_boosted_decision_trees



TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "href_$(hour_range_str)_$(data_subset_ratio)",

    training_X_and_labels_to_inclusion_probabilities   = (X, labels) -> max.(data_subset_ratio, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels) -> max.(data_subset_ratio, labels),

    bin_split_forecast_sample_count    = 200,
    max_iterations_without_improvement = 20,

    # Start with middle value for each parameter, plus some number of random choices, before beginning coordinate descent.
    random_start_count = 20,

    # Roughly factors of 1.78 (4 steps per power of 10)
    min_data_weight_in_leaf     = [100.0, 180.0, 320.0, 560.0, 1000.0, 1800.0, 3200.0, 5600.0, 10000.0, 18000.0, 32000.0, 56000.0, 100000.0, 180000.0, 320000.0, 560000.0, 1000000.0, 1800000.0, 3200000.0, 5600000.0, 10000000.0],
    l2_regularization           = [3.2],
    max_leaves                  = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35],
    max_depth                   = [3, 4, 5, 6, 7, 8],
    max_delta_score             = [0.56, 1.0, 1.8, 3.2, 5.6],
    learning_rate               = [0.063], # [0.025, 0.040, 0.063, 0.1, 0.16], # factors of 1.585 (5 steps per power of 10)
    feature_fraction            = [0.1, 0.25, 0.5, 0.75, 1.0],
    bagging_temperature         = [0.25]
  )
