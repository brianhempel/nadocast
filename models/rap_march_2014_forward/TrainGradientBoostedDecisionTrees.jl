import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import RAP

forecast_hour = parse(Int64, ENV["FORECAST_HOUR"])
forecast_hour_range = forecast_hour:forecast_hour

# 2
# 6
# 12
# 17

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.003"))
near_storm_ratio  = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.2"))
load_only         = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_f$(forecast_hour)_$(replace(string(Dates.now()), ":" => "."))"

# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.007 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 18782 for training. (2847 with tornadoes.)
# 3714 for validation.
# 3696 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 9334 positive and 90682 negative labels...computing bin splits...done.
# Loading training data
# done. 6129929 datapoints with 27222 features each.
# Loading validation data
# done. 1209459 datapoints with 27222 features each.
#
# Middle config:
# New best! Loss: 0.000843959
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0008319686
# Dict{Symbol,Real}(:max_depth => 7,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 0.75,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 18000.0)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.0008232763):
# Dict{Symbol,Real}(:max_depth => 7,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 0.75,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 5600.0)
#
# 321:26:32 elapsed


# $ FORECAST_HOUR=6 DATA_SUBSET_RATIO=0.007 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 18797 for training. (2842 with tornadoes.)
# 3705 for validation.
# 3701 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 8888 positive and 90694 negative labels...computing bin splits...done.
# Loading training data
# done. 6134462 datapoints with 27222 features each.
# Loading validation data
# done. 1205803 datapoints with 27222 features each.
#
# Middle config:
# New best! Loss: 0.0009312902
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0009229315
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 0.25,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 5.6e6)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.00091889914):
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 20,:l2_regularization => 3.2,:feature_fraction => 0.25,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 1.0e6)
#
# 241:44:55 elapsed


# $ FORECAST_HOUR=12 DATA_SUBSET_RATIO=0.007 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 18805 for training. (2851 with tornadoes.)
# 3691 for validation.
# 3695 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 9420 positive and 90681 negative labels...computing bin splits...done.
# Loading training data
# done. 6137395 datapoints with 27222 features each.
# Loading validation data
# done. 1202014 datapoints with 27222 features each.
#
# Middle config:
# New best! Loss: 0.0009834779
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0009807853
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320000.0)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.0009746317):
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 15,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 560000.0)
#
# 257:25:54 elapsed


# "RAP forecast 2018-06-05 00Z +18 does not have HGT:1000 mb:hour fcst::"
# Weird. labeled as "anl". Just removed it.
# "RAP forecast 2018-12-13 15Z +16 does not have HGT:950 mb:hour fcst::"
# Only has selected height levels. Removed it.

# $ FORECAST_HOUR=17 DATA_SUBSET_RATIO=0.007 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 18816 for training. (2855 with tornadoes.)
# 3693 for validation.
# 3686 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 8698 positive and 90688 negative labels...computing bin splits...done.
# Loading training data
# done. 6141096 datapoints with 27222 features each.
# Loading validation data
# done. 1202767 datapoints with 27222 features each.

# Middle config:
# New best! Loss: 0.0010086043
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)

# Best random:
# New best! Loss: 0.0009980386
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 35,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)

# Best after coordinate descent:
# Best hyperparameters (loss = 0.0009955188):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 35,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 18000.0)

# 66:07:37 elapsed (all data was loaded in failed runs)



# TAKE ONE MILLION, now with data through Oct 2020. But we'll remember the bin splits this time if we restart training after data loading.
# Also don't subset points so much within 100mi/90min of any storm event

# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.003 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # est size 370gb
# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.0015 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # est size 270gb
# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.0008 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # est size 270gb
# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.001 NEAR_STORM_RATIO=0.15 make train_gradient_boosted_decision_trees
# 20237 for training. (2963 with tornadoes.)
# 4070 for validation.
# 4060 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# Loading previously computed bin splits from rap_f2_0.001_0.15_samples_for_bin_splits/bin_splits
# Loading training data
# NOPE it crashed the training process because this process uses more memory over time, it seems.
# $ FORECAST_HOUR=2 LOAD_ONLY=true DATA_SUBSET_RATIO=0.001 NEAR_STORM_RATIO=0.15 make train_gradient_boosted_decision_trees
# 20237 for training. (2963 with tornadoes.)
# 4070 for validation.
# 4060 for testing.
# Loading previously computed bin splits from rap_f2_0.001_0.15_samples_for_bin_splits/bin_splits
# Loading training data
# done. 6310565 datapoints with 27222 features each.
# Loading validation data
# done. 1218579 datapoints with 27222 features each.
# 168:50:20 elapsed


TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    RAP.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "rap_f$(forecast_hour)_$(data_subset_ratio)_$(near_storm_ratio)",

    training_X_and_labels_to_inclusion_probabilities   = (X, labels, is_near_storm_event) -> max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels, is_near_storm_event) -> max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event, labels),
    load_only                                          = load_only,

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
