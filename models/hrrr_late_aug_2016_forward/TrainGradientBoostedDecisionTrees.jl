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

# Google archive only has up to f15 until 2016-8-25...and the Utah archive got rid of their 2016 data for some reason. So f17 is missing a couple months relative to the others.

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.0015"))
near_storm_ratio  = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.2"))
load_only         = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_f$(forecast_hour)_$(replace(string(Dates.now()), ":" => "."))"

# $ FORECAST_HOUR=12 DATA_SUBSET_RATIO=0.01 make train_gradient_boosted_decision_trees
# JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 10097 for training. (1868 with tornadoes.)
# 2065 for validation.
# 2020 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 20161 positive and 193071 negative labels...computing bin splits...done.
# Loading training data
# done. 9957559 datapoints with 18577 features each.
# Loading validation data
# done. 2039162 datapoints with 18577 features each.
#
# Middle config:
# New best! Loss: 0.0011770464
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0011733025
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 1.0,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320.0)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.0011668991):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 15,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 560.0)


# $ FORECAST_HOUR=2 DATA_SUBSET_RATIO=0.01 make train_gradient_boosted_decision_trees
# JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 10108 for training. (1873 with tornadoes.)
# 2066 for validation.
# 2019 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 19921 positive and 193093 negative labels...computing bin splits...done.
# Loading training data
# done. 9966102 datapoints with 18577 features each.
# Loading validation data
# done. 2039147 datapoints with 18577 features each.
#
# Middle config:
# New best! Loss: 0.0010063316
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0009928169
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 35,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 3200.0)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.0009847874):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.75,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 5600.0)



# Whoops, didn't have forecasts for ±1hr of all storm events. Fixed that.
#
# $ FORECAST_HOUR=6 DATA_SUBSET_RATIO=0.009 make train_gradient_boosted_decision_trees
# JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 11155 for training. (1798 with tornadoes.)
# 2273 for validation.
# 2206 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 18136 positive and 193094 negative labels...computing bin splits...done.
# Loading training data
# done. 9897675 datapoints with 18577 features each.
# Loading validation data
# done. 2018659 datapoints with 18577 features each.
# Trying Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
# Best validation loss: 0.0009561085    36.740812071 sec/tree
#
# Middle config:
# New best! Loss: 0.0009561085
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0009519324
# Dict{Symbol,Real}(:max_depth => 7,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 1.0,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 3.2e6)
#
# After coordinate descent:
# Best hyperparameters (loss = 0.0009482468):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.75,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 5.6e6)
#
# 289:52:10 elapsed

# $ FORECAST_HOUR=17 DATA_SUBSET_RATIO=0.009 make train_gradient_boosted_decision_trees
# JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 11160 for training. (1800 with tornadoes.)
# 2251 for validation.
# 2201 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 20591 positive and 192110 negative labels...computing bin splits...done.
# Loading training data
# done. 9905823 datapoints with 18577 features each.
# Loading validation data
# done. 1997355 datapoints with 18577 features each.
#
# Middle config:
# New best! Loss: 0.0010500954
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)



# TAKE ONE MILLION, now with data through Oct 2020. But we'll remember the bin splits this time if we restart training after data loading.
# Also don't subset points so much within 100mi/90min of any storm event

# $ FORECAST_HOUR=17 DATA_SUBSET_RATIO=0.009 make train_gradient_boosted_decision_trees # est training size: 1150gb
# $ FORECAST_HOUR=17 DATA_SUBSET_RATIO=0.001 make train_gradient_boosted_decision_trees # est training size: 650gb

# So 63,000gb is far away points and 600gb is near the storm
# $ FORECAST_HOUR=17 LOAD_ONLY=true DATA_SUBSET_RATIO=0.0015 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # Total size: 237gb
# 12782 for training. (1915 with tornadoes.)
# 2628 for validation.
# 2577 for testing.
# Loading previously computed bin splits from hrrr_f17_0.0015_0.2_samples_for_bin_splits/bin_splits
# Loading training data
# done. 11290932 datapoints with 18577 features each.
# Loading validation data
# done. 2335755 datapoints with 18577 features each.
# 121:37:24 elapsed
# Need 1 forecast for grid
# $ scp /Volumes/HRRR_1/hrrr/201608/20160824/hrrr_conus_sfc_20160824_t00z_f05.grib2 nadocaster:~/nadocast_dev/test_grib2s/HRRR_1/hrrr/201608/20160824/hrrr_conus_sfc_20160824_t00z_f05.grib2
# $ FORECASTS_ROOT=/home/brian/nadocast_dev/test_grib2s/ FORECAST_HOUR=17 DATA_SUBSET_RATIO=0.0015 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# Loading previously computed bin splits from hrrr_f17_0.0015_0.2_samples_for_bin_splits/bin_splits
# Loading training data
# done. 11290932 datapoints with 18577 features each.
# Loading validation data
# done. 2335755 datapoints with 18577 features each.
#
# Middle config:
# New best! Loss: 0.0010209968
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0010147324
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 18000.0)
#
# After coordinate descent (no effective change):
# Best hyperparameters (loss = 0.0010138382):
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 18000.0)
#
# 35:53:17 elapsed


# $ FORECAST_HOUR=12 LOAD_ONLY=true DATA_SUBSET_RATIO=0.0015 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 12783 for training. (1914 with tornadoes.)
# 2639 for validation.
# 2577 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 20306 positive and 193077 negative labels...computing bin splits...done.
# Loading training data
# done. 11296524 datapoints with 18577 features each.
# Loading validation data
# done. 2344934 datapoints with 18577 features each.
# 138:43:32 elapsed
# $


# $ FORECAST_HOUR=6 LOAD_ONLY=true DATA_SUBSET_RATIO=0.0015 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 12774 for training. (1913 with tornadoes.)
# 2651 for validation.
# 2581 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
#



TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    HRRR.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "hrrr_f$(forecast_hour)_$(data_subset_ratio)_$(near_storm_ratio)",

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
