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

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.012"))
near_storm_ratio  = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.2"))
load_only         = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

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
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 11141 for training. (1967 with tornadoes.)
# 2268 for validation.
# 2137 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 6550 positive and 71439 negative labels...computing bin splits...done.
# Loading training data
# done. 10105131 datapoints with 17758 features each.
# Loading validation data
# done. 2061642 datapoints with 17758 features each.
#
# Middle config:
# New best! Loss: 0.0011394698
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Remained the best config through random search and coordinate descent.
#
# Best hyperparameters (loss = 0.0011394698):
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
# 121:39:50 elapsed

# $ FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.025 make train_gradient_boosted_decision_trees
# ulimit -n 8192; JULIA_NUM_THREADS=16 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
# Loading tornadoes...
# Loading wind events...
# Loading hail events...
# 11121 for training. (1969 with tornadoes.)
# 2251 for validation.
# 2114 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 6673 positive and 71822 negative labels...computing bin splits...done.
# Loading training data
# done. 10083624 datapoints with 17758 features each.
# Loading validation data
# done. 2045972 datapoints with 17758 features each.
#
# Middle config:
# New best! Loss: 0.0011902072
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0011777475
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 180.0)
#
# Best after coordinate descent:
# New best! Loss: 0.0011718384
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 180.0)


# TAKE ONE MILLION, now with data through Oct 2020. But we'll remember the bin splits this time if we restart training after data loading.
# Also don't subset points so much within 100mi/90min of any storm event

# $ FORECAST_HOUR_RANGE=2:13 LOAD_ONLY=true DATA_SUBSET_RATIO=0.008 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # Est total size: 162gb
# $ FORECAST_HOUR_RANGE=2:13 LOAD_ONLY=true DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # Total size: 207gb
# 14719 for training. (2191 with tornadoes.)
# 3110 for validation.
# 2936 for testing.
# Loading training data
# done. 10259683 datapoints with 17758 features each.
# Loading validation data
# done. 2194771 datapoints with 17758 features each.
# 63:47:04 elapsed


# FORECAST_HOUR_RANGE=13:24 LOAD_ONLY=true DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 14710 for training. (2187 with tornadoes.)
# 3096 for validation.
# 2959 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 7430 positive and 71439 negative labels...computing bin splits...done.
# Loading training data




# TAKE ONE MILLION ONE! FORGOT THE BWD TERM IN THE pseudo-STP CALCULATION

# $ FORECAST_HOUR_RANGE=2:13 LOAD_ONLY=true DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees # Total size: 207gb
# 14719 for training. (2191 with tornadoes.)
# 3110 for validation.
# 2936 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 7262 positive and 71819 negative labels...computing bin splits...done.
# Loading training data
# done. 10259683 datapoints with 17758 features each.
# Loading validation data
# done. 2194771 datapoints with 17758 features each.
# 61:08:21 elapsed
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 14719 for training. (2191 with tornadoes.)
# 3110 for validation.
# 2936 for testing.
# Loading previously computed bin splits from href_f2-13_0.012_0.2_samples_for_bin_splits/bin_splits
# Loading training data
# done. 10259683 datapoints with 17758 features each.
# Loading validation data
# done. 2194771 datapoints with 17758 features each.
#
# Middle config:
# Best validation loss: 0.0010027429    17.78200787 sec/tree
# New best! Loss: 0.0010027429
#
# Best random:
# New best! Loss: 0.0009984318
# Dict{Symbol,Real}(:max_depth => 7,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 1800.0)
#
# Best after coordinate descent (no improvement):
# Best hyperparameters (loss = 0.0009984318):
# Dict{Symbol,Real}(:max_depth => 7,:max_delta_score => 0.56,:learning_rate => 0.063,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.1,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 1800.0)
#
# 46:11:57 elapsed

# sudo apt install sshfs
# Make sure ~/.ssh/config doesn't use screen
# mkdir ~/nadocaster2
# sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ FORECAST_HOUR_RANGE=13:24 LOAD_ONLY=true DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 14710 for training. (2187 with tornadoes.)
# 3096 for validation.
# 2959 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 7430 positive and 71439 negative labels...computing bin splits...done.
# done. 10250212 datapoints with 17758 features each.
# Loading validation data
# done. 2186551 datapoints with 17758 features each.
# 59:45:18 elapsed
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 14710 for training. (2187 with tornadoes.)
# 3096 for validation.
# 2959 for testing.
# Loading previously computed bin splits from href_f13-24_0.012_0.2_samples_for_bin_splits/bin_splits
# Loading training data
# done. 10250212 datapoints with 17758 features each.
# Loading validation data
# done. 2186551 datapoints with 17758 features each.
#
# Middle config:
# New best! Loss: 0.0010637761
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)



# $ FORECAST_HOUR_RANGE=24:35 LOAD_ONLY=true DATA_SUBSET_RATIO=0.012 NEAR_STORM_RATIO=0.2 make train_gradient_boosted_decision_trees
# 14712 for training. (2194 with tornadoes.)
# 3079 for validation.
# 2936 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 7256 positive and 71445 negative labels...computing bin splits...done.
# Loading training data
# done. 10245701 datapoints with 17758 features each.
# Loading validation data
# done. 2177360 datapoints with 17758 features each.
# 66:00:22 elapsed
# $


TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "href_$(hour_range_str)_$(data_subset_ratio)_$(near_storm_ratio)",

    training_X_and_labels_to_inclusion_probabilities   = (X, labels, is_near_storm_event) -> max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels, is_near_storm_event) -> max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event, labels),
    load_only                                          = load_only,

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
