import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import SREF


# SREF files come out 3-4 hours after run time

forecast_hour_range =
  if occursin(r"^\d+:\d+$", get(ENV, "FORECAST_HOUR_RANGE", ""))
    start, stop = map(str -> parse(Int64, str), split(ENV["FORECAST_HOUR_RANGE"], ":"))
    start:stop
  else
    2:38 # 1:87 # 4:39
  end

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.2"))

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(repr(Dates.now()), ":" => "."))"

# $ FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 6828 for training. (1150 with tornadoes.)
# 1436 for validation.
# 1304 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1106 positive and 10015 negative labels...computing bin splits...done.
# Loading training data
# done. 6925825 datapoints with 18759 features each.
#
# 64:23:23 elapsed Best hyperparameters (loss = 0.001277702): Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 8,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)

# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 6828 for training. (1150 with tornadoes.)
# 1436 for validation.
# 1304 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1129 positive and 10004 negative labels...computing bin splits...done.
# Loading training data
# done. 6925826 datapoints with 18759 features each.
# Loading validation data
# done. 1456007 datapoints with 18759 features each.
#
# 31:14:05 elapsed Best hyperparameters (loss = 0.0013720567): Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 18000.0)

# $ FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.15 make train_gradient_boosted_decision_trees
#
# 10242 for training. (1725 with tornadoes.)
# 2154 for validation.
# 1956 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1097 positive and 10012 negative labels...computing bin splits...done.
# Loading training data
# done. 7794745 datapoints with 18759 features each.
# Loading validation data
# done. 1638326 datapoints with 18759 features each.
#
# 28:33:08 elapsed Best hyperparameters (loss = 0.0013992075): Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 56000.0)

# Take 2, trying to see if tiling makes training faster. Seems to. Also accidently got a slightly better model.
#
# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 6828 for training. (1150 with tornadoes.)
# 1436 for validation.
# 1304 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1129 positive and 10004 negative labels...computing bin splits...done.
# Loading training data
# done. 6925826 datapoints with 18759 features each.
# Loading validation data
# done. 1456007 datapoints with 18759 features each.
#
# 21:47:20 elapsed Best hyperparameters (loss = 0.0013716344): Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)

# FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
# 6828 for training. (1150 with tornadoes.)
# 1436 for validation.
# 1304 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1129 positive and 10004 negative labels...computing bin splits...done.
# Loading training data
# done. 6925826 datapoints with 18759 features each.
# Loading validation data
# done. 1456007 datapoints with 18759 features each.
# Middle config:
# Trying Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)
# New best! Loss: 0.0013713938
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)
# ...
# Best after random: Loss: 0.0013599272 Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.32,:learning_rate => 0.1,:max_leaves => 30,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 1.0,:min_data_weight_in_leaf => 1.0e6)
#
# After coordinate descent from there:
# 123:55:50 elapsed Best hyperparameters (loss = 0.0013549898): Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.1,:max_leaves => 30,:l2_regularization => 1.8,:feature_fraction => 0.5,:bagging_temperature => 1.0,:min_data_weight_in_leaf => 1.8e6)

# min_gain_to_split, doesn't help
# but exploring more random options first did
#
# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 6828 for training. (1150 with tornadoes.)
# 1436 for validation.
# 1304 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1129 positive and 10004 negative labels...computing bin splits...done.
# Loading training data
# done. 6925826 datapoints with 18759 features each.
# Loading validation data
# done. 1456007 datapoints with 18759 features each.
# Middle config:
# Trying Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:min_gain_to_split => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)
# New best! Loss: 0.0013759497
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 5.6,:feature_fraction => 0.5,:min_gain_to_split => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)
#
# Best Random:
# New best! Loss: 0.0013581899
# Dict{Symbol,Real}(:max_depth => 9,:max_delta_score => 1.0,:learning_rate => 0.025,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 1.0,:min_gain_to_split => 0.03,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 560000.0)
#
# had to stop during coordinate descent, taking too long
#
# New best! Loss: 0.0013524442
# Dict{Symbol,Real}(:max_depth => 9,:max_delta_score => 1.0,:learning_rate => 0.016,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 1.0,:min_gain_to_split => 0.0,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 1.8e6)
# 315:09:24 elapsed

# Trying with regression in leaves. Doesn't seem to help.
#
# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# Loading training data
# done. 6925826 datapoints with 18759 features each.
# Loading validation data
# done. 1456007 datapoints with 18759 features each.
# Middle config:
# New best! Loss: 0.0013605118
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 10000,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 10000.0)
#
# Best random:
# New best! Loss: 0.0013546386
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.0,:learning_rate => 0.025,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 0.1,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 180000.0)
#
# Best hyperparameters (loss = 0.0013517824):
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.0,:learning_rate => 0.025,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 0.1,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320000.0)

# Okay a bit more data now (Mar Apr 2020)
#
# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 8159 for training. (1568 with tornadoes.)
# 1658 for validation.
# 1476 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# computing radius indices...done
# filtering to balance 1062 positive and 10008 negative labels...computing bin splits...done.
# Loading training data
# done. 8276969 datapoints with 18759 features each.
# Loading validation data
# done. 1681587 datapoints with 18759 features each.
#
# Middle config:
# New best! Loss: 0.0013790311
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 10000,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 100000.0)
#
# Power flickered and machine crashed, but above was still the best config near the end of the random exploration.

# Changed hyperparameter ranges a little.
#
# $ FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
#
# 8159 for training. (1568 with tornadoes.)
# 1658 for validation.
# 1476 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1062 positive and 10008 negative labels...computing bin splits...done.
# Loading training data
# done. 8276969 datapoints with 18759 features each.
# Loading validation data
# done. 1681587 datapoints with 18759 features each.
#
# Middle Config:
# New best! Loss: 0.0013916317
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 10000,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best Random:
# New best! Loss: 0.0013810864
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 560.0)
#
# Best hyperparameters (loss = 0.0013711528):
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 15,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320.0)
# 142:19:06 elapsed

# $ FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.2 make train_gradient_boosted_decision_trees
# 8173 for training. (1568 with tornadoes.)
# 1658 for validation.
# 1476 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 1032 positive and 10009 negative labels...computing bin splits...done.
# Loading training data
# done. 8291143 datapoints with 18759 features each.
# Loading validation data
# done. 1681584 datapoints with 18759 features each.
#
# Middle config:
# New best! Loss: 0.0013057644
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:min_data_to_regress_in_leaf => 10000,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.0012978732
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 0.56,:learning_rate => 0.1,:max_leaves => 12,:l2_regularization => 3.2,:feature_fraction => 1.0,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320000.0)
#
# Best after coordinate descent (4th round):
# Best hyperparameters (loss = 0.0012903158):
# Dict{Symbol,Real}(:max_depth => 8,:max_delta_score => 1.0,:learning_rate => 0.063,:max_leaves => 15,:l2_regularization => 3.2,:feature_fraction => 0.75,:min_data_to_regress_in_leaf => 9223372036854775807,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320000.0)
# 369:44:50 elapsed

# $ FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.15 make train_gradient_boosted_decision_trees
# 12212 for training. (2344 with tornadoes.)
# 2487 for validation.
# 2221 for testing.
# Preparing bin splits by sampling 200 training tornado hour forecasts
# filtering to balance 956 positive and 10015 negative labels...computing bin splits...done.
# Loading training data
# done. 9295159 datapoints with 18759 features each.
# Loading validation data
# done. 1892074 datapoints with 18759 features each.
#
# Middle config:
# New best! Loss: 0.0014386413
# Dict{Symbol,Real}(:max_depth => 5,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 10,:l2_regularization => 3.2,:feature_fraction => 0.5,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 32000.0)
#
# Best random:
# New best! Loss: 0.00143153
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 5.6,:learning_rate => 0.063,:max_leaves => 35,:l2_regularization => 3.2,:feature_fraction => 0.75,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 180.0)
#
# Best after coordinate descent (4th round):
# Best hyperparameters (loss = 0.00141136):
# Dict{Symbol,Real}(:max_depth => 6,:max_delta_score => 1.8,:learning_rate => 0.063,:max_leaves => 25,:l2_regularization => 3.2,:feature_fraction => 1.0,:bagging_temperature => 0.25,:min_data_weight_in_leaf => 320.0)
# 186:47:56 elapsed


TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    save_dir     = "sref_$(hour_range_str)_$(data_subset_ratio)",

    training_X_and_labels_to_inclusion_probabilities   = (X, labels) -> max.(data_subset_ratio, labels),
    validation_X_and_labels_to_inclusion_probabilities = (X, labels) -> max.(data_subset_ratio, labels),

    bin_split_forecast_sample_count    = 200,
    max_iterations_without_improvement = 20,

    # Start with middle value for each parameter, plus some number of random choices, before beginning coordinate descent.
    random_start_count = 20,

    # Roughly factors of 1.78 (4 steps per power of 10)
    # min_data_weight_in_leaf     = [100.0, 180.0, 320.0, 560.0, 1000.0, 1800.0, 3200.0, 5600.0, 10000.0, 18000.0, 32000.0, 56000.0, 100000.0, 180000.0, 320000.0, 560000.0, 1000000.0, 1800000.0, 3200000.0, 5600000.0, 10000000.0],
    # l2_regularization           = [3.2],
    # max_leaves                  = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35],
    # max_depth                   = [3, 4, 5, 6, 7, 8],
    # max_delta_score             = [0.56, 1.0, 1.8, 3.2, 5.6],
    # learning_rate               = [0.063], # [0.025, 0.040, 0.063, 0.1, 0.16], # factors of 1.585 (5 steps per power of 10)
    # feature_fraction            = [0.1, 0.25, 0.5, 0.75, 1.0],
    # bagging_temperature         = [0.25]
    min_data_weight_in_leaf     = [320.0],
    l2_regularization           = [0.0032, 0.0101, 0.032, 0.101, 0.32, 1.01, 3.2, 10.1, 32, 101.0, 320.0, 1010.0, 3200.0],
    max_leaves                  = [25],
    max_depth                   = [6],
    max_delta_score             = [1.8],
    learning_rate               = [0.063], # [0.025, 0.040, 0.063, 0.1, 0.16], # factors of 1.585 (5 steps per power of 10)
    feature_fraction            = [1.0],
    bagging_temperature         = [0.25]
  )
