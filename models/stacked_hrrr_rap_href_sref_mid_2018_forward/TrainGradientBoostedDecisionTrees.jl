import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import StackedHRRRRAPHREFSREF

# See the forecast_scheduling spreadsheet for Nadocast run and forecast hours (which match the latest HRRR used).

forecast_hour = parse(Int64, ENV["FORECAST_HOUR"]) # 1, 5, 11, or 16
forecast_hour_range = forecast_hour:forecast_hour

model_prefix = "gbdt_f$(forecast_hour)_$(replace(repr(Dates.now()), ":" => "."))"

stacked_forecasts = StackedHRRRRAPHREFSREF.forecasts()

# 616 for training. (118 with tornadoes.)
# 170 for validation.
# 118 for testing.
# looks like there were ~10 bad forecasts in training...

# filtering to balance 4127 positive and 4181733 negative labels...computing bin splits...done.

# Training   21867510 datapoints with 106 features each = 2,317,956,060 bytes
# Validation  6134450 datapoints with 106 features each =   650,251,700 bytes

# Best hyperparameters (loss = 0.0012386695): Dict{Symbol,Real}(:max_depth=>4,:max_delta_score=>3.0,:learning_rate=>0.005,:max_leaves=>8,:l2_regularization=>3.0,:feature_fraction=>0.7,:bagging_temperature=>0.25,:min_data_weight_in_leaf=>1500.0)

# 139:20:35 elapsed (~5.5 days)


TrainGBDTShared.train_multiple_annealing_rounds_with_coordinate_descent_hyperparameter_search(
    stacked_forecasts;
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    get_feature_engineered_data = StackedHRRRRAPHREFSREF.get_feature_engineered_data,

    annealing_rounds = 3,
    basal_inclusion_probability      = 1f0,
    prediction_inclusion_multiplier  = 1f0,
    validation_inclusion_probability = 1f0,

    bin_split_forecast_sample_count = 400,
    balance_labels_when_computing_bin_splits = true,
    max_iterations_without_improvement = 40,

    min_data_weight_in_leaf = [10.0, 15.0, 20.0, 35.0, 50.0, 70.0, 100.0, 150.0, 200.0, 350.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3500.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 35000.0, 50000.0, 70000.0, 100000.0, 150000.0, 200000.0],
    l2_regularization       = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    max_leaves              = [3, 4, 5, 6, 8, 10, 12, 15, 20],
    max_depth               = [2, 3, 4, 5, 6, 7],
    max_delta_score         = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 1000.0],
    learning_rate           = [0.02, 0.015, 0.01, 0.007, 0.005, 0.0035, 0.002, 0.0015, 0.001],
    feature_fraction        = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    bagging_temperature     = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  )
