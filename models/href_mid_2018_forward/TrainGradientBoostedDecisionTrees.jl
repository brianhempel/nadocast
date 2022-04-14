import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
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

event_types       = split(get(ENV, "EVENT_TYPE", ""), ",")
event_types       = event_types == [] ? nothing : event_types
data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.026"))
near_storm_ratio  = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.4"))
load_only         = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix        = model_prefix,
    save_dir            = "href_$(hour_range_str)_$(data_subset_ratio)_$(near_storm_ratio)",
    event_types         = event_types,
    load_only           = load_only,

    data_subset_ratio = data_subset_ratio,
    near_storm_ratio  = near_storm_ratio,

    bin_split_forecast_sample_count    = 300, # will be divided among the label types
    max_iterations_without_improvement = 20,

    # Start with middle value for each parameter, plus some number of random choices, before beginning coordinate descent.
    random_start_count = 20,
    max_hyperparameter_coordinate_descent_iterations = 2,

    # Roughly factors of 1.78 (4 steps per power of 10)
    min_data_weight_in_leaf     = [100.0, 180.0, 320.0, 560.0, 1000.0, 1800.0, 3200.0, 5600.0, 10000.0, 18000.0, 32000.0, 56000.0, 100000.0, 180000.0, 320000.0, 560000.0, 1000000.0, 1800000.0, 3200000.0, 5600000.0, 10000000.0],
    l2_regularization           = [3.2],
    max_leaves                  = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35],
    max_depth                   = [3, 4, 5, 6, 7, 8],
    max_delta_score             = [0.56, 1.0, 1.8, 3.2, 5.6],
    learning_rate               = [0.063], # [0.025, 0.040, 0.063, 0.1, 0.16], # factors of 1.585 (5 steps per power of 10)
    feature_fraction            = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    bagging_temperature         = [0.25]
  )


# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees

# $ FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees

# $ FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees