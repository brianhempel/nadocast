import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import RAP

forecast_hour = parse(Int64, ENV["FORECAST_HOUR"])
forecast_hour_range = forecast_hour:forecast_hour

model_prefix = "gbdt_f$(forecast_hour)_$(replace(repr(Dates.now()), ":" => "."))"

rap_forecasts = RAP.forecasts()
rap_forecasts = rap_forecasts[1:200:length(rap_forecasts)] # Subset the data



# Best so far Dict{Symbol,Real}(:max_depth=>5,:max_delta_score=>3.0,:learning_rate=>0.05,:max_leaves=>10,:l2_regularization=>3.0,:feature_fraction=>0.6,:bagging_temperature=>0.25,:min_data_weight_in_leaf=>10000.0)

TrainGBDTShared.train_multiple_annealing_rounds_with_coordinate_descent_hyperparameter_search(
    rap_forecasts;
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    get_feature_engineered_data = RAP.get_feature_engineered_data,

    annealing_rounds = 3,
    basal_inclusion_probability = 0.01f0,
    prediction_inclusion_multiplier = 1000.0f0,
    validation_inclusion_probability = 0.10f0,

    bin_split_forecast_sample_count = 5,
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
