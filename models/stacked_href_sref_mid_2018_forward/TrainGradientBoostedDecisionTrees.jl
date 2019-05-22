import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainGBDTShared

push!(LOAD_PATH, @__DIR__)
import StackedHREFSREF

# See the forecast_scheduling spreadsheet for Nadocast run and forecast hours (which match the latest HRRR used).
#
# First forecast hour where we might use HREF/SREF only is +17 (the first hour without 3 time lagged HRRRs)
# We can go out to +30 or +34 (depending on how old the HREF is).
#
# Make sure the size of the training range is a multiple of 6 so that
# we see each event the same number of times.
#
# Be conservative: choose latest range possible.

if ENV["HREF_NEWER_THAN_SREF"] == "true"
  forecast_hour_range = 23:34
  model_prefix = "gbdt_href_newer_nadocastf$(forecast_hour_range.start)-$(forecast_hour_range.stop)_$(replace(repr(Dates.now()), ":" => "."))"
  stacked_href_sref_prediction_forecasts = StackedHREFSREF.forecasts_with_href_newer_than_sref()
else
  forecast_hour_range = 19:30
  model_prefix = "gbdt_sref_newer_nadocastf$(forecast_hour_range.start)-$(forecast_hour_range.stop)_$(replace(repr(Dates.now()), ":" => "."))"
  stacked_href_sref_prediction_forecasts = StackedHREFSREF.forecasts_with_sref_newer_than_href()
end


TrainGBDTShared.train_multiple_annealing_rounds_with_coordinate_descent_hyperparameter_search(
    stacked_href_sref_prediction_forecasts;
    forecast_hour_range = forecast_hour_range,
    model_prefix = model_prefix,
    get_feature_engineered_data = StackedHREFSREF.get_feature_engineered_data,

    annealing_rounds = 3,
    basal_inclusion_probability      = 1f0,
    prediction_inclusion_multiplier  = 1f0,
    validation_inclusion_probability = 1f0,

    bin_split_forecast_sample_count = 400,
    balance_labels_when_computing_bin_splits = true,
    max_iterations_without_improvement = 40,

    min_data_weight_in_leaf = [10.0, 15.0, 20.0, 35.0, 50.0, 70.0, 100.0, 150.0, 200.0, 350.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3500.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 35000.0, 50000.0, 70000.0, 100000.0, 150000.0, 200000.0, 350000.0, 500000.0, 700000.0, 1000000.0, 1500000.0, 2000000.0, 3500000.0, 5000000.0, 7000000.0, 10000000.0],
    l2_regularization       = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    max_leaves              = [3, 4, 5, 6, 8, 10, 12, 15, 20],
    max_depth               = [2, 3, 4, 5, 6, 7],
    max_delta_score         = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 1000.0],
    learning_rate           = [0.02, 0.015, 0.01, 0.007, 0.005, 0.0035, 0.002, 0.0015, 0.001],
    feature_fraction        = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    bagging_temperature     = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  )
