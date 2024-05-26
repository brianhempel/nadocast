import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import TrainGBDTShared

must_load_from_disk = parse(Bool, get(ENV, "MUST_LOAD_FROM_DISK", "false"))

if !must_load_from_disk
  push!(LOAD_PATH, @__DIR__)
  import HREF
end


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

event_types        = split(get(ENV, "EVENT_TYPES", ""), ",")
event_types        = event_types == [""] ? nothing : event_types
data_subset_ratio  = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.05"))
near_storm_ratio   = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.5"))
all_hours          = parse(Bool, get(ENV, "ALL_HOURS", "true"))
load_only          = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))
distributed        = parse(Bool, get(ENV, "DISTRIBUTED", "false"))
only_features_path = get(ENV, "ONLY_FEATURES_PATH", "")
only_before        = Dates.DateTime(map(str -> parse(Int64, str), split(get(ENV, "ONLY_BEFORE", "2099-1-1"), "-"))...) + Dates.Hour(12)

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_2024-2005_features_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    must_load_from_disk ? [] : HREF.feature_engineered_forecasts_with_climatology(); # no windows
    forecast_hour_range = forecast_hour_range,
    model_prefix        = model_prefix,
    save_dir            = "href_2024-2005_features_$(hour_range_str)_$(data_subset_ratio)_$(near_storm_ratio)",
    event_types         = event_types,
    load_only           = load_only,
    must_load_from_disk = must_load_from_disk,
    use_mpi             = distributed,

    data_subset_ratio = data_subset_ratio,
    near_storm_ratio  = near_storm_ratio,
    just_hours_near_storm_events = !all_hours,

    only_features     = (only_features_path !=  "" ? readlines(only_features_path) : nothing), # can vary this without reloading the data
    only_before       = only_before,  # can vary this without reloading the data

    bin_split_forecast_sample_count    = 400, # will be divided among the label types
    max_iterations_without_improvement = 30,

    # Start with middle value for each parameter, plus some number of random choices, before beginning coordinate descent.
    random_start_count = 30,
    max_hyperparameter_coordinate_descent_iterations = 2,

    min_data_weight_in_leaf  = [100., 215., 464., 1000., 2150., 4640., 10000., 21500., 46400., 100000., 215000.],
    l2_regularization        = [0.001, 0.01, 0.1, 1.0, 10., 100., 1000., 10000, 100000.],
    max_leaves               = [6, 8, 10, 12, 15, 20, 25, 30, 35],
    max_depth                = [4, 5, 6, 7, 8],
    max_delta_score          = [0.56, 1.0, 1.8, 3.2],
    learning_rate            = [0.04],
    feature_fraction         = [0.018, 0.032, 0.056, 0.1, 0.18, 0.32, 0.5, 0.75, 1.0],
    second_opinion_weight    = [0.0, 0.01, 0.033, 0.1, 0.33, 1.0], # 0.0 = no second opinion. 1.0 = look at expected gains for sibling when choosing feature splits, choose feature that maximizes gains for both siblings. Inspired by Catboost choosing same split and feature across an entire level, so-called "oblivious" decision trees. But we are not going so far as to choose the same split point.
    normalize_second_opinion = [false, true], # true = make the best expected gain on the sibling match the leaf's best expected gain before applying the second opinion weight (in case of highly imbalanced nodes, this makes the leaf with more data count less)
    bagging_temperature      = [0.25]
  )


# $ FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.05 NEAR_STORM_RATIO=0.5 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_2:13_2024-2005.log
