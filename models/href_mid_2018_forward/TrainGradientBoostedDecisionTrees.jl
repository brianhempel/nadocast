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
data_subset_ratio  = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.03"))
near_storm_ratio   = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.4"))
load_only          = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))
distributed        = parse(Bool, get(ENV, "DISTRIBUTED", "false"))
only_features_path = get(ENV, "ONLY_FEATURES_PATH", "")
only_before        = Dates.DateTime(map(str -> parse(Int64, str), split(get(ENV, "ONLY_BEFORE", "2099-1-1"), "-"))...) + Dates.Hour(12)

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    must_load_from_disk ? [] : HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix        = model_prefix,
    save_dir            = "href_$(hour_range_str)_$(data_subset_ratio)_$(near_storm_ratio)",
    event_types         = event_types,
    load_only           = load_only,
    must_load_from_disk = must_load_from_disk,
    use_mpi             = distributed,

    data_subset_ratio = data_subset_ratio,
    near_storm_ratio  = near_storm_ratio,

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


# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees
# 23847 for training,     from 2018-06-29 00Z +2  to      2022-06-01 06Z +2.
#   (2853 with sig_wind,  from 2018-06-29 00Z +2  to      2022-06-01 00Z +5)
#   (1840 with sig_hail,  from 2018-06-29 00Z +2  to      2022-06-01 00Z +3)
#   (8101 with hail,      from 2018-06-29 00Z +2  to      2022-06-01 00Z +6)
#   (2853 with sig_wind_adj,      from 2018-06-29 00Z +2  to      2022-06-01 00Z +5)
#   (3611 with tornado,   from 2018-06-29 00Z +2  to      2022-06-01 00Z +4)
#   (12279 with wind_adj,         from 2018-06-29 00Z +2  to      2022-06-01 00Z +5)
#   (598 with sig_tornado,        from 2018-06-29 00Z +2  to      2022-05-30 18Z +4)
#   (12279 with wind,     from 2018-06-29 00Z +2  to      2022-06-01 00Z +5)
# 4824 for validation,    from 2018-06-30 00Z +13 to      2022-05-29 06Z +5.
# 4550 for testing,       from 2018-07-01 06Z +9  to      2022-05-30 06Z +5.
# Preparing bin splits by sampling 400 training forecasts with events
# sampling 24490 datapoints...computing bin splits...done.
# Loading training data



# $ USE_ALT_DISK=true LOAD_ONLY=true FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees
# 23806 for training,     from 2018-06-29 00Z +13 to      2022-05-31 18Z +14.
#   (2839 with sig_wind,  from 2018-06-29 00Z +13 to      2022-05-31 12Z +17)
#   (1832 with sig_hail,  from 2018-06-29 00Z +22 to      2022-05-31 12Z +15)
#   (8071 with hail,      from 2018-06-29 00Z +20 to      2022-05-31 12Z +18)
#   (2839 with sig_wind_adj,      from 2018-06-29 00Z +13 to      2022-05-31 12Z +17)
#   (3607 with tornado,   from 2018-06-29 00Z +15 to      2022-05-31 12Z +16)
#   (12251 with wind_adj,         from 2018-06-29 00Z +13 to      2022-05-31 12Z +17)
#   (594 with sig_tornado,        from 2018-07-03 12Z +19 to      2022-05-30 06Z +16)
#   (12251 with wind,     from 2018-06-29 00Z +13 to      2022-05-31 12Z +17)
# 4808 for validation,    from 2018-06-29 18Z +19 to      2022-05-28 18Z +17.
# 4568 for testing,       from 2018-06-30 18Z +21 to      2022-05-29 18Z +17.
# Preparing bin splits by sampling 400 training forecasts with events
# sampling 26565 datapoints...computing bin splits...done.
# Loading training data



# $ sshfs brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ cd ~/hd
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ USE_ALT_DISK=true LOAD_ONLY=true FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 JULIA_NUM_THREADS=$CORE_COUNT time julia --project=~/nadocast_dev ~/nadocast_dev/models/href_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
# 23778 for training,     from 2018-06-29 00Z +24 to      2022-05-31 06Z +26.
#   (2833 with sig_wind,  from 2018-06-29 00Z +25 to      2022-05-31 00Z +29)
#   (1823 with sig_hail,  from 2018-06-29 00Z +24 to      2022-05-31 00Z +27)
#   (8048 with hail,      from 2018-06-29 00Z +24 to      2022-05-31 06Z +24)
#   (2833 with sig_wind_adj,      from 2018-06-29 00Z +25 to      2022-05-31 00Z +29)
#   (3612 with tornado,   from 2018-07-03 00Z +31 to      2022-05-31 00Z +28)
#   (12229 with wind_adj,         from 2018-06-29 00Z +24 to      2022-05-31 00Z +29)
#   (596 with sig_tornado,        from 2018-07-03 00Z +31 to      2022-05-29 18Z +28)
#   (12229 with wind,     from 2018-06-29 00Z +24 to      2022-05-31 00Z +29)
# 4789 for validation,    from 2018-06-29 06Z +31 to      2022-05-28 06Z +29.
# 4528 for testing,       from 2018-06-30 06Z +33 to      2022-05-29 06Z +29.
# Preparing bin splits by sampling 400 training forecasts with events
# computing radius ranges...done
# sampling 21531 datapoints...computing bin splits...done.
# Loading training data




# on nadocaster2:
# mkdir ~/hd
# sshfs brian@nadocaster:/Volumes/hd/ ~/hd/

# on nadocaster:
# ln -s /Volumes/hd ~/hd

# on either:
# cd ~/hd
# JULIA_MPI_BINARY=system DISTRIBUTED=true JULIA_NUM_THREADS=$CORE_COUNT EVENT_TYPES=tornado,wind,wind_adj,hail,sig_tornado,sig_wind,sig_wind_adj,sig_hail MUST_LOAD_FROM_DISK=true FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 time mpirun -n 2 -wdir $(pwd) -hosts 192.168.1.112:1,192.168.1.121:1 -bind-to none julia --compiled-modules=no --project=/home/brian/nadocast_dev /home/brian/nadocast_dev/models/href_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl

