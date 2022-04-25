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
data_subset_ratio  = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.026"))
near_storm_ratio   = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.4"))
load_only          = parse(Bool,    get(ENV, "LOAD_ONLY", "false"))
distributed = parse(Bool, get(ENV, "DISTRIBUTED", "false"))
climatology_amount = get(ENV, "CLIMATOLOGY", "some") # options: none, minimal, some, all

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

    climatology_amount = climatology_amount, # can vary this without reloading the data

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
    feature_fraction            = [0.1, 0.25, 0.5, 0.75, 1.0],
    # feature_fraction            = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0], this is how we actually trained
    bagging_temperature         = [0.25]
  )


# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees
# 18752 for training,     from 2018-06-29 00Z +2  to      2021-12-31 18Z +6.
#   (1598 with sig_hail,  from 2018-06-29 00Z +2  to      2021-12-15 18Z +5)
#   (7205 with hail,      from 2018-06-29 00Z +2  to      2021-12-30 18Z +4)
#   (3107 with tornado,   from 2018-06-29 00Z +2  to      2021-12-31 18Z +5)
#   (480 with sig_tornado,        from 2018-06-29 00Z +2  to      2021-12-16 00Z +3)
#   (2547 with sig_wind,  from 2018-06-29 00Z +2  to      2021-12-30 18Z +3)
#   (11077 with wind,     from 2018-06-29 00Z +2  to      2021-12-30 18Z +4)
# 3818 for validation,    from 2018-06-30 06Z +8  to      2021-12-25 18Z +5.
# 3495 for testing,       from 2018-07-01 06Z +10 to      2021-12-06 06Z +5.
# Preparing bin splits by sampling 300 training forecasts with events
# sampling 26874 datapoints...computing bin splits...done.
# Loading training data



# $ sshfs brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees
# 18718 for training,     from 2018-06-29 00Z +13 to      2021-12-31 06Z +18.
#   (1590 with sig_hail,  from 2018-06-29 00Z +22 to      2021-12-15 06Z +17)
#   (7175 with hail,      from 2018-06-29 00Z +20 to      2021-12-30 06Z +16)
#   (3103 with tornado,   from 2018-06-29 00Z +15 to      2021-12-31 06Z +17)
#   (476 with sig_tornado,        from 2018-07-03 12Z +19 to      2021-12-15 12Z +15)
#   (2533 with sig_wind,  from 2018-06-29 00Z +13 to      2021-12-30 06Z +15)
#   (11050 with wind,     from 2018-06-29 00Z +13 to      2021-12-30 06Z +16)
# 3804 for validation,    from 2018-06-29 18Z +20 to      2021-12-25 06Z +17.
# 3509 for testing,       from 2018-06-30 18Z +22 to      2021-12-05 18Z +17.
# Preparing bin splits by sampling 300 training forecasts with events
# sampling 23991 datapoints...computing bin splits...done.
# Loading training data


# $ FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 make train_gradient_boosted_decision_trees
# 18688 for training,     from 2018-06-29 00Z +24 to      2021-12-31 00Z +24.
#   (1581 with sig_hail,  from 2018-06-29 00Z +24 to      2021-12-14 18Z +29)
#   (7152 with hail,      from 2018-06-29 00Z +24 to      2021-12-29 18Z +28)
#   (3108 with tornado,   from 2018-07-03 00Z +31 to      2021-12-30 18Z +29)
#   (478 with sig_tornado,        from 2018-07-03 00Z +31 to      2021-12-15 00Z +27)
#   (2527 with sig_wind,  from 2018-06-29 00Z +25 to      2021-12-29 18Z +27)
#   (11027 with wind,     from 2018-06-29 00Z +24 to      2021-12-29 18Z +28)
# 3787 for validation,    from 2018-06-29 06Z +32 to      2021-12-24 18Z +29.
# 3474 for testing,       from 2018-06-30 06Z +34 to      2021-12-05 06Z +29.
# Preparing bin splits by sampling 300 training forecasts with events
# computing radius ranges...done
# 300/~300 forecasts loaded.  9.504083029233334s each.  ~0.0 hours left.             left.
# sampling 29931 datapoints...computing bin splits...done.
# Loading training data
# done. 28310346 datapoints with 17412 features each.
# Loading validation data
# done. 5812153 datapoints with 17412 features each.



# on nadocaster2:
# mkdir ~/hd
# sshfs brian@nadocaster:/Volumes/hd/ ~/hd/

# on nadocaster:
# ln -s /Volumes/hd ~/hd

# on either:
# cd ~/hd
# JULIA_MPI_BINARY=system DISTRIBUTED=true JULIA_NUM_THREADS=$CORE_COUNT EVENT_TYPES=tornado,wind,hail,sig_tornado,sig_wind,sig_hail MUST_LOAD_FROM_DISK=true FORECAST_HOUR_RANGE=24:35 DATA_SUBSET_RATIO=0.026 NEAR_STORM_RATIO=0.4 time mpirun -n 2 -wdir $(pwd) -hosts 192.168.1.112:1,192.168.1.121:1 -bind-to none julia --compiled-modules=no --project=/home/brian/nadocast_dev /home/brian/nadocast_dev/models/href_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl

