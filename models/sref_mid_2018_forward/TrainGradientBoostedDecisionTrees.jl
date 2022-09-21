import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import TrainGBDTShared

must_load_from_disk = parse(Bool, get(ENV, "MUST_LOAD_FROM_DISK", "false"))

if !must_load_from_disk
  push!(LOAD_PATH, @__DIR__)
  import SREF
end


# SREF files come out 3-4 hours after run time

forecast_hour_range =
  if occursin(r"^\d+:\d+$", get(ENV, "FORECAST_HOUR_RANGE", ""))
    start, stop = map(str -> parse(Int64, str), split(ENV["FORECAST_HOUR_RANGE"], ":"))
    start:stop
  else
    2:38 # 1:87 # 4:39
  end

event_types        = split(get(ENV, "EVENT_TYPES", ""), ",")
event_types        = event_types == [""] ? nothing : event_types
load_only          = parse(Bool, get(ENV, "LOAD_ONLY", "false"))
distributed        = parse(Bool, get(ENV, "DISTRIBUTED", "false"))
climatology_amount = get(ENV, "CLIMATOLOGY", "all") # options: none, minimal, some, all


data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.3"))

near_storm_ratio = 1f0 # SREF is tiny

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    must_load_from_disk ? [] : SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix        = model_prefix,
    save_dir            = "sref_$(hour_range_str)_$(data_subset_ratio)",
    event_types         = event_types,
    load_only           = load_only,
    must_load_from_disk = must_load_from_disk,
    use_mpi             = distributed,

    data_subset_ratio = data_subset_ratio,
    near_storm_ratio  = near_storm_ratio,

    climatology_amount = climatology_amount, # can vary this without reloading the data

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


# $ FORECASTS_ROOT=../../test_grib2s LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.3 make train_gradient_boosted_decision_trees
# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ sshfs brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.3 make train_gradient_boosted_decision_trees
# 20843 for training,     from 2019-01-17 09Z +12 to      2022-06-01 03Z +5.
#   (2553 with sig_wind,  from 2019-01-24 03Z +8  to      2022-06-01 03Z +2)
#   (1706 with sig_hail,  from 2019-03-11 09Z +12 to      2022-05-31 21Z +6)
#   (7195 with hail,      from 2019-01-18 21Z +10 to      2022-06-01 03Z +3)
#   (2553 with sig_wind_adj,      from 2019-01-24 03Z +8  to      2022-06-01 03Z +2)
#   (3222 with tornado,   from 2019-01-17 15Z +8  to      2022-05-31 21Z +7)
#   (10700 with wind_adj,         from 2019-01-18 21Z +10 to      2022-06-01 03Z +2)
#   (548 with sig_tornado,        from 2019-03-12 15Z +9  to      2022-05-30 15Z +7)
#   (10700 with wind,     from 2019-01-18 21Z +10 to      2022-06-01 03Z +2)
# 4219 for validation,    from 2019-01-19 03Z +9  to      2022-05-29 09Z +2.
# 3854 for testing,       from 2019-01-20 03Z +10 to      2022-05-30 09Z +2.
# Preparing bin splits by sampling 400 training forecasts with events
# ...
# done. 26397033 datapoints with 14335 features each.
# Loading validation data
# done. 5359021 datapoints with 14335 features each.




# cd ~/hd
# FORECASTS_ROOT=/home/brian/nadocaster2/ USE_ALT_DISK=true LOAD_ONLY=true FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.3 JULIA_NUM_THREADS=$CORE_COUNT time julia --project=~/nadocast_dev ~/nadocast_dev/models/sref_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
# 20808 for training,     from 2019-01-17 03Z +18 to      2022-05-31 15Z +17.
#   (2550 with sig_wind,  from 2019-01-23 15Z +20 to      2022-05-31 15Z +14)
#   (1706 with sig_hail,  from 2019-03-11 03Z +18 to      2022-05-31 15Z +12)
#   (7192 with hail,      from 2019-01-18 09Z +22 to      2022-05-31 15Z +15)
#   (2550 with sig_wind_adj,      from 2019-01-23 15Z +20 to      2022-05-31 15Z +14)
#   (3222 with tornado,   from 2019-01-17 03Z +20 to      2022-05-31 15Z +13)
#   (10693 with wind_adj,         from 2019-01-18 09Z +22 to      2022-05-31 15Z +14)
#   (548 with sig_tornado,        from 2019-03-12 03Z +21 to      2022-05-30 09Z +13)
#   (10693 with wind,     from 2019-01-18 09Z +22 to      2022-05-31 15Z +14)
# 4220 for validation,    from 2019-01-18 15Z +21 to      2022-05-28 21Z +14.
# 3856 for testing,       from 2019-01-19 15Z +22 to      2022-05-29 21Z +14.
# Preparing bin splits by sampling 400 training forecasts with events
# sampling 3243 datapoints...computing bin splits...done.
# Loading training data


# cd ~/hd
# FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.2 JULIA_NUM_THREADS=$CORE_COUNT time julia --project=~/nadocast_dev ~/nadocast_dev/models/sref_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
# 31157 for training,     from 2019-01-16 09Z +36 to      2022-05-31 09Z +23.
#   (3823 with sig_wind,  from 2019-01-22 21Z +38 to      2022-05-31 03Z +26)
#   (2557 with sig_hail,  from 2019-03-10 09Z +36 to      2022-05-31 03Z +24)
#   (10774 with hail,     from 2019-01-17 21Z +34 to      2022-05-31 09Z +21)
#   (3823 with sig_wind_adj,      from 2019-01-22 21Z +38 to      2022-05-31 03Z +26)
#   (4820 with tornado,   from 2019-01-16 09Z +38 to      2022-05-31 03Z +25)
#   (16035 with wind_adj,         from 2019-01-17 21Z +34 to      2022-05-31 03Z +26)
#   (822 with sig_tornado,        from 2019-03-11 15Z +33 to      2022-05-29 21Z +25)
#   (16035 with wind,     from 2019-01-17 21Z +34 to      2022-05-31 03Z +26)
# 6330 for validation,    from 2019-01-18 03Z +33 to      2022-05-28 09Z +26.
# 5798 for testing,       from 2019-01-19 03Z +34 to      2022-05-29 09Z +26.
# Preparing bin splits by sampling 400 training forecasts with events
# sampling 4013 datapoints...computing bin splits...done.
# Loading training data





# time scp -C -r nadocaster2-remote:~/nadocast_dev/models/sref_mid_2018_forward/sref_f2-13_0.26_training ./



# on nadocaster2:
# mkdir ~/hd
# sshfs brian@nadocaster:/Volumes/hd/ ~/hd/

# on nadocaster:
# ln -s /Volumes/hd ~/hd

# on either:
# cd ~/hd
# JULIA_MPI_BINARY=system DISTRIBUTED=true JULIA_NUM_THREADS=$CORE_COUNT EVENT_TYPES=hail,sig_tornado,sig_wind,sig_hail MUST_LOAD_FROM_DISK=true FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.17 time mpirun -n 2 -wdir $(pwd) -hosts 192.168.1.112:1,192.168.1.121:1 -bind-to none julia --compiled-modules=no --project=/home/brian/nadocast_dev /home/brian/nadocast_dev/models/sref_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl
