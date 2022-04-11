import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
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

event_type = get(ENV, "EVENT_TYPE", nothing)
load_only  = parse(Bool, get(ENV, "LOAD_ONLY", "false"))

data_subset_ratio = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.26"))

near_storm_ratio = 1f0 # SREF is tiny

hour_range_str = "f$(forecast_hour_range.start)-$(forecast_hour_range.stop)"

model_prefix = "gbdt_3hr_window_3hr_min_mean_max_delta_$(hour_range_str)_$(replace(string(Dates.now()), ":" => "."))"

TrainGBDTShared.train_with_coordinate_descent_hyperparameter_search(
    SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts();
    forecast_hour_range = forecast_hour_range,
    model_prefix        = model_prefix,
    save_dir            = "sref_$(hour_range_str)_$(data_subset_ratio)",
    only_events_of_type = event_type,
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
    feature_fraction            = [0.1, 0.25, 0.5, 0.75, 1.0],
    bagging_temperature         = [0.25]
  )


# $ FORECASTS_ROOT=../../test_grib2s LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.26 make train_gradient_boosted_decision_trees
# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.26 make train_gradient_boosted_decision_trees
# 16177 for training,      from 2019-01-17 09Z +13 to 2021-12-31 21Z +3.
#   (1464 with sig_hail,   from 2019-03-11 09Z +12 to 2021-12-15 21Z +2)
#   (6299 with hail,       from 2019-01-18 21Z +10 to 2021-12-30 15Z +7)
#   (2718 with tornado,    from 2019-01-17 15Z +8  to 2021-12-31 21Z +2)
#   (430 with sig_tornado, from 2019-03-12 15Z +9  to 2021-12-15 21Z +6)
#   (2247 with sig_wind,   from 2019-01-24 03Z +8  to 2021-12-30 15Z +6)
#   (9498 with wind,       from 2019-01-18 21Z +10 to 2021-12-30 15Z +7)
# 3290 for validation,     from 2019-01-19 03Z +9  to 2021-12-25 21Z +2.
# 2922 for testing,        from 2019-01-20 03Z +11 to 2021-12-06 09Z +2.
# Preparing bin splits by sampling 300 training forecasts with events
# sampling 3415 datapoints...computing bin splits...done.
# Loading training data
# done. 23705020 datapoints with 18777 features each.
# Loading validation data
# done. 4839602 datapoints with 18777 features each.
# 416GB + 85GB

# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ sshfs brian@nadocaster2:/Volumes/ ~/nadocaster2/
# FORECASTS_ROOT=/home/brian/nadocaster2/ LOAD_ONLY=true FORECAST_HOUR_RANGE=12:23 DATA_SUBSET_RATIO=0.26 make train_gradient_boosted_decision_trees
# 16154 for training,      from 2019-01-17 03Z +19 to      2021-12-31 09Z +15.
#   (1464 with sig_hail,   from 2019-03-11 03Z +18 to      2021-12-15 09Z +14)
#   (6296 with hail,       from 2019-01-18 09Z +22 to      2021-12-30 09Z +13)
#   (2718 with tornado,    from 2019-01-17 03Z +20 to      2021-12-31 09Z +14)
#   (430 with sig_tornado, from 2019-03-12 03Z +21 to      2021-12-15 15Z +12)
#   (2244 with sig_wind,   from 2019-01-23 15Z +20 to      2021-12-30 09Z +12)
#   (9492 with wind,       from 2019-01-18 09Z +22 to      2021-12-30 09Z +13)
# 3290 for validation,     from 2019-01-18 15Z +21 to      2021-12-25 09Z +14).
# 2924 for testing,        from 2019-01-19 15Z +23 to      2021-12-05 21Z +14).
# Preparing bin splits by sampling 300 training forecasts with events
# sampling 3908 datapoints...computing bin splits...done.
# Loading training data
# done. 23669916 datapoints with 18777 features each.
# Loading validation data
# done. 4840284 datapoints with 18777 features each.


# LOAD_ONLY=true FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.26 make train_gradient_boosted_decision_trees
# 24182 for training,      from 2019-01-16 09Z +37 to      2021-12-31 03Z +21.
#   (2194 with sig_hail,   from 2019-03-10 09Z +36 to      2021-12-14 21Z +26)
#   (9430 with hail,       from 2019-01-17 21Z +34 to      2021-12-29 21Z +25)
#   (4064 with tornado,    from 2019-01-16 09Z +38 to      2021-12-30 21Z +26)
#   (645 with sig_tornado, from 2019-03-11 15Z +33 to      2021-12-15 03Z +24)
#   (3364 with sig_wind,   from 2019-01-22 21Z +38 to      2021-12-29 21Z +24)
#   (14232 with wind,      from 2019-01-17 21Z +34 to      2021-12-29 21Z +25)
# 4935 for validation,     from 2019-01-18 03Z +33 to      2021-12-24 21Z +26.
# 4393 for testing,        from 2019-01-19 03Z +35 to      2021-12-05 09Z +26.
# Preparing bin splits by sampling 300 training forecasts with events
# sampling 3365 datapoints...computing bin splits...done.
# Loading training data
# done. 35439117 datapoints with 18777 features each.
# Loading validation data
# done. 7261105 datapoints with 18777 features each.
# 621GB + 128GB TOO MUCH, do over

# LOAD_ONLY=true FORECAST_HOUR_RANGE=21:38 DATA_SUBSET_RATIO=0.17 make train_gradient_boosted_decision_trees
# 24182 for training,      from 2019-01-16 09Z +37 to      2021-12-31 03Z +21.
#   (2194 with sig_hail,   from 2019-03-10 09Z +36 to      2021-12-14 21Z +26)
#   (9430 with hail,       from 2019-01-17 21Z +34 to      2021-12-29 21Z +25)
#   (4064 with tornado,    from 2019-01-16 09Z +38 to      2021-12-30 21Z +26)
#   (645 with sig_tornado, from 2019-03-11 15Z +33 to      2021-12-15 03Z +24)
#   (3364 with sig_wind,   from 2019-01-22 21Z +38 to      2021-12-29 21Z +24)
#   (14232 with wind,      from 2019-01-17 21Z +34 to      2021-12-29 21Z +25)
# 4935 for validation,     from 2019-01-18 03Z +33 to      2021-12-24 21Z +26).
# 4393 for testing,        from 2019-01-19 03Z +35 to      2021-12-05 09Z +26).
# Preparing bin splits by sampling 300 training forecasts with events
# computing radius ranges...done
# 300/~300 forecasts loaded.  3.4334856009466663s each.  ~0.0 hours left.            left.
# sampling 3365 datapoints...computing bin splits...done.
# Loading training data
# done. 5103845 datapoints with 18777 features each.
# 12:57:59 elapsed
# 436G + 90G

# time scp -C -r nadocaster2-remote:~/nadocast_dev/models/sref_mid_2018_forward/sref_f2-13_0.26_training ./

