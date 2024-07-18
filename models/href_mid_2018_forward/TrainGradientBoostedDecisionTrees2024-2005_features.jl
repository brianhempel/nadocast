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
    nothing
  end

# when not using 3-hr windows:
# 1:12
# 13:24
# 25:36


event_types        = split(get(ENV, "EVENT_TYPES", ""), ",")
event_types        = event_types == [""] ? nothing : event_types
data_subset_ratio  = parse(Float32, get(ENV, "DATA_SUBSET_RATIO", "0.03"))
near_storm_ratio   = parse(Float32, get(ENV, "NEAR_STORM_RATIO", "0.4"))
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

    bin_split_forecast_sample_count    = 450, # will be divided among the label types
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


# $ FORECAST_HOUR_RANGE=1:12 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_1:12_2024-2005.log
# gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_wind_adj/1142_trees_loss_0.0013586894.model
# gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_sig_hail/953_trees_loss_0.0003080716.model
# $ CORE_COUNT=5 EVENT_TYPES=hail,wind,sig_wind_adj,sig_wind,tornado_life_risk,tornado,sig_tornado FORECAST_HOUR_RANGE=1:12 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_1:12_2024-2005.log2

# train_1:12_2024-2005.log:gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_wind_adj/1142_trees_loss_0.0013586894.model
# train_1:12_2024-2005.log:gbdt_2024-2005_features_f1-12_2024-05-29T05.01.20.957_sig_hail/953_trees_loss_0.0003080716.model
# train_1:12_2024-2005.log2:gbdt_2024-2005_features_f1-12_2024-06-13T07.09.03.411_hail/1245_trees_loss_0.0018700788.model
# train_1:12_2024-2005.log3:gbdt_2024-2005_features_f1-12_2024-06-25T14.35.47.253_wind/1210_trees_loss_0.0036398584.model
# train_1:12_2024-2005.log3:gbdt_2024-2005_features_f1-12_2024-06-25T14.35.47.253_sig_wind_adj/571_trees_loss_0.00021140957.model

# nadocaster3:
# $ CORE_COUNT=88 MUST_LOAD_FROM_DISK=true EVENT_TYPES=tornado_life_risk,tornado,sig_tornado FORECAST_HOUR_RANGE=1:12 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_1:12_2024-2005.log2
# train_1:12_2024-2005.log88:gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_tornado_life_risk/700_trees_loss_2.3796445e-5.model
# train_1:12_2024-2005.log88:gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_tornado/886_trees_loss_0.000570307.model
# train_1:12_2024-2005.log88:gbdt_2024-2005_features_f1-12_2024-06-17T09.17.19.881_sig_tornado/864_trees_loss_8.5906206e-5.model

# $ CORE_COUNT=44 MUST_LOAD_FROM_DISK=true EVENT_TYPES=tornado_life_risk,tornado,sig_tornado,wind,sig_wind,hail,sig_hail,wind_adj,sig_wind_adj FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_13:24_2024-2005.log
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_tornado_life_risk/444_trees_loss_2.5661777e-5.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_tornado/986_trees_loss_0.00061401847.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_tornado/656_trees_loss_9.254449e-5.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_wind/1226_trees_loss_0.0039208783.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_wind/879_trees_loss_0.0005461219.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_hail/1250_trees_loss_0.0020206557.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_sig_hail/422_trees_loss_0.00033432615.model
# train_13:24_2024-2005.log:gbdt_2024-2005_features_f13-24_2024-06-22T04.41.06.416_wind_adj/1250_trees_loss_0.00144605.model

# CORE_COUNT=44 MUST_LOAD_FROM_DISK=true EVENT_TYPES=wind_adj,sig_hail FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log4

# nadocaster-chance:
# CORE_COUNT=16 MUST_LOAD_FROM_DISK=true EVENT_TYPES=tornado FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log
# train_25:36_2024-2005.log2:gbdt_2024-2005_features_f25-36_2024-07-02T04.25.59.068_tornado/1217_trees_loss_0.0006426961.model

# CORE_COUNT=16 MUST_LOAD_FROM_DISK=true EVENT_TYPES=sig_wind_adj FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log

# nadocaster-wx-scan:
# CORE_COUNT=64 MUST_LOAD_FROM_DISK=true EVENT_TYPES=tornado_life_risk,sig_tornado,hail,sig_hail FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log

# gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_tornado_life_risk/773_trees_loss_2.4938929e-5.model
# gbdt_2024-2005_features_f25-36_2024-07-03T07.38.26.145_sig_tornado/639_trees_loss_9.4197974e-5.model

# nadocaster-wx-scan2:
# CORE_COUNT=64 MUST_LOAD_FROM_DISK=true EVENT_TYPES=wind,sig_wind,wind_adj,sig_wind_adj FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log



# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=13:24 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_13:24_2024-2005.log2
# 60513 for training,     from 2018-06-29 00Z +13 to      2024-02-27 18Z +17.
#   (4218 with sig_wind,  from 2018-06-29 00Z +13 to      2024-02-27 18Z +17)
#   (2754 with sig_hail,  from 2018-06-29 00Z +22 to      2024-02-27 18Z +16)
#   (11555 with hail,     from 2018-06-29 00Z +20 to      2024-02-27 18Z +17)
#   (4218 with sig_wind_adj,      from 2018-06-29 00Z +13 to      2024-02-27 18Z +17)
#   (4856 with tornado_life_risk,         from 2018-06-29 00Z +15 to      2024-02-27 18Z +17)
#   (4856 with tornado,   from 2018-06-29 00Z +15 to      2024-02-27 18Z +17)
#   (17386 with wind_adj,         from 2018-06-29 00Z +13 to      2024-02-27 18Z +17)
#   (848 with sig_tornado,        from 2018-07-03 12Z +19 to      2024-02-27 18Z +17)
#   (17386 with wind,     from 2018-06-29 00Z +13 to      2024-02-27 18Z +17)
# 12114 for validation,   from 2018-06-29 12Z +24 to      2024-02-24 18Z +17.
# 12263 for testing,      from 2018-06-30 12Z +24 to      2024-02-25 18Z +17.
# Preparing bin splits by sampling 450 training forecasts with events


# $ LOAD_ONLY=true FORECAST_HOUR_RANGE=25:36 DATA_SUBSET_RATIO=0.03 NEAR_STORM_RATIO=0.4 ONLY_BEFORE=2024-2-28 make train_gradient_boosted_decision_trees 2>&1 | tee -a train_25:36_2024-2005.log
# 60547 for training,     from 2018-06-29 00Z +25 to      2024-02-27 06Z +29.
#   (4200 with sig_wind,  from 2018-06-29 00Z +25 to      2024-02-27 06Z +29)
#   (2733 with sig_hail,  from 2018-06-29 00Z +25 to      2024-02-27 06Z +28)
#   (11504 with hail,     from 2018-06-29 00Z +25 to      2024-02-27 06Z +29)
#   (4200 with sig_wind_adj,      from 2018-06-29 00Z +25 to      2024-02-27 06Z +29)
#   (4854 with tornado_life_risk,         from 2018-07-03 00Z +31 to      2024-02-27 06Z +29)
#   (4854 with tornado,   from 2018-07-03 00Z +31 to      2024-02-27 06Z +29)
#   (17336 with wind_adj,         from 2018-06-29 00Z +25 to      2024-02-27 06Z +29)
#   (853 with sig_tornado,        from 2018-07-03 00Z +31 to      2024-02-27 06Z +29)
#   (17336 with wind,     from 2018-06-29 00Z +25 to      2024-02-27 06Z +29)
# 12120 for validation,   from 2018-06-29 00Z +36 to      2024-02-24 06Z +29.
# 12179 for testing,      from 2018-06-30 00Z +36 to      2024-02-25 06Z +29.
