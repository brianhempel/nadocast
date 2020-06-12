# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2020-06-02T03.31.32.492/231_trees_loss_0.001277702.model

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import SREF

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
