# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_tornado/389_trees_loss_0.001139937.model > f2-13_feature_importance_2021v1_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_tornado/271_trees_loss_0.0012234614.model > f12-23_feature_importance_2021v1_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-12T21.11.05.785_tornado/187_trees_loss_0.0012662631.model > f21-38_feature_importance_2021v1_tornado_models.txt


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import SREF

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
