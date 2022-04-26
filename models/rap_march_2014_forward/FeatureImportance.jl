# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2_2021-06-11T14.38.30.5/445_trees_loss_0.00081564416.model > f2_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f6_2021-06-04T14.25.56.451/297_trees_loss_0.0009202203.model > f6_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12_2021-06-07T05.23.57.904/477_trees_loss_0.0009632701.model > f12_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f17_2021-06-03T15.44.45.541/243_trees_loss_0.0009980382.model > f17_feature_importance_2020_models.txt


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import RAP

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  RAP.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
