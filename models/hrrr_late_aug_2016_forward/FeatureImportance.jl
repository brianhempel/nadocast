# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2_2021-05-21T22.37.26.544/463_trees_loss_0.0008549077.model > f2_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f6_2021-05-27T17.12.43.406/465_trees_loss_0.00092996453.model > f6_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12_2021-05-10T20.00.43.881/471_trees_loss_0.0010017317.model > f12_feature_importance_2020_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia-1.5 --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f17_2021-04-30T10.49.47.495/341_trees_loss_0.0010138382.model > f17_feature_importance_2020_models.txt


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import HRRR

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  HRRR.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
