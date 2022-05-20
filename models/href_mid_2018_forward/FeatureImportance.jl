# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_tornado/391_trees_loss_0.0010360148.model > f2-13_feature_importance_2021_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_tornado/317_trees_loss_0.001094988.model > f13-24_feature_importance_2021_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-16T14.36.46.241_tornado/308_trees_loss_0.0011393429.model > f24-35_feature_importance_2021_tornado_models.txt

# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_wind/414_trees_loss_0.006970079.model > f24-35_feature_importance_2021_wind_models.txt


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import HREF

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  HREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
