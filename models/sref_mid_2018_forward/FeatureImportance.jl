# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_tornado/389_trees_loss_0.001139937.model        > f2-13_feature_importance_2021_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_tornado/271_trees_loss_0.0012234614.model      > f13-24_feature_importance_2021_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-12T21.11.05.785_tornado/187_trees_loss_0.0012662631.model      > f24-35_feature_importance_2021_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_wind/527_trees_loss_0.0067868917.model          > f2-13_feature_importance_2021_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_wind/464_trees_loss_0.007258802.model          > f13-24_feature_importance_2021_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-13T01.36.58.700_wind/560_trees_loss_0.0074025104.model         > f24-35_feature_importance_2021_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_hail/365_trees_loss_0.00337179.model            > f2-13_feature_importance_2021_hail_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-16T10.56.41.459_hail/325_trees_loss_0.0036047401.model         > f13-24_feature_importance_2021_hail_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-20T09.45.13.302_hail/480_trees_loss_0.0037651379.model         > f24-35_feature_importance_2021_hail_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T08.46.42.718_sig_tornado/232_trees_loss_0.00019671764.model  > f2-13_feature_importance_2021_sig_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-20T04.30.47.008_sig_tornado/231_trees_loss_0.00020749163.model > f13-24_feature_importance_2021_sig_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-20T09.45.13.302_sig_tornado/189_trees_loss_0.00021580876.model > f24-35_feature_importance_2021_sig_tornado_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-22T08.00.21.554_sig_wind/211_trees_loss_0.0010130208.model      > f2-13_feature_importance_2021_sig_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-22T08.06.21.664_sig_wind/262_trees_loss_0.001065569.model      > f13-24_feature_importance_2021_sig_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-22T08.00.27.390_sig_wind/183_trees_loss_0.0010853795.model     > f24-35_feature_importance_2021_sig_wind_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-22T08.00.21.554_sig_hail/263_trees_loss_0.00052539556.model     > f2-13_feature_importance_2021_sig_hail_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_2022-04-22T08.06.21.664_sig_hail/290_trees_loss_0.0005572099.model     > f13-24_feature_importance_2021_sig_hail_models.txt
# FORECASTS_ROOT="../../test_grib2s" julia --project=../.. FeatureImportance.jl gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2022-04-22T08.00.27.390_sig_hail/319_trees_loss_0.0005862602.model     > f24-35_feature_importance_2021_sig_hail_models.txt

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import GBDTFeatureImportance

push!(LOAD_PATH, @__DIR__)
import SREF

gbdt_path = ARGS[1]

GBDTFeatureImportance.print_feature_importances(
  gbdt_path,
  SREF.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1]
)
