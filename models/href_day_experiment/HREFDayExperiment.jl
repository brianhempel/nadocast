module HREFDayExperiment

import Dates

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import ForecastCombinators
import Grids
import Inventories


push!(LOAD_PATH, (@__DIR__) * "/../shared")
import PredictionForecasts
import FeatureEngineeringShared
import Climatology

push!(LOAD_PATH, (@__DIR__) * "/../href_mid_2018_forward")
import HREF


const layer_blocks_to_make = [
  FeatureEngineeringShared.raw_features_block,
  FeatureEngineeringShared.twenty_five_mi_mean_block,
  FeatureEngineeringShared.fifty_mi_mean_block,
  FeatureEngineeringShared.hundred_mi_mean_block,
  # FeatureEngineeringShared.twenty_five_mi_forward_gradient_block,
  # FeatureEngineeringShared.twenty_five_mi_leftward_gradient_block,
  # FeatureEngineeringShared.twenty_five_mi_linestraddling_gradient_block,
  # FeatureEngineeringShared.fifty_mi_forward_gradient_block,
  # FeatureEngineeringShared.fifty_mi_leftward_gradient_block,
  # FeatureEngineeringShared.fifty_mi_linestraddling_gradient_block,
  # FeatureEngineeringShared.hundred_mi_forward_gradient_block,
  # FeatureEngineeringShared.hundred_mi_leftward_gradient_block,
  # FeatureEngineeringShared.hundred_mi_linestraddling_gradient_block,
]

function grid()
  HREF.grid()
end

function feature_engineered_hour_forecasts()
  FeatureEngineeringShared.feature_engineered_forecasts(
    HREF.forecasts();
    vector_wind_layers = HREF.vector_wind_layers,
    layer_blocks_to_make = layer_blocks_to_make,
    new_features_pre = HREF.extra_features
  )
end

function compute_min_mean_max!(nhours, nfeatures_per_hour, concated_data, out)
  nhours_f32 = Float32(nhours)

  Threads.@threads for feature_i in 1:nfeatures_per_hour
    for i in 1:size(concated_data,1)
      min_val = Inf32
      total   = 0f0
      max_val = -Inf32
      for concated_feature_i in feature_i:nfeatures_per_hour:size(concated_data,2)
        val = concated_data[i, concated_feature_i]
        min_val = min(val, min_val)
        total += val
        max_val = max(val, max_val)
      end
      out[i, feature_i*3 - 2] = min_val
      out[i, feature_i*3 - 1] = total / nhours_f32
      out[i, feature_i*3 - 0] = max_val
    end
  end

  ()
end

function min_mean_max_forecasts_with_climatology_etc()
  hour_forecasts = feature_engineered_hour_forecasts()

  day1_hourlies_concated, _day2_hourlies_concated, _fourhourlies_concated = ForecastCombinators.gather_daily_and_fourhourly(hour_forecasts)

  climatology_features = Climatology.climatology_features(grid(); run_datetime_to_simulation_version = HREF.run_datetime_to_simulation_version)

  function day_inventory(concated_forecast, concated_inventory)
    nhours = length(concated_forecast.based_on)
    nfeatures_per_hour = length(concated_inventory) ÷ nhours
    @assert nhours * nfeatures_per_hour == length(concated_inventory)

    date_str = last(concated_inventory).date_str

    out_lines = Inventories.InventoryLine[]
    for hour_line in concated_inventory[1:nfeatures_per_hour]
      push!(out_lines, Inventories.InventoryLine("", "", date_str, hour_line.abbrev, hour_line.level, hour_line.forecast_hour_str, hour_line.misc * " day min",  hour_line.feature_engineering))
      push!(out_lines, Inventories.InventoryLine("", "", date_str, hour_line.abbrev, hour_line.level, hour_line.forecast_hour_str, hour_line.misc * " day mean", hour_line.feature_engineering))
      push!(out_lines, Inventories.InventoryLine("", "", date_str, hour_line.abbrev, hour_line.level, hour_line.forecast_hour_str, hour_line.misc * " day max",  hour_line.feature_engineering))
    end
    for (feature_name, _) in climatology_features
      push!(out_lines, Inventories.InventoryLine( "", "", date_str, feature_name, "calculated", "day fcst", "", ""))
    end

    out_lines
  end

  function day_min_mean_max(concated_forecast, concated_data)
    nhours = length(concated_forecast.based_on)
    nfeatures_per_hour = size(concated_data, 2) ÷ nhours
    @assert nhours * nfeatures_per_hour == size(concated_data, 2)

    out = Array{Float32}(undef, (size(concated_data, 1), 3*nfeatures_per_hour + length(climatology_features)))

    compute_min_mean_max!(nhours, nfeatures_per_hour, concated_data, out)

    Threads.@threads for clim_feature_i in 1:length(climatology_features)
      _feature_name, compute_feature = climatology_features[clim_feature_i]
      out[:, 3*nfeatures_per_hour + clim_feature_i] = compute_feature(concated_forecast)
    end

    out
  end

  ForecastCombinators.map_forecasts(day1_hourlies_concated; inventory_transformer = day_inventory, data_transformer = day_min_mean_max, model_name = "HREFDayExperiment_Day1")
end

# Best hyperparameters:
# Dict{Symbol, Real}(:max_depth => 4, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 8,  :l2_regularization => 100.0,  :normalize_second_opinion => true,  :feature_fraction => 0.032, :second_opinion_weight => 0.033, :bagging_temperature => 0.25, :min_data_weight_in_leaf => 464.0)    gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T06.28.46.943_tornado/587_trees_loss_0.008584802.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 1.0,    :normalize_second_opinion => true,  :feature_fraction => 0.1,   :second_opinion_weight => 0.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215000.0) gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T07.23.03.252_tornado/200_trees_loss_0.009353732.model
# Dict{Symbol, Real}(:max_depth => 7, :max_delta_score => 1.0,  :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 100.0,  :normalize_second_opinion => false, :feature_fraction => 0.32,  :second_opinion_weight => 0.1,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 2150.0)   gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T06.28.46.943_wind/658_trees_loss_0.040234573.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 10.0,   :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T07.23.03.252_wind/781_trees_loss_0.04287363.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 1.0,    :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 0.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215000.0) gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T07.12.14.796_hail/516_trees_loss_0.021882312.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 0.56, :learning_rate => 0.04, :max_leaves => 30, :l2_regularization => 1000.0, :normalize_second_opinion => true,  :feature_fraction => 0.056, :second_opinion_weight => 1.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 100000.0) gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T08.11.17.582_hail/760_trees_loss_0.02368946.model
# Dict{Symbol, Real}(:max_depth => 8, :max_delta_score => 3.2,  :learning_rate => 0.04, :max_leaves => 20, :l2_regularization => 1.0,    :normalize_second_opinion => true,  :feature_fraction => 0.1,   :second_opinion_weight => 0.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215000.0) gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T07.12.14.796_sig_tornado/182_trees_loss_0.0013368124.model
# Dict{Symbol, Real}(:max_depth => 4, :max_delta_score => 1.0,  :learning_rate => 0.04, :max_leaves => 25, :l2_regularization => 1.0,    :normalize_second_opinion => true,  :feature_fraction => 0.018, :second_opinion_weight => 1.0,   :bagging_temperature => 0.25, :min_data_weight_in_leaf => 215.0)    gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T08.11.17.582_sig_tornado/219_trees_loss_0.0015899849.model



# (event_name, grib2_var_name, gbdt_0Z_day1, gbdt_12Z_day1)
# sig_tor was trained by accident...include it anyway
const models = [
  ("tornado",     "TORPROB",  "gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T07.23.03.252_tornado/200_trees_loss_0.009353732.model",      "gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T06.28.46.943_tornado/587_trees_loss_0.008584802.model"),
  ("wind",        "WINDPROB", "gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T07.23.03.252_wind/781_trees_loss_0.04287363.model",          "gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T06.28.46.943_wind/658_trees_loss_0.040234573.model"),
  ("hail",        "HAILPROB", "gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T08.11.17.582_hail/760_trees_loss_0.02368946.model",          "gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T07.12.14.796_hail/516_trees_loss_0.021882312.model"),
  ("sig_tornado", "STORPROB", "gbdt_day_experiment_min_mean_max_f35-35_2022-11-04T08.11.17.582_sig_tornado/219_trees_loss_0.0015899849.model", "gbdt_day_experiment_min_mean_max_f23-23_2022-11-04T07.12.14.796_sig_tornado/182_trees_loss_0.0013368124.model"),
]

function uncalibrated_day_prediction_forecasts()
  predictors = map(models) do (event_name, grib2_var_name, gbdt_0z_day1, gbdt_12z_day1)
    predict_0z  = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/" * gbdt_0z_day1)
    predict_12z = MemoryConstrainedTreeBoosting.load_unbinned_predictor((@__DIR__) * "/" * gbdt_12z_day1)

    predict(forecast, data) = begin
      if forecast.forecast_hour == 35 # 0Z
        predict_0z(data)
      elseif forecast.forecast_hour == 29 # 6Z, use the mean of the two
        0.5f0 .* (predict_0z(data) .+ predict_12z(data))
      elseif forecast.forecast_hour == 23  # 12Z
        predict_12z(data)
      elseif forecast.forecast_hour == 17 # 18Z
        predict_12z(data)
      else
        error("Forecast hour $(forecast.forecast_hour) is not a sensible HREF-based day 1 forecast hour!")
      end
    end

    (event_name, grib2_var_name, predict)
  end

  ForecastCombinators.disk_cache_forecasts(
    PredictionForecasts.simple_prediction_forecasts(min_mean_max_forecasts_with_climatology_etc(), predictors),
    "href_day_experiment_prediction_raw_2022_models_$(hash(models))"
  )
end

# grid = HREFDayExperiment.grid

# import Conus
# import Forecasts
# import Inventories
# import PlotMap

# function debug_plot(forecast)
#   inventory = Forecasts.inventory(forecast)
#   conus_mask = Conus.is_in_conus.(grid().latlons)
#   data = Forecasts.data(forecast)
#   for base_i in rand(1:length(inventory)÷3, 100)
#     for layer_i in 3*base_i-2 : 3*base_i
#       layer_data = @view data[:, layer_i]
#       lo, hi = extrema(@view layer_data[conus_mask])

#       str = Inventories.inventory_line_description(inventory[layer_i])
#       base_path = "feature_$(layer_i)_$(replace(str, r"[: ]" => "_"))_$(lo)_$(hi)"
#       println(base_path)

#       range = hi - lo
#       if range > 0
#         PlotMap.plot_fast(base_path, grid(), clamp.((layer_data .- lo) ./ range, 0f0, 1f0))
#       else
#         PlotMap.plot_fast(base_path, grid(), clamp.(layer_data .- lo, 0f0, 1f0))
#       end
#     end
#   end
# end


end # module HREFDayExperiment