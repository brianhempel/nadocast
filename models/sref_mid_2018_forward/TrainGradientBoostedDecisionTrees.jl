import Dates
# import DelimitedFiles
import Plots
import Random

import XGBoost

push!(LOAD_PATH, (@__DIR__) * "/../..")
import Conus
import Forecasts
import StormEvents
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF

model_prefix = "gbdt_$(Dates.now())"

all_sref_forecasts = SREF.forecasts()[1:37:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_on_grid, feature_count, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_on_grid_feature_count_train_validation_test(all_sref_forecasts)

conus_grid_bitmask = (conus_on_grid .== 1.0f0)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")


forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)

forecasts_per_stage = 50

function reset_epoch()
  global forecasts_remaining_in_epoch
  forecasts_remaining_in_epoch = Random.shuffle(train_forecasts)
end

function get_data_and_labels(forecasts)
  print("Loading")

  Xs = []
  Ys = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data_in_conus = data[conus_grid_bitmask, :]
    labels        = TrainingShared.forecast_labels(grid, forecast)[conus_grid_bitmask] :: Array{Float32,1}


    push!(Xs, data_in_conus)
    push!(Ys, labels)

    # if X == nothing
    #   X = data_in_conus
    #   Y = labels
    # else
    #   X = vcat(X, data_in_conus)
    #   Y = vcat(Y, labels) :: Array{Float32,1}
    # end

    print(".")
  end
  println("done.")

  if length(Xs) > 0
    (vcat(Xs...) :: Array{Float32,2}, vcat(Ys...) :: Array{Float32,1})
  else
    nothing
  end
end

function get_next_chunk()
  global forecasts_remaining_in_epoch

  stage_forecasts = collect(Iterators.take(forecasts_remaining_in_epoch, forecasts_per_stage))
  forecasts_remaining_in_epoch = forecasts_remaining_in_epoch[(forecasts_per_stage+1):length(forecasts_remaining_in_epoch)]

  get_data_and_labels(stage_forecasts)
end


booster_config = [
  "eta"              => 0.1, # learning rate (aka shrinkage rate)
  "min_child_weight" => 50,
  "max_leaves"       => 10,
  "reg_alpha"        => 0.1, # L1 regularization on term weights
  "reg_lambda"       => 0.1, # L2 regularization on term weights
  "tree_method"      => "hist",
  "grow_policy"      => "lossguide",
  "objective"        => "binary:logistic",
  "eval_metric"      => "logloss"
]

trees_per_stage = 5

stages = []

function run_stages(X; skip_final_sigmoid = false)
  predictions_so_far = zeros(Float32, size(X,1)) .- 5.0f0

  stage_X = XGBoost.DMatrix(X)
  for stage_i in 1:length(stages)
    stage_booster = stages[stage_i]
    XGBoost.set_info(stage_X, "base_margin", predictions_so_far)

    predictions_so_far =
      if stage_i == length(stages)
        XGBoost.predict(stage_booster, stage_X, output_margin = skip_final_sigmoid)
      else
        XGBoost.predict(stage_booster, stage_X, output_margin = true)
      end
  end

  predictions_so_far
end

function save_stage(stage_i, validation_loss)
  try
    mkdir("$(model_prefix)")
  catch
  end
  XGBoost.save(stages[stage_i], "$(model_prefix)/stage_$(stage_i)_loss_$(validation_loss).model")
end

# function test_loss(booster, forecasts)
#   test_loss            = 0.0
#   forecast_count       = 0.0
#   point_count_on_conus = sum(conus_on_grid)
#
#   for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
#     data_in_conus = data[conus_grid_bitmask, :]
#     labels = TrainingShared.forecast_labels(grid, forecast)[conus_grid_bitmask]
#
#     test_loss += Tracker.data(loss(transposed, labels)) / point_count_on_conus
#     forecast_count += 1
#     print(".")
#   end
#
#   test_loss / forecast_count
# end


# last_validation_loss = nothing
epoch_n = 1


print("Validation data ")
validation_X, validation_Y = get_data_and_labels(validation_forecasts)

while true
  # global last_validation_loss
  global epoch_n
  global stages

  while (chunk = get_next_chunk()) != nothing
    X, Y = chunk

    prior_predictions = run_stages(X, skip_final_sigmoid = true)

    stage_train_data = XGBoost.DMatrix(X, label = Y)
    XGBoost.set_info(stage_train_data, "base_margin", prior_predictions)

    # Train a few trees on this data chunk.
    stage_booster = XGBoost.xgboost(stage_train_data, trees_per_stage, param = booster_config)

    print("Calculating validation loss...")
    prior_validation_predictions = run_stages(validation_X, skip_final_sigmoid = true)

    stage_validation_data = XGBoost.DMatrix(validation_X, label = validation_Y)
    XGBoost.set_info(stage_validation_data, "base_margin", prior_validation_predictions)

    # e.g.: "[5]\tvalidation-logloss:0.583420"
    validation_loss_str = XGBoost.XGBoosterEvalOneIter(stage_booster.handle, convert(Int32, trees_per_stage), [stage_validation_data.handle], ["validation"], convert(UInt64, 1))
    validation_loss     = parse(Float32, split(validation_loss_str, ":")[2])
    println(validation_loss)

    push!(stages, stage_booster)
    save_stage(length(stages), validation_loss) # Save the newest stage.
  end

  # for forecast in validation_forecasts[[5,10,15]]
  #   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
  #   X = Forecasts.get_data(forecast)
  #   Y = TrainingShared.forecast_labels(grid, forecast)
  #   prefix   = "$(model_prefix)/epoch_$(epoch_n)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
  #   Plots.png(Grib2.plot(grid, run_stages(X)), "$(prefix)_predictions.png")
  #   Plots.png(Grib2.plot(grid, Y), "$(prefix)_labels.png")
  #   println("done.")
  # end

  reset_epoch()
  epoch_n += 1
end
