import Dates
# import DelimitedFiles
import Plots
import Random

import MagicTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts
import Grib2

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF

model_prefix = "gbdt_$(Dates.now())"

all_sref_forecasts = SREF.forecasts() #[1:11:21034] # Skip a bunch: more diversity, since there's always multiple forecasts for the same valid time

(grid, conus_on_grid, train_forecasts, validation_forecasts, test_forecasts) =
  TrainingShared.forecasts_grid_conus_on_grid_train_validation_test(all_sref_forecasts)

conus_grid_bitmask = (conus_on_grid .== 1.0f0)

println("$(length(train_forecasts)) for training.")
println("$(length(validation_forecasts)) for validation.")
println("$(length(test_forecasts)) for testing.")


forecasts_per_chunk = 350
bin_split_forecast_sample_count = 100

function get_data_and_labels(forecasts; X_transformer = identity)
  print("Loading")

  Xs = []
  Ys = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data = SREF.get_feature_engineered_data(forecast, data)

    data_in_conus = data[conus_grid_bitmask, :]
    labels        = TrainingShared.forecast_labels(grid, forecast)[conus_grid_bitmask] :: Array{Float32,1}

    push!(Xs, X_transformer(data_in_conus))
    push!(Ys, labels)

    print(".")
  end
  println("done.")

  if length(Xs) > 0
    (vcat(Xs...) :: Array{<:Number,2}, vcat(Ys...) :: Array{Float32,1})
  else
    nothing
  end
end

# Returns (X_binned, labels)
function get_data_and_labels_binned(forecasts, bin_splits)
  transformer(X) = MagicTreeBoosting.apply_bins(X, bin_splits)
  get_data_and_labels(forecasts, X_transformer = transformer)
end


# booster_config = [
#   "eta"              => 0.1, # learning rate (aka shrinkage rate)
#   "min_child_weight" => 50,
#   "max_leaves"       => 10,
#   "reg_alpha"        => 0.1, # L1 regularization on term weights
#   "reg_lambda"       => 0.1, # L2 regularization on term weights
#   "tree_method"      => "hist",
#   "grow_policy"      => "lossguide",
#   "objective"        => "binary:logistic",
#   "eval_metric"      => "logloss"
# ]

function save(epoch_i, validation_loss, bin_splits, trees)
  try
    mkdir("$(model_prefix)")
  catch
  end
  MagicTreeBoosting.save("$(model_prefix)/epoch_$(epoch_i)_loss_$(validation_loss).model", bin_splits, trees)
end

epoch_i    = 1
bin_splits = nothing
trees      = MagicTreeBoosting.Tree[MagicTreeBoosting.Leaf(-5.0,nothing, nothing)]
validation_X_binned, validation_y = (nothing, nothing)


println("Preparing bin splits by sampling $bin_split_forecast_sample_count training forecasts...")

(bin_sample_X, _) = get_data_and_labels(Iterators.take(Random.shuffle(train_forecasts), bin_split_forecast_sample_count))
bin_splits        = MagicTreeBoosting.prepare_bin_splits(bin_sample_X, 255)
bin_sample_X      = nothing # freeeeeeee

while length(trees) <= 150
  global epoch_i
  global bin_splits
  global trees
  global validation_X_binned
  global validation_y

  println("===== Epoch $epoch_i =====")

  validation_loss = nothing

  for chunk_of_forecasts in Iterators.partition(Random.shuffle(train_forecasts), forecasts_per_chunk)
    X_binned, y = get_data_and_labels_binned(chunk_of_forecasts, bin_splits)

    trees =
      MagicTreeBoosting.train_on_binned(
        X_binned, y,
        prior_trees             = trees,
        iteration_count         = 4,
        min_data_weight_in_leaf = 4000.0,
        l2_regularization       = 1.0,
        max_leaves              = 6,
        max_depth               = 4,
        max_delta_score         = 5.0,
        learning_rate           = 0.1,
        feature_fraction        = 0.5,
      )

    if validation_X_binned == nothing
      println("Loading validation data...")
      validation_X_binned, validation_y = get_data_and_labels_binned(validation_forecasts, bin_splits)
    end
    print("Predicting...")
    validation_ŷ    = MagicTreeBoosting.predict_on_binned(validation_X_binned, trees)
    validation_loss = sum(MagicTreeBoosting.logloss.(validation_y, validation_ŷ)) / length(validation_y)
    println("done.")

    println("Validation loss after chunk: $validation_loss")
  end

  save(epoch_i, validation_loss, bin_splits, trees)

  # for forecast in validation_forecasts[[5,10,15,30,40,50]]
  #   print("Plotting $(Forecasts.time_title(forecast)) (epoch+$(Forecasts.valid_time_in_seconds_since_epoch_utc(forecast))s)...")
  #   X = SREF.get_feature_engineered_data(forecast, Forecasts.get_data(forecast))
  #   y = TrainingShared.forecast_labels(grid, forecast)
  #   ŷ = MagicTreeBoosting.predict(X, bin_splits, trees)
  #   prefix = "$(model_prefix)/epoch_$(epoch_i)_forecast_$(replace(Forecasts.time_title(forecast), " " => "_"))"
  #   Plots.png(Grib2.plot(grid, Float32.(ŷ)), "$(prefix)_predictions.png")
  #   Plots.png(Grib2.plot(grid, y), "$(prefix)_labels.png")
  #   println("done.")
  # end

  epoch_i += 1
end

