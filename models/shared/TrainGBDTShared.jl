module TrainGBDTShared

import Random

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, @__DIR__)
import TrainingShared


struct EarlyStop <: Exception
end


# If we had infinite memory, we could train on all datapoints with all the forecast hours.
# However, we don't.
# Instead we include each datapoint with some probability (basal_inclusion_probability), and
# multiply the weight of that point by the inverse of that probability (a point included with
# probability 0.1 has its weight multiplied by 10).
#
# Positively labeled points are always included with probability 1.
#
# This is a crude mechanism for subsetting and weighting, so we train in multiple rounds
# of "annealing", increasing the resolution in areas more likely to contain positive
# labels (tornadoes). We use the most recently trained model and use the formula
# max(basal_inclusion_probability, prediction_from_last_model*prediction_inclusion_multiplier)
# as the inclusion probability for negatively labeled points. (But validation data uses only
# basal_inclusion_probability for inclusion probability of negatively labeled points.)
function train_multiple_annealing_rounds_with_coordinate_descent_hyperparameter_search(forecasts; annealing_rounds = 3, basal_inclusion_probability :: Float32, prediction_inclusion_multiplier :: Float32, config...)

  # Validation and Round 1 training
  X_and_labels_to_basal_inclusion_probabilities(X, labels) = begin
    map(1:size(X,1)) do i
      max(basal_inclusion_probability, labels[i])
    end
  end

  last_model_path = nothing

  for annealing_round_i in 1:annealing_rounds
    println("\nAnnealing round $annealing_round_i of $annealing_rounds...\n")

    if isnothing(last_model_path)
      last_model_path = train_with_coordinate_descent_hyperparameter_search(
          forecasts;
          training_X_and_labels_to_inclusion_probabilities   = X_and_labels_to_basal_inclusion_probabilities,
          validation_X_and_labels_to_inclusion_probabilities = X_and_labels_to_basal_inclusion_probabilities,
          config...
        )
    else
      last_model_bin_splits, last_model_trees = MemoryConstrainedTreeBoosting.load(last_model_path)

      X_and_labels_to_refined_inclusion_probabilities(X, labels) = begin
        predictions = MemoryConstrainedTreeBoosting.predict(X, last_model_bin_splits, last_model_trees)

        map(1:size(X,1)) do i
          max(basal_inclusion_probability, predictions[i]*prediction_inclusion_multiplier, labels[i])
        end
      end

      last_model_path = train_with_coordinate_descent_hyperparameter_search(
          forecasts;
          training_X_and_labels_to_inclusion_probabilities   = X_and_labels_to_refined_inclusion_probabilities,
          validation_X_and_labels_to_inclusion_probabilities = X_and_labels_to_basal_inclusion_probabilities,
          config...
        )
    end
  end

  last_model_path
end

# Returns path to best trained model
function train_with_coordinate_descent_hyperparameter_search(
    forecasts;
    forecast_hour_range = 1:10000,
    training_X_and_labels_to_inclusion_probabilities   = nothing,
    validation_X_and_labels_to_inclusion_probabilities = nothing,
    model_prefix = "",
    get_feature_engineered_data = nothing,
    bin_split_forecast_sample_count = 100,
    max_iterations_without_improvement = 20,
    configs...
  )

  (grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(forecasts, forecast_hour_range = forecast_hour_range)

  train_forecasts_with_tornadoes = filter(TrainingShared.forecast_is_tornado_hour, train_forecasts)

  println("$(length(train_forecasts)) for training. ($(length(train_forecasts_with_tornadoes)) with tornadoes.)")
  println("$(length(validation_forecasts)) for validation.")
  println("$(length(test_forecasts)) for testing.")


  # Returns (X_binned, labels, weights)
  get_data_labels_weights_binned(forecasts, bin_splits, X_and_labels_to_inclusion_probabilities) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, forecasts, X_transformer = transformer, X_and_labels_to_inclusion_probabilities = X_and_labels_to_inclusion_probabilities)
  end

  # Returns path
  save(validation_loss, bin_splits, trees) = begin
    try
      mkdir("$(model_prefix)")
    catch
    end
    MemoryConstrainedTreeBoosting.save("$(model_prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
  end

  println("Preparing bin splits by sampling $bin_split_forecast_sample_count training tornado hour forecasts")

  (bin_sample_X, _, _) = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts_with_tornadoes), bin_split_forecast_sample_count))
  bin_splits           = MemoryConstrainedTreeBoosting.prepare_bin_splits(bin_sample_X)
  bin_sample_X         = nothing # freeeeeeee

  println("done.")


  println("Loading training data")
  X_binned, y, weights = get_data_labels_weights_binned(train_forecasts, bin_splits, training_X_and_labels_to_inclusion_probabilities)
  println("done. $(size(X_binned,1)) datapoints with $(size(X_binned,2)) features each.")

  println("Loading validation data")
  validation_X_binned, validation_y, validation_weights = get_data_labels_weights_binned(validation_forecasts, bin_splits, validation_X_and_labels_to_inclusion_probabilities)
  println("done. $(size(validation_X_binned,1)) datapoints with $(size(validation_X_binned,2)) features each.")


  best_model_path = nothing
  best_loss = Inf32

  try_config(; config...) = begin
    validation_scores = nothing

    best_loss_for_config           = Inf32
    iterations_without_improvement = 0

    iteration_callback(trees) = begin
      new_tree = last(trees)

      if validation_scores == nothing
        validation_scores = MemoryConstrainedTreeBoosting.predict_on_binned(validation_X_binned, trees, output_raw_scores = true)
      else
        # print("Predicting...")
        validation_scores = MemoryConstrainedTreeBoosting.predict_on_binned(validation_X_binned, [new_tree], starting_scores = validation_scores, output_raw_scores = true)
      end
      validation_ŷ      = MemoryConstrainedTreeBoosting.σ.(validation_scores)
      validation_loss   = sum(MemoryConstrainedTreeBoosting.logloss.(validation_y, validation_ŷ) .* validation_weights) / sum(validation_weights)
      # println("done.")


      if validation_loss < best_loss_for_config
        best_loss_for_config           = validation_loss
        iterations_without_improvement = 0
        print("\rValidation loss: $validation_loss    ")
      else
        iterations_without_improvement += 1
      end

      if validation_loss < best_loss
        best_model_path = save(validation_loss, bin_splits, trees)
        best_loss = validation_loss
      end

      if iterations_without_improvement >= max_iterations_without_improvement
        throw(EarlyStop())
      end
    end

    try
      MemoryConstrainedTreeBoosting.train_on_binned(
        X_binned, y;
        prior_trees        = MemoryConstrainedTreeBoosting.Tree[MemoryConstrainedTreeBoosting.Leaf(-6.5)],
        weights            = weights,
        iteration_count    = Int64(round(30 / config[:learning_rate])),
        iteration_callback = iteration_callback,
        config...
      )
    catch expection
      println()
      if isa(expection, EarlyStop)
      else
        rethrow()
      end
    end

    best_loss_for_config
  end

  TrainingShared.coordinate_descent_hyperparameter_search(try_config; configs...)

  best_model_path
end

end # module TrainGBDTShared