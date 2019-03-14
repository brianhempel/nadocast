module TrainGBDTShared

import Random

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, @__DIR__)
import TrainingShared

struct EarlyStop <: Exception
end

function train_with_coordinate_descent_hyperparameter_search(
    forecasts;
    forecast_hour_range = 1:10000,
    X_and_labels_to_inclusion_probabilities = nothing,
    model_prefix = "",
    get_feature_engineered_data = nothing,
    bin_split_forecast_sample_count = 100,
    max_iterations_without_improvement = 20,
    configs...
  )

  (grid, conus_grid_bitmask, train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_grid_conus_grid_bitmask_train_validation_test(forecasts, forecast_hour_range = forecast_hour_range)

  println("$(length(train_forecasts)) for training.")
  println("$(length(validation_forecasts)) for validation.")
  println("$(length(test_forecasts)) for testing.")


  # Returns (X_binned, labels, weights)
  get_data_labels_weights_binned(forecasts, bin_splits) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, forecasts, X_transformer = transformer, X_and_labels_to_inclusion_probabilities = X_and_labels_to_inclusion_probabilities)
  end

  # # Returns (X_binned_compressed, labels, weights)
  # get_data_labels_weights_binned_compressed(forecasts, bin_splits) = begin
  #   ys                = Vector{Float32}[]
  #   weightss          = Vector{Float32}[]
  #   binned_compressed = nothing
  #   for forcast_chunk in Iterators.partition(forecasts, 10)
  #     (X_chunk, labels, weights) = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, forcast_chunk, X_and_labels_to_inclusion_probabilities = X_and_labels_to_inclusion_probabilities)
  #
  #     binned_compressed = MemoryConstrainedTreeBoosting.bin_and_compress(X_chunk, bin_splits, prior_data = binned_compressed)
  #
  #     push!(ys, labels)
  #     push!(weightss, weights)
  #   end
  #
  #   (MemoryConstrainedTreeBoosting.finalize_loading(binned_compressed), vcat(ys...), vcat(weightss...))
  # end

  save(validation_loss, bin_splits, trees) = begin
    try
      mkdir("$(model_prefix)")
    catch
    end
    MemoryConstrainedTreeBoosting.save("$(model_prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
  end

  println("Preparing bin splits by sampling $bin_split_forecast_sample_count training forecasts")

  (bin_sample_X, _, _) = TrainingShared.get_data_labels_weights(grid, conus_grid_bitmask, get_feature_engineered_data, Iterators.take(Random.shuffle(train_forecasts), bin_split_forecast_sample_count))
  bin_splits           = MemoryConstrainedTreeBoosting.prepare_bin_splits(bin_sample_X)
  bin_sample_X         = nothing # freeeeeeee

  println("done.")


  println("Loading training data")
  X_binned, y, weights = get_data_labels_weights_binned(train_forecasts, bin_splits)
  println("done.")

  println("Loading validation data")
  validation_X_binned, validation_y, validation_weights = get_data_labels_weights_binned(validation_forecasts, bin_splits)
  println("done.")


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
        save(validation_loss, bin_splits, trees)
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
end

end # module TrainGBDTShared