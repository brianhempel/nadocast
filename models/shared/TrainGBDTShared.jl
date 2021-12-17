module TrainGBDTShared

import Random
import Serialization

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, @__DIR__)
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


# # If we had infinite memory, we could train on all datapoints with all the forecast hours.
# # However, we don't.
# # Instead we include each datapoint with some probability (basal_inclusion_probability), and
# # multiply the weight of that point by the inverse of that probability (a point included with
# # probability 0.1 has its weight multiplied by 10).
# #
# # Positively labeled points are always included with probability 1.
# #
# # This is a crude mechanism for subsetting and weighting, so we train in multiple rounds
# # of "annealing", increasing the resolution in areas more likely to contain positive
# # labels (tornadoes). We use the most recently trained model and use the formula
# # max(basal_inclusion_probability, prediction_from_last_model*prediction_inclusion_multiplier)
# # as the inclusion probability for negatively labeled points. (But validation data uses only
# # basal_inclusion_probability for inclusion probability of negatively labeled points.)
# function train_multiple_annealing_rounds_with_coordinate_descent_hyperparameter_search(forecasts; annealing_rounds = 3, basal_inclusion_probability :: Float32, prediction_inclusion_multiplier :: Float32, validation_inclusion_probability :: Float32, config...)
#
#   X_and_labels_to_validation_inclusion_probabilities(labels, is_near_storm_event) =
#     map(1:length(labels)) do i
#       max(validation_inclusion_probability, labels[i])
#     end
#
#   last_model_path = nothing
#
#   for annealing_round_i in 1:annealing_rounds
#     println("\nAnnealing round $annealing_round_i of $annealing_rounds...\n")
#
#     if isnothing(last_model_path)
#       X_and_labels_to_basal_inclusion_probabilities(labels, is_near_storm_event) =
#         map(1:length(labels)) do i
#           max(basal_inclusion_probability, labels[i])
#         end
#
#       last_model_path = train_with_coordinate_descent_hyperparameter_search(
#           forecasts;
#           training_calc_inclusion_probabilities   = X_and_labels_to_basal_inclusion_probabilities,
#           validation_calc_inclusion_probabilities = X_and_labels_to_validation_inclusion_probabilities,
#           config...
#         )
#     else
#       last_model_bin_splits, last_model_trees = MemoryConstrainedTreeBoosting.load(last_model_path)
#
#       X_and_labels_to_refined_inclusion_probabilities(labels, is_near_storm_event) = begin
#         predictions = MemoryConstrainedTreeBoosting.predict(X, last_model_bin_splits, last_model_trees)
#
#         map(1:length(labels)) do i
#           max(basal_inclusion_probability, predictions[i]*prediction_inclusion_multiplier, labels[i])
#         end
#       end
#
#       last_model_path = train_with_coordinate_descent_hyperparameter_search(
#           forecasts;
#           training_calc_inclusion_probabilities   = X_and_labels_to_refined_inclusion_probabilities,
#           validation_calc_inclusion_probabilities = X_and_labels_to_validation_inclusion_probabilities,
#           config...
#         )
#     end
#   end
#
#   last_model_path
# end

# Returns path to best trained model
function train_with_coordinate_descent_hyperparameter_search(
    forecasts;
    forecast_hour_range = 1:10000,
    compute_forecast_labels = TrainingShared.compute_forecast_labels_ef0,
    training_calc_inclusion_probabilities   = nothing,
    validation_calc_inclusion_probabilities = nothing,
    load_only = false,
    model_prefix = "",
    prior_predictor = nothing,
    bin_split_forecast_sample_count = 100,
    balance_labels_when_computing_bin_splits = true,
    max_iterations_without_improvement = 20,
    save_dir,
    configs...
  )

  specific_save_dir(suffix) = save_dir * "_" * suffix

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(forecasts, forecast_hour_range = forecast_hour_range)

  train_forecasts_with_tornadoes = filter(TrainingShared.forecast_is_tornado_hour, train_forecasts)

  # for forecast in train_forecasts_with_tornadoes
  #   println(Forecasts.valid_utc_datetime(forecast))
  # end

  println("$(length(train_forecasts)) for training. ($(length(train_forecasts_with_tornadoes)) with tornadoes.)")
  println("$(length(validation_forecasts)) for validation.")
  println("$(length(test_forecasts)) for testing.")


  # Returns (X_binned, labels, weights)
  get_data_labels_weights_binned(forecasts, bin_splits, calc_inclusion_probabilities, save_suffix; prior_predictor = nothing) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.get_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, compute_forecast_labels = compute_forecast_labels, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor)
  end

  prepare_data_labels_weights_binned(forecasts, bin_splits, calc_inclusion_probabilities, save_suffix; prior_predictor = nothing) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.prepare_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, compute_forecast_labels = compute_forecast_labels, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor)
  end

  # Returns path
  save(validation_loss, bin_splits, trees) = begin
    try
      mkdir("$(model_prefix)")
    catch
    end
    MemoryConstrainedTreeBoosting.save("$(model_prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
  end

  bin_splits_path = joinpath(specific_save_dir("samples_for_bin_splits"), "bin_splits")
  bin_splits =
    if isfile(bin_splits_path)
      println("Loading previously computed bin splits from $bin_splits_path")
      bin_splits, _ = MemoryConstrainedTreeBoosting.load(bin_splits_path)
      bin_splits
    else
      println("Preparing bin splits by sampling $bin_split_forecast_sample_count training tornado hour forecasts")

      rng = Random.MersenneTwister(123456)
      (bin_sample_X, bin_sample_y, _) =
        TrainingShared.get_data_labels_weights(
          Iterators.take(Random.shuffle(rng, train_forecasts_with_tornadoes), bin_split_forecast_sample_count),
          compute_forecast_labels = compute_forecast_labels,
          calc_inclusion_probabilities = (forecast, labels) -> balance_labels_when_computing_bin_splits ? min.(max.(0.01f0, labels), training_calc_inclusion_probabilities(forecast, labels)) : ones(Float32, size(labels)),
          save_dir = specific_save_dir("samples_for_bin_splits")
        )
      if balance_labels_when_computing_bin_splits
        # Deterministic randomness for bin splitting
        # So we choose the same bin splits every time
        rng = Random.MersenneTwister(1234567)

        positive_indices = findall(bin_sample_y .>  0.5f0)
        negative_indices = findall(bin_sample_y .<= 0.5f0)
        print("filtering to balance $(length(positive_indices)) positive and $(length(negative_indices)) negative labels...")
        if length(positive_indices) > length(negative_indices)
          positive_indices = collect(Iterators.take(Random.shuffle(rng, positive_indices), length(negative_indices)))
        else
          negative_indices = collect(Iterators.take(Random.shuffle(rng, negative_indices), length(positive_indices)))
        end
        indicies_to_sample = sort(vcat(positive_indices, negative_indices)) # Vain hopes of gaining cache locality.

        bin_sample_X = bin_sample_X[indicies_to_sample, :]
      end
      print("computing bin splits...")
      bin_splits   = MemoryConstrainedTreeBoosting.prepare_bin_splits(bin_sample_X)
      MemoryConstrainedTreeBoosting.save(bin_splits_path, bin_splits, [])
      bin_sample_X = nothing # freeeeeeee
      bin_sample_y = nothing # freeeeeeee

      println("done.")

      bin_splits
    end


  if !load_only
    println("Loading training data")
    X_binned, y, weights = get_data_labels_weights_binned(train_forecasts, bin_splits, training_calc_inclusion_probabilities, "training"; prior_predictor = prior_predictor)
    println("done. $(size(X_binned,1)) datapoints with $(size(X_binned,2)) features each.")

    println("Loading validation data")
    validation_X_binned, validation_y, validation_weights = get_data_labels_weights_binned(validation_forecasts, bin_splits, validation_calc_inclusion_probabilities, "validation"; prior_predictor = prior_predictor)
    println("done. $(size(validation_X_binned,1)) datapoints with $(size(validation_X_binned,2)) features each.")
  else
    println("Loading training data")
    data_count, feature_count = prepare_data_labels_weights_binned(train_forecasts, bin_splits, training_calc_inclusion_probabilities, "training"; prior_predictor = prior_predictor)
    println("done. $(data_count) datapoints with $(feature_count) features each.")

    println("Loading validation data")
    validation_data_count, validation_feature_count = prepare_data_labels_weights_binned(validation_forecasts, bin_splits, validation_calc_inclusion_probabilities, "validation"; prior_predictor = prior_predictor)
    println("done. $(validation_data_count) datapoints with $(validation_feature_count) features each.")

    exit(0)
  end

  best_model_path = nothing
  best_loss = Inf32

  try_config(; config...) = begin

    best_loss_for_config = Inf32

    callback_to_track_validation_loss =
      MemoryConstrainedTreeBoosting.make_callback_to_track_validation_loss(
          validation_X_binned, validation_y;
          validation_weights                 = validation_weights,
          max_iterations_without_improvement = max_iterations_without_improvement
        )

    iteration_callback(trees) = begin
      validation_loss = callback_to_track_validation_loss(trees)

      if validation_loss < best_loss_for_config
        best_loss_for_config = validation_loss
      end

      if validation_loss < best_loss
        best_model_path = save(validation_loss, bin_splits, trees)

        validation_scores = MemoryConstrainedTreeBoosting.apply_trees(validation_X_binned, trees)
        write("$(model_prefix)/$(length(trees))_trees_loss_$(validation_loss).validation_scores", validation_scores)
        write("$(model_prefix)/validation_labels", validation_y)
        write("$(model_prefix)/validation_weights", validation_weights)

        best_loss = validation_loss
      end
    end

    MemoryConstrainedTreeBoosting.train_on_binned(
      X_binned, y;
      weights            = weights,
      iteration_count    = Int64(round(30 / config[:learning_rate])),
      iteration_callback = iteration_callback,
      config...
    )

    best_loss_for_config
  end

  TrainingShared.coordinate_descent_hyperparameter_search(try_config; configs...)

  best_model_path
end

end # module TrainGBDTShared