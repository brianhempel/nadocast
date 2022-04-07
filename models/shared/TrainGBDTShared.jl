module TrainGBDTShared

import Random
import Serialization

import MemoryConstrainedTreeBoosting

push!(LOAD_PATH, @__DIR__)
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts


# If we had infinite memory, we could train on all datapoints with all the forecast hours.
# However, we don't.
# Instead we include each datapoint with some probability, and
# multiply the weight of that point by the inverse of that probability (a point included with
# probability 0.1 has its weight multiplied by 10).
#
# Positively labeled points are always included with probability 1.


function train_with_coordinate_descent_hyperparameter_search(
    forecasts;
    forecast_hour_range = 1:10000,
    only_events_of_type = nothing,

    data_subset_ratio = data_subset_ratio,
    near_storm_ratio  = near_storm_ratio,

    load_only = false,
    model_prefix = "",
    prior_predictor = nothing,
    bin_split_forecast_sample_count = 100, # If not evenly divisible by the number of label types, a bit fewer might be used

    bin_splits_calc_inclusion_probabilities = (forecast, label_layers) -> max.(0.01f0, label_layers...),
    max_iterations_without_improvement = 20,
    save_dir,
    configs...
  )

  event_name_to_labeler = TrainingShared.event_name_to_labeler

  specific_save_dir(suffix) = save_dir * "_" * suffix

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(forecasts, forecast_hour_range = forecast_hour_range)

  println("$(length(train_forecasts)) for training.")
  train_forecasts_that_have_each_label = Dict{String, Vector{Forecasts.Forecast}}()
  for (name, forecast_has_event) in TrainingShared.event_name_to_forecast_predicate
    forecasts_with_event = filter(forecast_has_event, train_forecasts)
    train_forecasts_that_have_each_label[name] = forecasts_with_event
    println("  ($(length(forecasts_with_event)) with $name)")
  end
  label_type_count = length(train_forecasts_that_have_each_label)
  println("$(length(validation_forecasts)) for validation.")
  println("$(length(test_forecasts)) for testing.")

  bin_splits_path = joinpath(specific_save_dir("samples_for_bin_splits"), "bin_splits")
  bin_splits =
    if isfile(bin_splits_path)
      println("Loading previously computed bin splits from $bin_splits_path")
      bin_splits, _ = MemoryConstrainedTreeBoosting.load(bin_splits_path)
      bin_splits
    else
      # Deterministic randomness for bin splitting
      # So we choose the same bin splits every time, given the same forecasts
      rng = Random.MersenneTwister(123456)

      # Use forecasts with all different labeling (same number of each)

      sample_forecasts =
        reduce(vcat,
          map(values(train_forecasts_that_have_each_label)) do forecasts_with_event_type
            forecasts_per_type = div(bin_split_forecast_sample_count, label_type_count)
            collect(Iterators.take(Random.shuffle(rng, forecasts_with_event_type), forecasts_per_type))
          end
        )

      println("Preparing bin splits by sampling $(length(sample_forecasts)) training forecasts with events")

      (sample_X, sample_Ys, _) =
        TrainingShared.get_data_labels_weights(
          sample_forecasts,
          event_name_to_labeler = event_name_to_labeler,
          calc_inclusion_probabilities = bin_splits_calc_inclusion_probabilities,
          save_dir = specific_save_dir("samples_for_bin_splits")
        )

      # Balance the number of each kind of label, as well as negative data

      positive_indices_arrays = map(Y -> Random.shuffle(rng, findall(Y .>  0.5f0)), values(sample_Ys))
      negative_indices_arrays = map(Y -> Random.shuffle(rng, findall(Y .<= 0.5f0)), values(sample_Ys))

      indices_arrays = vcat(positive_indices_arrays, negative_indices_arrays)

      smallest_labelset_size = minimum(map(length, indices_arrays))

      indicies_to_sample = unique(sort(reduce(vcat, # Flatten, then sort in vain hopes of gaining cache locality.
        map(indices_arrays) do indices
          collect(Iterators.take(indices, smallest_labelset_size))
        end
      )))

      print("sampling $(length(indicies_to_sample)) datapoints...")

      sample_X = sample_X[indicies_to_sample, :]

      print("computing bin splits...")
      bin_splits = MemoryConstrainedTreeBoosting.prepare_bin_splits(sample_X)
      MemoryConstrainedTreeBoosting.save(bin_splits_path, bin_splits, [])

      # free memory
      sample_X                = nothing
      sample_Ys               = nothing
      positive_indices_arrays = nothing
      negative_indices_arrays = nothing
      indicies_to_sample      = nothing
      indices_arrays          = nothing

      println("done.")

      bin_splits
    end


  # Probability of 1 if any labeler considers it a positive gridpoint
  # Probability of near_storm_ratio if within 100mi or 90min of a storm event
  # Probability of data_subset_ratio otherwise
  calc_inclusion_probabilities(forecast, label_layers) = begin
    is_near_storm_event_layer = TrainingShared.compute_is_near_storm_event(forecast)
    max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event_layer, label_layers...)
  end

  # Returns (X_binned, Ys, weights)
  get_data_labels_weights_binned(forecasts, save_suffix) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.get_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, event_name_to_labeler = event_name_to_labeler, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor)
  end

  prepare_data_labels_weights_binned(forecasts, save_suffix) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.prepare_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, event_name_to_labeler = event_name_to_labeler, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor)
  end

  if !load_only
    println("Loading training data")
    X_binned, Ys, weights = get_data_labels_weights_binned(train_forecasts, "training")
    println("done. $(size(X_binned,1)) datapoints with $(size(X_binned,2)) features each.")

    println("Loading validation data")
    validation_X_binned, validation_Ys, validation_weights = get_data_labels_weights_binned(validation_forecasts, "validation")
    println("done. $(size(validation_X_binned,1)) datapoints with $(size(validation_X_binned,2)) features each.")
  else
    println("Loading training data")
    data_count, feature_count = prepare_data_labels_weights_binned(train_forecasts, "training")
    println("done. $(data_count) datapoints with $(feature_count) features each.")

    println("Loading validation data")
    validation_data_count, validation_feature_count = prepare_data_labels_weights_binned(validation_forecasts, "validation")
    println("done. $(validation_data_count) datapoints with $(validation_feature_count) features each.")

    exit(0)
  end


  for (event_name, labels) in Ys
    if !isnothing(only_events_of_type) && only_events_of_type != event_name
      continue
    end

    println("Training for $event_name with $(count(labels .> 0.5f0)) positive and $(count(labels .<= 0.5f0)) negative labels")

    best_model_path = nothing
    best_loss       = Inf32
    prefix          = "$(model_prefix)_$(event_name)"

    # Returns path
    save(validation_loss, bin_splits, trees) = begin
      try
        mkdir(prefix)
      catch
      end
      MemoryConstrainedTreeBoosting.save("$(prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
    end

    try_config(; config...) = begin

      best_loss_for_config = Inf32

      validation_labels = validation_Ys[event_name]
      callback_to_track_validation_loss =
        MemoryConstrainedTreeBoosting.make_callback_to_track_validation_loss(
            validation_X_binned, validation_labels;
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
          best_loss       = validation_loss

          # validation_scores = MemoryConstrainedTreeBoosting.apply_trees(validation_X_binned, trees)
          # write("$(prefix)/$(length(trees))_trees_loss_$(validation_loss).validation_scores", validation_scores)
          # write("$(prefix)/validation_labels", validation_y)
          # write("$(prefix)/validation_weights", validation_weights)

        end
      end

      MemoryConstrainedTreeBoosting.train_on_binned(
        X_binned, labels;
        weights            = weights,
        iteration_count    = Int64(round(50 / config[:learning_rate])),
        iteration_callback = iteration_callback,
        config...
      )

      best_loss_for_config
    end

    TrainingShared.coordinate_descent_hyperparameter_search(try_config; configs...)

    println()
    println(best_model_path)
    println()
  end

  ()
end

end # module TrainGBDTShared