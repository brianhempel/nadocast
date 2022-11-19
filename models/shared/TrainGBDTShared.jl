module TrainGBDTShared

import Mmap
import Random
import Serialization
using MPI

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

const Loss = MemoryConstrainedTreeBoosting.Loss

# brian@192.168.1.227:/home/brian/nadocast_dev:/home/brian/nadocast_dev/small_test/sref_f2-13_0.26_validation
function make_validation_server_callback(validation_server, event_name; max_iterations_without_improvement, mpi_comm = nothing)
  host, nadocast_dir, data_dir = split(validation_server, ":")
  server_command = `ssh $(host) "source ~/.profile; julia --threads auto --project=$nadocast_dir $nadocast_dir/models/shared/ValidationServer.jl $data_dir $event_name"`

  is_mpi_root(mpi_comm) = isnothing(mpi_comm) || MPI.Comm_rank(mpi_comm) == 0

  best_loss = Loss(Inf)
  iterations_without_improvement = 0

  if is_mpi_root(mpi_comm)
    server = open(server_command, "r+")
  end

  n_sent = 0

  iteration_callback(trees) = begin
    if is_mpi_root(mpi_comm)
      while n_sent < length(trees)
        Serialization.serialize(server, trees[n_sent+1])
        n_sent += 1
        validation_loss = Serialization.deserialize(server) :: Loss
      end
    end
    if !isnothing(mpi_comm)
      validation_loss = MPI.bcast(validation_loss, 0, mpi_comm)
    end

    if validation_loss < best_loss
      best_loss                      = validation_loss
      iterations_without_improvement = 0
    else
      iterations_without_improvement += 1
      if iterations_without_improvement >= max_iterations_without_improvement
        resize!(trees, length(trees) - max_iterations_without_improvement)
        throw(MemoryConstrainedTreeBoosting.EarlyStop())
      end
    end
    print("\rBest validation loss: $best_loss    ")

    validation_loss
  end

  iteration_callback
end


function train_with_coordinate_descent_hyperparameter_search(
    forecasts;
    forecast_hour_range = 1:10000,
    event_types = nothing,
    just_hours_near_storm_events = true,

    data_subset_ratio = 1.0,
    near_storm_ratio  = 1.0,
    event_name_to_labeler       = TrainingShared.event_name_to_labeler,
    compute_is_near_storm_event = TrainingShared.compute_is_near_storm_event,

    load_only = false,
    must_load_from_disk = false, # e.g. on a machine with no forecasts
    use_mpi = false, # must_load_from_disk must be true for distributed learning
    validation_server = nothing,
    model_prefix = "",
    prior_predictor = nothing,
    bin_split_forecast_sample_count = 100, # If not evenly divisible by the number of label types, a bit fewer might be used

    bin_splits_calc_inclusion_probabilities = (forecast, label_layers) -> max.(0.01f0, map(y -> y .> 0f0, label_layers)...),
    max_iterations_without_improvement = 20,
    save_dir,
    only_features = nothing, # nothing or ["CAPE:90-0 mb above ground:hour fcst:wt ens mean -1hr:", "CAPE:180-0 mb above ground:hour fcst:wt ens mean -1hr:", ...], can vary this without reloading the data
    only_before = Dates.DateTime(2099,1,1,12), # can vary this without reloading the data
    configs...
  )


  specific_save_dir(suffix) = save_dir * "_" * suffix

  if !must_load_from_disk
    (train_forecasts, validation_forecasts, test_forecasts) =
      TrainingShared.forecasts_train_validation_test(forecasts, forecast_hour_range = forecast_hour_range, just_hours_near_storm_events = just_hours_near_storm_events)

    forecasts_stats_str(forecasts) = begin
      sorted = sort(forecasts, alg=MergeSort, by=(fc -> (Forecasts.run_utc_datetime(fc), Forecasts.valid_utc_datetime(fc))))
      "\tfrom $(Forecasts.time_title(first(sorted)))\tto\t$(Forecasts.time_title(last(sorted)))"
    end

    println("$(length(train_forecasts)) for training, $(forecasts_stats_str(train_forecasts)).")
    train_forecasts_that_have_each_label = Dict{String, Vector{Forecasts.Forecast}}()
    for (name, forecast_has_event) in TrainingShared.event_name_to_forecast_predicate
      forecasts_with_event = filter(forecast_has_event, train_forecasts)
      train_forecasts_that_have_each_label[name] = forecasts_with_event
      println("  ($(length(forecasts_with_event)) with $name, $(forecasts_stats_str(forecasts_with_event)))")
    end
    label_type_count = length(train_forecasts_that_have_each_label)
    println("$(length(validation_forecasts)) for validation, $(forecasts_stats_str(validation_forecasts)).")
    println("$(length(test_forecasts)) for testing, $(forecasts_stats_str(test_forecasts)).")
  end

  bin_splits_path = joinpath(specific_save_dir("samples_for_bin_splits"), "bin_splits")
  bin_splits =
    if isfile(bin_splits_path) || must_load_from_disk
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


  # Probability of 1 if any labeler considers it a non-zero gridpoint
  # Probability of near_storm_ratio if within 100mi or 90min of a storm event
  # Probability of data_subset_ratio otherwise
  calc_inclusion_probabilities(forecast, label_layers) = begin
    is_near_storm_event_layer = compute_is_near_storm_event(forecast)
    max.(data_subset_ratio, near_storm_ratio .* is_near_storm_event_layer, map(y -> y .> 0f0, label_layers)...)
  end

  # Returns (X_binned, Ys, weights)
  get_data_labels_weights_binned(forecasts, save_suffix; only_features, only_before) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.get_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, event_name_to_labeler = event_name_to_labeler, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor, only_features = only_features, only_before = only_before)
  end

  prepare_data_labels_weights_binned(forecasts, save_suffix) = begin
    transformer(X) = MemoryConstrainedTreeBoosting.apply_bins(X, bin_splits)
    TrainingShared.prepare_data_labels_weights(forecasts, X_transformer = transformer, calc_inclusion_probabilities = calc_inclusion_probabilities, event_name_to_labeler = event_name_to_labeler, save_dir = specific_save_dir(save_suffix), prior_predictor = prior_predictor)
  end

  if use_mpi
    if must_load_from_disk
      MPI.Init()
      mpi_comm   = MPI.COMM_WORLD # The communication group for all processes
      root       = 0
      rank       = MPI.Comm_rank(mpi_comm) # Zero-indexed
      rank_count = MPI.Comm_size(mpi_comm) # Number of processes in cluster
    else
      println(stderr, "must_load_from_disk must be true if you want to distribute learning with MPI. (the initial dataset generation does not support MPI)")
      exit(1)
    end
  else
    mpi_comm = nothing
    root       = 0
    rank       = 0
    rank_count = 1
  end

  # If we are barely over memory, preferentially page out the validation data since only a fraction of it is used.
  # pageout_hint(arr) = Sys.islinux() ? (try Mmap.madvise!(arr, Mmap.MADV_COLD) catch end) : nothing
  pageout_hint(arr) = nothing

  if !load_only
    if only_before < Dates.now()
      println("Ignoring data after $only_before")
    end

    if isnothing(validation_server)
      # Load first so it page out first
      rank == root && println("Loading validation data")
      validation_X_binned, validation_Ys, validation_weights =
        if must_load_from_disk
          TrainingShared.read_data_labels_weights_from_disk(specific_save_dir("validation"); chunk_i = rank+1, chunk_count = rank_count, only_features = only_features, only_before = only_before)
        else
          get_data_labels_weights_binned(validation_forecasts, "validation", only_features = only_features, only_before = only_before)
        end
      print("done. $(size(validation_X_binned,1)) datapoints with $(size(validation_X_binned,2)) features each.\n")

      event_types = isnothing(event_types) ? collect(keys(validation_Ys)) : event_types

      pageout_hint(validation_X_binned)
      for event_name in keys(validation_Ys)
        if event_name != event_types[1]
          pageout_hint(validation_Ys[event_name])
        end
      end
    end

    rank == root && println("Loading training data")
    X_binned, Ys, weights =
      if must_load_from_disk
        TrainingShared.read_data_labels_weights_from_disk(specific_save_dir("training"); chunk_i = rank+1, chunk_count = rank_count, only_features = only_features, only_before = only_before)
      else
        get_data_labels_weights_binned(train_forecasts, "training", only_features = only_features, only_before = only_before)
      end
    print("done. $(size(X_binned,1)) datapoints with $(size(X_binned,2)) features each.\n")

    event_types = isnothing(event_types) ? collect(keys(Ys)) : event_types

    for event_name in keys(Ys)
      if event_name != event_types[1]
        pageout_hint(Ys[event_name])
      end
    end
  else
    println("Loading training data")
    data_count, feature_count = prepare_data_labels_weights_binned(train_forecasts, "training")
    println("done. $(data_count) datapoints with $(feature_count) features each.")

    println("Loading validation data")
    validation_data_count, validation_feature_count = prepare_data_labels_weights_binned(validation_forecasts, "validation")
    println("done. $(validation_data_count) datapoints with $(validation_feature_count) features each.")

    exit(0)
  end

  feature_file_path = joinpath(specific_save_dir("training"), "features.txt")
  feature_names = readlines(feature_file_path)

  feature_i_to_orig_feature_i =
    if isnothing(only_features)
      1:length(feature_names)
    else
      map(feat_name -> findfirst(isequal(feat_name), feature_names), only_features)
    end

  function restore_orig_feature_is(tree) :: MemoryConstrainedTreeBoosting.Tree
    if isa(tree, MemoryConstrainedTreeBoosting.Node)
      left  = restore_orig_feature_is(tree.left)
      right = restore_orig_feature_is(tree.right)
      MemoryConstrainedTreeBoosting.Node(feature_i_to_orig_feature_i[tree.feature_i], tree.split_i, left, right, [])
    else
      MemoryConstrainedTreeBoosting.Leaf(tree.Δscore)
    end
  end

  print_mpi(str) = rank == root && print(str)

  for event_name in event_types
    labels = Ys[event_name]
    if isnothing(validation_server)
      validation_labels = validation_Ys[event_name]
    end

    print("Training for $event_name with $(sum(labels)) positive and $(sum(1f0 .- labels)) negative labels\n")

    if !isnothing(only_features)
      for i in eachindex(only_features)
        print_mpi("Using $i $(only_features[i])\n")
      end
    end

    best_model_path = nothing
    best_loss       = Inf32
    prefix =
      if isnothing(only_features)
        "$(model_prefix)_$(event_name)"
      else
        "$(model_prefix)_$(event_name)_only_$(length(only_features))_features_$(hash(sort(unique(only_features))))"
      end

    # Returns path
    save(validation_loss, bin_splits, trees) = begin
      print("\tsaving...")
      try
        mkdir(prefix)
      catch
      end
      trees = map(restore_orig_feature_is, trees) # deep copy
      model_path = MemoryConstrainedTreeBoosting.save("$(prefix)/$(length(trees))_trees_loss_$(validation_loss).model", bin_splits, trees)
      print("done.\t")
      model_path
    end

    try_config(; config...) = begin

      isnothing(validation_server) && pageout_hint(validation_X_binned)

      best_loss_for_config = Inf32
      best_trees_for_config = nothing

      callback_to_track_validation_loss =
        if isnothing(validation_server)
          MemoryConstrainedTreeBoosting.make_callback_to_track_validation_loss(
              validation_X_binned, validation_labels;
              validation_weights                 = validation_weights,
              max_iterations_without_improvement = max_iterations_without_improvement,
              mpi_comm                           = mpi_comm
            )
        else
          make_validation_server_callback(
            validation_server, event_name,
            max_iterations_without_improvement = max_iterations_without_improvement,
            mpi_comm                           = mpi_comm
          )
        end

      iteration_callback(trees) = begin
        validation_loss = callback_to_track_validation_loss(trees)

        isnothing(validation_server) && pageout_hint(validation_X_binned) # unlikely to use same features twice in a row

        if validation_loss < best_loss_for_config
          best_loss_for_config  = validation_loss
          best_trees_for_config = map(MemoryConstrainedTreeBoosting.strip_tree_training_info, trees) # deep copy
        end
      end

      MemoryConstrainedTreeBoosting.train_on_binned(
        X_binned, labels;
        weights            = weights,
        iteration_count    = Int64(round(50 / config[:learning_rate])),
        iteration_callback = iteration_callback,
        mpi_comm           = mpi_comm,
        config...
      )

      if best_loss_for_config < best_loss
        rank == root && (best_model_path = save(best_loss_for_config, bin_splits, best_trees_for_config))
        best_loss = best_loss_for_config

        # validation_scores = MemoryConstrainedTreeBoosting.apply_trees(validation_X_binned, trees)
        # write("$(prefix)/$(length(trees))_trees_loss_$(validation_loss).validation_scores", validation_scores)
        # write("$(prefix)/validation_labels", validation_y)
        # write("$(prefix)/validation_weights", validation_weights)
      end

      best_loss_for_config
    end

    TrainingShared.coordinate_descent_hyperparameter_search(try_config; print = print_mpi, configs...)

    !isnothing(mpi_comm) && MPI.Barrier(mpi_comm)
    print_mpi("")
    print_mpi("$best_model_path\n")
    print_mpi("")
    !isnothing(mpi_comm) && MPI.Barrier(mpi_comm)

    pageout_hint(labels)
    isnothing(validation_server) && pageout_hint(validation_labels)
  end

  ()
end

end # module TrainGBDTShared
