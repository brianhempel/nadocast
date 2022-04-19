import Mmap
using Serialization
using MemoryConstrainedTreeBoosting

push!(LOAD_PATH, @__DIR__)
import TrainingShared

save_dir, event_name = ARGS


function read_data_labels_weights_from_disk()
  save_path(path) = joinpath(save_dir, path)

  weights = deserialize(save_path("weights.serialized"))

  data_file_names = sort(filter(file_name -> startswith(file_name, "data_"), readdir(save_dir)), by=(name -> parse(Int64, split(name, r"_|\.|-")[2])))

  data_count    = length(weights)
  feature_count = parse.(Int64, match(r"data_\d+_\d+x(\d+).serialized", data_file_names[1]))

  mmap_path = save_path("data.mmap")
  mmap_io =
    if isfile(mmap_path)
      # Already written.
      open(mmap_path)
    else
      open(mmap_path, "w+")
    end

  data = Mmap.mmap(mmap_io, Array{UInt8,2}, (data_count, feature_count); grow = true)

  if iswritable(mmap_io)
    row_i = 1

    for data_file_name in data_file_names
      forecast_row_count, forecast_feature_count = parse.(Int64, match(r"data_\d+_(\d+)x(\d+).serialized", data_file_name))

      @assert feature_count == forecast_feature_count

      forecast_data = deserialize(save_path(data_file_name))

      @assert forecast_row_count     = size(forecast_data, 1)
      @assert forecast_feature_count = size(forecast_data, 2)

      data[row_i:(row_i + forecast_row_count - 1), :] = forecast_data

      row_i += forecast_row_count
    end

    @assert row_i - 1 == data_count

    Mmap.sync!(data)
  end

  y = deserialize(save_path("labels-$event_name.serialized"))

  (data, y, weights)
end


function main()
  validation_X_binned, validation_y, validation_weights = read_data_labels_weights_from_disk()

  validation_scores = nothing

  trees = []
  while true
    new_tree =
      try
        deserialize(stdin) :: MemoryConstrainedTreeBoosting.Tree
      catch e
        isa(e, EOFError) ? exit(0) : rethrow()
      end

    if isnothing(validation_scores)
      validation_scores = predict_on_binned(validation_X_binned, [new_tree], output_raw_scores = true)
    else
      MemoryConstrainedTreeBoosting.apply_tree!(validation_X_binned, new_tree, validation_scores)
    end
    validation_loss = MemoryConstrainedTreeBoosting.compute_mean_logloss(validation_y, validation_scores; weights = validation_weights)

    serialize(stdout, validation_loss)
    flush(stdout)
  end
end

main()