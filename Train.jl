# Train a model
#
#
#
# $ julia Train.jl tor_days_only_model/logistic.jl
#
# Resume from a checkpoint:
#
# $ julia Train.jl tor_days_only_model/logistic.jl

if length(ARGS) == 0
  # error("Must provide a model script in a directory that includes a train.txt file containing a list of paths")
  error("Must provide a model script in a directory that includes a train.bindata file created by e.g. MakeTordnadoNeighborhoodsData.jl")
end

model_dir_and_type = ARGS[1]

model_dir    = dirname(model_dir_and_type)
model_script = basename(model_dir_and_type)
model_type   = replace(model_script, ".jl", "")

println("Loading Flux...")
using Flux
using Flux.Tracker
using BSON: @save, @load

# push!(headers, "lat")
# push!(headers, "lon")
# push!(headers, "lat_motion")
# push!(headers, "lon_motion")
# push!(headers, "plus 30 mins lat")
# push!(headers, "plus 30 mins lon")
# push!(headers, "minus 30 mins lat")
# push!(headers, "minus 30 mins lon")
# push!(headers, "tornado within 25mi")
# push!(headers, "training weight")

include("ReadFeatureData.jl")

feature_normalizers = include("NormalizingFactors.jl")

DO_BACKPROP = true # possibly overwritted in model script

include("$model_dir/$model_script")

epoch_n = 1

# if length(ARGS) == 2
#   saved_bson_path = ARGS[2]
#
#   println("Loading $saved_bson_path...")
#
#   @load saved_bson_path model optimizer
#
#   m = match(r"epoch_(\d+)", saved_bson_path)
#   if m !== nothing
#     epoch_n = Int64(m.captures[1])
#   end
#
#   println("Resuming training of $model_type $model_dir from $saved_bson_path...")
# else
#   println("Beginning training of $model_type $model_dir...")
# end

train_file_path    = "$model_dir/train-shuffled.binfeatures"
train_labels_path  = "$model_dir/train-shuffled.binlabels"
train_weights_path = "$model_dir/train-shuffled.binweights"
dev_file_path      = "$model_dir/dev2017+1.binfeatures"
dev_labels_path    = "$model_dir/dev2017+1.binlabels"
dev_weights_path   = "$model_dir/dev2017+1.binweights"

# files_to_process_at_a_time = 7
#
# files_to_process = []
#
# in_flight = []
#
# function start_grabbing_more_files()
#   for _ = 1:files_to_process_at_a_time
#     if length(files_to_process) > 0
#       file = pop!(files_to_process)
#       # println(file)
#       # println(files_to_process)
#       push!(in_flight, open(`julia read_grib.jl $file`))
#       # println(in_flight)
#     end
#   end
# end

# function get_more_data()
#   data = zeros(Float32, 1544, 0)
#
#   while length(in_flight) > 0
#     data_stream, proc = pop!(in_flight)
#     # success = wait(proc)
#     tmp_data_path = last(split(strip(String(read(data_stream))),"\n")) # last line is the tmp file with the data
#
#     # If reading grib file okay...
#     if endswith(tmp_data_path, ".bindata")
#       # println(tmp_data_path)
#       data_chunk = read(tmp_data_path, Float32, (1544,40870)) :: Array{Float32, 2}
#       open(`rm $tmp_data_path`)
#       data = hcat(data, data_chunk)
#       if any(isnan, data_chunk)
#         error("contains nans")
#       end
#       if any(isinf, data_chunk)
#         error("contains infs")
#       end
#     end
#     # println(in_flight)
#   end
#
#   data
# end

function save_model(blurb, pt_count, training_loss)
  time_str = replace("$(now())", ":", "-")
  save_path = "$(model_dir)/$(model_type)_$(time_str)_$(pt_count)_examples_$(blurb)_loss_$(training_loss).bson"

  @save save_path model optimizer

  println("Saved as $save_path")
end


# train_data = open(train_file_path, "r", )


train_data, train_data_file, train_point_count = open_data_file(train_file_path)

train_labels  = read(train_labels_path, Float32, train_point_count)
train_weights = read(train_weights_path, Float32, train_point_count)

dev_data,   dev_data_file,   dev_point_count   = open_data_file(dev_file_path)

dev_labels  = read(dev_labels_path, Float32, dev_point_count)
dev_weights = read(dev_weights_path, Float32, dev_point_count)


println("$train_point_count training points.")

pt_count            = 0 :: Int64

checkpoint_loss                  = 0.0
checkpoint_pt_count              = 0 :: Int64
last_checkpoint_time             = now()
checkpoint_interval              = 0.1 # Minutes
checkpoint_interval_growth_ratio = 1.5
checkpoint_interval_max          = 60.0 * 24.0


if isdefined(:pretrain)
  pretrain(train_data, train_labels, train_weights)
end

Flux.testmode!(model, false)


while true
  println("Epoch $epoch_n")

  epoch_loss     = 0.0
  epoch_pt_count = 0 :: Int64

  if isdefined(:trim_model_before_epoch)
    trim_model_before_epoch(model)
  end

  # files_to_process = train_files[shuffle(1:length(train_files))]
  #
  # start_grabbing_more_files()
  #
  # while length(files_to_process) > 0
  #   data = get_more_data()
  #   start_grabbing_more_files()
  #
  #   files_loss = 0.0
  #   pt_count   = 0
  #
  #   tor_prob     = 0.0
  #   tor_pt_count = 0
  #
  #   non_tor_prob     = 0.0
  #   non_tor_pt_count = 0

    # Shuffle points
    for i in 1:train_point_count
      example_weight = train_weights[i]
      if rand() <= example_weight
        expected = train_labels[i]

        if USE_NORMALIZING_FACTORS
          x = (@view train_data[:,i]) ./ feature_normalizers
        else
          x = @view train_data[:,i]
        end

        predicted = model(x)
        # show(predicted)
        loss      = loss_func(predicted, Float64(expected))
        # println(x)
        # print("$(maximum(abs.(x))), $(Flux.Tracker.data(predicted[1])), $expected, $(Flux.Tracker.data(loss))   ")
        # print("$(Flux.Tracker.data(loss)) ")
        # show(loss)
        # print("$(Flux.Tracker.data(loss)) ")
        # if expected > 0.5
        #   tor_prob     += Flux.Tracker.data(predicted[1])
        #   tor_pt_count += 1
        # else
        #   non_tor_prob     += Flux.Tracker.data(predicted[1])
        #   non_tor_pt_count += 1
        # end
        # files_loss += Flux.Tracker.data(loss)
        pt_count            += 1
        epoch_pt_count      += 1
        checkpoint_pt_count += 1
        epoch_loss          += Flux.Tracker.data(loss)
        checkpoint_loss     += Flux.Tracker.data(loss)
        if DO_BACKPROP
          back!(loss)
          optimizer()
        end

        if isdefined(:update_model)
          update_model(model, x, expected, loss)
        end

        if now() - last_checkpoint_time > Dates.Second(round(checkpoint_interval*60))
          checkpoint_pt_loss = checkpoint_loss / checkpoint_pt_count
          # println("Checkpoint loss $checkpoint_pt_loss")
          show_extra_training_info()

          save_model("checkpoint", pt_count, checkpoint_pt_loss)

          checkpoint_loss     = 0.0
          checkpoint_pt_count = 0 :: Int64
          checkpoint_interval = min(checkpoint_interval*checkpoint_interval_growth_ratio, checkpoint_interval_max)

          last_checkpoint_time = now()
        end
      end
    end
    # epoch_loss          += files_loss
    # epoch_pt_count      += pt_count
    # checkpoint_loss     += files_loss
    # checkpoint_pt_count += pt_count

    # println("Files loss $(files_loss/pt_count)")
    # println("Mean prob in tornado areas $(tor_prob/tor_pt_count)")
    # println("Mean prob in non-tornado areas $(non_tor_prob/non_tor_pt_count)")
    # prob_ratio = (tor_prob/tor_pt_count)/(non_tor_prob/non_tor_pt_count)
    # println("Prob ratio $prob_ratio")
  # end

  println("Epoch $epoch_n training loss $(epoch_loss/epoch_pt_count)")

  dev_weight_count = 0.0
  dev_loss         = 0.0

  Flux.testmode!(model, true)

  for i in 1:dev_point_count
    example_weight = dev_weights[i]
    expected       = dev_labels[i]
    x              = (@view dev_data[:,i]) ./ feature_normalizers
    predicted      = model(x)
    loss           = loss_func(predicted, Float64(expected))

    dev_weight_count += Float64(example_weight)
    dev_loss         += Flux.Tracker.data(loss)*Float64(example_weight)
  end

  Flux.testmode!(model, false)

  println("Epoch $(epoch_n) dev loss: $(dev_loss/dev_weight_count)")
  # The multiplier for bad loss to correct for when denominator was point count instead of weight: 1.06059

  show_extra_training_info()
  save_model("epoch_$(epoch_n)", pt_count, dev_loss/dev_weight_count)

  epoch_n += 1
end

close(train_data_file)
close(dev_data_file)