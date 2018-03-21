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
  error("Must provide a model script in a directory that includes a train.txt file containing a list of paths")
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

const FEATURE_COUNT = 1544-10 # First 10 "features" are actually not.

include("$model_dir/$model_script")

epoch_n = 1

if length(ARGS) == 2
  saved_bson_path = ARGS[2]

  println("Loading $saved_bson_path...")

  @load saved_bson_path model optimizer

  m = match(r"epoch_(\d+)", saved_bson_path)
  if m !== nothing
    epoch_n = Int64(m.captures[1])
  end

  println("Resuming training of $model_type $model_dir from $saved_bson_path...")
else
  println("Beginning training of $model_type $model_dir...")
end

train_files = split(strip(String(read("$model_dir/train.txt"))), "\n")
# dev_files   = split(strip(String(read("$model_dir/dev.txt"))), "\n")

files_to_process_at_a_time = 7

files_to_process = []

in_flight = []

function start_grabbing_more_files()
  for _ = 1:files_to_process_at_a_time
    if length(files_to_process) > 0
      file = pop!(files_to_process)
      # println(file)
      # println(files_to_process)
      push!(in_flight, open(`julia read_grib.jl $file`))
      # println(in_flight)
    end
  end
end

function get_more_data()
  data = zeros(Float32, 0, 1544)

  while length(in_flight) > 0
    data_stream, proc = pop!(in_flight)
    # success = wait(proc)
    tmp_data_path = last(split(strip(String(read(data_stream))),"\n")) # last line is the tmp file with the data

    # If reading grib file okay...
    if endswith(tmp_data_path, ".bindata")
      # println(tmp_data_path)
      data_chunk = read(tmp_data_path, Float32, (40870,1544)) :: Array{Float32, 2}
      open(`rm $tmp_data_path`)
      data = vcat(data, data_chunk)
      if any(isnan, data_chunk)
        error("contains nans")
      end
      if any(isinf, data_chunk)
        error("contains infs")
      end
    end
    # println(in_flight)
  end

  data
end

function save_model(blurb, training_loss)
  save_path = "$(model_dir)/$(model_type)_$(now())_$(blurb)_loss_$(training_loss).bson"

  @save save_path model optimizer

  println("Saved as $save_path")
end

feature_normalizers = include("NormalizingFactors.jl")

last_checkpoint_time = now()
checkpoint_loss     = 0.0
checkpoint_pt_count = 0 :: Int64

while true
  println("Epoch $epoch_n")

  epoch_loss     = 0.0
  epoch_pt_count = 0 :: Int64

  files_to_process = train_files[shuffle(1:length(train_files))]

  start_grabbing_more_files()

  while length(files_to_process) > 0
    data = get_more_data()
    start_grabbing_more_files()

    files_loss = 0.0
    pt_count   = 0

    tor_prob     = 0.0
    tor_pt_count = 0

    non_tor_prob     = 0.0
    non_tor_pt_count = 0

    # Shuffle rows
    for i in shuffle(1:size(data,1))
      example_weight = data[i,10]
      if rand() <= example_weight
        expected  = data[i,9]
        x         = data[i,11:size(data,2)] ./ feature_normalizers
        predicted = model(x)
        # show(predicted)
        loss      = loss_func(predicted, [Float64(expected)])
        # show(loss)
        # print("$(Flux.Tracker.data(loss)) ")
        if expected > 0.5
          tor_prob     += Flux.Tracker.data(predicted[1])
          tor_pt_count += 1
        else
          non_tor_prob     += Flux.Tracker.data(predicted[1])
          non_tor_pt_count += 1
        end
        files_loss += Flux.Tracker.data(loss)
        pt_count   += 1
        back!(loss)
        optimizer()
      end
    end
    epoch_loss          += files_loss
    epoch_pt_count      += pt_count
    checkpoint_loss     += files_loss
    checkpoint_pt_count += pt_count

    # println("Files loss $(files_loss/pt_count)")
    # println("Mean prob in tornado areas $(tor_prob/tor_pt_count)")
    # println("Mean prob in non-tornado areas $(non_tor_prob/non_tor_pt_count)")
    # prob_ratio = (tor_prob/tor_pt_count)/(non_tor_prob/non_tor_pt_count)
    # println("Prob ratio $prob_ratio")

    if now() - last_checkpoint_time > Dates.Minute(30)
      checkpoint_pt_loss = checkpoint_loss / checkpoint_pt_count
      println("Checkpoint loss $checkpoint_pt_loss")

      save_model("checkpoint", checkpoint_pt_loss)

      checkpoint_loss     = 0.0
      checkpoint_pt_count = 0 :: Int64
    end
  end

  println("Epoch $epoch_n loss $(epoch_loss/epoch_pt_count)")
  save_model("epoch_$(epoch_n)", epoch_loss/epoch_pt_count)

  epoch_n += 1
end