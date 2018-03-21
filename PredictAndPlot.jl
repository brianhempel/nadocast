# Usage:
#
# Loads a trained model, applies it to a grib2 file, and plots the results.
#
# $ julia PredictAndPlot.jl tiny_dev_model/logistic.bson rap_130_20170516_2200_001.grb2 [out_dir]
#

include("PlotMap.jl")

if length(ARGS) < 2
  error("Must provide a model script and a saved BSON file and a grib2 file")
end

# model_dir_and_type = ARGS[1]
saved_bson_path = ARGS[1]
grib2_path      = ARGS[2]
out_dir         = length(ARGS) == 3 ? ARGS[3] : "."

# model_dir    = dirname(model_dir_and_type)
# model_script = basename(model_dir_and_type)
# model_type   = replace(model_script, ".jl", "")

println("Loading Flux...")
using Flux
using BSON: @load

# const FEATURE_COUNT = 1544-10 # First 10 "features" are actually not.

# include("$model_dir/$model_script")

println("Loading $saved_bson_path...")

@load saved_bson_path model optimizer

println("Predicting tornado probabilities on $grib2_path with $saved_bson_path...")

println("Building features...")

feature_normalizers = include("NormalizingFactors.jl")

data_stream, proc = open(`julia read_grib.jl $grib2_path --all`)
tmp_data_path = last(split(strip(String(read(data_stream))),"\n")) # last line is the tmp file with the data

# If reading grib file okay...
if endswith(tmp_data_path, ".bindata")
  # println(tmp_data_path)
  data = read(tmp_data_path, Float32, (122067,1544)) :: Array{Float32, 2}
  open(`rm $tmp_data_path`)
  if any(isnan, data)
    error("contains nans")
  end
  if any(isinf, data)
    error("contains infs")
  end
else
  error("failed to build features")
end

println("Predicting...")

predictions = zeros(size(data,1))

for i in 1:size(data,1)
  x              = data[i,11:size(data,2)] ./ feature_normalizers
  predicted      = model(x)
  predictions[i] = Flux.Tracker.data(predicted[1])
end

println("Plotting...")

# Requires GMT >=6
plot_base_path = out_dir * "/" * basename(replace(grib2_path, ".grb2", "_tornado_probabilities"))
plot_map(plot_base_path, data[:,1], data[:,2], predictions)

println("$plot_base_path.pdf")
