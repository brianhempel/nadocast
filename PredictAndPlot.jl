# Usage:
#
# Loads a trained model, applies it to a grib2 file, and plots the results.
#
# $ julia PredictAndPlot.jl tor_neighborhoods_only_model/2factor_logistic.jl tor_neighborhoods_only_model/2factor_logistic_2018-03-21T12-42-08.883_151545_examples_checkpoint_loss_0.0798001117753203.bson  rap_130_20170516_2200_001.grb2
#

include("PlotMap.jl")

if length(ARGS) < 3
  error("Must provide a model script and a saved model file and a grib2 file")
end

# model_dir_and_type = ARGS[1]
model_dir_and_type = ARGS[1]
saved_model_path   = ARGS[2]
grib2_path         = ARGS[3]
out_dir            = length(ARGS) == 4 ? ARGS[4] : pwd()


include("ReadFeatureData.jl")

model_dir    = dirname(model_dir_and_type)
model_script = basename(model_dir_and_type)
model_type   = replace(model_script, ".jl", "")

USE_NORMALIZING_FACTORS = true

include("$model_dir/$model_script")

println("Loading $saved_model_path...")

model_load(saved_model_path)

println("Predicting tornado probabilities on $grib2_path with $saved_model_path...")

println("Building features...")

feature_normalizers = include("NormalizingFactors.jl")

data_stream, proc = open(`julia read_grib.jl $grib2_path --all`)
tmp_data_path = last(split(strip(String(read(data_stream))),"\n")) # last line is the tmp file with the data

# If reading grib file okay...
if endswith(tmp_data_path, ".bindata")
  # println(tmp_data_path)
  data = read_data_file(tmp_data_path)
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

predictions = zeros(size(data,2))
b1s = zeros(size(data,2))
f0s = zeros(size(data,2))
f1s = zeros(size(data,2))
# f2s = zeros(size(data,2))
# f3s = zeros(size(data,2))
# f4s = zeros(size(data,2))

if isdefined(:Flux)
  Flux.testmode!(model, true) # Turn off dropout
end

if isdefined(:model_predict_all)
  predictions = model_predict_all(data[11:size(data,1),:])
  # show(predictions)
else
  for i in 1:size(data,2)
    if USE_NORMALIZING_FACTORS
      x = (@view data[11:size(data,1),i]) ./ feature_normalizers
    else
      x = @view data[11:size(data,1),i]
    end
    # predictions[i] = Flux.Tracker.data(predicted[1])
    predictions[i] = model_prediction(x)
    # predicted      = model(x)
    # b1             = model.b1(x)
    # f0             = model.f0(x)
    # f1             = model.f1(x)
    # f2             = model.f2(x)
    # f3             = model.f3(x)
    # f4             = model.f4(x)
    # b1s[i]         = Flux.Tracker.data(b1[1])
    # f0s[i]         = Flux.Tracker.data(f0[1])
    # f1s[i]         = Flux.Tracker.data(f1[1])
    # f2s[i]         = Flux.Tracker.data(f2[1])
    # f3s[i]         = Flux.Tracker.data(f3[1])
    # f4s[i]         = Flux.Tracker.data(f4[1])
  end
end

println("Plotting...")

# Requires GMT >=6
plot_base_path = out_dir * "/" * basename(replace(grib2_path, r"\.gri?b2", "_tornado_probabilities_$(model_type)"))
plot_map(plot_base_path, data[1,:], data[2,:], predictions)
# plot_map(plot_base_path * "_binary_1", data[1,:], data[2,:], b1s)
# plot_map(plot_base_path * "_factor_0", data[1,:], data[2,:], f0s)
# plot_map(plot_base_path * "_factor_1", data[1,:], data[2,:], f1s)
# plot_map(plot_base_path * "4factor_1", data[1,:], data[2,:], f1s)
# plot_map(plot_base_path * "4factor_1", data[1,:], data[2,:], f1s)
# plot_map(plot_base_path * "4factor_2", data[1,:], data[2,:], f2s)
# plot_map(plot_base_path * "4factor_3", data[1,:], data[2,:], f3s)
# plot_map(plot_base_path * "4factor_4", data[1,:], data[2,:], f4s)

println("$plot_base_path.pdf")
