# Train a model with LightGBM
#
# $ julia TrainGBM.jl tor_days_only_model/lightgbm.jl


if length(ARGS) == 0
  # error("Must provide a model script in a directory that includes a train.txt file containing a list of paths")
  error("Must provide a model script in a directory that includes a train.bindata file created by e.g. MakeTornadoNeighborhoodsData.jl")
end

model_dir_and_script = ARGS[1]

MODEL_DIR    = dirname(model_dir_and_script)
model_script = basename(model_dir_and_script)
model_path   = "$MODEL_DIR/" * replace(model_script, ".jl", ".model")


include("ReadFeatureData.jl")

# epoch_n = 1

train_file_path    = "$MODEL_DIR/train.binfeatures"
train_labels_path  = "$MODEL_DIR/train.binlabels"
train_weights_path = "$MODEL_DIR/train.binweights"
dev_file_path      = "$MODEL_DIR/dev.binfeatures"
dev_labels_path    = "$MODEL_DIR/dev.binlabels"
dev_weights_path   = "$MODEL_DIR/dev.binweights"

train_data, train_data_file, train_point_count = open_data_file(train_file_path)

train_labels  = read(train_labels_path, Float32, train_point_count)
train_weights = read(train_weights_path, Float32, train_point_count)

dev_data,   dev_data_file,   dev_point_count   = open_data_file(dev_file_path)

dev_labels  = read(dev_labels_path, Float32, dev_point_count)
dev_weights = read(dev_weights_path, Float32, dev_point_count)


println("$train_point_count training points.")


# Load the estimator params
include("$MODEL_DIR/$model_script")


LightGBM.fit(
  estimator,
  train_data,
  train_labels,
  (dev_data, dev_labels);
  verbosity = 2,
  is_row_major = true, # Our data is in the columns, pretend it is row major and voila the data is in the rows for C
  weights = train_weights
  )

model_path = "$MODEL_DIR/lightgbm.model"
model_save(model_path)


close(train_data_file)
close(dev_data_file)