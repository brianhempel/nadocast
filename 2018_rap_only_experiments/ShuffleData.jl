# Data from e.g. MakeTornadoNeighborhoodsData.jl is sequential.
#
# We want it shuffled if we are going to use any SGD algorithm.
#
# $ julia ShuffleData.jl model/train.binfeatures model/train.binlabels model/train.binweights

if length(ARGS) != 3
  # error("Must provide a model script in a directory that includes a train.txt file containing a list of paths")
  error("Must provide binfeatures, binlabels, and binweights files to shuffle")
end

data_path    = ARGS[1]
labels_path  = ARGS[2]
weights_path = ARGS[3]
out_path     = data_path * ".shuffled"
out_labels_path  = labels_path * ".shuffled"
out_weights_path = weights_path * ".shuffled"

include("ReadFeatureData.jl")

# train_data = open(train_file_path, "r", )

data, data_file, point_count = open_data_file(data_path)
labels = read(labels_path, Float32, point_count)
weights = read(weights_path, Float32, point_count)

# With my SSD, about ~100,000 / min

out_file     = open(out_path, "w")
out_labels_file  = open(out_labels_path, "w")
out_weights_file = open(out_weights_path, "w")

for i in shuffle(1:point_count)
  write(out_file,         @view data[:,i])
  write(out_labels_file,  labels[i])
  write(out_weights_file, weights[i])
end

close(out_file)
close(data_file)
close(out_labels_file)
close(out_weights_file)
