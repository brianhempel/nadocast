# Find the best 1 or 2 or 3 features that, taken together, best discriminate tornados from non-tornadoes.
#
# $ julia TrainBestDiscriminators.jl tor_days_only_model/

import StatsBase

if length(ARGS) == 0
  # error("Must provide a model script in a directory that includes a train.txt file containing a list of paths")
  error("Must provide directory that includes a train.bindata file created by e.g. MakeTordnadoNeighborhoodsData.jl")
end

model_dir = ARGS[1]

train_file_path    = "$model_dir/train-shuffled.binfeatures"
train_labels_path  = "$model_dir/train-shuffled.binlabels"
train_weights_path = "$model_dir/train-shuffled.binweights"
dev_file_path      = "$model_dir/dev2017+1.binfeatures"
dev_labels_path    = "$model_dir/dev2017+1.binlabels"
dev_weights_path   = "$model_dir/dev2017+1.binweights"

include("ReadFeatureData.jl")

train_data, train_data_file, train_point_count = open_data_file(train_file_path)

train_labels  = read(train_labels_path, Float32, train_point_count)
train_weights = read(train_weights_path, Float32, train_point_count)

dev_data,   dev_data_file,   dev_point_count   = open_data_file(dev_file_path)

dev_labels  = read(dev_labels_path, Float32, dev_point_count)
dev_weights = read(dev_weights_path, Float32, dev_point_count)

println("$train_point_count training points.")

println("Building histogram of tornado points.")


tornado_points = train_data[:, 1.0f0 .== train_labels]


stdout_limited = IOContext(STDOUT, :display_size=>(100,60))
stdout_limited = IOContext(stdout_limited, :limit=>true)

show(stdout_limited, "text/plain", tornado_points)
println("")


# So far, best results are 7 bins, 4 dimensions. Tried 5,6,7,8,9 bins.
# 0.06335109658775055  PLPL:255-0 mb above ground:storm path max  HLCY:3000-0 m above ground:storm path max  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path mean
# (448, 400, 468, 416)
# Dev loss: 0.06787637968114556

const BIN_COUNT = 2
const EXPAND_TOP_BOTTOM_TOR_BINS = true && BIN_COUNT >= 3

function make_feature_splits(data)
  features_js_to_ignore = Int64[]
  splits = zeros(Float32, BIN_COUNT - 1, FEATURE_COUNT) # Seven splits for 8 bins (first and last bin extend to infinity)

  for j = 1:FEATURE_COUNT
    feature_vals = data[j,:]
    if BIN_COUNT == 2
      basic_splits = [StatsBase.median(feature_vals)]
    else
      basic_splits = StatsBase.quantile(feature_vals, (0:(BIN_COUNT - 2))/(BIN_COUNT - 2))
    end
    if EXPAND_TOP_BOTTOM_TOR_BINS && BIN_COUNT >= 3
      q1, q3       = StatsBase.quantile(feature_vals, [0.25, 0.75])
      iqr          = q3 - q1
    end
    range        = maximum(basic_splits) - minimum(basic_splits)

    if range == 0.0 && BIN_COUNT >= 3
      println("$(FEATURE_HEADERS[j]) has no range") # Several snow/ice/freezing rain features.
      push!(features_js_to_ignore, j)
    elseif unique(basic_splits) != basic_splits && BIN_COUNT >= 3
      println("$(FEATURE_HEADERS[j]) is poorly disributed $(basic_splits)") # A variety.
      push!(features_js_to_ignore, j)
    else
      # Expand end-bins to ensure they include q1-1.5*iqr and q3+1.5*iqr, in accordance with the standard test for outliers
      if EXPAND_TOP_BOTTOM_TOR_BINS && basic_splits[1] > q1-1.5*iqr && BIN_COUNT >= 3
        basic_splits[1] = q1-1.5*iqr
      end
      if EXPAND_TOP_BOTTOM_TOR_BINS && basic_splits[length(basic_splits)] < q3+1.5*iqr && BIN_COUNT >= 3
        basic_splits[length(basic_splits)] = q3+1.5*iqr
      end
      splits[:,j] = basic_splits
    end
  end

  splits, features_js_to_ignore
end

splits, features_js_to_ignore = make_feature_splits(tornado_points) # ~81 features ignored

# Read data in binned, LightGBM style (only 9.5 GB woot woot)

function make_bin_data(data, splits)
  bined_data  = zeros(UInt8, (size(data,2),size(data,1))) # Binned data rotated so features are in columns
  split_count = size(splits,1)

  for i = 1:size(data,2)
    for j = 1:size(data,1)
      val = data[j,i]
      for q = 1:split_count
        if val < splits[q,j]
          bined_data[i,j] = q
          break
        end
      end
      if bined_data[i,j] == 0 # No bin assigned yet
        bined_data[i,j] = val == splits[split_count,j] ? split_count : 1 + split_count
      end
    end
  end

  bined_data
end


println("Bining $train_file_path")

bined_train_data = make_bin_data(train_data, splits)
# bined_train_data = make_bin_data(train_data[:,1:500000], splits)
# train_weights = train_weights[1:500000]
# train_labels  = train_labels[1:500000]

println("Bining $dev_file_path")

bined_dev_data   = make_bin_data(dev_data, splits)
# dev_weights = dev_weights[1:100000]


function bins_train_loss(pos_bin_counts, neg_bin_counts)
  loss = 0.0

  for b = 1:length(pos_bin_counts)
    p = pos_bin_counts[b] / (pos_bin_counts[b] + neg_bin_counts[b])

    if p > 0.0
      loss += pos_bin_counts[b] * -log(p) # If we want to interpret as bits, should be log base 2, but need to do what Flux.jl does so we can compare with other models
    end
    if p < 1.0
      loss += neg_bin_counts[b] * -log(1-p)
    end
  end

  loss / (sum(pos_bin_counts) + sum(neg_bin_counts))
end


println("Finding best single discriminating dimension")

# Note this is training loss
function find_dimension_loss(j, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, bin_count) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, bin_count) - dumb_p # Smoothing so loss is never infinite

  for i = 1:size(bined_train_data,1)
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end


function find_one_dimension_loss(j1, bin_probs, bined_data, labels, weights)
  loss = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
    else
      loss += -log(1-p)*weights[i]
    end
  end

  loss / sum(weights)
end

dimension_losses    = zeros(Float32, FEATURE_COUNT)
dimension_bin_probs = []

for j = 1:FEATURE_COUNT
  if j in features_js_to_ignore
    dimension_losses[j] = Inf32
    push!(dimension_bin_probs, zeros(Float32, BIN_COUNT))
    continue
  end
  bin_probs, dimension_losses[j] = find_dimension_loss(j, bined_train_data, train_labels, train_weights, splits)
  push!(dimension_bin_probs, bin_probs)
  # println("$(dimension_losses[j])\t$(FEATURE_HEADERS[j])")
end

println("Best 20")

for j = Iterators.take(sortperm(dimension_losses), 20)
  println("$(dimension_losses[j])\t$(find_one_dimension_loss(j, dimension_bin_probs[j], bined_dev_data, dev_labels, dev_weights))\t$(splits[:,j])\t$(FEATURE_HEADERS[j])")
end

# # Best 20
# # 0.06439152  Float32[-10.0, 28.4375, 32.0, 34.5, 36.25, 38.0, 44.75]  REFD:4000 m above ground:storm path max
# # 0.06541891  Float32[0.0, 0.000725, 0.00125, 0.002, 0.0034, 0.0055, 0.0138]  PRATE:surface:storm path max
# # 0.06542256  Float32[0.0, 0.000156061, 0.000334329, 0.000544054, 0.000814577, 0.00129368, 0.00573576]  PRATE:surface:storm path 50mi mean
# # 0.06547563  Float32[-10.0, 28.75, 32.5, 35.0, 37.25, 39.75, 47.0]  REFD:1000 m above ground:storm path max
# # 0.065660775  Float32[0.0, 0.000101936, 0.000280649, 0.000517442, 0.000853192, 0.00146928, 0.00740213]  PRATE:surface:storm path mean
# # 0.065856606  Float32[-10.0, 29.25, 32.8125, 35.4375, 37.875, 40.8125, 48.1875]  REFC:entire atmosphere (considered as a single layer):storm path max
# # 0.06672716  Float32[-10.0, 0.520135, 9.28424, 15.8263, 21.4447, 26.7401, 38.4333]  REFD:4000 m above ground:storm path mean
# # 0.06676262  Float32[-3.28678, -1.19823, -0.863891, -0.627013, -0.410341, -0.190932, 0.755745]  VVEL:400 mb:storm path 50mi mean
# # 0.066764064  Float32[-3.5904, -1.25099, -0.895996, -0.645052, -0.417919, -0.196001, 0.78147]  VVEL:425 mb:storm path 50mi mean
# # 0.06676925  Float32[-3.84341, -1.30043, -0.922822, -0.661854, -0.422753, -0.197242, 0.770627]  VVEL:450 mb:storm path 50mi mean
# # 0.06677809  Float32[-2.94302, -1.13851, -0.827974, -0.602796, -0.398061, -0.180665, 0.68937]  VVEL:375 mb:storm path 50mi mean
# # 0.06679979  Float32[-4.19604, -1.38663, -0.956969, -0.676832, -0.431867, -0.197711, 0.62149]  VVEL:500 mb:storm path 50mi mean
# # 0.06680778  Float32[-4.04924, -1.34593, -0.942434, -0.671049, -0.426302, -0.196724, 0.714794]  VVEL:475 mb:storm path 50mi mean
# # 0.066833586  Float32[-2.70787, -1.07618, -0.784122, -0.569788, -0.376122, -0.167332, 0.604375]  VVEL:350 mb:storm path 50mi mean
# # 0.06683985  Float32[-4.31722, -1.41521, -0.965911, -0.678467, -0.436499, -0.196445, 0.674171]  VVEL:525 mb:storm path 50mi mean
# # 0.06686149  Float32[-10.0, 1.97426, 8.25609, 13.0526, 17.3769, 21.9313, 37.2582]  REFD:4000 m above ground:storm path 50mi mean
# # 0.066869594  Float32[-2.46512, -1.00592, -0.730374, -0.531203, -0.346844, -0.152682, 0.504948]  VVEL:325 mb:storm path 50mi mean
# # 0.06689867  Float32[-4.38393, -1.43187, -0.970389, -0.677494, -0.436581, -0.194114, 0.749171]  VVEL:550 mb:storm path 50mi mean
# # 0.06692485  Float32[-5.92105, -2.16457, -1.63531, -1.26092, -0.944889, -0.595943, 0.385136]  VVEL:325 mb:storm path min
# # 0.06693122  Float32[-5000.0, -1458.08, 1828.31, 4474.65, 6947.91, 9453.31, 14782.6]  HGT:convective cloud top level:storm path mean
#
#
# println("Finding best two discriminating dimensions")
#
# # Note this is training loss
# function find_two_dimension_loss(j1, j2, bined_train_data, train_labels, train_weights, splits)
#   split_count = size(splits,1)
#   bin_count   = split_count + 1
#   dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
#   pos_bin_counts = zeros(Float32, (bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
#   neg_bin_counts = ones(Float32, (bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
#   point_count    = size(bined_train_data,1)
#
#   for i = 1:point_count
#     if train_labels[i] == 1.0f0
#       pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2]] += train_weights[i]
#     else
#       neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2]] += train_weights[i]
#     end
#   end
#
#   loss = 0.0
#
#   for b = 1:(bin_count*bin_count)
#     p = pos_bin_counts[b] / (pos_bin_counts[b] + neg_bin_counts[b])
#
#     if p > 0.0
#       loss += pos_bin_counts[b] * -log(p) # If we want to interpret as bits, should be log base 2, but need to do what Flux.jl does so we can compare with other models
#     end
#     if p < 1.0
#       loss += neg_bin_counts[b] * -log(1-p)
#     end
#
#     if (b == 1 || b == bin_count) && pos_bin_counts[b] > 0.0
#       error("Bin $b should have no tornadoes for $(FEATURE_HEADERS[j]) but it had $(pos_bin_counts[b])")
#     end
#   end
#
#   loss / point_count
# end
#
# two_dimensions_to_try = Tuple{Int64,Int64}[]
#
# for j1 = Iterators.take(sortperm(dimension_losses), 200), j2 = 1:FEATURE_COUNT
#   if j2 in features_js_to_ignore
#     continue
#   end
#   push!(two_dimensions_to_try, (j1, j2))
# end
#
# two_dimension_losses = zeros(Float32, length(two_dimensions_to_try)) + Inf32
#
# for j12 in 1:length(two_dimensions_to_try)
#   j1, j2 = two_dimensions_to_try[j12]
#
#   two_dimension_losses[j12] = find_two_dimension_loss(j1, j2, bined_train_data, train_labels, train_weights, splits)
#   println("$(two_dimension_losses[j12])\t$(FEATURE_HEADERS[j1])\t$(FEATURE_HEADERS[j2])")
# end
#
#
# println("Best 50")
#
# for j12 = Iterators.take(sortperm(two_dimension_losses), 50)
#   j1, j2 = two_dimensions_to_try[j12]
#
#   println("$(two_dimension_losses[j12])\t$(FEATURE_HEADERS[j1])\t$(FEATURE_HEADERS[j2])")
# end

# Best 50
# 0.062518835  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path min
# 0.06256918  PRATE:surface:storm path 50mi mean  CAPE:surface:point
# 0.06258796  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:point
# 0.0626359  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path mean
# 0.062648326  PRATE:surface:storm path 50mi mean  CAPE:90-0 mb above ground:point
# 0.06268907  PRATE:surface:storm path 50mi mean  CAPE:surface:storm path mean
# 0.06278532  PRATE:surface:storm path 50mi mean  CAPE:90-0 mb above ground:storm path mean
# 0.06281193  PRATE:surface:storm path max  CAPE:surface:point
# 0.06281239  PRATE:surface:storm path max  CAPE:90-0 mb above ground:point
# 0.06284881  PRATE:surface:storm path 50mi mean  CAPE:surface:storm path max
# 0.06285526  PRATE:surface:storm path max  LFTX:500-1000 mb:point
# 0.062887944  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path 50mi mean
# 0.06290138  PRATE:surface:storm path max  LFTX:500-1000 mb:storm path min
# 0.06291911  PRATE:surface:storm path max  LFTX:500-1000 mb:storm path mean
# 0.062925145  PRATE:surface:storm path max  CAPE:surface:storm path mean
# 0.06294769  REFD:4000 m above ground:storm path max  LFTX:500-1000 mb:storm path min
# 0.0629484  REFD:4000 m above ground:storm path max  LFTX:500-1000 mb:point
# 0.06295984  PRATE:surface:storm path max  CAPE:90-0 mb above ground:storm path mean
# 0.06296337  PRATE:surface:storm path 50mi mean  CAPE:surface:storm path 50mi mean
# 0.06296719  REFD:4000 m above ground:storm path max  CAPE:surface:point
# 0.0629737  REFD:4000 m above ground:storm path max  VVEL:60-30 mb above ground:storm path 50mi mean
# 0.0629737  VVEL:60-30 mb above ground:storm path 50mi mean  REFD:4000 m above ground:storm path max
# 0.06300385  REFD:4000 m above ground:storm path max  LFTX:500-1000 mb:storm path mean
# 0.063010134  PRATE:surface:storm path 50mi mean  CAPE:90-0 mb above ground:storm path max
# 0.06304305  REFD:4000 m above ground:storm path max  CAPE:90-0 mb above ground:point
# 0.0630593  PRATE:surface:storm path 50mi mean  CAPE:90-0 mb above ground:storm path 50mi mean
# 0.06306147  REFD:4000 m above ground:storm path max  CAPE:surface:storm path mean
# 0.063088425  REFD:4000 m above ground:storm path max  VVEL:30-0 mb above ground:storm path 50mi mean
# 0.06309098  PRATE:surface:storm path 50mi mean  CAPE:90-0 mb above ground:storm path min
# 0.06309677  PRATE:surface:storm path mean  LFTX:500-1000 mb:storm path min
# 0.06310417  HGT:convective cloud top level:storm path 50mi mean  HLCY:3000-0 m above ground:storm path max
# 0.06310417  HLCY:3000-0 m above ground:storm path max  HGT:convective cloud top level:storm path 50mi mean
# 0.06312986  PRATE:surface:storm path 50mi mean  CAPE:surface:storm path min
# 0.063134395  PRATE:surface:storm path mean  LFTX:500-1000 mb:point
# 0.06313598  PRATE:surface:storm path max  CAPE:surface:storm path max
# 0.06313754  PRATE:surface:storm path 50mi mean  CAPE:180-0 mb above ground:point
# 0.06314801  REFD:1000 m above ground:storm path max  LFTX:500-1000 mb:storm path min
# 0.0631511  REFD:4000 m above ground:storm path max  VVEL:90-60 mb above ground:storm path 50mi mean
# 0.0631511  VVEL:90-60 mb above ground:storm path 50mi mean  REFD:4000 m above ground:storm path max
# 0.06315609  REFD:1000 m above ground:storm path max  CAPE:surface:point
# 0.06316165  REFD:1000 m above ground:storm path max  LFTX:500-1000 mb:point
# 0.063162565  REFD:4000 m above ground:storm path max  CAPE:90-0 mb above ground:storm path mean
# 0.06317672  PRATE:surface:storm path max  LFTX:500-1000 mb:storm path 50mi mean
# 0.063179664  REFD:4000 m above ground:storm path max  LFTX:500-1000 mb:storm path 50mi mean
# 0.06318092  REFD:1000 m above ground:storm path max  CAPE:90-0 mb above ground:point
# 0.06318095  PRATE:surface:storm path max  CAPE:90-0 mb above ground:storm path min
# 0.06318618  REFC:entire atmosphere (considered as a single layer):storm path max  CAPE:surface:point
# 0.06318711  REFC:entire atmosphere (considered as a single layer):storm path max  CAPE:90-0 mb above ground:point
# 0.063187934  REFC:entire atmosphere (considered as a single layer):storm path max  LFTX:500-1000 mb:storm path min
# 0.06319496  PRATE:surface:storm path mean  CAPE:surface:point


# Not exhaustive like above.

println("Exploring two discriminating dimensions")

# Note this is training loss
function find_two_dimension_bin_probs(j1, j2, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, (bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, (bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
  point_count    = size(bined_train_data,1)

  for i = 1:point_count
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end


function find_two_dimension_loss(j1, j2, bin_probs, bined_data, labels, weights)
  loss = 0.0
  excluded_positives = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1], bined_data[i,j2]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
      if p < 0.02
        excluded_positives += weights[i]
      end
    else
      loss += -log(1-p)*weights[i]
    end
  end

  (loss / sum(weights), excluded_positives / sum(weights .* labels))
end

function explore_two_dimension_loss(starting_j2)
  best_j1, best_j2 = 1, starting_j2
  best_loss = Inf32
  best_bin_probs = zeros(Float32, (BIN_COUNT, BIN_COUNT))

  any_progress = true
  while any_progress
    any_progress = false
    for j1 = 1:FEATURE_COUNT
      if j1 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_two_dimension_bin_probs(j1, best_j2, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j1 = j1
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])")
      end
    end
    for j2 = 1:FEATURE_COUNT
      if j2 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_two_dimension_bin_probs(best_j1, j2, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j2 = j2
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])")
      end
    end
  end

  best_bin_probs, best_loss, best_j1, best_j2
end

# Explore the space varying one dimension at a time until convergence.

# 468, 417  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path min
# 526 REFD:4000 m above ground:storm path max

best_bin_probs, _, best_j1, best_j2 = explore_two_dimension_loss(526)

println("Dev loss: $(find_two_dimension_loss(best_j1, best_j2, best_bin_probs, bined_dev_data, dev_labels, dev_weights))")
println("$(FEATURE_HEADERS[best_j1])\t$(splits[:,best_j1])")
println("$(FEATURE_HEADERS[best_j2])\t$(splits[:,best_j2])")
println("")





println("Exploring three discriminating dimensions")

# Note this is training loss
function find_three_dimension_bin_probs(j1, j2, j3, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, (bin_count, bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, (bin_count, bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
  point_count    = size(bined_train_data,1)

  for i = 1:point_count
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end


function find_three_dimension_loss(j1, j2, j3, bin_probs, bined_data, labels, weights)
  loss = 0.0
  excluded_positives = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1], bined_data[i,j2], bined_data[i,j3]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
      if p < 0.02
        excluded_positives += weights[i]
      end
    else
      loss += -log(1-p)*weights[i]
    end
  end

  (loss / sum(weights), excluded_positives / sum(weights .* labels))
end

function explore_three_dimension_loss(starting_j2, starting_j3)
  best_j1, best_j2, best_j3 = 1, starting_j2, starting_j3
  best_loss = Inf32
  best_bin_probs = zeros(Float32, (BIN_COUNT, BIN_COUNT, BIN_COUNT))

  any_progress = true
  while any_progress
    any_progress = false
    for j1 = 1:FEATURE_COUNT
      if j1 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_three_dimension_bin_probs(j1, best_j2, best_j3, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j1 = j1
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])")
      end
    end
    for j2 = 1:FEATURE_COUNT
      if j2 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_three_dimension_bin_probs(best_j1, j2, best_j3, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j2 = j2
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])")
      end
    end
    for j3 = 1:FEATURE_COUNT
      if j3 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_three_dimension_bin_probs(best_j1, best_j2, j3, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j3 = j3
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])")
      end
    end
  end

  best_bin_probs, best_loss, best_j1, best_j2, best_j3
end

# Explore the space varying one dimension at a time until convergence.

# 468, 417  PRATE:surface:storm path 50mi mean  LFTX:500-1000 mb:storm path min

best_bin_probs, _, best_j1, best_j2, best_j3 = explore_three_dimension_loss(best_j1, best_j2)

println("Dev loss: $(find_three_dimension_loss(best_j1, best_j2, best_j3, best_bin_probs, bined_dev_data, dev_labels, dev_weights))")
println("$(FEATURE_HEADERS[best_j1])\t$(splits[:,best_j1])")
println("$(FEATURE_HEADERS[best_j2])\t$(splits[:,best_j2])")
println("$(FEATURE_HEADERS[best_j3])\t$(splits[:,best_j3])")
println("")





println("Exploring four discriminating dimensions")

# Note this is training loss
function find_four_dimension_bin_probs(j1, j2, j3, j4, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, (bin_count, bin_count, bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, (bin_count, bin_count, bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
  point_count    = size(bined_train_data,1)

  for i = 1:point_count
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end

function find_four_dimension_loss(j1, j2, j3, j4, bin_probs, bined_data, labels, weights)
  loss = 0.0
  excluded_positives = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1], bined_data[i,j2], bined_data[i,j3], bined_data[i,j4]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
      if p < 0.02
        excluded_positives += weights[i]
      end
    else
      loss += -log(1-p)*weights[i]
    end
  end

  (loss / sum(weights), excluded_positives / sum(weights .* labels))
end

function explore_four_dimension_loss(starting_j2, starting_j3, starting_j4)
  best_j1, best_j2, best_j3, best_j4 = 1, starting_j2, starting_j3, starting_j4
  best_loss = Inf32
  best_bin_probs = zeros(Float32, (BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT))

  any_progress = true
  while any_progress
    any_progress = false
    for j1 = 1:FEATURE_COUNT
      if j1 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_four_dimension_bin_probs(j1, best_j2, best_j3, best_j4, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j1 = j1
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])")
      end
    end
    for j2 = 1:FEATURE_COUNT
      if j2 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_four_dimension_bin_probs(best_j1, j2, best_j3, best_j4, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j2 = j2
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])")
      end
    end
    for j3 = 1:FEATURE_COUNT
      if j3 in features_js_to_ignore
        continue
      end
      lbin_probs, loss = find_four_dimension_bin_probs(best_j1, best_j2, j3, best_j4, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j3 = j3
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])")
      end
    end
    for j4 = 1:FEATURE_COUNT
      if j4 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_four_dimension_bin_probs(best_j1, best_j2, best_j3, j4, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j4 = j4
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])")
      end
    end
  end

  best_bin_probs, best_loss, best_j1, best_j2, best_j3, best_j4
end


# Explore the space varying one dimension at a time until convergence.

best_bin_probs, _, best_j1, best_j2, best_j3, best_j4 = explore_four_dimension_loss(best_j1, best_j2, best_j3)

println((best_j1, best_j2, best_j3, best_j4))

println("Dev loss: $(find_four_dimension_loss(best_j1, best_j2, best_j3, best_j4, best_bin_probs, bined_dev_data, dev_labels, dev_weights))")
println("$(FEATURE_HEADERS[best_j1])\t$(splits[:,best_j1])")
println("$(FEATURE_HEADERS[best_j2])\t$(splits[:,best_j2])")
println("$(FEATURE_HEADERS[best_j3])\t$(splits[:,best_j3])")
println("$(FEATURE_HEADERS[best_j4])\t$(splits[:,best_j4])")
println("")



println("Exploring five discriminating dimensions")

# Note this is training loss
function find_five_dimension_bin_probs(j1, j2, j3, j4, j5, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, (bin_count, bin_count, bin_count, bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, (bin_count, bin_count, bin_count, bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
  point_count    = size(bined_train_data,1)

  for i = 1:point_count
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4], bined_train_data[i,j5]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4], bined_train_data[i,j5]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end

function find_five_dimension_loss(j1, j2, j3, j4, j5, bin_probs, bined_data, labels, weights)
  loss = 0.0
  excluded_positives = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1], bined_data[i,j2], bined_data[i,j3], bined_data[i,j4], bined_data[i,j5]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
      if p < 0.02
        excluded_positives += weights[i]
      end
    else
      loss += -log(1-p)*weights[i]
    end
  end

  (loss / sum(weights), excluded_positives / sum(weights .* labels))
end


function explore_five_dimension_loss(starting_j2, starting_j3, starting_j4, starting_j5)
  best_j1, best_j2, best_j3, best_j4, best_j5 = 1, starting_j2, starting_j3, starting_j4, starting_j5
  best_loss = Inf32
  best_bin_probs = zeros(Float32, (BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT))

  any_progress = true
  while any_progress
    any_progress = false
    for j1 = 1:FEATURE_COUNT
      if j1 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_five_dimension_bin_probs(j1, best_j2, best_j3, best_j4, best_j5, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j1 = j1
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])")
      end
    end
    for j2 = 1:FEATURE_COUNT
      if j2 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_five_dimension_bin_probs(best_j1, j2, best_j3, best_j4, best_j5, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j2 = j2
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])")
      end
    end
    for j3 = 1:FEATURE_COUNT
      if j3 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_five_dimension_bin_probs(best_j1, best_j2, j3, best_j4, best_j5, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j3 = j3
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])")
      end
    end
    for j4 = 1:FEATURE_COUNT
      if j4 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_five_dimension_bin_probs(best_j1, best_j2, best_j3, j4, best_j5, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j4 = j4
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])")
      end
    end
    for j5 = 1:FEATURE_COUNT
      if j5 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_five_dimension_bin_probs(best_j1, best_j2, best_j3, best_j4, j5, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j5 = j5
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])")
      end
    end
  end

  best_bin_probs, best_loss, best_j1, best_j2, best_j3, best_j4, best_j5
end


# Explore the space varying one dimension at a time until convergence.

function explore_and_evaluate_5d(starting_j2, starting_j3, starting_j4, starting_j5)
  best_bin_probs, best_loss, best_j1, best_j2, best_j3, best_j4, best_j5 = explore_five_dimension_loss(starting_j2, starting_j3, starting_j4, starting_j5)

  println((best_j1, best_j2, best_j3, best_j4, best_j5))

  println("Dev loss: $(find_five_dimension_loss(best_j1, best_j2, best_j3, best_j4, best_j5, best_bin_probs, bined_dev_data, dev_labels, dev_weights))")
  println("$(FEATURE_HEADERS[best_j1])\t$(splits[:,best_j1])")
  println("$(FEATURE_HEADERS[best_j2])\t$(splits[:,best_j2])")
  println("$(FEATURE_HEADERS[best_j3])\t$(splits[:,best_j3])")
  println("$(FEATURE_HEADERS[best_j4])\t$(splits[:,best_j4])")
  println("$(FEATURE_HEADERS[best_j5])\t$(splits[:,best_j5])")
  println("")

  best_loss, best_j1, best_j2, best_j3, best_j4, best_j5
end

best_loss, best_j1, best_j2, best_j3, best_j4, best_j5 = explore_and_evaluate_5d(best_j1, best_j2, best_j3, best_j4)

# # Start from 20 random dimensions and explore
# for _ = 1:20
#   starting_j2, starting_j3, starting_j4, starting_j5 = 1,1,1,1
#   while true
#     starting_j2, starting_j3, starting_j4, starting_j5 = rand(1:FEATURE_COUNT), rand(1:FEATURE_COUNT), rand(1:FEATURE_COUNT), rand(1:FEATURE_COUNT)
#     if starting_j2 in features_js_to_ignore
#       continue
#     end
#     if starting_j3 in features_js_to_ignore
#       continue
#     end
#     if starting_j4 in features_js_to_ignore
#       continue
#     end
#     if starting_j5 in features_js_to_ignore
#       continue
#     end
#     break
#   end
#   loss, j1, j2, j3, j4, j5 = explore_and_evaluate_5d(starting_j2, starting_j3, starting_j4, starting_j5)
#   if loss < best_loss
#     best_loss, best_j1, best_j2, best_j3, best_j4, best_j5 = loss, j1, j2, j3, j4, j5
#   end
# end
#
# println("Best (train loss):")
# explore_and_evaluate_5d(best_j2, best_j3, best_j4, best_j5)



println("Exploring six discriminating dimensions")

# Note this is training loss
function find_six_dimension_bin_probs(j1, j2, j3, j4, j5, j6, bined_train_data, train_labels, train_weights, splits)
  split_count = size(splits,1)
  bin_count   = split_count + 1
  dumb_p = sum(train_labels .* train_weights) / sum(train_weights)
  pos_bin_counts = zeros(Float32, (bin_count, bin_count, bin_count, bin_count, bin_count, bin_count)) + dumb_p # Smoothing so loss is never infinite
  neg_bin_counts = ones(Float32, (bin_count, bin_count, bin_count, bin_count, bin_count, bin_count)) - dumb_p # Smoothing so loss is never infinite
  point_count    = size(bined_train_data,1)

  for i = 1:point_count
    if train_labels[i] == 1.0f0
      pos_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4], bined_train_data[i,j5], bined_train_data[i,j6]] += train_weights[i]
    else
      neg_bin_counts[bined_train_data[i,j1], bined_train_data[i,j2], bined_train_data[i,j3], bined_train_data[i,j4], bined_train_data[i,j5], bined_train_data[i,j6]] += train_weights[i]
    end
  end

  bin_probs = pos_bin_counts ./ (pos_bin_counts .+ neg_bin_counts)

  bin_probs, bins_train_loss(pos_bin_counts, neg_bin_counts)
end

function find_six_dimension_loss(j1, j2, j3, j4, j5, j6, bin_probs, bined_data, labels, weights)
  loss = 0.0
  excluded_positives = 0.0

  for i = 1:size(bined_data,1)
    p = bin_probs[bined_data[i,j1], bined_data[i,j2], bined_data[i,j3], bined_data[i,j4], bined_data[i,j5], bined_data[i,j6]]

    if labels[i] == 1.0f0
      loss += -log(p)*weights[i]
      if p < 0.02
        excluded_positives += weights[i]
      end
    else
      loss += -log(1-p)*weights[i]
    end
  end

  (loss / sum(weights), excluded_positives / sum(weights .* labels))
end


function explore_six_dimension_loss(starting_j2, starting_j3, starting_j4, starting_j5, starting_j6)
  best_j1, best_j2, best_j3, best_j4, best_j5, best_j6 = 1, starting_j2, starting_j3, starting_j4, starting_j5, starting_j6
  best_loss = Inf32
  best_bin_probs = zeros(Float32, (BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT, BIN_COUNT))

  any_progress = true
  while any_progress
    any_progress = false
    for j1 = 1:FEATURE_COUNT
      if j1 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(j1, best_j2, best_j3, best_j4, best_j5, best_j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j1 = j1
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
    for j2 = 1:FEATURE_COUNT
      if j2 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(best_j1, j2, best_j3, best_j4, best_j5, best_j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j2 = j2
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
    for j3 = 1:FEATURE_COUNT
      if j3 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(best_j1, best_j2, j3, best_j4, best_j5, best_j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j3 = j3
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
    for j4 = 1:FEATURE_COUNT
      if j4 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(best_j1, best_j2, best_j3, j4, best_j5, best_j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j4 = j4
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
    for j5 = 1:FEATURE_COUNT
      if j5 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(best_j1, best_j2, best_j3, best_j4, j5, best_j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j5 = j5
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
    for j6 = 1:FEATURE_COUNT
      if j6 in features_js_to_ignore
        continue
      end
      bin_probs, loss = find_six_dimension_bin_probs(best_j1, best_j2, best_j3, best_j4, best_j5, j6, bined_train_data, train_labels, train_weights, splits)
      if loss < best_loss
        best_loss = loss
        best_j6 = j6
        best_bin_probs = bin_probs
        any_progress = true
        println("$(best_loss)\t$(FEATURE_HEADERS[best_j1])\t$(FEATURE_HEADERS[best_j2])\t$(FEATURE_HEADERS[best_j3])\t$(FEATURE_HEADERS[best_j4])\t$(FEATURE_HEADERS[best_j5])\t$(FEATURE_HEADERS[best_j6])")
      end
    end
  end

  best_bin_probs, best_loss, best_j1, best_j2, best_j3, best_j4, best_j5, best_j6
end


# Explore the space varying one dimension at a time until convergence.

function explore_and_evaluate_6d(starting_j2, starting_j3, starting_j4, starting_j5, starting_j6)
  best_bin_probs, best_loss, best_j1, best_j2, best_j3, best_j4, best_j5, best_j6 = explore_six_dimension_loss(starting_j2, starting_j3, starting_j4, starting_j5, starting_j6)

  println((best_j1, best_j2, best_j3, best_j4, best_j5, best_j6))

  println("Dev loss: $(find_six_dimension_loss(best_j1, best_j2, best_j3, best_j4, best_j5, best_j6, best_bin_probs, bined_dev_data, dev_labels, dev_weights))")
  println("$(FEATURE_HEADERS[best_j1])\t$(splits[:,best_j1])")
  println("$(FEATURE_HEADERS[best_j2])\t$(splits[:,best_j2])")
  println("$(FEATURE_HEADERS[best_j3])\t$(splits[:,best_j3])")
  println("$(FEATURE_HEADERS[best_j4])\t$(splits[:,best_j4])")
  println("$(FEATURE_HEADERS[best_j5])\t$(splits[:,best_j5])")
  println("$(FEATURE_HEADERS[best_j6])\t$(splits[:,best_j6])")
  println("")

  best_loss, best_j1, best_j2, best_j3, best_j4, best_j5, best_j6
end

best_loss, best_j1, best_j2, best_j3, best_j4, best_j5, best_j6 = explore_and_evaluate_6d(best_j1, best_j2, best_j3, best_j4, best_j5)



close(train_data_file)
close(dev_data_file)