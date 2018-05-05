# This experiment didn't work. Probably too many dimensions. Struggles to get below 0.14 training loss (VERY bad.)

# println("Loading Flux...")
# using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = false
DO_BACKPROP             = false

mutable struct KNN
  k::Int64
  w::Array{Float32,1} # Dimension weights, the only learned parameter
  nns::Array{Float32,2}
  nn_labels::Array{Float32,1}
end

# Flux.treelike(KNN)

function (knn::KNN)(x)
  ε = 1f-6
  k, w, nns, nn_labels = knn.k, knn.w, knn.nns, knn.nn_labels

  nns_available = size(nns,2)

  if nns_available > 0
    # L1 distance (did better than L2)

    distances = sum(abs.((x .- nns) .* w), 1)

    nn_is = sortperm(@view distances[1,:])

    if nns_available >= k
      ε + (1f0 - ε) * mean(nn_labels[@view nn_is[1:k]])
    else
      ε + (1f0 - ε) * mean(nn_labels)
    end
  else
    ε
  end
end


function find_random_i_with_label(labels, target_label)
  while true
    i = rand(1:length(labels))
    if labels[i] == target_label
      return i
    end
  end
end

function pretrain(data, labels, weights) # The weights will hardly matter but hey.
  global model
  global feature_normalizers

  println("Learning dimension weights for L1 distance metric")

  w = param(1.0f0 ./ feature_normalizers)

  optimizer = ADAM(params(w), 0.2 / 100_000)

  example_count = size(data, 2)

  checkpoint_loss = 0.0
  checkpoint_distance_counts = 0 :: Int64

  pos_checkpoint_loss = 0.0
  pos_checkpoint_distance_counts = 0 :: Int64

  neg_checkpoint_loss = 0.0
  neg_checkpoint_distance_counts = 0 :: Int64

  for iteration = 1:1_000_000
    # positive pair
    pi1 = find_random_i_with_label(labels, 1.0f0)
    pi2 = find_random_i_with_label(labels, 1.0f0)

    # opposite pair
    oi1 = find_random_i_with_label(labels, 0.0f0)
    oi2 = find_random_i_with_label(labels, 1.0f0)

    positive_distance = sum(abs.((data[:,pi2] .- data[:,pi1]) .* w))
    opposite_distance = sum(abs.((data[:,oi2] .- data[:,oi1]) .* w))

    if opposite_distance > 1e-10
      loss = positive_distance / opposite_distance
      back!(loss)
      optimizer()
      checkpoint_loss += Flux.Tracker.data(loss)
      checkpoint_distance_counts += 1
      pos_checkpoint_loss += Flux.Tracker.data(loss)
      pos_checkpoint_distance_counts += 1
    end

    # negative pair
    ni1 = find_random_i_with_label(labels, 0.0f0)
    ni2 = find_random_i_with_label(labels, 0.0f0)

    # opposite pair
    oi1 = find_random_i_with_label(labels, 0.0f0)
    oi2 = find_random_i_with_label(labels, 1.0f0)

    negative_distance = sum(abs.((data[:,ni2] .- data[:,ni1]) .* w))
    opposite_distance = sum(abs.((data[:,oi2] .- data[:,oi1]) .* w))

    if opposite_distance > 1e-10
      loss = negative_distance / opposite_distance
      back!(loss)
      optimizer()

      checkpoint_loss += Flux.Tracker.data(loss)
      checkpoint_distance_counts += 1
      neg_checkpoint_loss += Flux.Tracker.data(loss)
      neg_checkpoint_distance_counts += 1
    end

    if mod(iteration, 20_000) == 0 || iteration == 1
      println("$(iteration*2) backprops, same/opposite distance ratio: $(checkpoint_loss / checkpoint_distance_counts)\tneg/opposite $(neg_checkpoint_loss / neg_checkpoint_distance_counts)\tpos/opposite $(pos_checkpoint_loss / pos_checkpoint_distance_counts)")
      checkpoint_loss = 0.0
      checkpoint_distance_counts = 0
      pos_checkpoint_loss = 0.0
      pos_checkpoint_distance_counts = 0
      neg_checkpoint_loss = 0.0
      neg_checkpoint_distance_counts = 0
    end
  end

  w_raw = Array{Float32,1}(Flux.Tracker.data(w))

  show(w_raw)

  model =
    KNN(
      20,
      w_raw,
      zeros(Float32, FEATURE_COUNT,0),
      zeros(Float32, 0)
    )
end

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# optimizer = ADAM(params(model.w), 0.1 / points_per_epoch)
optimizer = ()
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.binarycrossentropy

ADD_RATE   = 0.2
TRIM_RATIO = 0.2

function update_model(model, x, label, example_loss)
  # Eventually want to calculate loss per nn and kick out the worst nn.
  if rand() < ADD_RATE * example_loss
    model.nns = hcat(model.nns, x)
    model.nn_labels = vcat(model.nn_labels, label)
    # push!(model.nn_labels, label)
  end
end

function trim_model_before_epoch(model)
  nns_count = size(model.nns,2)
  if nns_count > 0
    keep_count = Int64(ceil(nns_count * TRIM_RATIO))
    keep_is    = shuffle(1:nns_count)[1:keep_count]

    model.nns    = model.nns[:, keep_is]
    model.labels = model.labels[keep_is]
  end
end

function model_prediction(x)
  global model
  y = model(x)
  Flux.Tracker.data(y)
end

function show_extra_training_info()
  global model
  # println(Flux.Tracker.data(model.w))
  println(size(model.nns))
end

function model_load(saved_bson_path)
  # global model
  # global optimizer
  # @load saved_bson_path model optimizer
end
