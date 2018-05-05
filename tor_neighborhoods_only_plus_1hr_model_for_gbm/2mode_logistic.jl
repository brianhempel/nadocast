println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true

struct Modes{L}
  b1::L

  f0::L
  f1::L
end

Flux.treelike(Modes)

function (a::Modes)(x)
  f0, f1 = a.f0, a.f1

  b1 = 0.2 * a.b1(x)

  (0.4 + b1) .* f0(x) .+ (0.6 - b1) .* f1(x)
end

model =
  Modes(
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ)
  )

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 10.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.binarycrossentropy

function model_prediction(x)
  global model
  y = model(x)
  Flux.Tracker.data(y[1])
end


function show_extra_training_info()
  nothing
end

function model_load(saved_bson_path)
  global model
  global optimizer
  @load saved_bson_path model optimizer
end
