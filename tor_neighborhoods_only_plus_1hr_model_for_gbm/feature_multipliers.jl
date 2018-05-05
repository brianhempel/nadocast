println("Loading Flux...")
using Flux
using BSON: @load
import StatsBase

USE_NORMALIZING_FACTORS = true

struct Modes{L}
  w::L
  b::L
end

Flux.treelike(Modes)

function (a::Modes)(x)
  w, b = a.w, a.b

  gates = σ.(w .* x .+ b)

  StatsBase.geomean(gates)
  # prod(gates)
end

model =
  Modes(
    param(Flux.glorot_uniform(FEATURE_COUNT)),
    param(zeros(FEATURE_COUNT))
  )

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 100.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
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
