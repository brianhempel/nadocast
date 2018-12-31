println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true

struct Factors{L}
  f1::L
  f2::L
  f3::L
  f4::L
end

Flux.treelike(Factors)

function (a::Factors)(x)
  f1, f2, f3, f4 = a.f1, a.f2, a.f3, a.f4

  f1(x) .* f2(x) .* f3(x) .* f4(x)
end

model = Factors(Dense(FEATURE_COUNT, 1, σ), Dense(FEATURE_COUNT, 1, σ), Dense(FEATURE_COUNT, 1, σ), Dense(FEATURE_COUNT, 1, σ))

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
