println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true

struct Factors{L}
  gate::L
  cape0a::L
  cape250a::L
  cape500a::L
  cape750a::L
  cape1000a::L
  cape1250a::L
  cape1500a::L
  cape2000a::L
  cape2500a::L
  cape3500a::L
  cape5000a::L
  cape0b::L
  cape250b::L
  cape500b::L
  cape750b::L
  cape1000b::L
  cape1250b::L
  cape1500b::L
  cape2000b::L
  cape2500b::L
  cape3500b::L
  cape5000b::L
end

Flux.treelike(Factors)

function (a::Factors)(x)
  global normalizing_factors
  sbcape = x[32] * normalizing_factors[32] # storm path mean

  a.gate(x) .*
    if sbcape < 250.0
      w = sbcape / 250.0
      (1.0 - w) * a.cape0a(x) .* a.cape0b(x) .+ w * a.cape250a(x) .* a.cape250b(x)
    elseif sbcape < 500.0
      w = (sbcape - 250.0) / 250.0
      (1.0 - w) * a.cape250a(x) .* a.cape250b(x) .+ w * a.cape500a(x) .* a.cape500b(x)
    elseif sbcape < 750.0
      w = (sbcape - 500.0) / 250.0
      (1.0 - w) * a.cape500a(x) .* a.cape500b(x) .+ w * a.cape750a(x) .* a.cape750b(x)
    elseif sbcape < 1000.0
      w = (sbcape - 750.0) / 250.0
      (1.0 - w) * a.cape750a(x) .* a.cape750b(x) .+ w * a.cape1000a(x) .* a.cape1000b(x)
    elseif sbcape < 1250.0
      w = (sbcape - 1000.0) / 250.0
      (1.0 - w) * a.cape1000a(x) .* a.cape1000b(x) .+ w * a.cape1250a(x) .* a.cape1250b(x)
    elseif sbcape < 1500.0
      w = (sbcape - 1250.0) / 250.0
      (1.0 - w) * a.cape1250a(x) .* a.cape1250b(x) .+ w * a.cape1500a(x) .* a.cape1500b(x)
    elseif sbcape < 2000.0
      w = (sbcape - 1500.0) / 500.0
      (1.0 - w) * a.cape1500a(x) .* a.cape1500b(x) .+ w * a.cape2000a(x) .* a.cape2000b(x)
    elseif sbcape < 2500.0
      w = (sbcape - 2000.0) / 500.0
      (1.0 - w) * a.cape2000a(x) .* a.cape2000b(x) .+ w * a.cape2500a(x) .* a.cape2500b(x)
    elseif sbcape < 3500.0
      w = (sbcape - 2500.0) / 1000.0
      (1.0 - w) * a.cape2500a(x) .* a.cape2500b(x) .+ w * a.cape3500a(x) .* a.cape3500b(x)
    elseif sbcape < 5000.0
      w = (sbcape - 3500.0) / 1500.0
      (1.0 - w) * a.cape3500a(x) .* a.cape3500b(x) .+ w * a.cape5000a(x) .* a.cape5000b(x)
    else
      a.cape5000a(x) .* a.cape5000b(x)
    end
end

model =
  Factors(
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ)
  )

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 3.5 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
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
