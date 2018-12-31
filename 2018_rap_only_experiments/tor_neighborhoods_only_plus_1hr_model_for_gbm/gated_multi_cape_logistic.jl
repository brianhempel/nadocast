println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true

struct Factors{L}
  gate::L
  cape0::L
  cape250::L
  cape500::L
  cape750::L
  cape1000::L
  cape1250::L
  cape1500::L
  cape2000::L
  cape2500::L
  cape3500::L
  cape5000::L
end

Flux.treelike(Factors)

function (a::Factors)(x)
  global normalizing_factors
  sbcape = x[32] * normalizing_factors[32] # storm path mean

  a.gate(x) .*
    if sbcape < 250.0
      w = sbcape / 250.0
      (1.0 - w) * a.cape0(x) .+ w * a.cape250(x)
    elseif sbcape < 500.0
      w = (sbcape - 250.0) / 250.0
      (1.0 - w) * a.cape250(x) .+ w * a.cape500(x)
    elseif sbcape < 750.0
      w = (sbcape - 500.0) / 250.0
      (1.0 - w) * a.cape500(x) .+ w * a.cape750(x)
    elseif sbcape < 1000.0
      w = (sbcape - 750.0) / 250.0
      (1.0 - w) * a.cape750(x) .+ w * a.cape1000(x)
    elseif sbcape < 1250.0
      w = (sbcape - 1000.0) / 250.0
      (1.0 - w) * a.cape1000(x) .+ w * a.cape1250(x)
    elseif sbcape < 1500.0
      w = (sbcape - 1250.0) / 250.0
      (1.0 - w) * a.cape1250(x) .+ w * a.cape1500(x)
    elseif sbcape < 2000.0
      w = (sbcape - 1500.0) / 500.0
      (1.0 - w) * a.cape1500(x) .+ w * a.cape2000(x)
    elseif sbcape < 2500.0
      w = (sbcape - 2000.0) / 500.0
      (1.0 - w) * a.cape2000(x) .+ w * a.cape2500(x)
    elseif sbcape < 3500.0
      w = (sbcape - 2500.0) / 1000.0
      (1.0 - w) * a.cape2500(x) .+ w * a.cape3500(x)
    elseif sbcape < 5000.0
      w = (sbcape - 3500.0) / 1500.0
      (1.0 - w) * a.cape3500(x) .+ w * a.cape5000(x)
    else
      a.cape5000(x)
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
    Dense(FEATURE_COUNT, 1, σ)
  )

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 2.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
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
