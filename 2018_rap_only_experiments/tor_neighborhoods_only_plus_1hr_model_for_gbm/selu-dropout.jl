# Self-normalizing SELU NN

USE_NORMALIZING_FACTORS = true


# Based on https://github.com/FluxML/Flux.jl/blob/master/src/layers/normalisation.jl
#
# But with an offset zero as recommended for SELU by https://arxiv.org/pdf/1706.02515.pdf

mutable struct AlphaDropout{F}
  p::F
  active::Bool
end

function AlphaDropout(p)
  @assert 0 ≤ p ≤ 1
  AlphaDropout{typeof(p)}(p, true)
end


const λ = 1.0507009873554804934193349852946
const α = 1.6732632423543772848170429916717
const α_prime = -λ*α

function (a::AlphaDropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  q = 1 - a.p

  @inbounds for i=1:length(y)
    y[i] = y[i] > a.p ? 1 / q : α_prime
  end
  return y .* x
end

_testmode!(a::AlphaDropout, test) = (a.active = !test)


hidden_size  = 50
dropout_rate = 0.10

model = Chain(
  Dense(FEATURE_COUNT, hidden_size, selu),
  AlphaDropout(dropout_rate), # https://arxiv.org/pdf/1706.02515.pdf recommends values of 0.05 to 0.10
  Dense(hidden_size, hidden_size, selu),
  AlphaDropout(dropout_rate),
  Dense(hidden_size, hidden_size, selu),
  AlphaDropout(dropout_rate),
  Dense(hidden_size, 1, σ),
  (x -> x[1])
)

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 10.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.binarycrossentropy

function show_extra_training_info()
  nothing
end