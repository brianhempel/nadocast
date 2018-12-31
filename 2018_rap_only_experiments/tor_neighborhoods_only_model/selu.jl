# Self-normalizing SELU NN

hidden_size = 10

model = Chain(
  Dense(FEATURE_COUNT, hidden_size, selu),
  Dense(hidden_size, hidden_size, selu),
  Dense(hidden_size, 1, σ)
)

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 10.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.binarycrossentropy

function show_extra_training_info()
  nothing
end