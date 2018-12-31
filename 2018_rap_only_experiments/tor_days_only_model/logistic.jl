# Simple logistic regression

model     = Dense(FEATURE_COUNT, 1, σ)

# β2 is the memory of the variance. Need to give ADAM a lot of memory because our positive samples are so spread out.
optimizer = ADAM(params(model), 0.02 / 40000; β1 = 0.99, β2 = 0.9999999, ϵ = 1e-07)
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.mse
