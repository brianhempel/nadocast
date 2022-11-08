module LogisticRegression

import LinearAlgebra

function parallel_iterate(f, count)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads :static for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * count, Threads.nthreads()) + 1
    stop  = div( thread_i    * count, Threads.nthreads())
    thread_results[thread_i] = f(start:stop)
  end

  if isa(thread_results[1], Tuple)
    # Mangling so you get a tuple of arrays.
    Tuple(collect.(zip(thread_results...)))
  else
    thread_results
  end
end

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0

σ(x) = one(x) / (one(x) + exp(-x))
logloss(y, ŷ) = -y*log(ŷ + ε) - (one(ŷ) - y)*log(one(ŷ) - ŷ + ε)
@inline ∇logloss(y, ŷ) = ŷ - y
@inline ∇∇logloss(ŷ)   = ŷ * (one(ŷ) - ŷ)

function Σdloss_bs_Σloss_hessian(X, y, weights, bs, iis)
  # Σloss         = zero(Loss)
  bs                         = Float64.(bs)
  feature_count              = length(bs)
  Σdloss_bs                  = zeros(Float64, feature_count)
  Σloss_hessian              = zeros(Float64, (feature_count, feature_count))
  feature_val                = zeros(Float64, feature_count)
  feature_val[feature_count] = 1.0

  @inbounds for ii in iis
    logit = bs[feature_count]
    for j in 1:(feature_count-1)
      feature_val[j] = Float64(X[ii, j])
      logit         += bs[j] * feature_val[j]
    end
    ŷ_i = σ(logit)
    y_i = Float64(y[ii])

    # Σloss += logloss(y_i, ŷ_i) * weights[ii]

    dloss  = ∇logloss(y_i, ŷ_i) * Float64(weights[ii])
    ddloss = ∇∇logloss(ŷ_i)     * Float64(weights[ii])

    for j in 1:feature_count
      Σdloss_bs[j] += dloss * feature_val[j]
    end

    # https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method#the-math-putting-it-all-together

    for i in 1:feature_count
      for j in 1:feature_count
        Σloss_hessian[i,j] += ddloss*feature_val[i]*feature_val[j]
      end
    end
  end

  (Σdloss_bs, Σloss_hessian)
end

function fit(X, y, weights; iteration_count = 30, l2_regularization = 0.0)
  total_weight = sum(Float64.(weights))

  feature_count = size(X, 2) + 1
  bs = zeros(Float32, feature_count)
  diagonalizer = LinearAlgebra.diagm(ones(Float64, feature_count))

  converged = false

  l2_reg_hessian = LinearAlgebra.diagm(ones(Float64, feature_count) .* l2_regularization)

  # println(bs)
  # println(total_logloss(X, bs, y, weights))
  for iteration_i = 1:iteration_count

    thread_Σdloss_bss, thread_Σloss_hessians = parallel_iterate(length(y)) do thread_range
      Σdloss_bs_Σloss_hessian(X, y, weights, bs, thread_range)
    end

    dloss_vec    = reduce(+, thread_Σdloss_bss)     ./ total_weight
    loss_hessian = reduce(+, thread_Σloss_hessians) ./ total_weight

    # println(dloss_vec)

    for j in 1:feature_count
      loss_hessian[j,j] += ε
    end

    # https://stats.stackexchange.com/a/156719
    dloss_vec    .+= l2_regularization .* bs
    loss_hessian .+= l2_reg_hessian

    step =
      try
        try
          inv(loss_hessian) * dloss_vec
        catch exception
          if isa(exception, LinearAlgebra.SingularException)
            # Diagonalize.
            inv(loss_hessian .* diagonalizer) * dloss_vec ./ length(bs)
          else
            rethrow()
          end
        end
      catch exception
        if isa(exception, LinearAlgebra.LAPACKException) # Hessian is all zeros.
          break
        else
          rethrow()
        end
      end

    # bs -= dloss_vec * 0.2
    bs -= Float32.(step)
    print("$bs\r")
    flush(stdout)
    # println(total_logloss(X, bs, y, weights))

    if (sum(abs.(step)) / length(step) < 1e-5 && sum(abs.(dloss_vec)) / length(dloss_vec) < 1e-5)
      converged = true
      break
    end
  end
  println()

  if !converged
    println("No convergence.")
  end

  bs
end

function predict(X, bs)
  feature_count = length(bs)
  ŷ = Vector{Float32}(undef, size(X, 1))

  parallel_iterate(length(ŷ)) do thread_range
    for i in thread_range
      logit = bs[feature_count]
      for j in 1:(feature_count-1)
        logit += X[i, j] * bs[j]
      end
      ŷ[i] = σ(logit)
    end
  end

  ŷ
end

# function total_logloss(X, bs, y, weights)
#   ŷ = predict(X, bs)
#   sum(logloss.(y, ŷ) .* weights) / sum(weights)
# end

# n = 100_000_000
# # n = 100_000
# X = rand(Float32, n,5)
# y = map(i -> (sum(X[i,:]) + 3*rand(Float32)) / (5+3) > 0.5 ? 1f0 : 0f0, 1:n)
# weights = rand(Float32, n) .+ 0.5f0

# bs = fit(X, y, weights)

# ŷ = predict(X, bs)

# for i in 1:n
#   println("$(y[i])\t$(ŷ[i])")
# end

end