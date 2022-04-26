module Metrics

using Random

# f should be a function that take an indices_range and returns a tuple of reduction values
#
# parallel_iterate will unzip those tuples into a tuple of arrays of reduction values and return that.
function parallel_iterate(f, count)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads for thread_i in 1:Threads.nthreads()
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

# Sample sort
function parallel_sort_perm_i64(arr)
  sample_count = Threads.nthreads() * 20
  if length(arr) < sample_count || Threads.nthreads() == 1
    return sortperm(arr; alg = Base.Sort.MergeSort)
  end

  rng = MersenneTwister(1234);

  samples = sort(map(_ -> arr[rand(rng, 1:length(arr))], 1:sample_count))

  bin_splits = map(thread_i -> samples[Int64(round(thread_i/Threads.nthreads()*sample_count))], 1:(Threads.nthreads() - 1))

  thread_bins_bins = map(_ -> map(_ -> Int64[], 1:Threads.nthreads()), 1:Threads.nthreads())
  Threads.@threads for i in 1:length(arr)
    thread_bins = thread_bins_bins[Threads.threadid()]

    x = arr[i]
    bin_i = Threads.nthreads()
    @inbounds for k in 1:length(bin_splits)
      if bin_splits[k] > x
        bin_i = k
        break
      end
    end

    push!(thread_bins[bin_i], Int64(i))
  end

  outs = map(_ -> Int64[], 1:Threads.nthreads())

  Threads.@threads for _ in 1:Threads.nthreads()
    my_thread_bins = map(thread_i -> thread_bins_bins[thread_i][Threads.threadid()], 1:Threads.nthreads())

    my_out = Vector{Int64}(undef, sum(length.(my_thread_bins)))

    my_i = 1
    for j = 1:length(my_thread_bins)
      bin = my_thread_bins[j]
      if length(bin) > 0
        my_out[my_i:(my_i + length(bin)-1)] = bin
        my_i += length(bin)
      end
    end

    sort!(my_out; alg = Base.Sort.MergeSort, by = (i -> arr[i]))
    outs[Threads.threadid()] = my_out
  end

  out = Vector{Int64}(undef, length(arr))

  Threads.@threads for _ in 1:Threads.nthreads()
    my_out = outs[Threads.threadid()]

    start_i = sum(length.(outs)[1:Threads.threadid()-1]) + 1

    out[start_i:(start_i+length(my_out)-1)] = my_out
  end

  out
end

# Sample sort, UInt32 for indices
function parallel_sort_perm_u32(arr)
  sample_count = Threads.nthreads() * 20
  if length(arr) < sample_count || Threads.nthreads() == 1
    return sortperm(arr; alg = Base.Sort.MergeSort)
  end

  rng = MersenneTwister(1234);

  samples = sort(map(_ -> arr[rand(rng, 1:length(arr))], 1:sample_count))

  bin_splits = map(thread_i -> samples[Int64(round(thread_i/Threads.nthreads()*sample_count))], 1:(Threads.nthreads() - 1))

  thread_bins_bins = map(_ -> map(_ -> UInt32[], 1:Threads.nthreads()), 1:Threads.nthreads())
  Threads.@threads for i in 1:length(arr)
    thread_bins = thread_bins_bins[Threads.threadid()]

    x = arr[i]
    bin_i = Threads.nthreads()
    @inbounds for k in 1:length(bin_splits)
      if bin_splits[k] > x
        bin_i = k
        break
      end
    end

    push!(thread_bins[bin_i], UInt32(i))
  end

  outs = map(_ -> UInt32[], 1:Threads.nthreads())

  Threads.@threads for _ in 1:Threads.nthreads()
    my_thread_bins = map(thread_i -> thread_bins_bins[thread_i][Threads.threadid()], 1:Threads.nthreads())

    my_out = Vector{UInt32}(undef, sum(length.(my_thread_bins)))

    my_i = 1
    for j = 1:length(my_thread_bins)
      bin = my_thread_bins[j]
      if length(bin) > 0
        my_out[my_i:(my_i + length(bin)-1)] = bin
        my_i += length(bin)
      end
    end

    sort!(my_out; alg = Base.Sort.MergeSort, by = (i -> arr[i]))
    outs[Threads.threadid()] = my_out
  end

  out = Vector{UInt32}(undef, length(arr))

  Threads.@threads for _ in 1:Threads.nthreads()
    my_out = outs[Threads.threadid()]

    start_i = sum(length.(outs)[1:Threads.threadid()-1]) + 1

    out[start_i:(start_i+length(my_out)-1)] = my_out
  end

  out
end

function parallel_sort_perm(arr)
  length(arr) <= typemax(UInt32) ? parallel_sort_perm_u32(arr) : parallel_sort_perm_i64(arr)
end

function parallel_apply_sort_perm(arr, perm)
  out = Vector{eltype(arr)}(undef, length(arr))

  @inbounds Threads.@threads for i in 1:length(perm)
    out[i] = arr[perm[i]]
  end

  out
end

function parallel_float64_sum(arr)
  thread_sums = parallel_iterate(length(arr)) do thread_range
    thread_sum = 0.0
    @inbounds for i in thread_range
      thread_sum += Float64(arr[i])
    end
    thread_sum
  end
  sum(thread_sums)
end

# This calc erroneously assumes all ŷ are distinct
function roc_auc(ŷ, y, weights; sort_perm = parallel_sort_perm(ŷ))
  y_sorted       = Vector{eltype(y)}(undef, length(y))
  weights_sorted = Vector{eltype(weights)}(undef, length(weights))

  # tpr = true_pos/total_pos
  # fpr = false_pos/total_neg
  # ROC is tpr vs fpr

  thread_pos_weights, thread_neg_weights = parallel_iterate(length(y)) do thread_range
    pos_weight = 0.0
    neg_weight = 0.0
    @inbounds for i in thread_range
      j                 = sort_perm[i]
      y_sorted[i]       = y[j]
      weights_sorted[i] = weights[j]
      if y_sorted[i] > 0.5f0
        pos_weight += Float64(weights_sorted[i])
      else
        neg_weight += Float64(weights_sorted[i])
      end
    end
    pos_weight, neg_weight
  end

  total_pos_weight = sum(thread_pos_weights)
  total_neg_weight = sum(thread_neg_weights)

  thread_aucs = parallel_iterate(length(y)) do thread_range
    true_pos_weight  = sum(@view thread_pos_weights[Threads.threadid():Threads.nthreads()])
    false_pos_weight = sum(@view thread_neg_weights[Threads.threadid():Threads.nthreads()])

    auc = 0.0

    last_fpr = false_pos_weight / total_neg_weight
    @inbounds for i in thread_range
      if y_sorted[i] > 0.5f0
        true_pos_weight -= Float64(weights_sorted[i])
      else
        false_pos_weight -= Float64(weights_sorted[i])
      end
      fpr = false_pos_weight / total_neg_weight
      tpr = true_pos_weight  / total_pos_weight
      if fpr != last_fpr
        auc += (last_fpr - fpr) * tpr
      end
      last_fpr = fpr
    end

    auc
  end

  sum(thread_aucs)
end

# More cache misses, less allocation. Slightly slower on my machine
function roc_auc_less_mem(ŷ, y, weights; sort_perm = parallel_sort_perm(ŷ))
  # tpr = true_pos/total_pos
  # fpr = false_pos/total_neg
  # ROC is tpr vs fpr

  thread_pos_weights, thread_neg_weights = parallel_iterate(length(y)) do thread_range
    pos_weight = 0.0
    neg_weight = 0.0
    @inbounds for i in thread_range
      j = sort_perm[i]
      if y[j] > 0.5f0
        pos_weight += Float64(weights[j])
      else
        neg_weight += Float64(weights[j])
      end
    end
    pos_weight, neg_weight
  end

  total_pos_weight = sum(thread_pos_weights)
  total_neg_weight = sum(thread_neg_weights)

  thread_aucs = parallel_iterate(length(y)) do thread_range
    true_pos_weight  = sum(@view thread_pos_weights[Threads.threadid():Threads.nthreads()])
    false_pos_weight = sum(@view thread_neg_weights[Threads.threadid():Threads.nthreads()])

    auc = 0.0

    last_fpr = false_pos_weight / total_neg_weight
    @inbounds for i in thread_range
      j = sort_perm[i]
      if y[j] > 0.5f0
        true_pos_weight -= Float64(weights[j])
      else
        false_pos_weight -= Float64(weights[j])
      end
      fpr = false_pos_weight / total_neg_weight
      tpr = true_pos_weight  / total_pos_weight
      if fpr != last_fpr
        auc += (last_fpr - fpr) * tpr
      end
      last_fpr = fpr
    end

    auc
  end

  sum(thread_aucs)
end


const ε = eps(0.1)

# This part is fast and does not need to be threaded.
function _au_pr_curve(ŷ_sorted, y_sorted, weights_sorted, total_pos_weight, total_weight)
  # Arrays are sorted from lowest score to highest
  # Assume everything above threshold is predicted
  # (Starting POD      = 100%)
  true_pos_weight      = total_pos_weight
  predicted_pos_weight = total_weight
  thresh               = zero(ŷ_sorted[1])
  last_sr              = true_pos_weight / predicted_pos_weight
  last_pod             = true_pos_weight / total_pos_weight
  area                 = 0.0

  i = 1
  @inbounds while i <= length(y_sorted)
    thresh = ŷ_sorted[i]
    @inbounds while i <= length(y_sorted) && ŷ_sorted[i] == thresh
      if y_sorted[i] > 0.5f0
        true_pos_weight -= Float64(weights_sorted[i])
      end
      predicted_pos_weight -= Float64(weights_sorted[i])
      i += 1
    end

    sr  = true_pos_weight / (predicted_pos_weight + ε)
    pod = true_pos_weight / total_pos_weight

    if pod != last_pod
      area += (last_pod - pod) * (0.5 * sr + 0.5 * last_sr)
    end

    last_sr  = sr
    last_pod = pod
  end

  area
end

# Area under the precison-recall curve (success ratio vs probability of detection)
# = area to the left of the performance diagram curve
# This calc correctly handles when ŷ are not distinct
function area_under_pr_curve(ŷ, y, weights; sort_perm = parallel_sort_perm(ŷ))
  ŷ_sorted       = Vector{eltype(ŷ)}(undef, length(ŷ))
  y_sorted       = Vector{eltype(y)}(undef, length(y))
  weights_sorted = Vector{eltype(weights)}(undef, length(weights))

  thread_pos_weights, thread_neg_weights = parallel_iterate(length(y)) do thread_range
    pos_weight = 0.0
    neg_weight = 0.0
    @inbounds for i in thread_range
      j                 = sort_perm[i]
      y_sorted[i]       = y[j]
      ŷ_sorted[i]       = ŷ[j]
      weights_sorted[i] = weights[j]
      if y_sorted[i] > 0.5f0
        pos_weight += Float64(weights_sorted[i])
      else
        neg_weight += Float64(weights_sorted[i])
      end
    end
    pos_weight, neg_weight
  end

  total_pos_weight = sum(thread_pos_weights)
  total_weight     = total_pos_weight + sum(thread_neg_weights)

  _au_pr_curve(ŷ_sorted, y_sorted, weights_sorted, total_pos_weight, total_weight)
end


# CSI = hits / (hits + false alarms + misses)
#     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)
#     = 1 / (1/POD + 1/(1-FAR) - 1)

function csi(ŷ, y, weights; sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort), total_weight = sum(Float64.(weights)), positive_weight = sum(y .* Float64.(weights)))
  y       = y[sort_perm]
  ŷ       = ŷ[sort_perm]
  weights = Float64.(weights[sort_perm])

  negative_weight = total_weight - positive_weight

  true_pos_weight  = positive_weight
  false_pos_weight = negative_weight
  false_neg_weight = 0.0

  # CSI = hits / (hits + false alarms + misses)
  #     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)

  pods = Float64[true_pos_weight / positive_weight]
  csis = Float64[true_pos_weight / (true_pos_weight + false_pos_weight + false_neg_weight)]

  for i in 1:length(y)
    if y[i] > 0.5f0
      true_pos_weight  -= weights[i]
      false_neg_weight += weights[i]
    else
      false_pos_weight -= weights[i]
    end

    pod = true_pos_weight / positive_weight
    csi = true_pos_weight / (true_pos_weight + false_pos_weight + false_neg_weight)

    push!(pods, pod)
    push!(csis, csi)
  end

  # CSIs for PODs 0.9, 0.8, ..., 0.1
  map(collect(0.9:-0.1:0.1)) do pod_threshold
    i = findfirst(pod -> pod < pod_threshold, pods)
    csis[i]
  end
end

function mean_csi(ŷ, y, weights)
  csis = csi(ŷ, y, weights)
  Float32(sum(csis) / length(csis))
end

end # module Metrics

# push!(LOAD_PATH, (@__DIR__))
# import Metrics
# import Random
# Random.seed!(0)
# y = rand(Bool, 100000)
# ŷ = y .* (1 .- rand(100000) .* rand(100000)) .+ (1 .- y) .* (rand(100000) .* rand(100000))
# weights = rand(100000)
# Metrics.roc_auc(ŷ, y, weights)
# Metrics.roc_auc_less_mem(ŷ, y, weights)
# # 0.9277449649005693

# Random.seed!(0)
# y = rand(Bool, 100_000_000)
# ŷ = y .* (1 .- rand(100_000_000) .* rand(100_000_000)) .+ (1 .- y) .* (rand(100_000_000) .* rand(100_000_000))
# weights = rand(100_000_000)
# Metrics.roc_auc(ŷ, y, weights)
# Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)

# Random.seed!(0)
# y = Float32.(rand(Bool, 6_000_000))
# ŷ = Float32.(y .* (1 .- rand(6_000_000) .* rand(6_000_000)) .+ (1 .- y) .* (rand(6_000_000) .* rand(6_000_000)))
# weights = Float32.(rand(6_000_000))
# Metrics.roc_auc(ŷ, y, weights)
# Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)
# @time Metrics.roc_auc(ŷ, y, weights)
# @time Metrics.roc_auc_less_mem(ŷ, y, weights)
