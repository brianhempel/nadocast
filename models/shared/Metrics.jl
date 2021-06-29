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
function parallel_sort_perm(arr)
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
    for k in 1:length(bin_splits)
      if bin_splits[k] > x
        bin_i = k
        break
      end
    end

    push!(thread_bins[bin_i], i)
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

function parallel_apply_sort_perm(arr, perm)
  out = Vector{eltype(arr)}(undef, length(arr))

  Threads.@threads for i in 1:length(perm)
    out[i] = arr[perm[i]]
  end

  out
end

function parallel_float64_sum(arr)
  thread_sums = parallel_iterate(length(arr)) do thread_range
    thread_sum = 0.0
    for i in thread_range
      thread_sum += Float64(arr[i])
    end
    thread_sum
  end
  sum(thread_sums)
end

function roc_auc(ŷ, y, weights; sort_perm = parallel_sort_perm(ŷ), total_weight = parallel_float64_sum(weights), positive_weight = parallel_float64_sum(y .* weights))
  y       = parallel_apply_sort_perm(y, sort_perm)
  ŷ       = parallel_apply_sort_perm(ŷ, sort_perm)
  weights = Float64.(parallel_apply_sort_perm(weights, sort_perm))

  negative_weight  = total_weight - positive_weight
  true_pos_weight  = positive_weight
  false_pos_weight = negative_weight

  # tpr = true_pos/total_pos
  # fpr = false_pos/total_neg
  # ROC is tpr vs fpr

  auc = 0.0

  last_fpr = false_pos_weight / negative_weight # = 1.0
  for i in 1:length(y)
    if y[i] > 0.5f0
      true_pos_weight -= weights[i]
    else
      false_pos_weight -= weights[i]
    end
    fpr = false_pos_weight / negative_weight
    tpr = true_pos_weight  / positive_weight
    if fpr != last_fpr
      auc += (last_fpr - fpr) * tpr
    end
    last_fpr = fpr
  end

  auc
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
