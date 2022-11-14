module Metrics

using Random

# f should be a function that take an indices_range and returns a tuple of reduction values
#
# parallel_iterate will unzip those tuples into a tuple of arrays of reduction values and return that.
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
  Threads.@threads :static for i in 1:length(arr)
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

  Threads.@threads :static for _ in 1:Threads.nthreads()
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

  Threads.@threads :static for _ in 1:Threads.nthreads()
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
  Threads.@threads :static for i in 1:length(arr)
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

  Threads.@threads :static for _ in 1:Threads.nthreads()
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

  Threads.@threads :static for _ in 1:Threads.nthreads()
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

  @inbounds Threads.@threads :static for i in 1:length(perm)
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
      pos_weight += Float64(weights_sorted[i] * y_sorted[i])
      neg_weight += Float64(weights_sorted[i] * (1f0 - y_sorted[i]))
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
      true_pos_weight  -= Float64(weights_sorted[i] * y_sorted[i])
      false_pos_weight -= Float64(weights_sorted[i] * (1f0 - y_sorted[i]))
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
  # tpr = true_pos/total_pos = POD = recall
  # fpr = false_pos/total_neg
  # ROC is tpr vs fpr

  thread_pos_weights, thread_neg_weights = parallel_iterate(length(y)) do thread_range
    pos_weight = 0.0
    neg_weight = 0.0
    @inbounds for i in thread_range
      j = sort_perm[i]
      pos_weight += Float64(weights[j] * y[j])
      neg_weight += Float64(weights[j] * (1f0 - y[j]))
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
      true_pos_weight  -= Float64(weights[j] * y[j])
      false_pos_weight -= Float64(weights[j] * (1f0 - y[j]))
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
  thresh               = -one(ŷ_sorted[1])
  last_sr              = true_pos_weight / predicted_pos_weight
  last_pod             = true_pos_weight / total_pos_weight
  area                 = 0.0

  i = 1
  @inbounds while i <= length(y_sorted)
    thresh = ŷ_sorted[i]
    @inbounds while i <= length(y_sorted) && ŷ_sorted[i] == thresh
      true_pos_weight -= Float64(weights_sorted[i] * y_sorted[i])
      predicted_pos_weight -= Float64(weights_sorted[i])
      i += 1
    end

    sr  = true_pos_weight / (predicted_pos_weight + ε)
    pod = true_pos_weight / total_pos_weight

    if pod != last_pod
      # area += (last_pod - pod) * (0.5 * sr + 0.5 * last_sr) # interpolated version
      area += (last_pod - pod) * last_sr # stairstep version
    end

    last_sr  = sr
    last_pod = pod
  end

  # close the curve:
  pod = 0.0
  # area += (last_pod - pod) * (0.5 * 0.0 + 0.5 * last_sr) # interpolated version
  area += (last_pod - pod) * last_sr # stairstep version

  Float32(area)
end

function _au_pr_curve_interploated(ŷ_sorted, y_sorted, weights_sorted, total_pos_weight, total_weight)
  # Arrays are sorted from lowest score to highest
  # Assume everything above threshold is predicted
  # (Starting POD      = 100%)
  true_pos_weight      = total_pos_weight
  predicted_pos_weight = total_weight
  thresh               = -one(ŷ_sorted[1])
  last_sr              = true_pos_weight / predicted_pos_weight
  last_pod             = true_pos_weight / total_pos_weight
  area                 = 0.0

  i = 1
  @inbounds while i <= length(y_sorted)
    @assert ŷ_sorted[i] > thresh
    thresh = ŷ_sorted[i]
    @inbounds while i <= length(y_sorted) && ŷ_sorted[i] == thresh
      true_pos_weight -= Float64(weights_sorted[i] * y_sorted[i])
      predicted_pos_weight -= Float64(weights_sorted[i])
      i += 1
    end

    true_pos_weight      = max(0.0, true_pos_weight)
    predicted_pos_weight = max(0.0, predicted_pos_weight)

    sr  = true_pos_weight / (predicted_pos_weight + ε)
    pod = true_pos_weight / total_pos_weight

    if pod != last_pod
      @assert pod < last_pod
      area += (last_pod - pod) * (0.5 * sr + 0.5 * last_sr) # interpolated version
      # area += (last_pod - pod) * last_sr # stairstep version
    end

    last_sr  = sr
    last_pod = pod
  end

  # close the curve:
  pod = 0.0
  area += (last_pod - pod) * (0.5 * 0.0 + 0.5 * last_sr) # interpolated version
  # area += (last_pod - pod) * last_sr # stairstep version

  Float32(area)
end

# Area under the precison-recall curve (success ratio vs probability of detection)
# = area to the left of the performance diagram curve
#
# Better than ROC-AUC when the classes are imbalanced
#
# The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
# Takaya Saito, Marc Rehmsmeier
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
#
# The area under the precision-recall curve as a performance metric for rare binary events
# Helen R. Sofaer, Jennifer A. Hoeting, Catherine S. Jarnevich
# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13140#:~:text=Guo%2C%202013).-,The%20area%20under%20the%20precision%2Drecall%20curve%20(AUC%2DPR,Davis%20%26%20Goadrich%2C%202006).
#
# The B-ROC curve is also the same, with different axes orientations.
# B-ROC Curves for the Assessment of Classifiers over Imbalanced Data Sets
# Alvaro A. Cárdenas and John S. Baras
#
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
      pos_weight += Float64(weights_sorted[i] * y_sorted[i])
      neg_weight += Float64(weights_sorted[i] * (1f0 - y_sorted[i]))
    end
    pos_weight, neg_weight
  end

  total_pos_weight = sum(thread_pos_weights)
  total_weight     = total_pos_weight + sum(thread_neg_weights)

  _au_pr_curve(ŷ_sorted, y_sorted, weights_sorted, total_pos_weight, total_weight)
end

# Bin the data instead of sorting it.
function area_under_pr_curve_fast(ŷ, y, weights; bin_count = 1000)
  threads_bin_Σŷ, threads_bin_Σy, threads_bin_Σweights = Metrics.parallel_iterate(length(y)) do thread_range
    bin_Σŷ       = map(bin_i -> (bin_i - 0.5) / bin_count * eps(Float64), 1:bin_count) # ensure empty bins still yield sorted ŷs
    bin_Σy       = zeros(Float64, bin_count)
    bin_Σweights = fill(eps(Float64), bin_count)

    @inbounds for i in thread_range
      bin_i = Int64(floor(ŷ[i] * bin_count)) + 1
      bin_i = min(bin_i, bin_count) # if ŷ[i] == 1.0
      bin_Σŷ[bin_i]       += ŷ[i] * weights[i]
      bin_Σy[bin_i]       += y[i] * weights[i]
      bin_Σweights[bin_i] += weights[i]
    end

    bin_Σŷ, bin_Σy, bin_Σweights
  end

  Σbin_Σŷ       = sum(threads_bin_Σŷ)
  Σbin_Σy       = sum(threads_bin_Σy)
  Σbin_Σweights = sum(threads_bin_Σweights)

  @assert length(Σbin_Σweights) == bin_count

  _au_pr_curve_interploated(Σbin_Σŷ ./ Σbin_Σweights, Σbin_Σy ./ Σbin_Σweights, Σbin_Σweights, sum(Σbin_Σy), sum(Σbin_Σweights))
end


# event_names is list of keys of Ys, one per column in Ŷ
function reliability_curves_midpoints(bin_count, Ŷ, Ys, event_names, weights, col_names=event_names)

  nmodels = size(Ŷ, 2)

  bins_Σŷ      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σy      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σweight = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_max     = map(_ -> ones(Float32, bin_count), 1:nmodels)

  for prediction_i in 1:nmodels
    y                     = Ys[event_names[prediction_i]]
    ŷ                     = @view Ŷ[:,prediction_i];
    sort_perm             = Metrics.parallel_sort_perm(ŷ);
    y_sorted              = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted              = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted        = Metrics.parallel_apply_sort_perm(weights, sort_perm);
    total_positive_weight = sum(Metrics.parallel_iterate(is -> sum(Float64.(view(y, is) .* view(weights, is))), length(y)))
    per_bin_pos_weight    = total_positive_weight / bin_count

    bin_i = 1
    for i in eachindex(y_sorted)
      if ŷ_sorted[i] > bins_max[prediction_i][bin_i]
        bin_i += 1
      end

      bins_Σŷ[prediction_i][bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
      bins_Σy[prediction_i][bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
      bins_Σweight[prediction_i][bin_i] += Float64(weights_sorted[i])

      if bins_Σy[prediction_i][bin_i] >= per_bin_pos_weight
        bins_max[prediction_i][bin_i] = ŷ_sorted[i]
      end
    end
  end

  for prediction_i in 1:nmodels
    print("ŷ_$(col_names[prediction_i]),y_$(col_names[prediction_i]),")
  end
  println()

  for bin_i in 1:bin_count
    for prediction_i in 1:nmodels
      Σŷ      = bins_Σŷ[prediction_i][bin_i]
      Σy      = bins_Σy[prediction_i][bin_i]
      Σweight = bins_Σweight[prediction_i][bin_i]

      mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      print("$(Float32(mean_ŷ)),$(Float32(mean_y)),")
    end
    println()
  end
end


# CSI = hits / (hits + false alarms + misses)
#     = true_pos_weight / (true_pos_weight + false_pos_weight + false_negative_weight)
#     = true_pos_weight / (painted_weight + false_negative_weight)
#     = 1 / (1/POD + 1/(1-FAR) - 1)

function csi(ŷ, y, weights, threshold)

  @assert length(y) == length(weights)
  @assert length(y) == length(ŷ)

  thread_painted_weights, thread_true_pos_weights, thread_false_neg_weights = parallel_iterate(length(y)) do thread_range
    painted_weight   = 0.0
    true_pos_weight  = 0.0
    false_neg_weight = 0.0

    @inbounds for i in thread_range
      if ŷ[i] >= threshold
        painted_weight += Float64(weights[i])
        true_pos_weight += Float64(weights[i] * y[i])
      else
        false_neg_weight += Float64(weights[i] * y[i])
      end
    end

    painted_weight, true_pos_weight, false_neg_weight
  end

  sum(thread_true_pos_weights) / (sum(thread_painted_weights) +  sum(thread_false_neg_weights))
end


function multiple_csis(ŷ, y, weights; sort_perm = sortperm(ŷ; alg = Base.Sort.MergeSort), total_weight = sum(Float64.(weights)), positive_weight = sum(y .* Float64.(weights)))
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
    true_pos_weight  -= weights[i] * y[i]
    false_neg_weight += weights[i] * y[i]
    false_pos_weight -= weights[i] * (1f0 - y[i])

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
  csis = multiple_csis(ŷ, y, weights)
  Float32(sum(csis) / length(csis))
end




function success_ratio(ŷ, y, weights, threshold)
  @assert length(y) == length(weights)
  @assert length(y) == length(ŷ)

  thread_painted_weights, thread_true_pos_weights = parallel_iterate(length(y)) do thread_range
    painted_weight  = 0.0
    true_pos_weight = 0.0

    @inbounds for i in thread_range
      if ŷ[i] >= threshold
        painted_weight  += Float64(weights[i])
        true_pos_weight += Float64(weights[i] * y[i])
      end
    end

    painted_weight, true_pos_weight
  end

  sum(thread_true_pos_weights) / sum(thread_painted_weights)
end

function probability_of_detection(ŷ, y, weights, threshold)
  @assert length(y) == length(weights)
  @assert length(y) == length(ŷ)

  thread_pos_weights, thread_true_pos_weights = parallel_iterate(length(y)) do thread_range
    pos_weight      = 0.0
    true_pos_weight = 0.0

    @inbounds for i in thread_range
      pos_weight += Float64(weights[i] * y[i])
      if ŷ[i] >= threshold
        true_pos_weight += Float64(weights[i] * y[i])
      end
    end

    pos_weight, true_pos_weight
  end

  sum(thread_true_pos_weights) / sum(thread_pos_weights)
end

# assumes weights is proportional to gridpoint area
function warning_ratio(ŷ, weights, threshold)
  @assert length(ŷ) == length(weights)

  thread_total_weight, thread_warned_weight = parallel_iterate(length(ŷ)) do thread_range
    total_weight  = 0.0
    warned_weight = 0.0

    @inbounds for i in thread_range
      total_weight += Float64(weights[i])
      if ŷ[i] >= threshold
        warned_weight += Float64(weights[i])
      end
    end

    total_weight, warned_weight
  end

  sum(thread_warned_weight) / sum(thread_total_weight)
end

end # module Metrics

# push!(LOAD_PATH, (@__DIR__))
# import Metrics
# import Random
# Random.seed!(0)
# y = rand(Bool, 100000)
# ŷ = y .* (1 .- rand(100000) .* rand(100000)) .+ (1 .- y) .* (rand(100000) .* rand(100000))
# weights = rand(100000)
# println(Metrics.area_under_pr_curve(ŷ, y, weights))


# Random.seed!(0)
# y = rand(Bool, 10_000_000)
# ŷ = y .* (1 .- rand(10_000_000) .* rand(10_000_000)) .+ (1 .- y) .* (rand(10_000_000) .* rand(10_000_000))
# weights = rand(10_000_000)
# Metrics.area_under_pr_curve(ŷ, y, weights)
# @time Metrics.area_under_pr_curve(ŷ, y, weights)


# push!(LOAD_PATH, "models/shared")
# import Metrics
# import Random
# Random.seed!(123)
# y = Float32.(rand(10) .> 0.5)
# ŷ = Float32.(rand(10))
# weights = Float32.(0.3 .+ rand(10))

# println(Metrics.roc_auc(ŷ, y, weights))
# # 0.5465203558941252
# println(Metrics.roc_auc_less_mem(ŷ, y, weights))
# # 0.5465203558941252
# println(Metrics.area_under_pr_curve(ŷ, y, weights))
# # 0.6239406578120267

# # example from "The Critical Success Index as an Indicator of Warning Skill" Schaeffer 1990
# # CSI should be 0.228
# y1 = vcat(ones(28), zeros(72), zeros(2680), ones(23))
# ŷ1 = vcat(ones(28), ones(72), zeros(2680), zeros(23))
# weights1 = ones(length(y1))
# println(Metrics.csi(ŷ1, y1, weights1, 0.5))

# println(Metrics.multiple_csis(ŷ, y, weights))
# # [0.426800418229519, 0.426800418229519, 0.36645769907418074, 0.36645769907418074, 0.2810901274487399, 0.2810901274487399, 0.18663856431533987, 0.18663856431533987, 0.0]

# println(Metrics.success_ratio(ŷ, y, weights, 0.5f0))
# # 0.5708225878059436
# println(Metrics.success_ratio(ŷ, y, weights, 0.7f0))
# # 0.6555857340032812

# println(Metrics.probability_of_detection(ŷ, y, weights, 0.5f0))
# # 0.7631754079849923
# println(Metrics.probability_of_detection(ŷ, y, weights, 0.7f0))
# # 0.5767308993912318

