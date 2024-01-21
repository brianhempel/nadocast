module Utils

import Dates

export parallel_map, parallel_filter, parallel_count, read_3rd_col, read_first_2_cols, read_first_3_cols, mean, tally, concat, concat_map, count_unique, seconds_to_hour_i, seconds_to_fourhour_i, seconds_to_convective_day_i, time_to_hour_i, time_to_fourhour_i, time_to_convective_day_i

function parallel_map(f, xs)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * length(xs), Threads.nthreads()) + 1
    stop  = div( thread_i    * length(xs), Threads.nthreads())
    thread_results[thread_i] = map(f, @view xs[start:stop])
  end

  out = vcat(thread_results...)
  @assert length(out) == length(xs)
  out
end

function parallel_filter(f, xs)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * length(xs), Threads.nthreads()) + 1
    stop  = div( thread_i    * length(xs), Threads.nthreads())
    thread_results[thread_i] = filter(f, @view xs[start:stop])
  end

  vcat(thread_results...)
end

function parallel_count(f, xs)
  thread_counts = Vector{Int64}(undef, Threads.nthreads())

  Threads.@threads for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * length(xs), Threads.nthreads()) + 1
    stop  = div( thread_i    * length(xs), Threads.nthreads())
    thread_counts[thread_i] = count(f, @view xs[start:stop])
  end

  sum(thread_counts)
end

# Read 3rd column of a lat,lon,val CSV
function read_3rd_col(path) :: Vector{Float64}
  grid_latlons = Tuple{Float64,Float64}[]
  grid_vals    = Float64[]

  headers = nothing
  for line in eachline(path)
    if isnothing(headers)
      headers = split(line, ',')
      @assert length(headers) == 3
      @assert headers[1] == "lat"
      @assert headers[2] == "lon"
      continue
    end
    lat, lon, val = parse.(Float64, split(line, ','))

    push!(grid_latlons, (lat, lon))
    push!(grid_vals, val)
  end

  grid_vals
end

# Read latlons of a lat,lon,val CSV
function read_first_2_cols(path) :: Vector{Tuple{Float64,Float64}}
  grid_latlons = Tuple{Float64,Float64}[]
  grid_vals    = Float64[]

  headers = nothing
  for line in eachline(path)
    if isnothing(headers)
      headers = split(line, ',')
      @assert length(headers) == 3
      @assert headers[1] == "lat"
      @assert headers[2] == "lon"
      continue
    end
    lat, lon, val = parse.(Float64, split(line, ','))

    push!(grid_latlons, (lat, lon))
    push!(grid_vals, val)
  end

  grid_latlons
end

function read_first_3_cols(path) :: Vector{Tuple{Float64,Float64,Float64}}
  tuples = Tuple{Float64,Float64,Float64}[]

  headers = nothing
  for line in eachline(path)
    if isnothing(headers)
      headers = split(line, ',')
      @assert length(headers) == 3
      @assert headers[1] == "lat"
      @assert headers[2] == "lon"
      continue
    end
    lat, lon, val = parse.(Float64, split(line, ','))

    push!(tuples, (lat, lon, val))
  end

  tuples
end



mean(xs) = sum(xs) / length(xs)

# Count number of each element. It's called "tally" in Ruby.
#
# > tally([1,5,1,1,5])
# Dict{Int64, Int64} with 2 entries:
#   5 => 2
#   1 => 3
function tally(xs :: Vector{T}) where T
  counts = Dict{T,Int64}()
  for x in xs
    counts[x] = get(counts, x, 0) + 1
  end
  counts
end

concat(xs)        = collect(Iterators.flatten(xs))
concat_map(f, xs) = concat(map(f, xs))
count_unique(xs)  = length(unique(xs))

const MINUTE = 60
const HOUR   = 60*MINUTE
const DAY    = 24*HOUR

seconds_to_hour_i(sec)           = sec ÷ HOUR
seconds_to_fourhour_i(sec)       = sec ÷ (HOUR * 4)
seconds_to_convective_day_i(sec) = (sec - 12*HOUR) ÷ DAY

time_to_hour_i(time)           = seconds_to_hour_i(Int64(Dates.datetime2unix(time)))
time_to_fourhour_i(time)       = seconds_to_fourhour_i(Int64(Dates.datetime2unix(time)))
time_to_convective_day_i(time) = seconds_to_convective_day_i(Int64(Dates.datetime2unix(time)))

end