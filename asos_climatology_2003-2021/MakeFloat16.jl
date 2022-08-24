for path in readdir()
  endswith(path, ".csv") || continue
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

  write(replace(path, ".csv" => ".float16.bin"), Float16.(grid_vals))
end
