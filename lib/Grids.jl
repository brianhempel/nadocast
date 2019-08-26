module Grids

import Serialization
# import JLD # Tried using JLD but there was a problem on read-in, and the written file was 25x times larger than it needed to be.

push!(LOAD_PATH, @__DIR__)
import GeoUtils

struct Grid
  height               :: Int64 # Element count, after crop and downsample
  width                :: Int64 # Element count, after crop and downsample
  crop                 :: Union{Nothing, Tuple{UnitRange{Int64}, UnitRange{Int64}}} # (West:East crop range, South:North crop range) on original indicies
  crop_mask            :: Union{Colon, BitArray{1}}
  crop_height          :: Int64 # Element count before downsample
  crop_width           :: Int64 # Element count before downsample
  downsample           :: Int64 # 2x 3x 4x etc. (1x is no downsampling)
  original_height      :: Int64 # Element count before crop and downsample
  original_width       :: Int64 # Element count before crop and downsample
  min_lat              :: Float64
  max_lat              :: Float64
  min_lon              :: Float64
  max_lon              :: Float64
  latlons              :: Array{Tuple{Float64,Float64}, 1} # Ordering is row-major: W -> E, S -> N
  point_areas_sq_miles :: Array{Float64, 1}                # Ordering is row-major: W -> E, S -> N
  point_weights        :: Array{Float64, 1}                # Ordering is row-major: W -> E, S -> N
  point_heights_miles  :: Array{Float64, 1}                # Ordering is row-major: W -> E, S -> N
  point_widths_miles   :: Array{Float64, 1}                # Ordering is row-major: W -> E, S -> N
end

function to_file(path :: String, grid :: Grid)
  # JLD.save(path, "grid", grid)
  open(path, "w") do file
    Serialization.serialize(file, grid)
  end
end

function from_file(path :: String) :: Grids.Grid
  # JLD.load(path, "grid")
  open(path, "r") do file
    Serialization.deserialize(file) :: Grids.Grid
  end
end

function latlons_to_csv(path :: String, grid :: Grid)
  latlons_to_csv(path, grid.latlons)
end

function latlons_to_csv(path :: String, latlons :: Array{Tuple{Float64,Float64}, 1})
  open(path, "w") do file
    for (lat, lon) in latlons
      println(file, "$lat,$lon")
    end
  end
end

function get_grid_i(grid :: Grid, (s_to_n_row, w_to_e_col) :: Tuple{Int64, Int64}) :: Int64
  if w_to_e_col < 1
    error("Error indexing into grid, asked for column $w_to_e_col")
  elseif w_to_e_col > grid.width
    error("Error indexing into grid, asked for column $w_to_e_col")
  elseif s_to_n_row < 1
    error("Error indexing into grid, asked for row $s_to_n_row")
  elseif s_to_n_row > grid.height
    error("Error indexing into grid, asked for row $s_to_n_row")
  end
  grid.width*(s_to_n_row-1) + w_to_e_col
end


# Returns the index into the grid to return the point closest to the given lat-lon coordinate.
#
# "Closest" is simple 2D Euclidean distance on the lat-lon plane.
# It's "wrong" but since neighboring grid points are always close, it's not very wrong.
# And happily our grids don't cross -180/180.
#
# Binaryish search
function latlon_to_closest_grid_i(grid :: Grid, (lat, lon) :: Tuple{Float64, Float64}) :: Int64
  s_to_n_row = div(grid.height, 2)
  w_to_e_col = div(grid.width, 2)

  vertical_step_size   = div(grid.height, 2) - 1
  horizontal_step_size = div(grid.width, 2)  - 1

  latlon_to_closest_grid_i_search(
    grid,
    (lat, lon),
    (s_to_n_row, w_to_e_col),
    (vertical_step_size, horizontal_step_size)
  )
end

function latlon_to_closest_grid_i_search(grid :: Grid, (target_lat, target_lon) :: Tuple{Float64, Float64}, (s_to_n_row, w_to_e_col) :: Tuple{Int64, Int64}, (vertical_step_size, horizontal_step_size) :: Tuple{Int64, Int64}) :: Int64

  best_distance_squared = 10000000.0^2 # Best distance in "degrees"
  center_is_best = false
  best_s_to_n_row, best_w_to_e_col = (1, 1)

  # down row
  s_to_n_row_to_test = s_to_n_row - vertical_step_size

  if s_to_n_row_to_test >= 1
    w_to_e_col_to_test = w_to_e_col - horizontal_step_size
    if w_to_e_col_to_test >= 1
      flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
      (lat, lon) = grid.latlons[flat_i]
      distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
      if distance_squared < best_distance_squared
        best_distance_squared = distance_squared
        best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
        center_is_best = false
      end
    end

    w_to_e_col_to_test = w_to_e_col
    flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
    (lat, lon) = grid.latlons[flat_i]
    distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
    if distance_squared < best_distance_squared
      best_distance_squared = distance_squared
      best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
      center_is_best = false
    end

    w_to_e_col_to_test = w_to_e_col + horizontal_step_size
    if w_to_e_col_to_test <= grid.width
      flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
      (lat, lon) = grid.latlons[flat_i]
      distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
      if distance_squared < best_distance_squared
        best_distance_squared = distance_squared
        best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
        center_is_best = false
      end
    end
  end

  # center row
  s_to_n_row_to_test = s_to_n_row

  w_to_e_col_to_test = w_to_e_col - horizontal_step_size
  if w_to_e_col_to_test >= 1
    flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
    (lat, lon) = grid.latlons[flat_i]
    distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
    if distance_squared < best_distance_squared
      best_distance_squared = distance_squared
      best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
      center_is_best = false
    end
  end

  w_to_e_col_to_test = w_to_e_col
  flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
  (lat, lon) = grid.latlons[flat_i]
  distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
  if distance_squared < best_distance_squared
    best_distance_squared = distance_squared
    best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
    center_is_best = true
  end

  w_to_e_col_to_test = w_to_e_col + horizontal_step_size
  if w_to_e_col_to_test <= grid.width
    flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
    (lat, lon) = grid.latlons[flat_i]
    distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
    if distance_squared < best_distance_squared
      best_distance_squared = distance_squared
      best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
      center_is_best = false
    end
  end

  # up row
  s_to_n_row_to_test = s_to_n_row + vertical_step_size

  if s_to_n_row_to_test <= grid.height
    w_to_e_col_to_test = w_to_e_col - horizontal_step_size
    if w_to_e_col_to_test >= 1
      flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
      (lat, lon) = grid.latlons[flat_i]
      distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
      if distance_squared < best_distance_squared
        best_distance_squared = distance_squared
        best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
        center_is_best = false
      end
    end

    w_to_e_col_to_test = w_to_e_col
    flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
    (lat, lon) = grid.latlons[flat_i]
    distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
    if distance_squared < best_distance_squared
      best_distance_squared = distance_squared
      best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
      center_is_best = false
    end

    w_to_e_col_to_test = w_to_e_col + horizontal_step_size
    if w_to_e_col_to_test <= grid.width
      flat_i = get_grid_i(grid, (s_to_n_row_to_test, w_to_e_col_to_test))
      (lat, lon) = grid.latlons[flat_i]
      distance_squared = (lat-target_lat)^2 + (lon-target_lon)^2
      if distance_squared < best_distance_squared
        best_distance_squared = distance_squared
        best_s_to_n_row, best_w_to_e_col = (s_to_n_row_to_test, w_to_e_col_to_test)
        center_is_best = false
      end
    end
  end

  if center_is_best && vertical_step_size == 1 && horizontal_step_size == 1
    get_grid_i(grid, (best_s_to_n_row, best_w_to_e_col))
  else
    new_vertical_step_size   = max(1, div(vertical_step_size, 2))
    new_horizontal_step_size = max(1, div(horizontal_step_size, 2))
    latlon_to_closest_grid_i_search(
      grid,
      (target_lat, target_lon),
      (best_s_to_n_row, best_w_to_e_col),
      (new_vertical_step_size, new_horizontal_step_size)
    )
  end
end

# Searches gridpoints outward in a diamond shape from the given center until
# (a) it starts finding points matching the predicate, and after that until
# (b) it adds a layer to the diamond and finds no new points matching the predicate
function diamond_search(predicate, grid :: Grid, center_i :: Int64, center_j :: Int64) :: Vector{Int64}
  any_found_this_diamond  = false
  still_searching_on_grid = true
  matching_flat_is = []

  height  = grid.height
  width   = grid.width
  latlons = grid.latlons

  is_on_grid(w_to_e_col, s_to_n_row) = begin
    w_to_e_col >= 1 && w_to_e_col <= width &&
    s_to_n_row >= 1 && s_to_n_row <= height
  end

  get_grid_latlon_and_flat_i(w_to_e_col, s_to_n_row) = begin
    flat_i = grid.width*(s_to_n_row-1) + w_to_e_col
    (latlons[flat_i], flat_i)
  end

  r = 0 :: Int64
  while still_searching_on_grid && (isempty(matching_flat_is) || any_found_this_diamond)
    any_found_this_diamond  = false
    still_searching_on_grid = false

    # search in this order:
    #
    #     3
    #   3   2
    # 4   •   2
    #   4   1
    #     1

    if r == 0
      i, j = center_i, center_j
      if is_on_grid(i, j)
        still_searching_on_grid = true
        latlon, flat_i = get_grid_latlon_and_flat_i(i, j)
        if predicate(latlon)
          any_found_this_diamond = true
          push!(matching_flat_is, flat_i)
        end
      end
    else
      for k = 0:(r-1)
        i, j = center_i+k, center_j-r+k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          latlon, flat_i = get_grid_latlon_and_flat_i(i, j)
          if predicate(latlon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i+r-k, center_j+k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          latlon, flat_i = get_grid_latlon_and_flat_i(i, j)
          if predicate(latlon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i-k, center_j+r-k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          latlon, flat_i = get_grid_latlon_and_flat_i(i, j)
          if predicate(latlon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i-r+k, center_j-k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          latlon, flat_i = get_grid_latlon_and_flat_i(i, j)
          if predicate(latlon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

      end
    end

    r += 1
  end

  sort(matching_flat_is)
end


# For each grid point, return the indices of all grid points within so many miles.
function radius_grid_is(grid, miles) :: Vector{Vector{Int64}}
  radius_is = map(_ -> Int64[], 1:grid.height*grid.width)

  for j in 1:grid.height
    for i in 1:grid.width
      flat_i = grid.width*(j-1) + i

      latlon = grid.latlons[flat_i]

      radius_is[flat_i] =
        diamond_search(grid, i, j) do candidate_latlon
          GeoUtils.instantish_distance(candidate_latlon, latlon) <= miles * GeoUtils.METERS_PER_MILE
        end
    end
  end

  # Re-allocate to ensure cache locality.
  for flat_i in 1:length(radius_is)
    radius_is[flat_i] = radius_is[flat_i][1:length(radius_is[flat_i])]
  end

  radius_is
end

# For making 100mi mean is with a 50mi hole cut out of it.
function radius_grid_is_less_other_is(grid, miles, grid_is_to_subtract) :: Vector{Vector{Int64}}
  radius_is = Grids.radius_grid_is(grid, miles)

  for flat_i in 1:grid.height*grid.width
    is_to_subtract = grid_is_to_subtract[flat_i]

    radius_is[flat_i] =
      filter(radius_is[flat_i]) do i
        !(i in is_to_subtract)
      end
  end

  # Re-allocate to ensure cache locality.
  # Not sure this makes any difference though.
  for flat_i in 1:length(radius_is)
    radius_is[flat_i] = radius_is[flat_i][1:length(radius_is[flat_i])]
  end

  radius_is
end

# Returns a function that takes a single layer and upsamples it to the higher resolution grid.
#
# Nearest neighbor. It's blocky.
function get_upsampler(low_res_grid, high_res_grid)
  low_res_grid_is_on_high_res_grid = map(high_res_grid.latlons) do latlon
    Grids.latlon_to_closest_grid_i(low_res_grid, latlon)
  end

  upsampler(low_res_layer) = begin
    # print("Upsampling...")
    out = low_res_layer[low_res_grid_is_on_high_res_grid]
    # println("done.")
    out
  end

  upsampler
end

function _latlon_euclidean_distance_squared((lat1, lon1), (lat2, lon2))
  (lat2-lat1)^2 + (lon2-lon1)^2
end

# Returns a function that takes a single layer and upsamples it to the higher resolution grid.
#
# May have artifacts at the edges.
#
# Not really bilinear or bicubic, but not blocky.
function get_interpolating_upsampler(low_res_grid, high_res_grid)
  low_res_grid_spacing_miles = max(maximum(low_res_grid.point_heights_miles), maximum(low_res_grid.point_widths_miles))
  nearby_low_res_grid_is     = radius_grid_is(low_res_grid, 3.0 * low_res_grid_spacing_miles)

  low_res_grid_is_and_weights_on_high_res_grid = map(high_res_grid.latlons) do latlon
    closest_low_res_i = Grids.latlon_to_closest_grid_i(low_res_grid, latlon)
    close_low_res_grid_is = nearby_low_res_grid_is[closest_low_res_i]
    close_low_res_grid_i_distance_squared = _latlon_euclidean_distance_squared(low_res_grid.latlons[closest_low_res_i], latlon)

    ne_low_res_is = filter(low_res_i -> low_res_grid.latlons[low_res_i][1] >= latlon[1] && low_res_grid.latlons[low_res_i][2] >= latlon[2], close_low_res_grid_is)
    nw_low_res_is = filter(low_res_i -> low_res_grid.latlons[low_res_i][1] >= latlon[1] && low_res_grid.latlons[low_res_i][2] <  latlon[2], close_low_res_grid_is)
    se_low_res_is = filter(low_res_i -> low_res_grid.latlons[low_res_i][1] <  latlon[1] && low_res_grid.latlons[low_res_i][2] >= latlon[2], close_low_res_grid_is)
    sw_low_res_is = filter(low_res_i -> low_res_grid.latlons[low_res_i][1] <  latlon[1] && low_res_grid.latlons[low_res_i][2] <  latlon[2], close_low_res_grid_is)

    # "distances"
    ne_distances_is = map(low_res_i -> (_latlon_euclidean_distance_squared(low_res_grid.latlons[low_res_i], latlon), low_res_i), ne_low_res_is)
    nw_distances_is = map(low_res_i -> (_latlon_euclidean_distance_squared(low_res_grid.latlons[low_res_i], latlon), low_res_i), nw_low_res_is)
    se_distances_is = map(low_res_i -> (_latlon_euclidean_distance_squared(low_res_grid.latlons[low_res_i], latlon), low_res_i), se_low_res_is)
    sw_distances_is = map(low_res_i -> (_latlon_euclidean_distance_squared(low_res_grid.latlons[low_res_i], latlon), low_res_i), sw_low_res_is)

    ne_distance_squared, ne_i =
      if isempty(ne_distances_is)
        (close_low_res_grid_i_distance_squared, closest_low_res_i)
      else
        minimum(ne_distances_is)
      end

    nw_distance_squared, nw_i =
      if isempty(nw_distances_is)
        (close_low_res_grid_i_distance_squared, closest_low_res_i)
      else
        minimum(nw_distances_is)
      end

    se_distance_squared, se_i =
      if isempty(se_distances_is)
        (close_low_res_grid_i_distance_squared, closest_low_res_i)
      else
        minimum(se_distances_is)
      end

    sw_distance_squared, sw_i =
      if isempty(sw_distances_is)
        (close_low_res_grid_i_distance_squared, closest_low_res_i)
      else
        minimum(sw_distances_is)
      end

    ne_inverse_distance = 1.0 / (0.00000001 + √ne_distance_squared)
    nw_inverse_distance = 1.0 / (0.00000001 + √nw_distance_squared)
    se_inverse_distance = 1.0 / (0.00000001 + √se_distance_squared)
    sw_inverse_distance = 1.0 / (0.00000001 + √sw_distance_squared)

    total_inverse_distance = ne_inverse_distance + nw_inverse_distance + se_inverse_distance + sw_inverse_distance

    ne_weight = Float32(ne_inverse_distance / total_inverse_distance)
    nw_weight = Float32(nw_inverse_distance / total_inverse_distance)
    se_weight = Float32(se_inverse_distance / total_inverse_distance)
    sw_weight = Float32(sw_inverse_distance / total_inverse_distance)

    ( [ne_i,      nw_i,      se_i,      sw_i]
    , [ne_weight, nw_weight, se_weight, sw_weight]
    )
  end

  upsampler(low_res_layer) = begin
    print("Interpolating upsampling...")
    out = map(low_res_grid_is_and_weights_on_high_res_grid) do low_res_is_and_weights
      low_res_is, low_res_weights = low_res_is_and_weights
      sum(low_res_layer[low_res_is] .* low_res_weights)
    end
    println("done.")
    out
  end

  upsampler
end


end # module Grids