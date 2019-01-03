module Grids

import Serialization
# import JLD # Tried using JLD but there was a problem on read-in, and the written file was 25x times larger than it needed to be.

struct Grid
  height  :: Int64 # Element count
  width   :: Int64 # Element count
  min_lat :: Float64
  max_lat :: Float64
  min_lon :: Float64
  max_lon :: Float64
  latlons :: Array{Tuple{Float64,Float64}, 1} # Ordering is row-major: W -> E, S -> N
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

  if s_to_n_row_to_test > 1
    w_to_e_col_to_test = w_to_e_col - horizontal_step_size
    if w_to_e_col_to_test > 1
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
  if w_to_e_col_to_test > 1
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
    if w_to_e_col_to_test > 1
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


end # module Grid