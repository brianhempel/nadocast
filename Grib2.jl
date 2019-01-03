module Grib2

import DelimitedFiles # For readdlm

push!(LOAD_PATH, ".")
import Plots
import Grids

# Read grib2 grid.
#
# Ordering is row-major: W -> E, S -> N
#
# Returns a Grid.Grid structure (see Grid.jl) which contains the grid size and the lat-lon coordinates of all grid points.
function read_grid(grib2_path) :: Grids.Grid
  # Read the grid from the grib file. Returns four columns: row_index, col_index, lat, lon
  #
  # -end means stop after first (sub)message.
  # -inv /dev/null redirects the inventory listing. Otherwise it goes to stdout and there's otherwise no way to turn it off.
  all_pts = open(grid_csv -> DelimitedFiles.readdlm(grid_csv, ',', Float64; header=false), `wgrib2 $grib2_path -end -inv /dev/null -gridout -`)
  all_pts[:, 4] = [lon > 180 ? lon - 360 : lon for lon in all_pts[:, 4]]

  height = Int64(maximum(all_pts[:,2]))
  width  = Int64(maximum(all_pts[:,1]))

  min_lat = minimum(all_pts[:,3])
  max_lat = maximum(all_pts[:,3])
  min_lon = minimum(all_pts[:,4])
  max_lon = maximum(all_pts[:,4])

  latlons = map(flat_i -> (all_pts[flat_i,3], all_pts[flat_i,4]), 1:(width*height)) :: Array{Tuple{Float64,Float64},1}

  Grids.Grid(height, width, min_lat, max_lat, min_lon, max_lon, latlons)
end

# Read the inventory.
#
# Returns 8 column String array (no headers). Columns 4 and 5 are the abbreviation and description.
#
# Selected headers: message.submessage, position, -,              abbrev, description,             -,             -,             inventory number
# Sample row:       "4",                "956328", "d=2018062900", "CAPE", "180-0 mb above ground", "7 hour fcst", "wt ens mean", "n=4"
function read_inventory(grib2_path)
  # -s indicates "simple inventory"
  # -n indicates to add the inventory number
  open(`wgrib2 $grib2_path -s -n`) do inv
    DelimitedFiles.readdlm(inv, ':', String; header=false, use_mmap=false, quotes=false)
  end
end

# Read out the given layers into a binary Float32 array.
#
# `inventory` is a filtered set of lines from read_inventory above.
#
# Returns grid_length by layer_count Float32 array of values.
function read_layers_data_raw(grib2_path, inventory)
  # -i says to read the inventory from stdin
  # -headers says to print the layer size before and after
  # -inv /dev/null redirects the inventory listing. Otherwise it goes to stdout and there's otherwise no way to turn it off.
  wgrib2 = open(`wgrib2 $grib2_path -i -header -inv /dev/null -bin -`, "r+")

  layer_count = size(inventory,1)

  # Tell wgrib2 which layers we want.
  for layer_i = 1:layer_count
    layer_to_fetch = inventory[layer_i, :]
    # Only need first two columns (message.submessage and position) plus newline
    println(wgrib2, layer_to_fetch[1] * ":" * layer_to_fetch[2])
  end
  close(wgrib2.in)

  output_values_initialized = false

  layer_value_count = UInt32(0)
  values            = Array{Float32,2}[]

  # Read out the data in those layers.
  # Each layer is prefixed and postfixed by a 32bit integer indicating the byte size of the layer's data.
  for layer_i = 1:layer_count
    layer_to_fetch = inventory[layer_i, :]
    # abbrev, desc      = normalize_abbrev_and_desc(layer_to_fetch[4], layer_to_fetch[5])
    # layer_key         = abbrev * ":" * desc

    if !output_values_initialized
      layer_value_count         = div(read(wgrib2, UInt32), 4)
      values                    = zeros(Float32, (layer_value_count,layer_count))
      output_values_initialized = true
    else
      this_layer_value_count = div(read(wgrib2, UInt32), 4)
      if this_layer_value_count != layer_value_count
        error("value count mismatch, expected $layer_value_count for each layer but $layer_to_fetch has $this_layer_value_count values")
      end
    end

    values[:,layer_i] = reinterpret(Float32, read(wgrib2, layer_value_count*4))

    this_layer_value_count = div(read(wgrib2, UInt32), 4)
    if this_layer_value_count != layer_value_count
      error("value count mismatch, expected $layer_value_count for each layer but $layer_to_fetch has $this_layer_value_count values")
    end

    # if desc == "cloud base" || desc == "cloud top" || abbrev == "RETOP"
    #   # Handle undefined
    #   values = map((v -> (v > 25000.0f0 || v < -1000.0f0) ? 25000.0f0 : v), values)
    # end
    # println(grid_length)
    # println(values)
    # layer_to_data[layer_key] = values
    # println(grid_length_again)
  end

  # Sanity check that incoming stream is empty
  if !eof(wgrib2.out)
    error("wgrib2 sending more data than expected!")
  end

  # Set undefineds to 0 instead of 9.999f20
  map!(v -> v == 9.999f20 ? 0.0f0 : v, values, values)

  values
end

function latlon_to_value_no_interpolation(grid, layer_data, (lat, lon))
  flat_i = Grids.latlon_to_closest_grid_i(grid, (lat, lon))
  layer_data[flat_i]
end

function plot(grid :: Grids.Grid, layer_data :: Array{Float32,1})
  resolution_degrees = 0.1

  Plots.plot(Plots.heatmap(
    grid.min_lon:resolution_degrees:grid.max_lon,
    grid.min_lat:resolution_degrees:grid.max_lat,
    (lon, lat) -> latlon_to_value_no_interpolation(grid, layer_data, (lat, lon)),
    fill=true
  ))
end

end # module Grib2
