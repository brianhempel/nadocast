module Grib2

import DelimitedFiles # For readdlm
import JSON # For reading layer_name_normalization_substitutions.json
import Plots
import Dates
import Printf

push!(LOAD_PATH, @__DIR__)

import Cache
import GeoUtils
import Grids
import Inventories

# Deeply traverse root_path to find grib2 file paths
function all_grib2_file_paths_in(root_path)
  grib2_paths = []
  if !isdir(root_path)
    println("")
    println("$root_path does not exist! Can't read Grib2s!!")
    println("")
    return grib2_paths
  end
  for (dir_path, _, file_names) in walkdir(root_path)
    for file_name in file_names
      if endswith(file_name, ".grb2") || endswith(file_name, ".grib2")
        push!(grib2_paths, joinpath(dir_path, file_name)) # path to files
      end
    end
  end
  grib2_paths
end


### Grib2 Functions

# Read grib2 grid. Optional integer downsample parameter does what you would expect.
#
# If provided, crop is applied before downsample. Format: (W:E index range, S:N index range)
#
# Ordering is row-major: W -> E, S -> N
#
# Returns a Grid.Grid structure (see Grid.jl) which contains the grid size and the lat-lon coordinates of all grid points.
function read_grid(grib2_path; crop = nothing, downsample = 1) :: Grids.Grid
  # print("Reading grid $grib2_path...")

  out = Cache.cached([grib2_path], "read_grid_downsample_$(downsample)x") do

    # Read the grid from the grib file. Returns four columns: row_index, col_index, lat, lon
    #
    # -end means stop after first (sub)message.
    # -inv /dev/null redirects the inventory listing. Otherwise it goes to stdout and there's otherwise no way to turn it off.
    all_pts = open(grid_csv -> DelimitedFiles.readdlm(grid_csv, ',', Float64; header=false), `wgrib2 $grib2_path -end -inv /dev/null -gridout -`)
    all_pts[:, 4] = [lon > 180 ? lon - 360 : lon for lon in all_pts[:, 4]]

    raw_height = Int64(maximum(all_pts[:,2]))
    raw_width  = Int64(maximum(all_pts[:,1]))

    if isnothing(crop)
      crop_mask   = (:)
      crop_pts    = all_pts
      crop_height = raw_height
      crop_width  = raw_width
    else
      crop_x_range, crop_y_range = crop

      crop_mask   = map(x_i -> x_i in crop_x_range, all_pts[:,1]) .& map(y_i -> y_i in crop_y_range, all_pts[:,2])
      crop_pts    = all_pts[crop_mask, :]
      crop_height = Int64(maximum(crop_pts[:,2]) - minimum(crop_pts[:,2]) + 1)
      crop_width  = Int64(maximum(crop_pts[:,1]) - minimum(crop_pts[:,1]) + 1)
    end

    downsample_y_range = div(downsample+1,2):downsample:crop_height # 1:1:crop_height or 1:2:crop_height or 2:3:crop_height or 2:4:crop_height
    downsample_x_range = div(downsample+1,2):downsample:crop_width  # 1:1:crop_width  or 1:2:crop_width  or 2:3:crop_width  or 2:4:crop_width

    height = length(downsample_y_range)
    width  = length(downsample_x_range)

    downsampled_pts = zeros(Float64, (height*width, 2))

    downsampled_flat_i = 1
    for j in downsample_y_range
      for i in downsample_x_range
        crop_flat_i = crop_width*(j-1) + i

        if isodd(downsample)
          downsampled_pts[downsampled_flat_i, 1:2] = crop_pts[crop_flat_i, 3:4]
        else
          right_flat_i    = crop_flat_i
          left_flat_i     = crop_width*(j-1) + min(i+1, crop_width)
          up_right_flat_i = crop_width*(min(j+1,crop_height)-1) + i
          up_left_flat_i  = crop_width*(min(j+1,crop_height)-1) + min(i+1, crop_width)

          # Dumb flat mean, no sphere math. Should be close enough--always small distances
          crop_flat_is = [right_flat_i, left_flat_i, up_right_flat_i, up_left_flat_i]
          lat = sum(crop_pts[crop_flat_is, 3]) / 4
          lon = sum(crop_pts[crop_flat_is, 4]) / 4

          downsampled_pts[downsampled_flat_i, 1:2] = [lat, lon]
        end

        downsampled_flat_i += 1
      end
    end

    min_lat = minimum(downsampled_pts[:,1])
    max_lat = maximum(downsampled_pts[:,1])
    min_lon = minimum(downsampled_pts[:,2])
    max_lon = maximum(downsampled_pts[:,2])

    latlons = map(flat_i -> (downsampled_pts[flat_i,1], downsampled_pts[flat_i,2]), 1:(height*width)) :: Array{Tuple{Float64,Float64},1}

    # println("Estimating point areas...")

    # Estimate area represented by each grid point

    point_areas_sq_miles = zeros(Float64, height*width)
    point_heights_miles  = zeros(Float64, height*width)
    point_widths_miles   = zeros(Float64, height*width)

    for j = 1:height
      for i = 1:width
        flat_i = (j-1)*width + i
        lat, lon = latlons[flat_i]

        wlon = i > 1      ? latlons[flat_i-1][2]     : -1000.0
        elon = i < width  ? latlons[flat_i+1][2]     : -1000.0
        slat = j > 1      ? latlons[flat_i-width][1] : -1000.0
        nlat = j < height ? latlons[flat_i+width][1] : -1000.0

        w_distance = wlon > -1000.0 ? GeoUtils.distance((lat, lon), (lat, wlon)) / 2.0 / GeoUtils.METERS_PER_MILE : 0.0
        e_distance = elon > -1000.0 ? GeoUtils.distance((lat, lon), (lat, elon)) / 2.0 / GeoUtils.METERS_PER_MILE : 0.0
        s_distance = slat > -1000.0 ? GeoUtils.distance((lat, lon), (slat, lon)) / 2.0 / GeoUtils.METERS_PER_MILE : 0.0
        n_distance = nlat > -1000.0 ? GeoUtils.distance((lat, lon), (nlat, lon)) / 2.0 / GeoUtils.METERS_PER_MILE : 0.0

        w_distance = w_distance == 0.0 ? e_distance : w_distance
        e_distance = e_distance == 0.0 ? w_distance : e_distance
        s_distance = s_distance == 0.0 ? n_distance : s_distance
        n_distance = n_distance == 0.0 ? s_distance : n_distance

        sw_area = w_distance * s_distance
        se_area = e_distance * s_distance
        nw_area = w_distance * n_distance
        ne_area = e_distance * n_distance

        point_areas_sq_miles[flat_i] = sw_area + se_area + nw_area + ne_area
        point_heights_miles[flat_i]  = s_distance + n_distance
        point_widths_miles[flat_i]   = w_distance + e_distance
      end
    end

    point_weights = point_areas_sq_miles / maximum(point_areas_sq_miles)

    Grids.Grid(height, width, crop, crop_mask, crop_height, crop_width, downsample, raw_height, raw_width, min_lat, max_lat, min_lon, max_lon, latlons, point_areas_sq_miles, point_weights, point_heights_miles, point_widths_miles)
  end

  # println("done.")
  out
end

layer_name_normalization_substitutions = JSON.parse(open((@__DIR__) * "/layer_name_normalization_substitutions.json")) :: Dict{String,Any}


# Read the inventory. Field names are immediately normalized using the substitutions in layer_name_normalization_substitutions.json.
#
# Selected headers: message.submessage, position, -,              abbrev, level,                   forecast_time, misc,          inventory number
# Sample row:       "4",                "956328", "d=2018062900", "CAPE", "180-0 mb above ground", "7 hour fcst", "wt ens mean", "n=4"
#
# Probability files have an extra second-to-last column that just says "probability forecast" or "Neighborhood Probability"
function read_inventory(grib2_path) :: Vector{Inventories.InventoryLine}
  # print("Reading inventory $grib2_path...")

  out = Cache.cached([grib2_path], "read_inventory") do

    # -s indicates "simple inventory"
    # -n indicates to add the inventory number
    table =
      open(`wgrib2 $grib2_path -s -n`) do inv
        # c.f. find_common_layers.rb
        normalized_raw_inventory = read(inv, String)
        for (old, replacement) in layer_name_normalization_substitutions
          normalized_raw_inventory = replace(normalized_raw_inventory, old => replacement)
        end

        DelimitedFiles.readdlm(IOBuffer(normalized_raw_inventory), ':', String; header=false, use_mmap=false, quotes=false)
      end

    mapslices(row -> Inventories.InventoryLine(row[1], row[2], row[3], row[4], row[5], row[6], row[7], ""), table, dims = [2])[:,1]
  end

  # println("done.")
  out
end

# Set undefineds to 0 instead of 9.999f20
function zero_undefs!(arr)
  for i in eachindex(arr)
    v = arr[i]
    arr[i] = v == 9.999f20 ? 0.0f0 : v
  end
  ()
end

function apply_mask!(in, out, mask :: BitArray{1})
  j = 1
  for i in eachindex(in)
    if mask[i]
      out[j] = in[i]
      j += 1
    end
  end
  ()
end

function apply_mask!(in, out, mask :: Colon)
  out[:] = in
  ()
end

# Read out the given layers into a binary Float32 array.
#
# `inventory` is a filtered set of lines from read_inventory above.
# If cropping or downsampling, crop_downsample_grid must be provided.
#
# Julia reading happens to be really slow for Pipes b/c everything is read byte-by-byte, thereby making a bajillion syscalls.
# It's faster to have wgrib2 dump to a file and then read in the file, so that's what the code below does.
#
# Returns grid_length by layer_count Float32 array of values.
#
# 1.6s of 15s loading time, and 2.5M allocations
function read_layers_data_raw(grib2_path, inventory; crop_downsample_grid = nothing) :: Array{Float32, 2}
  # print("Reading data $grib2_path...")
  if isnothing(crop_downsample_grid)
    raw_height  = -1
    raw_width   = -1
    crop_mask   = (:)
    crop_height = -1
    crop_width  = -1
    crop_downsampled_value_count = UInt32(0)
    downsample  = 1
  else
    raw_height = crop_downsample_grid.original_height
    raw_width  = crop_downsample_grid.original_width

    crop_mask   = crop_downsample_grid.crop_mask
    crop_height = crop_downsample_grid.crop_height # Before downsampling
    crop_width  = crop_downsample_grid.crop_width  # Before downsampling

    crop_downsampled_value_count = UInt32(crop_downsample_grid.width * crop_downsample_grid.height)
    downsample = crop_downsample_grid.downsample
  end

  # out = Cache.cached([grib2_path], "read_layers_data_raw_downsample_$(downsample)x", inventory) do

  # -i says to read the inventory from stdin
  # -headers says to print the layer size before and after
  # -inv /dev/null redirects the inventory listing. Otherwise it goes to stdout and there's otherwise no way to turn it off.

  if Sys.KERNEL == :Darwin && !isdir("/Volumes/RAMDisk")
    println("Creating RAM disk for grib2 loading")
    disk_path = strip(read(`hdiutil attach -nomount ram://8388608`, String)) # 4GB ram disk
    run(`diskutil erasevolume HFS+ RAMDisk $disk_path`)
  end

  # print("opening wgrib2...")
  temp_path = tempname()
  if isdir("/dev/shm") # Use Ubuntu RAM disk if available
    temp_path = replace(temp_path, r".*/" => "/dev/shm/")
  elseif isdir("/Volumes/RAMDisk")
    temp_path = replace(temp_path, r".*/" => "/Volumes/RAMDisk/")
  end

  layer_count = length(inventory)

  # Establish the number of expected values per layer and the output array.

  # "1:0:npts=1905141\n"
  expected_layer_raw_value_count = parse(Int64, split(read(`wgrib2 $grib2_path -npts -end`, String), "=")[2])

  if isnothing(crop_downsample_grid)
    crop_downsampled_value_count = expected_layer_raw_value_count
  elseif expected_layer_raw_value_count != raw_width*raw_height
    error("$grib2_path has $expected_layer_raw_value_count values per layer but the crop_downsample_grid expects $(raw_width*raw_height) original values per layer.")
  end

  # out = zeros(Float32, )
  out = Array{Float32}(undef, (crop_downsampled_value_count,layer_count))

  # As of Julia 1.3, I/O is thread-safe.
  # And the preloader should have put the file into the disk cache.
  # So we can safely random access the file without killing the disk with seeks.
  # So we can run multiple instances of wgrib2 to read the different layers in parallel!

  thread_range(thread_id, count) = begin
    start = div((thread_id-1) * count, Threads.nthreads()) + 1
    stop  = div( thread_id    * count, Threads.nthreads())
    start:stop
  end

  thread_wgrib2_inv_path(thread_id) = temp_path * "_inventory_to_get_$(thread_id)"
  thread_wgrib2_out_path(thread_id) = temp_path * "_thread_$(thread_id)"

  wgrib2s = map(1:Threads.nthreads()) do thread_i
    layer_is = thread_range(thread_i, layer_count)

    out_path = thread_wgrib2_out_path(thread_i)

    # Okay it turns out wgrib2 is already multithreaded in this case.
    # But outputing separate files will allow us to downsample in threads.

    wgrib2 = open(`wgrib2 $grib2_path -i -header -inv /dev/null -bin $out_path -ncpu 1`, "r+")

    # print("asking for layers...")
    # Tell wgrib2 which layers we want.

    for layer_i in layer_is
      layer_to_fetch = inventory[layer_i]
      # Only need first two columns (message.submessage and position) plus newline
      println(wgrib2, layer_to_fetch.message_dot_submessage * ":" * layer_to_fetch.position_str)
    end

    close(wgrib2.in)

    wgrib2
  end

  # I don't think wait is thread-safe. It's lock-tastic. Hence the above isn't threaded (though it doesn't really need to be).
  # print("waiting...")
  map(wait, wgrib2s)

  count_after_crop_before_downsample = isa(crop_mask, Colon) ? expected_layer_raw_value_count : count(crop_mask)

  Threads.@threads for layer_is in map(thread_i -> thread_range(thread_i, layer_count), 1:Threads.nthreads())

    out_path = thread_wgrib2_out_path(Threads.threadid())

    read_scratch = Array{Float32}(undef, expected_layer_raw_value_count)
    crop_scratch = Array{Float32}(undef, count_after_crop_before_downsample)

    # print("reading layers")
    wgrib2_out = open(out_path)

    for layer_i in layer_is

      layer_to_fetch = inventory[layer_i]

      # Each layer is prefixed and postfixed by a 32bit integer indicating the byte size of the layer's data.

      this_layer_value_count = div(read(wgrib2_out, UInt32), 4)
      if this_layer_value_count != expected_layer_raw_value_count
        error("value count mismatch, expected $expected_layer_raw_value_count for each layer but $layer_to_fetch has $this_layer_value_count values")
      end

      read!(wgrib2_out, read_scratch)

      if downsample == 1
        apply_mask!(read_scratch, (@view out[:,layer_i]), crop_mask)
        zero_undefs!(@view out[:,layer_i])
      else
        # print("downsampling...")
        apply_mask!(read_scratch, crop_scratch, crop_mask)
        zero_undefs!(crop_scratch)
        layer_values_2d = reshape(crop_scratch, (crop_width, crop_height))

        _do_downsample!(downsample, crop_height, crop_width, layer_values_2d, layer_i, out)
      end

      this_layer_value_count = div(read(wgrib2_out, UInt32), 4)
      if this_layer_value_count != expected_layer_raw_value_count
        error("value count mismatch, expected $expected_layer_raw_value_count for each layer but $layer_to_fetch has $this_layer_value_count values")
      end

      # if desc == "cloud base" || desc == "cloud top" || abbrev == "RETOP"
      #   # Handle undefined
      #   out = map((v -> (v > 25000.0f0 || v < -1000.0f0) ? 25000.0f0 : v), out)
      # end

      # print(".")
    end

    # Sanity check that incoming stream is empty
    if !eof(wgrib2_out)
      error("wgrib2 sending more data than expected!")
    end
    close(wgrib2_out)
  end

  # run(`rm $out_path`) in threads was trouble.
  for thread_i in 1:Threads.nthreads()
    # print("removing temp file...")
    rm(thread_wgrib2_out_path(thread_i), force=true)
  end

  # println("done.")
  out
end

# Mutates out. Downsamples the given layer and places the result in the approprate location in out.
function _do_downsample!(downsample, crop_height, crop_width, layer_values_2d, layer_i, out)
  downsample_y_range = div(downsample+1,2):downsample:crop_height # 1:1:crop_height or 1:2:crop_height or 2:3:crop_height or 2:4:crop_height
  downsample_x_range = div(downsample+1,2):downsample:crop_width  # 1:1:crop_width  or 1:2:crop_width  or 2:3:crop_width  or 2:4:crop_width

  downleft_delta_i = div(downsample+1,2) - 1
  upright_delta_i  = -downleft_delta_i + downsample - 1

  downsampled_flat_i = 1
  @inbounds for j in downsample_y_range
    for i in downsample_x_range
      downsampled_value       = 0.0f0
      downsampled_value_count = 0.0f0

      for j2 in (j - downleft_delta_i):min(j + upright_delta_i, crop_height)
        for i2 in (i - downleft_delta_i):min(i + upright_delta_i, crop_width)
          downsampled_value       += layer_values_2d[i2, j2]
          downsampled_value_count += 1.0f0
        end
      end

      out[downsampled_flat_i, layer_i] = downsampled_value / downsampled_value_count

      downsampled_flat_i += 1
    end
  end

  ()
end


function write_15km_HREF_probs_grib2(probs :: Vector; run_time :: Dates.DateTime, forecast_hour :: Union{Int64,Tuple{Int64,Int64}}, event_type :: String, out_name :: String)
  var_name = Dict(
    "tornado"     => "TORPROB",
    "wind"        => "WINDPROB",
    "hail"        => "HAILPROB",
    "sig_tornado" => "STORPROB",
    "sig_wind"    => "SWINDPROB",
    "sig_hail"    => "SHAILPROB",
  )[event_type]

  date_str = Printf.@sprintf "%04d%02d%02d%02d" Dates.year(run_time) Dates.month(run_time) Dates.day(run_time) Dates.hour(run_time)
  fcst_str =
    if isa(forecast_hour, Int64)
      "$forecast_hour hour fcst"
    else
      near_hr, far_hr = forecast_hour
      "$near_hr-$far_hr hour ave fcst"
    end

  tmp_path = tempname()
  write(tmp_path, Float32.(probs .* 100f0))

  grib2_template_path = (@__DIR__) * "/href_one_field_for_grid_cropped_3x_downsampled.grib2"

  run(`wgrib2 $grib2_template_path -no_header -import_bin $tmp_path -set_var $var_name -set_date $date_str -set_ftime $fcst_str -set_prob 1 1 3 0 100 -set_grib_type jpeg -grib_out $out_name`)

  rm(tmp_path)

  ()
end


### Utility

function latlon_to_value_no_interpolation(grid, layer_data, (lat, lon))
  flat_i = Grids.latlon_to_closest_grid_i(grid, (lat, lon))
  layer_data[flat_i]
end

function plot(grid :: Grids.Grid, layer_data :: Array{<:Number,1})
  resolution_degrees = 0.1

  Plots.plot(Plots.heatmap(
    grid.min_lon:resolution_degrees:grid.max_lon,
    grid.min_lat:resolution_degrees:grid.max_lat,
    (lon, lat) -> latlon_to_value_no_interpolation(grid, layer_data, (lat, lon)),
    fill=true
  ))
end



end # module Grib2
