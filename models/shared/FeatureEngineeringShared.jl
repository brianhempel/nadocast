module FeatureEngineeringShared

using Printf

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Cache
import Forecasts
import ForecastCombinators
import GeoUtils
import Grids
import Inventories

# Data loading (including reading and downsampling and feature engineering)
#
#  Performance counter stats for 'system wide':
#
#       9,601,096.66 msec cpu-clock                 #   31.998 CPUs utilized
#          3,283,448      context-switches          #    0.342 K/sec
#             66,050      cpu-migrations            #    0.007 K/sec
#        121,015,667      page-faults               #    0.013 M/sec
# 13,613,969,087,148      cycles                    #    1.418 GHz                      (39.99%)
# 13,648,607,135,453      stalled-cycles-frontend   #  100.25% frontend cycles idle     (49.99%)
# 10,514,390,725,051      stalled-cycles-backend    #   77.23% backend cycles idle      (50.00%)
# 11,870,389,799,395      instructions              #    0.87  insn per cycle
#                                                   #    1.15  stalled cycles per insn  (60.00%)
#  1,624,047,459,493      branches                  #  169.152 M/sec                    (60.00%)
#      7,294,827,058      branch-misses             #    0.45% of all branches          (60.00%)
#  3,593,999,313,044      L1-dcache-loads           #  374.332 M/sec                    (31.93%)
#    448,398,167,181      L1-dcache-load-misses     #   12.48% of all L1-dcache hits    (32.89%)
#     78,116,932,543      LLC-loads                 #    8.136 M/sec                    (20.00%)
#     19,414,613,907      LLC-load-misses           #   24.85% of all LL-cache hits     (29.99%)
#
#      300.052759769 seconds time elapsed


feature_block_names = [
  "",
  "25mi mean",
  "50mi mean",
  "100mi mean",
  "25mi forward grad",
  "25mi leftward grad",
  "25mi linestraddling grad",
  "50mi forward grad",
  "50mi leftward grad",
  "50mi linestraddling grad",
  "100mi forward grad",
  "100mi leftward grad",
  "100mi linestraddling grad",
]

# Corresponding to the above.
# The gradient blocks are dependent on the mean blocks.
raw_features_block                           = 1
twenty_five_mi_mean_block                    = 2
fifty_mi_mean_block                          = 3
hundred_mi_mean_block                        = 4
twenty_five_mi_forward_gradient_block        = 5
twenty_five_mi_leftward_gradient_block       = 6
twenty_five_mi_linestraddling_gradient_block = 7
fifty_mi_forward_gradient_block              = 8
fifty_mi_leftward_gradient_block             = 9
fifty_mi_linestraddling_gradient_block       = 10
hundred_mi_forward_gradient_block            = 11
hundred_mi_leftward_gradient_block           = 12
hundred_mi_linestraddling_gradient_block     = 13
all_layer_blocks                             = collect(1:length(feature_block_names))

# Chosen by inspecting 2021v1 model feature usage.
fewer_grad_blocks = [
  raw_features_block,
  twenty_five_mi_mean_block,
  fifty_mi_mean_block,
  hundred_mi_mean_block,
  # twenty_five_mi_forward_gradient_block,
  # twenty_five_mi_leftward_gradient_block,
  # twenty_five_mi_linestraddling_gradient_block,
  fifty_mi_forward_gradient_block,
  fifty_mi_leftward_gradient_block,
  # fifty_mi_linestraddling_gradient_block,
  hundred_mi_forward_gradient_block,
  hundred_mi_leftward_gradient_block,
  hundred_mi_linestraddling_gradient_block,
]

# Returns twenty_five_mi_mean_is2, fifty_mi_mean_is2, hundred_mi_mean_is2
function compute_mean_is2(grid)
  print("computing radius ranges...")
  point_size_km           = 1.61 * sqrt(grid.point_areas_sq_miles[div(length(grid.point_areas_sq_miles), 2)])
  cache_folder            = @sprintf "grid_%.1fkm_%dx%d_downsample_%dx_%.2f_%.2f_%.2f-%.2f" point_size_km grid.width grid.height grid.downsample grid.min_lat grid.max_lat grid.min_lon grid.max_lon
  twenty_five_mi_mean_is2 = Cache.cached(() -> Grids.radius_grid_is2(grid, 25.0),  [cache_folder], "twenty_five_mi_mean_is2")
  fifty_mi_mean_is2       = Cache.cached(() -> Grids.radius_grid_is2(grid, 50.0),  [cache_folder], "fifty_mi_mean_is2")
  hundred_mi_mean_is2     = Cache.cached(() -> Grids.radius_grid_is2(grid, 100.0), [cache_folder], "hundred_mi_mean_is2")
  println("done")

  return (twenty_five_mi_mean_is2, fifty_mi_mean_is2, hundred_mi_mean_is2)
end

# new_features_pre should be a list of pairs of (feature_name, compute_feature_function(grid, inventory, data))
function feature_engineered_forecasts(base_forecasts; vector_wind_layers, layer_blocks_to_make, new_features_pre = [])

  if isempty(base_forecasts)
    return base_forecasts
  end

  layer_blocks_to_make = sort(layer_blocks_to_make) # Layers always produced in a particular order regardless of given order. It's a set.

  grid = base_forecasts[1].grid

  twenty_five_mi_mean_is2, fifty_mi_mean_is2, hundred_mi_mean_is2 = compute_mean_is2(grid)

  inventory_transformer(base_forecast, base_inventory) = begin

    new_inventory = Inventories.InventoryLine[]

    new_features_pre_lines =
      map(new_features_pre) do (feature_name, _compute_feature)
        Inventories.InventoryLine(
          "",                                   # message_dot_submessage
          "",                                   # position_str
          base_inventory[1].date_str,
          feature_name, # abbrev
          "calculated",                         # level
          "hour fcst",                          # forecast_hour_str
          "",                                   # misc
          ""                                    # feature_engineering
        )
      end

    base_inventory_with_pre_fields = vcat(base_inventory, new_features_pre_lines)

    for output_block_i in layer_blocks_to_make
      block_name = feature_block_names[output_block_i]
      for inventory_line in base_inventory_with_pre_fields
        push!(new_inventory, Inventories.revise_with_feature_engineering(inventory_line, block_name))
      end
    end

    new_inventory
  end

  data_transformer(base_forecast, base_data) = begin
    # println("Feature engineering $(base_forecast.model_name) $(Forecasts.time_title(base_forecast))...")

    out = make_data(
      grid,
      Forecasts.inventory(base_forecast), # 100k allocs each time
      base_data,
      vector_wind_layers,
      layer_blocks_to_make,
      twenty_five_mi_mean_is2,
      fifty_mi_mean_is2,
      hundred_mi_mean_is2;
      new_features_pre = new_features_pre
    )

    # println("done.")
    out
  end

  ForecastCombinators.map_forecasts(base_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end

function feature_range(block_i, pre_feature_count)
  (block_i-1)*pre_feature_count+1:block_i*pre_feature_count
end



function point_mean(point_mean_is2, len, row_cumsums)
  val = 0.0

  @inbounds for (row, range) in point_mean_is2
    val += row_cumsums[range.stop+1, row] - row_cumsums[range.start, row]
  end

  Float32(val / len)
end

# Mutates row_cumsums
function calc_row_cumsums!(grid, feature_data, row_cumsums)
  width = grid.width

  for j in 1:grid.height
    row_offset = width*(j-1)

    row_sum = 0.0
    row_cumsums[1,j] = row_sum

    @inbounds for i in 1:width
      flat_i = row_offset + i
      row_sum += Float64(feature_data[flat_i])
      row_cumsums[i+1,j] = row_sum
    end
  end

  ()
end

function make_3mean_layers!(grid, feature_data, mean_is2_1, mean_is2_2, mean_is2_3, mean_is2_lens_1, mean_is2_lens_2, mean_is2_lens_3, out1, out2, out3, row_cumsums)
  # out         = zeros(Float32, size(feature_data))
  # row_cumsums = Array{Float64}(undef, (grid.width+1, grid.height))

  calc_row_cumsums!(grid, feature_data, row_cumsums)

  @inbounds for grid_i in 1:length(grid.latlons)
    out1[grid_i] = point_mean(mean_is2_1[grid_i], mean_is2_lens_1[grid_i], row_cumsums)
    out2[grid_i] = point_mean(mean_is2_2[grid_i], mean_is2_lens_2[grid_i], row_cumsums)
    out3[grid_i] = point_mean(mean_is2_3[grid_i], mean_is2_lens_3[grid_i], row_cumsums)
  end

  ()
end


function make_mean_layers2!(
    out, pre_feature_count, grid,
    twenty_five_mi_mean_features_range, twenty_five_mi_mean_is2,
    fifty_mi_mean_features_range,       fifty_mi_mean_is2,
    hundred_mi_mean_features_range,     hundred_mi_mean_is2
  )

  calc_lens(grid_is2) = begin
    lens = Array{Int64}(undef, length(grid_is2))
    Threads.@threads for grid_i in 1:length(grid_is2)
      len = 0
      for (_, range) in grid_is2[grid_i]
        len += length(range)
      end
      lens[grid_i] = len
    end
    lens
  end

  twenty_five_mi_mean_is2_lens  = calc_lens(twenty_five_mi_mean_is2)
  fifty_mi_mean_is2_lens        = calc_lens(fifty_mi_mean_is2)
  hundred_mi_mean_is2_lens      = calc_lens(hundred_mi_mean_is2)

  thread_row_cumsums = map(_ -> Array{Float64}(undef, (grid.width+1, grid.height)), 1:Threads.nthreads())

  Threads.@threads for pre_layer_feature_i in 1:pre_feature_count

    twenty_five_mi_mean_feature_i = pre_layer_feature_i - 1 + twenty_five_mi_mean_features_range.start
    fifty_mi_mean_feature_i       = pre_layer_feature_i - 1 + fifty_mi_mean_features_range.start
    hundred_mi_mean_feature_i     = pre_layer_feature_i - 1 + hundred_mi_mean_features_range.start

    make_3mean_layers!(
      grid, (@view out[:, pre_layer_feature_i]),
      twenty_five_mi_mean_is2, fifty_mi_mean_is2, hundred_mi_mean_is2,
      twenty_five_mi_mean_is2_lens, fifty_mi_mean_is2_lens, hundred_mi_mean_is2_lens,
      (@view out[:, twenty_five_mi_mean_feature_i]),
      (@view out[:, fifty_mi_mean_feature_i]),
      (@view out[:, hundred_mi_mean_feature_i]),
      thread_row_cumsums[Threads.threadid()]
    )
  end

  ()
end

function uv_normalize(u, v)
  if u == 0f0 && v == 0f0
    (1f0, 0f0)
  else
    scale = max(abs(u), abs(v)) # Ensure small numbers don't get squared off to zero
    u /= scale
    v /= scale
    len = √(u^2 + v^2)
    (u / len, v / len)
  end
end

# Normalizes in place, but also returns the vectors
function normalize_uvs!(us, vs)
  @assert length(us) == length(vs)

  @inbounds Threads.@threads for flat_i in 1:length(us)
    normalized_u, normalized_v = @inbounds uv_normalize(us[flat_i], vs[flat_i])
    us[flat_i] = normalized_u
    vs[flat_i] = normalized_v
  end

  us, vs
end

# Mutates all the something_is arguments.
#
# Once we pull this out into a function then our allocations here go way down...to zero I believe.
function compute_directional_is!(
    height, width,
    point_heights_miles, point_widths_miles,
    # us, vs,
    normalized_us, normalized_vs,
    twenty_five_mi_forward_is, twenty_five_mi_backward_is, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is,
    fifty_mi_forward_is,       fifty_mi_backward_is,       fifty_mi_leftward_is,       fifty_mi_rightward_is,
    hundred_mi_forward_is,     hundred_mi_backward_is,     hundred_mi_leftward_is,     hundred_mi_rightward_is
  )
  Threads.@threads for j in 1:height
    @inbounds for i in 1:width
      flat_i = width*(j-1) + i

      # normalized_u, normalized_v = uv_normalize(us[flat_i], vs[flat_i])
      # normalized_u = normalized_us[flat_i]

      point_height = Float32(point_heights_miles[flat_i])
      point_width  = Float32(point_widths_miles[flat_i]) # On the SREF grid, point_width and point_height are always nearly equal, <1.0% difference.

      delta_25mi_j  = round(Int64, normalized_vs[flat_i] * 25.0f0  / point_height)
      delta_25mi_i  = round(Int64, normalized_us[flat_i] * 25.0f0  / point_width)
      delta_50mi_j  = round(Int64, normalized_vs[flat_i] * 50.0f0  / point_height)
      delta_50mi_i  = round(Int64, normalized_us[flat_i] * 50.0f0  / point_width)
      delta_100mi_j = round(Int64, normalized_vs[flat_i] * 100.0f0 / point_height)
      delta_100mi_i = round(Int64, normalized_us[flat_i] * 100.0f0 / point_width)

      forward_25mi_j    = clamp(j+delta_25mi_j,  1, height)
      forward_25mi_i    = clamp(i+delta_25mi_i,  1, width)
      forward_50mi_j    = clamp(j+delta_50mi_j,  1, height)
      forward_50mi_i    = clamp(i+delta_50mi_i,  1, width)
      forward_100mi_j   = clamp(j+delta_100mi_j, 1, height)
      forward_100mi_i   = clamp(i+delta_100mi_i, 1, width)

      backward_25mi_j   = clamp(j-delta_25mi_j,  1, height)
      backward_25mi_i   = clamp(i-delta_25mi_i,  1, width)
      backward_50mi_j   = clamp(j-delta_50mi_j,  1, height)
      backward_50mi_i   = clamp(i-delta_50mi_i,  1, width)
      backward_100mi_j  = clamp(j-delta_100mi_j, 1, height)
      backward_100mi_i  = clamp(i-delta_100mi_i, 1, width)

      leftward_25mi_j   = clamp(j+delta_25mi_i,  1, height)
      leftward_25mi_i   = clamp(i-delta_25mi_j,  1, width)
      leftward_50mi_j   = clamp(j+delta_50mi_i,  1, height)
      leftward_50mi_i   = clamp(i-delta_50mi_j,  1, width)
      leftward_100mi_j  = clamp(j+delta_100mi_i, 1, height)
      leftward_100mi_i  = clamp(i-delta_100mi_j, 1, width)

      rightward_25mi_j  = clamp(j-delta_25mi_i,  1, height)
      rightward_25mi_i  = clamp(i+delta_25mi_j,  1, width)
      rightward_50mi_j  = clamp(j-delta_50mi_i,  1, height)
      rightward_50mi_i  = clamp(i+delta_50mi_j,  1, width)
      rightward_100mi_j = clamp(j-delta_100mi_i, 1, height)
      rightward_100mi_i = clamp(i+delta_100mi_j, 1, width)

      twenty_five_mi_forward_is[flat_i]   = width*(forward_25mi_j-1)    + forward_25mi_i
      twenty_five_mi_backward_is[flat_i]  = width*(backward_25mi_j-1)   + backward_25mi_i
      twenty_five_mi_leftward_is[flat_i]  = width*(leftward_25mi_j-1)   + leftward_25mi_i
      twenty_five_mi_rightward_is[flat_i] = width*(rightward_25mi_j-1)  + rightward_25mi_i
      fifty_mi_forward_is[flat_i]         = width*(forward_50mi_j-1)    + forward_50mi_i
      fifty_mi_backward_is[flat_i]        = width*(backward_50mi_j-1)   + backward_50mi_i
      fifty_mi_leftward_is[flat_i]        = width*(leftward_50mi_j-1)   + leftward_50mi_i
      fifty_mi_rightward_is[flat_i]       = width*(rightward_50mi_j-1)  + rightward_50mi_i
      hundred_mi_forward_is[flat_i]       = width*(forward_100mi_j-1)   + forward_100mi_i
      hundred_mi_backward_is[flat_i]      = width*(backward_100mi_j-1)  + backward_100mi_i
      hundred_mi_leftward_is[flat_i]      = width*(leftward_100mi_j-1)  + leftward_100mi_i
      hundred_mi_rightward_is[flat_i]     = width*(rightward_100mi_j-1) + rightward_100mi_i
    end
  end

  ()
end

# To avoid broadcast allocs :/
# Mutates out
function compute_linestraddling_gradient!(out, out_feature_i, mean_feature_i, forward_is, backward_is, leftward_is, rightward_is)
  @inbounds for i in 1:size(out,1)
    out[i, out_feature_i] =
      out[forward_is[i],   mean_feature_i] +
      out[backward_is[i],  mean_feature_i] -
      out[leftward_is[i],  mean_feature_i] -
      out[rightward_is[i], mean_feature_i]
  end

  ()
end

# To avoid broadcast allocs :/
# Mutates out
function compute_gradient!(out, out_feature_i, mean_feature_i, forward_is, backward_is)
  @inbounds for i in 1:size(out,1)
    out[i, out_feature_i] =
      out[forward_is[i],  mean_feature_i] -
      out[backward_is[i], mean_feature_i]
  end

  ()
end

# Mutates out
function rotate_uv_layers!(out, u_i, v_i, rot_coses, rot_sines)
  @inbounds for i in 1:size(out,1)
    u = out[i, u_i]
    v = out[i, v_i]
    out[i, u_i] = u * rot_coses[i] - v * rot_sines[i]
    out[i, v_i] = u * rot_sines[i] + v * rot_coses[i]
  end

  ()
end

# Mutates out
# function rotate_and_perhaps_center_uv_layers!(out, u_i, v_i, mean_wind_lower_half_atmosphere_us, mean_wind_lower_half_atmosphere_vs, rot_coses, rot_sines, should_center)
#   if should_center
#     @inbounds for i in 1:size(out,1)
#       u = out[i, u_i] - mean_wind_lower_half_atmosphere_us[i]
#       v = out[i, v_i] - mean_wind_lower_half_atmosphere_vs[i]
#       out[i, u_i] = u * rot_coses[i] - v * rot_sines[i]
#       out[i, v_i] = u * rot_sines[i] + v * rot_coses[i]
#     end
#   else
#     @inbounds for i in 1:size(out,1)
#       u = out[i, u_i]
#       v = out[i, v_i]
#       out[i, u_i] = u * rot_coses[i] - v * rot_sines[i]
#       out[i, v_i] = u * rot_sines[i] + v * rot_coses[i]
#     end
#   end

#   ()
# end



function get_feature_i(inventory, feature_key)
  findfirst(inventory) do inventory_line
    Inventories.inventory_line_key(inventory_line) == feature_key
  end
end


function lapse_rate_from_ensemble_mean!(get_layer, out, lo_key, hi_key)
  lo_layer_tmp = get_layer("TMP:$(lo_key):hour fcst:wt ens mean")
  hi_layer_tmp = get_layer("TMP:$(hi_key):hour fcst:wt ens mean")
  lo_layer_hgt = get_layer("HGT:$(lo_key):hour fcst:wt ens mean")
  hi_layer_hgt = get_layer("HGT:$(hi_key):hour fcst:wt ens mean")
  out .= (lo_layer_tmp .- hi_layer_tmp) ./ ((hi_layer_hgt .- lo_layer_hgt .+ eps(1f0)) .* 0.001f0)
end

# Formulas from mixing ratio calculator, by Tim Brice and Todd Hall
# https://www.weather.gov/epz/wxcalc_mixingratio
function mixing_ratio(dpt_K, mb)
  dpt_C    = dpt_K - 273.15f0
  vap_pres = 6.11f0 * 10f0^(7.5f0*dpt_C / (237.7f0 + dpt_C))
  out = 621.97f0 * vap_pres/(mb - vap_pres)
  isnan(out) ? 0f0 : out
end

function compute_divergence_threaded!(grid, out, u_data, v_data)
  @assert length(grid.point_widths_miles) == length(out)
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)

  # out =  Array{Float32}(undef, length(grid.point_widths_miles))

  width = grid.width

  Threads.@threads for j in 2:(grid.height - 1)
    row_offset = width*(j-1)

    @inbounds for i in 2:(width - 1)
      flat_i = row_offset + i

      dx = (grid.point_widths_miles[flat_i]  + 0.5*grid.point_widths_miles[flat_i - 1]      + 0.5*grid.point_widths_miles[flat_i + 1])      * GeoUtils.METERS_PER_MILE
      dy = (grid.point_heights_miles[flat_i] + 0.5*grid.point_heights_miles[flat_i - width] + 0.5*grid.point_heights_miles[flat_i + width]) * GeoUtils.METERS_PER_MILE

      divergence =
        (u_data[flat_i + 1]     - u_data[flat_i - 1])     / Float32(dx) +
        (v_data[flat_i + width] - v_data[flat_i - width]) / Float32(dy)

      out[flat_i] = divergence * 100_000f0
    end

    # West/East edges, copy the neighbor.
    out[row_offset + 1]     = out[row_offset + 2]
    out[row_offset + width] = out[row_offset + width - 1]
  end

  # South/North edges, copy the neighbor.
  for i in 1:width
    out[i]                         = out[width + i]
    out[width*(grid.height-1) + i] = out[width*(grid.height-2) + i]
  end

  ()
end

function compute_abs_vorticity_threaded!(grid, out, u_data, v_data)
  @assert length(grid.point_widths_miles) == length(out)
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)

  # out =  Array{Float32}(undef, length(grid.point_widths_miles))

  width = grid.width

  Threads.@threads for j in 2:(grid.height - 1)
    row_offset = width*(j-1)

    @inbounds for i in 2:(width - 1)
      flat_i = row_offset + i

      dx = (grid.point_widths_miles[flat_i]  + 0.5*grid.point_widths_miles[flat_i - 1]      + 0.5*grid.point_widths_miles[flat_i + 1])      * GeoUtils.METERS_PER_MILE
      dy = (grid.point_heights_miles[flat_i] + 0.5*grid.point_heights_miles[flat_i - width] + 0.5*grid.point_heights_miles[flat_i + width]) * GeoUtils.METERS_PER_MILE

      # https://en.wikipedia.org/wiki/Vorticity#Mathematical_definition
      # dv_y/dx - dv_x/dy
      vorticity =
        (v_data[flat_i + 1]     - v_data[flat_i - 1])     / Float32(dx) +
        (u_data[flat_i + width] - u_data[flat_i - width]) / Float32(dy)

      out[flat_i] = abs(vorticity * 100_000f0)
    end

    # West/East edges, copy the neighbor.
    out[row_offset + 1]     = out[row_offset + 2]
    out[row_offset + width] = out[row_offset + width - 1]
  end

  # South/North edges, copy the neighbor.
  for i in 1:width
    out[i]                         = out[width + i]
    out[width*(grid.height-1) + i] = out[width*(grid.height-2) + i]
  end

  ()
end


# Let's estimate supercell likelihood to try not to miss events like Kansas 2019-5-17
#
# Reports:    https://www.spc.noaa.gov/climo/reports/190517_rpts.html
# Prediction: https://twitter.com/nadocast/status/1129390878269874176
#
# Algorithm to estimate initial forcing:
#
# 1. If SCP > 0, follow the bunkers storm motion upstream until SCP < 0.
# 2. Walk upstream one more hour. Now we've reached (presumably) the initiation zone.
# 3. Begin walking forwards. Sum up convergence. Continue walking/summing for 2 at least hours, continue until convergence <= 2.
# 4. The above mechanism estimates the amount of initiation forcing.
#
# Other considerations:
# 1. Abort and return 0 if the max SCP along the path < 1.
# 2. Abort and return 0 if we hit the edge.
#
# Actually maybe should just do 3-hour and 6-hour and 9-hour upstream convergence sum, gated by SCP > 1 + 1 hour.

function compute_upstream_mean(j_float :: Float32, i_float :: Float32, n_steps :: Int64, width, height, u_steps, v_steps, feature_data)
  value   = 0f0
  weight  = 0.00001f0

  @inbounds for _ in 1:n_steps
    j_closest = clamp(Int64(round(j_float)), 1, height)
    i_closest = clamp(Int64(round(i_float)), 1, width)
    flat_i_closest = width*(j_closest-1) + i_closest

    value  += feature_data[flat_i_closest]
    weight += 1f0

    j_float -= v_steps[flat_i_closest]
    i_float -= u_steps[flat_i_closest]
  end

  value / weight
end


# Follow wind vectors upstream so many hours and compute an average of the given feature.
#
# No interpolation along the way.
#
# If out is provided, mutates out
function compute_upstream_mean_threaded(; grid, u_data, v_data, feature_data, hours, step_size = 10*60, out = Array{Float32}(undef, length(grid.point_widths_miles)))
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)
  @assert length(grid.point_widths_miles) == length(feature_data)

  # out =  Array{Float32}(undef, length(grid.point_widths_miles))

  width  = grid.width
  height = grid.height

  u_steps = Float32(step_size) .* u_data ./ Float32.(grid.point_widths_miles  .* GeoUtils.METERS_PER_MILE)
  v_steps = Float32(step_size) .* v_data ./ Float32.(grid.point_heights_miles .* GeoUtils.METERS_PER_MILE)

  n_steps = 1 + hours*60*60÷step_size

  Threads.@threads for j in 1:height
    for i in 1:width
      flat_i = width*(j-1) + i
      out[flat_i] = compute_upstream_mean(Float32(j), Float32(i), n_steps, width, height, u_steps, v_steps, feature_data)
    end
  end

  out
end

# Sometimes needed before compute_upstream_mean_threaded
function meanify_threaded(feature_data, mean_is)
  out = zeros(Float32, size(feature_data))

  Threads.@threads for grid_i in 1:length(feature_data)
    val = 0f0

    @inbounds for near_i in mean_is[grid_i]
      val += feature_data[near_i]
    end

    out[grid_i] = val / Float32(length(mean_is[grid_i]))
  end

  out
end

function meanify_threaded2(grid, feature_data, mean_is2)
  out         = zeros(Float32, size(feature_data))
  row_cumsums = Array{Float64}(undef, (grid.width+1, grid.height))

  @assert (grid.width * grid.height) == length(feature_data)
  @assert length(mean_is2) == length(feature_data)

  width = grid.width

  Threads.@threads for j in 1:grid.height
    row_offset = width*(j-1)

    row_sum = 0.0
    row_cumsums[1,j] = row_sum

    @inbounds for i in 1:width
      flat_i = row_offset + i
      row_sum += Float64(feature_data[flat_i])
      row_cumsums[i+1,j] = row_sum
    end
  end

  Threads.@threads for grid_i in 1:length(feature_data)
    val = 0f0
    len = 0f0

    @inbounds for (row, range) in mean_is2[grid_i]
      val += Float32(row_cumsums[range.stop+1, row] - row_cumsums[range.start, row])
      len += length(range)
    end

    out[grid_i] = val / len
  end

  out
end



function make_data(
      grid                      :: Grids.Grid,
      inventory                 :: Vector{Inventories.InventoryLine},
      data                      :: Array{Float32,2},
      vector_wind_layers        :: Vector{String},
      layer_blocks_to_make      :: Vector{Int64}, # List of indices. See top of this file.
      twenty_five_mi_mean_is2   :: Vector{Vector{Tuple{Int64,UnitRange{Int64}}}}, # If not using, pass an empty vector.
      fifty_mi_mean_is2         :: Vector{Vector{Tuple{Int64,UnitRange{Int64}}}}, # If not using, pass an empty vector.
      hundred_mi_mean_is2       :: Vector{Vector{Tuple{Int64,UnitRange{Int64}}}};  # If not using, pass an empty vector.
      new_features_pre = [] # list of pairs of (feature_name, compute_feature_function(grid, get_layer_function))
    ) :: Array{Float32,2}


  grid_point_count  = size(data,1)
  raw_feature_count = size(data,2)
  pre_feature_count = raw_feature_count + length(new_features_pre)
  height            = grid.height
  width             = grid.width


  # Output for each feature:
  # raw data (but with relativized winds) + new_feature_pre
  # 25mi mean
  # 50mi mean
  # 100mi mean
  # 25mi forward grad, 25mi leftward grad, 25mi linestraddling grad
  # 50mi forward grad, 50mi leftward grad, 50mi linestraddling grad
  # 100mi forward grad, 100mi leftward grad, 100mi linestraddling grad
  #
  # And then a lone forecast hour field
  # And maybe feature interaction terms after that
  # feature_interaction_terms_count = length(feature_interaction_terms)


  # out = Array{Float32}(undef, (grid_point_count, length(layer_blocks_to_make)*pre_feature_count + 1 + feature_interaction_terms_count))
  out = Array{Float32}(undef, (grid_point_count, length(layer_blocks_to_make)*pre_feature_count))

  Threads.@threads for j in 1:raw_feature_count
    out[:,j] = @view data[:,j]
  end

  feature_key_to_i = Dict{String,Int64}()
  feature_keys = map(Inventories.inventory_line_key, inventory)

  for i in 1:length(inventory)
    feature_key_to_i[feature_keys[i]] = i
  end

  # Also load the pre computed features so later such features
  # can reference features computed earlier.
  #
  # Referenced by their plain name, not the inventory feature key
  for new_pre_feature_i in 1:length(new_features_pre)
    i = length(inventory) + new_pre_feature_i
    new_feature_name, _ = new_features_pre[new_pre_feature_i]
    feature_key_to_i[new_feature_name] = i
  end

  get_layer(key) = @view out[:, feature_key_to_i[key]]

  # Can't thread here. Terms may depend on prior computed terms.
  for new_pre_feature_i in 1:length(new_features_pre)
    new_feature_name, compute_new_feature_pre = new_features_pre[new_pre_feature_i]
    # out[:, raw_feature_count + new_pre_feature_i] = compute_new_feature_pre(grid, get_layer)
    compute_new_feature_pre(grid, get_layer, @view out[:, raw_feature_count + new_pre_feature_i])
  end

  block_i = 2

  if twenty_five_mi_mean_block in layer_blocks_to_make
    twenty_five_mi_mean_features_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if fifty_mi_mean_block in layer_blocks_to_make
    fifty_mi_mean_features_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if hundred_mi_mean_block in layer_blocks_to_make
    hundred_mi_mean_features_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
    twenty_five_mi_forward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
    twenty_five_mi_leftward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
    twenty_five_mi_linestraddling_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if fifty_mi_forward_gradient_block in layer_blocks_to_make
    fifty_mi_forward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if fifty_mi_leftward_gradient_block in layer_blocks_to_make
    fifty_mi_leftward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
    fifty_mi_linestraddling_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if hundred_mi_forward_gradient_block in layer_blocks_to_make
    hundred_mi_forward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if hundred_mi_leftward_gradient_block in layer_blocks_to_make
    hundred_mi_leftward_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end

  if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
    hundred_mi_linestraddling_gradient_range = feature_range(block_i, pre_feature_count)
    block_i += 1
  end


  should_make_twenty_five_mi_mean_block = twenty_five_mi_mean_block in layer_blocks_to_make
  should_make_fifty_mi_mean_block       = fifty_mi_mean_block       in layer_blocks_to_make
  should_make_hundred_mi_mean_block     = hundred_mi_mean_block     in layer_blocks_to_make

  if should_make_twenty_five_mi_mean_block || should_make_fifty_mi_mean_block || should_make_hundred_mi_mean_block
    # Don't want to deal with this
    @assert should_make_twenty_five_mi_mean_block
    @assert should_make_fifty_mi_mean_block
    @assert should_make_hundred_mi_mean_block

    make_mean_layers2!(
        out, pre_feature_count, grid,
        twenty_five_mi_mean_features_range, twenty_five_mi_mean_is2,
        fifty_mi_mean_features_range,       fifty_mi_mean_is2,
        hundred_mi_mean_features_range,     hundred_mi_mean_is2
      )
  end


  # Assume motion is in new_features_pre
  # HREF composite reflectivity movement near storm events is most correlated with mean between Bunkers and 500mb wind
  motion_us = get_layer("U½STM½500mb")[:]
  motion_vs = get_layer("V½STM½500mb")[:]

  # motion_angles = atan.(motion_vs, motion_us)

  # if any(isnan, motion_angles)
  #   error("nan wind angle")
  # end

  # rot_coses = cos.(-motion_angles)
  # rot_sines = sin.(-motion_angles)

  motion_normalized_us, motion_normalized_vs = normalize_uvs!(motion_us, motion_vs)

  # @assert rot_coses ≈ cos.(motion_angles)
  # @assert motion_normalized_us ≈ rot_coses
  # @assert .-motion_normalized_vs ≈ rot_sines

  rot_coses = motion_normalized_us
  rot_sines = .-motion_normalized_vs

  # Compute several "convolution" kernels by sampling the mean layers.

  # Forward gradient
  #
  #     -     +
  #   - - - + + +
  #     -     +
  #

  # Leftward gradient
  #
  #       +
  #     + + +
  #       +
  #       -
  #     - - -
  #       -

  # Linestradling (the +/- regions actually overlap a bit but you get the idea)
  #
  #         -
  #       - - -
  #     +   -   +
  #   + + +   + + +
  #     +   -   +
  #       - - -
  #         -

  # is_on_grid(w_to_e_col, s_to_n_row) = begin
  #   w_to_e_col >= 1 && w_to_e_col <= width &&
  #   s_to_n_row >= 1 && s_to_n_row <= height
  # end

  twenty_five_mi_forward_is   = Array{Int64}(undef, grid_point_count)
  twenty_five_mi_backward_is  = Array{Int64}(undef, grid_point_count)
  twenty_five_mi_leftward_is  = Array{Int64}(undef, grid_point_count)
  twenty_five_mi_rightward_is = Array{Int64}(undef, grid_point_count)
  fifty_mi_forward_is         = Array{Int64}(undef, grid_point_count)
  fifty_mi_backward_is        = Array{Int64}(undef, grid_point_count)
  fifty_mi_leftward_is        = Array{Int64}(undef, grid_point_count)
  fifty_mi_rightward_is       = Array{Int64}(undef, grid_point_count)
  hundred_mi_forward_is       = Array{Int64}(undef, grid_point_count)
  hundred_mi_backward_is      = Array{Int64}(undef, grid_point_count)
  hundred_mi_leftward_is      = Array{Int64}(undef, grid_point_count)
  hundred_mi_rightward_is     = Array{Int64}(undef, grid_point_count)

  compute_directional_is!(
      height, width,
      grid.point_heights_miles, grid.point_widths_miles,
      # mean_wind_lower_half_atmosphere_us, mean_wind_lower_half_atmosphere_vs,
      # motion_us, motion_vs,
      motion_normalized_us, motion_normalized_vs,
      twenty_five_mi_forward_is, twenty_five_mi_backward_is, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is,
      fifty_mi_forward_is,       fifty_mi_backward_is,       fifty_mi_leftward_is,       fifty_mi_rightward_is,
      hundred_mi_forward_is,     hundred_mi_backward_is,     hundred_mi_leftward_is,     hundred_mi_rightward_is
    )

  if should_make_twenty_five_mi_mean_block
    Threads.@threads for pre_layer_feature_i in 1:pre_feature_count
    # for pre_layer_feature_i in 1:pre_feature_count
      twenty_five_mi_mean_feature_i = pre_layer_feature_i + twenty_five_mi_mean_features_range.start - 1

      # twenty_five_mi_forward_vals   = @view out[twenty_five_mi_forward_is,   twenty_five_mi_mean_feature_i]
      # twenty_five_mi_backward_vals  = @view out[twenty_five_mi_backward_is,  twenty_five_mi_mean_feature_i]
      # twenty_five_mi_leftward_vals  = @view out[twenty_five_mi_leftward_is,  twenty_five_mi_mean_feature_i]
      # twenty_five_mi_rightward_vals = @view out[twenty_five_mi_rightward_is, twenty_five_mi_mean_feature_i]

      if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
        twenty_five_mi_forward_gradient_i         = pre_layer_feature_i + twenty_five_mi_forward_gradient_range.start - 1
        # out[:, twenty_five_mi_forward_gradient_i] = twenty_five_mi_forward_vals .- twenty_five_mi_backward_vals
        compute_gradient!(out, twenty_five_mi_forward_gradient_i, twenty_five_mi_mean_feature_i, twenty_five_mi_forward_is, twenty_five_mi_backward_is)
      end

      if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
        twenty_five_mi_leftward_gradient_i         = pre_layer_feature_i + twenty_five_mi_leftward_gradient_range.start - 1
        # out[:, twenty_five_mi_leftward_gradient_i] = twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
        compute_gradient!(out, twenty_five_mi_leftward_gradient_i, twenty_five_mi_mean_feature_i, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is)
      end

      if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
        twenty_five_mi_linestraddling_gradient_i         = pre_layer_feature_i + twenty_five_mi_linestraddling_gradient_range.start - 1
        compute_linestraddling_gradient!(out, twenty_five_mi_linestraddling_gradient_i, twenty_five_mi_mean_feature_i, twenty_five_mi_forward_is, twenty_five_mi_backward_is, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is)
        # out[:, twenty_five_mi_linestraddling_gradient_i] = twenty_five_mi_forward_vals .+ twenty_five_mi_backward_vals .- twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
      end
    end
  end

  if should_make_fifty_mi_mean_block
    Threads.@threads for pre_layer_feature_i in 1:pre_feature_count
    # for pre_layer_feature_i in 1:pre_feature_count
      fifty_mi_mean_feature_i = pre_layer_feature_i + fifty_mi_mean_features_range.start - 1

      # fifty_mi_forward_vals   = @view out[fifty_mi_forward_is,   fifty_mi_mean_feature_i]
      # fifty_mi_backward_vals  = @view out[fifty_mi_backward_is,  fifty_mi_mean_feature_i]
      # fifty_mi_leftward_vals  = @view out[fifty_mi_leftward_is,  fifty_mi_mean_feature_i]
      # fifty_mi_rightward_vals = @view out[fifty_mi_rightward_is, fifty_mi_mean_feature_i]

      if fifty_mi_forward_gradient_block in layer_blocks_to_make
        fifty_mi_forward_gradient_i         = pre_layer_feature_i + fifty_mi_forward_gradient_range.start - 1
        # out[:, fifty_mi_forward_gradient_i] = fifty_mi_forward_vals .- fifty_mi_backward_vals
        compute_gradient!(out, fifty_mi_forward_gradient_i, fifty_mi_mean_feature_i, fifty_mi_forward_is, fifty_mi_backward_is)
      end

      if fifty_mi_leftward_gradient_block in layer_blocks_to_make
        fifty_mi_leftward_gradient_i         = pre_layer_feature_i + fifty_mi_leftward_gradient_range.start - 1
        # out[:, fifty_mi_leftward_gradient_i] = fifty_mi_leftward_vals .- fifty_mi_rightward_vals
        compute_gradient!(out, fifty_mi_leftward_gradient_i, fifty_mi_mean_feature_i, fifty_mi_leftward_is, fifty_mi_rightward_is)
      end

      if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
        fifty_mi_linestraddling_gradient_i         = pre_layer_feature_i + fifty_mi_linestraddling_gradient_range.start - 1
        compute_linestraddling_gradient!(out, fifty_mi_linestraddling_gradient_i, fifty_mi_mean_feature_i, fifty_mi_forward_is, fifty_mi_backward_is, fifty_mi_leftward_is, fifty_mi_rightward_is)
        # out[:, fifty_mi_linestraddling_gradient_i] = fifty_mi_forward_vals .+ fifty_mi_backward_vals .- fifty_mi_leftward_vals .- fifty_mi_rightward_vals
      end
    end
  end

  if should_make_hundred_mi_mean_block
    Threads.@threads for pre_layer_feature_i in 1:pre_feature_count
    # for pre_layer_feature_i in 1:pre_feature_count
      hundred_mi_mean_feature_i = pre_layer_feature_i + hundred_mi_mean_features_range.start - 1

      hundred_mi_forward_vals   = @view out[hundred_mi_forward_is,   hundred_mi_mean_feature_i]
      hundred_mi_backward_vals  = @view out[hundred_mi_backward_is,  hundred_mi_mean_feature_i]
      hundred_mi_leftward_vals  = @view out[hundred_mi_leftward_is,  hundred_mi_mean_feature_i]
      hundred_mi_rightward_vals = @view out[hundred_mi_rightward_is, hundred_mi_mean_feature_i]

      if hundred_mi_forward_gradient_block in layer_blocks_to_make
        hundred_mi_forward_gradient_i         = pre_layer_feature_i + hundred_mi_forward_gradient_range.start - 1
        # out[:, hundred_mi_forward_gradient_i] = hundred_mi_forward_vals  .- hundred_mi_backward_vals
        compute_gradient!(out, hundred_mi_forward_gradient_i, hundred_mi_mean_feature_i, hundred_mi_forward_is, hundred_mi_backward_is)
      end

      if hundred_mi_leftward_gradient_block in layer_blocks_to_make
        hundred_mi_leftward_gradient_i         = pre_layer_feature_i + hundred_mi_leftward_gradient_range.start - 1
        # out[:, hundred_mi_leftward_gradient_i] = hundred_mi_leftward_vals .- hundred_mi_rightward_vals
        compute_gradient!(out, hundred_mi_leftward_gradient_i, hundred_mi_mean_feature_i, hundred_mi_leftward_is, hundred_mi_rightward_is)
      end

      if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
        hundred_mi_linestraddling_gradient_i         = pre_layer_feature_i + hundred_mi_linestraddling_gradient_range.start - 1
        compute_linestraddling_gradient!(out, hundred_mi_linestraddling_gradient_i, hundred_mi_mean_feature_i, hundred_mi_forward_is, hundred_mi_backward_is, hundred_mi_leftward_is, hundred_mi_rightward_is)
        # out[:, hundred_mi_linestraddling_gradient_i] = hundred_mi_forward_vals  .+ hundred_mi_backward_vals .- hundred_mi_leftward_vals .- hundred_mi_rightward_vals
      end
    end
  end

  Threads.@threads for wind_layer_key in vector_wind_layers
  # for wind_layer_key in vector_wind_layers
    if wind_layer_key == "VCSH:6000-0 m above ground:hour fcst:"
      layer_key_u = "VUCSH:6000-0 m above ground:hour fcst:"
      layer_key_v = "VVCSH:6000-0 m above ground:hour fcst:"
    elseif wind_layer_key == "VCSH:1000-0 m above ground:hour fcst:"
      layer_key_u = "VUCSH:1000-0 m above ground:hour fcst:"
      layer_key_v = "VVCSH:1000-0 m above ground:hour fcst:"
    else
      layer_key_u = "U" * wind_layer_key
      layer_key_v = "V" * wind_layer_key
    end

    # And rotate everrrryything.

    raw_layer_u_i = feature_key_to_i[layer_key_u]
    raw_layer_v_i = feature_key_to_i[layer_key_v]

    rotate_uv_layers!(out, raw_layer_u_i, raw_layer_v_i, rot_coses, rot_sines)

    if twenty_five_mi_mean_block in layer_blocks_to_make
      twenty_five_mi_mean_u_i = raw_layer_u_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_v_i = raw_layer_v_i + twenty_five_mi_mean_features_range.start - 1
      rotate_uv_layers!(out, twenty_five_mi_mean_u_i, twenty_five_mi_mean_v_i, rot_coses, rot_sines)
    end

    if fifty_mi_mean_block in layer_blocks_to_make
      fifty_mi_mean_u_i = raw_layer_u_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_v_i = raw_layer_v_i + fifty_mi_mean_features_range.start - 1
      rotate_uv_layers!(out, fifty_mi_mean_u_i, fifty_mi_mean_v_i, rot_coses, rot_sines)
    end

    if hundred_mi_mean_block in layer_blocks_to_make
      hundred_mi_mean_u_i = raw_layer_u_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_v_i = raw_layer_v_i + hundred_mi_mean_features_range.start - 1
      rotate_uv_layers!(out, hundred_mi_mean_u_i, hundred_mi_mean_v_i, rot_coses, rot_sines)
    end

    if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
      twenty_five_mi_forward_gradient_u_i = raw_layer_u_i + twenty_five_mi_forward_gradient_range.start - 1
      twenty_five_mi_forward_gradient_v_i = raw_layer_v_i + twenty_five_mi_forward_gradient_range.start - 1
      rotate_uv_layers!(out, twenty_five_mi_forward_gradient_u_i, twenty_five_mi_forward_gradient_v_i, rot_coses, rot_sines)
    end

    if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
      twenty_five_mi_leftward_gradient_u_i = raw_layer_u_i + twenty_five_mi_leftward_gradient_range.start - 1
      twenty_five_mi_leftward_gradient_v_i = raw_layer_v_i + twenty_five_mi_leftward_gradient_range.start - 1
      rotate_uv_layers!(out, twenty_five_mi_leftward_gradient_u_i, twenty_five_mi_leftward_gradient_v_i, rot_coses, rot_sines)
    end

    if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
      twenty_five_mi_linestraddling_gradient_u_i = raw_layer_u_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      twenty_five_mi_linestraddling_gradient_v_i = raw_layer_v_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      rotate_uv_layers!(out, twenty_five_mi_linestraddling_gradient_u_i, twenty_five_mi_linestraddling_gradient_v_i, rot_coses, rot_sines)
    end

    if fifty_mi_forward_gradient_block in layer_blocks_to_make
      fifty_mi_forward_gradient_u_i = raw_layer_u_i + fifty_mi_forward_gradient_range.start - 1
      fifty_mi_forward_gradient_v_i = raw_layer_v_i + fifty_mi_forward_gradient_range.start - 1
      rotate_uv_layers!(out, fifty_mi_forward_gradient_u_i, fifty_mi_forward_gradient_v_i, rot_coses, rot_sines)
    end

    if fifty_mi_leftward_gradient_block in layer_blocks_to_make
      fifty_mi_leftward_gradient_u_i = raw_layer_u_i + fifty_mi_leftward_gradient_range.start - 1
      fifty_mi_leftward_gradient_v_i = raw_layer_v_i + fifty_mi_leftward_gradient_range.start - 1
      rotate_uv_layers!(out, fifty_mi_leftward_gradient_u_i, fifty_mi_leftward_gradient_v_i, rot_coses, rot_sines)
    end

    if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
      fifty_mi_linestraddling_gradient_u_i = raw_layer_u_i + fifty_mi_linestraddling_gradient_range.start - 1
      fifty_mi_linestraddling_gradient_v_i = raw_layer_v_i + fifty_mi_linestraddling_gradient_range.start - 1
      rotate_uv_layers!(out, fifty_mi_linestraddling_gradient_u_i, fifty_mi_linestraddling_gradient_v_i, rot_coses, rot_sines)
    end

    if hundred_mi_forward_gradient_block in layer_blocks_to_make
      hundred_mi_forward_gradient_u_i = raw_layer_u_i + hundred_mi_forward_gradient_range.start - 1
      hundred_mi_forward_gradient_v_i = raw_layer_v_i + hundred_mi_forward_gradient_range.start - 1
      rotate_uv_layers!(out, hundred_mi_forward_gradient_u_i, hundred_mi_forward_gradient_v_i, rot_coses, rot_sines)
    end

    if hundred_mi_leftward_gradient_block in layer_blocks_to_make
      hundred_mi_leftward_gradient_u_i      = raw_layer_u_i + hundred_mi_leftward_gradient_range.start - 1
      hundred_mi_leftward_gradient_v_i      = raw_layer_v_i + hundred_mi_leftward_gradient_range.start - 1
      rotate_uv_layers!(out, hundred_mi_leftward_gradient_u_i, hundred_mi_leftward_gradient_v_i, rot_coses, rot_sines)
    end

    if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
      hundred_mi_linestraddling_gradient_u_i = raw_layer_u_i + hundred_mi_linestraddling_gradient_range.start - 1
      hundred_mi_linestraddling_gradient_v_i = raw_layer_v_i + hundred_mi_linestraddling_gradient_range.start - 1
      rotate_uv_layers!(out, hundred_mi_linestraddling_gradient_u_i, hundred_mi_linestraddling_gradient_v_i, rot_coses, rot_sines)
    end
  end

  out
end

end # module FeatureEngineeringShared