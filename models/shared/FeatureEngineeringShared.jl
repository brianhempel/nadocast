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

# leftover_fields = [
#   "div(forecast hour, 10)",
# ]
forecast_hour_feature_post(grid) =
  ( "div(forecast hour, 10)"
  , forecast -> fill(Float32(div(forecast.forecast_hour, 10)), length(grid.latlons))
  )

# Returns twenty_five_mi_mean_is, unique_fifty_mi_mean_is, unique_hundred_mi_mean_is
function compute_mean_is(grid)
  print("computing radius indices...")
  point_size_km = 1.61 * sqrt(grid.point_areas_sq_miles[div(length(grid.point_areas_sq_miles), 2)])
  cache_folder = @sprintf "grid_%.1fkm_%dx%d_downsample_%dx_%.2f_%.2f_%.2f-%.2f" point_size_km grid.width grid.height grid.downsample grid.min_lat grid.max_lat grid.min_lon grid.max_lon
  twenty_five_mi_mean_is    = Cache.cached(() -> Grids.radius_grid_is(grid, 25.0),                                                                       [cache_folder], "twenty_five_mi_mean_is")
  unique_fifty_mi_mean_is   = Cache.cached(() -> Grids.radius_grid_is_less_other_is(grid, 50.0, twenty_five_mi_mean_is),                                 [cache_folder], "unique_fifty_mi_mean_is")
  unique_hundred_mi_mean_is = Cache.cached(() -> Grids.radius_grid_is_less_other_is(grid, 100.0, vcat(twenty_five_mi_mean_is, unique_fifty_mi_mean_is)), [cache_folder], "unique_hundred_mi_mean_is")
  println("done")

  return (twenty_five_mi_mean_is, unique_fifty_mi_mean_is, unique_hundred_mi_mean_is)
end

# new_features_pre should be a list of pairs of (feature_name, compute_feature_function(grid, inventory, data))
function feature_engineered_forecasts(base_forecasts; vector_wind_layers, layer_blocks_to_make, new_features_pre = [])

  if isempty(base_forecasts)
    return base_forecasts
  end

  layer_blocks_to_make = sort(layer_blocks_to_make) # Layers always produced in a particular order regardless of given order. It's a set.

  grid = base_forecasts[1].grid

  twenty_five_mi_mean_is, unique_fifty_mi_mean_is, unique_hundred_mi_mean_is = compute_mean_is(grid)

  inventory_transformer(base_forecast, base_inventory) = begin

    new_inventory = Inventories.InventoryLine[]

    new_features_pre_lines =
      map(new_features_pre) do feature_name_and_compute_function
        Inventories.InventoryLine(
          "",                                   # message_dot_submessage
          "",                                   # position_str
          base_inventory[1].date_str,
          feature_name_and_compute_function[1], # abbrev
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

    # for leftover_field in leftover_fields
    #   leftover_line =
    #     Inventories.InventoryLine(
    #       "",             # message_dot_submessage
    #       "",             # position_str
    #       base_inventory[1].date_str,
    #       leftover_field, # abbrev
    #       "calculated",   # level
    #       "hour fcst",    # forecast_hour_str
    #       "",             # misc
    #       ""              # feature_engineering
    #     )
    #
    #   push!(new_inventory, leftover_line)
    # end

    # for interaction_is in feature_interaction_terms
    #   names_of_interacting_features =
    #     map(interaction_is) do feature_i
    #       line = new_inventory[feature_i]
    #       join([line.abbrev, line.level, Inventories.generic_forecast_hour_str(line.forecast_hour_str), line.misc, line.feature_engineering], ".")
    #     end
    #
    #   interaction_line =
    #     Inventories.InventoryLine(
    #       "",             # message_dot_submessage
    #       "",             # position_str
    #       base_inventory[1].date_str,
    #       join(names_of_interacting_features, " * "), # abbrev
    #       "calculated",   # level
    #       "hour fcst",    # forecast_hour_str
    #       "interaction",  # misc
    #       ""              # feature_engineering
    #     )
    #
    #   push!(new_inventory, interaction_line)
    # end

    new_inventory
  end

  data_transformer(base_forecast, base_data) = begin
    # println("Feature engineering $(base_forecast.model_name) $(Forecasts.time_title(base_forecast))...")

    out = make_data(
      grid,
      Forecasts.inventory(base_forecast),
      base_data,
      vector_wind_layers,
      layer_blocks_to_make,
      twenty_five_mi_mean_is,
      unique_fifty_mi_mean_is,
      unique_hundred_mi_mean_is;
      new_features_pre = new_features_pre
    )

    # println("done.")
    out
  end

  ForecastCombinators.map_forecasts(base_forecasts; inventory_transformer = inventory_transformer, data_transformer = data_transformer)
end


# function feature_i_to_name(inventory :: Vector{Inventories.InventoryLine}, layer_blocks_to_make :: Vector{Int64}, feature_i :: Int64; feature_interaction_terms = []) :: String
#   pre_feature_count = length(inventory)
#
#   layer_blocks_to_make = sort(layer_blocks_to_make) # Layers always produced in a particular order regardless of given order. It's a set.
#
#   output_block_i, feature_i_in_block = divrem(feature_i - 1, pre_feature_count)
#
#   output_block_i += 1
#   feature_i_in_block += 1
#
#   if output_block_i >= 1 && output_block_i <= length(layer_blocks_to_make)
#     block_name = feature_block_names[layer_blocks_to_make[output_block_i]]
#     return Inventories.inventory_line_key(inventory[feature_i_in_block]) * ":" * block_name
#   elseif output_block_i > length(layer_blocks_to_make)
#     leftover_feature_i = feature_i - length(layer_blocks_to_make)*pre_feature_count
#
#     if leftover_feature_i <= length(leftover_fields)
#       return join([leftover_fields[leftover_feature_i], "calculated", "hour fcst", "calculated", ""], ":")
#     end
#
#     interaction_feature_i = leftover_feature_i - length(leftover_fields)
#
#     if interaction_feature_i <= length(feature_interaction_terms)
#       names_of_interacting_features =
#         map(feature_interaction_terms[interaction_feature_i]) do feature_i
#           feature_i_to_name(inventory, layer_blocks_to_make, feature_i, feature_interaction_terms = feature_interaction_terms)
#         end
#
#       return join(names_of_interacting_features, "*")
#     end
#   end
#
#   "Unknown feature $feature_i"
# end

function feature_range(block_i, pre_feature_count)
  (block_i-1)*pre_feature_count+1:block_i*pre_feature_count
end

function make_mean_part(out, mean_is, total1, total2, total3, total4, pre_layer_feature1_i, pre_layer_feature2_i, pre_layer_feature3_i, pre_layer_feature4_i)
  @inbounds for near_i in mean_is
    total1 += out[near_i, pre_layer_feature1_i]
    total2 += out[near_i, pre_layer_feature2_i]
    total3 += out[near_i, pre_layer_feature3_i]
    total4 += out[near_i, pre_layer_feature4_i]
    # total5 += out[near_i, pre_layer_feature5_i]
    # total6 += out[near_i, pre_layer_feature6_i]
    # total7 += out[near_i, pre_layer_feature7_i]
    # total8 += out[near_i, pre_layer_feature8_i]
    # n      += 1.0f0
  end

  (total1, total2, total3, total4)
end

# Make 25mi, 50mi, and 100mi mean layers
#
# Mutates out
#
# This is still the majority time-consumer.
function make_mean_layers(
    out, pre_feature_count, grid_point_count,
    should_make_twenty_five_mi_mean_block, twenty_five_mi_mean_features_range, twenty_five_mi_mean_is,
    should_make_fifty_mi_mean_block,       fifty_mi_mean_features_range,       unique_fifty_mi_mean_is,
    should_make_hundred_mi_mean_block,     hundred_mi_mean_features_range,     unique_hundred_mi_mean_is
  )

  # SREF numbers:
  # 2 at a time: 3.71s per 5 forecasts
  # 3 at a time: 3.55s per 5 forecasts
  # 4 at a time: 3.35s per 5 forecasts
  # 5 at a time: 3.35s per 5 forecasts
  # 6 at a time: 3.30s per 5 forecasts
  # 7 at a time: 3.34s per 5 forecasts
  # 8 at a time: 3.33s per 5 forecasts

  # HREF numbers:
  # 4 at a time: 9.7s per 3
  # 6 at a time: 9.7s per 3
  # 8 at a time: 9.8s per 3

  Threads.@threads for pre_layer_feature1_i in 1:4:pre_feature_count
  # for pre_layer_feature1_i in 1:4:pre_feature_count
    pre_layer_feature2_i = min(pre_layer_feature1_i + 1, pre_feature_count)
    pre_layer_feature3_i = min(pre_layer_feature1_i + 2, pre_feature_count)
    pre_layer_feature4_i = min(pre_layer_feature1_i + 3, pre_feature_count)
    # pre_layer_feature5_i = min(pre_layer_feature1_i + 4, pre_feature_count)
    # pre_layer_feature6_i = min(pre_layer_feature1_i + 5, pre_feature_count)
    # pre_layer_feature7_i = min(pre_layer_feature1_i + 6, pre_feature_count)
    # pre_layer_feature8_i = min(pre_layer_feature1_i + 7, pre_feature_count)

    if should_make_twenty_five_mi_mean_block
      twenty_five_mi_mean_feature1_i   = pre_layer_feature1_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_feature2_i   = pre_layer_feature2_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_feature3_i   = pre_layer_feature3_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_feature4_i   = pre_layer_feature4_i + twenty_five_mi_mean_features_range.start - 1
      # twenty_five_mi_mean_feature5_i   = pre_layer_feature5_i + twenty_five_mi_mean_features_range.start - 1
      # twenty_five_mi_mean_feature6_i   = pre_layer_feature6_i + twenty_five_mi_mean_features_range.start - 1
      # twenty_five_mi_mean_feature7_i   = pre_layer_feature7_i + twenty_five_mi_mean_features_range.start - 1
      # twenty_five_mi_mean_feature8_i   = pre_layer_feature8_i + twenty_five_mi_mean_features_range.start - 1
    end

    if should_make_fifty_mi_mean_block
      fifty_mi_mean_feature1_i   = pre_layer_feature1_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_feature2_i   = pre_layer_feature2_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_feature3_i   = pre_layer_feature3_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_feature4_i   = pre_layer_feature4_i + fifty_mi_mean_features_range.start - 1
      # fifty_mi_mean_feature5_i   = pre_layer_feature5_i + fifty_mi_mean_features_range.start - 1
      # fifty_mi_mean_feature6_i   = pre_layer_feature6_i + fifty_mi_mean_features_range.start - 1
      # fifty_mi_mean_feature7_i   = pre_layer_feature7_i + fifty_mi_mean_features_range.start - 1
      # fifty_mi_mean_feature8_i   = pre_layer_feature8_i + fifty_mi_mean_features_range.start - 1
    end

    if should_make_hundred_mi_mean_block
      hundred_mi_mean_feature1_i = pre_layer_feature1_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_feature2_i = pre_layer_feature2_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_feature3_i = pre_layer_feature3_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_feature4_i = pre_layer_feature4_i + hundred_mi_mean_features_range.start - 1
      # hundred_mi_mean_feature5_i = pre_layer_feature5_i + hundred_mi_mean_features_range.start - 1
      # hundred_mi_mean_feature6_i = pre_layer_feature6_i + hundred_mi_mean_features_range.start - 1
      # hundred_mi_mean_feature7_i = pre_layer_feature7_i + hundred_mi_mean_features_range.start - 1
      # hundred_mi_mean_feature8_i = pre_layer_feature8_i + hundred_mi_mean_features_range.start - 1
    end

    @inbounds for flat_i in 1:grid_point_count
      total1 = 0.0f0
      total2 = 0.0f0
      total3 = 0.0f0
      total4 = 0.0f0
      # total5 = 0.0f0
      # total6 = 0.0f0
      # total7 = 0.0f0
      # total8 = 0.0f0
      n      = 0.0f0
      # for near_i in twenty_five_mi_mean_is[flat_i]
      #   total1 += out[near_i, pre_layer_feature1_i]
      #   total2 += out[near_i, pre_layer_feature2_i]
      #   total3 += out[near_i, pre_layer_feature3_i]
      #   total4 += out[near_i, pre_layer_feature4_i]
      #   # total5 += out[near_i, pre_layer_feature5_i]
      #   # total6 += out[near_i, pre_layer_feature6_i]
      #   # total7 += out[near_i, pre_layer_feature7_i]
      #   # total8 += out[near_i, pre_layer_feature8_i]
      #   n      += 1.0f0
      # end
      total1, total2, total3, total4 =
        make_mean_part(
            out, twenty_five_mi_mean_is[flat_i],
            total1, total2, total3, total4,
            pre_layer_feature1_i, pre_layer_feature2_i, pre_layer_feature3_i, pre_layer_feature4_i
          )
      n += Float32(length(twenty_five_mi_mean_is[flat_i]))
      if should_make_twenty_five_mi_mean_block
        out[flat_i, twenty_five_mi_mean_feature1_i] = total1 / n
        out[flat_i, twenty_five_mi_mean_feature2_i] = total2 / n
        out[flat_i, twenty_five_mi_mean_feature3_i] = total3 / n
        out[flat_i, twenty_five_mi_mean_feature4_i] = total4 / n
        # out[flat_i, twenty_five_mi_mean_feature5_i] = total5 / n
        # out[flat_i, twenty_five_mi_mean_feature6_i] = total6 / n
        # out[flat_i, twenty_five_mi_mean_feature7_i] = total7 / n
        # out[flat_i, twenty_five_mi_mean_feature8_i] = total8 / n
      end
      if should_make_fifty_mi_mean_block || should_make_hundred_mi_mean_block
        # for near_i in unique_fifty_mi_mean_is[flat_i]
        #   total1 += out[near_i, pre_layer_feature1_i]
        #   total2 += out[near_i, pre_layer_feature2_i]
        #   total3 += out[near_i, pre_layer_feature3_i]
        #   total4 += out[near_i, pre_layer_feature4_i]
        #   # total5 += out[near_i, pre_layer_feature5_i]
        #   # total6 += out[near_i, pre_layer_feature6_i]
        #   # total7 += out[near_i, pre_layer_feature7_i]
        #   # total8 += out[near_i, pre_layer_feature8_i]
        #   n      += 1.0f0
        # end
        total1, total2, total3, total4 =
          make_mean_part(
              out, unique_fifty_mi_mean_is[flat_i],
              total1, total2, total3, total4,
              pre_layer_feature1_i, pre_layer_feature2_i, pre_layer_feature3_i, pre_layer_feature4_i
            )
        n += Float32(length(unique_fifty_mi_mean_is[flat_i]))
        if should_make_fifty_mi_mean_block
          out[flat_i, fifty_mi_mean_feature1_i] = total1 / n
          out[flat_i, fifty_mi_mean_feature2_i] = total2 / n
          out[flat_i, fifty_mi_mean_feature3_i] = total3 / n
          out[flat_i, fifty_mi_mean_feature4_i] = total4 / n
          # out[flat_i, fifty_mi_mean_feature5_i] = total5 / n
          # out[flat_i, fifty_mi_mean_feature6_i] = total6 / n
          # out[flat_i, fifty_mi_mean_feature7_i] = total7 / n
          # out[flat_i, fifty_mi_mean_feature8_i] = total8 / n
        end
      end
      if should_make_hundred_mi_mean_block
        total1, total2, total3, total4 =
          make_mean_part(
              out, unique_hundred_mi_mean_is[flat_i],
              total1, total2, total3, total4,
              pre_layer_feature1_i, pre_layer_feature2_i, pre_layer_feature3_i, pre_layer_feature4_i
            )
        n += Float32(length(unique_hundred_mi_mean_is[flat_i]))
        # for near_i in unique_hundred_mi_mean_is[flat_i]
        #   total1 += out[near_i, pre_layer_feature1_i]
        #   total2 += out[near_i, pre_layer_feature2_i]
        #   total3 += out[near_i, pre_layer_feature3_i]
        #   total4 += out[near_i, pre_layer_feature4_i]
        #   # total5 += out[near_i, pre_layer_feature5_i]
        #   # total6 += out[near_i, pre_layer_feature6_i]
        #   # total7 += out[near_i, pre_layer_feature7_i]
        #   # total8 += out[near_i, pre_layer_feature8_i]
        #   n       += 1.0f0
        # end
        out[flat_i, hundred_mi_mean_feature1_i] = total1 / n
        out[flat_i, hundred_mi_mean_feature2_i] = total2 / n
        out[flat_i, hundred_mi_mean_feature3_i] = total3 / n
        out[flat_i, hundred_mi_mean_feature4_i] = total4 / n
        # out[flat_i, hundred_mi_mean_feature5_i] = total5 / n
        # out[flat_i, hundred_mi_mean_feature6_i] = total6 / n
        # out[flat_i, hundred_mi_mean_feature7_i] = total7 / n
        # out[flat_i, hundred_mi_mean_feature8_i] = total8 / n
      end
    end
  end

  ()
end

function uv_normalize(u, v)
  ε = 1f-8
  len = √(u^2 + v^2) + ε
  (u / len, v / len)
end

# Mutates all the something_is arguments.
#
# Once we pull this out into a function then our allocations here go way down...to zero I believe.
function compute_directional_is(
    height, width,
    point_heights_miles, point_widths_miles,
    mean_wind_lower_half_atmosphere_us, mean_wind_lower_half_atmosphere_vs,
    twenty_five_mi_forward_is, twenty_five_mi_backward_is, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is,
    fifty_mi_forward_is,       fifty_mi_backward_is,       fifty_mi_leftward_is,       fifty_mi_rightward_is,
    hundred_mi_forward_is,     hundred_mi_backward_is,     hundred_mi_leftward_is,     hundred_mi_rightward_is
  )
  for j in 1:height
    for i in 1:width
      flat_i = width*(j-1) + i

      relative_u, relative_v = uv_normalize(mean_wind_lower_half_atmosphere_us[flat_i], mean_wind_lower_half_atmosphere_vs[flat_i])

      point_height = Float32(point_heights_miles[flat_i])
      point_width  = Float32(point_widths_miles[flat_i]) # On the SREF grid, point_width and point_height are always nearly equal, <1.0% difference.

      delta_25mi_j  = round(Int64, relative_v * 25.0f0  / point_height)
      delta_25mi_i  = round(Int64, relative_u * 25.0f0  / point_width)
      delta_50mi_j  = round(Int64, relative_v * 50.0f0  / point_height)
      delta_50mi_i  = round(Int64, relative_u * 50.0f0  / point_width)
      delta_100mi_j = round(Int64, relative_v * 100.0f0 / point_height)
      delta_100mi_i = round(Int64, relative_u * 100.0f0 / point_width)

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

function get_feature_i(inventory, feature_key)
  findfirst(inventory) do inventory_line
    Inventories.inventory_line_key(inventory_line) == feature_key
  end
end

function feature_key_to_interaction_feature_name(feature_key)
  replace(feature_key, r"[: ]" => "")
end

# Returns (feature_name, compute_feature_function(grid, inventory, data))
function make_interaction_feature(feature_keys :: Vector{String})
  compute_feature_function(grid, get_layer) = begin
    out = ones(Float32, length(grid.latlons))
    for feature_key in feature_keys
      out .*= get_layer(feature_key)
    end
    out
  end
  feature_name = join(map(feature_key_to_interaction_feature_name, feature_keys), "*")
  (feature_name, compute_feature_function)
end

function compute_divergence_threaded(grid, u_data, v_data)
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)

  out =  Array{Float32}(undef, length(grid.point_widths_miles))

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

  out
end

function compute_vorticity_threaded(grid, u_data, v_data)
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)

  out =  Array{Float32}(undef, length(grid.point_widths_miles))

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

      out[flat_i] = vorticity * 100_000f0
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

  out
end

# Follow wind vectors upstream so many hours and compute an average of the given feature.
#
# No interpolation along the way.
function compute_upstream_mean_threaded(; grid, u_data, v_data, feature_data, hours, step_size = 10*60)
  @assert length(grid.point_widths_miles) == length(u_data)
  @assert length(grid.point_widths_miles) == length(v_data)
  @assert length(grid.point_widths_miles) == length(feature_data)

  out =  Array{Float32}(undef, length(grid.point_widths_miles))

  width  = grid.width
  height = grid.height

  Threads.@threads for j in 1:height
    for i in 1:width
      j_float = j
      i_float = i
      value   = 0f0
      weight  = 0.00001f0
      seconds = 0f0

      @inbounds while true
        j_closest = clamp(Int64(round(j_float)), 1, height)
        i_closest = clamp(Int64(round(i_float)), 1, width)
        flat_i_closest = width*(j_closest-1) + i_closest

        value  += feature_data[flat_i_closest]
        weight += 1f0

        seconds += step_size

        if seconds > hours*60*60
          break
        end

        j_float -= step_size * v_data[flat_i_closest] / Float32(grid.point_heights_miles[flat_i_closest] * GeoUtils.METERS_PER_MILE)
        i_float -= step_size * u_data[flat_i_closest] / Float32(grid.point_widths_miles[flat_i_closest]  * GeoUtils.METERS_PER_MILE)
      end

      out[width*(j-1) + i] = value / weight
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



function make_data(
      grid                      :: Grids.Grid,
      inventory                 :: Vector{Inventories.InventoryLine},
      data                      :: Array{Float32,2},
      vector_wind_layers        :: Vector{String},
      layer_blocks_to_make      :: Vector{Int64}, # List of indices. See top of this file.
      twenty_five_mi_mean_is    :: Vector{Vector{Int64}}, # If not using, pass an empty vector.
      unique_fifty_mi_mean_is   :: Vector{Vector{Int64}}, # If not using, pass an empty vector.
      unique_hundred_mi_mean_is :: Vector{Vector{Int64}};  # If not using, pass an empty vector.
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

  out[:,1:raw_feature_count] = data


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
    out[:, raw_feature_count + new_pre_feature_i] = compute_new_feature_pre(grid, get_layer)
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

  # forecast_hour_layer_i = feature_range(block_i, pre_feature_count).start
  #
  # feature_interaction_terms_range = forecast_hour_layer_i+1:forecast_hour_layer_i+feature_interaction_terms_count




  should_make_twenty_five_mi_mean_block = twenty_five_mi_mean_block in layer_blocks_to_make
  should_make_fifty_mi_mean_block       = fifty_mi_mean_block       in layer_blocks_to_make
  should_make_hundred_mi_mean_block     = hundred_mi_mean_block     in layer_blocks_to_make

  if should_make_twenty_five_mi_mean_block || should_make_fifty_mi_mean_block || should_make_hundred_mi_mean_block
    make_mean_layers(
        out, pre_feature_count, grid_point_count,
        should_make_twenty_five_mi_mean_block, twenty_five_mi_mean_features_range, twenty_five_mi_mean_is,
        should_make_fifty_mi_mean_block,       fifty_mi_mean_features_range,       unique_fifty_mi_mean_is,
        should_make_hundred_mi_mean_block,     hundred_mi_mean_features_range,     unique_hundred_mi_mean_is
      )
  end



  # Make 0-5500m(ish) mean wind.

  mean_wind_lower_half_atmosphere_us = Array{Float32}(undef, grid_point_count)
  mean_wind_lower_half_atmosphere_vs = Array{Float32}(undef, grid_point_count)

  total_weight = 0.0f0

  # Not density weighted. Not even reasonably weighted lol.

  if "GRD:10 m above ground:hour fcst:wt ens mean" in vector_wind_layers
    # SREF
    mean_wind_lower_half_atmosphere_us .= get_layer("UGRD:10 m above ground:hour fcst:wt ens mean")
    mean_wind_lower_half_atmosphere_vs .= get_layer("VGRD:10 m above ground:hour fcst:wt ens mean")
    total_weight += 1.0f0
  elseif "GRD:10 m above ground:hour fcst:" in vector_wind_layers
    # RAP, HRRR
    mean_wind_lower_half_atmosphere_us .= get_layer("UGRD:10 m above ground:hour fcst:")
    mean_wind_lower_half_atmosphere_vs .= get_layer("VGRD:10 m above ground:hour fcst:")
    total_weight += 1.0f0
  else
    # HREF
    fill!(mean_wind_lower_half_atmosphere_us, 0.0f0)
    fill!(mean_wind_lower_half_atmosphere_vs, 0.0f0)
  end

  for mb in 950:-25:500
    if "GRD:$mb mb:hour fcst:wt ens mean" in vector_wind_layers
      # HREF/SREF
      mean_wind_lower_half_atmosphere_us .+= get_layer("UGRD:$mb mb:hour fcst:wt ens mean")
      mean_wind_lower_half_atmosphere_vs .+= get_layer("VGRD:$mb mb:hour fcst:wt ens mean")
      total_weight += 1.0f0
    elseif "GRD:$mb mb:hour fcst:" in vector_wind_layers
      # RAP, HRRR
      mean_wind_lower_half_atmosphere_us .+= get_layer("UGRD:$mb mb:hour fcst:")
      mean_wind_lower_half_atmosphere_vs .+= get_layer("VGRD:$mb mb:hour fcst:")
      total_weight += 1.0f0
    elseif mb == 600
      # HRRR/HREF don't have 600mb.
      # And the HRRR/HREF levels (surface, 925mb, 850mb, 700mb, 500mb) are likely too biased towards the lower atmosphere.
      # Estimate 600mb winds into the mix.
      if "GRD:700 mb:hour fcst:wt ens mean" in vector_wind_layers
        # HREF
        mean_wind_lower_half_atmosphere_us .+= 0.5f0 .* get_layer("UGRD:700 mb:hour fcst:wt ens mean")
        mean_wind_lower_half_atmosphere_vs .+= 0.5f0 .* get_layer("VGRD:700 mb:hour fcst:wt ens mean")
        mean_wind_lower_half_atmosphere_us .+= 0.5f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean")
        mean_wind_lower_half_atmosphere_vs .+= 0.5f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean")
        total_weight += 1.0f0
      else
        # HRRR
        mean_wind_lower_half_atmosphere_us .+= 0.5f0 .* get_layer("UGRD:700 mb:hour fcst:")
        mean_wind_lower_half_atmosphere_vs .+= 0.5f0 .* get_layer("VGRD:700 mb:hour fcst:")
        mean_wind_lower_half_atmosphere_us .+= 0.5f0 .* get_layer("UGRD:500 mb:hour fcst:")
        mean_wind_lower_half_atmosphere_vs .+= 0.5f0 .* get_layer("VGRD:500 mb:hour fcst:")
        total_weight += 1.0f0
      end
    end
  end

  mean_wind_lower_half_atmosphere_us /= total_weight
  mean_wind_lower_half_atmosphere_vs /= total_weight



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

  compute_directional_is(
      height, width,
      grid.point_heights_miles, grid.point_widths_miles,
      mean_wind_lower_half_atmosphere_us, mean_wind_lower_half_atmosphere_vs,
      twenty_five_mi_forward_is, twenty_five_mi_backward_is, twenty_five_mi_leftward_is, twenty_five_mi_rightward_is,
      fifty_mi_forward_is,       fifty_mi_backward_is,       fifty_mi_leftward_is,       fifty_mi_rightward_is,
      hundred_mi_forward_is,     hundred_mi_backward_is,     hundred_mi_leftward_is,     hundred_mi_rightward_is
    )

  if should_make_twenty_five_mi_mean_block
    Threads.@threads for pre_layer_feature_i in 1:pre_feature_count
    # for pre_layer_feature_i in 1:pre_feature_count
      twenty_five_mi_mean_feature_i = pre_layer_feature_i + twenty_five_mi_mean_features_range.start - 1

      twenty_five_mi_forward_vals   = @view out[twenty_five_mi_forward_is,   twenty_five_mi_mean_feature_i]
      twenty_five_mi_backward_vals  = @view out[twenty_five_mi_backward_is,  twenty_five_mi_mean_feature_i]
      twenty_five_mi_leftward_vals  = @view out[twenty_five_mi_leftward_is,  twenty_five_mi_mean_feature_i]
      twenty_five_mi_rightward_vals = @view out[twenty_five_mi_rightward_is, twenty_five_mi_mean_feature_i]

      if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
        twenty_five_mi_forward_gradient_i         = pre_layer_feature_i + twenty_five_mi_forward_gradient_range.start - 1
        out[:, twenty_five_mi_forward_gradient_i] = twenty_five_mi_forward_vals .- twenty_five_mi_backward_vals
      end

      if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
        twenty_five_mi_leftward_gradient_i         = pre_layer_feature_i + twenty_five_mi_leftward_gradient_range.start - 1
        out[:, twenty_five_mi_leftward_gradient_i] = twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
      end

      if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
        twenty_five_mi_linestraddling_gradient_i         = pre_layer_feature_i + twenty_five_mi_linestraddling_gradient_range.start - 1
        out[:, twenty_five_mi_linestraddling_gradient_i] = twenty_five_mi_forward_vals .+ twenty_five_mi_backward_vals .- twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
      end
    end
  end

  if should_make_fifty_mi_mean_block
    Threads.@threads for pre_layer_feature_i in 1:pre_feature_count
    # for pre_layer_feature_i in 1:pre_feature_count
      fifty_mi_mean_feature_i = pre_layer_feature_i + fifty_mi_mean_features_range.start - 1

      fifty_mi_forward_vals   = @view out[fifty_mi_forward_is,   fifty_mi_mean_feature_i]
      fifty_mi_backward_vals  = @view out[fifty_mi_backward_is,  fifty_mi_mean_feature_i]
      fifty_mi_leftward_vals  = @view out[fifty_mi_leftward_is,  fifty_mi_mean_feature_i]
      fifty_mi_rightward_vals = @view out[fifty_mi_rightward_is, fifty_mi_mean_feature_i]

      if fifty_mi_forward_gradient_block in layer_blocks_to_make
        fifty_mi_forward_gradient_i         = pre_layer_feature_i + fifty_mi_forward_gradient_range.start - 1
        out[:, fifty_mi_forward_gradient_i] = fifty_mi_forward_vals .- fifty_mi_backward_vals
      end

      if fifty_mi_leftward_gradient_block in layer_blocks_to_make
        fifty_mi_leftward_gradient_i         = pre_layer_feature_i + fifty_mi_leftward_gradient_range.start - 1
        out[:, fifty_mi_leftward_gradient_i] = fifty_mi_leftward_vals .- fifty_mi_rightward_vals
      end

      if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
        fifty_mi_linestraddling_gradient_i         = pre_layer_feature_i + fifty_mi_linestraddling_gradient_range.start - 1
        out[:, fifty_mi_linestraddling_gradient_i] = fifty_mi_forward_vals .+ fifty_mi_backward_vals .- fifty_mi_leftward_vals .- fifty_mi_rightward_vals
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
        out[:, hundred_mi_forward_gradient_i] = hundred_mi_forward_vals  .- hundred_mi_backward_vals
      end

      if hundred_mi_leftward_gradient_block in layer_blocks_to_make
        hundred_mi_leftward_gradient_i         = pre_layer_feature_i + hundred_mi_leftward_gradient_range.start - 1
        out[:, hundred_mi_leftward_gradient_i] = hundred_mi_leftward_vals .- hundred_mi_rightward_vals
      end

      if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
        hundred_mi_linestraddling_gradient_i         = pre_layer_feature_i + hundred_mi_linestraddling_gradient_range.start - 1
        out[:, hundred_mi_linestraddling_gradient_i] = hundred_mi_forward_vals  .+ hundred_mi_backward_vals .- hundred_mi_leftward_vals .- hundred_mi_rightward_vals
      end
    end
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



  # Make ~500m - ~5000m shear vector relative to which we will rotate the winds


  if "GRD:10 m above ground:hour fcst:wt ens mean" in vector_wind_layers
    # SREF
    mean_wind_lower_atmosphere_us  = 0.5f0 .* (get_layer("UGRD:10 m above ground:hour fcst:wt ens mean") .+ get_layer("UGRD:850 mb:hour fcst:wt ens mean"))
    mean_wind_lower_atmosphere_vs  = 0.5f0 .* (get_layer("VGRD:10 m above ground:hour fcst:wt ens mean") .+ get_layer("VGRD:850 mb:hour fcst:wt ens mean"))
    mean_wind_middle_atmosphere_us = 0.5f0 .* (get_layer("UGRD:600 mb:hour fcst:wt ens mean")            .+ get_layer("UGRD:500 mb:hour fcst:wt ens mean"))
    mean_wind_middle_atmosphere_vs = 0.5f0 .* (get_layer("VGRD:600 mb:hour fcst:wt ens mean")            .+ get_layer("VGRD:500 mb:hour fcst:wt ens mean"))
  elseif "GRD:925 mb:hour fcst:wt ens mean" in vector_wind_layers
    # HREF
    mean_wind_lower_atmosphere_us  = 0.75f0 .* get_layer("UGRD:925 mb:hour fcst:wt ens mean")  .+  0.25f0 .* get_layer("UGRD:850 mb:hour fcst:wt ens mean")
    mean_wind_lower_atmosphere_vs  = 0.75f0 .* get_layer("VGRD:925 mb:hour fcst:wt ens mean")  .+  0.25f0 .* get_layer("VGRD:850 mb:hour fcst:wt ens mean")
    mean_wind_middle_atmosphere_us = 0.25f0 .* get_layer("UGRD:700 mb:hour fcst:wt ens mean")  .+  0.75f0 .* get_layer("UGRD:500 mb:hour fcst:wt ens mean")
    mean_wind_middle_atmosphere_vs = 0.25f0 .* get_layer("VGRD:700 mb:hour fcst:wt ens mean")  .+  0.75f0 .* get_layer("VGRD:500 mb:hour fcst:wt ens mean")
  elseif "GRD:950 mb:hour fcst:" in vector_wind_layers
    # RAP
    mean_wind_lower_atmosphere_us  = 0.5f0 .* (get_layer("UGRD:80 m above ground:hour fcst:") .+ get_layer("UGRD:950 mb:hour fcst:"))
    mean_wind_lower_atmosphere_vs  = 0.5f0 .* (get_layer("VGRD:80 m above ground:hour fcst:") .+ get_layer("VGRD:950 mb:hour fcst:"))
    mean_wind_middle_atmosphere_us = 0.5f0 .* (get_layer("UGRD:600 mb:hour fcst:")            .+ get_layer("UGRD:500 mb:hour fcst:"))
    mean_wind_middle_atmosphere_vs = 0.5f0 .* (get_layer("VGRD:600 mb:hour fcst:")            .+ get_layer("VGRD:500 mb:hour fcst:"))
  else
    # HRRR
    mean_wind_lower_atmosphere_us  = 0.5f0 .* (get_layer("UGRD:80 m above ground:hour fcst:") .+ get_layer("UGRD:925 mb:hour fcst:"))
    mean_wind_lower_atmosphere_vs  = 0.5f0 .* (get_layer("VGRD:80 m above ground:hour fcst:") .+ get_layer("VGRD:925 mb:hour fcst:"))
    mean_wind_middle_atmosphere_us = 0.25f0 .* get_layer("UGRD:700 mb:hour fcst:")  .+  0.75f0 .* get_layer("UGRD:500 mb:hour fcst:")
    mean_wind_middle_atmosphere_vs = 0.25f0 .* get_layer("VGRD:700 mb:hour fcst:")  .+  0.75f0 .* get_layer("VGRD:500 mb:hour fcst:")
  end

  mean_wind_angles = atan.(mean_wind_middle_atmosphere_vs .- mean_wind_lower_atmosphere_vs, mean_wind_middle_atmosphere_us .- mean_wind_lower_atmosphere_us)

  if any(isnan, mean_wind_angles)
    error("nan wind angle")
  end

  rot_coses = cos.(-mean_wind_angles)
  rot_sines = sin.(-mean_wind_angles)


  # Center wind vectors around 0-6km(ish) mean wind
  # Rotate winds to align to the ~500m - ~5500m shear vector (inspired by Bunker's storm motion, which we are not calculating yet)

  # Save some allocations by using a pre-existing scratch_us and scratch_vs
  rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, u_i, v_i, should_center) = begin
    if should_center
      scratch_us .= out[:, u_i] .- mean_wind_lower_half_atmosphere_us
      scratch_vs .= out[:, v_i] .- mean_wind_lower_half_atmosphere_vs
    else
      scratch_us .= out[:, u_i] # Don't use @view because u and v are each used in the calculation of the other
      scratch_vs .= out[:, v_i]
    end

    out[:, u_i] = (scratch_us .* rot_coses) .- (scratch_vs .* rot_sines)
    out[:, v_i] = (scratch_us .* rot_sines) .+ (scratch_vs .* rot_coses)
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

    # Center the non-gradient layers (gradients are relativized already)
    # And rotate everrrryything.

    raw_layer_u_i = feature_key_to_i[layer_key_u]
    raw_layer_v_i = feature_key_to_i[layer_key_v]

    scratch_us = Array{Float32}(undef, grid_point_count)
    scratch_vs = Array{Float32}(undef, grid_point_count)

    rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, raw_layer_u_i, raw_layer_v_i, true)

    if twenty_five_mi_mean_block in layer_blocks_to_make
      twenty_five_mi_mean_u_i = raw_layer_u_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_v_i = raw_layer_v_i + twenty_five_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, twenty_five_mi_mean_u_i, twenty_five_mi_mean_v_i, true)
    end

    if fifty_mi_mean_block in layer_blocks_to_make
      fifty_mi_mean_u_i = raw_layer_u_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_v_i = raw_layer_v_i + fifty_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, fifty_mi_mean_u_i, fifty_mi_mean_v_i, true)
    end

    if hundred_mi_mean_block in layer_blocks_to_make
      hundred_mi_mean_u_i = raw_layer_u_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_v_i = raw_layer_v_i + hundred_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, hundred_mi_mean_u_i, hundred_mi_mean_v_i, true)
    end

    if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
      twenty_five_mi_forward_gradient_u_i = raw_layer_u_i + twenty_five_mi_forward_gradient_range.start - 1
      twenty_five_mi_forward_gradient_v_i = raw_layer_v_i + twenty_five_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, twenty_five_mi_forward_gradient_u_i, twenty_five_mi_forward_gradient_v_i, false)
    end

    if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
      twenty_five_mi_leftward_gradient_u_i = raw_layer_u_i + twenty_five_mi_leftward_gradient_range.start - 1
      twenty_five_mi_leftward_gradient_v_i = raw_layer_v_i + twenty_five_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, twenty_five_mi_leftward_gradient_u_i, twenty_five_mi_leftward_gradient_v_i, false)
    end

    if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
      twenty_five_mi_linestraddling_gradient_u_i = raw_layer_u_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      twenty_five_mi_linestraddling_gradient_v_i = raw_layer_v_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, twenty_five_mi_linestraddling_gradient_u_i, twenty_five_mi_linestraddling_gradient_v_i, false)
    end

    if fifty_mi_forward_gradient_block in layer_blocks_to_make
      fifty_mi_forward_gradient_u_i = raw_layer_u_i + fifty_mi_forward_gradient_range.start - 1
      fifty_mi_forward_gradient_v_i = raw_layer_v_i + fifty_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, fifty_mi_forward_gradient_u_i, fifty_mi_forward_gradient_v_i, false)
    end

    if fifty_mi_leftward_gradient_block in layer_blocks_to_make
      fifty_mi_leftward_gradient_u_i = raw_layer_u_i + fifty_mi_leftward_gradient_range.start - 1
      fifty_mi_leftward_gradient_v_i = raw_layer_v_i + fifty_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, fifty_mi_leftward_gradient_u_i, fifty_mi_leftward_gradient_v_i, false)
    end

    if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
      fifty_mi_linestraddling_gradient_u_i = raw_layer_u_i + fifty_mi_linestraddling_gradient_range.start - 1
      fifty_mi_linestraddling_gradient_v_i = raw_layer_v_i + fifty_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, fifty_mi_linestraddling_gradient_u_i, fifty_mi_linestraddling_gradient_v_i, false)
    end

    if hundred_mi_forward_gradient_block in layer_blocks_to_make
      hundred_mi_forward_gradient_u_i = raw_layer_u_i + hundred_mi_forward_gradient_range.start - 1
      hundred_mi_forward_gradient_v_i = raw_layer_v_i + hundred_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, hundred_mi_forward_gradient_u_i, hundred_mi_forward_gradient_v_i, false)
    end

    if hundred_mi_leftward_gradient_block in layer_blocks_to_make
      hundred_mi_leftward_gradient_u_i      = raw_layer_u_i + hundred_mi_leftward_gradient_range.start - 1
      hundred_mi_leftward_gradient_v_i      = raw_layer_v_i + hundred_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, hundred_mi_leftward_gradient_u_i, hundred_mi_leftward_gradient_v_i, false)
    end

    if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
      hundred_mi_linestraddling_gradient_u_i = raw_layer_u_i + hundred_mi_linestraddling_gradient_range.start - 1
      hundred_mi_linestraddling_gradient_v_i = raw_layer_v_i + hundred_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(scratch_us, scratch_vs, hundred_mi_linestraddling_gradient_u_i, hundred_mi_linestraddling_gradient_v_i, false)
    end
  end



  # Extra layer: the forecast hour.
  #
  # Only 10 hour resolution. Don't want to overfit.
  # out[:, forecast_hour_layer_i] = repeat([Float32(div(forecast_hour, 10))], grid_point_count)

  # # Now any interaction terms (multiplication of previously computed terms; computed in order so an interaction term could use a prior interaction term.)
  # for interaction_terms_i in 1:length(feature_interaction_terms)
  #   interaction_term_is = feature_interaction_terms[interaction_terms_i]
  #   layer_i             = feature_interaction_terms_range.start + interaction_terms_i - 1
  #
  #   out[:, layer_i] = @view reduce(*, (@view out[:, interaction_term_is]), dims = 2)[:,1]
  # end

  out
end

end # module FeatureEngineeringShared