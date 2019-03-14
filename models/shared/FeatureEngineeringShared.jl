module FeatureEngineeringShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Grids
import Inventories


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

leftover_fields = [
  "div(forecast hour, 10)",
]

function feature_i_to_name(inventory :: Vector{Inventories.InventoryLine}, layer_blocks_to_make :: Vector{Int64}, feature_i :: Int64; feature_interaction_terms = []) :: String
  raw_feature_count = length(inventory)

  layer_blocks_to_make = sort(layer_blocks_to_make) # Layers always produced in a particular order regardless of given order. It's a set.

  output_block_i, feature_i_in_block = divrem(feature_i - 1, raw_feature_count)

  output_block_i += 1
  feature_i_in_block += 1

  if output_block_i >= 1 && output_block_i <= length(layer_blocks_to_make)
    block_name = feature_block_names[layer_blocks_to_make[output_block_i]]
    return Inventories.inventory_line_key(inventory[feature_i_in_block]) * ":" * block_name
  elseif output_block_i > length(layer_blocks_to_make)
    leftover_feature_i = feature_i - length(layer_blocks_to_make)*raw_feature_count

    if leftover_feature_i <= length(leftover_fields)
      return join([leftover_fields[leftover_feature_i], "calculated", "hour fcst", "calculated", ""], ":")
    end

    interaction_feature_i = leftover_feature_i - length(leftover_fields)

    if interaction_feature_i <= length(feature_interaction_terms)
      names_of_interacting_features =
        map(feature_interaction_terms[interaction_feature_i]) do feature_i
          feature_i_to_name(inventory, layer_blocks_to_make, feature_i, feature_interaction_terms = feature_interaction_terms)
        end

      return join(names_of_interacting_features, "*")
    end
  end

  "Unknown feature $feature_i"
end

function feature_range(block_i, raw_feature_count)
  (block_i-1)*raw_feature_count+1:block_i*raw_feature_count
end

function make_data(
      grid                      :: Grids.Grid,
      forecast                  :: Forecasts.Forecast,
      data                      :: Array{Float32,2},
      vector_wind_layers        :: Vector{String},
      layer_blocks_to_make      :: Vector{Int64}, # List of indices. See top of this file.
      twenty_five_mi_mean_is    :: Vector{Vector{Int64}}, # If not using, pass an empty vector.
      unique_fifty_mi_mean_is   :: Vector{Vector{Int64}}, # If not using, pass an empty vector.
      unique_hundred_mi_mean_is :: Vector{Vector{Int64}};  # If not using, pass an empty vector.
      feature_interaction_terms = []
    ) :: Array{Float32,2}
  inventory = Forecasts.inventory(forecast)

  feature_keys = map(Inventories.inventory_line_key, inventory)

  feature_key_to_i = Dict{String,Int64}()

  grid_point_count  = size(data,1)
  raw_feature_count = size(data,2)
  height            = grid.height
  width             = grid.width

  for i in 1:length(inventory)
    feature_key_to_i[feature_keys[i]] = i
  end

  get_layer(key) = @view data[:, feature_key_to_i[key]]

  # Output for each feature:
  # raw data (but with relativized winds)
  # 25mi mean
  # 50mi mean
  # 100mi mean
  # 25mi forward grad, 25mi leftward grad, 25mi linestraddling grad
  # 50mi forward grad, 50mi leftward grad, 50mi linestraddling grad
  # 100mi forward grad, 100mi leftward grad, 100mi linestraddling grad
  #
  # And then a lone forecast hour field
  feature_interaction_terms_count = length(feature_interaction_terms)

  out = Array{Float32}(undef, (grid_point_count, length(layer_blocks_to_make)*raw_feature_count + 1 + feature_interaction_terms_count))

  out[:,1:raw_feature_count] = data

  block_i = 2

  if twenty_five_mi_mean_block in layer_blocks_to_make
    twenty_five_mi_mean_features_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if fifty_mi_mean_block in layer_blocks_to_make
    fifty_mi_mean_features_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if hundred_mi_mean_block in layer_blocks_to_make
    hundred_mi_mean_features_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
    twenty_five_mi_forward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
    twenty_five_mi_leftward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
    twenty_five_mi_linestraddling_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if fifty_mi_forward_gradient_block in layer_blocks_to_make
    fifty_mi_forward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if fifty_mi_leftward_gradient_block in layer_blocks_to_make
    fifty_mi_leftward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
    fifty_mi_linestraddling_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if hundred_mi_forward_gradient_block in layer_blocks_to_make
    hundred_mi_forward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if hundred_mi_leftward_gradient_block in layer_blocks_to_make
    hundred_mi_leftward_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
    hundred_mi_linestraddling_gradient_range = feature_range(block_i, raw_feature_count)
    block_i += 1
  end

  forecast_hour_layer_i = feature_range(block_i, raw_feature_count).start

  feature_interaction_terms_range = forecast_hour_layer_i+1:forecast_hour_layer_i+feature_interaction_terms_count



  # Make 25mi, 50mi, and 100mi mean layers

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

  should_make_twenty_five_mi_mean_block = twenty_five_mi_mean_block in layer_blocks_to_make
  should_make_fifty_mi_mean_block       = fifty_mi_mean_block       in layer_blocks_to_make
  should_make_hundred_mi_mean_block    = hundred_mi_mean_block     in layer_blocks_to_make


  if should_make_twenty_five_mi_mean_block || should_make_fifty_mi_mean_block || should_make_hundred_mi_mean_block

    Threads.@threads for raw_layer_feature1_i in 1:4:raw_feature_count
      raw_layer_feature2_i = min(raw_layer_feature1_i + 1, raw_feature_count)
      raw_layer_feature3_i = min(raw_layer_feature1_i + 2, raw_feature_count)
      raw_layer_feature4_i = min(raw_layer_feature1_i + 3, raw_feature_count)
      # raw_layer_feature5_i = min(raw_layer_feature1_i + 4, raw_feature_count)
      # raw_layer_feature6_i = min(raw_layer_feature1_i + 5, raw_feature_count)
      # raw_layer_feature7_i = min(raw_layer_feature1_i + 6, raw_feature_count)
      # raw_layer_feature8_i = min(raw_layer_feature1_i + 7, raw_feature_count)

      if should_make_twenty_five_mi_mean_block
        twenty_five_mi_mean_feature1_i   = raw_layer_feature1_i + twenty_five_mi_mean_features_range.start - 1
        twenty_five_mi_mean_feature2_i   = raw_layer_feature2_i + twenty_five_mi_mean_features_range.start - 1
        twenty_five_mi_mean_feature3_i   = raw_layer_feature3_i + twenty_five_mi_mean_features_range.start - 1
        twenty_five_mi_mean_feature4_i   = raw_layer_feature4_i + twenty_five_mi_mean_features_range.start - 1
        # twenty_five_mi_mean_feature5_i   = raw_layer_feature5_i + twenty_five_mi_mean_features_range.start - 1
        # twenty_five_mi_mean_feature6_i   = raw_layer_feature6_i + twenty_five_mi_mean_features_range.start - 1
        # twenty_five_mi_mean_feature7_i   = raw_layer_feature7_i + twenty_five_mi_mean_features_range.start - 1
        # twenty_five_mi_mean_feature8_i   = raw_layer_feature8_i + twenty_five_mi_mean_features_range.start - 1
      end

      if should_make_fifty_mi_mean_block
        fifty_mi_mean_feature1_i   = raw_layer_feature1_i + fifty_mi_mean_features_range.start - 1
        fifty_mi_mean_feature2_i   = raw_layer_feature2_i + fifty_mi_mean_features_range.start - 1
        fifty_mi_mean_feature3_i   = raw_layer_feature3_i + fifty_mi_mean_features_range.start - 1
        fifty_mi_mean_feature4_i   = raw_layer_feature4_i + fifty_mi_mean_features_range.start - 1
        # fifty_mi_mean_feature5_i   = raw_layer_feature5_i + fifty_mi_mean_features_range.start - 1
        # fifty_mi_mean_feature6_i   = raw_layer_feature6_i + fifty_mi_mean_features_range.start - 1
        # fifty_mi_mean_feature7_i   = raw_layer_feature7_i + fifty_mi_mean_features_range.start - 1
        # fifty_mi_mean_feature8_i   = raw_layer_feature8_i + fifty_mi_mean_features_range.start - 1
      end

      if should_make_hundred_mi_mean_block
        hundred_mi_mean_feature1_i = raw_layer_feature1_i + hundred_mi_mean_features_range.start - 1
        hundred_mi_mean_feature2_i = raw_layer_feature2_i + hundred_mi_mean_features_range.start - 1
        hundred_mi_mean_feature3_i = raw_layer_feature3_i + hundred_mi_mean_features_range.start - 1
        hundred_mi_mean_feature4_i = raw_layer_feature4_i + hundred_mi_mean_features_range.start - 1
        # hundred_mi_mean_feature5_i = raw_layer_feature5_i + hundred_mi_mean_features_range.start - 1
        # hundred_mi_mean_feature6_i = raw_layer_feature6_i + hundred_mi_mean_features_range.start - 1
        # hundred_mi_mean_feature7_i = raw_layer_feature7_i + hundred_mi_mean_features_range.start - 1
        # hundred_mi_mean_feature8_i = raw_layer_feature8_i + hundred_mi_mean_features_range.start - 1
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
        for near_i in twenty_five_mi_mean_is[flat_i]
          total1 += data[near_i, raw_layer_feature1_i]
          total2 += data[near_i, raw_layer_feature2_i]
          total3 += data[near_i, raw_layer_feature3_i]
          total4 += data[near_i, raw_layer_feature4_i]
          # total5 += data[near_i, raw_layer_feature5_i]
          # total6 += data[near_i, raw_layer_feature6_i]
          # total7 += data[near_i, raw_layer_feature7_i]
          # total8 += data[near_i, raw_layer_feature8_i]
          n      += 1.0f0
        end
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
          for near_i in unique_fifty_mi_mean_is[flat_i]
            total1 += data[near_i, raw_layer_feature1_i]
            total2 += data[near_i, raw_layer_feature2_i]
            total3 += data[near_i, raw_layer_feature3_i]
            total4 += data[near_i, raw_layer_feature4_i]
            # total5 += data[near_i, raw_layer_feature5_i]
            # total6 += data[near_i, raw_layer_feature6_i]
            # total7 += data[near_i, raw_layer_feature7_i]
            # total8 += data[near_i, raw_layer_feature8_i]
            n      += 1.0f0
          end
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
          for near_i in unique_hundred_mi_mean_is[flat_i]
            total1 += data[near_i, raw_layer_feature1_i]
            total2 += data[near_i, raw_layer_feature2_i]
            total3 += data[near_i, raw_layer_feature3_i]
            total4 += data[near_i, raw_layer_feature4_i]
            # total5 += data[near_i, raw_layer_feature5_i]
            # total6 += data[near_i, raw_layer_feature6_i]
            # total7 += data[near_i, raw_layer_feature7_i]
            # total8 += data[near_i, raw_layer_feature8_i]
            n       += 1.0f0
          end
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

  end



  # Make 0-5500km(ish) mean wind.

  mean_wind_lower_half_atmosphere_us = Array{Float32}(undef, grid_point_count)
  mean_wind_lower_half_atmosphere_vs = Array{Float32}(undef, grid_point_count)

  total_weight = 0.0f0

  # Not density weighted. Not even reasonably weighted lol.

  if "GRD:10 m above ground:hour fcst:wt ens mean" in vector_wind_layers
    # SREF
    mean_wind_lower_half_atmosphere_us .= get_layer("UGRD:10 m above ground:hour fcst:wt ens mean")
    mean_wind_lower_half_atmosphere_vs .= get_layer("VGRD:10 m above ground:hour fcst:wt ens mean")
    total_weight += 1.0f0
  elseif "GRD:925 mb:hour fcst:wt ens mean" in vector_wind_layers
    # HREF
    mean_wind_lower_half_atmosphere_us .= get_layer("UGRD:925 mb:hour fcst:wt ens mean")
    mean_wind_lower_half_atmosphere_vs .= get_layer("VGRD:925 mb:hour fcst:wt ens mean")
    total_weight += 1.0f0
  elseif "GRD:10 m above ground:hour fcst:" in vector_wind_layers
    # RAP
    mean_wind_lower_half_atmosphere_us .= get_layer("UGRD:10 m above ground:hour fcst:")
    mean_wind_lower_half_atmosphere_vs .= get_layer("VGRD:10 m above ground:hour fcst:")
    total_weight += 1.0f0
  end

  for mb in 950:-50:500
    if "GRD:$mb mb:hour fcst:wt ens mean" in vector_wind_layers
      # HREF/SREF
      mean_wind_lower_half_atmosphere_us .+= get_layer("UGRD:$mb mb:hour fcst:wt ens mean")
      mean_wind_lower_half_atmosphere_vs .+= get_layer("VGRD:$mb mb:hour fcst:wt ens mean")
      total_weight += 1.0f0
    elseif "GRD:$mb mb:hour fcst:" in vector_wind_layers
      # RAP
      mean_wind_lower_half_atmosphere_us .+= get_layer("UGRD:$mb mb:hour fcst:")
      mean_wind_lower_half_atmosphere_vs .+= get_layer("VGRD:$mb mb:hour fcst:")
      total_weight += 1.0f0
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

  uv_normalize(u, v) = begin
    ε = 1f-8
    len = √(u^2 + v^2) + ε
    (u / len, v / len)
  end

  is_on_grid(w_to_e_col, s_to_n_row) = begin
    w_to_e_col >= 1 && w_to_e_col <= width &&
    s_to_n_row >= 1 && s_to_n_row <= height
  end

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

  for j in 1:height
    for i in 1:width
      flat_i = width*(j-1) + i

      relative_u, relative_v = uv_normalize(mean_wind_lower_half_atmosphere_us[flat_i], mean_wind_lower_half_atmosphere_vs[flat_i])

      point_height = Float32(grid.point_heights_miles[flat_i])
      point_width  = Float32(grid.point_widths_miles[flat_i]) # On the SREF grid, point_width and point_height are always nearly equal, <1.0% difference.

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
      hundred_mi_forward_is[flat_i]      = width*(forward_100mi_j-1)   + forward_100mi_i
      hundred_mi_backward_is[flat_i]     = width*(backward_100mi_j-1)  + backward_100mi_i
      hundred_mi_leftward_is[flat_i]     = width*(leftward_100mi_j-1)  + leftward_100mi_i
      hundred_mi_rightward_is[flat_i]    = width*(rightward_100mi_j-1) + rightward_100mi_i
    end
  end

  if should_make_twenty_five_mi_mean_block
    Threads.@threads for raw_layer_feature_i in 1:raw_feature_count
      twenty_five_mi_mean_feature_i = raw_layer_feature_i + twenty_five_mi_mean_features_range.start - 1

      twenty_five_mi_forward_vals   = @view out[twenty_five_mi_forward_is,   twenty_five_mi_mean_feature_i]
      twenty_five_mi_backward_vals  = @view out[twenty_five_mi_backward_is,  twenty_five_mi_mean_feature_i]
      twenty_five_mi_leftward_vals  = @view out[twenty_five_mi_leftward_is,  twenty_five_mi_mean_feature_i]
      twenty_five_mi_rightward_vals = @view out[twenty_five_mi_rightward_is, twenty_five_mi_mean_feature_i]

      if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
        twenty_five_mi_forward_gradient_i         = raw_layer_feature_i + twenty_five_mi_forward_gradient_range.start - 1
        out[:, twenty_five_mi_forward_gradient_i] = twenty_five_mi_forward_vals .- twenty_five_mi_backward_vals
      end

      if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
        twenty_five_mi_leftward_gradient_i         = raw_layer_feature_i + twenty_five_mi_leftward_gradient_range.start - 1
        out[:, twenty_five_mi_leftward_gradient_i] = twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
      end

      if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
        twenty_five_mi_linestraddling_gradient_i         = raw_layer_feature_i + twenty_five_mi_linestraddling_gradient_range.start - 1
        out[:, twenty_five_mi_linestraddling_gradient_i] = twenty_five_mi_forward_vals .+ twenty_five_mi_backward_vals .- twenty_five_mi_leftward_vals .- twenty_five_mi_rightward_vals
      end
    end
  end

  if should_make_fifty_mi_mean_block
    Threads.@threads for raw_layer_feature_i in 1:raw_feature_count
      fifty_mi_mean_feature_i = raw_layer_feature_i + fifty_mi_mean_features_range.start - 1

      fifty_mi_forward_vals   = @view out[fifty_mi_forward_is,   fifty_mi_mean_feature_i]
      fifty_mi_backward_vals  = @view out[fifty_mi_backward_is,  fifty_mi_mean_feature_i]
      fifty_mi_leftward_vals  = @view out[fifty_mi_leftward_is,  fifty_mi_mean_feature_i]
      fifty_mi_rightward_vals = @view out[fifty_mi_rightward_is, fifty_mi_mean_feature_i]

      if fifty_mi_forward_gradient_block in layer_blocks_to_make
        fifty_mi_forward_gradient_i         = raw_layer_feature_i + fifty_mi_forward_gradient_range.start - 1
        out[:, fifty_mi_forward_gradient_i] = fifty_mi_forward_vals .- fifty_mi_backward_vals
      end

      if fifty_mi_leftward_gradient_block in layer_blocks_to_make
        fifty_mi_leftward_gradient_i         = raw_layer_feature_i + fifty_mi_leftward_gradient_range.start - 1
        out[:, fifty_mi_leftward_gradient_i] = fifty_mi_leftward_vals .- fifty_mi_rightward_vals
      end

      if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
        fifty_mi_linestraddling_gradient_i         = raw_layer_feature_i + fifty_mi_linestraddling_gradient_range.start - 1
        out[:, fifty_mi_linestraddling_gradient_i] = fifty_mi_forward_vals .+ fifty_mi_backward_vals .- fifty_mi_leftward_vals .- fifty_mi_rightward_vals
      end
    end
  end

  if should_make_hundred_mi_mean_block
    Threads.@threads for raw_layer_feature_i in 1:raw_feature_count
      hundred_mi_mean_feature_i = raw_layer_feature_i + hundred_mi_mean_features_range.start - 1

      hundred_mi_forward_vals   = @view out[hundred_mi_forward_is,   hundred_mi_mean_feature_i]
      hundred_mi_backward_vals  = @view out[hundred_mi_backward_is,  hundred_mi_mean_feature_i]
      hundred_mi_leftward_vals  = @view out[hundred_mi_leftward_is,  hundred_mi_mean_feature_i]
      hundred_mi_rightward_vals = @view out[hundred_mi_rightward_is, hundred_mi_mean_feature_i]

      if hundred_mi_forward_gradient_block in layer_blocks_to_make
        hundred_mi_forward_gradient_i         = raw_layer_feature_i + hundred_mi_forward_gradient_range.start - 1
        out[:, hundred_mi_forward_gradient_i] = hundred_mi_forward_vals  .- hundred_mi_backward_vals
      end

      if hundred_mi_leftward_gradient_block in layer_blocks_to_make
        hundred_mi_leftward_gradient_i         = raw_layer_feature_i + hundred_mi_leftward_gradient_range.start - 1
        out[:, hundred_mi_leftward_gradient_i] = hundred_mi_leftward_vals .- hundred_mi_rightward_vals
      end

      if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
        hundred_mi_linestraddling_gradient_i         = raw_layer_feature_i + hundred_mi_linestraddling_gradient_range.start - 1
        out[:, hundred_mi_linestraddling_gradient_i] = hundred_mi_forward_vals  .+ hundred_mi_backward_vals .- hundred_mi_leftward_vals .- hundred_mi_rightward_vals
      end
    end
  end



  # Make ~500m - ~500m shear vector relative to which we will rotate the winds


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
  else
    # RAP
    mean_wind_lower_atmosphere_us  = 0.5f0 .* (get_layer("UGRD:80 m above ground:hour fcst:") .+ get_layer("UGRD:950 mb:hour fcst:"))
    mean_wind_lower_atmosphere_vs  = 0.5f0 .* (get_layer("VGRD:80 m above ground:hour fcst:") .+ get_layer("VGRD:950 mb:hour fcst:"))
    mean_wind_middle_atmosphere_us = 0.5f0 .* (get_layer("UGRD:600 mb:hour fcst:")            .+ get_layer("UGRD:500 mb:hour fcst:"))
    mean_wind_middle_atmosphere_vs = 0.5f0 .* (get_layer("VGRD:600 mb:hour fcst:")            .+ get_layer("VGRD:500 mb:hour fcst:"))
  end

  mean_wind_angles = atan.(mean_wind_middle_atmosphere_vs .- mean_wind_lower_atmosphere_vs, mean_wind_middle_atmosphere_us .- mean_wind_lower_atmosphere_us)

  if any(isnan, mean_wind_angles)
    error("nan wind angle")
  end

  rot_coses = cos.(-mean_wind_angles)
  rot_sines = sin.(-mean_wind_angles)


  # Center wind vectors around 0-6km(ish) mean wind
  # Rotate winds to align to the ~500m - ~5500m shear vector (inspired by Bunker's storm motion, which we are not calculating yet)

  rotate_and_perhaps_center_uv_layers(u_i, v_i, should_center) = begin
    if should_center
      us = out[:, u_i] .- mean_wind_lower_half_atmosphere_us
      vs = out[:, v_i] .- mean_wind_lower_half_atmosphere_vs
    else
      us = out[:, u_i] # Don't use @view because u and v are each used in the calculation of the other
      vs = out[:, v_i]
    end

    out[:, u_i] = (us .* rot_coses) .- (vs .* rot_sines)
    out[:, v_i] = (us .* rot_sines) .+ (vs .* rot_coses)
  end

  Threads.@threads for wind_layer_key in vector_wind_layers
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

    rotate_and_perhaps_center_uv_layers(raw_layer_u_i, raw_layer_v_i, true)

    if twenty_five_mi_mean_block in layer_blocks_to_make
      twenty_five_mi_mean_u_i = raw_layer_u_i + twenty_five_mi_mean_features_range.start - 1
      twenty_five_mi_mean_v_i = raw_layer_v_i + twenty_five_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(twenty_five_mi_mean_u_i, twenty_five_mi_mean_v_i, true)
    end

    if fifty_mi_mean_block in layer_blocks_to_make
      fifty_mi_mean_u_i = raw_layer_u_i + fifty_mi_mean_features_range.start - 1
      fifty_mi_mean_v_i = raw_layer_v_i + fifty_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(fifty_mi_mean_u_i, fifty_mi_mean_v_i, true)
    end

    if hundred_mi_mean_block in layer_blocks_to_make
      hundred_mi_mean_u_i = raw_layer_u_i + hundred_mi_mean_features_range.start - 1
      hundred_mi_mean_v_i = raw_layer_v_i + hundred_mi_mean_features_range.start - 1
      rotate_and_perhaps_center_uv_layers(hundred_mi_mean_u_i, hundred_mi_mean_v_i, true)
    end

    if twenty_five_mi_forward_gradient_block in layer_blocks_to_make
      twenty_five_mi_forward_gradient_u_i = raw_layer_u_i + twenty_five_mi_forward_gradient_range.start - 1
      twenty_five_mi_forward_gradient_v_i = raw_layer_v_i + twenty_five_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(twenty_five_mi_forward_gradient_u_i, twenty_five_mi_forward_gradient_v_i, false)
    end

    if twenty_five_mi_leftward_gradient_block in layer_blocks_to_make
      twenty_five_mi_leftward_gradient_u_i = raw_layer_u_i + twenty_five_mi_leftward_gradient_range.start - 1
      twenty_five_mi_leftward_gradient_v_i = raw_layer_v_i + twenty_five_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(twenty_five_mi_leftward_gradient_u_i, twenty_five_mi_leftward_gradient_v_i, false)
    end

    if twenty_five_mi_linestraddling_gradient_block in layer_blocks_to_make
      twenty_five_mi_linestraddling_gradient_u_i = raw_layer_u_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      twenty_five_mi_linestraddling_gradient_v_i = raw_layer_v_i + twenty_five_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(twenty_five_mi_linestraddling_gradient_u_i, twenty_five_mi_linestraddling_gradient_v_i, false)
    end

    if fifty_mi_forward_gradient_block in layer_blocks_to_make
      fifty_mi_forward_gradient_u_i = raw_layer_u_i + fifty_mi_forward_gradient_range.start - 1
      fifty_mi_forward_gradient_v_i = raw_layer_v_i + fifty_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(fifty_mi_forward_gradient_u_i, fifty_mi_forward_gradient_v_i, false)
    end

    if fifty_mi_leftward_gradient_block in layer_blocks_to_make
      fifty_mi_leftward_gradient_u_i = raw_layer_u_i + fifty_mi_leftward_gradient_range.start - 1
      fifty_mi_leftward_gradient_v_i = raw_layer_v_i + fifty_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(fifty_mi_leftward_gradient_u_i, fifty_mi_leftward_gradient_v_i, false)
    end

    if fifty_mi_linestraddling_gradient_block in layer_blocks_to_make
      fifty_mi_linestraddling_gradient_u_i = raw_layer_u_i + fifty_mi_linestraddling_gradient_range.start - 1
      fifty_mi_linestraddling_gradient_v_i = raw_layer_v_i + fifty_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(fifty_mi_linestraddling_gradient_u_i, fifty_mi_linestraddling_gradient_v_i, false)
    end

    if hundred_mi_forward_gradient_block in layer_blocks_to_make
      hundred_mi_forward_gradient_u_i = raw_layer_u_i + hundred_mi_forward_gradient_range.start - 1
      hundred_mi_forward_gradient_v_i = raw_layer_v_i + hundred_mi_forward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(hundred_mi_forward_gradient_u_i, hundred_mi_forward_gradient_v_i, false)
    end

    if hundred_mi_leftward_gradient_block in layer_blocks_to_make
      hundred_mi_leftward_gradient_u_i      = raw_layer_u_i + hundred_mi_leftward_gradient_range.start - 1
      hundred_mi_leftward_gradient_v_i      = raw_layer_v_i + hundred_mi_leftward_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(hundred_mi_leftward_gradient_u_i, hundred_mi_leftward_gradient_v_i, false)
    end

    if hundred_mi_linestraddling_gradient_block in layer_blocks_to_make
      hundred_mi_linestraddling_gradient_u_i = raw_layer_u_i + hundred_mi_linestraddling_gradient_range.start - 1
      hundred_mi_linestraddling_gradient_v_i = raw_layer_v_i + hundred_mi_linestraddling_gradient_range.start - 1
      rotate_and_perhaps_center_uv_layers(hundred_mi_linestraddling_gradient_u_i, hundred_mi_linestraddling_gradient_v_i, false)
    end
  end



  # Extra layer: the forecast hour.
  #
  # Only 10 hour resolution. Don't want to overfit.
  out[:, forecast_hour_layer_i] = repeat([Float32(div(forecast.forecast_hour, 10))], grid_point_count)

  # Now any interaction terms (multiplication of previously computed terms; computed in order so an interaction term could use a prior interaction term.)
  for interaction_terms_i in 1:length(feature_interaction_terms)
    interaction_term_is = feature_interaction_terms[interaction_terms_i]
    layer_i             = feature_interaction_terms_range.start + interaction_terms_i - 1

    out[:, layer_i] = @view reduce(*, (@view out[:, interaction_term_is]), dims = 2)[:,1]
  end

  out
end

end # module FeatureEngineeringShared