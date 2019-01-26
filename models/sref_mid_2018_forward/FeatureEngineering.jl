module FeatureEngineering

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import GeoUtils
import Grids
import Inventories


vector_wind_layers = [
  "GRD:10 m above ground:hour fcst:wt ens mean",
  "GRD:1000 mb:hour fcst:wt ens mean",
  "GRD:850 mb:hour fcst:wt ens mean",
  "GRD:700 mb:hour fcst:wt ens mean",
  "GRD:600 mb:hour fcst:wt ens mean",
  "GRD:600 mb:hour fcst:wt ens mean",
  "GRD:300 mb:hour fcst:wt ens mean",
  "GRD:250 mb:hour fcst:wt ens mean",
]

_fifty_mi_is          = Vector{Int64}[]
_unique_hundred_mi_is = Vector{Int64}[]

# Grid point indicies within 50mi
#
# Grid should only ever be SREF.grid(), since we do cache the result. But don't want to import SREF (that'd be circular).
function fifty_mi_is(grid :: Grids.Grid) :: Vector{Vector{Int64}}
  global _fifty_mi_is

  if isempty(_fifty_mi_is)
    _fifty_mi_is = Grids.radius_grid_is(grid, 50.0)
  end

  _fifty_mi_is
end

# Grid point indicies within 100mi but not within 50mi
#
# Grid should only ever be SREF.grid(), since we do cache the result. But don't want to import SREF (that'd be circular).
function unique_hundred_mi_is(grid :: Grids.Grid) :: Vector{Vector{Int64}}
  global _unique_hundred_mi_is
  # unique_hundred_mi_is =  # *

  if isempty(_unique_hundred_mi_is)
    _unique_hundred_mi_is = Grids.radius_grid_is(grid, 100.0)

    fifty_mi_mean_is = fifty_mi_is(grid)
    for flat_i in 1:grid.height*grid.width
      fifty_mi_indices = fifty_mi_mean_is[flat_i]

      _unique_hundred_mi_is[flat_i] =
        filter(_unique_hundred_mi_is[flat_i]) do i
          !(i in fifty_mi_indices)
        end
    end

    # Re-allocate to ensure cache locality.
    for flat_i in 1:length(_unique_hundred_mi_is)
      _unique_hundred_mi_is[flat_i] = _unique_hundred_mi_is[flat_i][1:length(_unique_hundred_mi_is[flat_i])]
    end
  end

  _unique_hundred_mi_is
end

function mean(xs)
  sum(xs) / length(xs)
end

function is_wind_key(key)
  replace(key, r"\A[UV]" => "") in vector_wind_layers
end

function make_data(grid :: Grids.Grid, forecast :: Forecasts.Forecast, data :: Array{Float32,2}) :: Array{Float32,2}
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

  non_wind_feature_is = filter(feature_i -> !is_wind_key(feature_keys[feature_i]), 1:length(feature_keys))

  # Output for each feature:
  # raw data (but with relativized winds)
  # 50mi mean
  # 100mi mean
  # 50mi forward grad, 50mi leftward grad, 50mi linestraddling grad
  # 100mi forward grad, 100mi leftward grad, 100mi linestraddling grad
  #
  # And then a lone forecast hour field
  out = Array{Float32}(undef, (grid_point_count, 9*raw_feature_count + 1))

  out[:,1:raw_feature_count] = data

  fifty_mi_mean_features_range            = 1*raw_feature_count+1:2*raw_feature_count
  hundred_mi_mean_features_range          = 2*raw_feature_count+1:3*raw_feature_count
  fifty_mi_forward_gradient_range         = 3*raw_feature_count+1:4*raw_feature_count
  fifty_mi_leftward_gradient_range        = 4*raw_feature_count+1:5*raw_feature_count
  fifty_mi_linestradling_gradient_range   = 5*raw_feature_count+1:6*raw_feature_count
  hundred_mi_forward_gradient_range       = 6*raw_feature_count+1:7*raw_feature_count
  hundred_mi_leftward_gradient_range      = 7*raw_feature_count+1:8*raw_feature_count
  hundred_mi_linestradling_gradient_range = 8*raw_feature_count+1:9*raw_feature_count
  forecast_hour_layer_i                   = 9*raw_feature_count + 1



  # Make 50mi and 100mi mean layers

  fifty_mi_mean_is          = fifty_mi_is(grid)
  unique_hundred_mi_mean_is = unique_hundred_mi_is(grid) # 100mi indices less the 50mi indices

  # 2 at a time: 3.71s per 5 forecasts
  # 3 at a time: 3.55s per 5 forecasts
  # 4 at a time: 3.35s per 5 forecasts
  # 5 at a time: 3.35s per 5 forecasts
  # 6 at a time: 3.30s per 5 forecasts
  # 7 at a time: 3.34s per 5 forecasts
  # 8 at a time: 3.33s per 5 forecasts

  for raw_layer_feature1_i in 1:6:raw_feature_count
    raw_layer_feature2_i = min(raw_layer_feature1_i + 1, raw_feature_count)
    raw_layer_feature3_i = min(raw_layer_feature1_i + 2, raw_feature_count)
    raw_layer_feature4_i = min(raw_layer_feature1_i + 3, raw_feature_count)
    raw_layer_feature5_i = min(raw_layer_feature1_i + 4, raw_feature_count)
    raw_layer_feature6_i = min(raw_layer_feature1_i + 5, raw_feature_count)
    # raw_layer_feature7_i = min(raw_layer_feature1_i + 6, raw_feature_count)
    # raw_layer_feature8_i = min(raw_layer_feature1_i + 7, raw_feature_count)

    fifty_mi_mean_feature1_i   = raw_layer_feature1_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_feature2_i   = raw_layer_feature2_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_feature3_i   = raw_layer_feature3_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_feature4_i   = raw_layer_feature4_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_feature5_i   = raw_layer_feature5_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_feature6_i   = raw_layer_feature6_i + fifty_mi_mean_features_range.start - 1
    # fifty_mi_mean_feature7_i   = raw_layer_feature7_i + fifty_mi_mean_features_range.start - 1
    # fifty_mi_mean_feature8_i   = raw_layer_feature8_i + fifty_mi_mean_features_range.start - 1

    hundred_mi_mean_feature1_i = raw_layer_feature1_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_feature2_i = raw_layer_feature2_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_feature3_i = raw_layer_feature3_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_feature4_i = raw_layer_feature4_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_feature5_i = raw_layer_feature5_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_feature6_i = raw_layer_feature6_i + hundred_mi_mean_features_range.start - 1
    # hundred_mi_mean_feature7_i = raw_layer_feature7_i + hundred_mi_mean_features_range.start - 1
    # hundred_mi_mean_feature8_i = raw_layer_feature8_i + hundred_mi_mean_features_range.start - 1

    @inbounds for flat_i in 1:grid_point_count
      total1 = 0.0f0
      total2 = 0.0f0
      total3 = 0.0f0
      total4 = 0.0f0
      total5 = 0.0f0
      total6 = 0.0f0
      # total7 = 0.0f0
      # total8 = 0.0f0
      n      = 0.0f0
      for near_i in fifty_mi_mean_is[flat_i]
        total1 += data[near_i, raw_layer_feature1_i]
        total2 += data[near_i, raw_layer_feature2_i]
        total3 += data[near_i, raw_layer_feature3_i]
        total4 += data[near_i, raw_layer_feature4_i]
        total5 += data[near_i, raw_layer_feature5_i]
        total6 += data[near_i, raw_layer_feature6_i]
        # total7 += data[near_i, raw_layer_feature7_i]
        # total8 += data[near_i, raw_layer_feature8_i]
        n      += 1.0f0
      end
      out[flat_i, fifty_mi_mean_feature1_i] = total1 / n
      out[flat_i, fifty_mi_mean_feature2_i] = total2 / n
      out[flat_i, fifty_mi_mean_feature3_i] = total3 / n
      out[flat_i, fifty_mi_mean_feature4_i] = total4 / n
      out[flat_i, fifty_mi_mean_feature5_i] = total5 / n
      out[flat_i, fifty_mi_mean_feature6_i] = total6 / n
      # out[flat_i, fifty_mi_mean_feature7_i] = total7 / n
      # out[flat_i, fifty_mi_mean_feature8_i] = total8 / n
      for near_i in unique_hundred_mi_mean_is[flat_i]
        total1 += data[near_i, raw_layer_feature1_i]
        total2 += data[near_i, raw_layer_feature2_i]
        total3 += data[near_i, raw_layer_feature3_i]
        total4 += data[near_i, raw_layer_feature4_i]
        total5 += data[near_i, raw_layer_feature5_i]
        total6 += data[near_i, raw_layer_feature6_i]
        # total7 += data[near_i, raw_layer_feature7_i]
        # total8 += data[near_i, raw_layer_feature8_i]
        n       += 1.0f0
      end
      out[flat_i, hundred_mi_mean_feature1_i] = total1 / n
      out[flat_i, hundred_mi_mean_feature2_i] = total2 / n
      out[flat_i, hundred_mi_mean_feature3_i] = total3 / n
      out[flat_i, hundred_mi_mean_feature4_i] = total4 / n
      out[flat_i, hundred_mi_mean_feature5_i] = total5 / n
      out[flat_i, hundred_mi_mean_feature6_i] = total6 / n
      # out[flat_i, hundred_mi_mean_feature7_i] = total7 / n
      # out[flat_i, hundred_mi_mean_feature8_i] = total8 / n
    end
  end



  # Make 0-6km(ish) mean wind.

  mean_wind_10m_to_500mb_us = Array{Float32}(undef, grid_point_count)
  mean_wind_10m_to_500mb_vs = Array{Float32}(undef, grid_point_count)

  total_weight = 0.0f0

  # Not density weighted, but trying to estimate the depth of the layer represented.

  mean_wind_10m_to_500mb_us .= get_layer("UGRD:10 m above ground:hour fcst:wt ens mean") .* 0.75f0
  mean_wind_10m_to_500mb_vs .= get_layer("VGRD:10 m above ground:hour fcst:wt ens mean") .* 0.75f0
  total_weight += 0.75f0

  mean_wind_10m_to_500mb_us .+= get_layer("UGRD:850 mb:hour fcst:wt ens mean") .* 1.5f0
  mean_wind_10m_to_500mb_vs .+= get_layer("VGRD:850 mb:hour fcst:wt ens mean") .* 1.5f0
  total_weight += 1.5f0

  mean_wind_10m_to_500mb_us .+= get_layer("UGRD:700 mb:hour fcst:wt ens mean") .* 1.25f0
  mean_wind_10m_to_500mb_vs .+= get_layer("VGRD:700 mb:hour fcst:wt ens mean") .* 1.25f0
  total_weight += 1.25f0

  mean_wind_10m_to_500mb_us .+= get_layer("UGRD:600 mb:hour fcst:wt ens mean")
  mean_wind_10m_to_500mb_vs .+= get_layer("VGRD:600 mb:hour fcst:wt ens mean")
  total_weight += 1.0f0

  mean_wind_10m_to_500mb_us .+= get_layer("UGRD:500 mb:hour fcst:wt ens mean")
  mean_wind_10m_to_500mb_vs .+= get_layer("VGRD:500 mb:hour fcst:wt ens mean")
  total_weight += 1.0f0

  mean_wind_10m_to_500mb_us /= total_weight
  mean_wind_10m_to_500mb_vs /= total_weight



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

  fifty_mi_forward_is     = Array{Int64}(undef, grid_point_count)
  fifty_mi_backward_is    = Array{Int64}(undef, grid_point_count)
  fifty_mi_leftward_is    = Array{Int64}(undef, grid_point_count)
  fifty_mi_rightward_is   = Array{Int64}(undef, grid_point_count)
  hundred_mi_forward_is   = Array{Int64}(undef, grid_point_count)
  hundred_mi_backward_is  = Array{Int64}(undef, grid_point_count)
  hundred_mi_leftward_is  = Array{Int64}(undef, grid_point_count)
  hundred_mi_rightward_is = Array{Int64}(undef, grid_point_count)

  for j in 1:height
    for i in 1:width
      flat_i = width*(j-1) + i

      relative_u, relative_v = uv_normalize(mean_wind_10m_to_500mb_us[flat_i], mean_wind_10m_to_500mb_vs[flat_i])

      point_height = Float32(grid.point_heights_miles[flat_i])
      point_width  = Float32(grid.point_widths_miles[flat_i]) # On the SREF grid, point_width and point_height are always nearly equal, <1.0% difference.

      delta_50mi_j  = round(Int64, relative_v * 50.0f0  / point_height)
      delta_50mi_i  = round(Int64, relative_u * 50.0f0  / point_width)
      delta_100mi_j = round(Int64, relative_v * 100.0f0 / point_height)
      delta_100mi_i = round(Int64, relative_u * 100.0f0 / point_width)

      forward_50mi_j    = clamp(j+delta_50mi_j,  1, height)
      forward_50mi_i    = clamp(i+delta_50mi_i,  1, width)
      forward_100mi_j   = clamp(j+delta_100mi_j, 1, height)
      forward_100mi_i   = clamp(i+delta_100mi_i, 1, width)

      backward_50mi_j   = clamp(j-delta_50mi_j,  1, height)
      backward_50mi_i   = clamp(i-delta_50mi_i,  1, width)
      backward_100mi_j  = clamp(j-delta_100mi_j, 1, height)
      backward_100mi_i  = clamp(i-delta_100mi_i, 1, width)

      leftward_50mi_j   = clamp(j+delta_50mi_i,  1, height)
      leftward_50mi_i   = clamp(i-delta_50mi_j,  1, width)
      leftward_100mi_j  = clamp(j+delta_100mi_i, 1, height)
      leftward_100mi_i  = clamp(i-delta_100mi_j, 1, width)

      rightward_50mi_j  = clamp(j-delta_50mi_i,  1, height)
      rightward_50mi_i  = clamp(i+delta_50mi_j,  1, width)
      rightward_100mi_j = clamp(j-delta_100mi_i, 1, height)
      rightward_100mi_i = clamp(i+delta_100mi_j, 1, width)

      fifty_mi_forward_is[flat_i]     = width*(forward_50mi_j-1)    + forward_50mi_i
      fifty_mi_backward_is[flat_i]    = width*(backward_50mi_j-1)   + backward_50mi_i
      fifty_mi_leftward_is[flat_i]    = width*(leftward_50mi_j-1)   + leftward_50mi_i
      fifty_mi_rightward_is[flat_i]   = width*(rightward_50mi_j-1)  + rightward_50mi_i
      hundred_mi_forward_is[flat_i]   = width*(forward_100mi_j-1)   + forward_100mi_i
      hundred_mi_backward_is[flat_i]  = width*(backward_100mi_j-1)  + backward_100mi_i
      hundred_mi_leftward_is[flat_i]  = width*(leftward_100mi_j-1)  + leftward_100mi_i
      hundred_mi_rightward_is[flat_i] = width*(rightward_100mi_j-1) + rightward_100mi_i
    end
  end

  for raw_layer_feature_i in 1:raw_feature_count
    fifty_mi_mean_feature_i           = raw_layer_feature_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_forward_gradient_i       = raw_layer_feature_i + fifty_mi_forward_gradient_range.start - 1
    fifty_mi_leftward_gradient_i      = raw_layer_feature_i + fifty_mi_leftward_gradient_range.start - 1
    fifty_mi_linestradling_gradient_i = raw_layer_feature_i + fifty_mi_linestradling_gradient_range.start - 1

    fifty_mi_forward_vals   = @view out[fifty_mi_forward_is,   fifty_mi_mean_feature_i]
    fifty_mi_backward_vals  = @view out[fifty_mi_backward_is,  fifty_mi_mean_feature_i]
    fifty_mi_leftward_vals  = @view out[fifty_mi_leftward_is,  fifty_mi_mean_feature_i]
    fifty_mi_rightward_vals = @view out[fifty_mi_rightward_is, fifty_mi_mean_feature_i]

    out[:, fifty_mi_forward_gradient_i]       = fifty_mi_forward_vals  .- fifty_mi_backward_vals
    out[:, fifty_mi_leftward_gradient_i]      = fifty_mi_leftward_vals .- fifty_mi_rightward_vals
    out[:, fifty_mi_linestradling_gradient_i] = fifty_mi_forward_vals  .+ fifty_mi_backward_vals .- fifty_mi_leftward_vals .- fifty_mi_rightward_vals
  end

  for raw_layer_feature_i in 1:raw_feature_count
    hundred_mi_mean_feature_i           = raw_layer_feature_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_forward_gradient_i       = raw_layer_feature_i + hundred_mi_forward_gradient_range.start - 1
    hundred_mi_leftward_gradient_i      = raw_layer_feature_i + hundred_mi_leftward_gradient_range.start - 1
    hundred_mi_linestradling_gradient_i = raw_layer_feature_i + hundred_mi_linestradling_gradient_range.start - 1

    hundred_mi_forward_vals   = @view out[hundred_mi_forward_is,   hundred_mi_mean_feature_i]
    hundred_mi_backward_vals  = @view out[hundred_mi_backward_is,  hundred_mi_mean_feature_i]
    hundred_mi_leftward_vals  = @view out[hundred_mi_leftward_is,  hundred_mi_mean_feature_i]
    hundred_mi_rightward_vals = @view out[hundred_mi_rightward_is, hundred_mi_mean_feature_i]

    out[:, hundred_mi_forward_gradient_i]       = hundred_mi_forward_vals  .- hundred_mi_backward_vals
    out[:, hundred_mi_leftward_gradient_i]      = hundred_mi_leftward_vals .- hundred_mi_rightward_vals
    out[:, hundred_mi_linestradling_gradient_i] = hundred_mi_forward_vals  .+ hundred_mi_backward_vals .- hundred_mi_leftward_vals .- hundred_mi_rightward_vals
  end



  # Make ~500m - ~5500m shear vector relative to which we will rotate the winds

  mean_wind_10m_to_850mb_us   = 0.5f0 .* (get_layer("UGRD:10 m above ground:hour fcst:wt ens mean") .+ get_layer("UGRD:850 mb:hour fcst:wt ens mean"))
  mean_wind_10m_to_850mb_vs   = 0.5f0 .* (get_layer("VGRD:10 m above ground:hour fcst:wt ens mean") .+ get_layer("VGRD:850 mb:hour fcst:wt ens mean"))
  mean_wind_600mb_to_500mb_us = 0.5f0 .* (get_layer("UGRD:600 mb:hour fcst:wt ens mean") .+ get_layer("UGRD:500 mb:hour fcst:wt ens mean"))
  mean_wind_600mb_to_500mb_vs = 0.5f0 .* (get_layer("VGRD:600 mb:hour fcst:wt ens mean") .+ get_layer("VGRD:500 mb:hour fcst:wt ens mean"))

  mean_wind_angles = atan.(mean_wind_600mb_to_500mb_vs .- mean_wind_10m_to_850mb_vs, mean_wind_600mb_to_500mb_us .- mean_wind_10m_to_850mb_us)

  if any(isnan, mean_wind_angles)
    error("nan wind angle")
  end

  rot_coses = cos.(-mean_wind_angles)
  rot_sines = sin.(-mean_wind_angles)



  # Center wind vectors around 0-6km(ish) mean wind
  # Rotate winds to align to the ~500m - ~5500m shear vector (inspired by Bunker's storm motion, which we are not calculating yet)

  for wind_layer_key in vector_wind_layers
    layer_key_u = "U" * wind_layer_key
    layer_key_v = "V" * wind_layer_key

    raw_layer_u_i                         = feature_key_to_i[layer_key_u]
    raw_layer_v_i                         = feature_key_to_i[layer_key_v]
    fifty_mi_mean_u_i                     = raw_layer_u_i + fifty_mi_mean_features_range.start - 1
    fifty_mi_mean_v_i                     = raw_layer_v_i + fifty_mi_mean_features_range.start - 1
    hundred_mi_mean_u_i                   = raw_layer_u_i + hundred_mi_mean_features_range.start - 1
    hundred_mi_mean_v_i                   = raw_layer_v_i + hundred_mi_mean_features_range.start - 1
    fifty_mi_forward_gradient_u_i         = raw_layer_u_i + fifty_mi_forward_gradient_range.start - 1
    fifty_mi_forward_gradient_v_i         = raw_layer_v_i + fifty_mi_forward_gradient_range.start - 1
    fifty_mi_leftward_gradient_u_i        = raw_layer_u_i + fifty_mi_leftward_gradient_range.start - 1
    fifty_mi_leftward_gradient_v_i        = raw_layer_v_i + fifty_mi_leftward_gradient_range.start - 1
    fifty_mi_linestradling_gradient_u_i   = raw_layer_u_i + fifty_mi_linestradling_gradient_range.start - 1
    fifty_mi_linestradling_gradient_v_i   = raw_layer_v_i + fifty_mi_linestradling_gradient_range.start - 1
    hundred_mi_forward_gradient_u_i       = raw_layer_u_i + hundred_mi_forward_gradient_range.start - 1
    hundred_mi_forward_gradient_v_i       = raw_layer_v_i + hundred_mi_forward_gradient_range.start - 1
    hundred_mi_leftward_gradient_u_i      = raw_layer_u_i + hundred_mi_leftward_gradient_range.start - 1
    hundred_mi_leftward_gradient_v_i      = raw_layer_v_i + hundred_mi_leftward_gradient_range.start - 1
    hundred_mi_linestradling_gradient_u_i = raw_layer_u_i + hundred_mi_linestradling_gradient_range.start - 1
    hundred_mi_linestradling_gradient_v_i = raw_layer_v_i + hundred_mi_linestradling_gradient_range.start - 1

    # Center the non-gradient layers (gradients are relativized already)

    us = out[:, raw_layer_u_i] .- mean_wind_10m_to_500mb_us
    vs = out[:, raw_layer_v_i] .- mean_wind_10m_to_500mb_vs

    fifty_mi_mean_us = out[:, fifty_mi_mean_u_i] .- mean_wind_10m_to_500mb_us
    fifty_mi_mean_vs = out[:, fifty_mi_mean_v_i] .- mean_wind_10m_to_500mb_vs

    hundred_mi_mean_us = out[:, hundred_mi_mean_u_i] .- mean_wind_10m_to_500mb_us
    hundred_mi_mean_vs = out[:, hundred_mi_mean_v_i] .- mean_wind_10m_to_500mb_vs

    # Grab the other layers.... (Don't use @view because u and v are each used in the calculation of the other)

    fifty_mi_forward_gradient_us         = out[:, fifty_mi_forward_gradient_u_i]
    fifty_mi_forward_gradient_vs         = out[:, fifty_mi_forward_gradient_v_i]
    fifty_mi_leftward_gradient_us        = out[:, fifty_mi_leftward_gradient_u_i]
    fifty_mi_leftward_gradient_vs        = out[:, fifty_mi_leftward_gradient_v_i]
    fifty_mi_linestradling_gradient_us   = out[:, fifty_mi_linestradling_gradient_u_i]
    fifty_mi_linestradling_gradient_vs   = out[:, fifty_mi_linestradling_gradient_v_i]
    hundred_mi_forward_gradient_us       = out[:, hundred_mi_forward_gradient_u_i]
    hundred_mi_forward_gradient_vs       = out[:, hundred_mi_forward_gradient_v_i]
    hundred_mi_leftward_gradient_us      = out[:, hundred_mi_leftward_gradient_u_i]
    hundred_mi_leftward_gradient_vs      = out[:, hundred_mi_leftward_gradient_v_i]
    hundred_mi_linestradling_gradient_us = out[:, hundred_mi_linestradling_gradient_u_i]
    hundred_mi_linestradling_gradient_vs = out[:, hundred_mi_linestradling_gradient_v_i]

    # And rotate everrrryything.

    out[:, raw_layer_u_i]                         = (us                                   .* rot_coses) .- (vs                                   .* rot_sines)
    out[:, raw_layer_v_i]                         = (us                                   .* rot_sines) .+ (vs                                   .* rot_coses)
    out[:, fifty_mi_mean_u_i]                     = (fifty_mi_mean_us                     .* rot_coses) .- (fifty_mi_mean_vs                     .* rot_sines)
    out[:, fifty_mi_mean_v_i]                     = (fifty_mi_mean_us                     .* rot_sines) .+ (fifty_mi_mean_vs                     .* rot_coses)
    out[:, hundred_mi_mean_u_i]                   = (hundred_mi_mean_us                   .* rot_coses) .- (hundred_mi_mean_vs                   .* rot_sines)
    out[:, hundred_mi_mean_v_i]                   = (hundred_mi_mean_us                   .* rot_sines) .+ (hundred_mi_mean_vs                   .* rot_coses)
    out[:, fifty_mi_forward_gradient_u_i]         = (fifty_mi_forward_gradient_us         .* rot_coses) .- (fifty_mi_forward_gradient_vs         .* rot_sines)
    out[:, fifty_mi_forward_gradient_v_i]         = (fifty_mi_forward_gradient_us         .* rot_sines) .+ (fifty_mi_forward_gradient_vs         .* rot_coses)
    out[:, fifty_mi_leftward_gradient_u_i]        = (fifty_mi_leftward_gradient_us        .* rot_coses) .- (fifty_mi_leftward_gradient_vs        .* rot_sines)
    out[:, fifty_mi_leftward_gradient_v_i]        = (fifty_mi_leftward_gradient_us        .* rot_sines) .+ (fifty_mi_leftward_gradient_vs        .* rot_coses)
    out[:, fifty_mi_linestradling_gradient_u_i]   = (fifty_mi_linestradling_gradient_us   .* rot_coses) .- (fifty_mi_linestradling_gradient_vs   .* rot_sines)
    out[:, fifty_mi_linestradling_gradient_v_i]   = (fifty_mi_linestradling_gradient_us   .* rot_sines) .+ (fifty_mi_linestradling_gradient_vs   .* rot_coses)
    out[:, hundred_mi_forward_gradient_u_i]       = (hundred_mi_forward_gradient_us       .* rot_coses) .- (hundred_mi_forward_gradient_vs       .* rot_sines)
    out[:, hundred_mi_forward_gradient_v_i]       = (hundred_mi_forward_gradient_us       .* rot_sines) .+ (hundred_mi_forward_gradient_vs       .* rot_coses)
    out[:, hundred_mi_leftward_gradient_u_i]      = (hundred_mi_leftward_gradient_us      .* rot_coses) .- (hundred_mi_leftward_gradient_vs      .* rot_sines)
    out[:, hundred_mi_leftward_gradient_v_i]      = (hundred_mi_leftward_gradient_us      .* rot_sines) .+ (hundred_mi_leftward_gradient_vs      .* rot_coses)
    out[:, hundred_mi_linestradling_gradient_u_i] = (hundred_mi_linestradling_gradient_us .* rot_coses) .- (hundred_mi_linestradling_gradient_vs .* rot_sines)
    out[:, hundred_mi_linestradling_gradient_v_i] = (hundred_mi_linestradling_gradient_us .* rot_sines) .+ (hundred_mi_linestradling_gradient_vs .* rot_coses)
  end



  # Final layer: the forecast hour.
  #
  # Only 10 hour resolution. Don't want to overfit.
  out[:, forecast_hour_layer_i] = repeat([Float32(div(forecast.forecast_hour, 10))], grid_point_count)



  out
end

end # module FeatureEngineering