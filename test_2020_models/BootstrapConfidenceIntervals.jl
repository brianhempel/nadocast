import DelimitedFiles
import Random
import Statistics

table, headers = DelimitedFiles.readdlm("test_0z.csv", ','; header = true)

headers = String.(headers[1,:])

# AbstractString["yymmdd" "spc" "nadocast" "spc_painted_sq_mi_0.02" "spc_true_positive_sq_mi_0.02" "spc_false_negative_sq_mi_0.02" "nadocast_painted_sq_mi_0.02" "nadocast_true_positive_sq_mi_0.02" "nadocast_false_negative_sq_mi_0.02" "spc_painted_sq_mi_0.05" "spc_true_positive_sq_mi_0.05" "spc_false_negative_sq_mi_0.05" "nadocast_painted_sq_mi_0.05" "nadocast_true_positive_sq_mi_0.05" "nadocast_false_negative_sq_mi_0.05" "spc_painted_sq_mi_0.1" "spc_true_positive_sq_mi_0.1" "spc_false_negative_sq_mi_0.1" "nadocast_painted_sq_mi_0.1" "nadocast_true_positive_sq_mi_0.1" "nadocast_false_negative_sq_mi_0.1" "spc_painted_sq_mi_0.15" "spc_true_positive_sq_mi_0.15" "spc_false_negative_sq_mi_0.15" "nadocast_painted_sq_mi_0.15" "nadocast_true_positive_sq_mi_0.15" "nadocast_false_negative_sq_mi_0.15" "spc_painted_sq_mi_0.3" "spc_true_positive_sq_mi_0.3" "spc_false_negative_sq_mi_0.3" "nadocast_painted_sq_mi_0.3" "nadocast_true_positive_sq_mi_0.3" "nadocast_false_negative_sq_mi_0.3" "spc_painted_sq_mi_0.45" "spc_true_positive_sq_mi_0.45" "spc_false_negative_sq_mi_0.45" "nadocast_painted_sq_mi_0.45" "nadocast_true_positive_sq_mi_0.45" "nadocast_false_negative_sq_mi_0.45" "spc_painted_sq_mi_0.6" "spc_true_positive_sq_mi_0.6" "spc_false_negative_sq_mi_0.6" "nadocast_painted_sq_mi_0.6" "nadocast_true_positive_sq_mi_0.6" "nadocast_false_negative_sq_mi_0.6"]

ε = Float64(eps(1f0))

for treatment in ["spc", "nadocast"]
  for threshold in ["0.02", "0.05", "0.1", "0.15", "0.3", "0.45", "0.6"]
    painted_sq_mi        = Float64.(table[:, findfirst(isequal("$(treatment)_painted_sq_mi_$(threshold)"), headers)])
    true_positive_sq_mi  = Float64.(table[:, findfirst(isequal("$(treatment)_true_positive_sq_mi_$(threshold)"), headers)])
    false_negative_sq_mi = Float64.(table[:, findfirst(isequal("$(treatment)_false_negative_sq_mi_$(threshold)"), headers)])

    bootstraps = map(1:1_000_000) do _
      painted, true_pos, false_neg = 0.001, 0.0, 0.001
      for _ in 1:length(painted_sq_mi)
        i = rand(1:length(painted_sq_mi))
        painted   += painted_sq_mi[i]
        true_pos  += true_positive_sq_mi[i]
        false_neg += false_negative_sq_mi[i]
      end
      success_ratio = true_pos / painted
      pod           = true_pos / (true_pos + false_neg)
      (success_ratio, pod)
    end

    # println("$(treatment)_$(threshold)")
    # for (sr, pod) in bootstraps[1:100]
    #   println("$(Float32(sr)),$(Float32(pod))")
    # end

    # use median for shape center
    base_success_ratio   = Statistics.quantile(map(sr_pod -> sr_pod[1], bootstraps), 0.5)
    base_pod             = Statistics.quantile(map(sr_pod -> sr_pod[2], bootstraps), 0.5)
    base                 = (base_success_ratio, base_pod)

    segment_deg = 4
    segment_rad = segment_deg/360*2π
    confidence_circle_points = []
    for deg in -180:segment_deg:179
      rad = deg/360*2π
      θ1 = rad - segment_rad/2
      θ2 = rad + segment_rad/2
      unit_vec = (cos(rad), sin(rad)) # This is a little conservative, actually, drawing the segments as triangles instead of arcs, using projections instead of distance
      in_segment = filter(bootstraps) do sr_pod
        dx, dy = sr_pod .- base
        angle = atan(dy, dx)
        projected = sum(unit_vec .* (dx, dy))
        between(angle, θ1, θ2) = θ1 <= angle && angle < θ2
        projected > 0 && (between(angle, θ1, θ2) || between(angle+2π, θ1, θ2) || between(angle-2π, θ1, θ2))
      end
      projected = map(sr_pod -> sum(unit_vec .* (sr_pod .- base)), in_segment)
      boundary = length(projected) == 0 ? 0.0 : Statistics.quantile(projected, 0.95)
      # isoceles triangle with height boundary, point at base, in direction of rad
      θ1_vec = (cos(θ1), sin(θ1)) ./ cos(segment_rad/2)
      θ2_vec = (cos(θ2), sin(θ2)) ./ cos(segment_rad/2)
      push!(confidence_circle_points, (boundary .* θ1_vec) .+ base)
      push!(confidence_circle_points, (boundary .* θ2_vec) .+ base)
    end

    # Make it less jaggy (this is conservative because it only expands the region area)
    confidence_circle_points = filter(confidence_circle_points) do sr_pod
      dx, dy = sr_pod .- base
      angle = atan(dy, dx)
      d = sqrt(dx^2 + dy^2)
      # If there's any point at the same angle, but further from the base, drop this point
      !any(confidence_circle_points) do sr_pod₂
        dx₂, dy₂ = sr_pod₂ .- base
        angle₂ = atan(dy₂, dx₂)
        d₂ = sqrt(dx₂^2 + dy₂^2)
        d₂ > d && abs(angle₂ - angle) < segment_deg/100/360*2π
      end
    end

    println("$(treatment)_$(threshold)")
    for (sr, pod) in confidence_circle_points
      println("$(Float32(sr)),$(Float32(pod))")
    end
  end
end
