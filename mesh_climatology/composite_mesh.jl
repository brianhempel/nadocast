import StatsBase
using PNGFiles

# function filter_mesh(img, mm)
#   black       = PNGFiles.GrayA{PNGFiles.N0f8}(0.0,1.0)
#   transparent = PNGFiles.GrayA{PNGFiles.N0f8}(0.0,0.0)

#   hue_threshold = Dict(30 => 20, 20 => 40)[mm]

#   map(img) do color
#     hsla = convert(PNGFiles.HSLA, color)
#     if hsla.alpha > 0.9 && hsla.s > 0.9 && (hsla.h > 260.0 || hsla.h < hue_threshold)
#       black
#     else
#       transparent
#     end
#   end
# end

# for (dir_path, _, file_names) in walkdir(@__DIR__)
#   for file_name in file_names
#     if contains(file_name, r"\Amesh_\d\d\d\d-\d\d-\d\d\.png")
#       img = PNGFiles.load(joinpath(dir_path, file_name))
#       new_img = filter_mesh(img, 20)
#       PNGFiles.save(joinpath(dir_path, replace(file_name, ".png" => "_>20mm.png")), new_img; compression_level = 9)
#       new_img = filter_mesh(img, 30)
#       PNGFiles.save(joinpath(dir_path, replace(file_name, ".png" => "_>30mm.png")), new_img; compression_level = 9)
#     end
#   end
# end

function sum_counts(mm)
  counts = zeros(Int64, 630, 920)

  for (dir_path, _, file_names) in walkdir(@__DIR__)
    for file_name in file_names
      if contains(file_name, r"\Amesh_\d\d\d\d-\d\d-\d\d_>\d\dmm\.png") && endswith(file_name, "_>$(mm)mm.png")
        counts .+= map(color -> color.alpha > 0.5 ? 1 : 0, PNGFiles.load(joinpath(dir_path, file_name)))
      end
    end
  end

  max_count = maximum(counts)
  percentile_99p995 = Int64(round(StatsBase.percentile(counts[:], 99.995)))
  percentile_99p99  = Int64(round(StatsBase.percentile(counts[:], 99.99)))
  percentile_99p9   = Int64(round(StatsBase.percentile(counts[:], 99.9)))
  percentile_99     = Int64(round(StatsBase.percentile(counts[:], 99)))
  println("$(mm)  Max: $max_count  99.995th percentile: $percentile_99p995  99.99th percentile: $percentile_99p99  99.9th percentile: $percentile_99p9  99th percentile: $percentile_99")
  counts_colored = map(counts) do count
    PNGFiles.Gray{PNGFiles.N0f8}(min(1.0, count / (percentile_99p99 + 1)))
  end
  PNGFiles.save(joinpath(@__DIR__, "$(mm)mm_days.png"), counts_colored; compression_level = 9)
end

sum_counts(20)
sum_counts(30)