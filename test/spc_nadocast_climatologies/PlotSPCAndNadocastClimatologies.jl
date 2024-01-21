# cribbed from the adjusted wind repo
# hence the extra lib files here

push!(LOAD_PATH, joinpath(@__DIR__))

import Grids
using Utils
import GMTPlot


grid = Grids.grid_130_cropped

plots = [
  ("../total_prob_nadocast_hail_600_days_0z_spc_calibrated.csv", 600, "Nadocast 0Z Hail, Average Day Probability", "colors_8.cpt"),
  ("../total_prob_spc_hail_600_days_0z_spc_calibrated.csv",      600, "SPC 6Z Hail, Average Day Probability",      "colors_8.cpt"),
  ("../total_prob_reports_hail_600_days_0z_spc_calibrated.csv",  600, "Hail Reports, Average Day Probability",     "colors_8.cpt"),
]

for (csv_path, ndays, title, colors) in plots
  println(csv_path)

  gridded_13km_total_prob = zeros(Float32, length(grid.latlons))
  gridded_13km_weight     = zeros(Float32, length(grid.latlons))

  for line in readlines(csv_path)[2:end]
    lat, lon, prob = parse.(Float64, split(line, ","))

    grid_i = Grids.latlon_to_closest_grid_i(grid, (lat, lon))

    gridded_13km_total_prob[grid_i] += prob
    gridded_13km_weight[grid_i]     += 1
  end

  gridded_13km = gridded_13km_total_prob ./ (gridded_13km_weight .+ eps(Float32)) ./ ndays .* 100

  # GMTPlot.plot_map("severe_thunderstorm_warning_days_per_year_blurred_more", Grids.grid_130_cropped.latlons, gridded_counts_13km_blurred_more; title = "Severe Thunderstorm Warning Days/Year", label_contours = true, steps = 7, zlow = 0, zhigh = 14, colors = colors_path_14)
  GMTPlot.plot_map(replace(csv_path, ".csv" => ""), grid.latlons, gridded_13km; title = title, nearest_neighbor = false, label_contours = true, steps = nothing, colors = colors)
end