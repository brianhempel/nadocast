push!(LOAD_PATH, joinpath(@__DIR__, ".."))

import Grids
using Utils
import GMTPlot


grid = Grids.grid_130_cropped

# These files are generated in the Nadocast repo.
plots = [
  ("total_prob_nadocast_wind_183_days_12z_absolutely_calibrated.csv",     183,  "Nadocast 12Z Wind, Average Day Probability",          "colors_8.cpt"),
  ("total_prob_nadocast_wind_adj_183_days_12z_absolutely_calibrated.csv", 183,  "Nadocast 12Z Wind Adjusted, Average Day Probability", "colors_3.cpt"),
  ("total_prob_wind_4017_days_13z_spc_all.csv",                           4017, "SPC 13Z Wind Outlook, Average Day Probability",       "colors_1.5.cpt"),
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