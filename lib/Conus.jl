module Conus

import DelimitedFiles

push!(LOAD_PATH, @__DIR__)
import Grids
import Grib2

# https://en.wikipedia.org/wiki/List_of_extreme_points_of_the_United_States
const s_extreme = 24.0
const n_extreme = 50.0
const w_extreme = -125.0
const e_extreme = -66.0

# For excluding Alaska, Hawaii, Puerto Rico tornadoes
function is_in_conus_bounding_box((lat, lon))
  lat > s_extreme && lat < n_extreme && lon > w_extreme && lon < e_extreme
end

# Same cropping as in HREF.jl
const href_cropped_5km_grid = Grib2.read_grid((@__DIR__) * "/href_one_field_for_grid.grib2", crop = ((1+214):(1473 - 99), (1+119):(1025-228))) :: Grids.Grid

# See MakeConusGrid for how to create HREF_conus_latlons.csv
const conus_mask_href_cropped_5km_grid = begin
  latlons, _ = DelimitedFiles.readdlm((@__DIR__) * "/HREF_conus_latlons.csv", Float64; header = true)
  conus_mask = falses(length(href_cropped_5km_grid.latlons))

  for (lat, lon) in eachrow(latlons)
    conus_mask[Grids.latlon_to_closest_grid_i(href_cropped_5km_grid, (lat, lon))] = true
  end

  conus_mask
end

# Based on closest HREF 5km gridpoint
function is_in_conus(latlon :: Tuple{Float64, Float64}) :: Bool
  flat_i = Grids.latlon_to_closest_grid_i(href_cropped_5km_grid, latlon)
  conus_mask_href_cropped_5km_grid[flat_i]
end

end # module Conus
