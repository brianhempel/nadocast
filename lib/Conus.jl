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


const rap_130_grid = Grib2.read_grid((@__DIR__) * "/rap_130_one_field_for_grid.grb2") :: Grids.Grid

# conus_on_rap_130_grid.txt produced by:
#
# import Grib2
# import MakeConusGrid
# import DelimitedFiles # For readdlm
#
# DelimitedFiles.writedlm("conus_on_rap_130_grid.txt", MakeConusGrid.grid_to_conus(Grib2.read_grid("test_grib2s/rap_130_20180319_1400_012.grb2")))
const conus_layer_data_on_rap_130_grid = DelimitedFiles.readdlm((@__DIR__) * "/conus_on_rap_130_grid.txt", Float32)[:,1] :: Array{Float32,1} # 0.0/1.0 indicator layer of conus

function is_in_conus(latlon :: Tuple{Float64, Float64}) :: Bool
  flat_i = Grids.latlon_to_closest_grid_i(rap_130_grid, latlon)
  conus_layer_data_on_rap_130_grid[flat_i] > 0.5f0
end

end # module Conus