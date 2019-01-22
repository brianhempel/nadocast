module Conus

import DelimitedFiles

push!(LOAD_PATH, @__DIR__)
import Grids

# rap_130_grid.bin produced by:
#
# import Grids
# import Grib2
#
# Grids.to_file("rap_130_grid.bin", Grib2.read_grid("test_grib2s/rap_130_20180319_1400_012.grb2"))
const rap_130_grid = Grids.from_file((@__DIR__) * "/rap_130_grid.bin") :: Grids.Grid

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