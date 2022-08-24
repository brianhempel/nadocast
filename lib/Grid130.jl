module Grid130

export GRID_130_CROPPED

push!(LOAD_PATH, @__DIR__)
import Grib2
import Grids

# 13km grid
# Same cropping as in RAP.jl
const GRID_130_CROPPED = Grib2.read_grid((@__DIR__) * "/a_file_with_grid_130_cropped.grib2",) :: Grids.Grid

end