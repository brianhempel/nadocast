module HREF15KMGrid

export HREF_CROPPED_15KM_GRID

push!(LOAD_PATH, @__DIR__)
import Grib2
import Grids

# Same cropping and 3x downsampling as in HREF.jl
const HREF_CROPPED_15KM_GRID =
  Grib2.read_grid(
    (@__DIR__) * "/href_one_field_for_grid.grib2",
    crop = ((1+214):(1473 - 99), (1+119):(1025-228)),
    downsample = 3
  ) :: Grids.Grid

end