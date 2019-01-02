module Conus

import DelimitedFiles
import Grids

# rap_130_grid.bin produced by:
#
# import Grids
# import Grib2
#
# Grids.to_file("rap_130_grid.bin", Grib2.read_grid("test_grib2s/rap_130_20180319_1400_012.grb2"))
const rap_130_grid = Grids.from_file("rap_130_grid.bin") :: Grids.Grid

# conus_on_rap_130_grid.txt produced by:
#
# import Grib2
# import DelimitedFiles # For readdlm
#
# DelimitedFiles.writedlm("conus_on_rap_130_grid.txt", grid_to_conus(Grib2.read_grid("test_grib2s/rap_130_20180319_1400_012.grb2")))
const conus_layer_data_on_rap_130_grid = DelimitedFiles.readdlm("conus_on_rap_130_grid.txt", Float32)[:,1] :: Array{Float32,1} # 0.0/1.0 indicator layer of conus

function is_in_conus(latlon :: Tuple{Float64, Float64}) :: Bool
  flat_i = Grids.lat_lon_to_closest_grid_i(rap_130_grid, latlon)
  conus_layer_data_on_rap_130_grid[flat_i] > 0.5f0
end

# We only need GDAL/ArchGDAL for this function.
#
# Returns a grid layer of 0.0/1.0 values indicating where CONUS is.
#
# Really slow. About 4 hours for the RAP grid. Hence the cached result above (conus_layer_data_on_rap_130_grid).
import GDAL
import ArchGDAL

function grid_to_conus(grid :: Grids.Grid) :: Array{Float32,1}
  ArchGDAL.registerdrivers() do
    ArchGDAL.read("geo_regions/geo_layers/ln_us/ln_us.shp") do dataset # Have to use this original file. GDAL is finicky about the geometry created by attempting to simplify it.
      layer = ArchGDAL.getlayer(dataset, 0)
      ArchGDAL.getfeature(layer, 0) do feature
        conus = ArchGDAL.getgeom(feature)
        # point = ArchGDAL.createpoint(lon, lat)
        points = map(latlon -> ArchGDAL.createpoint(latlon[2], latlon[1]), grid.lat_lons) # Possibly a memory leak here.
        map(point -> ArchGDAL.contains(conus, point) ? 1.0f0 : 0.0f0, points)
      end
    end
  end
end

end # module Conus