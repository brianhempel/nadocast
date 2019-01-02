module MakeConusGrid

# This file is only for producing conus_on_rap_130_grid.txt, a 0.0/1.0 indicator
# grid of where the continental united states is.
#
# If that file exists, then this code can be ignored.
#
#
# conus_on_rap_130_grid.txt may be reproduced by:
#
# import Grib2
# import DelimitedFiles # For readdlm
# import MakeConusGrid
#
# DelimitedFiles.writedlm("conus_on_rap_130_grid.txt", MakeConusGrid.grid_to_conus(Grib2.read_grid("test_grib2s/rap_130_20180319_1400_012.grb2")))


import Grids
import GDAL
import ArchGDAL

# Returns a grid layer of 0.0/1.0 values indicating where CONUS is.
#
# Really slow. About 4 hours for the RAP grid. Hence the cached result "conus_on_rap_130_grid.txt".
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

end # module MakeConusGrid