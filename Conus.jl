module Conus

import GDAL
import ArchGDAL
import Grids

function is_in_conus((lat, lon) :: Tuple{Float64, Float64}) :: Bool
  ArchGDAL.registerdrivers() do
    ArchGDAL.read("geo_regions/geo_layers/ln_us/ln_us.shp") do dataset # Have to use this original file. GDAL is finicky about the geometry created by attempting to simplify it.
      # print(dataset)
      layer = ArchGDAL.getlayer(dataset, 0)
      # println(layer)
      ArchGDAL.getfeature(layer, 0) do feature
        # println(feature)
        conus = ArchGDAL.getgeom(feature)
        # # ArchGDAL.closerings!(conus)
        # println(conus)
        point = ArchGDAL.createpoint(lon, lat)
        # point
        # point = ArchGDAL.pointonsurface(conus)
        ArchGDAL.contains(conus, point)
      end
    end
  end
end

# Returns a layer of 0.0/1.0 indicating where CONUS is.
function grid_to_conus(grid :: Grids.Grid) :: Array{Float32,1}
  ArchGDAL.registerdrivers() do
    ArchGDAL.read("geo_regions/geo_layers/ln_us/ln_us.shp") do dataset # Have to use this original file. GDAL is finicky about the geometry created by attempting to simplify it.
      layer = ArchGDAL.getlayer(dataset, 0)
      ArchGDAL.getfeature(layer, 0) do feature
        conus = ArchGDAL.getgeom(feature)
        # point = ArchGDAL.createpoint(lon, lat)
        points = map(latlon -> ArchGDAL.createpoint(latlon[2], latlon[1]), grid.lat_lons) # Probably a memory leak here.
        map(point -> ArchGDAL.contains(conus, point) ? 1.0f0 : 0.0f0, points)
      end
    end
  end
end

end # module Conus