module SPC

# For reading SPC Convective Outlook regions.
#
# Rasterizes onto the given grid.
#
# Asking for

import ArchGDAL


# Read the SPC probability shapefile and returns BitArray mask for the given grid.
#
# Shapefile url: https://www.spc.noaa.gov/products/outlook/archive/2019/day1otlk_20190812_1300-shp.zip
#
# Should work for tornadoes, wind, and hail (all use "DN" field to label the regions).
#
# rasterize_prob_regions(Conus.href_cropped_5km_grid, 0.02, "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp")
function rasterize_prob_regions(grid, threshold_prob, shapefile_path)

  mask = falses(length(grid.latlons))

  ArchGDAL.registerdrivers() do
    ArchGDAL.importEPSG(4326) do wgs84
    # ArchGDAL.importEPSG(4269) do nad83
      ArchGDAL.read(shapefile_path) do dataset
        # println(dataset)
        @assert ArchGDAL.nlayer(dataset) == 1
        layer = ArchGDAL.getlayer(dataset, 0)
        # println(layer)
        # println(ArchGDAL.getlayerdefn(layer))
        feature_count = ArchGDAL.nfeature(layer)
        # println("$feature_count features")
        for feature_i in 0:(feature_count-1)
          ArchGDAL.getfeature(layer, feature_i) do feature
            # println(feature)
            # println("DN: $(ArchGDAL.getfield(feature, "DN"))")
            # DN field contains the probability level (in percent: 2, 5, 10 etc)
            if ArchGDAL.getfield(feature, "DN") / 100 >= threshold_prob - 10*eps(threshold_prob)
              geom = ArchGDAL.getgeom(feature)
              # println(geom)
              spatialref = ArchGDAL.getspatialref(geom)
              # println(spatialref)
              ArchGDAL.createcoordtrans(spatialref, wgs84) do transform
                ArchGDAL.transform!(geom, transform)
                # println(geom)
              end
              ArchGDAL.createpoint(0,0) do point
                _add_geom_to_mask!(mask, grid.latlons, geom, point)
                # test_point(-90.3, 39.9, geom, point)
                # test_point(-91.9, 41.7, geom, point)
                # test_point(-91.9, 41.8, geom, point)
                # test_point(-96.03, 45.15, geom, point)
                # test_point(-95.95, 45.15, geom, point)
              end
            end
          end
        end
      end
    end
  end

  mask
end


# Mutates mask. On HREF grid, ~3-5s per geom.
function _add_geom_to_mask!(mask, grid_latlons, geom, point)
  bounds = ArchGDAL.getenvelope(geom)

  for i in 1:length(grid_latlons)
    lat, lon = grid_latlons[i]

    if mask[i] || lon > bounds.MaxX || lon < bounds.MinX || lat > bounds.MaxY || lat < bounds.MinY
      continue
    end

    ArchGDAL.setpoint!(point, 0, lon, lat)

    if ArchGDAL.contains(geom, point)
      mask[i] = true
    end
  end

  ()
end



# Helpful scratch below:
#
# # function test_point(lon, lat, geom, point)
# #   ArchGDAL.setpoint!(point, 0, lon, lat)
# #   println("$lon, $lat: $(ArchGDAL.contains(geom, point))")
# # end
#
# import ArchGDAL
#
# push!(LOAD_PATH, "./lib")
#
# import SPC
# import Conus
# import Grids
#
# threshold = 0.02
# grid = Conus.href_cropped_5km_grid
# # shapefile_path = "geo_regions/ln_us/ln_us.shp"
# shapefile_path = "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp"
#
# @time mask = SPC.rasterize_prob_regions(grid, threshold, "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp")
#
# target_latlons = grid.latlons[mask]
#
# Grids.latlons_to_csv("tor_latlons.csv", target_latlons)

end
