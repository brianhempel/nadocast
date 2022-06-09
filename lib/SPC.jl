module SPC

# For reading SPC Convective Outlook regions.
#
# Rasterizes onto the given grid.
#
# Asking for a threshold probability of 0.02 will give you the union of the 2%, 5%, 10%, etc regions.

import ArchGDAL

push!(LOAD_PATH, @__DIR__)
import Cache
import Grids


# Read the SPC probability shapefile and return the threshold probabilities in the file
# Returned array is unique, sorted, range 0.0-1.0
function threshold_probs(shapefile_path)
  thresholds = Float64[]

  ArchGDAL.registerdrivers() do
    ArchGDAL.importEPSG(4326) do wgs84
    # ArchGDAL.importEPSG(4269) do nad83
      ArchGDAL.read(shapefile_path) do dataset
        @assert ArchGDAL.nlayer(dataset) == 1
        layer = ArchGDAL.getlayer(dataset, 0)
        feature_count = ArchGDAL.nfeature(layer)
        for feature_i in 0:(feature_count-1)
          ArchGDAL.getfeature(layer, feature_i) do feature
            # DN field contains the probability level (in percent: 2, 5, 10 etc)
            dn =
              if isa(ArchGDAL.getfield(feature, "DN"), String)
                # println("DN: $(ArchGDAL.getfield(feature, "DN"))")
                parse(Float64, ArchGDAL.getfield(feature, "DN"))
              else
                ArchGDAL.getfield(feature, "DN")
              end
            push!(thresholds, Float64(dn) * 0.01)
          end
        end
      end
    end
  end

  sort(unique(thresholds))
end


function latlons_on_spacial_ref(grid, spatialref)
  wkt_str = ArchGDAL.toWKT(spatialref)

  Cache.cached([Grids.grid_cache_folder(grid)], "xys_reprojected_$(hash(wkt_str))") do
    ArchGDAL.importEPSG(4326) do wgs84
      ArchGDAL.createcoordtrans(wgs84, spatialref) do transform
        ArchGDAL.createpoint(0,0) do point
          println(wkt_str)
          map(grid.latlons) do (lat, lon)
            ArchGDAL.setpoint!(point, 0, lon, lat)
            ArchGDAL.transform!(point, transform)
            x = ArchGDAL.getx(point, 0)
            y = ArchGDAL.gety(point, 0)
            (x, y)
          end
        end
      end
    end
  end
end


# Read the SPC probability shapefile and returns BitArray mask for the given grid.
#
# Shapefile url: https://www.spc.noaa.gov/products/outlook/archive/2019/day1otlk_20190812_1300-shp.zip
#
# Should work for tornadoes, wind, and hail (all use "DN" field to label the regions).
#
# rasterize_prob_regions(Conus.href_cropped_5km_grid(), 0.02, "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp")
function rasterize_prob_regions(grid, threshold_prob, shapefile_path)

  mask = falses(length(grid.latlons))

  ArchGDAL.registerdrivers() do
    ArchGDAL.importEPSG(4326) do wgs84
    # ArchGDAL.importEPSG(4269) do nad83
      ArchGDAL.read(shapefile_path) do dataset
        println(shapefile_path)
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
            # DN field contains the probability level (in percent: 2, 5, 10 etc)
            dn =
              if isa(ArchGDAL.getfield(feature, "DN"), String)
                # println("DN: $(ArchGDAL.getfield(feature, "DN"))")
                parse(Int64, ArchGDAL.getfield(feature, "DN"))
              else
                ArchGDAL.getfield(feature, "DN")
              end
            if dn / 100 >= threshold_prob * 0.999
              geom = ArchGDAL.getgeom(feature)
              # println(geom)
              spatialref = ArchGDAL.getspatialref(geom)
              # println(spatialref)
              xys = latlons_on_spacial_ref(grid, spatialref)
              _add_geom_to_mask!(mask, xys, geom)
            end
          end
        end
      end
    end
  end

  mask
end

# ArchGDAL.intersects
# Slow as snot. quadtree speeds it a littl

mutable struct Quadtree
  on_edge   :: Bool
  inside    :: Bool # only valid if on_edge is false
  minX      :: Float64
  maxX      :: Float64
  minY      :: Float64
  maxY      :: Float64
  # Child quadrants
  midX      :: Float64
  midY      :: Float64
  top_left  :: Union{Quadtree,Nothing}
  top_right :: Union{Quadtree,Nothing}
  bot_left  :: Union{Quadtree,Nothing}
  bot_right :: Union{Quadtree,Nothing}
end

function make_rect(f, minX, maxX, minY, maxY)
  ArchGDAL.createpolygon(f, [
    [minX, maxY],
    [maxX, maxY],
    [maxX, minY],
    [minX, minY],
    [minX, maxY],
  ])
end

function make_quadree(geom, minX, maxX, minY, maxY, min_size)
  if min(maxX-minX, maxY-minY) < min_size
    return nothing
  end

  midX = 0.5*(minX + maxX)
  midY = 0.5*(minY + maxY)

  make_rect(minX, maxX, minY, maxY) do rect
    try
      if ArchGDAL.contains(geom, rect)
        Quadtree(false, true, minX, maxX, minY, maxY, midX, midY, nothing, nothing, nothing, nothing)
      elseif ArchGDAL.disjoint(geom, rect)
        Quadtree(false, false, minX, maxX, minY, maxY, midX, midY, nothing, nothing, nothing, nothing)
      else
        Quadtree(true, false, minX, maxX, minY, maxY, midX, midY,
          make_quadree(geom, minX, midX, midY, maxY, min_size),
          make_quadree(geom, midX, maxX, midY, maxY, min_size),
          make_quadree(geom, minX, midX, minY, midY, min_size),
          make_quadree(geom, midX, maxX, minY, midY, min_size)
        )
      end
    catch err
      if isa(err, ArchGDAL.GDAL.GDALError)
        println(geom)
        println(ArchGDAL.toWKT(geom))
        println(rect)
        rethrow()
      else
        rethrow()
      end
    end
  end
end

function test_point(x, y, point, geom)
  ArchGDAL.setpoint!(point, 0, x, y)
  ArchGDAL.contains(geom, point)
end

function in_geom_quadtree(tree, x, y, point, geom)
  if tree.on_edge
    if x <= tree.midX && y >= tree.midY
      isnothing(tree.top_left)  ? test_point(x, y, point, geom) : in_geom_quadtree(tree.top_left,  x, y, point, geom)
    elseif x >= tree.midX && y >= tree.midY
      isnothing(tree.top_right) ? test_point(x, y, point, geom) : in_geom_quadtree(tree.top_right, x, y, point, geom)
    elseif x <= tree.midX && y <= tree.midY
      isnothing(tree.bot_left)  ? test_point(x, y, point, geom) : in_geom_quadtree(tree.bot_left,  x, y, point, geom)
    else
      isnothing(tree.bot_right) ? test_point(x, y, point, geom) : in_geom_quadtree(tree.bot_right, x, y, point, geom)
    end
  else
    tree.inside
  end
end

# Mutates mask. On HREF grid, ~3-5s per geom.
function _add_geom_to_mask!(mask, xys, geom)
  bounds = ArchGDAL.getenvelope(geom)

  # print("making quadtree...")
  x1, y1 = xys[1]
  x2, y2 = xys[2]
  min_size = max(abs(x2-x1), abs(y2-y1)) * 3
  quadtree = make_quadree(geom, bounds.MinX, bounds.MaxX, bounds.MinY, bounds.MaxY, min_size)
  # println("done.")
  # println(quadtree)

  ArchGDAL.createpoint(0,0) do point
    for i in 1:length(xys)
      if mask[i]
        continue
      end

      x, y = xys[i]

      if x > bounds.MaxX || x < bounds.MinX || y > bounds.MaxY || y < bounds.MinY
        continue
      end

      # mask[i] = test_point(x, y, point, geom)
      mask[i] = isnothing(quadtree) ? test_point(x, y, point, geom) : in_geom_quadtree(quadtree, x, y, point, geom)
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
# grid = Conus.href_cropped_5km_grid()
# # shapefile_path = "geo_regions/ln_us/ln_us.shp"
# shapefile_path = "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp"
#
# @time mask = SPC.rasterize_prob_regions(grid, threshold, "day1otlk_20190812_1300-shp/day1otlk_20190812_1300_torn.shp")
#
# target_latlons = grid.latlons[mask]
#
# Grids.latlons_to_csv("tor_latlons.csv", target_latlons)

end
