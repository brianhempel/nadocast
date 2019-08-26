module MakeConusGrid

# This file is only to explain how to produce HREF_conus_latlons.csv
#
# "Vector > Geoprocessing Tools > Clip" seems to work in QGIS, if you can get the grid in and out.
#
# Make a Grid for QGIS:
#
# push!(LOAD_PATH, (@__DIR__) * "/../models/href_mid_2018_forward")
# import HREF
# import Grids
# Grids.latlons_to_csv("HREF_cropped_latlons.csv", HREF.original_grid_cropped())
#
# Open geo_regions/ln_us/ln_us.shp and HREF_cropped_latlons.csv in QGIS
# Vector > Geoprocessing Tools > Clip
# Layer > Save As (CSV)
#
#
# Then Conus.jl will do this:
#
# import DelimitedFiles
# latlons, _ = DelimitedFiles.readdlm("HREF_conus_latlons.csv", Float64; header = true)
# conus_mask = falses(length(HREF.original_grid_cropped().latlons))
# for (lat, lon) in eachrow(latlons)
#   conus_mask[Grids.latlon_to_closest_grid_i(HREF.original_grid_cropped(), (lat, lon))] = true
# end

end # module MakeConusGrid