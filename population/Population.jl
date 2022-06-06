# julia --project=.. Population.jl

# The ASCII grid files look like:

# ncols         8640
# nrows         4320
# xllcorner     -180
# yllcorner     -90
# cellsize      0.041666666666667
# NODATA_value  -9999
# -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 ...
# -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 ...
# -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 ...
# ...


function load_2pt5_min_data(zip_path) :: Matrix{Float32}
  grid_lines = map(strip, readlines(`unzip -p $zip_path $(replace(zip_path, "_asc.zip" => ".asc", "-" => "_"))`))

  # Convert to 2D array of Float32's, with -9999 recoded as 0
  vcat(
    map(grid_lines[7:length(grid_lines)]) do line
      transpose(map(cell -> max(0f0, parse(Float32, cell)), split(line)))
    end...
  )
end

data = load_2pt5_min_data("gpw-v4-population-density-rev11_2020_2pt5_min_asc.zip")

function lookup(data, lat, lon)
  x_per_cell = 360.0 / size(data, 2)
  y_per_cell = 180.0 / size(data, 1)

  x_i = round(Int64, (lon + 180.0) / x_per_cell, RoundDown) + 1
  y_i = size(data, 1) - round(Int64, (lat + 90.0)  / y_per_cell, RoundDown)

  # println((y_i, x_i))

  data[y_i, x_i]
end

import PNGFiles
using PNGFiles.ImageCore.ColorTypes
using PNGFiles.ImageCore.ColorTypes.FixedPointNumbers
PNGFiles.save("data_raw.png", Gray.(clamp.(data ./ 2_000,0.0,1.0)))


# The densities are based on the land area, not the gridpoint area
# So we have to convert the density back to gridpoint area

land_areas  = load_2pt5_min_data("gpw-v4-land-water-area-rev11_landareakm_2pt5_min_asc.zip")
water_areas = load_2pt5_min_data("gpw-v4-land-water-area-rev11_waterareakm_2pt5_min_asc.zip")

corrected = data .* land_areas ./ (land_areas .+ water_areas .+ eps(1f0))

PNGFiles.save("data_corrected.png", Gray.(clamp.(corrected ./ 2_000,0.0,1.0)))

PNGFiles.save("data_sqrt_corrected.png", Gray.(clamp.(sqrt.(corrected) ./ 100,0.0,1.0)))


push!(LOAD_PATH, (@__DIR__) * "/../lib")

# import GeoUtils
import Grib2
import Grids

# Same cropping and 3x downsampling as in HREF.jl
HREF_CROPPED_15KM_GRID =
  Grib2.read_grid(
    (@__DIR__) * "/../lib/href_one_field_for_grid.grib2",
    crop = ((1+214):(1473 - 99), (1+119):(1025-228)),
    downsample = 3
  ) :: Grids.Grid

pop_density_on_15km_grid = Vector{Float32}(undef, length(HREF_CROPPED_15KM_GRID.latlons))

height = HREF_CROPPED_15KM_GRID.height
width  = HREF_CROPPED_15KM_GRID.width
latlons = HREF_CROPPED_15KM_GRID.latlons


# 9x antialias because why not
# definitely need to do some antialiasing

# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ N ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ W ⋅ ⋅ ⋅ ⋅ • • • • X • • • • ⋅ ⋅ ⋅ ⋅ E ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ • • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ 0 • • • • • • • • ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ S ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅
# ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅ ⋅

const antialias_n = 9

for j in 1:height
  for i in 1:width
    flat_i = (j-1)*width + i
    # lat, lon = latlons[flat_i]

    # This is wrong on the edges, but the edges don't matter.
    wlatlon = i > 1      ? latlons[flat_i-1]     : latlons[flat_i] # W in diagram
    elatlon = i < width  ? latlons[flat_i+1]     : latlons[flat_i] # E in diagram
    slatlon = j > 1      ? latlons[flat_i-width] : latlons[flat_i] # S in diagram
    nlatlon = j < height ? latlons[flat_i+width] : latlons[flat_i] # N in diagram

    we_vec = elatlon .- wlatlon
    sn_vec = nlatlon .- slatlon

    # latlon vectors between the tiny dots in the diagram
    dx = we_vec ./ (antialias_n * 2)
    dy = sn_vec ./ (antialias_n * 2)

    # point 0 in diagram
    latlon0 = latlons[flat_i] .- (dx .* (antialias_n / 2 - 0.5)) .- (dy .* (antialias_n / 2 - 0.5))

    out = 0f0
    for x_i in 0:(antialias_n - 1)
      for y_i in 0:(antialias_n - 1)
        y, x = latlon0 .+ (dx .* x_i) .+ (dy .* y_i)
        out += lookup(corrected, y, x)
        # if flat_i == 30_000
        #   println((y, x)) # to check in grib viewer
        # end
      end
    end

    # if flat_i == 30_000
    #   exit(1)
    # end

    pop_density_on_15km_grid[flat_i] = out / antialias_n / antialias_n
  end
end

write("pop_density_on_15km_grid.float16.bin", Float16.(pop_density_on_15km_grid))


import Dates

# So I can check in a grib viewer if it looks right
# The Grib2 quantization has a habit of over-quantizing the low density regions, so DO NOT USE this. Use the Float16 version above.
# The Grib2 only has 406 discrete levels, compared to 17861 for the Float16 version.
Grib2.write_15km_HREF_probs_grib2(
  pop_density_on_15km_grid;
  run_time = Dates.DateTime(2020,1,1,0,0,0),
  forecast_hour = 0,
  event_type = "tornado",
  out_name = "pop_density_on_15km_grid_but_do_not_use.grib2",
)

# Grib2.write_15km_HREF_probs_grib2(
#   map(yx -> lookup(corrected, yx[1], yx[2]), latlons);
#   run_time = Dates.DateTime(2020,1,1,0,0,0),
#   forecast_hour = 0,
#   event_type = "tornado",
#   out_name = "pop_density_on_15km_grid_no_aa.grib2",
# )


PNGFiles.save("data_corrected_on_15km_grid.png", Gray.(clamp.(transpose(reshape(pop_density_on_15km_grid, (width, height)))[height:-1:1,:] ./ 2_000,0.0,1.0)))
PNGFiles.save("data_corrected_on_15km_grid_log2.png", Gray.(clamp.(transpose(reshape(log.(pop_density_on_15km_grid) ./ log(2), (width, height)))[height:-1:1,:] ./ maximum(log.(pop_density_on_15km_grid) ./ log(2)),0.0,1.0)))