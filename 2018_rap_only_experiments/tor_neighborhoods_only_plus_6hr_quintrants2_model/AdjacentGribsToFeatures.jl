# Evolution of read_grib.jl
# Provide two adjacent pressure-level RAP grib2s (e.g. +5 hr and +6 hr forecasts)

# # RUC has an HLCY coded as "surface" but is probably 0-3km helicity per https://ruc.noaa.gov/ruc/fslparms/13km/ruc13_grib.table-nov2008 and http://www.nws.noaa.gov/om/tpb/448body.htm (surface doesn't really make sense)
# # Same for UV of storm motion.
# # CAPE 255-0mb is MUCAPE https://ruc.noaa.gov/forum/f2/Welcome.cgi/read/2145
# #
# # 4LFTX and LFTX ??
# #
# # Low-level (180-0 mb agl, 90-0 mb agl) CAPE/CIN not added until 2014-02-25-1200. How to handle?? (Estimate for old data??)
# #
# # For now, simply only using data after 2014-02-25-1200
#


# Following fields exluded b/c they are cumulative:

# SSRUN:surface
# NCPCP:surface
# BGRUN:surface
# WEASD:surface
# ACPCP:surface

# It *does* appear that there are multiple accumulation periods in the forecast (i.e. always a 1-hour period somewhere for all of the above)
# May look for that.

# ARGS = ["test_grib2s/rap_130_20170516_1700_005.grb2", "test_grib2s/rap_130_20170516_1700_006.grb2"]

if length(ARGS) < 2
  error("must provide a grib2 file to read")
end

IS_TRAINING = !(length(ARGS) > 2 && ARGS[3] == "--all")

TORNADO_NEIGHBORHOODS_ONLY = length(ARGS) > 2 && ARGS[3] == "--250mi_tornado_neighborhoods"

println("Importing libraries...")

push!(LOAD_PATH, ".")

import GeoUtils
import TimeZones

prior_forecast_hour_grib2_path = ARGS[1] # "rap_130_20170516_1700_005.grb2"
grib2_path                     = ARGS[2] # "rap_130_20170516_1700_006.grb2"

grib2_file_name = last(split(grib2_path, "/"))

# grib2_winds_latlon_aligned_path = "tmp/winds_latlon_aligned_tmp_$grib2_file_name"

out_path = "tmp/$(IS_TRAINING ? "training_" : "")$(replace(grib2_file_name, ".grb2", "")).bindata"


forecast_hour_str = match(r"(\d\d)\.gri?b2", grib2_file_name).captures[1]

forecast_hour = parse(Int64, forecast_hour_str)

if IS_TRAINING
  println("Preparing time...")

  utc = TimeZones.tz"UTC"

  year_str, month_str, day_str, run_hour_str, _ = match(r"_130_(\d\d\d\d)(\d\d)(\d\d)_(\d\d)00_(\d\d\d)\.grb2", grib2_file_name).captures

  valid_time       = TimeZones.ZonedDateTime(parse(Int64,year_str),parse(Int64,month_str),parse(Int64,day_str),parse(Int64,run_hour_str),0,0, utc) + Dates.Hour(forecast_hour)
  valid_start_time = valid_time - Dates.Minute(30)
  valid_end_time   = valid_time + Dates.Minute(30)

  println(valid_start_time," to ",valid_end_time)
end


stdout_limited = IOContext(STDOUT, :display_size=>(100,60))
stdout_limited = IOContext(stdout_limited, :limit=>true)

### Common Fields ###
# readdlm conveniently filters out lines starting with "#"
layers = readdlm(IOBuffer("""4LFTX:180-0 mb above ground
ABSV:500 mb
CAPE:180-0 mb above ground
CAPE:255-0 mb above ground
CAPE:90-0 mb above ground
CAPE:surface
# CFRZR:surface
# CICEP:surface
CIN:180-0 mb above ground
CIN:255-0 mb above ground
CIN:90-0 mb above ground
CIN:surface
CRAIN:surface
# CSNOW:surface
DEPR:2 m above ground
DPT:2 m above ground
EPOT:surface
GUST:surface
HCDC:high cloud layer
# HGT:0C isotherm
HGT:100 mb
HGT:1000 mb
# HGT:125 mb
HGT:150 mb
# HGT:175 mb
HGT:200 mb
HGT:225 mb
# HGT:250 mb
# HGT:275 mb
HGT:300 mb
# HGT:325 mb
# HGT:350 mb
# HGT:375 mb
HGT:400 mb
# HGT:425 mb
# HGT:450 mb
# HGT:475 mb
HGT:500 mb
# HGT:525 mb
# HGT:550 mb
# HGT:575 mb
HGT:600 mb
# HGT:625 mb
# HGT:650 mb
# HGT:675 mb
HGT:700 mb
# HGT:725 mb
# HGT:750 mb
# HGT:775 mb
HGT:800 mb
# HGT:825 mb
HGT:850 mb
# HGT:875 mb
HGT:900 mb
# HGT:925 mb
HGT:950 mb
# HGT:975 mb
HGT:cloud base
HGT:cloud top
HGT:convective cloud top level
HGT:equilibrium level
# HGT:highest tropospheric freezing level
HGT:lowest level of the wet bulb zero
HGT:planetary boundary layer
HGT:surface
HLCY:1000-0 m above ground
HLCY:3000-0 m above ground
HPBL:surface
LCDC:low cloud layer
LFTX:500-1000 mb
LTNG:surface
MCDC:middle cloud layer
MSLMA:mean sea level
MSTAV:0 m underground
PLPL:255-0 mb above ground
POT:2 m above ground
POT:tropopause
PRATE:surface
# PRES:0C isotherm
PRES:80 m above ground
# PRES:highest tropospheric freezing level
PRES:max wind
PRES:surface
PRES:tropopause
PWAT:entire atmosphere (considered as a single layer)
REFC:entire atmosphere (considered as a single layer)
REFD:1000 m above ground
REFD:4000 m above ground
RETOP:entire atmosphere (considered as a single layer)
RH:0C isotherm
RH:100 mb
RH:1000 mb
RH:120-90 mb above ground
# RH:125 mb
RH:150 mb
RH:150-120 mb above ground
# RH:175 mb
RH:180-150 mb above ground
# RH:2 m above ground
RH:200 mb
# RH:225 mb
RH:250 mb
# RH:275 mb
RH:30-0 mb above ground
RH:300 mb
# RH:325 mb
RH:350 mb
# RH:375 mb
RH:400 mb
# RH:425 mb
RH:450 mb
# RH:475 mb
RH:500 mb
# RH:525 mb
RH:550 mb
# RH:575 mb
RH:60-30 mb above ground
RH:600 mb
# RH:625 mb
RH:650 mb
# RH:675 mb
RH:700 mb
# RH:725 mb
RH:750 mb
# RH:775 mb
RH:800 mb
# RH:825 mb
RH:850 mb
# RH:875 mb
RH:90-60 mb above ground
RH:900 mb
# RH:925 mb
RH:950 mb
# RH:975 mb
RH:highest tropospheric freezing level
# SNOD:surface
SPFH:2 m above ground
SPFH:80 m above ground
TCDC:entire atmosphere (considered as a single layer)
TMP:100 mb
# TMP:1000 mb
# TMP:120-90 mb above ground
TMP:125 mb
TMP:150 mb
TMP:150-120 mb above ground
TMP:175 mb
TMP:180-150 mb above ground
# TMP:2 m above ground
TMP:200 mb
# TMP:225 mb
TMP:250 mb
# TMP:275 mb
TMP:30-0 mb above ground
TMP:300 mb
# TMP:325 mb
TMP:350 mb
# TMP:375 mb
TMP:400 mb
# TMP:425 mb
TMP:450 mb
# TMP:475 mb
TMP:500 mb
# TMP:525 mb
TMP:550 mb
# TMP:575 mb
TMP:60-30 mb above ground
TMP:600 mb
# TMP:625 mb
TMP:650 mb
# TMP:675 mb
TMP:700 mb
# TMP:725 mb
TMP:750 mb
# TMP:775 mb
TMP:80 m above ground
TMP:800 mb
# TMP:825 mb
TMP:850 mb
# TMP:875 mb
TMP:90-60 mb above ground
TMP:900 mb
# TMP:925 mb
TMP:950 mb
# TMP:975 mb
TMP:surface
TMP:tropopause
# UGRD:10 m above ground
UGRD:100 mb
UGRD:1000 mb
UGRD:120-90 mb above ground
# UGRD:125 mb
UGRD:150 mb
UGRD:150-120 mb above ground
# UGRD:175 mb
UGRD:180-150 mb above ground
UGRD:200 mb
# UGRD:225 mb
UGRD:250 mb
# UGRD:275 mb
UGRD:30-0 mb above ground
UGRD:300 mb
# UGRD:325 mb
UGRD:350 mb
# UGRD:375 mb
UGRD:400 mb
# UGRD:425 mb
UGRD:450 mb
# UGRD:475 mb
UGRD:500 mb
# UGRD:525 mb
UGRD:550 mb
# UGRD:575 mb
UGRD:60-30 mb above ground
UGRD:600 mb
# UGRD:625 mb
UGRD:650 mb
# UGRD:675 mb
UGRD:700 mb
# UGRD:725 mb
UGRD:750 mb
# UGRD:775 mb
UGRD:80 m above ground
UGRD:800 mb
# UGRD:825 mb
UGRD:850 mb
# UGRD:875 mb
UGRD:90-60 mb above ground
UGRD:900 mb
# UGRD:925 mb
UGRD:950 mb
# UGRD:975 mb
UGRD:max wind
UGRD:tropopause
USTM:u storm motion
# VGRD:10 m above ground
VGRD:100 mb
VGRD:1000 mb
VGRD:120-90 mb above ground
# VGRD:125 mb
VGRD:150 mb
VGRD:150-120 mb above ground
# VGRD:175 mb
VGRD:180-150 mb above ground
VGRD:200 mb
# VGRD:225 mb
VGRD:250 mb
# VGRD:275 mb
VGRD:30-0 mb above ground
VGRD:300 mb
# VGRD:325 mb
VGRD:350 mb
# VGRD:375 mb
VGRD:400 mb
# VGRD:425 mb
VGRD:450 mb
# VGRD:475 mb
VGRD:500 mb
# VGRD:525 mb
VGRD:550 mb
# VGRD:575 mb
VGRD:60-30 mb above ground
VGRD:600 mb
# VGRD:625 mb
VGRD:650 mb
# VGRD:675 mb
VGRD:700 mb
# VGRD:725 mb
VGRD:750 mb
# VGRD:775 mb
VGRD:80 m above ground
VGRD:800 mb
# VGRD:825 mb
VGRD:850 mb
# VGRD:875 mb
VGRD:90-60 mb above ground
VGRD:900 mb
# VGRD:925 mb
VGRD:950 mb
# VGRD:975 mb
VGRD:max wind
VGRD:tropopause
VIS:surface
VSTM:v storm motion
VUCSH:6000-0 m above ground
VVCSH:6000-0 m above ground
VVEL:100 mb
VVEL:1000 mb
# VVEL:120-90 mb above ground
# VVEL:125 mb
# VVEL:150 mb
# VVEL:150-120 mb above ground
# VVEL:175 mb
# VVEL:180-150 mb above ground
VVEL:200 mb
# VVEL:225 mb
# VVEL:250 mb
# VVEL:275 mb
VVEL:30-0 mb above ground
VVEL:300 mb
# VVEL:325 mb
# VVEL:350 mb
# VVEL:375 mb
VVEL:400 mb
# VVEL:425 mb
# VVEL:450 mb
# VVEL:475 mb
VVEL:500 mb
# VVEL:525 mb
# VVEL:550 mb
# VVEL:575 mb
# VVEL:60-30 mb above ground
VVEL:600 mb
# VVEL:625 mb
# VVEL:650 mb
# VVEL:675 mb
VVEL:700 mb
# VVEL:725 mb
# VVEL:750 mb
# VVEL:775 mb
VVEL:800 mb
# VVEL:825 mb
# VVEL:850 mb
# VVEL:875 mb
# VVEL:90-60 mb above ground
VVEL:900 mb
# VVEL:925 mb
VVEL:950 mb
# VVEL:975 mb"""), ':', String; header=false, use_mmap=false, quotes=false) :: Array{String,2}

uv_layers = [
  # ("UGRD:10 m above ground",       "VGRD:10 m above ground"),
  ("UGRD:80 m above ground",       "VGRD:80 m above ground"),
  ("UGRD:100 mb",                  "VGRD:100 mb"),
  ("UGRD:120-90 mb above ground",  "VGRD:120-90 mb above ground"),
  # ("UGRD:125 mb",                  "VGRD:125 mb"),
  ("UGRD:150 mb",                  "VGRD:150 mb"),
  ("UGRD:150-120 mb above ground", "VGRD:150-120 mb above ground"),
  # ("UGRD:175 mb",                  "VGRD:175 mb"),
  ("UGRD:180-150 mb above ground", "VGRD:180-150 mb above ground"),
  ("UGRD:200 mb",                  "VGRD:200 mb"),
  # ("UGRD:225 mb",                  "VGRD:225 mb"),
  ("UGRD:250 mb",                  "VGRD:250 mb"),
  # ("UGRD:275 mb",                  "VGRD:275 mb"),
  ("UGRD:30-0 mb above ground",    "VGRD:30-0 mb above ground"),
  ("UGRD:300 mb",                  "VGRD:300 mb"),
  # ("UGRD:325 mb",                  "VGRD:325 mb"),
  ("UGRD:350 mb",                  "VGRD:350 mb"),
  # ("UGRD:375 mb",                  "VGRD:375 mb"),
  ("UGRD:400 mb",                  "VGRD:400 mb"),
  # ("UGRD:425 mb",                  "VGRD:425 mb"),
  ("UGRD:450 mb",                  "VGRD:450 mb"),
  # ("UGRD:475 mb",                  "VGRD:475 mb"),
  ("UGRD:500 mb",                  "VGRD:500 mb"),
  # ("UGRD:525 mb",                  "VGRD:525 mb"),
  ("UGRD:550 mb",                  "VGRD:550 mb"),
  # ("UGRD:575 mb",                  "VGRD:575 mb"),
  ("UGRD:60-30 mb above ground",   "VGRD:60-30 mb above ground"),
  ("UGRD:600 mb",                  "VGRD:600 mb"),
  # ("UGRD:625 mb",                  "VGRD:625 mb"),
  ("UGRD:650 mb",                  "VGRD:650 mb"),
  # ("UGRD:675 mb",                  "VGRD:675 mb"),
  ("UGRD:700 mb",                  "VGRD:700 mb"),
  # ("UGRD:725 mb",                  "VGRD:725 mb"),
  ("UGRD:750 mb",                  "VGRD:750 mb"),
  # ("UGRD:775 mb",                  "VGRD:775 mb"),
  ("UGRD:800 mb",                  "VGRD:800 mb"),
  # ("UGRD:825 mb",                  "VGRD:825 mb"),
  ("UGRD:850 mb",                  "VGRD:850 mb"),
  # ("UGRD:875 mb",                  "VGRD:875 mb"),
  ("UGRD:90-60 mb above ground",   "VGRD:90-60 mb above ground"),
  ("UGRD:900 mb",                  "VGRD:900 mb"),
  # ("UGRD:925 mb",                  "VGRD:925 mb"),
  ("UGRD:950 mb",                  "VGRD:950 mb"),
  # ("UGRD:975 mb",                  "VGRD:975 mb"),
  ("UGRD:1000 mb",                 "VGRD:1000 mb"),
  ("UGRD:max wind",                "VGRD:max wind"),
  ("UGRD:tropopause",              "VGRD:tropopause"),
  ("USTM:u storm motion",          "VSTM:v storm motion"),
  ("VUCSH:6000-0 m above ground",  "VVCSH:6000-0 m above ground")
] :: Array{Tuple{String,String},1}



println("Reading grid latlons...")

# Figure out latlons for grid points

all_pts = open(grid -> readdlm(grid, ','; header=false), `wgrib2 $grib2_path -end -inv /dev/null -gridout -`)

all_pts[:, 4] = [lon > 180 ? lon - 360 : lon for lon in all_pts[:, 4]]

# Grid is W to E, S to N

const grid_width  = Int64(maximum(all_pts[:,1]))
const grid_height = Int64(maximum(all_pts[:,2]))

println("min lat ",minimum(all_pts[:,3]))
println("max lat ",maximum(all_pts[:,3]))
println("min lon ",minimum(all_pts[:,4]))
println("max lon ",maximum(all_pts[:,4]))

const grid_lat_lons = map(flat_i -> (all_pts[flat_i,3], all_pts[flat_i,4]), 1:(grid_width*grid_height)) :: Array{Tuple{Float64,Float64},1}

function get_grid_i(w_to_e_col :: Int64, s_to_n_row :: Int64) :: Int64
  if w_to_e_col < 1
    error("Error indexing into grid, asked for column $w_to_e_col")
  elseif w_to_e_col > grid_width
    error("Error indexing into grid, asked for column $w_to_e_col")
  elseif s_to_n_row < 1
    error("Error indexing into grid, asked for row $s_to_n_row")
  elseif s_to_n_row > grid_height
    error("Error indexing into grid, asked for row $s_to_n_row")
  end
  grid_width*(s_to_n_row-1) + w_to_e_col
end

function is_on_grid(w_to_e_col :: Int64, s_to_n_row :: Int64) :: Bool
  if w_to_e_col < 1
    false
  elseif w_to_e_col > grid_width
    false
  elseif s_to_n_row < 1
    false
  elseif s_to_n_row > grid_height
    false
  else
    true
  end
end

# function get_grid_lat_lon_for_flat_i(flat_i :: Int64) :: Tuple{Float64,Float64}
#   lat = all_pts[flat_i,3]
#   lon = all_pts[flat_i,4]
#   (lat, lon)
# end

function get_grid_lat_lon_and_flat_i(w_to_e_col :: Int64, s_to_n_row :: Int64) :: Tuple{Float64,Float64,Int64}
  flat_i   = get_grid_i(w_to_e_col, s_to_n_row)
  lat, lon = grid_lat_lons[flat_i]
  (lat, lon, flat_i)
end


# The first time we use broadcast function like .== or .> we seem to pay a considerable JIT cost or something.
# The below three functions are thus faster.

function find_i(arr, target)
  for i = 1:length(arr)
    if arr[i] == target
      return i
    end
  end
  return -1
end

function broadcast_gt(arr, cutoff)
  [x > cutoff for x=arr]
end

function broadcast_lt(arr, cutoff)
  [x < cutoff for x=arr]
end


# It's actually not correct to align winds to lat/lon to average them.
#
# I think the correct way would be to average the rotation vectors at the center of the earth.
#
# But we need lat/lon aligned winds so we can follow the storm motion. So let's roll with it for now.

println("Aligning winds to lat/lon...")

# Not going to bother with points on the top/bottom row.

println("Finding rotation angles...")

# Elements of rotation matrices
coses = ones(Float32, grid_width*grid_height)
sines = zeros(Float32, grid_width*grid_height)

for j = 2:(grid_height-1)
  for i = 1:grid_width
    lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
    s_lat, s_lon, s_flat_i = get_grid_lat_lon_and_flat_i(i, j-1)
    n_lat, n_lon, n_flat_i = get_grid_lat_lon_and_flat_i(i, j+1)

    n_angle, _ = GeoUtils.azimuths(lat, lon, n_lat, n_lon)
    _, s_angle = GeoUtils.azimuths(s_lat, s_lon, lat, lon)

    flat_i = get_grid_i(i, j)

    angle = -(n_angle + s_angle) / 2.0 / 180.0 * π

    coses[flat_i] = cos(angle)
    sines[flat_i] = sin(angle)
  end
end

# Verified against wgrib2's rotation and we're within 5%.
function rotate_grid_winds_to_lat_lon(us, vs)
  lon_us = (us .* coses) .- (vs .* sines)
  lat_vs = (us .* sines) .+ (vs .* coses)

  (lon_us, lat_vs)
end

# Normalize wind angle relative to storm motion (convert to polar: ground speed + relative direction)

# # UV to polar
# function uv_to_r_theta(pair::Tuple{Float32,Float32})::Tuple{Float32,Float32}
#   u, v = pair
#   u_v_to_r_theta(u, v)
# end

function u_v_to_r_theta(u::Float32, v::Float32)::Tuple{Float32,Float32}
  if u == 0.0 && v == 0.0
    return (0.0, 0.0)
  end
  r     = sqrt(u^2 + v^2)
  theta = atan2(v, u) # Angle, in radians
  theta = mod(theta + π, 2π) - π # Turns π into -π (otherwise a passthrough here)
  (Float32(r), Float32(theta))
end



println("Reading inventory lines...")

function normalize_abbrev_and_desc(abbrev, desc)
  if abbrev == "MSTAV"
    return ("MSTAV", "0 m underground")
  elseif abbrev == "USTM"
    return ("USTM", "u storm motion")
  elseif abbrev == "VSTM"
    return ("VSTM", "v storm motion")
  elseif abbrev == "HLCY" && desc == "surface"
    return ("HLCY", "3000-0 m above ground")
  end
  if abbrev == "DIST"
    abbrev = "HGT"
  end
  desc = replace(desc, "0-6000 m above ground", "6000-0 m above ground")
  desc = replace(desc, r"\bentire atmosphere\z", "entire atmosphere (considered as a single layer)")
  desc = replace(desc, r"\b3 m underground\b", "surface") # BGRUN:3 m underground -> BGRUN:surface

  return (abbrev, desc)
end

function normalize_line(row)
  abbrev, desc = normalize_abbrev_and_desc(row[4], row[5])
  row[4:5] = [abbrev, desc]
  return row
end

function desiredLayerToInventoryLine(forecast_hour, inventory_lines_normalized, desired)
  desired_abbrev = desired[1]
  desired_desc   = desired[2]
  desired_hour   = "$forecast_hour hour fcst"
  # I hate this language.
  matching_i =
    findfirst([[desired_abbrev, desired_desc, desired_hour] == inventory_lines_normalized[i,4:6] for i=1:size(inventory_lines_normalized,1)])

  if matching_i > 0
    return inventory_lines_normalized[matching_i, :]
  else
    error("Could not find inventory_lines_normalized row for $desired_abbrev $desired_desc $desired_hour")
  end
end


function get_raw_data(grib2_path, forecast_hour, layers, uv_layers)
  # Read the inventory of the file so we can corrolate it to figure out which layers we want.
  inventory_lines = open(`wgrib2 $grib2_path -s -n`) do inv
    readdlm(inv, ':', String; header=false, use_mmap=false, quotes=false)
  end

  # show(stdout_limited, "text/plain", inventory_lines)

  println("Fetching desired layers...")

  inventory_lines_normalized =
    mapslices(normalize_line, inventory_lines, 2)

  # show(stdout_limited, "text/plain", inventory_lines_normalized)

  layers_to_fetch = mapslices(desired -> desiredLayerToInventoryLine(forecast_hour, inventory_lines_normalized, desired), layers, 2)

  # show(stdout_limited, "text/plain", layers_to_fetch)

  layer_to_data = Dict{String,Array{Float32}}()

  # If you don't redirect inventory to /dev/null, it goes to stdout. No way to turn inventory off.
  (from_wgrib2, to_wgrib2, wgrib2) = readandwrite(`wgrib2 $grib2_path -i -header -inv /dev/null -bin -`)

  # Tell wgrib2 which layers we want.
  for layer_i = 1:size(layers_to_fetch,1)
    layer_to_fetch = layers_to_fetch[layer_i, :]
    # Only need first two columns (message.submessage and position) plus newline
    println(to_wgrib2, layer_to_fetch[1] * ":" * layer_to_fetch[2])
  end
  close(to_wgrib2)

  # Read out the data in those layers.
  for layer_i = 1:size(layers_to_fetch,1)
    layer_to_fetch    = layers_to_fetch[layer_i, :]
    abbrev, desc      = normalize_abbrev_and_desc(layer_to_fetch[4], layer_to_fetch[5])
    layer_key         = abbrev * ":" * desc
    grid_length       = read(from_wgrib2, UInt32)
    values            = read(from_wgrib2, Float32, div(grid_length, 4))
    grid_length_again = read(from_wgrib2, UInt32)

    if desc == "cloud base" || desc == "cloud top" || abbrev == "RETOP"
      # Handle undefined
      values = map((v -> (v > 25000.0f0 || v < -1000.0f0) ? 25000.0f0 : v), values)
    end
    # println(grid_length)
    # println(values)
    layer_to_data[layer_key] = values
    # println(grid_length_again)
  end

  # Sanity check that incoming stream is empty
  if !eof(from_wgrib2)
    error("wgrib2 sending more data than expected!")
  end

  # show(stdout_limited, "text/plain", layer_to_data["USTM:u storm motion"][grid_width:(grid_width*grid_height - grid_width)])
  # show(stdout_limited, "text/plain", layer_to_data["VSTM:v storm motion"][grid_width:(grid_width*grid_height - grid_width)])


  println("Rotating...")

  for uv_layer in uv_layers
    u_layer_key, v_layer_key = uv_layer

    rotated_us, rotated_vs = rotate_grid_winds_to_lat_lon(layer_to_data[u_layer_key], layer_to_data[v_layer_key])

    layer_to_data[u_layer_key] = rotated_us
    layer_to_data[v_layer_key] = rotated_vs
  end

  # open(`rm $grib2_winds_latlon_aligned_path`)

  println("Consolidating data...")
  regular_layer_order = Tuple{String,String}[]

  consolidated_data = zeros(Float32, size(layers,1), grid_width*grid_height)
  i = 1

  for layer_i in 1:size(layers,1)
    abbrev, desc = layers[layer_i,:]

    if abbrev == "UGRD" || abbrev == "USTM" || abbrev == "VUCSH"
    elseif abbrev == "VGRD" || abbrev == "VSTM" || abbrev == "VVCSH"
    else
      layer_key = abbrev * ":" * desc
      push!(regular_layer_order, (abbrev, desc))
      consolidated_data[i,:] = layer_to_data[layer_key]
      i += 1
    end
  end

  wind_layer_order = Tuple{String,String}[]

  const first_wind_layer_i = i

  storm_motion_us_j = -1
  storm_motion_vs_j = -1

  for layer_i in 1:size(layers,1)
    abbrev, desc = layers[layer_i,:]

    # print("$abbrev ")

    if abbrev == "UGRD" || abbrev == "USTM" || abbrev == "VUCSH"
      u_abbrev, u_desc = abbrev :: String, desc :: String
      v_abbrev = abbrev == "VUCSH" ? "VVCSH" : "V" * u_abbrev[2:4]
      v_desc   = abbrev == "USTM" ? "v storm motion" : u_desc

      u_layer_key = u_abbrev * ":" * u_desc
      v_layer_key = v_abbrev * ":" * v_desc

      if abbrev == "USTM"
        storm_motion_us_j = i
        storm_motion_vs_j = i +1
      end

      consolidated_data[i,:] = layer_to_data[u_layer_key]
      i += 1

      consolidated_data[i,:] = layer_to_data[v_layer_key]
      i += 1

      push!(wind_layer_order, (u_abbrev, u_desc))

    elseif abbrev == "VGRD" || abbrev == "VSTM" || abbrev == "VVCSH"
    else
    end
  end

  (consolidated_data, regular_layer_order, wind_layer_order, storm_motion_us_j, storm_motion_vs_j)
end


consolidated_data, regular_layer_order, wind_layer_order, storm_motion_us_j, storm_motion_vs_j = get_raw_data(grib2_path, forecast_hour, layers, uv_layers)
prior_hour_consolidated_data, _, _, _, _                                                       = get_raw_data(prior_forecast_hour_grib2_path, forecast_hour-1, layers, uv_layers)




if IS_TRAINING
  println("Loading training grid points set...")

  # Figure out which grid points are for training

  training_pts = readdlm("grid_xys_26_miles_inside_1_mile_outside_conus.csv", ','; header=false)[:, 1:2]
  training_pts_set = Set{Tuple{Int32,Int32}}(Set())

  function lat_lon_to_key(lat :: Float64, lon :: Float64)
    (Int32(round(lat*1000)), Int32(round(lon*1000)))
  end

  for i in 1:size(training_pts,1)
    push!(training_pts_set, lat_lon_to_key(training_pts[i,2], training_pts[i,1]))
  end

  train_pts = all_pts[Bool[(lat_lon_to_key(all_pts[i, 3], all_pts[i,4]) in training_pts_set) for i in 1:size(all_pts,1)], :]

  println("min training lat ",minimum(train_pts[:,3]))
  println("max training lat ",maximum(train_pts[:,3]))
  println("min training lon ",minimum(train_pts[:,4]))
  println("max training lon ",maximum(train_pts[:,4]))

  if size(train_pts,1) != length(training_pts_set)
    error("Grid error: grid used in $grib2_path does not match grid used to determine training points")
  end


  println("Estimating point areas...")

  # Estimate area represented by each grid point

  point_areas = zeros(Float32, grid_width*grid_height,1)

  max_point_area = 0.0f0

  for j = 1:grid_height
    for i = 1:grid_width
      lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
      # We should never need the area weights on the edges of the grid, but so any kind of handling here is okay.
      wlon = i > 1           ? all_pts[flat_i-1,4]          : lon
      elon = i < grid_width  ? all_pts[flat_i+1,4]          : lon
      slat = j > 1           ? all_pts[flat_i-grid_width,3] : lat
      nlat = j < grid_height ? all_pts[flat_i+grid_width,3] : lat

      w_distance = GeoUtils.distance(lat, lon, lat, wlon) / 2.0 / GeoUtils.METERS_PER_MILE
      e_distance = GeoUtils.distance(lat, lon, lat, elon) / 2.0 / GeoUtils.METERS_PER_MILE
      s_distance = GeoUtils.distance(lat, lon, slat, lon) / 2.0 / GeoUtils.METERS_PER_MILE
      n_distance = GeoUtils.distance(lat, lon, nlat, lon) / 2.0 / GeoUtils.METERS_PER_MILE

      sw_area = w_distance * s_distance
      se_area = e_distance * s_distance
      nw_area = w_distance * n_distance
      ne_area = e_distance * n_distance

      point_areas[flat_i] = sw_area + se_area + nw_area + ne_area

      if lat_lon_to_key(lat, lon) in training_pts_set && point_areas[flat_i] > max_point_area
        max_point_area = point_areas[flat_i]
      end
    end
  end

  point_weights = point_areas / max_point_area

  # Transpose features to row per point

  # headers = ["i", "j", "lat", "lon", "sq_miles"]
  #
  # tranposed_data = zeros(Float32, size(all_pts,1), size(all_pts,2) + size(layers,1))
  # tranposed_data[:,1:size(all_pts,2)] = all_pts
  #
  # out_col = length(headers) + 1
  # for layer_i in 1:size(layers,1)
  #   abbrev, desc = layers[layer_i,:]
  #   # println((abbrev, desc))
  #   # desc   = layer[layer_i][2]
  #   if abbrev == "UGRD"
  #     abbrev = "RGRD"
  #   elseif abbrev == "VGRD"
  #     abbrev = "TGRD"
  #   elseif abbrev == "USTM"
  #     abbrev = "RSTM"
  #     desc   = "latlon relative storm motion speed"
  #   end
  #
  #   # if abbrev == "VSTM"
  #   #   # Don't include absolute direction as a feature.
  #   # else
  #   layer_key = abbrev * ":" * desc
  #   push!(headers, layer_key)
  #   tranposed_data[:, out_col] = layer_to_data[layer_key]
  #   out_col += 1
  #   # end
  # end
  #
  # println("headers")
  # show(stdout_limited, "text/plain", headers)
  # println("tranposed_data")
  # show(stdout_limited, "text/plain", tranposed_data)


  # Build final feature set

  # Find min/max/mean/start-end-diff within 25mi of +/- 30 min storm motion

  # function relativize_angle(thetaAndRef)
  #   theta, ref = thetaAndRef
  #   mod(theta - ref + π, 2π) - π
  # end

  # relative_thetas = map(thetaAndRef -> relativize_angle(thetaAndRef), zip(thetas, storm_motion_thetas))
else
  training_pts_set = Set{Tuple{Int32,Int32}}(Set())
  point_weights    = ones(Float32, grid_width*grid_height,1)
end

const ONE_MINUTE = 60.0
const repeated = Base.Iterators.repeated

# Search outward in squre rings until no more points in region are found.
#
# Works for any predicate region that is:
#  (a) fully connected (not split into disconnected regions)
#  (b) no "skinny parts" that could slip between grid points
#
# Circles and extruded circles are good. (What we use.)
function square_ring_search(predicate, center_i :: Int64, center_j :: Int64) :: Array{Int64,1}
  any_found_this_ring     = false
  still_searching_on_grid = true
  matching_flat_is = []

  r = 0
  while still_searching_on_grid && (isempty(matching_flat_is) || any_found_this_ring)
    any_found_this_ring     = false
    still_searching_on_grid = false

    # search in this order:
    #
    # 4 3 3 3 3
    # 4       2
    # 4   •   2
    # 4       2
    # 1 1 1 1 2

    if r == 0
      ring = [(center_i, center_j)]
    else
      sw_se = zip(center_i-r:center_i+r-1, repeated(center_j-r))
      se_ne = zip(repeated(center_i+r), center_j-r:center_j+r-1)
      ne_nw = zip(center_i+r:-1:center_i-r+1, repeated(center_j+r))
      nw_sw = zip(repeated(center_i-r), center_j+r:-1:center_j-r+1)
      ring  = collect(Base.Iterators.flatten((sw_se, se_ne, ne_nw, nw_sw)))
    end

    for ring_i = 1:length(ring)
      i, j = ring[ring_i]
      if is_on_grid(i, j)
        still_searching_on_grid = true
        lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
        if predicate(lat, lon)
          any_found_this_ring = true
          push!(matching_flat_is, flat_i)
        end
      end
    end

    r += 1
  end

  # print(" $r ")

  matching_flat_is
end

# A little faster. (Each layer increases by 4 grid points rather than 8, so we have more opportunities to terminate.)
function diamond_search(predicate, center_i :: Int64, center_j :: Int64) :: Array{Int64,1}
  any_found_this_diamond  = false
  still_searching_on_grid = true
  matching_flat_is = []

  r = 0 :: Int64
  # rs = Int64[0]
  while still_searching_on_grid && (isempty(matching_flat_is) || any_found_this_diamond)
    any_found_this_diamond  = false
    still_searching_on_grid = false

    # search in this order:
    #
    #     3
    #   3   2
    # 4   •   2
    #   4   1
    #     1

    if r == 0
      # diamond = [(center_i, center_j)]
      i, j = center_i, center_j
      if is_on_grid(i, j)
        still_searching_on_grid = true
        lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
        if predicate(lat, lon)
          any_found_this_diamond = true
          push!(matching_flat_is, flat_i)
        end
      end
    else
      for k = 0:(r-1)
        i, j = center_i+k, center_j-r+k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
          if predicate(lat, lon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i+r-k, center_j+k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
          if predicate(lat, lon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i-k, center_j+r-k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
          if predicate(lat, lon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

        i, j = center_i-r+k, center_j-k
        if is_on_grid(i, j)
          still_searching_on_grid = true
          lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
          if predicate(lat, lon)
            any_found_this_diamond = true
            push!(matching_flat_is, flat_i)
          end
        end

      end
      # Too many allocations???
      # s_e = map(k -> (center_i+k, center_j-r+k), rs)
      # e_n = map(k -> (center_i+r-k, center_j+k), rs)
      # n_w = map(k -> (center_i-k, center_j+r-k), rs)
      # w_s = map(k -> (center_i-r+k, center_j-k), rs)
      # diamond = [s_e; e_n; n_w; w_s]
      # s_e = zip(center_i:center_i+r-1,    center_j-r:center_j-1)
      # e_n = zip(center_i+r:-1:center_i+1, center_j:center_j+r-1)
      # n_w = zip(center_i:-1:center_i-r+1, center_j+r:-1:center_j+1)
      # w_s = zip(center_i-r:center_i-1,    center_j:-1:center_j-r+1)
      # diamond = collect(Base.Iterators.flatten((s_e, e_n, n_w, w_s))) # this line is slow
    end

    # for diamond_i = 1:length(diamond)
    #   i, j = diamond[diamond_i]
    #   if is_on_grid(i, j)
    #     still_searching_on_grid = true
    #     lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
    #     if predicate(lat, lon)
    #       any_found_this_diamond = true
    #       push!(matching_flat_is, flat_i)
    #     end
    #   end
    # end

    r += 1
    # push!(rs, r)
  end

  # print(" $r ")

  matching_flat_is
end



if IS_TRAINING
  println("Loading tornadoes...")

  tornado_rows, tornado_headers = readdlm("tornadoes.csv",','; header=true)

  # show(stdout_limited, "text/plain", tornado_headers)
  # println("")
  # show(stdout_limited, "text/plain", tornado_headers .== "begin_time_seconds")
  # println("")

  # println("finding headers")

  start_seconds_col_i, = find_i(tornado_headers, "begin_time_seconds")
  end_seconds_col_i,   = find_i(tornado_headers, "end_time_seconds")
  start_lat_col_i,     = find_i(tornado_headers, "begin_lat")
  start_lon_col_i,     = find_i(tornado_headers, "begin_lon")
  end_lat_col_i,       = find_i(tornado_headers, "end_lat")
  end_lon_col_i,       = find_i(tornado_headers, "end_lon")

  # println("datetime2unix")

  valid_start_seconds = Dates.datetime2unix(DateTime(valid_start_time))
  valid_end_seconds   = Dates.datetime2unix(DateTime(valid_end_time))

  # println("filtering relevant tornadoes")

  relevant_tornadoes = tornado_rows[broadcast_gt(tornado_rows[:, end_seconds_col_i], valid_start_seconds), :]
  relevant_tornadoes = relevant_tornadoes[broadcast_lt(relevant_tornadoes[:, start_seconds_col_i], valid_end_seconds), :]

  # if size(relevant_tornadoes,1) > 0
  #   relevant_tornadoes = mapslices(relevant_tornadoes, 2) do row
  #     # tornado_start_time = TimeZones.ZonedDateTime(Dates.unix2datetime(row[start_seconds_col_i]), utc)
  #     # tornado_end_time   = TimeZones.ZonedDateTime(Dates.unix2datetime(row[end_seconds_col_i]), utc)
  #     row
  #   end
  # end

  # println("translating relevant tornadoes")

  tornado_segments = map(1:size(relevant_tornadoes,1)) do i
    start_seconds = Int64(relevant_tornadoes[i, start_seconds_col_i])
    end_seconds   = Int64(relevant_tornadoes[i, end_seconds_col_i])
    duration      = end_seconds - start_seconds
    start_lat     = Float64(relevant_tornadoes[i, start_lat_col_i])
    start_lon     = Float64(relevant_tornadoes[i, start_lon_col_i])
    end_lat       = Float64(relevant_tornadoes[i, end_lat_col_i])
    end_lon       = Float64(relevant_tornadoes[i, end_lon_col_i])

    if duration == 0
      ( start_lat
      , start_lon
      , end_lat
      , end_lon
      )
    else
      if start_seconds >= valid_start_seconds
        seg_start_lat = start_lat
        seg_start_lon = start_lon
      else
        start_ratio = Float64(valid_start_seconds - start_seconds) / duration
        seg_start_lat, seg_start_lon = GeoUtils.ratio_on_segment(start_lat, start_lon, end_lat, end_lon, start_ratio)
      end

      if end_seconds <= valid_end_seconds
        seg_end_lat = end_lat
        seg_end_lon = end_lon
      else
        end_ratio = Float64(valid_end_seconds - start_seconds) / duration
        seg_end_lat, seg_end_lon = GeoUtils.ratio_on_segment(start_lat, start_lon, end_lat, end_lon, end_ratio)
      end

      ( seg_start_lat
      , seg_start_lon
      , seg_end_lat
      , seg_end_lon
      )
    end
  end :: Array{Tuple{Float64,Float64,Float64,Float64},1}

  # println("relevant tornadoes")
  # show(stdout_limited, "text/plain", relevant_tornadoes)

  println("tornado segments")
  show(stdout_limited, "text/plain", tornado_segments)
  println("")
  # max_error = 0.0
  # println(STDERR, tornado_segments)
else
  tornado_segments = Tuple{Float64,Float64,Float64,Float64}[]
  # println(STDERR, tornado_segments)
end


# r_theta_data = zeros(Float32, size(wind_layer_order,1)*2,grid_width*grid_height)
#
# # println(first_wind_layer_i)
# # println(length(wind_layer_order))
#
# for winds_layer_i in 1:length(wind_layer_order)
#   u_data = consolidated_data[first_wind_layer_i+winds_layer_i*2-2,:]
#   v_data = consolidated_data[first_wind_layer_i+winds_layer_i*2-1,:]
#
#   for k in 1:length(u_data)
#     r, theta = u_v_to_r_theta(u_data[k], v_data[k])
#     r_theta_data[winds_layer_i*2-1, k] = r
#     r_theta_data[winds_layer_i*2, k]   = theta
#   end
#   # if abbrev == "UGRD" || abbrev == "USTM"
#   #   u_abbrev, u_desc = abbrev, desc
#   #   v_abbrev = "V" * u_abbrev[2:4]
#   #   v_desc   = abbrev == "USTM" ? "v storm motion" : u_desc
#   #
#   #   u_layer_key = u_abbrev * ":" * u_desc
#   #   v_layer_key = v_abbrev * ":" * v_desc
#   #
#   #   push!(wind_layer_order, (u_abbrev, u_desc))
#   #   consolidated_data[i,:] = layer_to_data[u_layer_key]
#   #   i += 1
#   #
#   #   push!(wind_layer_order, (v_abbrev, v_desc))
#   #   consolidated_data[i,:] = layer_to_data[v_layer_key]
#   #   i += 1
#   # elseif abbrev == "VGRD" || abbrev == "VSTM"
#   # else
#   # end
# end



# println(i)
# show(stdout_limited, "text/plain", consolidated_data)

# exit(1)

println("Building features...")

# storm_motion_lon_us = layer_to_data["USTM:u storm motion"] :: Array{Float32,1}
# storm_motion_lat_vs = layer_to_data["VSTM:v storm motion"] :: Array{Float32,1}

function relativize_angle(theta::Float32, ref::Float32)::Float32
  Float32(mod(theta - ref + π, 2π) - π)
end


# Profile.init(delay=0.01)

function makeFeatures(prior_hour_data::Array{Float32,2}, data::Array{Float32,2})
  out_rows = Array{Float32}[]
  headers  = String[]
  first_pt = true

  # Avoid NaN gradients at the map edges
  padding = 20
  x_range = (1 + padding):(grid_width  - padding)
  y_range = (1 + padding):(grid_height - padding)

  for j = y_range
    # println("$j")
    for i = x_range
      lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)

      # skip if not a training point
      if !IS_TRAINING || (lat_lon_to_key(lat, lon) in training_pts_set)

        out_row = Float32[]

        is_close_to_tornado = false
        is_in_tornado_neighborhood = false
        for (tlat1, tlon1, tlat2, tlon2) in tornado_segments
          distance_to_tornado = GeoUtils.instant_distance_to_line(lat, lon, tlat1, tlon1, tlat2, tlon2)
          if distance_to_tornado <= 25.0 * GeoUtils.METERS_PER_MILE
            is_close_to_tornado = true
          end
          if distance_to_tornado <= 250.0 * GeoUtils.METERS_PER_MILE
            is_in_tornado_neighborhood = true
          end
        end
        if TORNADO_NEIGHBORHOODS_ONLY && !is_in_tornado_neighborhood
          continue
        end
        # if is_close_to_tornado
        #   print("t")
        # end

        lat_motion = data[storm_motion_us_j, flat_i] :: Float32 # m / s
        lon_motion = data[storm_motion_vs_j, flat_i] :: Float32 # m / S

        storm_r, storm_theta = u_v_to_r_theta(lon_motion, lat_motion)

        # Doesn't seem to save any time.
        # plus_30_mins_lat,  plus_30_mins_lon  = GeoUtils.instant_integrate_velocity(lat, lon,  lat_motion,  lon_motion, 30*ONE_MINUTE)
        # minus_30_mins_lat, minus_30_mins_lon = GeoUtils.instant_integrate_velocity(lat, lon, -lat_motion, -lon_motion, 30*ONE_MINUTE)
        # plus_30_mins_lat,  plus_30_mins_lon  = GeoUtils.integrate_velocity(lat, lon,  Float64(lat_motion),  Float64(lon_motion), 30*ONE_MINUTE)
        # minus_30_mins_lat, minus_30_mins_lon = GeoUtils.integrate_velocity(lat, lon, -Float64(lat_motion), -Float64(lon_motion), 30*ONE_MINUTE)
        minus_60_mins_lat, minus_60_mins_lon = GeoUtils.integrate_velocity(lat, lon, -Float64(lat_motion), -Float64(lon_motion), 60*ONE_MINUTE)

        # MakeTornadoNeighborhoodsData.jl and other scripts expect 10 columns to ignore
        push!(out_row, Float32(lat))
        push!(out_row, Float32(lon))
        push!(out_row, Float32(lat_motion))
        push!(out_row, Float32(lon_motion))
        push!(out_row, Float32(0))
        push!(out_row, Float32(0))
        push!(out_row, Float32(minus_60_mins_lat))
        push!(out_row, Float32(minus_60_mins_lon))
        push!(out_row, is_close_to_tornado ? 1.0f0 : 0.0f0)
        push!(out_row, point_weights[flat_i])
        if first_pt
          push!(headers, "lat")
          push!(headers, "lon")
          push!(headers, "lat_motion")
          push!(headers, "lon_motion")
          push!(headers, "not plus 30 mins lat")
          push!(headers, "not plus 30 mins lon")
          push!(headers, "minus 60 mins lat")
          push!(headers, "minus 60 mins lon")
          push!(headers, "tornado within 25mi")
          push!(headers, "training weight")
        end

        # err1, errd1 = GeoUtils.compare_integrate_velocity(lat, lon,  lat_motion,  lon_motion, 30*ONE_MINUTE)
        # err2, errd2 = GeoUtils.compare_integrate_velocity(lat, lon, -lat_motion, -lon_motion, 30*ONE_MINUTE)
        #
        # if errd1 > GeoUtils.METERS_PER_MILE * .05 || errd2 > GeoUtils.METERS_PER_MILE * .05
        #   println((err1, err2, errd1/GeoUtils.METERS_PER_MILE, errd2/GeoUtils.METERS_PER_MILE))
        # end

        # println((minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat,  plus_30_mins_lon))

        # find indices within 50 miles of -30 mins to +30 mins storm path
        # so the model can learn to compare this point or storm region to the near environment
        # flat_is_within_50mi_and_30_mins_of_storm =
        #   diamond_search(i, j) do lat, lon
        #     GeoUtils.instant_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon) <= 50.0 * GeoUtils.METERS_PER_MILE
        #   end

        # flat_is_within_50mi =
        #   diamond_search(i, j) do candidate_lat, candidate_lon
        #     GeoUtils.instant_distance(candidate_lat, candidate_lon, lat, lon) <= 50.0 * GeoUtils.METERS_PER_MILE
        #   end

        flat_is_within_25mi =
          diamond_search(i, j) do candidate_lat, candidate_lon
            GeoUtils.instant_distance(candidate_lat, candidate_lon, lat, lon) <= 25.0 * GeoUtils.METERS_PER_MILE
          end

        # 56 miles makes the 25mi circle roughly the same area as the four quadrants around it
        flat_is_between_25mi_and_56mi =
          diamond_search(i, j) do candidate_lat, candidate_lon
            distance = GeoUtils.instant_distance(candidate_lat, candidate_lon, lat, lon)
            distance >  25.0 * GeoUtils.METERS_PER_MILE &&
            distance <= 56.0 * GeoUtils.METERS_PER_MILE
          end

        flat_is_within_56mi = [flat_is_within_25mi; flat_is_between_25mi_and_56mi]

        flat_is_within_25mi_60_mins_ago =
          diamond_search(i, j) do candidate_lat, candidate_lon
            GeoUtils.instant_distance(candidate_lat, candidate_lon, minus_60_mins_lat, minus_60_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
          end

        flat_is_within_56mi_60_mins_ago =
          diamond_search(i, j) do candidate_lat, candidate_lon
            GeoUtils.instant_distance(candidate_lat, candidate_lon, minus_60_mins_lat, minus_60_mins_lon) <= 56.0 * GeoUtils.METERS_PER_MILE
          end

        flat_is_ahead  = Int64[]
        flat_is_behind = Int64[]
        flat_is_left   = Int64[]
        flat_is_right  = Int64[]

        for ring_flat_i in flat_is_between_25mi_and_56mi
          ring_lat, ring_lon = grid_lat_lons[ring_flat_i]

          ring_angle, _ = GeoUtils.azimuths(lat, lon, ring_lat, ring_lon)

          # note ring_angle is northward, clockwise in degrees
          # we want eastward, counterclockwise in radians
          relative_ring_theta = relativize_angle(Float32((-ring_angle / 180.0 * π) + π/2), storm_theta)

          # print("$lat,$lon\tto $ring_lat,$ring_lon\tis $relative_ring_theta versus $storm_theta, ")

          if relative_ring_theta >= -π/4 && relative_ring_theta < π/4
            # println("(ahead)")
            push!(flat_is_ahead, ring_flat_i)
          elseif relative_ring_theta >= π/4 && relative_ring_theta < 3π/4
            # println("(left)")
            push!(flat_is_left, ring_flat_i)
          elseif relative_ring_theta >= -3π/4 && relative_ring_theta < -π/4
            # println("(right)")
            push!(flat_is_right, ring_flat_i)
          else
            # println("(behind)")
            push!(flat_is_behind, ring_flat_i)
          end
        end

        # # find indices within 25 miles of -30 mins to +30 mins storm path
        # flat_is_within_25mi_and_30_mins_of_storm =
        #   # I'm guessing diamond_search is faster than walking through the flat_is_within_50mi_and_30_mins_of_storm
        #   diamond_search(i, j) do lat, lon
        #     # d = GeoUtils.instant_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon)
        #     # if d > 20.0 * GeoUtils.METERS_PER_MILE
        #     #   fast_error_pct = GeoUtils.compare_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon)
        #     #   global max_error = max(max_error, fast_error_pct)
        #     #   if fast_error_pct > 0.4
        #     #     println((fast_error_pct, d, lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon))
        #     #   end
        #     # end
        #     # GeoUtils.distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon, 1.0) <= 25.0 * GeoUtils.METERS_PER_MILE
        #     GeoUtils.instant_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        #   end
        #
        # # sort!(flat_is_within_25mi_and_30_mins_of_storm) # May help the prefetcher later.
        #
        # # find indices within 25 miles of -30 mins storm location
        # flat_is_within_25mi_of_storm_30_mins_ago =
        #   filter(flat_is_within_25mi_and_30_mins_of_storm) do candidate_flat_i
        #     candiate_lat, candidate_lon = grid_lat_lons[candidate_flat_i]
        #
        #     # lightning, instant, instantish, fast, fastish = GeoUtils.compare_distances(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon)
        #     # if instant > 0.004
        #     #   d = GeoUtils.distance(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon)
        #     #   println((d, lightning, instant, instantish, fast, fastish))
        #     # end
        #     GeoUtils.instant_distance(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        #   end
        #
        # # find indices within 25 miles of +30 mins storm location
        # flat_is_within_25mi_of_storm_30_mins_from_now =
        #   filter(flat_is_within_25mi_and_30_mins_of_storm) do candidate_flat_i
        #     candiate_lat, candidate_lon = grid_lat_lons[candidate_flat_i]
        #     GeoUtils.instant_distance(candiate_lat, candidate_lon, plus_30_mins_lat, plus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        #   end




        data_at_point             = data[:, flat_i]
        data_at_point_60_mins_ago = prior_hour_data[:, flat_i]

        data_within_25mi                             = data[:,flat_is_within_25mi]
        data_within_25mi_of_here_60_mins_ago         = prior_hour_data[:,flat_is_within_25mi]
        data_within_25mi_where_storm_was_60_mins_ago = prior_hour_data[:,flat_is_within_25mi_60_mins_ago]

        data_within_56mi                             = data[:,flat_is_within_56mi]
        data_within_56mi_of_here_60_mins_ago         = prior_hour_data[:,flat_is_within_56mi]
        data_within_56mi_where_storm_was_60_mins_ago = prior_hour_data[:,flat_is_within_56mi_60_mins_ago]

        data_ahead  = data_within_56mi[:,indexin(flat_is_ahead,  flat_is_within_56mi)]
        data_behind = data_within_56mi[:,indexin(flat_is_behind, flat_is_within_56mi)]
        data_left   = data_within_56mi[:,indexin(flat_is_left,   flat_is_within_56mi)]
        data_right  = data_within_56mi[:,indexin(flat_is_right,  flat_is_within_56mi)]

        # more_data_around_storm_path   = data[:,flat_is_within_50mi_and_30_mins_of_storm]
        # data_around_storm_path        = more_data_around_storm_path[:,indexin(flat_is_within_25mi_and_30_mins_of_storm, flat_is_within_50mi_and_30_mins_of_storm)]
        # polar_winds_around_storm_path = r_theta_data[:,flat_is_within_25mi_and_30_mins_of_storm]
        # data_around_30_mins_ago       = data_around_storm_path[:,indexin(flat_is_within_25mi_of_storm_30_mins_ago, flat_is_within_25mi_and_30_mins_of_storm)]
        # data_around_30_mins_from_now  = data_around_storm_path[:,indexin(flat_is_within_25mi_of_storm_30_mins_from_now, flat_is_within_25mi_and_30_mins_of_storm)]

        # for each regular layer...
        k = 1
        for abbrev_desc in regular_layer_order
          # push!(out_row, data_at_point[k])
          # push!(out_row, mean(@view data_around_storm_path[k,:]))
          # push!(out_row, minimum(@view data_around_storm_path[k,:]))
          # push!(out_row, maximum(@view data_around_storm_path[k,:]))
          # push!(out_row, mean(@view data_around_30_mins_from_now[k,:]) - mean(@view data_around_30_mins_ago[k,:]))
          # push!(out_row, mean(@view more_data_around_storm_path[k,:]))
          #
          # if first_pt
          #   abbrev, desc = abbrev_desc
          #   push!(headers, abbrev * ":" * desc * ":point")
          #   push!(headers, abbrev * ":" * desc * ":storm path mean")
          #   push!(headers, abbrev * ":" * desc * ":storm path min")
          #   push!(headers, abbrev * ":" * desc * ":storm path max")
          #   push!(headers, abbrev * ":" * desc * ":storm path gradient")
          #   push!(headers, abbrev * ":" * desc * ":storm path 50mi mean")
          # end

          push!(out_row, data_at_point[k])
          push!(out_row, mean(@view data_within_25mi[k,:]))
          push!(out_row, mean(@view data_within_56mi[k,:]))
          push!(out_row, mean(@view data_ahead[k,:]) - mean(@view data_behind[k,:]))
          push!(out_row, mean(@view data_left[k,:]) - mean(@view data_right[k,:]))
          if abbrev_desc == ("CAPE", "90-0 mb above ground") || abbrev_desc == ("LFTX", "500-1000 mb")
            push!(out_row, data_at_point_60_mins_ago[k])
          end
          push!(out_row, mean(@view data_within_25mi_of_here_60_mins_ago[k,:]))
          push!(out_row, mean(@view data_within_56mi_of_here_60_mins_ago[k,:]))
          push!(out_row, mean(@view data_within_25mi_where_storm_was_60_mins_ago[k,:]))
          push!(out_row, mean(@view data_within_56mi_where_storm_was_60_mins_ago[k,:]))


          if first_pt
            abbrev, desc = abbrev_desc
            push!(headers, abbrev * ":" * desc * ":point")
            push!(headers, abbrev * ":" * desc * ":25mi mean")
            push!(headers, abbrev * ":" * desc * ":56mi mean")
            push!(headers, abbrev * ":" * desc * ":25mi-56mi forward gradient")
            push!(headers, abbrev * ":" * desc * ":25mi-56mi leftward gradient")
            if abbrev_desc == ("CAPE", "90-0 mb above ground") || abbrev_desc == ("LFTX", "500-1000 mb")
              push!(headers, abbrev * ":" * desc * ":point -1hr")
            end
            push!(headers, abbrev * ":" * desc * ":25mi mean -1hr")
            push!(headers, abbrev * ":" * desc * ":56mi mean -1hr")
            push!(headers, abbrev * ":" * desc * ":25mi mean storm location -1hr")
            push!(headers, abbrev * ":" * desc * ":56mi mean storm location -1hr")
          end

          k += 1
        end

        # for each wind layer...
        for winds_layer_i in 1:length(wind_layer_order)
          abbrev, desc = wind_layer_order[winds_layer_i]
          if abbrev[1] == 'U' || abbrev == "VUCSH" # Handle u and v layers together, so skip v layers.
            abbrev_root = abbrev == "VUCSH" ? "VCSH" : abbrev[2:4]
            r_unit      = abbrev == "VUCSH" ? "shear magnitude 1/s" : "speed m/s"

            # point value
            r, theta = u_v_to_r_theta(data_at_point[k], data_at_point[k+1])
            # r_60_mins_ago, theta_60_mins_ago = u_v_to_r_theta(data_at_point_60_mins_ago[k], data_at_point_60_mins_ago[k+1])
            push!(out_row, r)
            # push!(out_row, r_60_mins_ago)
            if abbrev != "USTM" # Always relativized to zero
              relative_theta = relativize_angle(theta, storm_theta)
              push!(out_row, relative_theta)
              # push!(out_row, abs(relative_theta))
            end
            # relative_theta_60_mins_ago = relativize_angle(theta_60_mins_ago, storm_theta)
            # push!(out_row, relative_theta_60_mins_ago)
            if first_pt
              if abbrev != "USTM"
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":point")
                # push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":point -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":point")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":point")
              else
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion:point")
                # push!(headers, "R" * abbrev_root * ":$r_unit storm motion:point -1hr")
              end
              # push!(headers, "T" * abbrev_root * ":angle radians from point storm motion " * desc * ":point -1hr")
            end

            # local mean
            u_mean                             = mean(@view data_within_25mi[k,:])
            v_mean                             = mean(@view data_within_25mi[k+1,:])
            u_mean_within_56mi                 = mean(@view data_within_56mi[k,:])
            v_mean_within_56mi                 = mean(@view data_within_56mi[k+1,:])
            du_mean_ahead                      = mean(@view data_ahead[k,:])   - mean(@view data_behind[k,:])
            dv_mean_ahead                      = mean(@view data_ahead[k+1,:]) - mean(@view data_behind[k+1,:])
            du_mean_left                       = mean(@view data_left[k,:])    - mean(@view data_right[k,:])
            dv_mean_left                       = mean(@view data_left[k+1,:])  - mean(@view data_right[k+1,:])

            u_mean_60_mins_ago                 = mean(@view data_within_25mi_of_here_60_mins_ago[k,:])
            v_mean_60_mins_ago                 = mean(@view data_within_25mi_of_here_60_mins_ago[k+1,:])
            u_mean_within_56mi_60_mins_ago     = mean(@view data_within_56mi_of_here_60_mins_ago[k,:])
            v_mean_within_56mi_60_mins_ago     = mean(@view data_within_56mi_of_here_60_mins_ago[k+1,:])

            u_mean_where_storm_was_60_mins_ago             = mean(@view data_within_25mi_where_storm_was_60_mins_ago[k,:])
            v_mean_where_storm_was_60_mins_ago             = mean(@view data_within_25mi_where_storm_was_60_mins_ago[k+1,:])
            u_mean_within_56mi_where_storm_was_60_mins_ago = mean(@view data_within_56mi_where_storm_was_60_mins_ago[k,:])
            v_mean_within_56mi_where_storm_was_60_mins_ago = mean(@view data_within_56mi_where_storm_was_60_mins_ago[k+1,:])


            mean_r, mean_theta                                                         = u_v_to_r_theta(u_mean, v_mean)
            mean_r_within_56mi, mean_theta_within_56mi                                 = u_v_to_r_theta(u_mean_within_56mi, v_mean_within_56mi)
            mean_dr_ahead, mean_dtheta_ahead                                           = u_v_to_r_theta(du_mean_ahead, dv_mean_ahead)
            mean_dr_left, mean_dtheta_left                                             = u_v_to_r_theta(du_mean_left, dv_mean_left)

            mean_r_60_mins_ago, mean_theta_60_mins_ago                                 = u_v_to_r_theta(u_mean_60_mins_ago, v_mean_60_mins_ago)
            mean_r_within_56mi_60_mins_ago, mean_theta_within_56mi_60_mins_ago         = u_v_to_r_theta(u_mean_within_56mi_60_mins_ago, v_mean_within_56mi_60_mins_ago)

            mean_r_where_storm_was_60_mins_ago, mean_theta_where_storm_was_60_mins_ago                         = u_v_to_r_theta(u_mean_where_storm_was_60_mins_ago, v_mean_where_storm_was_60_mins_ago)
            mean_r_within_56mi_where_storm_was_60_mins_ago, mean_theta_within_56mi_where_storm_was_60_mins_ago = u_v_to_r_theta(u_mean_within_56mi_where_storm_was_60_mins_ago, v_mean_within_56mi_where_storm_was_60_mins_ago)


            push!(out_row, mean_r)
            relative_theta = relativize_angle(mean_theta, storm_theta)
            push!(out_row, relative_theta)
            # push!(out_row, abs(relative_theta))

            push!(out_row, mean_r_within_56mi)
            relative_theta_within_56mi = relativize_angle(mean_theta_within_56mi, storm_theta)
            push!(out_row, relative_theta_within_56mi)
            # push!(out_row, abs(relative_theta_within_56mi))

            push!(out_row, mean_dr_ahead)
            relative_dtheta_ahead = relativize_angle(mean_dtheta_ahead, storm_theta)
            push!(out_row, relative_dtheta_ahead)
            # push!(out_row, abs(relative_dtheta_ahead))

            push!(out_row, mean_dr_left)
            relative_dtheta_left = relativize_angle(mean_dtheta_left, storm_theta)
            push!(out_row, relative_dtheta_left)
            # push!(out_row, abs(relative_dtheta_left))


            push!(out_row, mean_r_60_mins_ago)
            relative_theta_60_mins_ago = relativize_angle(mean_theta_60_mins_ago, storm_theta)
            push!(out_row, relative_theta_60_mins_ago)
            # push!(out_row, abs(relative_theta_theta_60_mins_ago))

            push!(out_row, mean_r_within_56mi_60_mins_ago)
            relative_theta_within_56mi_60_mins_ago = relativize_angle(mean_theta_within_56mi_60_mins_ago, storm_theta)
            push!(out_row, relative_theta_within_56mi_60_mins_ago)
            # push!(out_row, abs(relative_theta_within_56mi_60_mins_ago))


            push!(out_row, mean_r_where_storm_was_60_mins_ago)
            relative_theta_where_storm_was_60_mins_ago = relativize_angle(mean_theta_where_storm_was_60_mins_ago, storm_theta)
            push!(out_row, relative_theta_where_storm_was_60_mins_ago)
            # push!(out_row, abs(relative_theta_where_storm_was_60_mins_ago))

            push!(out_row, mean_r_within_56mi_where_storm_was_60_mins_ago)
            relative_theta_within_56mi_where_storm_was_60_mins_ago = relativize_angle(mean_theta_within_56mi_where_storm_was_60_mins_ago, storm_theta)
            push!(out_row, relative_theta_within_56mi_where_storm_was_60_mins_ago)
            # push!(out_row, abs(relative_theta_within_56mi_where_storm_was_60_mins_ago))


            if first_pt
              if abbrev != "USTM"
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":25mi mean")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":25mi mean")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi mean")
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":56mi mean")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":56mi mean")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":56mi mean")
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":25mi-56mi forward gradient")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":25mi-56mi forward gradient")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi-56mi forward gradient")
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":25mi-56mi leftward gradient")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":25mi-56mi leftward gradient")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi-56mi leftward gradient")

                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":25mi mean -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":25mi mean -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi mean -1hr")
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":56mi mean -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":56mi mean -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi mean -1hr")

                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":25mi mean storm location -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":25mi mean storm location -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi mean storm location -1hr")
                push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":56mi mean storm location -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":56mi mean storm location -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":25mi mean storm location -1hr")
              else
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":25mi mean")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":25mi mean")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi mean")
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":56mi mean")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":56mi mean")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * desc * ":56mi mean")
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":25mi-56mi forward gradient")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":25mi-56mi forward gradient")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi-56mi forward gradient")
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":25mi-56mi leftward gradient")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":25mi-56mi leftward gradient")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi-56mi leftward gradient")

                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":25mi mean -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":25mi mean -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi mean -1hr")
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":56mi mean -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":56mi mean -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi mean -1hr")

                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":25mi mean storm location -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":25mi mean storm location -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi mean storm location -1hr")
                push!(headers, "R" * abbrev_root * ":$r_unit storm motion"                               * ":56mi mean storm location -1hr")
                push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion" * ":56mi mean storm location -1hr")
                # push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion" * ":25mi mean storm location -1hr")
              end
            end

            # # local min
            # rs     = polar_winds_around_storm_path[winds_layer_i*2-1, :]
            # thetas = polar_winds_around_storm_path[winds_layer_i*2, :]
            # push!(out_row, minimum(rs))
            # push!(out_row, minimum(thetas))
            #
            # # local max
            # push!(out_row, maximum(rs))
            # push!(out_row, maximum(thetas))


            # min_r, min_theta = u_v_to_r_theta(data_around_storm_path[k,1], data_around_storm_path[k+1,1])
            # max_r, max_theta = min_r, min_theta
            # for l = 2:length(flat_is_within_25mi_and_30_mins_of_storm)
            #   r, theta = u_v_to_r_theta(data_around_storm_path[k,l], data_around_storm_path[k+1,l])
            #   theta = relativize_angle(theta, storm_theta)
            #   if r < min_r
            #     min_r = r
            #   end
            #   if r > max_r
            #     max_r = r
            #   end
            #   if theta < min_theta
            #     min_theta = theta
            #   end
            #   if theta > max_theta
            #     max_theta = theta
            #   end
            # end
            # push!(out_row, min_r)
            # push!(out_row, min_theta)
            # push!(out_row, max_r)
            # push!(out_row, max_theta)
            # push!(out_row, max(abs(min_theta), abs(max_theta)))
            # if first_pt
            #   if abbrev != "USTM"
            #     push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":storm path min")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":storm path min")
            #     push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":storm path max")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":storm path max")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":storm path max")
            #   else
            #     push!(headers, "R" * abbrev_root * ":$r_unit storm motion:storm path min")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion:storm path min")
            #     push!(headers, "R" * abbrev_root * ":$r_unit storm motion:storm path max")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion:storm path max")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion:storm path max")
            #   end
            # end
            #
            # # version below is slow...perhaps because of allocations.
            # # polar_vectors = map(uv_to_r_theta, zip(us_around_storm_path, vs_around_storm_path))
            # # rs     = map(first, polar_vectors)
            # # thetas = map(r_theta -> relativize_angle(r_theta[2], storm_theta),  polar_vectors)
            # # push!(out_row, minimum(rs))
            # # push!(out_row, minimum(thetas))
            # #
            # # # local max
            # # push!(out_row, maximum(rs))
            # # push!(out_row, maximum(thetas))
            #
            # # us_around_30_mins_ago      = data_around_30_mins_ago[k,:]
            # # vs_around_30_mins_ago      = data_around_30_mins_ago[k+1,:]
            # # us_around_30_mins_from_now = data_around_30_mins_from_now[k,:]
            # # vs_around_30_mins_from_now = data_around_30_mins_from_now[k+1,:]
            #
            #


            # # gradient in storm direction
            # delta_u = mean(@view data_around_30_mins_from_now[k,:]) - mean(@view data_around_30_mins_ago[k,:])
            # delta_v = mean(@view data_around_30_mins_from_now[k+1,:]) - mean(@view data_around_30_mins_ago[k+1,:])
            # mean_delta_r, mean_delta_theta = u_v_to_r_theta(delta_u, delta_v)
            # push!(out_row, mean_delta_r)
            # relative_mean_delta_theta = relativize_angle(mean_delta_theta, storm_theta)
            # push!(out_row, relative_mean_delta_theta)
            # push!(out_row, abs(relative_mean_delta_theta))
            # if first_pt
            #   if abbrev != "USTM"
            #     push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":storm path gradient")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":storm path gradient")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":storm path gradient")
            #   else
            #     push!(headers, "R" * abbrev_root * ":$r_unit storm motion:storm path gradient")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion:storm path gradient")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion:storm path gradient")
            #   end
            # end
            #
            #
            # # 50mi mean
            # u_50mi_mean = mean(@view more_data_around_storm_path[k,:])
            # v_50mi_mean = mean(@view more_data_around_storm_path[k+1,:])
            # mean_50mi_r, mean_50mi_theta = u_v_to_r_theta(u_50mi_mean, v_50mi_mean)
            # push!(out_row, mean_50mi_r)
            # relative_mean_50mi_theta = relativize_angle(mean_50mi_theta, storm_theta)
            # push!(out_row, relative_mean_50mi_theta)
            # push!(out_row, abs(relative_mean_50mi_theta))
            # if first_pt
            #   if abbrev != "USTM"
            #     push!(headers, "R" * abbrev_root * ":$r_unit "                                              * desc * ":storm path 50mi mean")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion "                * desc * ":storm path 50mi mean")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion " * desc * ":storm path 50mi mean")
            #   else
            #     push!(headers, "R" * abbrev_root * ":$r_unit storm motion:storm path 50mi mean")
            #     push!(headers, "T" * abbrev_root * ":angle radians from point storm motion storm motion:storm path 50mi mean")
            #     push!(headers, "T" * abbrev_root * ":absolute value angle radians from point storm motion storm motion:storm path 50mi mean")
            #   end
            # end

            k += 2
          end
        end

        push!(out_rows, out_row)
        first_pt = false
      end
    end
  end

  (headers, out_rows)
end

headers, out_rows = makeFeatures(prior_hour_consolidated_data, consolidated_data)

# println(headers)

# open("training.csv", "w") do f
#   writedlm(f, [headers], ',')
#   writedlm(f, out_rows[[6607, 14570, 11574, 34094, 12879, 5687, 27445, 7043, 37298, 37411, 11986, 31411, 12133, 8139, 10896, 30805, 16827, 32259, 11552, 9795, 32675, 16649, 21560, 629, 2758, 2046, 21768, 18020, 34494, 20742, 13081, 19156, 27910, 10503, 1756, 37709, 7693, 4233, 10974, 28505, 35292, 17995, 9313, 12762, 37430, 1790, 12494, 34689, 23741, 6236, 17549, 11166, 14494, 25405, 28983, 25011, 20006, 17108, 28070, 1670, 10739, 31648, 15594, 26814, 21439, 16716, 29435, 2441, 27199, 9130, 21846, 7085, 32196, 1456, 26125, 12281, 7683, 30286, 35499, 13931, 4163, 1728, 403, 17275, 22366, 8175, 2672, 31774, 17210, 13522, 35876, 32889, 18157, 28099, 16052, 13984, 36386, 29747, 728, 4622]], ',')
# end


# Requires GMT >=6
# include("PlotMap.jl")
# function plot_data_col(col_i)
#   header = headers[col_i]
#   println(header)
#   base_path = "plots/" * replace(header, r" |:|/", "_")
#   plot_map(base_path, map(row -> row[1], out_rows), map(row -> row[2], out_rows), map(row -> row[col_i], out_rows))
# end

# for col_i in 1:length(headers)
#   plot_data_col(col_i)
# end
# plot_data_col(1076) # point storm speed
# plot_data_col(9) # tornadoes
# plot_data_col(10) # training weight
# plot_data_col(16) # point cape
# plot_data_col(846) # surface tmp
# plot_data_col(847) # surface tmp
# plot_data_col(848) # surface tmp
# plot_data_col(849) # surface tmp
# plot_data_col(850) # surface tmp

# Put each data point in a column.
#
# This allows us to add more data points simply by appending (because Julia is column-major).
out_flat = zeros(Float32, length(headers), length(out_rows))

for i in 1:length(out_rows)
  out_flat[:,i] = out_rows[i]
end

# show(stdout_limited, "text/plain", out_flat)

# println("headers")
map(println, headers)
# println("\n")
# println("Max magnitudes per column (to prevent exploding gradients)")
# mapslices(x -> println(maximum(abs.(x))), out_flat, 2)

write(out_path, out_flat)

println("\n")
println(out_path)


# Profile.print()
# Profile.print(format=:flat, noisefloor=2.0, sortedby=:count)

# @code_warntype makeFeatures(consolidated_data)
# @code_warntype u_v_to_r_theta(1.0,2.0)

# println(max_error)
