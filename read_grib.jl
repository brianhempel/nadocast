# puts "### Common Fields ###"
#
# # RUC has an HLCY coded as "surface" but is probably 0-3km helicity per https://ruc.noaa.gov/ruc/fslparms/13km/ruc13_grib.table-nov2008 and http://www.nws.noaa.gov/om/tpb/448body.htm (surface doesn't really make sense)
# # Same for UV of storm motion.
# # CAPE 255-0mb is MUCAPE https://ruc.noaa.gov/forum/f2/Welcome.cgi/read/2145
# #
# # 4LFTX and LFTX ??
# #
# # Low-level (180-0 mb agl, 90-0 mb agl) CAPE/CIN not added until 2014-02-25-1200. How to handle?? (Estimate for old data??)
# #
#


# Following fields exluded b/c they are cumulative:

# SSRUN:surface
# NCPCP:surface
# BGRUN:surface
# WEASD:surface
# ACPCP:surface

# It *does* appear that there are multiple accumulation periods in the forecast (i.e. always a 1-hour period somewhere for all of the above)
# May look for that.

push!(LOAD_PATH, ".")

import GeoUtils
import TimeZones

utc = TimeZones.tz"UTC"

grib2_path = "rap_130_20170516_2200_001.grb2"

year_str, month_str, day_str, run_hour_str, forcast_hour_str = match(r"_130_(\d\d\d\d)(\d\d)(\d\d)_(\d\d)00_(\d\d\d)\.grb2", grib2_path).captures

valid_time       = TimeZones.ZonedDateTime(parse(Int64,year_str),parse(Int64,month_str),parse(Int64,day_str),parse(Int64,run_hour_str),0,0, utc) + Base.Dates.Hour(parse(Int64,forcast_hour_str))
valid_start_time = valid_time - Base.Dates.Minute(30)
valid_end_time   = valid_time + Base.Dates.Minute(30)

println(valid_start_time," to ",valid_end_time)



stdout_limited = IOContext(STDOUT, :display_size=>(100,60))
stdout_limited = IOContext(stdout_limited, :limit=>true)

### Common Fields ###
layers = readdlm(IOBuffer("""ABSV:500 mb
CAPE:255-0 mb above ground
CAPE:surface
CFRZR:surface
CICEP:surface
CIN:255-0 mb above ground
CIN:surface
CRAIN:surface
CSNOW:surface
DEPR:2 m above ground
HGT:cloud base
HGT:cloud top
DPT:2 m above ground
EPOT:surface
GUST:surface
HGT:0C isotherm
HGT:100 mb
HGT:125 mb
HGT:150 mb
HGT:175 mb
HGT:200 mb
HGT:225 mb
HGT:250 mb
HGT:275 mb
HGT:300 mb
HGT:325 mb
HGT:350 mb
HGT:375 mb
HGT:400 mb
HGT:425 mb
HGT:450 mb
HGT:475 mb
HGT:500 mb
HGT:525 mb
HGT:550 mb
HGT:575 mb
HGT:600 mb
HGT:625 mb
HGT:650 mb
HGT:675 mb
HGT:700 mb
HGT:725 mb
HGT:750 mb
HGT:775 mb
HGT:800 mb
HGT:825 mb
HGT:850 mb
HGT:875 mb
HGT:900 mb
HGT:925 mb
HGT:950 mb
HGT:975 mb
HGT:1000 mb
HGT:convective cloud top level
HGT:equilibrium level
HGT:highest tropospheric freezing level
HGT:lowest level of the wet bulb zero
HGT:surface
HLCY:1000-0 m above ground
HLCY:3000-0 m above ground
HPBL:surface
MSLMA:mean sea level
MSTAV:0 m underground
POT:2 m above ground
POT:tropopause
PRATE:surface
PRES:0C isotherm
PRES:highest tropospheric freezing level
PRES:max wind
PRES:surface
PRES:tropopause
PWAT:entire atmosphere (considered as a single layer)
REFC:entire atmosphere (considered as a single layer)
REFD:1000 m above ground
REFD:4000 m above ground
RH:0C isotherm
RH:100 mb
RH:120-90 mb above ground
RH:125 mb
RH:150 mb
RH:150-120 mb above ground
RH:175 mb
RH:180-150 mb above ground
RH:2 m above ground
RH:200 mb
RH:225 mb
RH:250 mb
RH:275 mb
RH:30-0 mb above ground
RH:300 mb
RH:325 mb
RH:350 mb
RH:375 mb
RH:400 mb
RH:425 mb
RH:450 mb
RH:475 mb
RH:500 mb
RH:525 mb
RH:550 mb
RH:575 mb
RH:60-30 mb above ground
RH:600 mb
RH:625 mb
RH:650 mb
RH:675 mb
RH:700 mb
RH:725 mb
RH:750 mb
RH:775 mb
RH:800 mb
RH:825 mb
RH:850 mb
RH:875 mb
RH:90-60 mb above ground
RH:900 mb
RH:925 mb
RH:950 mb
RH:975 mb
RH:1000 mb
RH:highest tropospheric freezing level
SNOD:surface
SPFH:2 m above ground
TMP:100 mb
TMP:120-90 mb above ground
TMP:125 mb
TMP:150 mb
TMP:150-120 mb above ground
TMP:175 mb
TMP:180-150 mb above ground
TMP:2 m above ground
TMP:200 mb
TMP:225 mb
TMP:250 mb
TMP:275 mb
TMP:30-0 mb above ground
TMP:300 mb
TMP:325 mb
TMP:350 mb
TMP:375 mb
TMP:400 mb
TMP:425 mb
TMP:450 mb
TMP:475 mb
TMP:500 mb
TMP:525 mb
TMP:550 mb
TMP:575 mb
TMP:60-30 mb above ground
TMP:600 mb
TMP:625 mb
TMP:650 mb
TMP:675 mb
TMP:700 mb
TMP:725 mb
TMP:750 mb
TMP:775 mb
TMP:800 mb
TMP:825 mb
TMP:850 mb
TMP:875 mb
TMP:90-60 mb above ground
TMP:900 mb
TMP:925 mb
TMP:950 mb
TMP:975 mb
TMP:1000 mb
TMP:surface
TMP:tropopause
USTM:u storm motion
VSTM:v storm motion
UGRD:10 m above ground
VGRD:10 m above ground
UGRD:100 mb
VGRD:100 mb
UGRD:120-90 mb above ground
VGRD:120-90 mb above ground
UGRD:125 mb
VGRD:125 mb
UGRD:150 mb
VGRD:150 mb
UGRD:150-120 mb above ground
VGRD:150-120 mb above ground
UGRD:175 mb
VGRD:175 mb
UGRD:180-150 mb above ground
VGRD:180-150 mb above ground
UGRD:200 mb
VGRD:200 mb
UGRD:225 mb
VGRD:225 mb
UGRD:250 mb
VGRD:250 mb
UGRD:275 mb
VGRD:275 mb
UGRD:30-0 mb above ground
VGRD:30-0 mb above ground
UGRD:300 mb
VGRD:300 mb
UGRD:325 mb
VGRD:325 mb
UGRD:350 mb
VGRD:350 mb
UGRD:375 mb
VGRD:375 mb
UGRD:400 mb
VGRD:400 mb
UGRD:425 mb
VGRD:425 mb
UGRD:450 mb
VGRD:450 mb
UGRD:475 mb
VGRD:475 mb
UGRD:500 mb
VGRD:500 mb
UGRD:525 mb
VGRD:525 mb
UGRD:550 mb
VGRD:550 mb
UGRD:575 mb
VGRD:575 mb
UGRD:60-30 mb above ground
VGRD:60-30 mb above ground
UGRD:600 mb
VGRD:600 mb
UGRD:625 mb
VGRD:625 mb
UGRD:650 mb
VGRD:650 mb
UGRD:675 mb
VGRD:675 mb
UGRD:700 mb
VGRD:700 mb
UGRD:725 mb
VGRD:725 mb
UGRD:750 mb
VGRD:750 mb
UGRD:775 mb
VGRD:775 mb
UGRD:800 mb
VGRD:800 mb
UGRD:825 mb
VGRD:825 mb
UGRD:850 mb
VGRD:850 mb
UGRD:875 mb
VGRD:875 mb
UGRD:90-60 mb above ground
VGRD:90-60 mb above ground
UGRD:900 mb
VGRD:900 mb
UGRD:925 mb
VGRD:925 mb
UGRD:950 mb
VGRD:950 mb
UGRD:975 mb
VGRD:975 mb
UGRD:1000 mb
VGRD:1000 mb
UGRD:max wind
VGRD:max wind
UGRD:tropopause
VGRD:tropopause
VIS:surface
VVEL:100 mb
VVEL:120-90 mb above ground
VVEL:125 mb
VVEL:150 mb
VVEL:150-120 mb above ground
VVEL:175 mb
VVEL:180-150 mb above ground
VVEL:200 mb
VVEL:225 mb
VVEL:250 mb
VVEL:275 mb
VVEL:30-0 mb above ground
VVEL:300 mb
VVEL:325 mb
VVEL:350 mb
VVEL:375 mb
VVEL:400 mb
VVEL:425 mb
VVEL:450 mb
VVEL:475 mb
VVEL:500 mb
VVEL:525 mb
VVEL:550 mb
VVEL:575 mb
VVEL:60-30 mb above ground
VVEL:600 mb
VVEL:625 mb
VVEL:650 mb
VVEL:675 mb
VVEL:700 mb
VVEL:725 mb
VVEL:750 mb
VVEL:775 mb
VVEL:800 mb
VVEL:825 mb
VVEL:850 mb
VVEL:875 mb
VVEL:90-60 mb above ground
VVEL:900 mb
VVEL:925 mb
VVEL:950 mb
VVEL:975 mb
VVEL:1000 mb"""), ':', String; header=false, use_mmap=false, quotes=false)

# Read the inventory of the file so we can corrolate it to figure out which layers we want.
inventory_lines = open(`wgrib2 $grib2_path -s -n`) do inv
  readdlm(inv, ':', String; header=false, use_mmap=false, quotes=false)
end

# unique_inventories_ignoring_number = unique_inventories.map do |inventory|
#   inventory.map do |abbrev, desc|
#     next(["MSTAV", "0 m underground"]) if abbrev == "MSTAV"
#     next(["USTM", "u storm motion"]) if abbrev == "USTM"
#     next(["VSTM", "v storm motion"]) if abbrev == "VSTM"
#     next(["HLCY", "3000-0 m above ground"]) if abbrev == "HLCY" && desc == "surface"
#     [
#       abbrev.gsub(/\bDIST\b/, "HGT"), # Also changed from m to gpm (geo-potential meters); probably not a big deal.
#       desc.
#         gsub("0-6000 m above ground", "6000-0 m above ground").
#         # gsub(/\bsfc\b/, "surface").
#         # gsub(/\bgnd\b/, "ground").
#         # gsub(/\bhigh trop\b/, "highest tropospheric").
#         # gsub(/\blvl\b|\blev\b/, "level").
#         # gsub(/\batmos col\b/, "entire atmosphere (considered as a single layer)").
#         gsub(/\bentire atmosphere\z/, "entire atmosphere (considered as a single layer)").
#         # gsub(/\b300 cm down\b|\b3 m underground\b/, "surface"). # BGRUN:300 cm down -> BGRUN:surface
#         gsub(/\b3 m underground\b/, "surface") # BGRUN:3 m underground -> BGRUN:surface
#         # gsub(/\bMSL\b/, "mean sea level").
#         # gsub(/\bconvect-cld top\b/, "convective cloud top level").
#         # gsub(/\bmax e-pot-temp\b/, "maximum equivalent potential temperature").
#         # gsub(/\bmax wind level\b/, "max wind").
#         # gsub(/\bof wet bulb\b/, "of the wet bulb").
#         # gsub(/\bcld\b/, "cloud")
#     ]
#   end
# end

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

# show(stdout_limited, "text/plain", inventory_lines)

inventory_lines_normalized =
  mapslices(normalize_line, inventory_lines, 2)

# show(stdout_limited, "text/plain", inventory_lines_normalized)

function desiredLayerToInventoryLine(desired)
  desiredAbbrev = desired[1]
  desiredDesc   = desired[2]
  # I hate this language.
  matching_i =
    findfirst([[desiredAbbrev, desiredDesc] == inventory_lines_normalized[i,4:5] for i=1:size(inventory_lines_normalized,1)])

  if matching_i > 0
    return inventory_lines_normalized[matching_i, :]
  else
    error("Could not find inventory_lines_normalized row for $desiredAbbrev $desiredDesc")
  end
end

layers_to_fetch = mapslices(desiredLayerToInventoryLine, layers, 2)

# show(stdout_limited, "text/plain", layers_to_fetch)

layer_to_data = Dict{String,Array{Float64}}()

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
  # println(grid_length)
  # println(values)
  layer_to_data[layer_key] = values
  # println(grid_length_again)
end

# Sanity check that incoming stream is empty
if !eof(from_wgrib2)
  error("wgrib2 sending more data than expected!")
end

# Normalize wind angle relative to storm motion (convert to polar: ground speed + relative direction)

# UV to polar
function uv_to_r_theta(pair)
  u, v = pair
  u_v_to_r_theta(u, v)
end

function u_v_to_r_theta(u, v)
  if u == 0.0 && v == 0.0
    return (0.0, 0.0)
  end
  r     = sqrt(u^2 + v^2)
  theta = atan2(v, u) # Angle, in radians
  theta = mod(theta + π, 2π) - π # Turns π into -π (otherwise a passthrough here)
  (r, theta)
end

# storm_motion_us = layer_to_data["USTM:u storm motion"]
# storm_motion_vs = layer_to_data["VSTM:v storm motion"]
#
# storm_motion_polar_vectors = map(uv_to_r_theta, zip(storm_motion_us, storm_motion_vs))
# storm_motion_rs     = map(first, storm_motion_polar_vectors)
# storm_motion_thetas = map(last,  storm_motion_polar_vectors)

uv_layers_excluding_storm_motion = [
  ("UGRD:10 m above ground",       "VGRD:10 m above ground"),
  ("UGRD:100 mb",                  "VGRD:100 mb"),
  ("UGRD:120-90 mb above ground",  "VGRD:120-90 mb above ground"),
  ("UGRD:125 mb",                  "VGRD:125 mb"),
  ("UGRD:150 mb",                  "VGRD:150 mb"),
  ("UGRD:150-120 mb above ground", "VGRD:150-120 mb above ground"),
  ("UGRD:175 mb",                  "VGRD:175 mb"),
  ("UGRD:180-150 mb above ground", "VGRD:180-150 mb above ground"),
  ("UGRD:200 mb",                  "VGRD:200 mb"),
  ("UGRD:225 mb",                  "VGRD:225 mb"),
  ("UGRD:250 mb",                  "VGRD:250 mb"),
  ("UGRD:275 mb",                  "VGRD:275 mb"),
  ("UGRD:30-0 mb above ground",    "VGRD:30-0 mb above ground"),
  ("UGRD:300 mb",                  "VGRD:300 mb"),
  ("UGRD:325 mb",                  "VGRD:325 mb"),
  ("UGRD:350 mb",                  "VGRD:350 mb"),
  ("UGRD:375 mb",                  "VGRD:375 mb"),
  ("UGRD:400 mb",                  "VGRD:400 mb"),
  ("UGRD:425 mb",                  "VGRD:425 mb"),
  ("UGRD:450 mb",                  "VGRD:450 mb"),
  ("UGRD:475 mb",                  "VGRD:475 mb"),
  ("UGRD:500 mb",                  "VGRD:500 mb"),
  ("UGRD:525 mb",                  "VGRD:525 mb"),
  ("UGRD:550 mb",                  "VGRD:550 mb"),
  ("UGRD:575 mb",                  "VGRD:575 mb"),
  ("UGRD:60-30 mb above ground",   "VGRD:60-30 mb above ground"),
  ("UGRD:600 mb",                  "VGRD:600 mb"),
  ("UGRD:625 mb",                  "VGRD:625 mb"),
  ("UGRD:650 mb",                  "VGRD:650 mb"),
  ("UGRD:675 mb",                  "VGRD:675 mb"),
  ("UGRD:700 mb",                  "VGRD:700 mb"),
  ("UGRD:725 mb",                  "VGRD:725 mb"),
  ("UGRD:750 mb",                  "VGRD:750 mb"),
  ("UGRD:775 mb",                  "VGRD:775 mb"),
  ("UGRD:800 mb",                  "VGRD:800 mb"),
  ("UGRD:825 mb",                  "VGRD:825 mb"),
  ("UGRD:850 mb",                  "VGRD:850 mb"),
  ("UGRD:875 mb",                  "VGRD:875 mb"),
  ("UGRD:90-60 mb above ground",   "VGRD:90-60 mb above ground"),
  ("UGRD:900 mb",                  "VGRD:900 mb"),
  ("UGRD:925 mb",                  "VGRD:925 mb"),
  ("UGRD:950 mb",                  "VGRD:950 mb"),
  ("UGRD:975 mb",                  "VGRD:975 mb"),
  ("UGRD:1000 mb",                 "VGRD:1000 mb"),
  ("UGRD:max wind",                "VGRD:max wind"),
  ("UGRD:tropopause",              "VGRD:tropopause")
]

# for (u_layer_key, v_layer_key) in uv_layers_excluding_storm_motion
#   us = layer_to_data[u_layer_key]
#   vs = layer_to_data[v_layer_key]
#
#   polar_vectors   = map(uv_to_r_theta, zip(us, vs))
#   rs              = map(first, polar_vectors) # speeds
#   thetas          = map(last,  polar_vectors) # grid relative angles
#
#   # Will relativize angles against storm motion later. (For a point, relative to storm motion at that point; but for computing area statistics around that point, should be relative to that point as well)
#
#   # delete!(layer_to_data, u_layer_key)
#   # delete!(layer_to_data, v_layer_key)
#
#   r_layer_key     = replace(u_layer_key, r"^U", "R")
#   theta_layer_key = replace(v_layer_key, r"^V", "T")
#
#   layer_to_data[r_layer_key]     = rs
#   layer_to_data[theta_layer_key] = thetas
# end

# delete!(layer_to_data, "USTM:u storm motion")
# delete!(layer_to_data, "VSTM:v storm motion")


# Rotate storm winds to lat/lon

# http://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/tricks.wgrib2
#
# (40) How do I get the wind speed and direction? conversion between earth and grid relative winds.
#
#      Calculating the wind speed is easy (UGRD^2 + VGRD^2)^0.5
#   use: wgrib2 IN.grb -wind_speed WND.grb
#
#      Calculating the wind direction can be tricky.
#   For global files, the UGRD is the wind to the east
#   and VGRD is the wind to the north (earth relative).
#   You can use: wgrib2 IN.grb -wind_dir WND.grb
#
#   For Lambert conformal and polar stereographic files,
#   UGRD is the wind from grid point (i,j) to (i+1,j).
#   VGRD is the wind from grid point (i,j) to (i,j+1).
#   This is call grid relative winds and -wind_dir doesn't work.
#   However, the -new_grid option can change the winds
#   to earth relative.  So step 1 is to convert the winds to earth relative.
#
#   wgrib2 IN.grb -new_grid_winds earth -new_grid `grid_defn.pl IN.grb` OUT.grb
#
#   The script grid_defn.pl returns the definition of the grid defintion of IN.grb
#   in -new_grid format.      http://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2.scripts/
#
#   Step 2 is to calculate the wind speed and direction.
#
#   wgrib2 OUT.grb -wind_speed WND.grb -wind_dir WND.grb


grid_defn = split(String(read(`perl grid_defn.pl $grib2_path`)))
# println(grid_defn)

storm_winds_temp_file = "storm_winds_latlon_aligned_tmp_$grib2_path"

(to_wgrib2, wgrib2) = open(`wgrib2 $grib2_path -i -inv /dev/null -new_grid_winds earth -new_grid $grid_defn $storm_winds_temp_file`, "w")

# storm_wind_layers = [["USTM" "u storm motion"]; ["VSTM" "v storm motion"]]
# storm_wind_layers_to_fetch = mapslices(desiredLayerToInventoryLine, storm_wind_layers, 2)

ustm_layer_to_fetch = desiredLayerToInventoryLine(["USTM" "u storm motion"])
vstm_layer_to_fetch = desiredLayerToInventoryLine(["VSTM" "v storm motion"])
println(to_wgrib2, ustm_layer_to_fetch[1] * ":" * ustm_layer_to_fetch[2])
println(to_wgrib2, vstm_layer_to_fetch[1] * ":" * vstm_layer_to_fetch[2])

close(to_wgrib2)
close(wgrib2)
wait(wgrib2)

(from_wgrib2, wgrib2) = open(`wgrib2 $storm_winds_temp_file -header -inv /dev/null -bin -`)

grid_length         = read(from_wgrib2, UInt32)
storm_motion_lon_us = read(from_wgrib2, Float32, div(grid_length, 4))
grid_length_again   = read(from_wgrib2, UInt32)

grid_length         = read(from_wgrib2, UInt32)
storm_motion_lat_vs = read(from_wgrib2, Float32, div(grid_length, 4))
grid_length_again   = read(from_wgrib2, UInt32)

# Sanity check that incoming stream is empty
if !eof(from_wgrib2)
  error("wgrib2 sending more data than expected!")
end

# storm_motion_polar_vectors_latlon_aligned = map(uv_to_r_theta, zip(storm_motion_lon_us, storm_motion_lat_vs))
# storm_motion_rs_latlon_aligned            = map(first, storm_motion_polar_vectors_latlon_aligned)
# storm_motion_thetas_latlon_aligned        = map(last,  storm_motion_polar_vectors_latlon_aligned)

# println("storm_motion_rs")
# show(stdout_limited, "text/plain", storm_motion_rs)
# println("storm_motion_rs2")
# show(stdout_limited, "text/plain", storm_motion_rs_latlon_aligned)
# println("storm_motion_thetas")
# show(stdout_limited, "text/plain", storm_motion_thetas)
# println("storm_motion_thetas2")
# show(stdout_limited, "text/plain", storm_motion_thetas_latlon_aligned)

# run(`rm $storm_winds_temp_file`)

# layer_to_data["RSTM:latlon relative storm motion speed"] = storm_motion_rs_latlon_aligned
# layer_to_data["TSTM:latlon relative storm motion angle"] = storm_motion_thetas_latlon_aligned


# Figure out latlons for grid points

all_pts = open(grid -> readdlm(grid, ','; header=false), `wgrib2 $grib2_path -end -inv /dev/null -gridout -`)

all_pts[:, 4] = [lon > 180 ? lon - 360 : lon for lon in all_pts[:, 4]]

# Grid is W to E, S to N

const grid_width  = Int64(maximum(all_pts[:,1]))
const grid_height = Int64(maximum(all_pts[:,2]))

function get_grid_i(w_to_e_col, s_to_n_row)
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

function is_on_grid(w_to_e_col, s_to_n_row)
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

function get_grid_lat_lon_for_flat_i(flat_i)
  lat = all_pts[flat_i,3]
  lon = all_pts[flat_i,4]
  (lat, lon)
end

function get_grid_lat_lon_and_flat_i(w_to_e_col, s_to_n_row)
  flat_i   = get_grid_i(w_to_e_col, s_to_n_row)
  lat, lon = get_grid_lat_lon_for_flat_i(flat_i)
  (lat, lon, flat_i)
end



# Estimate area represented by each grid point

point_areas = zeros(grid_width*grid_height,1)

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
  end
end

all_pts = hcat(all_pts, point_areas)


# Transpose features to row per point

# headers = ["i", "j", "lat", "lon", "sq_miles"]
#
# tranposed_data = zeros(size(all_pts,1), size(all_pts,2) + size(layers,1))
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



# Figure out which grid points are for training

training_pts = readdlm("grid_xys_26_miles_inside_1_mile_outside_conus.csv", ','; header=false)[:, 1:2]
training_pts_set = Set{Tuple{Int32,Int32}}(Set())

function lat_lon_to_key(lat, lon)
  (Int32(round(lat*1000)), Int32(round(lon*1000)))
end

for i in 1:size(training_pts,1)
  push!(training_pts_set, lat_lon_to_key(training_pts[i,2], training_pts[i,1]))
end

train_pts = all_pts[Bool[(lat_lon_to_key(all_pts[i, 3], all_pts[i,4]) in training_pts_set) for i in 1:size(all_pts,1)], :]

if size(train_pts,1) != length(training_pts_set)
  error("Grid error: grid used in $grib2_path does not match grid used to determine training points")
end


# Build final feature set

# Find min/max/mean/start-end-diff within 25mi of +/- 30 min storm motion

function relativize_angle(theta, ref)
  mod(theta - ref + π, 2π) - π
end

# function relativize_angle(thetaAndRef)
#   theta, ref = thetaAndRef
#   mod(theta - ref + π, 2π) - π
# end

# relative_thetas = map(thetaAndRef -> relativize_angle(thetaAndRef), zip(thetas, storm_motion_thetas))

const ONE_MINUTE = 60.0
const repeated = Base.Iterators.repeated

# Search outward in squre rings until no more points in region are found.
#
# Works for any predicate region that is:
#  (a) fully connected (not split into disconnected regions)
#  (b) no "skinny parts" that could slip between grid points
#
# Circles and extruded circles are good. (What we use.)
function square_ring_search(predicate, center_i, center_j)
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
function diamond_search(predicate, center_i, center_j)
  any_found_this_diamond  = false
  still_searching_on_grid = true
  matching_flat_is = []

  r = 0
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
      diamond = [(center_i, center_j)]
    else
      s_e = zip(center_i:center_i+r-1,    center_j-r:center_j-1)
      e_n = zip(center_i+r:-1:center_i+1, center_j:center_j+r-1)
      n_w = zip(center_i:-1:center_i-r+1, center_j+r:-1:center_j+1)
      w_s = zip(center_i-r:center_i-1,    center_j:-1:center_j-r+1)
      diamond = collect(Base.Iterators.flatten((s_e, e_n, n_w, w_s)))
    end

    for diamond_i = 1:length(diamond)
      i, j = diamond[diamond_i]
      if is_on_grid(i, j)
        still_searching_on_grid = true
        lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)
        if predicate(lat, lon)
          any_found_this_diamond = true
          push!(matching_flat_is, flat_i)
        end
      end
    end

    r += 1
  end

  # print(" $r ")

  matching_flat_is
end

tornado_rows, tornado_headers = readdlm("tornadoes.csv",','; header=true)

start_seconds_col_i, = find(tornado_headers .== "begin_time_seconds")
end_seconds_col_i,   = find(tornado_headers .== "end_time_seconds")
start_lat_col_i,     = find(tornado_headers .== "begin_lat")
start_lon_col_i,     = find(tornado_headers .== "begin_lon")
end_lat_col_i,       = find(tornado_headers .== "end_lat")
end_lon_col_i,       = find(tornado_headers .== "end_lon")

valid_start_seconds = Dates.datetime2unix(DateTime(valid_start_time))
valid_end_seconds   = Dates.datetime2unix(DateTime(valid_end_time))

relevant_tornadoes = tornado_rows[tornado_rows[:, end_seconds_col_i] .> valid_start_seconds, :]
relevant_tornadoes = relevant_tornadoes[relevant_tornadoes[:, start_seconds_col_i] .< valid_end_seconds,: ]

if size(relevant_tornadoes,1) > 0
  relevant_tornadoes = mapslices(relevant_tornadoes, 2) do row
    # tornado_start_time = TimeZones.ZonedDateTime(Dates.unix2datetime(row[start_seconds_col_i]), utc)
    # tornado_end_time   = TimeZones.ZonedDateTime(Dates.unix2datetime(row[end_seconds_col_i]), utc)
    row
  end
end

tornado_segments = map(1:size(relevant_tornadoes,1)) do i
  start_seconds = relevant_tornadoes[i, start_seconds_col_i]
  end_seconds   = relevant_tornadoes[i, end_seconds_col_i]
  duration      = end_seconds - start_seconds
  start_lat     = relevant_tornadoes[i, start_lat_col_i]
  start_lon     = relevant_tornadoes[i, start_lon_col_i]
  end_lat       = relevant_tornadoes[i, end_lat_col_i]
  end_lon       = relevant_tornadoes[i, end_lon_col_i]

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
end

println("relevant tornadoes")
show(stdout_limited, "text/plain", relevant_tornadoes)

println("tornado segments")
show(stdout_limited, "text/plain", tornado_segments)
# max_error = 0.0

out_rows = Array{Float32}[]

for j = 1:grid_height
  println("$j")
  for i = 1:grid_width
    lat, lon, flat_i = get_grid_lat_lon_and_flat_i(i, j)

    # skip if not a training point
    if lat_lon_to_key(lat, lon) in training_pts_set

      is_close_to_tornado = false
      for (tlat1, tlon1, tlat2, tlon2) in tornado_segments
        if GeoUtils.instant_distance_to_line(lat, lon, tlat1, tlon1, tlat2, tlon2) <= 25.0 * GeoUtils.METERS_PER_MILE
          is_close_to_tornado = true
        end
      end
      if is_close_to_tornado
        print("t")
      end

      lat_motion = storm_motion_lat_vs[flat_i] # m / s
      lon_motion = storm_motion_lon_us[flat_i] # m / S

      # Doesn't seem to save any time.
      # plus_30_mins_lat,  plus_30_mins_lon  = GeoUtils.instant_integrate_velocity(lat, lon,  lat_motion,  lon_motion, 30*ONE_MINUTE)
      # minus_30_mins_lat, minus_30_mins_lon = GeoUtils.instant_integrate_velocity(lat, lon, -lat_motion, -lon_motion, 30*ONE_MINUTE)
      plus_30_mins_lat,  plus_30_mins_lon  = GeoUtils.integrate_velocity(lat, lon,  lat_motion,  lon_motion, 30*ONE_MINUTE)
      minus_30_mins_lat, minus_30_mins_lon = GeoUtils.integrate_velocity(lat, lon, -lat_motion, -lon_motion, 30*ONE_MINUTE)

      # err1, errd1 = GeoUtils.compare_integrate_velocity(lat, lon,  lat_motion,  lon_motion, 30*ONE_MINUTE)
      # err2, errd2 = GeoUtils.compare_integrate_velocity(lat, lon, -lat_motion, -lon_motion, 30*ONE_MINUTE)
      #
      # if errd1 > GeoUtils.METERS_PER_MILE * .05 || errd2 > GeoUtils.METERS_PER_MILE * .05
      #   println((err1, err2, errd1/GeoUtils.METERS_PER_MILE, errd2/GeoUtils.METERS_PER_MILE))
      # end

      # println((minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat,  plus_30_mins_lon))

      # find indices within 25 miles of -30 mins to +30 mins storm path
      flat_is_within_25mi_and_30_mins_of_storm =
        diamond_search(i, j) do lat, lon
          # d = GeoUtils.instant_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon)
          # if d > 20.0 * GeoUtils.METERS_PER_MILE
          #   fast_error_pct = GeoUtils.compare_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon)
          #   global max_error = max(max_error, fast_error_pct)
          #   if fast_error_pct > 0.4
          #     println((fast_error_pct, d, lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon))
          #   end
          # end
          # GeoUtils.distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon, 1.0) <= 25.0 * GeoUtils.METERS_PER_MILE
          GeoUtils.instant_distance_to_line(lat, lon, minus_30_mins_lat, minus_30_mins_lon, plus_30_mins_lat, plus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        end

      # find indices within 25 miles of -30 mins storm location
      flat_is_within_25mi_of_storm_30_mins_ago =
        filter(flat_is_within_25mi_and_30_mins_of_storm) do candidate_flat_i
          candiate_lat, candidate_lon = get_grid_lat_lon_for_flat_i(candidate_flat_i)

          # lightning, instant, instantish, fast, fastish = GeoUtils.compare_distances(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon)
          # if instant > 0.004
          #   d = GeoUtils.distance(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon)
          #   println((d, lightning, instant, instantish, fast, fastish))
          # end
          GeoUtils.instant_distance(candiate_lat, candidate_lon, minus_30_mins_lat, minus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        end

      # find indices within 25 miles of +30 mins storm location
      flat_is_within_25mi_of_storm_30_mins_from_now =
        filter(flat_is_within_25mi_and_30_mins_of_storm) do candidate_flat_i
          candiate_lat, candidate_lon = get_grid_lat_lon_for_flat_i(candidate_flat_i)
          GeoUtils.instant_distance(candiate_lat, candidate_lon, plus_30_mins_lat, plus_30_mins_lon) <= 25.0 * GeoUtils.METERS_PER_MILE
        end


      out_row = Float32[]

      storm_r, storm_theta = u_v_to_r_theta(lon_motion, lat_motion)

      # for each layer...
      for layer_i in 1:size(layers,1)
        abbrev, desc = layers[layer_i,:]
        layer_key = abbrev * ":" * desc

        # if wind layer, relativize angle against storm motion
        if abbrev == "UGRD"
          u_layer_key = layer_key
          v_layer_key = "V" * u_layer_key[2:length(u_layer_key)

          # Need lat-lon relative u/v
          # So can compare direction correctly to take mean across multiple grid points
          # (Whether uv or polar, reference needs to be constant across points for a mean to be meaningful)

          # u_layer = layer_to_data[u_layer_key]
          # v_layer = layer_to_data[v_layer_key]
          #
          # pt_r, pt_theta = u_v_to_r_theta(u_layer[flat_i], v_layer[flat_i])
          # pt_theta_rel   = relativize_angle(pt_theta, storm_theta)
          #
          #
          # us_around_storm_path       = u_layer[flat_is_within_25mi_and_30_mins_of_storm]
          # vs_around_storm_path       = v_layer[flat_is_within_25mi_and_30_mins_of_storm]
          # us_around_30_mins_ago      = u_layer[flat_is_within_25mi_of_storm_30_mins_ago]
          # vs_around_30_mins_ago      = v_layer[flat_is_within_25mi_of_storm_30_mins_ago]
          # us_around_30_mins_from_now = u_layer[flat_is_within_25mi_of_storm_30_mins_from_now]
          # vs_around_30_mins_from_now = v_layer[flat_is_within_25mi_of_storm_30_mins_from_now]
          #
          # polar_vectors_around_storm_path = map(uv_to_r_theta, zip(us_around_storm_path, vs_around_storm_path))
          # rs_around_storm_path            = map(first, polar_vectors_around_storm_path)
          # thetas_around_storm_path        = map(last,  polar_vectors_around_storm_path)


          # u_v_to_r_theta
          # ustm_layer_to_fetch = desiredLayerToInventoryLine(["USTM" "u storm motion"])
          # vstm_layer_to_fetch = desiredLayerToInventoryLine(["VSTM" "v storm motion"])
          # println(to_wgrib2, ustm_layer_to_fetch[1] * ":" * ustm_layer_to_fetch[2])
          # println(to_wgrib2, vstm_layer_to_fetch[1] * ":" * vstm_layer_to_fetch[2])
          #
          # close(to_wgrib2)
          # close(wgrib2)
          # wait(wgrib2)
          #
          # (from_wgrib2, wgrib2) = open(`wgrib2 $storm_winds_temp_file -header -inv /dev/null -bin -`)
          #
          # grid_length         = read(from_wgrib2, UInt32)
          # storm_motion_lon_us = read(from_wgrib2, Float32, div(grid_length, 4))
          # grid_length_again   = read(from_wgrib2, UInt32)
          #
          # grid_length         = read(from_wgrib2, UInt32)
          # storm_motion_lat_vs = read(from_wgrib2, Float32, div(grid_length, 4))
          # grid_length_again   = read(from_wgrib2, UInt32)
          #
          # # Sanity check that incoming stream is empty
          # if !eof(from_wgrib2)
          #   error("wgrib2 sending more data than expected!")
          # end
          #
          # storm_motion_polar_vectors_latlon_aligned = map(uv_to_r_theta, zip(storm_motion_lon_us, storm_motion_lat_vs))
          # storm_motion_rs_latlon_aligned            = map(first, storm_motion_polar_vectors_latlon_aligned)
          # storm_motion_thetas_latlon_aligned        = map(last,  storm_motion_polar_vectors_latlon_aligned)

          # println("storm_motion_rs")
          # show(stdout_limited, "text/plain", storm_motion_rs)
          # println("storm_motion_rs2")
          # show(stdout_limited, "text/plain", storm_motion_rs_latlon_aligned)
          # println("storm_motion_thetas")
          # show(stdout_limited, "text/plain", storm_motion_thetas)
          # println("storm_motion_thetas2")
          # show(stdout_limited, "text/plain", storm_motion_thetas_latlon_aligned)

          # run(`rm $storm_winds_temp_file`)

          # layer_to_data["RSTM:latlon relative storm motion speed"] = storm_motion_rs_latlon_aligned
          # layer_to_data["TSTM:latlon relative storm motion angle"] = storm_motion_thetas_latlon_aligned

        elseif abbrev == "TGRD"

        elseif abbrev == "USTM"

        elseif abbrev == "TSTM"

        else
          layer_data = layer_to_data[layer_key]

          values_around_storm_path       = layer_data[flat_is_within_25mi_and_30_mins_of_storm]
          values_around_30_mins_ago      = layer_data[flat_is_within_25mi_of_storm_30_mins_ago]
          values_around_30_mins_from_now = layer_data[flat_is_within_25mi_of_storm_30_mins_from_now]

          # point value
          # local mean
          # local min
          # local max
          # gradient in storm direction

          push!(out_row, layer_data[flat_i])
          push!(out_row, mean(values_around_storm_path))
          push!(out_row, minimum(values_around_storm_path))
          push!(out_row, maximum(values_around_storm_path))
          push!(out_row, mean(values_around_30_mins_from_now) - mean(values_around_30_mins_ago))
        end



        # add to output

        # grab within 25 miles of -30 mins to +30 mins storm path
        # if wind layer, relativize angle against storm motion
        # find mean, min, max
        # add to output

        # grab within 25 miles of -30 mins storm location
        # if wind layer, relativize angle against storm motion
        # find mean
        # grab within 25 miles of +30 mins storm location
        # if wind layer, relativize angle against storm motion
        # find mean
        # add gradient to output
      end

      push!(out_rows, out_rows)
    end
  end
end

# println(max_error)
