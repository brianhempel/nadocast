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

grib2_path = "rap_130_20170515_0000_001.grb2"

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

show(stdout_limited, "text/plain", inventory_lines)

inventory_lines_normalized =
  mapslices(normalize_line, inventory_lines, 2)

show(stdout_limited, "text/plain", inventory_lines_normalized)

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

show(stdout_limited, "text/plain", layers_to_fetch)

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
  if u == 0.0 && v == 0.0
    return (0.0, 0.0)
  end
  r     = sqrt(u^2 + v^2)
  theta = atan2(v, u) # Angle, in radians
  theta = mod(theta + π, 2π) - π # Turns π into -π (otherwise a passthrough here)
  (r, theta)
end

storm_motion_us = layer_to_data["USTM:u storm motion"]
storm_motion_vs = layer_to_data["VSTM:v storm motion"]

storm_motion_polar_vectors = map(uv_to_r_theta, zip(storm_motion_us, storm_motion_vs))
storm_motion_rs     = map(first, storm_motion_polar_vectors)
storm_motion_thetas = map(last,  storm_motion_polar_vectors)

uv_layers_to_relativize = [
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

function relativize_angle(thetaAndRef)
  theta, ref = thetaAndRef
  mod(theta - ref + π, 2π) - π
end

for (u_layer_key, v_layer_key) in uv_layers_to_relativize
  us = layer_to_data[u_layer_key]
  vs = layer_to_data[v_layer_key]

  polar_vectors   = map(uv_to_r_theta, zip(us, vs))
  rs              = map(first, storm_motion_polar_vectors)
  thetas          = map(last,  storm_motion_polar_vectors)
  relative_thetas = map(thetaAndRef -> relativize_angle(thetaAndRef), zip(thetas, storm_motion_thetas))

  delete!(layer_to_data, u_layer_key)
  delete!(layer_to_data, v_layer_key)

  r_layer_key     = replace(u_layer_key, r"^U", "R")
  theta_layer_key = replace(u_layer_key, r"^U", "T")

  layer_to_data[r_layer_key]     = rs
  layer_to_data[theta_layer_key] = relative_thetas
end

delete!(layer_to_data, "USTM:u storm motion")
delete!(layer_to_data, "VSTM:v storm motion")


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
println(grid_defn)

storm_winds_temp_file = "storm_winds_latlon_aligned_tmp.grib2"

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

storm_motion_polar_vectors2 = map(uv_to_r_theta, zip(storm_motion_lon_us, storm_motion_lat_vs))
storm_motion_rs2     = map(first, storm_motion_polar_vectors2)
storm_motion_thetas2 = map(last,  storm_motion_polar_vectors2)

println("storm_motion_rs")
show(stdout_limited, "text/plain", storm_motion_rs)
println("storm_motion_rs2")
show(stdout_limited, "text/plain", storm_motion_rs2)
println("storm_motion_thetas")
show(stdout_limited, "text/plain", storm_motion_thetas)
println("storm_motion_thetas2")
show(stdout_limited, "text/plain", storm_motion_thetas2)

run(`rm $storm_winds_temp_file`)

layer_to_data["RSTM:latlon relative storm motion speed"] = storm_motion_rs2
layer_to_data["TSTM:latlon relative storm motion angle"] = storm_motion_thetas2

# Find min/max/mean/start-end-diff within 25mi of +/- 30 min storm motion


