module GeoUtils

export FEET_PER_METER, METERS_PER_MILE, DEGREES_PER_METER
export distance, distance_and_midpoint, integrate_velocity, waypoints, distance_to_line

import Proj4

const EARTH_RADIUS_METERS = 6_371_229.0

const FEET_PER_METER  = 100.0 / 2.54 / 12.0
const METERS_PER_MILE = 5280.0 / FEET_PER_METER

const METERS_PER_RADIAN = EARTH_RADIUS_METERS
const DEGREES_PER_METER = 360.0 / (METERS_PER_RADIAN*2*π)
const METERS_PER_DEGREE = METERS_PER_RADIAN*2*π / 360.0

# GRIB2 docs says earth shape 6 (in RAP) is "Earth assumed spherical with radius = 6,371,229.0 m"
# http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table3-2.shtml

# But we'll do our math on an ellipsoid because...it's more correct, I guess.

wgs84 = Proj4.Projection("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
# Julia wrapper doesn't set the geod by default :(
major_axis, eccentricity_squared = Proj4._get_spheroid_defn(wgs84.rep)
wgs84.geod = Proj4.geod_geodesic(major_axis, 1-sqrt(1-eccentricity_squared))

# distance(32.902, -94.0431, 32.9308, -94.0211)
function distance(lat1, lon1, lat2, lon2)
  distance, _, _ = Proj4._geod_inverse(wgs84.geod, [lon1, lat1], [lon2, lat2])
  distance
end

# Dumb flattening method.
function lightning_distance(lat1, lon1, lat2, lon2)
  mean_lat = (lat1 + lat2) / 2.0 / 180.0 * π
  dlat     = (lat2 - lat1) / 180.0 * π
  dlon     = (lon2 - lon1) / 180.0 * π

  EARTH_RADIUS_METERS * √(dlat^2 + (cos(mean_lat)dlon)^2)
end

# FCC method below with some terms removed, but performs just as well.
# And way better than haversine for the kinds of distances we are dealing with.
# Error < 0.005% for the distances we are using.
# Precondition: longitudes don't cross over (raw lon2-lon1 < 180)
function instant_distance(lat1 :: Float64, lon1 :: Float64, lat2 :: Float64, lon2 :: Float64) :: Float64
  mean_lat = (lat1 + lat2) / 2.0 / 180.0 * π
  dlat     = lat2 - lat1
  dlon     = lon2 - lon1

  k1 = 111.13209 - 0.56605cos(2*mean_lat)
  k2 = 111.41513cos(mean_lat) - 0.09455cos(3*mean_lat)

  √((k1*dlat)^2 + (k2*dlon)^2) * 1000.0
end

# Compared to actual calculation on an ellipsoid, error is greatest when
# close to the line (since geodesic follows a different path).
# For the scales we are concerned about, error is always < 0.45%.
# Precondition: longitudes don't cross over (raw lon2-lon1 < 180)
function instant_distance_to_line(lat :: Float64, lon :: Float64, lat1 :: Float64, lon1 :: Float64, lat2 :: Float64, lon2 :: Float64) :: Float64
  mean_lat = (lat + lat1 + lat2) / 3.0 / 180.0 * π

  k1 = 111.13209 - 0.56605cos(2*mean_lat)
  k2 = 111.41513cos(mean_lat) - 0.09455cos(3*mean_lat)

  # Translate so endpoint 1 is the origin.
  x2 = (lon2 - lon1) * k2
  y2 = (lat2 - lat1) * k1
  x  = (lon  - lon1) * k2
  y  = (lat  - lat1) * k1

  # Unit vector...
  segment_length = √(x2^2 + y2^2)
  ux2 = x2 / segment_length
  uy2 = y2 / segment_length

  # Project onto line
  distance_from_origin_on_line = x*ux2 + y*uy2

  if distance_from_origin_on_line <= 0.0
    # Closer to endpoint 1 (the origin)
    √(x^2 + y^2) * 1000.0
  elseif distance_from_origin_on_line >= segment_length
    # Closer to endpoint 2
    √((x-x2)^2 + (y-y2)^2) * 1000.0
  else
    # Closer to somewhere on the line
    x_proj = ux2 * distance_from_origin_on_line
    y_proj = uy2 * distance_from_origin_on_line

    √((x-x_proj)^2 + (y-y_proj)^2) * 1000.0
  end
end

# On the example grib2 file seems never to be off by more than a tenth of a mile (<0.4% error for our 25mi queries).
# But doesn't really save any time.
function instant_integrate_velocity(lat, lon, lat_m_per_s, lon_m_per_s, seconds)
  mean_lat = (lat + lat + lat_m_per_s*seconds/METERS_PER_DEGREE) / 2.0 / 180.0 * π

  k1 = 111.13209 - 0.56605cos(2*mean_lat)
  k2 = 111.41513cos(mean_lat) - 0.09455cos(3*mean_lat)

  ( lat + lat_m_per_s*seconds/1000.0/k1
  , lon + lon_m_per_s*seconds/1000.0/k2
  )
end


# FCC method, per Wikipedia https://en.wikipedia.org/wiki/Geographical_distance#Ellipsoidal_Earth_projected_to_a_plane
# Surprisingly good! Generally much less than 0.01% error over short distances, and not completely awful over long distances.
function instantish_distance(lat1, lon1, lat2, lon2)
  mean_lat = (lat1 + lat2) / 2.0 / 180.0 * π
  dlat     = lat2 - lat1
  dlon     = lon2 - lon1

  k1 = 111.13209 - 0.56605cos(2*mean_lat) + 0.00120cos(4*mean_lat)
  k2 = 111.41513cos(mean_lat) - 0.09455cos(3*mean_lat) + 0.00012cos(5*mean_lat)

  √((k1*dlat)^2 + (k2*dlon)^2) * 1000.0
end

# Haversine distance, per Wikipedia.
function fast_distance(lat1, lon1, lat2, lon2)
  lat1 = lat1/180.0*π
  lat2 = lat2/180.0*π
  half_dlat = (lat2 - lat1) / 2.0
  half_dlon = (lon2 - lon1) / 2.0 / 180.0 * π

  2.0asin(√(sin(half_dlat)^2 + cos(lat1)cos(lat2)sin(half_dlon)^2)) * METERS_PER_RADIAN
end

# Haversine distance for short distances, w/earth sphere radius based on WGS 84 ellipsoid.
# Not clearly better than Haversine.
function fastish_distance(lat1, lon1, lat2, lon2)
  lat1 = lat1/180.0*π
  lat2 = lat2/180.0*π
  half_dlat = (lat2 - lat1) / 2.0
  half_dlon = (lon2 - lon1) / 2.0 / 180.0 * π

  dradians = 2.0asin(√(sin(half_dlat)^2 + cos(lat1)cos(lat2)sin(half_dlon)^2))

  mean_lat = (lat1 + lat2) / 2.0
  semimajor_r = 6378137
  lat_flattening = 1.0/298.257223563 * (1.0 - cos(mean_lat))

  dradians * semimajor_r * (1.0 - lat_flattening)
end

function compare_distances(lat1, lon1, lat2, lon2)
  best       = distance(lat1, lon1, lat2, lon2)
  lightning  = lightning_distance(lat1, lon1, lat2, lon2)
  instant    = instant_distance(lat1, lon1, lat2, lon2)
  instantish = instantish_distance(lat1, lon1, lat2, lon2)
  fast       = fast_distance(lat1, lon1, lat2, lon2)
  fastish    = fastish_distance(lat1, lon1, lat2, lon2)

  ( abs(lightning - best)/best*100.0
  , abs(instant - best)/best*100.0
  , abs(instantish - best)/best*100.0
  , abs(fast - best)/best*100.0
  , abs(fastish-best)/best*100.0
  )
end

# distance_and_midpoint(32.902, -94.0431, 32.9308, -94.0211)
function distance_and_midpoint(lat1, lon1, lat2, lon2)
  distance, point_1_azimuth, point_2_azimuth = Proj4._geod_inverse(wgs84.geod, [lon1, lat1], [lon2, lat2])
  midpoint = deepcopy([lon1, lat1]) # call is destructive :(
  Proj4._geod_direct!(wgs84.geod, midpoint, point_1_azimuth, distance / 2.0)
  (distance, reverse(midpoint))
end

function ratio_on_segment(lat1 :: Float64, lon1 :: Float64, lat2 :: Float64, lon2 :: Float64, ratio :: Float64) :: Tuple{Float64,Float64}
  distance, point_1_azimuth, point_2_azimuth = Proj4._geod_inverse(wgs84.geod, [lon1, lat1], [lon2, lat2])
  ratio_point = deepcopy([lon1, lat1]) # call is destructive :(
  Proj4._geod_direct!(wgs84.geod, ratio_point, point_1_azimuth, distance * ratio)
  (ratio_point[2], ratio_point[1])
end


# Returns endpoint
function integrate_velocity(lat :: Float64, lon :: Float64, lat_m_per_s :: Float64, lon_m_per_s :: Float64, seconds :: Float64) :: Tuple{Float64,Float64}
  # Recall atan2 is (y,x). For geodesics, azimuth is clockwise from north.
  azimuth = 90.0 - atan2(lat_m_per_s, lon_m_per_s) * 180 / π
  m_per_s = sqrt(lat_m_per_s^2 + lon_m_per_s^2)

  point = deepcopy([lon, lat]) # call is destructive :(
  Proj4._geod_direct!(wgs84.geod, point, azimuth, m_per_s * seconds)
  (point[2], point[1])
end


function compare_integrate_velocity(lat, lon, lat_m_per_s, lon_m_per_s, seconds)
  best_lat, best_lon = integrate_velocity(lat, lon, lat_m_per_s, lon_m_per_s, seconds)
  inst_lat, inst_lon = instant_integrate_velocity(lat, lon, lat_m_per_s, lon_m_per_s, seconds)

  error_d = distance(best_lat, best_lon, inst_lat, inst_lon)
  d       = distance(lat, lon, best_lat, best_lon)

  (error_d / d * 100.0, error_d)
end


####### Translated from geographiclib geodesic.h ########

mutable struct geod_geodesicline
  lat1::Cdouble                # the starting latitude
  lon1::Cdouble                # the starting longitude
  azi1::Cdouble                # the starting azimuth
  a::Cdouble                   # the equatorial radius
  f::Cdouble                   # the flattening
  salp1::Cdouble               # sine of \e azi1
  calp1::Cdouble               # cosine of \e azi1
  a13::Cdouble                 # arc length to reference point
  s13::Cdouble                 # distance to reference point

  # internals
  b::Cdouble
  c2::Cdouble
  f1::Cdouble
  salp0::Cdouble
  calp0::Cdouble
  k2::Cdouble
  ssig1::Cdouble
  csig1::Cdouble
  dn1::Cdouble
  stau1::Cdouble
  ctau1::Cdouble
  somg1::Cdouble
  comg1::Cdouble
  A1m1::Cdouble
  A2m1::Cdouble
  A3c::Cdouble
  B11::Cdouble
  B21::Cdouble
  B31::Cdouble
  A4::Cdouble
  B41::Cdouble
  C1a::NTuple{7, Cdouble}
  C1pa::NTuple{7, Cdouble}
  C2a::NTuple{7, Cdouble}
  C3a::NTuple{6, Cdouble}
  C4a::NTuple{6, Cdouble}

  caps::Cuint              # the capabilities

  geod_geodesicline() = new()
end

GEOD_NONE          = Cuint(0)                                  # Calculate nothing
GEOD_LATITUDE      = Cuint(1<<7)   | Cuint(0)                  # Calculate latitude
GEOD_LONGITUDE     = Cuint(1<<8)   | Cuint(1<<3)               # Calculate longitude
GEOD_AZIMUTH       = Cuint(1<<9)   | Cuint(0)                  # Calculate azimuth
GEOD_DISTANCE      = Cuint(1<<10)  | Cuint(1<<0)               # Calculate distance
GEOD_DISTANCE_IN   = Cuint(1<<11)  | Cuint(1<<0) | Cuint(1<<1) # Allow distance as input
GEOD_REDUCEDLENGTH = Cuint(1<<12)  | Cuint(1<<0) | Cuint(1<<2) # Calculate reduced length
GEOD_GEODESICSCALE = Cuint(1<<13)  | Cuint(1<<0) | Cuint(1<<2) # Calculate geodesic scale
GEOD_AREA          = Cuint(1<<14)  | Cuint(1<<4)               # Calculate area
GEOD_ALL           = Cuint(0x7F80) | Cuint(0x1F)               # Calculate everything

GEOD_NOFLAGS     = Cuint(0)     # No flags
GEOD_ARCMODE     = Cuint(1<<0)  # Position given in terms of arc distance
GEOD_LONG_UNROLL = Cuint(1<<15) # Unroll the longitude

###########################################################


# waypoints(32.902, -94.0431, 32.9308, -94.0211, METERS_PER_MILE)
function waypoints(lat1, lon1, lat2, lon2, step) # step in meters

  # struct geod_geodesic g;
  # struct geod_geodesicline l;
  # double lat[101], lon[101];
  # int i;
  # geod_init(&g, 6378137, 1/298.257223563);
  # geod_inverseline(&l, &g, 40.64, -73.78, 1.36, 103.99,
  #                  GEOD_LATITUDE | GEOD_LONGITUDE);
  # for (i = 0; i <= 100; ++i) {
  #   geod_genposition(&l, GEOD_ARCMODE, i * l.a13 * 0.01,
  #                    lat + i, lon + i, 0, 0, 0, 0, 0, 0);
  #   printf("%.5f %.5f\n", lat[i], lon[i]);
  # }
  desicline = geod_geodesicline()

  ccall((:geod_inverseline, Proj4.libproj), Void,
        (Ptr{Void}, Ptr{Void}, Cdouble, Cdouble, Cdouble, Cdouble, Cuint),
        pointer_from_objref(desicline), pointer_from_objref(wgs84.geod),
        lat1, lon1, lat2, lon2, GEOD_LATITUDE | GEOD_LONGITUDE)

  degree_step = Cdouble(DEGREES_PER_METER * step)
  degree_max  = desicline.a13

  range = Cdouble(0):(degree_step):(degree_max)
  lats  = zeros(Cdouble, length(range) + 1)
  lons  = zeros(Cdouble, length(range) + 1)

  i = 1
  for degree = range
    ccall((:geod_genposition, Proj4.libproj), Void,
          (Ptr{Void}, Cuint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          pointer_from_objref(desicline), GEOD_ARCMODE, degree, pointer(lats,i), pointer(lons,i), Ptr{Cdouble}(0), Ptr{Cdouble}(0), Ptr{Cdouble}(0), Ptr{Cdouble}(0), Ptr{Cdouble}(0), Ptr{Cdouble}(0));

    i += 1
  end
  lats[i] = lat2
  lons[i] = lon2

  return zip(lats, lons)
end




# max_error assumes triangle inequality holds, which, well, it should for all our queries. Not smart enough to know if it does in general.
# distance_to_line(32.902, -94.0431, 32.902, -94.0431, 32.9308, -94.0211, 1.0)
function distance_to_line(lat, lon, lat1, lon1, lat2, lon2, max_error)
  d1 = distance(lat, lon, lat1, lon1)
  d2 = distance(lat, lon, lat2, lon2)

  line_length, (midLat, midLon) = distance_and_midpoint(lat1, lon1, lat2, lon2)

  if d1 < d2
    if line_length < max_error
      d1
    else
      distance_to_line(lat, lon, lat1, lon1, midLat, midLon, max_error)
    end
  else
    if line_length < max_error
      d2
    else
      distance_to_line(lat, lon, midLat, midLon, lat2, lon2, max_error)
    end
  end
end


function compare_distance_to_line(lat, lon, lat1, lon1, lat2, lon2)
  best      = distance_to_line(lat, lon, lat1, lon1, lat2, lon2, 1.0) # 1 meter
  instant   = instant_distance_to_line(lat, lon, lat1, lon1, lat2, lon2)
  # one_meter = distance_to_line(lat, lon, lat1, lon1, lat2, lon2, 1.0)

  abs(instant - best)/best*100.0
end


end