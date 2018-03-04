module GeoUtils

export FEET_PER_METER, METERS_PER_MILE, DEGREES_PER_METER
export distance, distance_and_midpoint, waypoints, distance_to_line

import Proj4

const FEET_PER_METER  = 100.0 / 2.54 / 12.0
const METERS_PER_MILE = 5280.0 / FEET_PER_METER

const DEGREES_PER_METER = 360.0 / (6_371_229*2*pi)

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

# distance_and_midpoint(32.902, -94.0431, 32.9308, -94.0211)
function distance_and_midpoint(lat1, lon1, lat2, lon2)
  distance, point_1_azimuth, point_2_azimuth = Proj4._geod_inverse(wgs84.geod, [lon1, lat1], [lon2, lat2])
  midpoint = deepcopy([lon1, lat1]) # call is destructive :(
  Proj4._geod_direct!(wgs84.geod, midpoint, mean([point_1_azimuth,point_2_azimuth]), distance / 2.0)
  (distance, reverse(midpoint))
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


end