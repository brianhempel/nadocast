import Dates

# Commented out parts are to figure out when severe gusts do not get coded as t-storm reports

# Source: https://www.nhc.noaa.gov/data/#hurdat
#
# Cite: Landsea, C. W. and J. L. Franklin, 2013: Atlantic Hurricane Database Uncertainty and Presentation of a New Database Format. Mon. Wea. Rev., 141, 3576-3592.
const atlantic_in_path = joinpath(@__DIR__, "hurdat2-1851-2022-050423.txt")
const pacific_in_path  = joinpath(@__DIR__, "hurdat2-nepac-1949-2022-050423.txt")
const out_path         = joinpath(@__DIR__, "tropical_cyclones_2018-2022.csv")

try_parse_i64(str) = try
  parse(Int64, str)
catch e
  -1
end

tracks_by_storm_id = Dict{String, Any}()

time_str(datetime) = replace(string(datetime),  "T" => " ") * " UTC"

function do_it()
  # There are two header lines. We can ignore the second
  headers      = nothing
  units_header = nothing
  storm_id     = nothing
  name         = nothing

  for line in eachline(`cat $atlantic_in_path $pacific_in_path`)
    row = strip.(split(line, ','))

    if length(row) == 4
      # new storm
      storm_id, name, _, _ = row
      continue
    else
      yyyymmdd, hhmm, _,
      status,
      lat_str, lon_str,
      knots,
      _mb,
      ne_34knot_radius_nm_str,
      se_34knot_radius_nm_str,
      sw_34knot_radius_nm_str,
      nw_34knot_radius_nm_str,
      _ = row
    end

    year  = parse(Int64, yyyymmdd[1:4])
    month = parse(Int64, yyyymmdd[5:6])
    day   = parse(Int64, yyyymmdd[7:8])
    hour  = parse(Int64, hhmm[1:2])
    min   = parse(Int64, hhmm[3:4])

    year >= 2018 && year <= 2022 || continue

    print("\r$(yyyymmdd)")

    lat_str = endswith(lat_str, "N") ?       replace(lat_str, "N" => "") : "-" * replace(lat_str, "S" => "")
    lon_str = endswith(lon_str, "W") ? "-" * replace(lon_str, "W" => "") :       replace(lon_str, "E" => "")

    radii = [
      parse(Int64, ne_34knot_radius_nm_str),
      parse(Int64, se_34knot_radius_nm_str),
      parse(Int64, sw_34knot_radius_nm_str),
      parse(Int64, nw_34knot_radius_nm_str),
    ]

    track_pt = (
      Dates.DateTime(year, month, day, hour, min),
      storm_id,
      name,
      status,
      lat_str,
      lon_str,
      parse(Int64, knots),
      maximum(radii)
    )

    if !haskey(tracks_by_storm_id, storm_id)
      tracks_by_storm_id[storm_id] = []
    end
    push!(tracks_by_storm_id[storm_id], track_pt)
  end

  open(out_path, "w") do out
    println(out, join([
      "begin_time_str",
      "begin_time_seconds",
      "end_time_str",
      "end_time_seconds",
      "id",
      "name",
      "status",
      "knots",
      "max_radius_34_knot_winds_nmiles",
      "begin_lat",
      "begin_lon",
      "end_lat",
      "end_lon",
      # "in_conus_bounding_box",
      # "wind_events_within_100mi",
      # "wind_events_within_200mi",
      # "wind_events_within_250mi",
      # "wind_events_within_300mi",
      # "wind_events_within_400mi",
      # "wind_events_within_500mi",
      # "severe_gusts_within_100mi",
      # "severe_gusts_within_200mi",
      # "severe_gusts_within_250mi",
      # "severe_gusts_within_300mi",
      # "severe_gusts_within_400mi",
      # "severe_gusts_within_500mi",
    ], ","))

    for (_, track_pts) in sort(collect(tracks_by_storm_id), by = (id_tps -> id_tps[2][1][1]))
      last_track_pt = nothing
      for track_pt in track_pts
        if !isnothing(last_track_pt)
          time_1, storm_id, name, status_1,  lat_str_1, lon_str_1, knots_1, radii_1 = last_track_pt
          time_2, _,        _,    _status_2, lat_str_2, lon_str_2, knots_2, radii_2 = track_pt

          print("\r$(time_str(time_1))")

          # begin_latlon = parse.(Float64, (lat_str_1, lon_str_1))
          # end_latlon   = parse.(Float64, (lat_str_2, lon_str_2))

          println(out, join([
            time_str(time_1),
            Int64(Dates.datetime2unix(time_1)),
            time_str(time_2),
            Int64(Dates.datetime2unix(time_2)),
            storm_id,
            name,
            status_1,
            max(knots_1, knots_2),
            max(radii_1, radii_2),
            lat_str_1,
            lon_str_1,
            lat_str_2,
            lon_str_2,
            # Grids.is_in_conus_bounding_box(begin_latlon) || Grids.is_in_conus_bounding_box(end_latlon),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 100, time_1, time_2, begin_latlon, end_latlon)),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 200, time_1, time_2, begin_latlon, end_latlon)),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 250, time_1, time_2, begin_latlon, end_latlon)),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 300, time_1, time_2, begin_latlon, end_latlon)),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 400, time_1, time_2, begin_latlon, end_latlon)),
            # length(WindEvents.number_of_events_near_segment(WindEvents.conus_severe_wind_events, 500, time_1, time_2, begin_latlon, end_latlon)),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 100, time_1, time_2, begin_latlon, end_latlon),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 200, time_1, time_2, begin_latlon, end_latlon),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 250, time_1, time_2, begin_latlon, end_latlon),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 300, time_1, time_2, begin_latlon, end_latlon),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 400, time_1, time_2, begin_latlon, end_latlon),
            # Gusts.number_of_gusts_near_segment(Gusts.severe_gusts, 500, time_1, time_2, begin_latlon, end_latlon),
          ], ","))
        end
        last_track_pt = track_pt
      end
    end
  end
end

do_it()