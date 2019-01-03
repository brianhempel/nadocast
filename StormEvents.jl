module StormEvents

import DelimitedFiles

push!(LOAD_PATH, ".")
import GeoUtils
# import TimeZones

# const utc = TimeZones.tz"UTC"

struct Tornado
  start_seconds_from_utc_epoch :: Int64
  end_seconds_from_utc_epoch   :: Int64
  duration_seconds             :: Int64
  start_latlon                 :: Tuple{Float64, Float64}
  end_latlon                   :: Tuple{Float64, Float64}
end

const tornadoes = begin
  # println("Loading tornadoes...")

  tornado_rows, tornado_headers = DelimitedFiles.readdlm("storm_data/tornadoes.csv",','; header=true)

  tornado_headers = tornado_headers[1,:] # 1x9 array to 9-element vector.

  start_seconds_col_i = findfirst(isequal("begin_time_seconds"), tornado_headers)
  end_seconds_col_i   = findfirst(isequal("end_time_seconds"), tornado_headers)
  start_lat_col_i     = findfirst(isequal("begin_lat"), tornado_headers)
  start_lon_col_i     = findfirst(isequal("begin_lon"), tornado_headers)
  end_lat_col_i       = findfirst(isequal("end_lat"), tornado_headers)
  end_lon_col_i       = findfirst(isequal("end_lon"), tornado_headers)

  row_to_tornado(row) = begin
    start_seconds = row[start_seconds_col_i]
    end_seconds   = row[end_seconds_col_i]
    duration      = end_seconds - start_seconds
    start_latlon  = (row[start_lat_col_i], row[start_lon_col_i])
    end_latlon    = (row[end_lat_col_i],   row[end_lon_col_i])

    Tornado(start_seconds, end_seconds, duration, start_latlon, end_latlon)
  end

  mapslices(row_to_tornado, tornado_rows, dims = [2])[:,1]
end :: Vector{Tornado}

# Returns list of (start_latlon, end_latlon) of where the tornadoes were around during the time period.
function tornado_segments_around_time(seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}
  period_start_seconds = seconds_from_utc_epoch - seconds_before_and_after
  period_end_seconds   = seconds_from_utc_epoch + seconds_before_and_after

  is_relevant_tornado(tornado) = begin
    tornado.end_seconds_from_utc_epoch   > period_start_seconds &&
    tornado.start_seconds_from_utc_epoch < period_end_seconds
  end

  relevant_tornadoes = filter(is_relevant_tornado, tornadoes)

  tornado_to_segment(tornado) = begin
    start_seconds = tornado.start_seconds_from_utc_epoch
    end_seconds   = tornado.end_seconds_from_utc_epoch
    duration      = tornado.duration_seconds
    start_latlon  = tornado.start_latlon
    end_latlon    = tornado.end_latlon

    # Turns out no special case is needed for tornadoes of 0 duration.

    if start_seconds >= period_start_seconds
      seg_start_latlon = start_latlon
    else
      start_ratio = Float64(period_start_seconds - start_seconds) / duration
      seg_start_latlon = GeoUtils.ratio_on_segment(start_latlon, end_latlon, start_ratio)
    end

    if end_seconds <= period_end_seconds
      seg_end_latlon = end_latlon
    else
      end_ratio = Float64(period_end_seconds - start_seconds) / duration
      seg_end_latlon = GeoUtils.ratio_on_segment(start_latlon, end_latlon, end_ratio)
    end

    (seg_start_latlon, seg_end_latlon)
  end

  map(tornado_to_segment, relevant_tornadoes)
end

end # module StormEvents