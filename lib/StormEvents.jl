module StormEvents

import Dates
import DelimitedFiles

push!(LOAD_PATH, @__DIR__)
import Conus
import GeoUtils
import Grids

HOUR = 60*60
DAY  = 24*HOUR

struct TornadoSeverity
  ef_rating :: Int64
end

struct WindSeverity
  knots :: Float64
end

struct HailSeverity
  inches :: Float64
end

struct Event
  start_seconds_from_epoch_utc :: Int64
  end_seconds_from_epoch_utc   :: Int64
  start_latlon                 :: Tuple{Float64, Float64}
  end_latlon                   :: Tuple{Float64, Float64}
  severity                     :: Union{Nothing, TornadoSeverity, WindSeverity, HailSeverity}
end

function seconds_to_convective_days_since_epoch_utc(seconds_from_epoch_utc :: Int64) :: Int64
  fld(seconds_from_epoch_utc - 12*HOUR, DAY)
end

function start_time_in_convective_days_since_epoch_utc(event :: Event) :: Int64
  seconds_to_convective_days_since_epoch_utc(event.start_seconds_from_epoch_utc)
end

function end_time_in_convective_days_since_epoch_utc(event :: Event) :: Int64
  seconds_to_convective_days_since_epoch_utc(event.end_seconds_from_epoch_utc)
end


_tornadoes            = nothing
_wind_events          = nothing
_hail_events          = nothing
_conus_tornado_events = nothing
_conus_wind_events    = nothing
_conus_hail_events    = nothing
_conus_events         = nothing

function event_looks_okay(event :: Event) :: Bool
  duration = event.end_seconds_from_epoch_utc - event.start_seconds_from_epoch_utc
  if duration >= 4*HOUR
    println("Event starting $(Dates.unix2datetime(event.start_seconds_from_epoch_utc)) ending $(Dates.unix2datetime(event.end_seconds_from_epoch_utc)) is $(duration / HOUR) hours long! discarding")
    false
  elseif duration < 0
    println("Event starting $(Dates.unix2datetime(event.start_seconds_from_epoch_utc)) ending $(Dates.unix2datetime(event.end_seconds_from_epoch_utc)) is $(duration / MINUTE) minutes long! discarding")
    false
  else
    true
  end
end

function read_events_csv(path) ::Vector{Event}
  event_rows, event_headers = DelimitedFiles.readdlm(path, ','; header=true)

  event_headers = event_headers[1,:] # 1x9 array to 9-element vector.

  start_seconds_col_i = findfirst(isequal("begin_time_seconds"), event_headers)
  end_seconds_col_i   = findfirst(isequal("end_time_seconds"), event_headers)
  start_lat_col_i     = findfirst(isequal("begin_lat"), event_headers)
  start_lon_col_i     = findfirst(isequal("begin_lon"), event_headers)
  end_lat_col_i       = findfirst(isequal("end_lat"), event_headers)
  end_lon_col_i       = findfirst(isequal("end_lon"), event_headers)
  ef_col_i            = findfirst(isequal("f_scale"), event_headers)
  knots_col_i         = findfirst(isequal("speed"), event_headers)
  inches_col_i        = findfirst(isequal("inches"), event_headers)

  row_to_event(row) = begin
    start_seconds = row[start_seconds_col_i]
    end_seconds   = row[end_seconds_col_i]

    if isa(row[start_lat_col_i], Real)
      start_latlon  = (row[start_lat_col_i], row[start_lon_col_i])
      end_latlon    = (row[end_lat_col_i],   row[end_lon_col_i])
    elseif row[start_lat_col_i] == "" || row[start_lat_col_i] == "LA" || row[start_lat_col_i] == "NJ" || row[start_lat_col_i] == "TN"
      # Some wind events are not geocoded. One LSR event is geocoded as "LA,32.86,LA,32.86"
      start_latlon = (NaN, NaN)
      end_latlon   = (NaN, NaN)
    else
      # If some wind events are not geocoded, DelimitedFiles treats the column as strings, I believe.
      start_latlon  = (parse(Float64, row[start_lat_col_i]), parse(Float64, row[start_lon_col_i]))
      end_latlon    = (parse(Float64, row[end_lat_col_i]),   parse(Float64, row[end_lon_col_i]))
    end

    severity =
      if !isnothing(ef_col_i)
        TornadoSeverity(row[ef_col_i] == "" ? 0 : row[ef_col_i])
      elseif !isnothing(knots_col_i)
        WindSeverity(row[knots_col_i] == -1 ? 50.0 : row[knots_col_i])
      elseif !isnothing(inches_col_i)
        HailSeverity(row[inches_col_i])
      else
        nothing
      end

    Event(start_seconds, end_seconds, start_latlon, end_latlon, severity)
  end

  events_raw = mapslices(row_to_event, event_rows, dims = [2])[:,1]
  filter(event_looks_okay, events_raw)
end

function tornadoes() :: Vector{Event}
  global _tornadoes

  if isnothing(_tornadoes)
    _tornadoes = begin
      println("Loading tornadoes...")
      read_events_csv((@__DIR__) * "/../storm_data/tornadoes.csv")
    end
  end

  _tornadoes
end

function wind_events() :: Vector{Event}
  global _wind_events

  if isnothing(_wind_events)
    _wind_events = begin
      println("Loading wind events...")
      read_events_csv((@__DIR__) * "/../storm_data/wind_events.csv")
    end
  end

  _wind_events
end

function hail_events() :: Vector{Event}
  global _hail_events

  if isnothing(_hail_events)
    _hail_events = begin
      println("Loading hail events...")
      read_events_csv((@__DIR__) * "/../storm_data/hail_events.csv")
    end
  end

  _hail_events
end

function conus_tornado_events() :: Vector{Event}
  global _conus_tornado_events

  if isnothing(_conus_tornado_events)
    _conus_tornado_events = filter(tornadoes()) do tornado
      # Exclude Alaska, Hawaii, Puerto Rico
      Conus.is_in_conus_bounding_box(tornado.start_latlon) || Conus.is_in_conus_bounding_box(tornado.end_latlon)
    end
  end

  _conus_tornado_events
end

function conus_wind_events() :: Vector{Event}
  global _conus_wind_events

  if isnothing(_conus_wind_events)
    _conus_wind_events = filter(wind_events()) do wind_event
      # Exclude Alaska, Hawaii, Puerto Rico
      Conus.is_in_conus_bounding_box(wind_event.start_latlon) || Conus.is_in_conus_bounding_box(wind_event.end_latlon)
    end
  end

  _conus_wind_events
end

function conus_hail_events() :: Vector{Event}
  global _conus_hail_events

  if isnothing(_conus_hail_events)
    _conus_hail_events = filter(hail_events()) do hail_event
      # Exclude Alaska, Hawaii, Puerto Rico
      Conus.is_in_conus_bounding_box(hail_event.start_latlon) || Conus.is_in_conus_bounding_box(hail_event.end_latlon)
    end
  end

  _conus_hail_events
end

function conus_events() :: Vector{Event}
  global _conus_events

  if isnothing(_conus_events)
    _conus_events = sort(vcat(conus_tornado_events(), conus_wind_events(), conus_hail_events()), by = (event -> event.start_seconds_from_epoch_utc))
  end

  _conus_events
end


# Returns a data layer on the grid with 0.0/1.0 indicators of points within x miles of the tornadoes
# The grid is still square: "CONUS" here only means Puerto Rico/Alaska/Hawaii etc events have been filtered out
function grid_to_conus_tornado_neighborhoods(grid :: Grids.Grid, miles :: Float64, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64; rating_range = 0:5) :: Vector{Float32}
  events = filter(event -> event.severity.ef_rating in rating_range, conus_tornado_events())
  grid_to_event_neighborhoods(events, grid, miles, seconds_from_utc_epoch, seconds_before_and_after)
end

# Returns a data layer on the grid with 0.0/1.0 indicators of points within x miles of any storm event
# The grid is still square: "CONUS" here only means Puerto Rico/Alaska/Hawaii etc events have been filtered out
function grid_to_conus_event_neighborhoods(grid :: Grids.Grid, miles :: Float64, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Float32}
  grid_to_event_neighborhoods(conus_events(), grid, miles, seconds_from_utc_epoch, seconds_before_and_after)
end



function grid_to_event_neighborhoods(events :: Vector{Event}, grid :: Grids.Grid, miles :: Float64, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Float32}
  event_segments = event_segments_around_time(events, seconds_from_utc_epoch, seconds_before_and_after)

  is_near_event(latlon) = begin
    is_near = false

    for (latlon1, latlon2) in event_segments
      meters_away = GeoUtils.instant_meters_to_line(latlon, latlon1, latlon2)
      if meters_away <= miles * GeoUtils.METERS_PER_MILE
        is_near = true
      end
    end

    is_near
  end

  out = Vector{Float32}(undef, length(grid.latlons))

  Threads.@threads for grid_i in 1:length(grid.latlons)
    out[grid_i] = is_near_event(grid.latlons[grid_i]) ? 1.0f0 : 0.0f0
  end

  out
end

# # Returns list of (start_latlon, end_latlon) of where the tornadoes were around during the time period.
# function tornado_segments_around_time(seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}
#   event_segments_around_time(tornadoes(), seconds_from_utc_epoch, seconds_before_and_after)
# end

function event_segments_around_time(events :: Vector{Event}, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}
  period_start_seconds = seconds_from_utc_epoch - seconds_before_and_after
  period_end_seconds   = seconds_from_utc_epoch + seconds_before_and_after

  is_relevant_event(event) = begin
    event.end_seconds_from_epoch_utc   > period_start_seconds &&
    event.start_seconds_from_epoch_utc < period_end_seconds
  end

  relevant_events = filter(is_relevant_event, events)

  event_to_segment(event) = begin
    start_seconds = event.start_seconds_from_epoch_utc
    end_seconds   = event.end_seconds_from_epoch_utc
    start_latlon  = event.start_latlon
    end_latlon    = event.end_latlon

    duration = event.end_seconds_from_epoch_utc - event.start_seconds_from_epoch_utc

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

  map(event_to_segment, relevant_events)
end

# Set of seconds, each on the hour.
function event_hours_set_in_seconds_from_epoch_utc(events, event_time_window_half_size)
  event_hours_set = Set{Int64}()

  for event in events
    event_time_range =
      (event.start_seconds_from_epoch_utc - event_time_window_half_size):(event.end_seconds_from_epoch_utc + event_time_window_half_size - 1)

    for hour_from_epoch in fld(event_time_range.start, HOUR):fld(event_time_range.stop, HOUR)
      hour_second = hour_from_epoch*HOUR
      if hour_second in event_time_range
        push!(event_hours_set, hour_second)
      end
    end
  end

  event_hours_set
end


end # module StormEvents