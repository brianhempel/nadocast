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
  knots     :: Float64
  sustained :: Bool # false == gust
  measured  :: Bool
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

is_severe_tornado(tornado) = true
is_severe_wind(wind_event) = wind_event.severity.knots  >= 50.0
is_severe_hail(hail_event) = hail_event.severity.inches >= 1.0

is_estimated_wind(wind_event) = !wind_event.severity.measured
is_measured_wind(wind_event)  = wind_event.severity.measured

is_sig_tornado(tornado) = tornado.severity.ef_rating >= 2
is_sig_wind(wind_event) = wind_event.severity.knots  >= 65.0
is_sig_hail(hail_event) = hail_event.severity.inches >= 2.0


function seconds_to_convective_days_since_epoch_utc(seconds_from_epoch_utc :: Int64) :: Int64
  fld(seconds_from_epoch_utc - 12*HOUR, DAY)
end

function convective_days_since_epoch_to_seconds_utc(day_i :: Int64) :: Int64
  day_i * DAY + 12*HOUR
end

function start_time_in_convective_days_since_epoch_utc(event :: Event) :: Int64
  seconds_to_convective_days_since_epoch_utc(event.start_seconds_from_epoch_utc)
end

function end_time_in_convective_days_since_epoch_utc(event :: Event) :: Int64
  seconds_to_convective_days_since_epoch_utc(event.end_seconds_from_epoch_utc)
end


_tornadoes                          = nothing
_wind_events                        = nothing
_hail_events                        = nothing
_conus_tornado_events               = nothing
_conus_wind_events                  = nothing
_conus_hail_events                  = nothing
_conus_events                       = nothing # not-quite severe events are still used for the ±1hr, 100mi radius of near storm negative data
_conus_severe_wind_events           = nothing
_conus_estimated_severe_wind_events = nothing
_conus_measured_severe_wind_events  = nothing
_conus_severe_hail_events           = nothing
_conus_severe_events                = nothing
_conus_sig_tornado_events           = nothing
_conus_sig_wind_events              = nothing
_conus_estimated_sig_wind_events    = nothing
_conus_measured_sig_wind_events     = nothing
_conus_sig_hail_events              = nothing
_conus_sig_severe_events            = nothing

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
  speed_type_col_i    = findfirst(isequal("speed_type"), event_headers)
  source_col_i        = findfirst(isequal("source"), event_headers)
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
        WindSeverity(row[knots_col_i] == -1 ? 50.0 : row[knots_col_i], row[speed_type_col_i] == "sustained", row[source_col_i] == "measured")
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


function conus_severe_wind_events() :: Vector{Event}
  global _conus_severe_wind_events
  isnothing(_conus_severe_wind_events) && (_conus_severe_wind_events = filter(is_severe_wind, conus_wind_events()))
  _conus_severe_wind_events
end

function conus_estimated_severe_wind_events() :: Vector{Event}
  global _conus_estimated_severe_wind_events
  isnothing(_conus_estimated_severe_wind_events) && (_conus_estimated_severe_wind_events = filter(is_estimated_wind, conus_severe_wind_events()))
  _conus_estimated_severe_wind_events
end

function conus_measured_severe_wind_events() :: Vector{Event}
  global _conus_measured_severe_wind_events
  isnothing(_conus_measured_severe_wind_events) && (_conus_measured_severe_wind_events = filter(is_measured_wind, conus_severe_wind_events()))
  _conus_measured_severe_wind_events
end

function conus_severe_hail_events() :: Vector{Event}
  global _conus_severe_hail_events
  isnothing(_conus_severe_hail_events) && (_conus_severe_hail_events = filter(is_severe_hail, conus_hail_events()))
  _conus_severe_hail_events
end

function conus_severe_events() :: Vector{Event}
  global _conus_severe_events

  if isnothing(_conus_severe_events)
    _conus_severe_events = sort(vcat(conus_tornado_events(), conus_severe_wind_events(), conus_severe_hail_events()), by = (event -> event.start_seconds_from_epoch_utc))
  end

  _conus_severe_events
end


function conus_sig_tornado_events() :: Vector{Event}
  global _conus_sig_tornado_events
  isnothing(_conus_sig_tornado_events) && (_conus_sig_tornado_events = filter(is_sig_tornado, conus_tornado_events()))
  _conus_sig_tornado_events
end

function conus_sig_wind_events() :: Vector{Event}
  global _conus_sig_wind_events
  isnothing(_conus_sig_wind_events) && (_conus_sig_wind_events = filter(is_sig_wind, conus_wind_events()))
  _conus_sig_wind_events
end

function conus_estimated_sig_wind_events() :: Vector{Event}
  global _conus_estimated_sig_wind_events
  isnothing(_conus_estimated_sig_wind_events) && (_conus_estimated_sig_wind_events = filter(is_sig_wind, conus_estimated_severe_wind_events()))
  _conus_estimated_sig_wind_events
end

function conus_measured_sig_wind_events() :: Vector{Event}
  global _conus_measured_sig_wind_events
  isnothing(_conus_measured_sig_wind_events) && (_conus_measured_sig_wind_events = filter(is_sig_wind, conus_measured_severe_wind_events()))
  _conus_measured_sig_wind_events
end

function conus_sig_hail_events() :: Vector{Event}
  global _conus_sig_hail_events
  isnothing(_conus_sig_hail_events) && (_conus_sig_hail_events = filter(is_sig_hail, conus_hail_events()))
  _conus_sig_hail_events
end

function conus_sig_severe_events() :: Vector{Event}
  global _conus_sig_severe_events

  if isnothing(_conus_sig_severe_events)
    _conus_sig_severe_events = sort(vcat(conus_sig_tornado_events(), conus_sig_wind_events(), conus_sig_hail_events()), by = (event -> event.start_seconds_from_epoch_utc))
  end

  _conus_sig_severe_events
end



# Returns a data layer on the grid with 0.0/1.0 indicators of points within x miles of any storm event
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

  Threads.@threads :static for grid_i in 1:length(grid.latlons)
    out[grid_i] = is_near_event(grid.latlons[grid_i]) ? 1.0f0 : 0.0f0
  end

  out
end

# Returns a data layer on the grid with 0.0 to 1.0 indicators of points within x miles of any storm event
function grid_to_adjusted_event_neighborhoods(events :: Vector{Event}, grid :: Grids.Grid, normalization_grid :: Grids.Grid, gridded_normalization :: Vector{Float32}, miles :: Float64, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Float32}
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

  Threads.@threads :static for grid_i in 1:length(grid.latlons)
    pt_out = 0f0

    latlon = grid.latlons[grid_i]
    if is_near_event(latlon)
      for (latlon1, latlon2) in event_segments
        meters_away = GeoUtils.instant_meters_to_line(latlon, latlon1, latlon2)
        if meters_away <= miles * GeoUtils.METERS_PER_MILE
          factor1 = Grids.lookup_nearest(normalization_grid, gridded_normalization, latlon1)
          factor2 = Grids.lookup_nearest(normalization_grid, gridded_normalization, latlon2)
          factor = 0.5f0 * factor1 + 0.5f0 * factor2
          pt_out += factor
        end
      end
    end

    out[grid_i] = min(1f0, pt_out)
  end

  out
end

# # Returns list of (start_latlon, end_latlon) of where the tornadoes were around during the time period.
# function tornado_segments_around_time(seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}
#   event_segments_around_time(tornadoes(), seconds_from_utc_epoch, seconds_before_and_after)
# end

# f should be a function that take an indices_range and returns a tuple of reduction values
#
# parallel_iterate will unzip those tuples into a tuple of arrays of reduction values and return that.
function parallel_iterate(f, count)
  thread_results = Vector{Any}(undef, Threads.nthreads())

  Threads.@threads :static for thread_i in 1:Threads.nthreads()
  # for thread_i in 1:Threads.nthreads()
    start = div((thread_i-1) * count, Threads.nthreads()) + 1
    stop  = div( thread_i    * count, Threads.nthreads())
    thread_results[thread_i] = f(start:stop)
  end

  if isa(thread_results[1], Tuple)
    # Mangling so you get a tuple of arrays.
    Tuple(collect.(zip(thread_results...)))
  else
    thread_results
  end
end

# probably only a win if the predicate is expensive or only a few items pass the predicate
function parallel_filter(f, arr)
  arrs = parallel_iterate(length(arr)) do thread_range
    filter(f, @view arr[thread_range])
  end

  vcat(arrs...)
end

function event_segments_around_time(events :: Vector{Event}, seconds_from_utc_epoch :: Int64, seconds_before_and_after :: Int64) :: Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}
  period_start_seconds = seconds_from_utc_epoch - seconds_before_and_after
  period_end_seconds   = seconds_from_utc_epoch + seconds_before_and_after

  is_relevant_event(event) = begin
    (event.end_seconds_from_epoch_utc  > period_start_seconds &&
    event.start_seconds_from_epoch_utc < period_end_seconds) ||
    # Zero-duration events exactly on the boundary count in the later period
    (event.start_seconds_from_epoch_utc == period_start_seconds && event.end_seconds_from_epoch_utc == period_start_seconds)
  end

  relevant_events = parallel_filter(is_relevant_event, events)

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
      # This math is correct
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