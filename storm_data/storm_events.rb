require "csv"

TORNADOES_CSV_PATH   = File.expand_path("../tornadoes.csv", __FILE__)
HAIL_EVENTS_CSV_PATH = File.expand_path("../hail_events.csv", __FILE__)
WIND_EVENTS_CSV_PATH = File.expand_path("../wind_events.csv", __FILE__)

module StormEvent
  def on_ground_during(time_range)
    start_time < time_range.end &&
    end_time   > time_range.begin
  end

  def in_conus?
    ((24..50).cover?(start_lat) && (-125..-66).cover?(start_lon)) ||
    ((24..50).cover?(end_lat)   && (-125..-66).cover?(end_lon))
  end
end

class Tornado < Struct.new(:start_time, :end_time, :rating, :start_lat, :start_lon, :end_lat, :end_lon)
  include StormEvent
end

#  begin_time_str,begin_time_seconds,end_time_str,end_time_seconds,kind,inches,begin_lat,begin_lon,end_lat,end_lon
class HailEvent < Struct.new(:start_time, :end_time, :kind, :inches, :start_lat, :start_lon, :end_lat, :end_lon)
  include StormEvent
end

# begin_time_str,begin_time_seconds,end_time_str,end_time_seconds,kind,speed,speed_type,begin_lat,begin_lon,end_lat,end_lon
class WindEvent < Struct.new(:start_time, :end_time, :kind, :speed, :speed_type, :start_lat, :start_lon, :end_lat, :end_lon)
  include StormEvent
end


TORNADOES = CSV.read(TORNADOES_CSV_PATH).drop(1).map do |start_time_str, start_seconds_str, end_time_str, end_seconds_str, rating_str, start_lat_str, start_lon_str, end_lat_str, end_lon_str|
  Tornado.new(
    Time.at(Integer(start_seconds_str)).utc,
    Time.at(Integer(end_seconds_str)).utc,
    rating_str.to_i,
    Float(start_lat_str),
    Float(start_lon_str),
    Float(end_lat_str),
    Float(end_lon_str),
  )
end

HAIL_EVENTS = CSV.read(HAIL_EVENTS_CSV_PATH).drop(1).map do |start_time_str, start_seconds_str, end_time_str, end_seconds_str, kind_str, inches_str, start_lat_str, start_lon_str, end_lat_str, end_lon_str|
  HailEvent.new(
    Time.at(Integer(start_seconds_str)).utc,
    Time.at(Integer(end_seconds_str)).utc,
    kind_str,
    Float(inches_str),
    Float(start_lat_str),
    Float(start_lon_str),
    Float(end_lat_str),
    Float(end_lon_str),
  )
end

# Some wind events are not geocoded. One LSR event is geocoded as "LA,32.86,LA,32.86"
WIND_EVENTS = CSV.read(WIND_EVENTS_CSV_PATH).drop(1).map do |start_time_str, start_seconds_str, end_time_str, end_seconds_str, kind_str, speed_str, speed_type_str, start_lat_str, start_lon_str, end_lat_str, end_lon_str|
  if !start_lat_str || start_lat_str == "LA"
    start_lat = Float::NAN
    start_lon = Float::NAN
    end_lat   = Float::NAN
    end_lon   = Float::NAN
  else
    start_lat = Float(start_lat_str)
    start_lon = Float(start_lon_str)
    end_lat   = Float(end_lat_str)
    end_lon   = Float(end_lon_str)
  end
  WindEvent.new(
    Time.at(Integer(start_seconds_str)).utc,
    Time.at(Integer(end_seconds_str)).utc,
    kind_str,
    Float(speed_str),
    speed_type_str,
    start_lat,
    start_lon,
    end_lat,
    end_lon,
  )
end

STORM_EVENTS = TORNADOES + HAIL_EVENTS + WIND_EVENTS


def rap_to_time(rap_str)
  # rap_130_20161008_1800_001
  yyyy = rap_str[8..11]
  mm   = rap_str[12..13]
  dd   = rap_str[14..15]
  hh   = rap_str[17..18]
  fh   = rap_str[23..24]

  require "time"

  Time.parse("#{yyyy}-#{mm}-#{dd} #{hh}:00 +0000") + 60*60*fh.to_i
end

def rap_to_time_range(rap_str)
  rap_to_time(rap_str)-30*60...rap_to_time(rap_str)+30*60
end

HOUR = 60*60

# Ported from models/shared/TrainingShared.jl
#
# Returns a set of Time objects representing the set of hours covered by the given storm events that are in the CONUS.
def conus_event_hours_set(events, event_time_window_half_size)
  events.select(&:in_conus?).flat_map do |event|
    event_time_range = (event.start_time.to_f.round - event_time_window_half_size ... event.end_time.to_f.round + event_time_window_half_size)

    (event_time_range.begin / HOUR .. event_time_range.end / HOUR).map do |hour_from_epoch|
      hour_second = hour_from_epoch*HOUR
      if event_time_range.include?(hour_second)
        Time.at(hour_second).utc
      end
    end.compact
  end.to_set
end