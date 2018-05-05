require "csv"

class Tornado < Struct.new(:start_time, :end_time, :rating, :start_lat, :start_lon, :end_lat, :end_lon)
  def on_ground_during(time_range)
    start_time < time_range.end &&
    end_time   > time_range.begin
  end
end

TORNADOES = CSV.read("tornadoes.csv").drop(1).map do |start_time_str, start_seconds_str, end_time_str, end_seconds_str, rating_str, start_lat_str, start_lon_str, end_lat_str, end_lon_str|
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