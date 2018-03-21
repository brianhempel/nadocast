require "date"
require "csv"

# Prompts for data range and spits out a list of grib files.
#
# Usage:
#
# $ ruby make_grib_file_list.rb > my_model/train.txt


def int_prompt(prompt)
  STDERR.puts prompt
  Integer(gets)
end

def prompt_for_date(prefix)
  year  = int_prompt "#{prefix} year?"
  month = int_prompt "#{prefix} month?"
  day   = int_prompt "#{prefix} day?"

  Date.new(year, month, day)
end

TRAIN_START_DATE = prompt_for_date("Start")
TRAIN_END_DATE   = prompt_for_date("End")

FORECAST_HOUR    = int_prompt("Forcast hour (1-18)?")

STDERR.puts "Only hours with tornadoes (n/y)?"
if gets =~ /^[yY]/
  ONLY_TORNADO_HOURS = true
else
  ONLY_TORNADO_HOURS = false
end

STDERR.puts "Subsample ratio [0.0-1.0]?"
SUBSAMPLE_RATIO = Float(gets)

TRAIN_DATES     = (TRAIN_START_DATE..TRAIN_END_DATE).to_a # 2005-10-31 is first +1hr with inv, no forecasts during Jan 2008, 2008-10-30 is when 13km RUC consistently available, 2008-11-17-1200 is first RUC with simulated reflectivity; also schema more stable (during 2007 CAPE and others only included during certain periods, presumably if relevant or not)
HOURS_OF_DAY   = (0..23).to_a
# BASE_URL       = FORECAST_HOURS.max <= 1 ? "https://nomads.ncdc.noaa.gov/data/rucanl" : "https://nomads.ncdc.noaa.gov/data/rap130" # Based a single sample, files in these two folders are identical if they exist in both
BASE_DIRECTORY = "/Volumes/Tornadoes/rap"
MIN_FILE_BYTES = 10_000_000
BAD_FILES      = %w[
  rap_130_20140928_1800_001.inv
]
# Possibly bad: ruc2anl_252_20091210_0500_001.inv missing fields 209+
# ruc2anl_252_20110601_1000_001.inv
# ruc2anl_252_20110601_1100_001.inv

hours_with_tornadoes = 0
file_count           = 0

ONE_MINUTE = 60
ONE_HOUR   = 60*ONE_MINUTE

Tornado = Struct.new(:start_time, :end_time, :rating, :start_lat, :start_lon, :end_lat, :end_lon)

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

TRAIN_DATES.product(HOURS_OF_DAY).each do |date, run_hour|
  year_month        = "%04d%02d"     % [date.year, date.month]
  year_month_day    = "%04d%02d%02d" % [date.year, date.month, date.day]
  run_hour_str      = "%02d00"       % [run_hour]
  forecast_hour_str = "%03d"         % [FORECAST_HOUR]
  rap_or_ruc        =
    if year_month_day < "20120509"
      "ruc2anl_130"
    else
      "rap_130"
    end
  file_name  = "#{rap_or_ruc}_#{year_month_day}_#{run_hour_str}_#{forecast_hour_str}.grb2"
  next if BAD_FILES.include?(file_name)
  # url_to_get = "#{BASE_URL}/#{year_month}/#{year_month_day}/#{file_name}" # e.g. "https://nomads.ncdc.noaa.gov/data/rap130/201612/20161231/rap_130_20161231_0600_002.grb2"
  directory  = "#{BASE_DIRECTORY}/#{year_month}/#{year_month_day}"
  path       = "#{directory}/#{file_name}"

  if (File.size(path) rescue 0) >= MIN_FILE_BYTES

    valid_time       = Time.new(date.year, date.month, date.day, run_hour,00,00,"+00:00") + FORECAST_HOUR*ONE_HOUR
    valid_start_time = valid_time - 30*ONE_MINUTE
    valid_end_time   = valid_time + 30*ONE_MINUTE

    any_tornadoes =
      TORNADOES.any? do |tornado|
        # ignore tornadoes in Hawaii, Alaska, and Puerto Rico
        # min training lat 25.719
        # max training lat 48.647
        # min training lon -123.95500000000001
        # max training lon -67.89800000000002
        tornado.start_time < valid_end_time &&
        tornado.end_time   > valid_start_time &&
        (
          ((24..50).cover?(tornado.start_lat) && (-125..-66).cover?(tornado.start_lon)) ||
          ((24..50).cover?(tornado.end_lat)   && (-125..-66).cover?(tornado.end_lon))
        )
      end

    if !ONLY_TORNADO_HOURS || any_tornadoes
      if rand <= SUBSAMPLE_RATIO
        hours_with_tornadoes += any_tornadoes ? 1 : 0
        puts path
        file_count += 1
      end
    end
  end
end

STDERR.puts "#{hours_with_tornadoes}/#{file_count} hours with tornadoes"

