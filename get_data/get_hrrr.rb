#!/usr/bin/env ruby

require "date"
require "fileutils"
require File.expand_path("../../storm_data/storm_events.rb", __FILE__)

FROM_ARCHIVE    = ARGV.include?("--from-archive")
DRY_RUN         = ARGV.include?("--dry-run")
DELETE_UNNEEDED = ARGV.include?("--delete-unneeded") # Delete files in time range not associated with storm events.

# The reason we have run hour 1,2,3,5,6,7,11,12,13,16,17,18 forecasts for storm events since mid-2018 is for training the stacked models. :/

# RUN_HOURS=8,9,10 FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb

# For getting the HRRRs associated with the SREF/HREF forecasts we have
# START_DATE=2018-6-25 FORECAST_HOURS=1,2,3,5,6,7,11,12,13,16,17,18 ruby get_hrrr.rb --from-archive

RUN_HOURS      = ENV["RUN_HOURS"]&.split(",")&.map(&:to_i) || (0..23).to_a
FORECAST_HOURS = ENV["FORECAST_HOURS"]&.split(",")&.map(&:to_i) || [2, 6, 12, 18]
BASE_DIRECTORY = "/Volumes/HRRR_1/hrrr"
MIN_FILE_BYTES = 80_000_000
THREAD_COUNT   = Integer((DRY_RUN && "1") || ENV["THREAD_COUNT"] || (FROM_ARCHIVE ? "2" : "4"))

MINUTE = 60
HOUR   = 60*MINUTE

def alt_location(directory)
  directory.sub(/^\/Volumes\/HRRR_1\//, "/Volumes/HRRR_2/")
end

loop { break if Dir.exists?("/Volumes/HRRR_1/"); puts "Waiting for HRRR_1 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/HRRR_2/"); puts "Waiting for HRRR_2 to mount..."; sleep 4 }

def ymd_to_date(ymd_str)
  Date.new(Integer(ymd_str[0...4]), ymd_str[4...6].to_i, ymd_str[6...8].to_i)
end

class Forecast < Struct.new(:run_date, :run_hour, :forecast_hour)
  def year_month_day
    "%04d%02d%02d" % [run_date.year, run_date.month, run_date.day]
  end

  def year_month
    "%04d%02d" % [run_date.year, run_date.month]
  end

  def run_hour_str
    "%02d" % [run_hour]
  end

  def forecast_hour_str
    "%02d" % [forecast_hour]
  end

  def valid_time
    Time.utc(run_date.year, run_date.month, run_date.day, run_hour) + forecast_hour*HOUR
  end

  def file_name
    "hrrr_conus_sfc_#{year_month_day}_t#{run_hour_str}z_f#{forecast_hour_str}.grib2"
  end

  def archive_url
    "https://pando-rgw01.chpc.utah.edu/hrrr/sfc/#{year_month_day}/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def ncep_url
    "https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def directory
    "#{BASE_DIRECTORY}/#{year_month}/#{year_month_day}"
  end

  def path
    "#{directory}/#{file_name}"
  end

  def alt_directory
    alt_location(directory)
  end

  def alt_path
    alt_location(path)
  end

  def make_directories!
    return if DRY_RUN
    system("mkdir -p #{directory} 2> /dev/null")
    system("mkdir -p #{alt_directory} 2> /dev/null")
  end

  def downloaded?
    (File.size(path) rescue 0) >= MIN_FILE_BYTES
  end

  def ensure_downloaded!(from_archive: false)
    url_to_get = from_archive ? archive_url : ncep_url
    make_directories!
    unless downloaded?
      puts "#{url_to_get} -> #{path}"
      return if DRY_RUN
      data = `curl -f -s --show-error #{url_to_get}`
      if $?.success? && data.size >= MIN_FILE_BYTES
        File.write(path, data)
        File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < MIN_FILE_BYTES
      end
    end
  end

  def remove!
    if File.exists?(path)
      puts "REMOVE #{path}"
      FileUtils.rm(path) unless DRY_RUN
    end
    if File.exists?(alt_path)
      puts "REMOVE #{alt_path}"
      FileUtils.rm(alt_path) unless DRY_RUN
    end
  end
end

if FROM_ARCHIVE # Storm event hours only, for now. Would be 12TB for all +2 +6 +12 +18 forecasts.
  start_date_parts = ENV["START_DATE"]&.split("-")&.map(&:to_i) || [2016,7,15]

  # https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180101/hrrr.t00z.wrfsfcf00.grib2
  DATES = (Date.new(*start_date_parts)..Date.today).to_a

  storm_event_times =
    conus_event_hours_set(STORM_EVENTS, 30*MINUTE)

  forecasts_in_range =
    DATES.product(RUN_HOURS, (0..18).to_a).map do |date, run_hour, forecast_hour|
      Forecast.new(date, run_hour, forecast_hour)
    end

  forecasts_to_get =
    forecasts_in_range.select do |forecast|
      FORECAST_HOURS.include?(forecast.forecast_hour) && storm_event_times.include?(forecast.valid_time)
    end

  forecasts_to_remove = DELETE_UNNEEDED ? (forecasts_in_range - forecasts_to_get) : []

  forecasts_to_remove.each(&:remove!)
else
  # https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.20190220/conus/hrrr.t02z.wrfsfcf18.grib2
  ymds = `curl -s https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/`.scan(/hrrr\.(\d{8})\//).flatten.uniq
  forecasts_to_get = ymds.product(RUN_HOURS, FORECAST_HOURS).map { |ymd, run_hour, forecast_hour| Forecast.new(ymd_to_date(ymd), run_hour, forecast_hour) }
end

# hrrr.t02z.wrfsfcf18.grib2

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      forecast_to_get.ensure_downloaded!(from_archive: FROM_ARCHIVE)
    end
  end
end

threads.each(&:join)
