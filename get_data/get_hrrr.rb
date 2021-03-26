#!/usr/bin/env ruby

require "date"
require "fileutils"
require File.expand_path("../../storm_data/storm_events.rb", __FILE__)
require File.expand_path("../forecast.rb", __FILE__)

FROM_ARCHIVE    = ARGV.include?("--from-archive")
DRY_RUN         = ARGV.include?("--dry-run")
DELETE_UNNEEDED = ARGV.include?("--delete-unneeded") # Delete files in time range not associated with storm events.

# Runs through 2018-5 are stored on HRRR_1
# Runs 2018-6 onward are stored on HRRR_2

# For training:
# --from-archive flag implies storm event hours±1 only (HALF_WINDOW_SIZE below)
# Don't have enough disk space to store them all at once :(
# Download next set after one set has loaded and is training.
# FORECAST_HOURS=2,6,11,12,13,18 ruby get_hrrr.rb --from-archive --delete-unneeded
# FORECAST_HOURS=1,2,3,6,12,18 ruby get_hrrr.rb --from-archive --delete-unneeded
# THREAD_COUNT=2 FORECAST_HOURS=2,5,6,7,12,18 ruby get_hrrr.rb --from-archive --delete-unneeded
# FORECAST_HOURS=2,6,12,16,17,18 ruby get_hrrr.rb --from-archive --delete-unneeded

# Forecaster runs these:
# RUN_HOURS=8,9,10   FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb
# RUN_HOURS=12,13,14 FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb

# For getting the HRRRs associated with the SREF/HREF forecasts use this (although we don't have these, currently)
# START_DATE=2018-6-25 FORECAST_HOURS=1,2,3,5,6,7,11,12,13,16,17,18 ruby get_hrrr.rb --from-archive

# Want three hour windows. Don't have storage yet for all, so we'll start with +11,+12,+13 and build from there.

RUN_HOURS        = ENV["RUN_HOURS"]&.split(",")&.map(&:to_i) || (0..23).to_a
FORECAST_HOURS   = ENV["FORECAST_HOURS"]&.split(",")&.map(&:to_i) || [2, 6, 11, 12, 13, 18]
MIN_FILE_BYTES   = 80_000_000
THREAD_COUNT     = Integer((DRY_RUN && "1") || ENV["THREAD_COUNT"] || (FROM_ARCHIVE ? "1" : "4"))
HALF_WINDOW_SIZE = 90*MINUTE # Grab forecasts valid within this many minutes of a geocoded storm event

loop { break if Dir.exists?("/Volumes/HRRR_1/"); puts "Waiting for HRRR_1 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/HRRR_2/"); puts "Waiting for HRRR_2 to mount..."; sleep 4 }

class HRRRForecast < Forecast
  # def year_month_day
  #   "%04d%02d%02d" % [run_date.year, run_date.month, run_date.day]
  # end
  #
  # def year_month
  #   "%04d%02d" % [run_date.year, run_date.month]
  # end
  #
  # def run_hour_str
  #   "%02d" % [run_hour]
  # end
  #
  # def forecast_hour_str
  #   "%02d" % [forecast_hour]
  # end
  #
  # def valid_time
  #   Time.utc(run_date.year, run_date.month, run_date.day, run_hour) + forecast_hour*HOUR
  # end

  def file_name
    "hrrr_conus_sfc_#{year_month_day}_t#{run_hour_str}z_f#{forecast_hour_str}.grib2"
  end

  def archive_url
    "https://pando-rgw01.chpc.utah.edu/hrrr/sfc/#{year_month_day}/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def alt_archive_url
    "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def ncep_url
    "https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def base_directory
    if ([run_date.year, run_date.month] <=> [2018, 5]) <= 0
      "/Volumes/HRRR_1/hrrr"
    else
      "/Volumes/HRRR_2/hrrr"
    end
  end

  # def directory
  #   "#{base_directory}/#{year_month}/#{year_month_day}"
  # end

  # def path
  #   "#{directory}/#{file_name}"
  # end

  def alt_directory
    # directory.sub(/^\/Volumes\/HRRR_1\//, "/Volumes/HRRR_2/")
    raise "no alt directory for HRRR: the online archive serves as the backup"
  end

  def alt_path
    nil
  end
  #
  # def make_directories!
  #   return if DRY_RUN
  #   system("mkdir -p #{directory} 2> /dev/null")
  #   if alt_path
  #     system("mkdir -p #{alt_directory} 2> /dev/null")
  #   end
  # end

  def min_file_bytes
    MIN_FILE_BYTES
  end

  # def downloaded?
  #   (File.size(path) rescue 0) >= min_file_bytes
  # end
  #
  # def ensure_downloaded!(from_archive: false)
  #   url_to_get = from_archive ? archive_url : ncep_url
  #   make_directories!
  #   unless downloaded?
  #     puts "#{url_to_get} -> #{path}"
  #     return if DRY_RUN
  #     data = `curl -f -s --show-error #{url_to_get}`
  #     if $?.success? && data.size >= min_file_bytes
  #       File.write(path, data)
  #       if alt_path
  #         File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < min_file_bytes
  #       end
  #     end
  #   end
  # end
  #
  # def remove!
  #   if File.exists?(path)
  #     puts "REMOVE #{path}"
  #     FileUtils.rm(path) unless DRY_RUN
  #   end
  #   if alt_path && File.exists?(alt_path)
  #     puts "REMOVE #{alt_path}"
  #     FileUtils.rm(alt_path) unless DRY_RUN
  #   end
  # end
end

if FROM_ARCHIVE # Storm event hours only, for now. Would be 12TB for all +2 +6 +12 +18 forecasts.
  start_date_parts = ENV["START_DATE"]&.split("-")&.map(&:to_i) || [2016,7,15]

  # https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180101/hrrr.t00z.wrfsfcf00.grib2
  DATES = (Date.new(*start_date_parts)..Date.today).to_a

  storm_event_times =
    conus_event_hours_set(STORM_EVENTS, HALF_WINDOW_SIZE)

  forecasts_in_range =
    DATES.product(RUN_HOURS, (0..18).to_a).map do |date, run_hour, forecast_hour|
      HRRRForecast.new(date, run_hour, forecast_hour)
    end

  forecasts_to_get =
    forecasts_in_range.select do |forecast|
      FORECAST_HOURS.include?(forecast.forecast_hour) &&
        (storm_event_times.include?(forecast.valid_time)
        || storm_event_times.include?(forecast.valid_time + HOUR)
        || storm_event_times.include?(forecast.valid_time - HOUR))
    end

  forecasts_to_remove = DELETE_UNNEEDED ? (forecasts_in_range - forecasts_to_get) : []

  forecasts_to_remove.each(&:remove!)
else
  # https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.20190220/conus/hrrr.t02z.wrfsfcf18.grib2
  ymds = `curl -s https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/`.scan(/hrrr\.(\d{8})\//).flatten.uniq
  forecasts_to_get = ymds.product(RUN_HOURS, FORECAST_HOURS).map { |ymd, run_hour, forecast_hour| HRRRForecast.new(ymd_to_date(ymd), run_hour, forecast_hour) }
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
