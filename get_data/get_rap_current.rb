#!/usr/bin/env ruby

require "date"
require File.expand_path("../forecast.rb", __FILE__)

def ymd_to_date(ymd_str)
  Date.new(Integer(ymd_str[0...4]), ymd_str[4...6].to_i, ymd_str[6...8].to_i)
end

# NOMADS has 1 week to 1 year ago of data.

FROM_NOMADS = ARGV.include?("--from-nomads") # (ARGV[0] == "--from-archive")
DRY_RUN     = ARGV.include?("--dry-run")

# RUN_HOURS=8,9,10 FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 ruby get_rap_current.rb

RUN_HOURS      = ENV["RUN_HOURS"]&.split(",")&.map(&:to_i) || (0..23).to_a
FORECAST_HOURS = ENV["FORECAST_HOURS"]&.split(",")&.map(&:to_i) || (1..21).to_a
MIN_FILE_BYTES = 10_000_000
BAD_FILES      = %w[]
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || (FROM_NOMADS ? "2" : "4"))


loop { break if Dir.exists?("/Volumes/RAP_1/"); puts "Waiting for RAP_1 to mount...";  sleep 4 }
loop { break if Dir.exists?("/Volumes/RAP_2/"); puts "Waiting for RAP_2 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/RAP_3/"); puts "Waiting for RAP_3 to mount...";  sleep 4 }
loop { break if Dir.exists?("/Volumes/RAP_4/"); puts "Waiting for RAP_4 to mount..."; sleep 4 }

class RAPForecast < Forecast
  def file_name
    "rap_130_#{year_month_day}_#{run_hour_str}00_0#{forecast_hour_str}.grb2"
  end

  def archive_url
    if FROM_NOMADS
      "https://nomads.ncdc.noaa.gov/data/rap130/#{year_month}/#{year_month_day}/rap_130_#{year_month_day}_#{run_hour}00_0#{forecast_hour_str}.grb2"
    else
      raise "see get_rap_archived.rb for pulling from NCDC's long term storage request system"
    end
  end

  def ncep_url
    "https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.#{year_month_day}/rap.t#{run_hour_str}z.awp130pgrbf#{forecast_hour_str}.grib2"
  end

  def base_directory
    if run_date.year <= 2018
      "/Volumes/RAP_1/rap" # and backup copy on RAP_2
    else
      "/Volumes/RAP_3/rap" # and backup copy on RAP_4
    end
  end

  # Backup location
  def alt_directory
    directory.sub(/^\/Volumes\/RAP_1\//, "/Volumes/RAP_2/").sub(/^\/Volumes\/RAP_3\//, "/Volumes/RAP_4/")
  end

  def alt_path
    "#{alt_directory}/#{file_name}"
  end

  def min_file_bytes
    MIN_FILE_BYTES
  end
end


if FROM_NOMADS
  DATES = (Date.today - 365..Date.today).to_a

  # storm_event_times =
  #   conus_event_hours_set(STORM_EVENTS, HALF_WINDOW_SIZE)

  forecasts_in_range =
    DATES.product(RUN_HOURS, (0..21).to_a).map do |date, run_hour, forecast_hour|
      RAPForecast.new(date, run_hour, forecast_hour)
    end

  forecasts_to_get =
    forecasts_in_range.select do |forecast|
      FORECAST_HOURS.include?(forecast.forecast_hour)
    end

  forecasts_to_remove = DELETE_UNNEEDED ? (forecasts_in_range - forecasts_to_get) : []

  forecasts_to_remove.each(&:remove!)
else
  # https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.20180319/rap.t00z.awp130pgrbf02.grib2
  ymds = `curl -s https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/`.scan(/rap\.(\d{8})\//).flatten.uniq
  forecasts_to_get = ymds.product(RUN_HOURS, FORECAST_HOURS).map { |ymd, run_hour, forecast_hour| RAPForecast.new(ymd_to_date(ymd), run_hour, forecast_hour) }
end

# rap.t00z.awp130pgrbf02.grib2
# rap_130_20161228_0900_018.grb2

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      forecast_to_get.ensure_downloaded!(from_archive: false)
    end
  end
end

threads.each(&:join)
