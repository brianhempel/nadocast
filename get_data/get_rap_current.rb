#!/usr/bin/env ruby

require "date"

FROM_NOMADS = false # (ARGV[0] == "--from-archive")

# RUN_HOURS=8,9,10 FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 ruby get_rap_current.rb

RUN_HOURS      = ENV["RUN_HOURS"]&.split(",")&.map(&:to_i) || (0..23).to_a
FORECAST_HOURS = ENV["FORECAST_HOURS"]&.split(",")&.map(&:to_i) || (1..21).to_a
BASE_DIRECTORY = "/Volumes/RAP_1/rap"
MIN_FILE_BYTES = 10_000_000
BAD_FILES      = %w[]
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || (FROM_NOMADS ? "2" : "4"))

MINUTE = 60
HOUR   = 60*MINUTE

def alt_location(directory)
  directory.sub(/^\/Volumes\/RAP_1\//, "/Volumes/RAP_2/")
end

loop { break if Dir.exists?("/Volumes/RAP_1/"); puts "Waiting for RAP_1 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/RAP_2/"); puts "Waiting for RAP_2 to mount..."; sleep 4 }

if FROM_NOMADS
  STDERR.puts "This script hasn't yet been updated to grab files from Nomads."
  exit 1
else
  # https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.20180319/rap.t00z.awp130pgrbf02.grib2
  YMDS = `curl -s https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/`.scan(/rap\.(\d{8})\//).flatten.uniq
  forecasts_to_get = YMDS.product(RUN_HOURS, FORECAST_HOURS)
end

# rap.t00z.awp130pgrbf02.grib2
# rap_130_20161228_0900_018.grb2

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, forecast_hour = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]
      forecast_hour_str = "%02d" % [forecast_hour]

      file_name         = "rap_130_#{year_month_day}_#{run_hour_str}00_0#{forecast_hour_str}.grb2"
      next if BAD_FILES.include?(file_name)
      if FROM_NOMADS
      else
        url_to_get = "https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.#{year_month_day}/rap.t#{run_hour_str}z.awp130pgrbf#{forecast_hour_str}.grib2"
      end
      directory         = "#{BASE_DIRECTORY}/#{year_month}/#{year_month_day}"
      path              = "#{directory}/#{file_name}"
      alt_directory     = alt_location(directory)
      alt_path          = alt_location(path)

      system("mkdir -p #{directory} 2> /dev/null")
      system("mkdir -p #{alt_location(alt_directory)} 2> /dev/null")
      if (File.size(path) rescue 0) < MIN_FILE_BYTES
        puts "#{url_to_get} -> #{path}"
        data = `curl -f -s --show-error #{url_to_get}`
        if $?.success? && data.size >= MIN_FILE_BYTES
          File.write(path, data)
          File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < MIN_FILE_BYTES
        end
      end
    end
  end
end

threads.each(&:join)
