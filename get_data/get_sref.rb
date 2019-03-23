#!/usr/bin/env ruby

require "date"

# https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/
# https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.20180626/09/ensprod/sref.t09z.pgrb212.mean_1hrly.grib2

# Files available 3.5-4hrs after run time
#
# 1hrly files contain hours 1-38 not divisible by 3
# 3hrly files contain hours anl,3-87 divisible by 3

TYPES          = ["mean_1hrly", "mean_3hrly", "prob_1hrly", "prob_3hrly"]
YMDS           = `curl -s https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/`.scan(/\bsref\.(\d{8})\//).flatten.uniq
HOURS_OF_DAY   = [3, 9, 15, 21]
BASE_DIRECTORY = "/Volumes/SREF_HREF_1/sref"
MIN_FILE_BYTES = 20_000_000
BAD_FILES      = %w[]
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || "4")

def alt_location(directory)
  directory.sub(/^\/Volumes\/SREF_HREF_1\//, "/Volumes/SREF_HREF_2/")
end

loop { break if Dir.exists?("/Volumes/SREF_HREF_1/"); puts "Waiting for SREF_HREF_1 to mount...";  sleep 4 }
loop { break if Dir.exists?("/Volumes/SREF_HREF_2/"); puts "Waiting for SREF_HREF_2 to mount..."; sleep 4 }


# https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/href.20180629/ensprod/href.t00z.conus.prob.f01.grib2

forecasts_to_get = YMDS.product(HOURS_OF_DAY, TYPES)

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, type = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]

      file_name         = "sref_#{year_month_day}_t#{run_hour_str}z_#{type}.grib2"
      next if BAD_FILES.include?(file_name)
      url_to_get        = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.#{year_month_day}/#{run_hour_str}/ensprod/sref.t#{run_hour_str}z.pgrb212.#{type}.grib2"
      directory         = "#{BASE_DIRECTORY}/#{year_month}/#{year_month_day}"
      path              = "#{directory}/#{file_name}"
      alt_directory     = alt_location(directory)
      alt_path          = alt_location(path)

      system("mkdir -p #{directory} 2> /dev/null")
      system("mkdir -p #{alt_directory} 2> /dev/null")
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
