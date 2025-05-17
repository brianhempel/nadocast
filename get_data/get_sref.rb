#!/usr/bin/env ruby

require "date"
require "set"

# https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/
# https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.20180626/09/ensprod/sref.t09z.pgrb212.mean_1hrly.grib2

# Files available 3.5-4hrs after run time
#
# 1hrly files contain hours 1-38 not divisible by 3
# 3hrly files contain hours anl,3-87 divisible by 3

def time(&block)
  start_t = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  out     = yield
  end_t   = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  [out, end_t - start_t]
end

# FTPPROD is ftp only now
# FTPPROD = "ftpprd.ncep.noaa.gov"
NOMADS  = "nomads.ncep.noaa.gov/pub"

# ymds_1, ftpprod_time = time { `curl -s https://#{FTPPROD}/data/nccf/com/href/prod/`.scan(/\bhref\.(\d{8})\//).flatten.uniq }
ymds_2, nomads_time  = time { `curl -s https://#{NOMADS}/data/nccf/com/href/prod/`.scan(/\bhref\.(\d{8})\//).flatten.uniq }

# if ([ymds_1.size, -ftpprod_time] <=> [ymds_2.size, -nomads_time]) > 0
#   DOMAIN = FTPPROD
#   YMDS   = ymds_1
# else
DOMAIN = NOMADS
YMDS   = ymds_2
# end

PRIMARY_FORECASTS_ROOT = ENV["FORECASTS_ROOT"] || "/Volumes" # || "/Volumes/hd2/DATA_TEMP"
BACKUP_FORECASTS_ROOT  = ENV["BACKUP_FORECASTS_ROOT"] || "/Volumes/hd2/DATA_TEMP"

loop { break if Dir.exist?("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3/"); puts "Waiting for SREF_HREF_3 to mount..."; sleep 4 }
loop { break if Dir.exist?("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4/"); puts "Waiting for SREF_HREF_4 to mount..."; sleep 4 }

def mb_available(path)
  # Filesystem     1M-blocks  Used Available Use% Mounted on
  # /dev/sdb1        5676931 32903   5357855   1% /media/brian/hd2
  `df -m #{path}`.split("\n")[1].split[3].to_f
end

if [mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3/"), mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4/")].min < 10_000
  puts "#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3 space: #{mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3/")}MB"
  puts "#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4 space: #{mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4/")}MB"
  puts "Not enough space, using backup location: #{BACKUP_FORECASTS_ROOT}"
  FORECASTS_ROOT = BACKUP_FORECASTS_ROOT
else
  FORECASTS_ROOT = PRIMARY_FORECASTS_ROOT
end

TYPES          = ["mean_1hrly", "mean_3hrly", "prob_1hrly", "prob_3hrly"]
HOURS_OF_DAY   = [3, 9, 15, 21]
BASE_DIRECTORY_1 = "#{FORECASTS_ROOT}/SREF_HREF_1/sref"
BASE_DIRECTORY_2 = "#{FORECASTS_ROOT}/SREF_HREF_3/sref"
MIN_FILE_BYTES = 20_000_000
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || "2")

AVAILABLE_FOR_DOWNLOAD = YMDS.product(HOURS_OF_DAY).flat_map do |ymd, run_hour|
  run_hour_str = "%02d" % [run_hour]
  remote_files = `curl -s https://#{DOMAIN}/data/nccf/com/sref/prod/sref.#{ymd}/#{run_hour_str}/ensprod/`.scan(/\bsref\.t[\.0-9a-z_]+/).grep(/pgrb212.*grib2$/)
  remote_files.map { |name| "https://#{DOMAIN}/data/nccf/com/sref/prod/sref.#{ymd}/#{run_hour_str}/ensprod/#{name}" }
end.to_set

def alt_location(directory)
  directory.sub(/^#{FORECASTS_ROOT}\/SREF_HREF_1\//, "#{FORECASTS_ROOT}/SREF_HREF_2/").sub(/^#{FORECASTS_ROOT}\/SREF_HREF_3\//, "#{FORECASTS_ROOT}/SREF_HREF_4/")
end


# https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/href.20180629/ensprod/href.t00z.conus.prob.f01.grib2

forecasts_to_get = YMDS.product(HOURS_OF_DAY, TYPES)

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, type = forecast_to_get
      year_month   = year_month_day[0...6]
      run_hour_str = "%02d" % [run_hour]

      file_name    = "sref_#{year_month_day}_t#{run_hour_str}z_#{type}.grib2"
      url_to_get   = "https://#{DOMAIN}/data/nccf/com/sref/prod/sref.#{year_month_day}/#{run_hour_str}/ensprod/sref.t#{run_hour_str}z.pgrb212.#{type}.grib2"
      if AVAILABLE_FOR_DOWNLOAD.include?(url_to_get)
        base_directory    = year_month[0...4].to_i < 2021 ? BASE_DIRECTORY_1 : BASE_DIRECTORY_2
        directory         = "#{base_directory}/#{year_month}/#{year_month_day}"
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
            File.write(alt_path, data) if Dir.exist?(alt_directory) && (File.size(alt_path) rescue 0) < MIN_FILE_BYTES
          end
        end
      end
    end
  end
end

threads.each(&:join)
