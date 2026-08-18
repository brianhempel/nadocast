#!/usr/bin/env ruby

require "date"
require "set"

# Files available 2.5hrs after run time

# try to use whichever domain is responding best...

def time(&block)
  start_t = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  out     = yield
  end_t   = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  [out, end_t - start_t]
end

# FTPPROD is ftp only now
# FTPPROD = "ftpprd.ncep.noaa.gov"
NOMADS  = "nomads.ncep.noaa.gov/pub"

# SCN 26-48: parallel "para" feed since Aug 11 2026, becomes "prod" at implementation Oct 6 2026.
# Only ~2 days are retained here, vs ~10 on the old S3 prototype bucket (see get_refs_s3.rb).
REFS_FEED = ENV["REFS_FEED"] || "prod"

# ymds_1, ftpprod_time = time { `curl -s https://#{FTPPROD}/data/nccf/com/href/prod/`.scan(/\bhref\.(\d{8})\//).flatten.uniq }
ymds_2, nomads_time  = time { `curl --http1.1 -s https://#{NOMADS}/data/nccf/com/refs/#{REFS_FEED}/`.scan(/\brefs\.(\d{8})\//).flatten.uniq }

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

TYPES          = ["mean", "prob"]
HOURS_OF_DAY   = [00, 06, 12, 18]
FORECAST_HOURS = (01..60).to_a
BASE_DIRECTORY_1 = "#{FORECASTS_ROOT}/SREF_HREF_1/refs"
BASE_DIRECTORY_2 = "#{FORECASTS_ROOT}/SREF_HREF_3/refs"
MIN_FILE_BYTES = 25_000_000
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || "4")

# REFS nests a cycle directory under the day, where HREF had one ensprod/ per day.
AVAILABLE_FOR_DOWNLOAD = YMDS.product(HOURS_OF_DAY).flat_map do |ymd, run_hour|
  dir_url      = "https://#{DOMAIN}/data/nccf/com/refs/#{REFS_FEED}/refs.#{ymd}/#{"%02d" % [run_hour]}/ensprod"
  remote_files = `curl --http1.1 -s #{dir_url}/`.scan(/\brefs\.t[\.0-9a-z_]+/).grep(/conus\.grib2$/)
  remote_files.map { |name| "#{dir_url}/#{name}" }
end.to_set

def alt_location(directory)
  directory.sub(/^#{FORECASTS_ROOT}\/SREF_HREF_1\//, "#{FORECASTS_ROOT}/SREF_HREF_2/").sub(/^#{FORECASTS_ROOT}\/SREF_HREF_3\//, "#{FORECASTS_ROOT}/SREF_HREF_4/")
end


# https://nomads.ncep.noaa.gov/pub/data/nccf/com/refs/para/refs.20260818/00/ensprod/refs.t00z.prob.f01.conus.grib2
# (REFS puts the type before the domain; HREF had domain before type.)

forecasts_to_get = YMDS.product(HOURS_OF_DAY, FORECAST_HOURS, TYPES)

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, forecast_hour, type = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]
      forecast_hour_str = "%02d" % [forecast_hour]

      file_name         = "refs_conus_#{year_month_day}_t#{run_hour_str}z_#{type}_f#{forecast_hour_str}.grib2"
      url_to_get        = "https://#{DOMAIN}/data/nccf/com/refs/#{REFS_FEED}/refs.#{year_month_day}/#{run_hour_str}/ensprod/refs.t#{run_hour_str}z.#{type}.f#{forecast_hour_str}.conus.grib2"
      if AVAILABLE_FOR_DOWNLOAD.include?(url_to_get)
        base_directory    = year_month[0...4].to_i < 2021 ? BASE_DIRECTORY_1 : BASE_DIRECTORY_2
        directory         = "#{base_directory}/#{year_month}/#{year_month_day}"
        path              = "#{directory}/#{file_name}"
        alt_directory     = alt_location(directory)
        alt_path          = alt_location(path)

        system("mkdir -p #{directory} 2> /dev/null")
        system("mkdir -p #{alt_location(alt_directory)} 2> /dev/null")
        if (File.size(path) rescue 0) < MIN_FILE_BYTES
          puts "#{url_to_get} -> #{path}"
          data = `curl --http1.1 -f -s --show-error #{url_to_get}`
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
