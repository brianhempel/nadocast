#!/usr/bin/env ruby

require "cgi"
require "date"
require "set"

# Files available 2.5hrs after run time

# https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260801/00/enspost/refs.t00z.mean.f01.conus.grib2
# https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260801/00/enspost/refs.t00z.prob.f01.conus.grib2
#
# The bucket only keeps ~10 days, so this has to run often enough to keep up.
#
# rrfs_a/      is the fuller developer feed (its enspost/ also holds thousands of graphics/*.gif)
# rrfs_public/ is the subset NOAA says matches what will be served operationally over NOMADS
RRFS_S3   = "noaa-rrfs-pds.s3.amazonaws.com"
S3_PREFIX = ENV["REFS_S3_PREFIX"] || "rrfs_public/"

# Lists one "directory" level, exactly like the bucket's index.html viewer: a single GET
# with delimiter=/ so S3 rolls everything below up into CommonPrefixes. A directory level
# here is only tens of entries, so this never needs continuation tokens.
def s3_dirs(prefix)
  xml = `curl -f -s --show-error "https://#{RRFS_S3}/?list-type=2&delimiter=%2F&prefix=#{CGI.escape(prefix)}"`
  puts xml
  STDERR.puts "Failed to list #{prefix}" unless $?.success?
  # S3 echoes the requested prefix back in its own <Prefix> element; drop it.
  xml.scan(/<Prefix>([^<]+)<\/Prefix>/).flatten.uniq - [prefix]
end

# Lists the object keys under a prefix, following continuation tokens.
def s3_keys(prefix)
  keys  = []
  token = nil

  loop do
    url  = "https://#{RRFS_S3}/?list-type=2&max-keys=1000&prefix=#{CGI.escape(prefix)}"
    url += "&continuation-token=#{CGI.escape(token)}" if token

    xml = `curl -f -s --show-error "#{url}"`
    unless $?.success?
      STDERR.puts "Failed to list #{prefix}"
      break
    end

    keys += xml.scan(/<Key>([^<]+)<\/Key>/).flatten

    token = xml[/<NextContinuationToken>([^<]+)<\/NextContinuationToken>/, 1]
    break unless token
  end

  keys
end

# rrfs_public/ also holds rrfs.YYYYMMDD/, so match refs. specifically.
YMDS = s3_dirs(S3_PREFIX).map { |dir| dir[/\brefs\.(\d{8})\//, 1] }.compact.uniq.sort

p YMDS

PRIMARY_FORECASTS_ROOT = ENV["FORECASTS_ROOT"] || "/Volumes" # || "/Volumes/hd2/DATA_TEMP"
BACKUP_FORECASTS_ROOT  = ENV["BACKUP_FORECASTS_ROOT"] || "/Volumes/hd2/DATA_TEMP"

loop { break if Dir.exist?("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3/"); puts "Waiting for SREF_HREF_3 to mount..."; sleep 4 }
loop { break if Dir.exist?("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4/"); puts "Waiting for SREF_HREF_4 to mount..."; sleep 4 }

def mb_available(path)
  # Filesystem     1M-blocks  Used Available Use% Mounted on
  # /dev/sdb1        5676931 32903   5357855   1% /media/brian/hd2
  `df -m #{path}`.split("\n")[1].split[3].to_f
end

if [mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_3/"), mb_available("#{PRIMARY_FORECASTS_ROOT}/SREF_HREF_4/")].min < 1_000
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
MIN_FILE_BYTES = 20_000_000
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || "4")

ymds_hours_to_list = YMDS.product(HOURS_OF_DAY)

# The ensemble product directory is "enspost" in the prototype feed but becomes "ensprod"
# for REFS v1.0 (SCN 26-48), so discover whichever one each cycle actually has.
PRODUCT_DIRS = {} # [year_month_day, run_hour] => "rrfs_public/refs.20260801/00/enspost/"

AVAILABLE_FOR_DOWNLOAD = Set.new

threads = (THREAD_COUNT * 2).times.map do
  Thread.new do
    while ymds_hour = ymds_hours_to_list.shift
      ymd, run_hour = ymds_hour
      run_hour_str = "%02d" % [run_hour]
      subdirs      = s3_dirs("#{S3_PREFIX}refs.#{ymd}/#{run_hour_str}/")
      product_dir  = subdirs.find { |dir| dir.end_with?("/ensprod/") } || subdirs.find { |dir| dir.end_with?("/enspost/") }
      next unless product_dir

      PRODUCT_DIRS[[ymd, run_hour]] = product_dir

      # Listing per type keeps each response to a single page (~480 keys) instead of paging
      # through every product and domain in the cycle.
      TYPES.flat_map do |type|
        s3_keys("#{product_dir}refs.t#{run_hour_str}z.#{type}.").grep(/\.conus\.grib2$/).map { |key| "https://#{RRFS_S3}/#{key}" }
      end.each do |url|
        AVAILABLE_FOR_DOWNLOAD << url
      end
    end
  end
end

threads.each(&:join)

def alt_location(directory)
  directory.sub(/^#{FORECASTS_ROOT}\/SREF_HREF_1\//, "#{FORECASTS_ROOT}/SREF_HREF_2/").sub(/^#{FORECASTS_ROOT}\/SREF_HREF_3\//, "#{FORECASTS_ROOT}/SREF_HREF_4/")
end

# https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.20260801/00/enspost/refs.t00z.prob.f01.conus.grib2
# Note REFS orders the name type-before-domain, where HREF was domain-before-type.

forecasts_to_get = YMDS.product(HOURS_OF_DAY, FORECAST_HOURS, TYPES)

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, forecast_hour, type = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]
      forecast_hour_str = "%02d" % [forecast_hour]

      file_name         = "refs_conus_#{year_month_day}_t#{run_hour_str}z_#{type}_f#{forecast_hour_str}.grib2"
      product_dir       = PRODUCT_DIRS[[year_month_day, run_hour]]
      url_to_get        = product_dir && "https://#{RRFS_S3}/#{product_dir}refs.t#{run_hour_str}z.#{type}.f#{forecast_hour_str}.conus.grib2"
      if url_to_get && AVAILABLE_FOR_DOWNLOAD.include?(url_to_get)
        base_directory    = year_month[0...4].to_i < 2021 ? BASE_DIRECTORY_1 : BASE_DIRECTORY_2
        directory         = "#{base_directory}/#{year_month}/#{year_month_day}"
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
            File.write(alt_path, data) if Dir.exist?(alt_directory) && (File.size(alt_path) rescue 0) < MIN_FILE_BYTES
          end
        end
      end
    end
  end
end

threads.each(&:join)
