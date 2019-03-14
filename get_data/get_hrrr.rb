require "date"
require File.expand_path("../../storm_data/tornadoes.rb", __FILE__)

FROM_ARCHIVE = (ARGV[0] == "--from-archive")

RUN_HOURS      = (0..23).to_a
FORECAST_HOURS = [2, 6, 12, 18]
BASE_DIRECTORY = "/Volumes/HRRR_1/hrrr"
MIN_FILE_BYTES = 80_000_000
BAD_FILES      = %w[]
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || (FROM_ARCHIVE ? "2" : "4"))

MINUTE = 60
HOUR   = 60*MINUTE

def alt_location(directory)
  directory.sub(/^\/Volumes\/HRRR_1\//, "/Volumes/HRRR_2/")
end

loop { break if Dir.exists?("/Volumes/HRRR_1/"); puts "Waiting for HRRR_1 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/HRRR_2/"); puts "Waiting for HRRR_2 to mount..."; sleep 4 }

if FROM_ARCHIVE # Tornado hours only, for now. Would be 12TB for all +2 +6 +12 +18 forecasts.
  # https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180101/hrrr.t00z.wrfsfcf00.grib2
  DATES = (Date.new(2016,7,15)..Date.today).to_a

  # This filtering takes five minutes.
  forecasts_to_get =
    DATES.product(RUN_HOURS, FORECAST_HOURS).filter do |date, run_hour, forecast_hour|
      valid_time = Time.utc(date.year, date.month, date.day, run_hour) + forecast_hour*HOUR
      valid_start_time = valid_time - 30*MINUTE
      valid_end_time   = valid_time + 30*MINUTE
      TORNADOES.any? do |tornado|
        tornado.on_ground_during(valid_start_time...valid_end_time) && tornado.in_conus?
      end
    end.map do |date, run_hour, forecast_hour|
      ["%04d%02d%02d" % [date.year, date.month, date.day], run_hour, forecast_hour]
    end
else
  # https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.20190220/conus/hrrr.t02z.wrfsfcf18.grib2
  YMDS = `curl -s https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/`.scan(/hrrr\.(\d{8})\//).flatten.uniq
  forecasts_to_get = YMDS.product(RUN_HOURS, FORECAST_HOURS)
end

# hrrr.t02z.wrfsfcf18.grib2

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, forecast_hour = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]
      forecast_hour_str = "%02d" % [forecast_hour]

      file_name         = "hrrr_conus_sfc_#{year_month_day}_t#{run_hour_str}z_f#{forecast_hour_str}.grib2"
      next if BAD_FILES.include?(file_name)
      if FROM_ARCHIVE
        url_to_get = "https://pando-rgw01.chpc.utah.edu/hrrr/sfc/#{year_month_day}/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
      else
        url_to_get = "https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
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
