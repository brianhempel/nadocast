#!/usr/bin/env ruby

# downloads HRRR and resamples it to 5km (saves ~2x space)
# spot checks showed that the resampling looks a lot better at 5km than 6km or 9km

require "date"
require "fileutils"
# require File.expand_path("../../storm_data/storm_events.rb", __FILE__)
require File.expand_path("../forecast.rb", __FILE__)

FROM_ARCHIVE    = ARGV.include?("--from-archive")
DRY_RUN         = ARGV.include?("--dry-run")
# DELETE_UNNEEDED = ARGV.include?("--delete-unneeded") # Delete files in time range not associated with storm events.

# $ wgrib2 -grid -end hrrr.t00z.wrfsfcf15.grib2
# 1:0:grid_template=30:winds(grid):
# 	Lambert Conformal: (1799 x 1059) input WE:SN output WE:SN res 8
# 	Lat1 21.138123 Lon1 237.280472 LoV 262.500000
# 	LatD 38.500000 Latin1 38.500000 Latin2 38.500000
# 	LatSP 0.000000 LonSP 0.000000
# 	North Pole (1799 x 1059) Dx 3000.000000 m Dy 3000.000000 m

# Lambert conic conformal
#
# wgrib2 -new_grid_interpolation bilinear -new_grid lambert:lov:latin1:latin2:lad lon0:nx:dx lat0:ny:dy outfile
#
# lad = latin2
# latin2 = latini lad = latin1
# lov = longitude (degrees) where y axis is parallel to meridian
# latin1 = first latitude from pole which cuts the secant cone
# latin2 = second latitude from pole which cuts the secant cone
# lad = latitude (degrees) where dx and dy are specified
# lat0, lon0 = degrees of lat/lon for 1st grid point
# nx = number of grid points in X direction
# ny = number of grid points in Y direction
# dx = grid cell size in meters in × direction
# dy = grid cell size in meters in y direction
# note: if latin2 >= 0, the north pole is on proj plane
# if latin2 < 0, the south pole is on proj plane
#
# best looking resampling is 5km bilinear, then 6km budget (2x), then 7.5km budget
#
# 1x                 wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:262.5:38.5:38.5:38.5 237.280472:1799:3000 21.138123:1059:3000 hrrr.t00z.wrfsfcf15.resampled-1x.grib2
# 2x                 wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:262.5:38.5:38.5:38.5 237.280472:900:6000 21.138123:530:6000 hrrr.t00z.wrfsfcf15.resampled-2x.grib2
# 3x                 wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:262.5:38.5:38.5:38.5 237.280472:600:9000 21.138123:353:9000 hrrr.t00z.wrfsfcf15.resampled-3x.grib2
# 5km HREF cropped   wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:265:25:25:25 234.906:1161:5079 19.858000:678:5079 hrrr.t00z.wrfsfcf15.resampled-5km.grib2
# 5km HREF cropped   wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation budget   -new_grid lambert:265:25:25:25 234.906:1161:5079 19.858000:678:5079 hrrr.t00z.wrfsfcf15.resampled-5km-budget.grib2
# 7.5km HREF cropped wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:265:25:25:25 234.906:774:7618.5 19.858000:452:7618.5 hrrr.t00z.wrfsfcf15.resampled-7.5km.grib2
# 7.5km HREF cropped wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation budget   -new_grid lambert:265:25:25:25 234.906:774:7618.5 19.858000:452:7618.5 hrrr.t00z.wrfsfcf15.resampled-7.5km-budget.grib2
# 15km HREF cropped  wgrib2 hrrr.t00z.wrfsfcf15.grib2 -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:265:25:25:25 234.906:387:15237 19.858000:226:15237 hrrr.t00z.wrfsfcf15.resampled-15km.grib2


# Google archive only has up to f15 until 2016-8-25...and the Utah archive got rid of their 2016 data for some reason.
# 2016-8-24 is where the 250mb winds, lightning, and surface RH appear, which our models assume.
# If we use the older data, we'll have to disregard those

# For training:
#
# Don't have enough disk space to store them all at once :(
# Download next set after one set has loaded and is training.

# THREAD_COUNT=3 FORECAST_HOURS=1,6,12,18 ruby get_hrrr.rb --from-archive

# Forecaster runs these:
# RUN_HOURS=8,9,10   FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb
# RUN_HOURS=12,13,14 FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb

RUN_HOURS            = ENV["RUN_HOURS"]&.split(",")&.map(&:to_i) || (0..23).to_a
FORECAST_HOURS       = ENV["FORECAST_HOURS"]&.split(",")&.map(&:to_i) || [1, 6, 12, 18]
VALIDATION_RUN_HOURS = ENV["VALIDATION_RUN_HOURS"]&.split(",")&.map(&:to_i) || []
TEST_RUN_HOURS       = ENV["TEST_RUN_HOURS"]&.split(",")&.map(&:to_i) || []
MIN_FILE_BYTES       = 30_000_000
THREAD_COUNT         = Integer((DRY_RUN && "1") || ENV["THREAD_COUNT"] || (FROM_ARCHIVE ? "4" : "8"))
# HALF_WINDOW_SIZE     = 90*MINUTE # Grab forecasts valid within this many minutes of a geocoded storm event


# loop { break if Dir.exists?("/Volumes/HRRR_1/"); puts "Waiting for HRRR_1 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/HRRR_2/"); puts "Waiting for HRRR_2 to mount..."; sleep 4 }

class HRRRForecast < Forecast
  def file_name
    "hrrr_conus_sfc_5km_#{year_month_day}_t#{run_hour_str}z_f#{forecast_hour_str}.grib2"
  end

  def archive_url
    "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def alt_archive_url
    "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def ncep_url
    "https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.#{year_month_day}/conus/hrrr.t#{run_hour_str}z.wrfsfcf#{forecast_hour_str}.grib2"
  end

  def base_directory
    "/Volumes/HRRR_2/hrrr"
  end

  def alt_directory
    raise "no alt directory for HRRR: the online archive serves as the backup"
  end

  def alt_path
    nil
  end

  def min_file_bytes
    MIN_FILE_BYTES
  end

  def ensure_downloaded!(from_archive: false)
    url_to_get = from_archive ? archive_url : ncep_url
    make_directories!
    unless downloaded?
      puts "#{url_to_get} -> #{path}"
      return if DRY_RUN
      # The pando HRRR archive server doesn't like it when TLS 1.3 is tried before TLS 1.2
      # data = `curl -f -s --show-error --tlsv1.2 --tls-max 1.2 #{url_to_get}`

      data = `curl -f -s --show-error #{url_to_get} | wgrib2 - -set_grib_type same -new_grid_winds grid -new_grid_interpolation bilinear -new_grid lambert:265:25:25:25 234.906000:1161:5079 19.858000:678:5079 -`

      if $?.success? && data.size >= min_file_bytes
        File.write(path, data)
      elsif from_archive && alt_archive_url
        url_to_get = alt_archive_url
        puts "#{url_to_get} -> #{path}"
        return if DRY_RUN
        data = `curl -f -s --show-error #{url_to_get}`
        if $?.success? && data.size >= min_file_bytes
          File.write(path, data)
          if alt_path
            File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < min_file_bytes
          end
        end
      end
    end
  end
end

if FROM_ARCHIVE
  start_date_parts = (ENV["START_DATE"] || "2018-7-13").split("-")&.map(&:to_i) # HRRRv3 started sometime on 2018-7-12, but not at 0z so we'll start on 2018-7-13

  DATES     = (Date.new(*start_date_parts)..Date.today).to_a
  SATURDAYS = DATES.select(&:saturday?)
  SUNDAYS   = DATES.select(&:sunday?)

  # storm_event_times =
  #   conus_event_hours_set(STORM_EVENTS, HALF_WINDOW_SIZE)

  # This is a little conservative because we aren't computing backwards from the valid time
  forecasts_in_range =
    DATES.product(RUN_HOURS, FORECAST_HOURS).map do |date, run_hour, forecast_hour|
      HRRRForecast.new(date, run_hour, forecast_hour)
    end

  # forecasts_to_get =
  #   forecasts_in_range.select do |forecast|
  #     FORECAST_HOURS.include?(forecast.forecast_hour) &&
  #       (storm_event_times.include?(forecast.valid_time) ||
  #        storm_event_times.include?(forecast.valid_time + HOUR) ||
  #        storm_event_times.include?(forecast.valid_time - HOUR))
  #   end

  validation_forecasts_to_get =
    SATURDAYS.product(VALIDATION_RUN_HOURS, (1..18).to_a).map do |date, run_hour, forecast_hour|
      HRRRForecast.new(date, run_hour, forecast_hour)
    end

  test_forecasts_to_get =
    SUNDAYS.product(TEST_RUN_HOURS, (1..18).to_a).map do |date, run_hour, forecast_hour|
      HRRRForecast.new(date, run_hour, forecast_hour)
    end

  forecasts_to_get = (forecasts_in_range + validation_forecasts_to_get + test_forecasts_to_get).uniq

  # forecasts_to_remove = DELETE_UNNEEDED ? (forecasts_in_range - forecasts_to_get) : []

  # forecasts_to_remove.each(&:remove!)
else
  # https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/hrrr.20190220/conus/hrrr.t02z.wrfsfcf18.grib2
  ymds = ENV["FORECAST_DATE"] ? [ENV["FORECAST_DATE"].gsub("-","")] : `curl -s https://ftp.ncep.noaa.gov/data/nccf/com/hrrr/prod/`.scan(/hrrr\.(\d{8})\//).flatten.uniq
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
