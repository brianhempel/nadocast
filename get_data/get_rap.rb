require "date"

# Modify these constants.
INV_OR_GRIB    = %w[inv grb2][1]
DATES          = (Date.new(2017,9,6)..Date.new(2018,9,7)-1).to_a # 2005-10-31 is first +1hr with inv, no forecasts during Jan 2008, 2008-10-30 is when 13km RUC consistently available, 2008-11-17-1200 is first RUC with simulated reflectivity; also schema more stable (during 2007 CAPE and others only included during certain periods, presumably if relevant or not)
HOURS_OF_DAY   = (0..23).to_a
FORECAST_HOURS = [1,2,5,6,11,12,17,18] # RAP forcasts 0 - 21 hours ahead (0-18 before fall 2016). +0 is analysis, +1 barely comes out before its valid time, so +2 hours is the first real forecast.
BASE_URL       = FORECAST_HOURS.max <= 1 ? "https://nomads.ncdc.noaa.gov/data/rucanl" : "https://nomads.ncdc.noaa.gov/data/rap130" # Based a single sample, files in these two folders are identical if they exist in both
BASE_DIRECTORY = INV_OR_GRIB == "inv" ? "inventory_files" : "/Volumes/RAP_1/rap"
MIN_FILE_BYTES = INV_OR_GRIB == "inv" ? 1000 : 10_000_000
BAD_FILES      = %w[
  rap_130_20140928_1800_001.inv
]
# Possibly bad: ruc2anl_252_20091210_0500_001.inv missing fields 209+
# ruc2anl_252_20110601_1000_001.inv
# ruc2anl_252_20110601_1100_001.inv
THREAD_COUNT   = INV_OR_GRIB == "inv" ? 8 : 4

def alt_location(directory)
  directory.sub(/^\/Volumes\/RAP_1\//, "/Volumes/RAP_2/")
end

loop { break if Dir.exists?("/Volumes/RAP_1/"); puts "Waiting for RAP_1 to mount...";  sleep 4 }
loop { break if Dir.exists?("/Volumes/RAP_2/"); puts "Waiting for RAP_2 to mount..."; sleep 4 }

forecasts_to_get = DATES.product(HOURS_OF_DAY, FORECAST_HOURS)

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      date, run_hour, forecast_hour = forecast_to_get
      year_month        = "%04d%02d"     % [date.year, date.month]
      year_month_day    = "%04d%02d%02d" % [date.year, date.month, date.day]
      run_hour_str      = "%02d00"       % [run_hour]
      forecast_hour_str = "%03d"         % [forecast_hour]
      rap_or_ruc        =
        if year_month_day < "20120509"
          if INV_OR_GRIB == "inv" && forecast_hour <= 1
            if year_month_day < "20070101"
              "ruc2_252"
            elsif year_month_day < "20080201"
              "ruc2anl_252"
            elsif year_month_day < "20081030"
              "ruc2_252"
            else
              "ruc2anl_130"
            end
          else
            "ruc2anl_130"
          end
        else
          "rap_130"
        end
      file_name         = "#{rap_or_ruc}_#{year_month_day}_#{run_hour_str}_#{forecast_hour_str}.#{INV_OR_GRIB}"
      next if BAD_FILES.include?(file_name)
      url_to_get        = "#{BASE_URL}/#{year_month}/#{year_month_day}/#{file_name}" # e.g. "https://nomads.ncdc.noaa.gov/data/rap130/201612/20161231/rap_130_20161231_0600_002.grb2"
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
