require "date"

# Modify these constants.
INV_OR_GRIB    = %w[inv grb2][1]
DATES          = (Date.new(2016,11,18)..Date.new(2018,2,16)-1).to_a # 2005-10-31 is first +1hr with inv, no forecasts during Jan 2008, 2008-10-30 is when 13km RUC consistently available, 2008-11-17-1200 is first RUC with simulated reflectivity; also schema more stable (during 2007 CAPE and others only included during certain periods, presumably if relevant or not)
HOURS_OF_DAY   = (0..23).to_a
FORECAST_HOURS = (1..1).to_a # RAP forcasts 0 - 18 hours ahead.
BASE_URL       = FORECAST_HOURS.max <= 1 ? "https://nomads.ncdc.noaa.gov/data/rucanl" : "https://nomads.ncdc.noaa.gov/data/rap130" # Based a single sample, files in these two folders are identical if they exist in both
BASE_DIRECTORY = INV_OR_GRIB == "inv" ? "inventory_files" : "/Volumes/Tornadoes/rap"
MIN_FILE_BYTES = INV_OR_GRIB == "inv" ? 1000 : 10_000_000
BAD_FILES      = %w[
  rap_130_20140928_1800_001.inv
]
# Possibly bad: ruc2anl_252_20091210_0500_001.inv missing fields 209+
# ruc2anl_252_20110601_1000_001.inv
# ruc2anl_252_20110601_1100_001.inv
THREAD_COUNT   = INV_OR_GRIB == "inv" ? 8 : 4



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

      system("mkdir -p #{directory} 2> /dev/null")
      if (File.size(path) rescue 0) < MIN_FILE_BYTES
        puts "#{url_to_get} -> #{path}"
        data = `curl -f -s --show-error #{url_to_get}`
        if data.size >= MIN_FILE_BYTES
          File.write(path, data)
        end
      end
    end
  end
end

threads.each(&:join)
