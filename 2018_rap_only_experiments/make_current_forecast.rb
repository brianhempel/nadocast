model_jl = ARGV[0] || raise("Need to provide a Julia model to use for prediction")
model    = ARGV[1] || raise("Need to provide a saved model to use for prediction")

THREAD_COUNT = Integer(ARGV[2] || "4")

ONE_MINUTE = 60

FORECASTING_FROM_ARCHIVE = ENV["FORECAST_FROM_ARCHIVE"].to_s.size > 0

# The files we want typically don't appear until pretty late in the hour.
RUN_TIME =
  if FORECASTING_FROM_ARCHIVE
    year_str, month_str, day_str, run_hour_str = ENV["FORECAST_FROM_ARCHIVE"].scan(/\d+/)  # "2018-07-19@13Z"
    Time.utc(year_str.to_i, month_str.to_i, day_str.to_i, run_hour_str.to_i, 0, 0)
  else
    Time.now.utc - Integer(ENV["FORECAST_DELAY_MINUTES"] || "45")*ONE_MINUTE
  end

# http://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.20180319/rap.t00z.awp130pgrbf02.grib2

RUN_DMY_STR  = "%04d%02d%02d" % [RUN_TIME.year, RUN_TIME.month, RUN_TIME.day]
RUN_HOUR     = RUN_TIME.hour
# run_hour_str = "%02d" % run_hour
DIR_STR      = "rap.#{RUN_DMY_STR}"

hours_to_forecast = (2..21).to_a


def get_grib2(forecast_hour)
  grib2_base_name = "rap.t%02dz.awp130pgrbf%02d.grib2" % [RUN_HOUR, forecast_hour]
  `mkdir forecast_grib2s/#{DIR_STR} 2> /dev/null`
  `mkdir forecasts/#{DIR_STR} 2> /dev/null`
  grib2_path = "forecast_grib2s/#{DIR_STR}/#{grib2_base_name}"
  grib2_url  =
    if FORECASTING_FROM_ARCHIVE
      year_month        = "%04d%02d" % [RUN_TIME.year, RUN_TIME.month]
      run_hour_str      = "%02d00"       % [RUN_HOUR]
      forecast_hour_str = "%03d"         % [forecast_hour]
      file_name         = "rap_130_#{RUN_DMY_STR}_#{run_hour_str}_#{forecast_hour_str}.grb2"
      "https://nomads.ncdc.noaa.gov/data/rap130/#{year_month}/#{RUN_DMY_STR}/#{file_name}" # e.g. "https://nomads.ncdc.noaa.gov/data/rap130/201612/20161231/rap_130_20161231_0600_002.grb2"
    else
      "http://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/#{DIR_STR}/#{grib2_base_name}"
    end
  curl_cmd   = "curl #{grib2_url}"

  if (File.size(grib2_path) rescue 0) <= 10_000_000
    puts curl_cmd
    grib2_data = `#{curl_cmd}`
    if grib2_data.size > 10_000_000
      File.write(grib2_path, grib2_data)
      grib2_path
    end
  else
    grib2_path
  end
end

grib2_paths = {}
TIMEOUT = 15*ONE_MINUTE

Thread.new do
  ([hours_to_forecast.min - 1] + hours_to_forecast).each do |forecast_hour|
    grib2_paths[forecast_hour] = get_grib2(forecast_hour)
    (TIMEOUT / 20).times do |n|
      unless grib2_paths[forecast_hour]
        puts "Waiting on #{RUN_HOUR}Z +#{forecast_hour}hr RAP file..."
        sleep 20
        grib2_paths[forecast_hour] = get_grib2(forecast_hour)
      end
    end
  end
  # p grib2_paths
end

hours_remaining_to_forecast = hours_to_forecast.dup

threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_hour = hours_remaining_to_forecast.shift
      timeout = TIMEOUT
      until grib2_paths[forecast_hour - 1] && grib2_paths[forecast_hour]
        timeout -= 1
        Thread.current.exit if timeout < 0
        sleep 1
      end

      prior_hour_grib2_path = grib2_paths[forecast_hour - 1]
      grib2_path            = grib2_paths[forecast_hour]
      grib2_base_name       = File.basename(grib2_path)
      system("julia PredictAndPlot.jl #{model_jl} #{model} #{prior_hour_grib2_path} #{grib2_path} forecasts/#{DIR_STR}")
      puts "forecasts/#{DIR_STR}/#{grib2_base_name.gsub(".grib2", "_tornado_probabilities.pdf")}"
    end
  end
end

threads.each(&:join)
