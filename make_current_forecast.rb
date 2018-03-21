model = ARGV[0] || raise("Need to provide a model to use for prediction")

ONE_MINUTE = 60

run_time = Time.now.utc - 60*ONE_MINUTE

# http://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/rap.20180319/rap.t00z.awp130pgrbf02.grib2

run_dmy_str  = "%04d%02d%02d" % [run_time.year, run_time.month, run_time.day]
run_hour     = run_time.hour
# run_hour_str = "%02d" % run_hour
dir_str      = "rap.#{run_dmy_str}"

(2..21).each do |forecast_hour|

  grib2_base_name = "rap.t%02dz.awp130pgrbf%02d.grib2" % [run_hour, forecast_hour]
  `mkdir forecast_grib2s/#{dir_str} 2> /dev/null`
  `mkdir forecasts/#{dir_str} 2> /dev/null`
  grib2_path = "forecast_grib2s/#{dir_str}/#{grib2_base_name}"
  grib2_url  = "http://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/#{dir_str}/#{grib2_base_name}"
  curl_cmd   = "curl #{grib2_url}"
  puts curl_cmd
  grib2_data = `#{curl_cmd}`
  if grib2_data.size > 10_000_000
    File.write(grib2_path, grib2_data)
    system("julia PredictAndPlot.jl #{model} #{grib2_path} forecasts/#{dir_str}")
    puts "forecasts/#{dir_str}/#{grib2_base_name.gsub(".grib2", "_tornado_probabilities.pdf")}"
  end
end
