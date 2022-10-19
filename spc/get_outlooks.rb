
require 'date'

# https://www.spc.noaa.gov/products/outlook/archive/2019/day1otlk_20190812_1300-shp.zip

# 2018-6-29 through 2022-5-31

# NOTE: I had to manually fix day1otlk_20200409_1300_wind.shp and day1otlk_20200407_1630_wind.shp and day1otlk_20200409_1630_wind.shp with the "Fix Geometries" tool in the "Processing Toolbox" pane in QGIS.

# DATES = (Date.new(2018,6,29)..Date.new(2019,1,7)).to_a
# DATES = (Date.new(2019,1,7)..Date.new(2022,2,22)).to_a
# DATES = (Date.new(2021,12,31)..Date.new(2022,2,22)).to_a
DATES = (Date.new(2022,6,1)..Date.new(2022,7,31)).to_a
HOURS_TO_GET = %w[1200 1300 1630] # 0600Z 1300Z and 1630Z SPC outlooks, corresponding to 0Z 10Z and 14Z Nadocast forecasts.

class Date
  def yyyymmdd
    "%04d%02d%02d" % [year, month, day]
  end

  def yyyymm
    "%04d%02d" % [year, month]
  end
end

DATES.each do |date|
  HOURS_TO_GET.each do |hour|
    file_name  = "day1otlk_#{date.yyyymmdd}_#{hour}-shp.zip"
    url_to_get = "https://www.spc.noaa.gov/products/outlook/archive/#{date.year}/#{file_name}"
    dir = date.yyyymm
    path       = "#{dir}/#{file_name}"
    if (File.size("#{dir}/day1otlk_#{date.yyyymmdd}_#{hour}_torn.shp") rescue 0) == 0
      system("mkdir -p #{dir} 2> /dev/null")
      puts "#{url_to_get} -> #{path}"
      data = `curl -f -s --show-error #{url_to_get}`
      File.write(path, data) if $?.success?
      system("unzip -d #{dir} #{path} && rm #{path}")
    end
  end
end
