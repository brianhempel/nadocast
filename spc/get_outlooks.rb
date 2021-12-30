
require 'date'

# https://www.spc.noaa.gov/products/outlook/archive/2019/day1otlk_20190812_1300-shp.zip

# 2019-01-7 through 2021-8-31

DATES = (Date.new(2019,1,7)..Date.new(2021,8,31)).to_a
HOURS_TO_GET = %w[1200 1300 1630] # 0600Z 1300Z and 1630Z SPC outlooks, corresponding to 0Z 10Z and 14Z Nadocast forecasts.

class Date
  def year_month_day
    "%04d%02d%02d" % [year, month, day]
  end

  def year_month
    "%04d%02d" % [year, month]
  end
end

DATES.each do |date|
  HOURS_TO_GET.each do |hour|
    file_name  = "day1otlk_#{date.year_month_day}_#{hour}-shp.zip"
    url_to_get = "https://www.spc.noaa.gov/products/outlook/archive/#{date.year}/#{file_name}"
    dir = date.year_month
    path       = "#{dir}/#{file_name}"
    if (File.size("#{dir}/day1otlk_#{date.year_month_day}_#{hour}_torn.shp") rescue 0) == 0
      system("mkdir -p #{dir} 2> /dev/null")
      puts "#{url_to_get} -> #{path}"
      data = `curl -f -s --show-error #{url_to_get}`
      File.write(path, data) if $?.success?
      system("unzip -d #{dir} #{path} && rm #{path}")
    end
  end
end
