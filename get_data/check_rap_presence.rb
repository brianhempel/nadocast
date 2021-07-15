#!/usr/bin/env ruby

# $ VALIDATION_RUN_HOURS=8,9,10,12,13,14 BASE_PATH=/Volumes/RAP_1/rap ruby check_rap_presence.rb 2014-2-1 2017-12-31

require 'date'

# "2019-1-10" to date
def str_to_date(str)
  year, month, day = str.split("-").map(&:to_i)
  Date.new(year, month, day)
end

validation_run_hours = ENV["VALIDATION_RUN_HOURS"]&.split(",")&.map(&:to_i) || []
base_path            = ENV["BASE_PATH"]

start_date = str_to_date(ARGV[0])
end_date   = str_to_date(ARGV[1])

dates     = (start_date..end_date).to_a
saturdays = dates.select(&:saturday?)

saturdays.each do |date|
  year_month_day = "%04d%02d%02d" % [date.year, date.month, date.day]
  year_month     = "%04d%02d"     % [date.year, date.month]
  validation_run_hours.each do |run_hour|
    run_hour_str = "%02d" % [run_hour]
    (1..18).each do |forecast_hour|
      forecast_hour_str = "%02d" % [forecast_hour]
      path = "#{base_path}/#{year_month}/#{year_month_day}/rap_130_#{year_month_day}_#{run_hour_str}00_0#{forecast_hour_str}.grb2"
      unless (File.size(path) rescue 0) >= 10_000_000
        puts "MISSING: #{path}"
      end
    end
  end
end
