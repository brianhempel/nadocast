require "date"
require "csv"

# Usage:   $ ruby get_spc_storm_reports.rb start_date end_date
# Example: $ ruby get_spc_storm_reports.rb 2017-12-31 2019-01-01 > tornadoes_from_spc_storm_reports.csv
#
# Dates are "convective days" i.e. 12Z - 12Z.
#
# Fetches tornadoes from SPC storm reports. Does not separate start/end time and
# is a less precise data source. But, the storm reports are available immediately
# and they're still online during the current gov't shutdown (cough cough).
#
# To run this script, see the Makefile at the project root.

# ROOT_URL = "https://www.spc.noaa.gov/climo/reports/181201_rpts_filtered.csv"

start_date_str, end_date_str = [ARGV[0], ARGV[1]]

unless start_date_str && end_date_str
  STDERR.puts "Expected start and end date, e.g.:"
  STDERR.puts "$ ruby get_spc_storm_reports.rb 2017-12-31 2019-01-01"
  exit 1
end

start_date = Date.parse(start_date_str)
end_date   = Date.parse(end_date_str)

MINUTE = 60
HOUR   = 60*MINUTE

print %w[
  begin_time_str
  begin_time_seconds
  end_time_str
  end_time_seconds
  f_scale
  begin_lat
  begin_lon
  end_lat
  end_lon
].to_csv

(start_date..end_date).each do |date|
  yymmdd = "%02d%02d%02d" % [date.year % 100, date.month, date.day]

  STDERR.puts "https://www.spc.noaa.gov/climo/reports/#{yymmdd}_rpts_filtered.csv"

  day_reports_csv_str = `curl https://www.spc.noaa.gov/climo/reports/#{yymmdd}_rpts_filtered.csv`
  day_reports_csv_str.gsub!('"', 'inch') # SPC CSV doesn't quote fields so " for "inch" messes us up.

  rows = CSV.parse(day_reports_csv_str, headers: true)

  # Grab only the tornado reports and discard the rest.
  # (The headers are repeated above each section)
  tornado_rows = rows.take_while { |row| row["Time"] =~ /\A\d+\z/ }

  tornado_rows.map! do |row|
    # 1955 => 2018-05-10 19:55:00 UTC
    hour, minute = row["Time"].to_i.divmod(100)

    hour += 24 if hour < 12 # Convective days are 12Z - 12Z, so times < 12 hours are the next UTC day.

    time = Time.new(date.year, date.month, date.day, 0,0,0, "+00:00") + hour*HOUR + minute*MINUTE

    [
      time.to_s.gsub("+0000", "UTC"),  # begin_time_str
      time.to_i,  # begin_time_seconds
      time.to_s.gsub("+0000", "UTC"),  # end_time_str
      time.to_i,  # end_time_seconds
      "-1",       # f_scale
      row["Lat"], # begin_lat
      row["Lon"], # begin_lon
      row["Lat"], # end_lat
      row["Lon"], # end_lon
    ]
  end

  tornado_rows.map(&:to_csv).sort.each do |row_csv_str|
    print row_csv_str
  end

rescue CSV::MalformedCSVError => e
  STDERR.puts e
  STDERR.puts e.message
  STDERR.puts day_reports_csv_str
end
