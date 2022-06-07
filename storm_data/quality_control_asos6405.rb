# $ ruby quality_control_asos6405.rb gusts_deduped.csv > gusts_deduped_qced.csv
#
# See Makefile task "get_measured_wind_gusts" at the project root.

require "csv"

headers_written = false

MAX_GUST_FACTOR = Integer(ENV["MAX_GUST_FACTOR"] || "10")

CSV.readlines(ARGV[0], headers: true).each_with_index do |row, i|
  puts row.headers.to_csv if i == 0

  gust_factor = row["gust_knots"].to_f / row["knots"].to_f

  if gust_factor >= 1 && gust_factor <= MAX_GUST_FACTOR
    puts row.to_csv
  else
    STDERR.puts "Bad row, gust factor #{"%.2f" % gust_factor}: #{row.to_csv}"
  end
end
