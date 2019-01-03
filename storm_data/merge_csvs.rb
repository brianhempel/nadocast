# Used when adding SPC storm reports to tornadoes.rb.
#
# Deduplicates and sorts rows, so should be idempotent.
#
# $ ruby merge_csvs.rb file1.csv file2.csv > out.csv
#
# See Makefile task "add_2018_spc_tornado_reports" at the project root.

require "csv"

csv_path_1, csv_path_2 = [ARGV[0], ARGV[1]]

unless csv_path_1 && csv_path_2
  STDERR.puts "Expected two CSV files, e.g.:"
  STDERR.puts "$ ruby merge_csvs.rb file1.csv file2.csv"
  exit 1
end

table_1 = CSV.read(csv_path_1, headers: true)
table_2 = CSV.read(csv_path_2, headers: true)

new_rows = (table_1.map(&:fields) | table_2.map(&:fields)).sort

puts table_1.headers.to_csv
new_rows.each do |row|
  puts row.to_csv
end
