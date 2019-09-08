# Used when adding SPC storm reports to tornadoes.csv.
#
# Merge any number of CSVs.
#
# Deduplicates and sorts rows, so should be idempotent.
#
# If given only one CSV, still deduplicates and sorts.
#
# $ ruby merge_csvs.rb file1.csv file2.csv file3.csv > out.csv
#
# See Makefile task "storm_events" at the project root.

require "csv"

csv_paths = ARGV

tables = csv_paths.map do |csv_path|
  CSV.read(csv_path, headers: true)
end

new_rows = tables.map { |table| table.map(&:fields) }.reduce([], :|).sort_by { |row| row.map(&:to_s) } # Prevent comparisons with nil

puts tables[0].headers.to_csv
new_rows.each do |row|
  puts row.to_csv
end
