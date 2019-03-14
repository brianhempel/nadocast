# Usage: $ ruby find_common_layers.rb path/to/folder [path_filter] [layer_filter]
#
# Reads a sample (n=500) of the grib2 inventories deeper than the given directory and outputs the common layers to STDOUT.
#
# Uncommon layers are output to STDERR.
#
# Optional `path_filter` is straight Ruby you can provide. The path is in a `path` variable--just be sure to return a boolean.

require "fileutils"
require "json"

NAME_NORMALIZATION_SUBSTITUTIONS = JSON.parse(File.read(File.expand_path("../layer_name_normalization_substitutions.json", __FILE__)))

root_path    = ARGV[0] || (STDERR.puts "Usage: $ ruby find_common_layers.rb path/to/folder [path_filter] [layer_filter]"; exit 1)
path_filter  = ARGV[1]
layer_filter = ARGV[2]
sample_count = 500


FileUtils.cd root_path

grib2_paths = Dir.glob("**/*").grep(/.gri?b2\z/i).select { |path| path_filter ? eval(path_filter) : true }


def normalize_inventory_line(line)
  NAME_NORMALIZATION_SUBSTITUTIONS.each do |target, replacement|
    line = line.gsub(target, replacement)
  end
  line
end

STDERR.print "Sampling #{sample_count} files to find common layers"

inventories =
  grib2_paths.
    sample(sample_count).
    map do |grib2_path|
      STDERR.print "."; STDERR.flush
      inventory = `wgrib2 #{grib2_path} -s -n`
      if $?.success?
        inventory
      else
        STDERR.print "SKIP"; STDERR.flush
        nil
      end
    end.
    compact.
    flat_map do |inventory_str|
      # STDERR.puts inventory_str
      # STDERR.puts
      inventory_str.
        split("\n").
        map      { |line| normalize_inventory_line(line) }.
        select   { |line| line =~ /:\d+ hour fcst:/ }. # Skip accumulation fields; for HREF they vary from forecast hour to forecast hour.
        select   { |line| line !~ /:APCP:/ }.          # Skip APCP Total Precipitation fields; for SREF they seem to be off by an hour and mess up the group_by below
        map      { |line| line.split(":") }.
        group_by { |_, _, _, _, _, x_hours_fcst, _, _| x_hours_fcst }. # SREF forecasts contain multiple forecast hours in the same file: split them apart.
        map      { |x_hours_fcst, inventory_lines| inventory_lines }
    end.map do |inventory_lines|
      inventory_lines.map { |_, _, _, abbrev, desc, x_hours_fcst, prob_level, _| [abbrev, desc, x_hours_fcst.gsub(/\s*\d+\s*/, ""), prob_level] }
    end

common   = inventories.reduce(:&)
uncommon = inventories.reduce(:|) - common

if layer_filter
  common_filtered = common.select { |abbrev, desc, hours_fcst, prob_level| eval(layer_filter) }
else
  common_filtered = common
end
rejected_by_filter = common - common_filtered


STDERR.puts
puts common_filtered.map { |key_parts| key_parts.join(":") }.join("\n")

if uncommon.any?
  STDERR.puts
  STDERR.puts "Uncommon layers:"
  STDERR.puts uncommon.map { |key_parts| key_parts.join(":") }.join("\n")
end

if rejected_by_filter.any?
  STDERR.puts
  STDERR.puts "Rejected layers:"
  STDERR.puts rejected_by_filter.map { |key_parts| key_parts.join(":") }.join("\n")
end
