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

# c.f. Inventories.jl
ENGLISH_NUMBERS = {
  0  => "zero",
  1  => "one",
  2  => "two",
  3  => "three",
  4  => "four",
  5  => "five",
  6  => "six",
  7  => "seven",
  8  => "eight",
  9  => "nine",
  10 => "ten",
  11 => "eleven",
  12 => "twelve",
  13 => "thirteen",
  14 => "fourteen",
  15 => "fifteen",
  16 => "sixteen",
  17 => "seventeen",
  18 => "eighteen",
  19 => "nineteen",
  20 => "twenty",
  30 => "thirty",
  40 => "fourty",
  50 => "fifty",
  60 => "sixty",
  70 => "seventy",
  80 => "eighty",
  90 => "ninety",
}

def int_to_english(n, elide_zero: false)
  if n < 0
    "negative " + int_to_english(-n)
  elsif n == 0 && elide_zero
    ""
  elsif n >= 1_000_000_000_000
    trillions, rest = n.divmod(1_000_000_000_000)
    int_to_english(trillions) + " trillion " + int_to_english(rest, elide_zero: true)
  elsif n >= 1_000_000_000
    billions, rest = n.divmod(1_000_000_000)
    int_to_english(billions) + " billion " + int_to_english(rest, elide_zero: true)
  elsif n >= 1_000_000
    millions, rest = n.divmod(1_000_000)
    int_to_english(millions) + " million " + int_to_english(rest, elide_zero: true)
  elsif n >= 1_000
    thousands, rest = n.divmod(1_000)
    int_to_english(thousands) + " thousand " + int_to_english(rest, elide_zero: true)
  elsif n >= 100
    hundreds, rest = n.divmod(100)
    int_to_english(hundreds) + " hundred " + int_to_english(rest, elide_zero: true)
  elsif ENGLISH_NUMBERS[n]
    ENGLISH_NUMBERS[n]
  else # 21 <= n <= 99 and n not divisible by 10
    tens, rest = n.divmod(10)
    ENGLISH_NUMBERS[tens*10] + "-" + ENGLISH_NUMBERS[rest]
  end.strip
end

# "7 hour fcst" => "hour fcst"
# "11-12 hour acc fcst" => "one hour long acc fcst"
# "11-12 hour max fcst" => "one hour long max fcst"
# c.f. Inventories.jl
def generic_forecast_hour_str(forecast_hour_str)
  # "11-12" => "one hour long"
  forecast_hour_str.gsub(/^\d+-\d+\s+hour/) do |range_str|
    start, stop = range_str.split(/[\- ]/)[0..1].map(&:to_i)
    int_to_english(stop - start) + " hour long"
  end.gsub(/\s*\d+\s*/, "") # "7 hour fcst" => "hour fcst"
end

# "7 hour fcst" => 7
# "11-12 hour acc fcst" => 12
# "11-12 hour max fcst" => 12
# c.f. Inventories.jl
def extract_forecast_hour(forecast_hour_str)
  Integer(forecast_hour_str[/(\d+) hour (\w+ )?fcst/, 1])
end

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
        select   { |line| line =~ /:(\d+-)?\d+ hour (\w+ )?fcst:/ }. # Exclude "0-1 day acc fcst"
        # select   { |line| line !~ /:APCP:/ }.          # Skip APCP Total Precipitation fields; for SREF they seem to be off by an hour and mess up the group_by below
        map      { |line| line.split(":") }.
        group_by { |_, _, _, _, _, x_hours_fcst_str, _, _| extract_forecast_hour(x_hours_fcst_str) }. # SREF forecasts contain multiple forecast hours in the same file: split them apart.
        map      { |x_hours_fcst, inventory_lines| inventory_lines }
    end.map do |inventory_lines|
      inventory_lines.map { |_, _, _, abbrev, desc, x_hours_fcst_str, prob_level, _| [abbrev, desc, generic_forecast_hour_str(x_hours_fcst_str), prob_level] }
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
