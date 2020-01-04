module Inventories

struct InventoryLine
  # "4", "956328", "d=2018062900", "CAPE", "180-0 mb above ground", "7 hour fcst", "wt ens mean", "n=4"
  message_dot_submessage :: String # "3" or "3.2"
  position_str           :: String # "956328"
  date_str               :: String # "d=2018062900"
  abbrev                 :: String # "CAPE"
  level                  :: String # "180-0 mb above ground"
  forecast_hour_str      :: String # "7 hour fcst" or "11-12 hour acc fcst" or "11-12 hour ave fcst"  or "11-12 hour max fcst"
  misc                   :: String # "wt ens mean" or "prob >2.54"
  feature_engineering    :: String # "" or "25mi mean" or "100mi forward grad" etc
end

function revise_with_feature_engineering(line :: InventoryLine, feature_engineering :: String) :: InventoryLine
  InventoryLine(
    line.message_dot_submessage,
    line.position_str,
    line.date_str,
    line.abbrev,
    line.level,
    line.forecast_hour_str,
    line.misc,
    feature_engineering
  )
end

function revise_with_misc(line :: InventoryLine, misc :: String) :: InventoryLine
  InventoryLine(
    line.message_dot_submessage,
    line.position_str,
    line.date_str,
    line.abbrev,
    line.level,
    line.forecast_hour_str,
    misc,
    line.feature_engineering
  )
end

struct FieldMissing <: Exception
  forecast_str :: String
  missing_key  :: String
  inventory    :: Vector{InventoryLine}
end

Base.showerror(io::IO, e::FieldMissing) = print(io, e.forecast_str, " is missing ", e.missing_key, ". Inventory: ", join(map(inventory_line_key, e.inventory), "\t"))

# c.f. extract_forecast_hour in find_common_layers.rb
function forecast_hour(line :: InventoryLine) :: Int64
  try
    if line.forecast_hour_str == "anl"
      return 0
    elseif occursin(" day ", line.forecast_hour_str) # "0-1 day acc fcst" We don't use these yet.
      return -1
    end
    hour_str, _ = match(r"(\d+) hour (\w+ )?fcst", line.forecast_hour_str).captures
    parse(Int64, hour_str)
  catch exception
    println("Bad line.forecast_hour_str: $(line.forecast_hour_str)")
    rethrow(exception)
  end
end

english_numbers = Dict(
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
  90 => "ninety"
)

function int_to_english(n; elide_zero = false)
  if n < 0
    "negative " * int_to_english(-n)
  elseif n == 0 && elide_zero
    ""
  elseif n >= 1_000_000_000_000
    trillions, rest = divrem(n, 1_000_000_000_000)
    strip(int_to_english(trillions) * " trillion " * int_to_english(rest, elide_zero = true))
  elseif n >= 1_000_000_000
    billions, rest = divrem(n, 1_000_000_000)
    strip(int_to_english(billions) * " billion " * int_to_english(rest, elide_zero = true))
  elseif n >= 1_000_000
    millions, rest = divrem(n, 1_000_000)
    strip(int_to_english(millions) * " million " * int_to_english(rest, elide_zero = true))
  elseif n >= 1_000
    thousands, rest = divrem(n, 1_000)
    strip(int_to_english(thousands) * " thousand " * int_to_english(rest, elide_zero = true))
  elseif n >= 100
    hundreds, rest = divrem(n, 100)
    strip(int_to_english(hundreds) * " hundred " * int_to_english(rest, elide_zero = true))
  elseif haskey(english_numbers, n)
    english_numbers[n]
  else # 21 <= n <= 99 and n not divisible by 10
    tens, rest = divrem(n, 10)
    english_numbers[tens*10] * "-" * english_numbers[rest]
  end
end

# "7 hour fcst" => "hour fcst"
# "11-12 hour acc fcst" => "one hour long acc fcst"
# "11-12 hour max fcst" => "one hour long max fcst"
# "0-1 day acc fcst" => "- day acc fcst" we aren't using these fields yet
# c.f. find_common_layers.rb
function generic_forecast_hour_str(forecast_hour_str)
  # "11-12" => "one hour long"
  duration_normalizer(str) = begin
    start, stop = map(num_str -> parse(Int64, num_str), split(str, r"[\- ]")[1:2])
    int_to_english(stop - start) * " hour long"
  end
  forecast_hour_str = replace(forecast_hour_str, r"^\d+-\d+\s+hour"  => duration_normalizer)
  forecast_hour_str = replace(forecast_hour_str, r"\s*\d+\s*" => "") # "7 hour fcst" => "hour fcst"
  forecast_hour_str
end

# "REFD:1000 m above ground:hour fcst:prob >40"
function inventory_line_key(line :: InventoryLine) :: String
  # "7 hour fcst" => "hour fcst"
  # c.f. find_common_layers.rb
  join([line.abbrev, line.level, generic_forecast_hour_str(line.forecast_hour_str), line.misc], ":")
end

# "REFD:1000 m above ground:7 hour fcst:prob >40"
function specific_inventory_line_key(line :: InventoryLine) :: String
  join([line.abbrev, line.level, line.forecast_hour_str, line.misc], ":")
end

# "REFD:1000 m above ground:hour fcst:prob >40:25mi mean"
function inventory_line_description(line :: InventoryLine) :: String
  # "7 hour fcst" => "hour fcst"
  # c.f. find_common_layers.rb
  join([line.abbrev, line.level, generic_forecast_hour_str(line.forecast_hour_str), line.misc, line.feature_engineering], ":")
end

end # module Inventories