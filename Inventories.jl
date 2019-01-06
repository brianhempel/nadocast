module Inventories

struct InventoryLine
  # "4", "956328", "d=2018062900", "CAPE", "180-0 mb above ground", "7 hour fcst", "wt ens mean", "n=4"
  message_dot_submessage :: String # "3" or "3.2"
  position_str           :: String # "956328"
  date_str               :: String # "d=2018062900"
  abbrev                 :: String # "CAPE"
  level                  :: String # "180-0 mb above ground"
  forecast_hour_str      :: String # "7 hour fcst" or "6-hour acc fcst"
  misc                   :: String # "wt ens mean" or "prob >2.54"
end

# "REFD:1000 m above ground:hour fcst:prob >40"
function inventory_line_key(line :: InventoryLine) :: String
  # "7 hour fcst" => "hour fcst"
  # c.f. find_common_layers.rb
  generic_forecast_hour_str = replace(line.forecast_hour_str, r"\s*\d+\s*" => "")
  join([line.abbrev, line.level, generic_forecast_hour_str, line.misc], ":")
end

# "REFD:1000 m above ground:7 hour fcst:prob >40"
function specific_inventory_line_key(line :: InventoryLine) :: String
  join([line.abbrev, line.level, line.forecast_hour_str, line.misc], ":")
end

end # module Inventories