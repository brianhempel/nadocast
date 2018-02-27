
system("mkdir inventory_changes 2> /dev/null")

unique_inventories = []

last_inventory_file = nil

Dir.glob("inventory_files/*/*/*.inv").each do |path|
  inventory = File.read(path).lines.map do |line|
    # 1:0:d=2012051023:HGT:1000 mb:1 hour fcst:
    n, _, _, abbrev, desc, hour = line.split(":")
    [abbrev, desc] if n
  end.compact.sort

  filtered_inventory = inventory.map { |fields| fields.join(":") }.join("\n") + "\n"

  filtered_path = "inventory_changes/#{File.basename(path)}"
  if !last_inventory_file
    puts "### #{filtered_path} ###"
    File.write(filtered_path, filtered_inventory)
    unique_inventories << inventory
    puts filtered_inventory
    puts
    last_inventory_file = filtered_path
  elsif filtered_inventory != File.read(last_inventory_file)
    puts "### #{filtered_path} ###"
    File.write(filtered_path, filtered_inventory)
    unique_inventories << inventory
    diff = `diff #{last_inventory_file} #{filtered_path}`
    puts diff
    puts
    last_inventory_file = filtered_path
  end
end

puts "### Common Fields ###"

# RUC has an HLCY coded as "surface" but is probably 0-3km helicity per https://ruc.noaa.gov/ruc/fslparms/13km/ruc13_grib.table-nov2008 and http://www.nws.noaa.gov/om/tpb/448body.htm (surface doesn't really make sense)
# Same for UV of storm motion.
# CAPE 255-0mb is MUCAPE https://ruc.noaa.gov/forum/f2/Welcome.cgi/read/2145
#
# 4LFTX and LFTX ??
#
# Low-level (180-0 mb agl, 90-0 mb agl) CAPE/CIN not added until 2014-02-25-1200. How to handle?? (Estimate for old data??)
#

unique_inventories_ignoring_number = unique_inventories.map do |inventory|
  inventory.map do |abbrev, desc|
    next(["MSTAV", "0 m underground"]) if abbrev == "MSTAV"
    next(["USTM", "u storm motion"]) if abbrev == "USTM"
    next(["VSTM", "v storm motion"]) if abbrev == "VSTM"
    next(["HLCY", "3000-0 m above ground"]) if abbrev == "HLCY" && desc == "surface"
    [
      abbrev.gsub(/\bDIST\b/, "HGT"), # Also changed from m to gpm (geo-potential meters); probably not a big deal.
      desc.
        gsub("0-6000 m above ground", "6000-0 m above ground").
        # gsub(/\bsfc\b/, "surface").
        # gsub(/\bgnd\b/, "ground").
        # gsub(/\bhigh trop\b/, "highest tropospheric").
        # gsub(/\blvl\b|\blev\b/, "level").
        # gsub(/\batmos col\b/, "entire atmosphere (considered as a single layer)").
        gsub(/\bentire atmosphere\z/, "entire atmosphere (considered as a single layer)").
        # gsub(/\b300 cm down\b|\b3 m underground\b/, "surface"). # BGRUN:300 cm down -> BGRUN:surface
        gsub(/\b3 m underground\b/, "surface") # BGRUN:3 m underground -> BGRUN:surface
        # gsub(/\bMSL\b/, "mean sea level").
        # gsub(/\bconvect-cld top\b/, "convective cloud top level").
        # gsub(/\bmax e-pot-temp\b/, "maximum equivalent potential temperature").
        # gsub(/\bmax wind level\b/, "max wind").
        # gsub(/\bof wet bulb\b/, "of the wet bulb").
        # gsub(/\bcld\b/, "cloud")
    ]
  end
end

common_inventory = unique_inventories_ignoring_number.reduce { |inv1, inv2| inv1 & inv2 }

common_inventory.each do |abbrev, desc|
  puts "#{abbrev}:#{desc}"
end

puts

puts "### Uncommon Fields ###"

uncommon_fields = unique_inventories_ignoring_number.flat_map do |fields|
  fields - common_inventory
end.uniq

uncommon_fields.each do |abbrev, desc|
  puts "#{abbrev}:#{desc}"
end
