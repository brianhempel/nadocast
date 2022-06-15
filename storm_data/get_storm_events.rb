require "date"
require "csv"

# Fetches the storm events database from (ENV["START_YEAR"] || 2014) through
# the current year (or ENV["STOP_YEAR"]) and outputs all the tornado, wind, and hail events.
#
# Primarily uses the storm events database because it has start and end times and locations.
#
# The previous year is finalized near the end of Spring of the following year.
#
# If ARGV[3] is "--add_spc_storm_reports", then SPC storm reports from the
# end of the storm events database until one week ago are included.
#
# To run this script, see the Makefile at the project root.

tornadoes_csv_out_path   = ARGV[0] || "tornadoes.csv"
wind_events_csv_out_path = ARGV[1] || "wind_events.csv"
hail_events_csv_out_path = ARGV[2] || "hail_events.csv"

START_YEAR = Integer(ENV["START_YEAR"] || "2014")
STOP_YEAR  = Integer(ENV["STOP_YEAR"]  || Time.now.year)

# Some longitudes are encoded with their decimal point apparently off by one (e.g. -812.15 instead of, presumably -81.215)
# Whether it's an insertion error or an off by one error is not clear so repair is attempted by assuming the start/end longitude
# should be the same and using the other if it looks good. Not perfect, but more reasonable for climatological purposes than
# discarding or not fixing.
DO_REPAIR_LATLONS = (ENV["BAD_LATLON_HANDLING"] == "repair")


# Part 1: Storm Events Database

ROOT_URL = "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

file_names = `curl #{ROOT_URL}`.scan(/StormEvents_details-ftp_v1\.0_d\d\d\d\d_c\d+\.csv\.gz/).uniq

if file_names.empty?
  STDERR.puts "Could not retreive storm event database file list from https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
  exit 1
end

tornadoes_file   = open(tornadoes_csv_out_path, "w")
wind_events_file = open(wind_events_csv_out_path, "w")
hail_events_file = open(hail_events_csv_out_path, "w")


BEGIN_END_TIMES_HEADERS = %w[
  begin_time_str
  begin_time_seconds
  end_time_str
  end_time_seconds
]

LAT_LON_HEADERS = %w[
  begin_lat
  begin_lon
  end_lat
  end_lon
]

tornadoes_file.print   (BEGIN_END_TIMES_HEADERS + %w[f_scale length width] + LAT_LON_HEADERS).to_csv
wind_events_file.print (BEGIN_END_TIMES_HEADERS + %w[kind speed speed_type source] + LAT_LON_HEADERS).to_csv
hail_events_file.print (BEGIN_END_TIMES_HEADERS + %w[kind inches] + LAT_LON_HEADERS).to_csv

# Event types:
# Astronomical Low Tide
# Avalanche
# Blizzard
# Coastal Flood
# Cold/Wind Chill
# Debris Flow
# Dense Fog
# Dense Smoke
# Drought
# Dust Devil
# Dust Storm
# Excessive Heat
# Extreme Cold/Wind Chill
# Flash Flood
# Flood
# Freezing Fog
# Frost/Freeze
# Funnel Cloud
# Hail *
# Heat
# Heavy Rain
# Heavy Snow
# High Surf
# High Wind (Not geocoded!)
# Hurricane
# Ice Storm
# Lake-Effect Snow
# Lakeshore Flood
# Lightning
# Marine Hail
# Marine High Wind
# Marine Hurricane/Typhoon
# Marine Strong Wind (weaker than "high wind")
# Marine Thunderstorm Wind
# Marine Tropical Depression
# Marine Tropical Storm
# Rip Current
# Seiche
# Sleet
# Sneakerwave
# Storm Surge/Tide
# Strong Wind (weaker than "high wind")
# Thunderstorm Wind *
# Tornado *
# Tropical Depression
# Tropical Storm
# Volcanic Ashfall
# Waterspout
# Wildfire
# Winter Storm
# Winter Weather

def begin_end_times(row)
  begin_year_month_str = row["BEGIN_YEARMONTH"]
  begin_day_str        = row["BEGIN_DAY"]
  begin_time_str       = row["BEGIN_TIME"]
  end_year_month_str   = row["END_YEARMONTH"]
  end_day_str          = row["END_DAY"]
  end_time_str         = row["END_TIME"]
  tz_offset_hrs        = row["CZ_TIMEZONE"][/-?\d+/].to_i

  begin_time = Time.new(begin_year_month_str.to_i / 100, begin_year_month_str.to_i % 100, begin_day_str.to_i, begin_time_str.to_i / 100, begin_time_str.to_i % 100, 00, "%+03d:00" % tz_offset_hrs)
  end_time   = Time.new(  end_year_month_str.to_i / 100,   end_year_month_str.to_i % 100,   end_day_str.to_i,   end_time_str.to_i / 100,   end_time_str.to_i % 100, 00, "%+03d:00" % tz_offset_hrs)

  [begin_time, end_time]
end

# begin_time_str,begin_time_seconds,end_time_str,end_time_seconds
# 2014-01-11 12:37:00 UTC,1389443820,2014-01-11 12:41:00 UTC,1389444060
def row_to_begin_end_time_cells(row)
  begin_time, end_time = begin_end_times(row)

  begin_end_time_cells(begin_time, end_time)
end


def begin_end_time_cells(begin_time, end_time)
  [
    begin_time.utc.to_s,
    begin_time.utc.to_i,
    end_time.utc.to_s,
    end_time.utc.to_i
  ]
end

# begin_lat,begin_lon,end_lat,end_lon
# 34.3328,-84.5286,34.3476,-84.4811
def row_to_lat_lon_cells(row)
  [
    row["BEGIN_LAT"] || row["Lat"],
    row["BEGIN_LON"] || row["Lon"],
    row["END_LAT"] || row["Lat"],
    row["END_LON"] || row["Lon"]
  ]
end

def valid_lat?(lat)
  (1..90).cover?(lat)
end

def valid_lon?(lon)
  (-180..-2).cover?(lon)
end

# Mutates row.
def perhaps_repair_latlons!(row)
  if DO_REPAIR_LATLONS
    if row["BEGIN_LAT"] && row["END_LAT"]
      if valid_lat?(row["BEGIN_LAT"].to_f) && !valid_lat?(row["END_LAT"].to_f)
        STDERR.puts "Repairing latitude #{row["END_LAT"]} to #{row["BEGIN_LAT"]}"
        row["END_LAT"] = row["BEGIN_LAT"]
      elsif !valid_lat?(row["BEGIN_LAT"].to_f) && valid_lat?(row["END_LAT"].to_f)
        STDERR.puts "Repairing latitude #{row["BEGIN_LAT"]} to #{row["END_LAT"]}"
        row["BEGIN_LAT"] = row["END_LAT"]
      end
    end
    if row["BEGIN_LON"] && row["END_LON"]
      if valid_lon?(row["BEGIN_LON"].to_f) && !valid_lon?(row["END_LON"].to_f)
        STDERR.puts "Repairing longitude #{row["END_LON"]} to #{row["BEGIN_LON"]}"
        row["END_LON"] = row["BEGIN_LON"]
      elsif !valid_lon?(row["BEGIN_LON"].to_f) && valid_lon?(row["END_LON"].to_f)
        STDERR.puts "Repairing longitude #{row["BEGIN_LON"]} to #{row["END_LON"]}"
        row["BEGIN_LON"] = row["END_LON"]
      end
    end
  end
end

def valid_lat_lon?(row)
  valid_lat?((row["BEGIN_LAT"] || row["Lat"]).to_f) &&
  valid_lon?((row["BEGIN_LON"] || row["Lon"]).to_f) &&
  valid_lat?((row["END_LAT"]   || row["Lat"]).to_f) &&
  valid_lon?((row["END_LON"]   || row["Lon"]).to_f)
end

last_storm_events_database_event_time = Time.new(START_YEAR)

(START_YEAR..STOP_YEAR).each do |year|
  file_name = file_names.grep(/v1\.0_d#{year}_/).last

  next unless file_name

  rows = CSV.parse(`curl #{ROOT_URL + file_name} | gunzip`, headers: true)

  # STDERR.puts "Event types: #{rows.map { |row| row["EVENT_TYPE"] }.uniq.sort}"

  tornado_rows = rows.select { |row| row["EVENT_TYPE"].strip == "Tornado" }
  wind_rows    = rows.select { |row| row["EVENT_TYPE"].strip == "Thunderstorm Wind" }
  hail_rows    = rows.select { |row| row["EVENT_TYPE"].strip == "Hail" }

  STDERR.puts "#{tornado_rows.count} tornado path pieces in #{year}"
  STDERR.puts "#{wind_rows.count} thunderstorm wind events in #{year}"
  STDERR.puts "#{hail_rows.count} hail events in #{year}"

  tornado_rows.select! { |row| perhaps_repair_latlons!(row); valid_lat_lon?(row) }
  tornado_rows.map! do |row|
    last_storm_events_database_event_time = [begin_end_times(row)[1], last_storm_events_database_event_time].max

    row_to_begin_end_time_cells(row) +
    [
      row["TOR_F_SCALE"][/\d+/], # f_scale
      row["TOR_LENGTH"].to_f,
      row["TOR_WIDTH"].to_i,
    ] +
    row_to_lat_lon_cells(row)
  end

  # EG = Estimated Gust, MG = Measured Gust, ES = Estimated Sustained, MS = Measured Sustained
  wind_type = {
    "EG" => "gust",
    "MG" => "gust",
    "ES" => "sustained",
    "MS" => "sustained",
  }

  wind_source = {
    "EG" => "estimated",
    "MG" => "measured",
    "ES" => "estimated",
    "MS" => "measured",
  }

  # Although we might use wind hours without geocodes for negative data, we aren't yet.
  wind_rows.select! { |row| perhaps_repair_latlons!(row); valid_lat_lon?(row) }
  # There are so few "sustained" thunderstorm winds events, and they all look like short events so no reason to worry about including or excluding them.
  # wind_rows.select! { |row| (wind_type[row["MAGNITUDE_TYPE"]] || row["MAGNITUDE_TYPE"]) != "sustained" }
  wind_rows.map! do |row|
    last_storm_events_database_event_time = [begin_end_times(row)[1], last_storm_events_database_event_time].max

    row_to_begin_end_time_cells(row) +
    [
      row["EVENT_TYPE"], # kind
      (row["MAGNITUDE"].to_s)[/[\d\.]+/] || "-1", # speed
      wind_type[row["MAGNITUDE_TYPE"]] || row["MAGNITUDE_TYPE"], # speed_type
      wind_source[row["MAGNITUDE_TYPE"]]
    ] +
    row_to_lat_lon_cells(row)
  end

  hail_rows.select! { |row| perhaps_repair_latlons!(row); valid_lat_lon?(row) }
  hail_rows.map! do |row|
    last_storm_events_database_event_time = [begin_end_times(row)[1], last_storm_events_database_event_time].max

    row_to_begin_end_time_cells(row) +
    [
      row["EVENT_TYPE"],
      (row["MAGNITUDE"].to_s)[/[\d\.]+/] || "-1", # Hail size
      # row["TOR_LENGTH"].to_f,
      # row["TOR_WIDTH"].to_i,
    ] +
    row_to_lat_lon_cells(row)
  end

  tornado_rows.map(&:to_csv).sort.each do |row_csv_str|
    tornadoes_file.print row_csv_str
  end

  wind_rows.map(&:to_csv).sort.each do |row_csv_str|
    wind_events_file.print row_csv_str
  end

  hail_rows.map(&:to_csv).sort.each do |row_csv_str|
    hail_events_file.print row_csv_str
  end
end


# Part 2: SPC Storm Reports from the end of the storm events database until 1 week ago

if ARGV[3] == "--add_spc_storm_reports"

  start_date = last_storm_events_database_event_time.to_date + 1
  end_date   = Date.today - 7

  puts "Adding SPC storm reports for #{start_date} through #{end_date}"

  MINUTE = 60
  HOUR   = 60*MINUTE

  def spc_storm_report_row_to_time(date, row)
    # 1955 => 2018-05-10 19:55:00 UTC
    hour, minute = row["Time"].to_i.divmod(100)

    hour += 24 if hour < 12 # Convective days are 12Z - 12Z, so times < 12 hours are the next UTC day.

    Time.new(date.year, date.month, date.day, 0,0,0, "+00:00") + hour*HOUR + minute*MINUTE
  end

  (start_date..end_date).each do |date|
    yymmdd = "%02d%02d%02d" % [date.year % 100, date.month, date.day]

    STDERR.puts "https://www.spc.noaa.gov/climo/reports/#{yymmdd}_rpts_filtered.csv"

    # Time,F_Scale,Location,County,State,Lat,Lon,Comments
    # 1954,UNK,2 E PLEASANT HILL,PIKE,IL,39.45,-90.84,NWS DAMAGE SURVEY CREW CONFIRMED AN EF-1 TORNADO NEAR PLEASANT HILL ILLINOIS. MAX WINDS WERE 98 MPH. PATH LENGTH WAS 1.68 MILES. PATH WIDTH WAS 25 YARDS. (LSX)
    # 2029,UNK,2 WNW DETROIT,PIKE,IL,39.63,-90.71,NWS SURVEY CREW CONFIRMS AN EF-0 TORNADO. MAX WINDS WERE 85 MPH. PATH LENGTH WAS 0.11 MILES. PATH WIDTH WAS 25 YARDS. (LSX)
    # Time,Speed,Location,County,State,Lat,Lon,Comments
    # 1528,59,17 S SAINT GEORGE ISLAN,GMZ755,FL,29.41,-84.86,GUST TO 59 MPH AT THE C TOWER AT 20 METER ELEVATION. (TAE)
    # 2017,UNK,3 N UNIONTOWN,PERRY,AL,32.5,-87.5,TREES DOWN NEAR THE INTERSECTION OF HWY 183 AND HEMLOCK RD. POSSIBLE TORNADO. TIME ESTIMATED FROM RADAR. (BMX)
    # Time,Size,Location,County,State,Lat,Lon,Comments
    # 1237,150,WATSON,DESHA,AR,33.89,-91.26,(LZK)
    # 1315,100,CLARKSDALE,COAHOMA,MS,34.2,-90.58,SOCIAL MEDIA REPORT OF QUARTER SIZED HAIL. (MEG)

    day_reports_csv_str = `curl https://www.spc.noaa.gov/climo/reports/#{yymmdd}_rpts_filtered.csv`
    day_reports_csv_str.gsub!('"', 'inch') # SPC CSV doesn't quote fields so " for "inch" messes us up.

    day_reports_tornado_csv_str, day_reports_wind_csv_str, day_reports_hail_csv_str = day_reports_csv_str.split(/^(?=Time)/)

    tornado_rows = CSV.parse(day_reports_tornado_csv_str, headers: true).each.to_a # as list of rows
    wind_rows    = CSV.parse(day_reports_wind_csv_str, headers: true).each.to_a # as list of rows
    hail_rows    = CSV.parse(day_reports_hail_csv_str, headers: true).each.to_a # as list of rows

    tornado_rows.select! { |row| valid_lat_lon?(row) }
    tornado_rows.map! do |row|
      time = spc_storm_report_row_to_time(date, row)

      begin_end_time_cells(time, time) +
      [
        "-1" # f_scale
      ] +
      row_to_lat_lon_cells(row)
    end

    # Wind rows are allowed to have bad geocodes.
    wind_rows.map! do |row|
      time = spc_storm_report_row_to_time(date, row)

      begin_end_time_cells(time, time) +
      [
        "Thunderstorm Wind", # kind
        (row["Speed"] == "UNK" ? "-1" : "%0.1f" % (row["Speed"].to_f * 0.868976)), # speed, convert to knots
        "gust", # speed_type
        (row["Comments"] =~ /\b(AWOS|ASOS|MESONET|\w*(weather|wx)\w* station|recorded|measured|anemometer)\b/i ? "measured" : "estimated"), # source
      ] +
      row_to_lat_lon_cells(row)
    end

    hail_rows.select! { |row| valid_lat_lon?(row) }
    hail_rows.map! do |row|
      time = spc_storm_report_row_to_time(date, row)

      begin_end_time_cells(time, time) +
      [
        "Hail", # kind
        (row["Size"] == "UNK" ? "-1" : row["Size"].to_i / 100.0), # inches
      ] +
      row_to_lat_lon_cells(row)
    end

    tornado_rows.map(&:to_csv).sort.each do |row_csv_str|
      tornadoes_file.print row_csv_str
    end

    wind_rows.map(&:to_csv).sort.each do |row_csv_str|
      wind_events_file.print row_csv_str
    end

    hail_rows.map(&:to_csv).sort.each do |row_csv_str|
      hail_events_file.print row_csv_str
    end

  rescue CSV::MalformedCSVError => e
    STDERR.puts e
    STDERR.puts e.message
    STDERR.puts day_reports_csv_str
  end
end

tornadoes_file.close
wind_events_file.close
hail_events_file.close
