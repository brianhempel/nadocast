require "csv"

# Fetches the storm events database from 2014 through the current year
# and outputs all the tornado events.
#
# Uses the storm events database because it has start and end times and locations.
#
# The previous year is finalized near the end of Spring of the following year.
#
# To run this script, see the Makefile at the project root.

ROOT_URL = "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

file_names = `curl #{ROOT_URL}`.scan(/StormEvents_details-ftp_v1\.0_d\d\d\d\d_c\d+\.csv\.gz/).uniq



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

# print %w[
#   begin_time_str
#   begin_time_seconds
#   BEGIN_DATE_TIME
#   end_time_str
#   end_time_seconds
#   END_DATE_TIME
#   CZ_TIMEZONE
#   f_scale
#   length
#   width
#   begin_lat
#   begin_lon
#   end_lat
#   end_lon
# ].to_csv


(2014..Time.now.year).each do |year|
  file_name = file_names.grep(/v1\.0_d#{year}_/).last

  next unless file_name

  rows = CSV.parse(`curl #{ROOT_URL + file_name} | gunzip`, headers: true)

  # STDERR.puts "Event types: #{rows.map { |row| row["EVENT_TYPE"] }.uniq.sort}"

  tornado_rows = rows.select { |row| row["EVENT_TYPE"].strip == "Tornado" }

  STDERR.puts "#{tornado_rows.count} tornado path pieces in #{year}"

  tornado_rows.map! do |row|
    begin_year_month_str = row["BEGIN_YEARMONTH"]
    begin_day_str        = row["BEGIN_DAY"]
    begin_time_str       = row["BEGIN_TIME"]
    end_year_month_str   = row["END_YEARMONTH"]
    end_day_str          = row["END_DAY"]
    end_time_str         = row["END_TIME"]
    tz_offset_hrs        = row["CZ_TIMEZONE"][/-?\d+/].to_i

    begin_time = Time.new(begin_year_month_str.to_i / 100, begin_year_month_str.to_i % 100, begin_day_str.to_i, begin_time_str.to_i / 100, begin_time_str.to_i % 100, 00, "%+03d:00" % tz_offset_hrs)
    end_time   = Time.new(  end_year_month_str.to_i / 100,   end_year_month_str.to_i % 100,   end_day_str.to_i,   end_time_str.to_i / 100,   end_time_str.to_i % 100, 00, "%+03d:00" % tz_offset_hrs)

    # TOR_F_SCALE  TOR_LENGTH  TOR_WIDTH
    #
    # BEGIN_LAT  BEGIN_LON  END_LAT  END_LON

    [
      begin_time.utc.to_s,
      begin_time.utc.to_i,
      # row["BEGIN_DATE_TIME"],
      end_time.utc.to_s,
      end_time.utc.to_i,
      # row["END_DATE_TIME"],
      # row["CZ_TIMEZONE"],
      row["TOR_F_SCALE"][/\d+/],
      # row["TOR_LENGTH"].to_f,
      # row["TOR_WIDTH"].to_i,
      row["BEGIN_LAT"],
      row["BEGIN_LON"],
      row["END_LAT"],
      row["END_LON"]
    ]
  end

  tornado_rows.map(&:to_csv).sort.each do |row_csv_str|
    print row_csv_str
  end
end
