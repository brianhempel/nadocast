# ruby process_asos6405.rb *.dat > gusts.csv



# WBAN#        local       utc                                           2min avg  3s gust
#              YYYYMMDDhhmmhhmm                                          dir knts  dir knts
# 13966KSPS SPS2021120100280628    0.066  N                              216    9  217    9

require "csv"
require "set"

# paths = Dir.glob("**/*.dat")
paths = ARGV

verbose = ENV["VERBOSE"] == "true"

good_row_counts_path = ENV["GOOD_ROW_COUNTS_PATH"] || "good_row_counts.csv"

MINUTE = 60
HOUR   = 60*MINUTE
DAY    = 24*HOUR

FEET_PER_METER  = 100.0 / 2.54 / 12.0
METERS_PER_MILE = 5280.0 / FEET_PER_METER

# FCC method, per Wikipedia https://en.wikipedia.org/wiki/Geographical_distance#Ellipsoidal_Earth_projected_to_a_plane
# Surprisingly good! Generally much less than 0.01% error over short distances, and not completely awful over long distances.
# Precondition: longitudes don't cross over (raw lon2-lon1 < 180)
def instantish_distance_miles(lat1, lon1, lat2, lon2)
  mean_lat = (lat1 + lat2) / 2.0 / 180.0 * Math::PI
  dlat     = lat2 - lat1
  dlon     = lon2 - lon1

  k1 = 111.13209 - 0.56605*Math.cos(2*mean_lat) + 0.00120*Math.cos(4*mean_lat)
  k2 = 111.41513*Math.cos(mean_lat) - 0.09455*Math.cos(3*mean_lat) + 0.00012*Math.cos(5*mean_lat)

  Math.sqrt((k1*dlat)**2.0 + (k2*dlon)**2.0) * 1000.0 / METERS_PER_MILE
end

class Array
  def mean
    sum / size.to_f
  end
end

class String
  LEADING_ZERO = /\A0/
  def trim_leading_zero
    sub(LEADING_ZERO, "")
  end
end

class Time
  def to_convective_day_i
    (utc.to_i - 12*HOUR) / DAY
  end

  def self.from_convective_day_i(day_i)
    Time.at(day_i * DAY + 12*HOUR).utc
  end
end

# using isd-history.csv:
# 751 stations
# 212 stations with gusts

# stations_path = File.expand_path("../isd-history.csv", __FILE__)
# stations = CSV.read(stations_path, headers: true).select { |row| row["WBAN"] && row["WBAN"] != "" && row["WBAN"] != "99999" }
# STDERR.puts stations.size
# <CSV::Row "USAF":"007026" "WBAN":"99999" "STATION NAME":"WXPOD 7026" "CTRY":"AF" "STATE":"" "ICAO":"" "LAT":"+00.000" "LON":"+000.000" "ELEV(M)":"+7026.0" "BEGIN":"20120713" "END":"20170822">
# wban_to_stations = stations.group_by { |row| row["WBAN"] }.to_h

# using mshr_enhanced.csv:
# 752 stations
# 212 stations with gusts

stations2_path = File.expand_path("../mshr_enhanced_wban_only.csv", __FILE__)
stations2 = CSV.read(stations2_path, headers: true).select { |row| row["WBAN_ID"] != "" && (row["FIPS_COUNTRY_CODE"] == "US" || row["FIPS_COUNTRY_CODE"] == "GQ" || row["FIPS_COUNTRY_CODE"] == "CQ") } # Guam and Mariana Islands have ASOS stations
# STDERR.puts stations2.size
# #<CSV::Row "SOURCE_ID":"AK-AB-1" "SOURCE":"COCORAHS" "BEGIN_DATE":"20070407" "END_DATE":"20100422" "STATION_STATUS":"CLOSED" "NCDCSTN_ID":"30039214" "ICAO_ID":"" "WBAN_ID":"" "FAA_ID":"" "NWSLI_ID":"" "WMO_ID":"" "COOP_ID":"" "TRANSMITTAL_ID":"AK-AB-1" "GHCND_ID":"US1AKAB0001" "NAME_PRINCIPAL":"ANCHORAGE 4.8 E" "NAME_PRINCIPAL_SHORT":"ANCHORAGE 4.8 E" "NAME_COOP":"" "NAME_COOP_SHORT":"" "NAME_PUBLICATION":"" "NAME_ALIAS":"" "NWS_CLIM_DIV":"" "NWS_CLIM_DIV_NAME":"" "STATE_PROV":"AK" "COUNTY":"ANCHORAGE BOROUGH" "NWS_ST_CODE":"50" "FIPS_COUNTRY_CODE":"US" "FIPS_COUNTRY_NAME":"UNITED STATES" "NWS_REGION":"" "NWS_WFO":"" "ELEV_GROUND":"230" "ELEV_GROUND_UNIT":"FEET" "ELEV_BAROM":"" "ELEV_BAROM_UNIT":"" "ELEV_AIR":"" "ELEV_AIR_UNIT":"" "ELEV_ZERODAT":"" "ELEV_ZERODAT_UNIT":"" "ELEV_UNK":"" "ELEV_UNK_UNIT":"" "LAT_DEC":"61.20468" "LON_DEC":"-149.75633" "LAT_LON_PRECISION":"" "RELOCATION":"" "UTC_OFFSET":"" "OBS_ENV":"LANDSFC" "PLATFORM":"COCORAHS" "GHCNMLT_ID":"" "COUNTY_FIPS_CODE":"02020" "DATUM_HORIZONTAL":"" "DATUM_VERTICAL":"" "LAT_LON_SOURCE":"" "IGRA_ID":"" "HPD_ID":"">
wban_to_stations2 = stations2.group_by { |row| row["WBAN_ID"] }.to_h

# regex = /^\d{5}\w{4} \w{3}(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/
# wind_stuff   = /\s(\d+)\s+(\d+)\s+(\d+)\s+(\d+)(?!.*\s\d+\s)\s.*\n\z/ # Should be the last four numbers in the row.
# five_numbers = /\s(?:(?:\d+)\s+){5}/

paths.each do |path|
  last_wban       = nil
  valid_stations2 = []

  entries_by_wban = {}
  wban_rows       = []

  final_rows_for_path = []

  File.read(path).lines.each do |line|
    #                     id  YYYYMM
    # asos/6405-2021/64050KMFI202112.dat

    # Do NOT try to recover garbled lines.
    if line.size != 113
      STDERR.puts "Bad line in #{path}: #{line.inspect}" if verbose
      next
    end

    # I think brackets in the line mean the station is in maintainence mode
    if line[66] == "[" || line[89] == "]"
      # STDERR.puts "Maintainence mode (maybe?) line in #{path}: #{line.inspect}" if verbose
      next
    end

    begin
      wban       = line[0...5]
      yyyymmdd   = line[13...21]
      year       = Integer(yyyymmdd[0...4])
      month      = Integer(yyyymmdd[4...6].trim_leading_zero)
      day        = Integer(yyyymmdd[6...8].trim_leading_zero)
      hr         = Integer(line[21...23].trim_leading_zero)
      min        = Integer(line[23...25].trim_leading_zero)
      utc_hr     = Integer(line[25...27].trim_leading_zero)
      utc_min    = Integer(line[27...29].trim_leading_zero)

      # deg        = Integer(line[69...74].strip)
      knots      = Integer(line[74...79].strip)
      # gust_deg   = Integer(line[79...84].strip)
      gust_knots = Integer(line[84...89].strip)
    rescue
      STDERR.puts "Bad line in #{path}: #{line.inspect}" if verbose
      next
      # # Try to recover bad lines: look for four consecutive integers
      # begin
      #   line = line.gsub("["," ").gsub("]"," ") # sometimes there are strange brackets
      #   raise "bad line" if line =~ five_numbers

      #   wban       = line[0...5]
      #   yyyymmdd   = line[13...21]
      #   year       = Integer(yyyymmdd[0...4])
      #   month      = Integer(yyyymmdd[4...6])
      #   day        = Integer(yyyymmdd[6...8])
      #   hr         = Integer(line[21...23])
      #   min        = Integer(line[23...25])
      #   utc_hr     = Integer(line[25...27])
      #   utc_min    = Integer(line[27...29])

      #   deg, knots, gust_deg, gust_knots = wind_stuff.match(line).captures.map { |int_str| Integer(int_str.strip) }
      #   raise "bad line" if !(0..360).include?(deg) || !(0..360).include?(gust_deg)
      # rescue
      #   STDERR.puts "Bad line in #{path}: #{line.inspect}"
      #   next
      # end
    end


    if wban != last_wban
      entries_by_wban[wban] ||= []
      wban_rows = entries_by_wban[wban]
      last_wban = wban
    end

    utc_offset = hr - utc_hr
    utc_offset += 24 if utc_offset <= -12
    utc_offset -= 24 if utc_offset > 12

    if !(-9..-3).include?(utc_offset)
      next
    end

    if !(2000..2099).include?(year) || !(1..12).include?(month) || !(1..31).include?(day) || !(00..23).include?(hr) || !(00..23).include?(utc_hr) || !(00..59).include?(min) || min != utc_min
      STDERR.puts "Bad line in #{path}: #{line.inspect}" if verbose
      next
    end

    time = Time.new(year, month, day, hr, min, 0, "%+03d:00" % utc_offset).utc

    wban_rows << [
      yyyymmdd,
      time.utc,
      knots,
      gust_knots,
    ]
  end

  entries_by_wban.each do |wban, rows|
    wban_stations = wban_to_stations2[wban]
    if !wban_stations
      STDERR.puts "No station records with WBAN #{wban}"
      STDERR.puts path
      next
    end

    # Per Cook 2022, there are duplicate records. We'll discard if they disagree.
    rows_deduped =
      rows
        .uniq
        .group_by { |row| row[1] } # UTC time
        .values
        .select { |rows_for_same_time| rows_for_same_time.size == 1 }
        .map    { |rows_for_same_time| rows_for_same_time[0] }
        .sort


    # There seems to be a calibration procedure that involves running the aneomometer up to 51knots and to 76knots
    #
    # Discard rows ±1hr of such an event.
    #
    # Detect such an event in ANY of the following cases.:
    #   Case 1. A 76gust AND within 10min there is also either (two additional 76gusts OR another 76gust and a 51gust OR two 51gusts OR a single 51avg/51gust record)
    #   Case 2. A 76avg/76gust record.
    #   Case 3. A 51avg/51gust record AND two additional 51gusts within 10min.

    rows1            = []
    times_to_discard = []
    rows_deduped.each_with_index do |row, i|
      t          = row[1]
      knots      = row[2]
      gust_knots = row[3]

      if knots == 76 && gust_knots == 76 # Case 2
        times_to_discard << t
      elsif gust_knots == 76 # Case 1
        rows_within_10_min =
          (i-10..i+10)
            .select { |j| j > 0 && j < rows_deduped.size }
            .select { |j| (rows_deduped[j][1] - t).abs <= 60*10 }
            .map    { |j| rows_deduped[j] }

        if rows_within_10_min.any? { |r| r[2] == 51 && r[3] == 51 } || rows_within_10_min.count { |r| r[3] == 51 } >= 2 || rows_within_10_min.count { |r| r[3] == 76 } >= 3 || (rows_within_10_min.count { |r| r[3] == 76 } >= 2 && rows_within_10_min.any? { |r| r[3] == 51 })
          times_to_discard << t
        end
      elsif knots == 51 && gust_knots == 51 # Case 3
        rows_within_10_min =
          (i-10..i+10)
            .select { |j| j > 0 && j < rows_deduped.size }
            .select { |j| (rows_deduped[j][1] - t).abs <= 60*10 }
            .map    { |j| rows_deduped[j] }

        if rows_within_10_min.count { |r| r[3] == 51 } >= 3
          times_to_discard << t
        end
      end
    end

    if times_to_discard == []
      rows1 = rows_deduped
    else
      rows_deduped.each do |row|
        t = row[1]
        unless times_to_discard.any? { |bad_t| t >= bad_t - HOUR && t <= bad_t + HOUR }
          rows1 << row
        end
      end
    end

    # Cook 2022 bird-generated gust removal algorithm
    # Gusts next to a data gap are usually erroneous, so do the following twice:
    #   Compute the 10-min average of gusts, centered on the time of interest, pretending gaps are 0-min long
    #   Remove any record next to a data gap if the gust is >1.5x the running average gust

    # Also, I don't care about gusts <=10knots

    rows2 = []
    rows1.each_with_index do |row, i|
      t = row[1]
      # actually 11-min average
      rows_for_averaging =
        (i-5..i+5)
          .select { |j| j > 0 && j < rows1.size }
          .select { |j| (rows1[j][1] - t).abs <= 60*15 } # don't include rows more than 15min away (i.e. on the other side of loooong gaps)
          .map    { |j| rows1[j] }

      # If next to a gap...
      if i == 0 || i == rows1.size-1 || rows1[i-1][1] + 60 != t || rows1[i+1][1] - 60 != t
        # Only keep row if <=1.5x the running average gust
        eleven_min_average_gust = rows_for_averaging.map { |r| r[3].to_f }.mean
        if row[3] <= 10 || row[3] <= 1.5*eleven_min_average_gust
          rows2 << row
        else
          STDERR.puts "Discarding bird-generated gust row #{row.to_csv.chomp} because:" if verbose
          rows_for_averaging.each { |r| STDERR.puts r.to_csv } if verbose
        end
      else
        rows2 << row
      end
    end

    # Do it again.
    rows3 = []
    rows2.each_with_index do |row, i|
      t = row[1]

      # actually 11-min average
      rows_for_averaging =
        (i-5..i+5)
          .select { |j| j > 0 && j < rows2.size }
          .select { |j| (rows2[j][1] - t).abs <= 60*15 } # don't include rows more than 15min away (i.e. on the other side of loooong gaps)
          .map    { |j| rows2[j] }

      # If next to a gap...
      if i == 0 || i == rows2.size-1 || rows2[i-1][1] + 60 != t || rows2[i+1][1] - 60 != t
        # Only keep row if <=1.5x the running average gust
        eleven_min_average_gust = rows_for_averaging.map { |r| r[3].to_f }.mean
        if row[3] <= 10 || row[3] <= 1.5*eleven_min_average_gust
          rows3 << row
        else
          STDERR.puts "Discarding bird-generated gust row #{row.to_csv.chomp} because:" if verbose
          rows_for_averaging.each { |r| STDERR.puts r.to_csv } if verbose
        end
      else
        rows3 << row
      end
    end

    # Discard rows with few neighbors. (Less than 3 other nearby rows available for averaging.)
    rows4 = []
    rows3.each_with_index do |row, i|
      t = row[1]

      # actually 11-min average
      rows_for_averaging =
        (i-5..i+5)
          .select { |j| j > 0 && j < rows3.size }
          .select { |j| (rows3[j][1] - t).abs <= 60*15 } # don't include rows more than 15min away (i.e. on the other side of loooong gaps)
          .map    { |j| rows3[j] }

      if rows_for_averaging.size >= 4
        rows4 << row
      else
        STDERR.puts "Discarding lonely row #{row.to_csv.chomp} because:" if verbose
        rows_for_averaging.each { |r| STDERR.puts r.to_csv } if verbose
      end
    end


    # Cook 2014 spurious gust removal algorithm
    # Remove single/double/triple spikes more than 30knots different from BOTH before and after
    # (Does not apply next to data gaps.)

    # Single
    rows5 = []
    rows4.each_with_index do |row, i|
      t = row[1]
      if i > 0 && i < rows4.size-1 && rows4[i-1][1] + 60 == t && rows4[i+1][1] - 60 == t
        if rows4[i][3] - rows4[i-1][3] > 30 && rows4[i][3] - rows4[i+1][3] > 30
          STDERR.puts "Discarding single spike:" if verbose
          STDERR.puts rows4[i-1].to_csv if verbose
          STDERR.puts rows4[i].to_csv if verbose
          STDERR.puts rows4[i+1].to_csv if verbose
          next
        end
      end
      rows5 << row
    end

    # Double
    rows6 = []
    n_to_skip = 0
    rows5.each_with_index do |row, i|
      t = row[1]
      if n_to_skip == 0 && i > 0 && i < rows5.size-2 && rows5[i-1][1] + 60 == t && rows5[i+1][1] - 60 == t && rows5[i+2][1] - 120 == t
        if rows5[i][3] - rows5[i-1][3] > 30 && rows5[i+1][3] - rows5[i+2][3] > 30
          n_to_skip = 2
          STDERR.puts "Discarding double spike:" if verbose
          STDERR.puts rows5[i-1].to_csv if verbose
          STDERR.puts rows5[i].to_csv if verbose
          STDERR.puts rows5[i+1].to_csv if verbose
          STDERR.puts rows5[i+2].to_csv if verbose
        end
      end
      if n_to_skip > 0
        n_to_skip -= 1
      else
        rows6 << row
      end
    end

    # Triple
    rows7 = []
    n_to_skip = 0
    rows6.each_with_index do |row, i|
      t = row[1]
      if n_to_skip == 0 && i > 0 && i < rows6.size-3 && rows6[i-1][1] + 60 == t && rows6[i+1][1] - 60 == t && rows6[i+2][1] - 120 == t && rows6[i+3][1] - 180 == t
        if rows6[i][3] - rows6[i-1][3] > 30 && rows6[i+2][3] - rows6[i+3][3] > 30
          n_to_skip = 3
          STDERR.puts "Discarding triple spike:" if verbose
          STDERR.puts rows6[i-1].to_csv if verbose
          STDERR.puts rows6[i].to_csv if verbose
          STDERR.puts rows6[i+1].to_csv if verbose
          STDERR.puts rows6[i+2].to_csv if verbose
          STDERR.puts rows6[i+3].to_csv if verbose
        end
      end
      if n_to_skip > 0
        n_to_skip -= 1
      else
        rows7 << row
      end
    end

    # Apply ASOS sonic aneomometer QC test 10, 2018+ version, b/c it was part of a software update during the period
    # Remove records with 2min avg <=6knots && 3sec gust >13knots && gust factor >2.5
    #
    # And do a sanity check that gust > avg and gust not >125knots max rated measurement
    rows8 = rows7.reject do |row|
      knots      = row[2]
      gust_knots = row[3]
      if knots <= 6 && gust_knots > 13 && gust_knots.to_f / knots > 2.5
        STDERR.puts "Discarding #{row.to_csv} for test 10 failure" if verbose
        true
      elsif knots > gust_knots + 1 || gust_knots > 125 # Sometimes the gust is 1knot below the sustained...probably a quirk of the averaging algs. Also, ASOS only rated up to 125knots
        STDERR.puts "Bad line in #{path}: #{line.inspect}" if verbose
        true
      else
        false
      end
    end

    last_yyyymmdd = nil
    lat_str = nil
    lon_str = nil
    valid_stations = []
    rows8.each do |yyyymmdd, utc_time, knots, gust_knots|
      if yyyymmdd != last_yyyymmdd
        lat_str = nil
        lon_str = nil
        last_yyyymmdd = yyyymmdd

        valid_stations = wban_stations.select { |row| yyyymmdd >= row["BEGIN_DATE"] && yyyymmdd <= row["END_DATE"] }

        if valid_stations.size == 0
          STDERR.puts "No station locations for WBAN #{wban} on #{yyyymmdd}"
          STDERR.puts path
          next
        end

        # Disambiguate upper air and radar stations.
        if valid_stations.size >= 2 && valid_stations.count { |station| station["OBS_ENV"].include?("LANDSFC") } >= 1
          valid_stations.select! { |station| station["OBS_ENV"].include?("LANDSFC") }
        end
        if valid_stations.size >= 2 && valid_stations.count { |station| station["PLATFORM"].include?("ASOS") } >= 1
          valid_stations.select! { |station| station["PLATFORM"].include?("ASOS") }
        end

        lat = valid_stations.map { |station| station["LAT_DEC"].to_f }.mean
        lon = valid_stations.map { |station| station["LON_DEC"].to_f }.mean

        # Ensure all stations are within 1 mile of centroid
        stations_good = valid_stations.all? { |station| instantish_distance_miles(station["LAT_DEC"].to_f, station["LON_DEC"].to_f, lat, lon) <= 1.0 }

        if !stations_good
          STDERR.puts "Conflicting station locations for WBAN #{wban} on #{yyyymmdd}:"
          STDERR.puts path
          # STDERR.puts valid_stations.map(&:to_csv).join("\n")
          STDERR.puts valid_stations.map(&:to_csv).join("\n")
          next
        end

        lat_str = ("%.4f" % lat).sub(/0+$/,"")
        lon_str = ("%.4f" % lon).sub(/0+$/,"")
      end

      if lat_str
        final_rows_for_path << [
          utc_time.to_s,
          utc_time.to_i,
          wban,
          valid_stations.first["NAME_PRINCIPAL"], # CHICAGO OHARE INTL AP
          valid_stations.first["STATE_PROV"], # IL
          valid_stations.first["COUNTY"], # COOK
          knots,
          gust_knots,
          lat_str,
          lon_str
        ]
      end
    end
  end

  open(good_row_counts_path, "a") do |good_rows_csv|
    final_rows_for_path
      .group_by { |r| [r[2], r[8], r[9]] } # WBAN, lat_str, lon_str
      .each do |(wban, lat_str, lon_str), location_rows|
        location_rows
          .map { |r| (r[1] - 12*HOUR) / DAY } # convective_day_i
          .tally
          .each do |day_i, good_rows_count|
            good_rows_csv.puts [
              Time.from_convective_day_i(day_i).strftime("%F"),
              day_i,
              wban,
              lat_str,
              lon_str,
              good_rows_count
            ].to_csv
          end
      end
  end

  final_rows_for_path.each do |row|
    knots      = row[6]
    gust_knots = row[7]
    if knots >= 40 || gust_knots >= 40
      puts row.to_csv
    end
  end

end
