# ruby process_asos6405.rb *.dat > gusts.csv



# WBAN#        local       utc                                           2min avg  3s gust
#              YYYYMMDDhhmmhhmm                                          dir knts  dir knts
# 13966KSPS SPS2021120100280628    0.066  N                              216    9  217    9

require "csv"
require "set"

# paths = Dir.glob("**/*.dat")
paths = ARGV

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

wban_set = Set.new
wban_gust_set = Set.new

regex = /^\d{5}\w{4} \w{3}(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/

paths.each do |path|
  last_wban       = nil
  valid_stations2 = []
  File.read(path).lines.each do |line|
    #                     id  YYYYMM
    # asos/6405-2021/64050KMFI202112.dat

    begin
      wban       = line[0...5]
      yyyymmdd   = line[13...21]
      knots      = line[74...79].strip.to_i # Missing will be encoded as 0
      gust_knots = line[84...89].strip.to_i # Missing will be encoded as 0
    rescue
      STDERR.puts "Bad line in #{path}: #{line.inspect}"
      next
    end

    if wban != last_wban
      begin
        # valid_stations = wban_to_stations[wban].select { |row| yyyymmdd >= row["BEGIN"] && yyyymmdd <= row["END"] }
        valid_stations2 = wban_to_stations2[wban].select { |row| yyyymmdd >= row["BEGIN_DATE"] && yyyymmdd <= row["END_DATE"] }
        wban_set << wban
        last_wban = wban
      rescue
        STDERR.puts "No station records with WBAN #{wban}"
        STDERR.puts path
        STDERR.puts line
        break
      end
    end

    # next unless valid_stations.size >= 1
    next unless valid_stations2.size >= 1

    if knots >= 50 || gust_knots >= 50

      year, month, day, hr, min, utc_hr, utc_min =
        begin
          regex.match(line).captures.map(&:to_i)
        rescue
          STDERR.puts "Bad line in #{path}: #{line.inspect}"
          next
        end

      utc_offset = hr - utc_hr
      utc_offset += 24 if utc_offset <= -12
      utc_offset -= 24 if utc_offset > 12

      time = Time.new(year, month, day, hr, min, 0, "%+03d:00" % utc_offset).utc

      # If multiple station entries, take centroid
      # lat = valid_stations.map { |station| station["LAT"].to_f }.mean
      # lon = valid_stations.map { |station| station["LON"].to_f }.mean
      lat = valid_stations2.map { |station| station["LAT_DEC"].to_f }.mean
      lon = valid_stations2.map { |station| station["LON_DEC"].to_f }.mean

      # Ensure all entires are within 1 mile of centroid
      # stations_good = valid_stations.all? { |station| instantish_distance_miles(station["LAT"].to_f, station["LON"].to_f, lat, lon) <= 1.0 }
      stations_good = valid_stations2.all? { |station| instantish_distance_miles(station["LAT_DEC"].to_f, station["LON_DEC"].to_f, lat, lon) <= 1.0 }

      if !stations_good
        STDERR.puts "Conflicting station locations for WBAN #{wban}:"
        STDERR.puts path
        STDERR.puts line
        # STDERR.puts valid_stations.map(&:to_csv).join("\n")
        STDERR.puts valid_stations2.map(&:to_csv).join("\n")
        break
      end

      wban_gust_set << wban

      puts [
        time.utc.to_s,
        time.utc.to_i,
        wban,
        valid_stations2.first["NAME_PRINCIPAL"], # CHICAGO OHARE INTL AP
        valid_stations2.first["STATE_PROV"], # IL
        valid_stations2.first["COUNTY"], # COOK
        knots,
        gust_knots,
        ("%.4f" % lat).sub(/0+$/,""),
        ("%.4f" % lon).sub(/0+$/,"")
      ].to_csv
    end
  end
end

STDERR.puts "#{wban_set.size} stations"
STDERR.puts "#{wban_gust_set.size} stations with gusts"
