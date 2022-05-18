# ruby mshr_enhanced_to_csv.rb > mshr_enhanced.csv

require "csv"


# Layout: https://www.ncei.noaa.gov/access/homr/file/MSHR_Enhanced_Table.txt

# MSHR_ENHANCED.TXT

# The Enhanced Master Station History Report is a list of basic, historical information for every station in the station history
# database.  The Enhanced MSHR version contains similar data elements and is in a similar format as the Standard/Legacy MSHR, but
# the Enhanced MSHR includes additional stations that are included in the GHCN-Daily dataset. Also, the Enhanced MSHR will
# incorporate additional station networks, including international stations/networks in the future.  The Enhanced MSHR additionally
# contains an expanded number of items such as station status, additional IDs, alternate station names, NWS region, NWS WFO, all elevations,
# latitude/longitude precision and source, and UTC offset.  Data elements are in user-friendly formats.

# The file is sorted by source, source_id, begin_date, end_date.

# This report is scheduled to be updated monthly.

# *** UPDATE, beginning with August 2014 report: New GHCNM-LT source, new columns appended: GHCNMLT_ID, COUNTY_FIPS_CODE, DATUM_HORIZONTAL,
#     DATUM_VERTICAL, LAT_LON_SOURCE.

# *** UPDATE, beginning with June 2016 report: New IGRA2 source, new columns appended: IGRA_ID, HPD_ID.

# *** UPDATE, June 2021: Documentation only, updated column descriptions and provided platform reference table (B).

# FIELD            LENGTH   FORMAT       POSITION      DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------------------------
# SOURCE_ID            20   X(20)        001-020       Unique identifier from source system, should be used in union with SOURCE.

# SOURCE               10   X(10)        022-031       Name of source NCEI system (MMS, ISIS, GHCND, GHCNM-LT, IGRA2 etc). These are used
#                                                      mostly by NCEI to determine the source system where a correction can be made if needed.
#                                                      For example, MMS was the precursor to HOMR and is a legacy reference, ISIS is a standalone
#                                                      system maintaining USCRN metadata, etc.

# BEGIN_DATE           8    YYYYMMDD     033-040       Beginning date of record, set to 00010101 if date is unknown.

# END_DATE             8    YYYYMMDD     042-049       Ending date of record, set to 99991231 if station is currently open.

# STATION_STATUS       20   X(20)        051-070       For the Cooperative Network, displays INACTIVE if station is currently
#                                                      inactive and not closed. END_DATE would be effective date of inactivation.
#                                                      For USCRN / USRCRN / AL USRCRN stations, the status will display either
#                                                      OPERATIONAL, NON-OPERATIONAL, CLOSED or ABANDONED.

# NCDCSTN_ID           20   X(20)        072-091       Unique identifier used by NCEI, primary key for HOMR station.

# ICAO_ID              20   X(20)        093-112       Used for geographical locations throughout the world, managed by
#                                                      the International Civil Aviation Organization.

# WBAN_ID              20   X(20)        114-133       Assigned by NCEI, used for digital data storage and general station
#                                                      identification purposes.

# FAA_ID               20   X(20)        135-154       Alpha-numeric, managed by USDT Federal Aviation Administration used
#                                                      for site identification of sites vital to navigation. Commonly
#                                                      referred to as "Call Sign".

# NWSLI_ID             20   X(20)        156-175       Alpha-numeric, location identifier assigned by the National Weather
#                                                      Service for use in real-time data transmissions and forecasts.

# WMO_ID               20   X(20)        177-196       Assigned by World Meteorological Organization, used for
#                                                      international weather data exchange and station documentation.

# COOP_ID              20   X(20)        198-217       Assigned by NCEI, first 2 digits represent state, last 4 digits
#                                                      are assigned numerically by alphabetical ordering of the station name.

# TRANSMITTAL_ID       20   X(20)        219-238       Holds miscellaneous IDs that do not fall into an officially sourced
#                                                      ID category that are needed in support of NCEI data datasets and ingests.

# GHCND_ID             20   X(20)        240-259       Populated if station is included in GHCN-Daily dataset.

# NAME_PRINCIPAL       100  X(100)       261-360       Name of station, upper case may contain characters, numbers or symbols.

# NAME_PRINCIPAL_SHORT 30   X(30)        362-391       Name of station, upper case may contain characters, numbers or symbols.

# NAME_COOP            100  X(100)       393-492       Coop station name as maintained by NWS Cooperative network on WS Form
#                                                      B-44, upper case may contain characters, numbers or symbols.

# NAME_COOP_SHORT      30   X(30)        494-523       Abbreviated Coop station name, limited to 30 characters, used by the
#                                                      NCEI Climatological Data publication.

# NAME_PUBLICATION     100  X(100)       525-624       Populated with station names from legacy IPB.STATION table (LCD, QCLCD,
#                                                      ISD, etc).

# NAME_ALIAS           100  X(100)       626-725       Any station alias known.

# NWS_CLIM_DIV         10   X(10)        727-736       Contains a number between 01 and 13 indicating climate division as
#                                                      determined by master divisional reference maps. Assigned by NCEI.

# NWS_CLIM_DIV_NAME    40   X(40)        738-777       Climate division name.

# STATE_PROV           10   X(10)        779-788       USPS two character alphabetic abbreviation for each state, uppercase.

# COUNTY               50   X(50)        790-839       Name of county, upper case.

# NWS_ST_CODE          2    X(2)         841-842       NWS state code.

# FIPS_COUNTRY_CODE    2    X(2)         844-845       FIPS country code.

# FIPS_COUNTRY_NAME    100  X(100)       847-946       FIPS country name.

# NWS_REGION           30   X(30)        948-977       NWS region (ALASKAN, CENTRAL, EASTERN, PACIFIC, SOUTHERN, WESTERN).

# NWS_WFO              10   X(10)        979-988       NWS weather forecast office.

# ELEV_GROUND          40   X(40)        990-1029      Ground elevation.  For Coop network, average elevation of the ground
#                                                      in a 20-meter(60ft) circle around the primary rain gauge. For 1st &
#                                                      2nd Order stations, elevation of the official temperature sensor for
#                                                      the station.

# ELEV_GROUND_UNIT     20   X(20)        1031-1050     Ground elevation unit (should always be FEET).

# ELEV_BAROM           40   X(40)        1052-1091     Barometric: Ivory point of the Barometer or Altimeter Setting.

# ELEV_BAROM_UNIT      20   X(20)        1093-1112     Barometric elevation unit (should always be FEET).

# ELEV_AIR             40   X(40)        1114-1153     Airport: Field, Aerodrome, or Runway elevation.

# ELEV_AIR_UNIT        20   X(20)        1155-1174     Airport elevation unit (should always be FEET).

# ELEV_ZERODAT         40   X(40)        1176-1215     Zero Datum of a River Gage.

# ELEV_ZERODAT_UNIT    20   X(20)        1217-1236     Zero datum elevation unit (should always be FEET).

# ELEV_UNK             40   X(40)        1238-1277     Elevation value, type unknown.

# ELEV_UNK_UNIT        20   X(20)        1279-1298     Unknown elevation unit (should always be FEET).

# LAT_DEC              20   X(20)        1300-1319     Decimal latitude, blank indicates North and "-" indicates South.

# LON_DEC              20   X(20)        1321-1340     Decimal longitude, blank indicates East and "-" indicates West.

# LAT_LON_PRECISION    10   X(10)        1342-1351     Indicates precision of source lat and lon, see Reference Table A) below.

# RELOCATION           62   X(62)        1353-1414     Distance and direction of station relocation expressed as a distance
#                                                      value (1-4 characters), space, distance units (2 character abbreviation),
#                                                      space, and direction (1-3 character 16-point cardinal direction). Date of
#                                                      relocation indicated by begin date of record.

# UTC_OFFSET           16   9(16)        1416-1431     Time zone, UTC offset.

# OBS_ENV              40   X(40)        1433-1472     Type of observing programs associated with the station (LANDSFC, RADAR,
#                                                      UPPERAIR, UNKNOWN), delimited by comma.

# PLATFORM             100  X(100)       1474-1573     Station type and/or platforms station participates in, delimited by comma.
#                                                      See Reference Table B) below.

# *** UPDATE, beginning with August 2014 report: New columns appended

# GHCNMLT_ID           20   X(20)        1575-1594     Populated if station is included in GHCN-Monthly Land Temperature dataset.

# COUNTY_FIPS_CODE     5    X(5)         1596-1600     FIPS county code.

# DATUM_HORIZONTAL     30   X(30)        1602-1631     Horizontal reference datum used to determine the spatial fix of the station.
#                                                      Only available in newer periods where GPS receivers were used.

# DATUM_VERTICAL       30   X(30)        1633-1662     Vertical reference datum used to determine the elevation of the station.
#                                                      Only available in newer periods where GPS receivers were used.

# LAT_LON_SOURCE       100  X(100)       1664-1763     Latitude/longitude data source.  Only available in newer periods where GPS
#                                                      receivers were used.

# *** UPDATE, beginning with June 2016 report: New columns appended

# IGRA_ID              20   X(20)        1765-1784     Populated if station is included in Integrated Global Radiosonde Archive (IGRA) dataset.

# HPD_ID               20   X(20)        1786-1805     Populated if station is included in Hourly Precipitation Data (HPD) dataset.
# ----------------------------------------------------------------------------------------------------------------------------------------
# Reference Tables

# A) LAT_LON_PRECISION CODES
# DD        Whole Degrees
# DDMM      Degrees, Whole Minutes
# DDMMSS    Degrees, Whole Minutes, Whole Seconds
# DDd       Decimal Degrees, to Tenths
# DDdd      Decimal Degrees, to Hundredths
# DDddd     Decimal Degrees, to Thousandths
# DDdddd    Decimal Degrees, to Ten Thousandths
# DDddddd   Decimal Degrees, to Hundred Thousandths
# DDMMm     Degrees, Decimal Minutes to Tenths
# DDMMmm    Degrees, Decimal Minutes to Hundredths
# DDMMmmm   Degrees, Decimal Minutes to Thousandths
# DDMMSSs   Degrees, Minutes, Decimal Seconds to Tenths
# DDMMSSss  Degrees, Minutes, Decimal Seconds to Hundredths
# DDMMSSss  Degrees, Minutes, Decimal Seconds to Hundredths

# B) PLATFORMS
# AIRSAMPLE       Air Sample Measurements of CO2, CH4, CO, N2O, H2, SF6 and isotopic ratios
# AIRWAYS         Surface Airways Observations
# AL USRCRN       Alabama U.S. Regional Climate Reference Network
# AMOS            AutoMated Observing Station
# ASOS            Automated Surface Observation System
# AWOS            Airway Weather Observation
# BALLOON         Radiosonde and pilot balloon soundings
# BASIC           Basic contract
# C-MAN           Coastal-Marine Automated Network
# CCOOP           Cellular Cooperative Observer Station
# CHARM           Cooperative Huntsville Area Rainfall Measurements
# COCORAHS        Community Collaborative Rain, Hail and Snow Network
# COOP            COOPerative station
# GSN             GCOS Surface Network
# HIDEN           Minnesota Volunteer Precipitation Observing Program
# MILITARY        Military Bases
# MSWS            Mountain States Weather Services
# NEPP            New England Pilot Project
# NERAIN          Nebraska Rainfall Assessment and Information Network
# NEXRAD          NEXT generation RADar
# NPN             NOAA Profiler Network
# ORC2C           Oregon Crest-to-Coast
# PLCD            Primary Local Climatological Data
# PRE-COOP        Pre-1900 Forts and Other Voluntary Observers
# RBCN            Regional Basic Climatological Network
# SURFRAD         Surface Radiation Budget
# SYNOPTIC        Synoptic reports (NWS)
# TDWR            Terminal Doppler Weather Radar
# UCN             Upper Colorado Network
# UNKNOWN         No platform could be determined
# UPPERAIR        Upper Air
# USCRN           U.S. Climate Reference Network
# USHCN           U.S. Historical Climatology Network
# USRCRN          U.S. Regional Climate Reference Network
# WXSVC           U.S. Weather Service%


# SOURCE_ID            20   X(20)        001-020       Unique identifier from source system, should be used in union with SOURCE.
# SOURCE               10   X(10)        022-031       Name of source NCEI system (MMS, ISIS, GHCND, GHCNM-LT, IGRA2 etc). These are used
# BEGIN_DATE           8    YYYYMMDD     033-040       Beginning date of record, set to 00010101 if date is unknown.
# END_DATE             8    YYYYMMDD     042-049       Ending date of record, set to 99991231 if station is currently open.
# STATION_STATUS       20   X(20)        051-070       For the Cooperative Network, displays INACTIVE if station is currently
#                                                      inactive and not closed. END_DATE would be effective date of inactivation.
#                                                      For USCRN / USRCRN / AL USRCRN stations, the status will display either
#                                                      OPERATIONAL, NON-OPERATIONAL, CLOSED or ABANDONED.

# NCDCSTN_ID           20   X(20)        072-091       Unique identifier used by NCEI, primary key for HOMR station.

# ICAO_ID              20   X(20)        093-112       Used for geographical locations throughout the world, managed by
#                                                      the International Civil Aviation Organization.

# WBAN_ID              20   X(20)        114-133       Assigned by NCEI, used for digital data storage and general station
#                                                      identification purposes.

# FAA_ID               20   X(20)        135-154       Alpha-numeric, managed by USDT Federal Aviation Administration used
#                                                      for site identification of sites vital to navigation. Commonly
#                                                      referred to as "Call Sign".

# NWSLI_ID             20   X(20)        156-175       Alpha-numeric, location identifier assigned by the National Weather
#                                                      Service for use in real-time data transmissions and forecasts.

# WMO_ID               20   X(20)        177-196       Assigned by World Meteorological Organization, used for
#                                                      international weather data exchange and station documentation.

# COOP_ID              20   X(20)        198-217       Assigned by NCEI, first 2 digits represent state, last 4 digits
#                                                      are assigned numerically by alphabetical ordering of the station name.

# TRANSMITTAL_ID       20   X(20)        219-238       Holds miscellaneous IDs that do not fall into an officially sourced
#                                                      ID category that are needed in support of NCEI data datasets and ingests.

# GHCND_ID             20   X(20)        240-259       Populated if station is included in GHCN-Daily dataset.

# NAME_PRINCIPAL       100  X(100)       261-360       Name of station, upper case may contain characters, numbers or symbols.

# NAME_PRINCIPAL_SHORT 30   X(30)        362-391       Name of station, upper case may contain characters, numbers or symbols.

# NAME_COOP            100  X(100)       393-492       Coop station name as maintained by NWS Cooperative network on WS Form
#                                                      B-44, upper case may contain characters, numbers or symbols.

# NAME_COOP_SHORT      30   X(30)        494-523       Abbreviated Coop station name, limited to 30 characters, used by the
#                                                      NCEI Climatological Data publication.

# NAME_PUBLICATION     100  X(100)       525-624       Populated with station names from legacy IPB.STATION table (LCD, QCLCD,
#                                                      ISD, etc).

# NAME_ALIAS           100  X(100)       626-725       Any station alias known.

# NWS_CLIM_DIV         10   X(10)        727-736       Contains a number between 01 and 13 indicating climate division as
#                                                      determined by master divisional reference maps. Assigned by NCEI.

# NWS_CLIM_DIV_NAME    40   X(40)        738-777       Climate division name.

# STATE_PROV           10   X(10)        779-788       USPS two character alphabetic abbreviation for each state, uppercase.

# COUNTY               50   X(50)        790-839       Name of county, upper case.

# NWS_ST_CODE          2    X(2)         841-842       NWS state code.

# FIPS_COUNTRY_CODE    2    X(2)         844-845       FIPS country code.

# FIPS_COUNTRY_NAME    100  X(100)       847-946       FIPS country name.

# NWS_REGION           30   X(30)        948-977       NWS region (ALASKAN, CENTRAL, EASTERN, PACIFIC, SOUTHERN, WESTERN).

# NWS_WFO              10   X(10)        979-988       NWS weather forecast office.

# ELEV_GROUND          40   X(40)        990-1029      Ground elevation.  For Coop network, average elevation of the ground
#                                                      in a 20-meter(60ft) circle around the primary rain gauge. For 1st &
#                                                      2nd Order stations, elevation of the official temperature sensor for
#                                                      the station.

# ELEV_GROUND_UNIT     20   X(20)        1031-1050     Ground elevation unit (should always be FEET).

# ELEV_BAROM           40   X(40)        1052-1091     Barometric: Ivory point of the Barometer or Altimeter Setting.

# ELEV_BAROM_UNIT      20   X(20)        1093-1112     Barometric elevation unit (should always be FEET).

# ELEV_AIR             40   X(40)        1114-1153     Airport: Field, Aerodrome, or Runway elevation.

# ELEV_AIR_UNIT        20   X(20)        1155-1174     Airport elevation unit (should always be FEET).

# ELEV_ZERODAT         40   X(40)        1176-1215     Zero Datum of a River Gage.

# ELEV_ZERODAT_UNIT    20   X(20)        1217-1236     Zero datum elevation unit (should always be FEET).

# ELEV_UNK             40   X(40)        1238-1277     Elevation value, type unknown.

# ELEV_UNK_UNIT        20   X(20)        1279-1298     Unknown elevation unit (should always be FEET).

# LAT_DEC              20   X(20)        1300-1319     Decimal latitude, blank indicates North and "-" indicates South.

# LON_DEC              20   X(20)        1321-1340     Decimal longitude, blank indicates East and "-" indicates West.

# LAT_LON_PRECISION    10   X(10)        1342-1351     Indicates precision of source lat and lon, see Reference Table A) below.

# RELOCATION           62   X(62)        1353-1414     Distance and direction of station relocation expressed as a distance
#                                                      value (1-4 characters), space, distance units (2 character abbreviation),
#                                                      space, and direction (1-3 character 16-point cardinal direction). Date of
#                                                      relocation indicated by begin date of record.

# UTC_OFFSET           16   9(16)        1416-1431     Time zone, UTC offset.

# OBS_ENV              40   X(40)        1433-1472     Type of observing programs associated with the station (LANDSFC, RADAR,
#                                                      UPPERAIR, UNKNOWN), delimited by comma.

# PLATFORM             100  X(100)       1474-1573     Station type and/or platforms station participates in, delimited by comma.
#                                                      See Reference Table B) below.

# *** UPDATE, beginning with August 2014 report: New columns appended

# GHCNMLT_ID           20   X(20)        1575-1594     Populated if station is included in GHCN-Monthly Land Temperature dataset.

# COUNTY_FIPS_CODE     5    X(5)         1596-1600     FIPS county code.

# DATUM_HORIZONTAL     30   X(30)        1602-1631     Horizontal reference datum used to determine the spatial fix of the station.
#                                                      Only available in newer periods where GPS receivers were used.

# DATUM_VERTICAL       30   X(30)        1633-1662     Vertical reference datum used to determine the elevation of the station.
#                                                      Only available in newer periods where GPS receivers were used.

# LAT_LON_SOURCE       100  X(100)       1664-1763     Latitude/longitude data source.  Only available in newer periods where GPS
#                                                      receivers were used.

# *** UPDATE, beginning with June 2016 report: New columns appended

# IGRA_ID              20   X(20)        1765-1784     Populated if station is included in Integrated Global Radiosonde Archive (IGRA) dataset.

# HPD_ID               20   X(20)        1786-1805     Populated if station is included in Hourly Precipitation Data (HPD) dataset.



# pbpaste | egrep "^# \w"
col_info = <<-ASDF
  SOURCE_ID            20   X(20)        001-020       Unique identifier from source system, should be used in union with SOURCE.
  SOURCE               10   X(10)        022-031       Name of source NCEI system (MMS, ISIS, GHCND, GHCNM-LT, IGRA2 etc). These are used
  BEGIN_DATE           8    YYYYMMDD     033-040       Beginning date of record, set to 00010101 if date is unknown.
  END_DATE             8    YYYYMMDD     042-049       Ending date of record, set to 99991231 if station is currently open.
  STATION_STATUS       20   X(20)        051-070       For the Cooperative Network, displays INACTIVE if station is currently
  NCDCSTN_ID           20   X(20)        072-091       Unique identifier used by NCEI, primary key for HOMR station.
  ICAO_ID              20   X(20)        093-112       Used for geographical locations throughout the world, managed by
  WBAN_ID              20   X(20)        114-133       Assigned by NCEI, used for digital data storage and general station
  FAA_ID               20   X(20)        135-154       Alpha-numeric, managed by USDT Federal Aviation Administration used
  NWSLI_ID             20   X(20)        156-175       Alpha-numeric, location identifier assigned by the National Weather
  WMO_ID               20   X(20)        177-196       Assigned by World Meteorological Organization, used for
  COOP_ID              20   X(20)        198-217       Assigned by NCEI, first 2 digits represent state, last 4 digits
  TRANSMITTAL_ID       20   X(20)        219-238       Holds miscellaneous IDs that do not fall into an officially sourced
  GHCND_ID             20   X(20)        240-259       Populated if station is included in GHCN-Daily dataset.
  NAME_PRINCIPAL       100  X(100)       261-360       Name of station, upper case may contain characters, numbers or symbols.
  NAME_PRINCIPAL_SHORT 30   X(30)        362-391       Name of station, upper case may contain characters, numbers or symbols.
  NAME_COOP            100  X(100)       393-492       Coop station name as maintained by NWS Cooperative network on WS Form
  NAME_COOP_SHORT      30   X(30)        494-523       Abbreviated Coop station name, limited to 30 characters, used by the
  NAME_PUBLICATION     100  X(100)       525-624       Populated with station names from legacy IPB.STATION table (LCD, QCLCD,
  NAME_ALIAS           100  X(100)       626-725       Any station alias known.
  NWS_CLIM_DIV         10   X(10)        727-736       Contains a number between 01 and 13 indicating climate division as
  NWS_CLIM_DIV_NAME    40   X(40)        738-777       Climate division name.
  STATE_PROV           10   X(10)        779-788       USPS two character alphabetic abbreviation for each state, uppercase.
  COUNTY               50   X(50)        790-839       Name of county, upper case.
  NWS_ST_CODE          2    X(2)         841-842       NWS state code.
  FIPS_COUNTRY_CODE    2    X(2)         844-845       FIPS country code.
  FIPS_COUNTRY_NAME    100  X(100)       847-946       FIPS country name.
  NWS_REGION           30   X(30)        948-977       NWS region (ALASKAN, CENTRAL, EASTERN, PACIFIC, SOUTHERN, WESTERN).
  NWS_WFO              10   X(10)        979-988       NWS weather forecast office.
  ELEV_GROUND          40   X(40)        990-1029      Ground elevation.  For Coop network, average elevation of the ground
  ELEV_GROUND_UNIT     20   X(20)        1031-1050     Ground elevation unit (should always be FEET).
  ELEV_BAROM           40   X(40)        1052-1091     Barometric: Ivory point of the Barometer or Altimeter Setting.
  ELEV_BAROM_UNIT      20   X(20)        1093-1112     Barometric elevation unit (should always be FEET).
  ELEV_AIR             40   X(40)        1114-1153     Airport: Field, Aerodrome, or Runway elevation.
  ELEV_AIR_UNIT        20   X(20)        1155-1174     Airport elevation unit (should always be FEET).
  ELEV_ZERODAT         40   X(40)        1176-1215     Zero Datum of a River Gage.
  ELEV_ZERODAT_UNIT    20   X(20)        1217-1236     Zero datum elevation unit (should always be FEET).
  ELEV_UNK             40   X(40)        1238-1277     Elevation value, type unknown.
  ELEV_UNK_UNIT        20   X(20)        1279-1298     Unknown elevation unit (should always be FEET).
  LAT_DEC              20   X(20)        1300-1319     Decimal latitude, blank indicates North and "-" indicates South.
  LON_DEC              20   X(20)        1321-1340     Decimal longitude, blank indicates East and "-" indicates West.
  LAT_LON_PRECISION    10   X(10)        1342-1351     Indicates precision of source lat and lon, see Reference Table A) below.
  RELOCATION           62   X(62)        1353-1414     Distance and direction of station relocation expressed as a distance
  UTC_OFFSET           16   9(16)        1416-1431     Time zone, UTC offset.
  OBS_ENV              40   X(40)        1433-1472     Type of observing programs associated with the station (LANDSFC, RADAR,
  PLATFORM             100  X(100)       1474-1573     Station type and/or platforms station participates in, delimited by comma.
  GHCNMLT_ID           20   X(20)        1575-1594     Populated if station is included in GHCN-Monthly Land Temperature dataset.
  COUNTY_FIPS_CODE     5    X(5)         1596-1600     FIPS county code.
  DATUM_HORIZONTAL     30   X(30)        1602-1631     Horizontal reference datum used to determine the spatial fix of the station.
  DATUM_VERTICAL       30   X(30)        1633-1662     Vertical reference datum used to determine the elevation of the station.
  LAT_LON_SOURCE       100  X(100)       1664-1763     Latitude/longitude data source.  Only available in newer periods where GPS
  IGRA_ID              20   X(20)        1765-1784     Populated if station is included in Integrated Global Radiosonde Archive (IGRA) dataset.
  HPD_ID               20   X(20)        1786-1805     Populated if station is included in Hourly Precipitation Data (HPD) dataset.
ASDF

# Headers
puts col_info.lines.map(&:split).map(&:first).to_csv

# Just like Julia ranges: One-based indexing, both ends inclusive
# "1786-1805" => [1786, 1805]
col_poses = col_info.lines.map(&:split).map do |row|
  start, stop = row[3].split("-")
  [ Integer(start.sub(/^0*/,"")), Integer(stop.sub(/^0*/,""))] # Integer doesn't like leading 0's, but I like that it will crash on bad input, unlike .to_i
end

wban_pos_start, wban_pos_stop = col_poses[col_info.lines.find_index { |line| line.split.first == "WBAN_ID" }]
wban_range = wban_pos_start-1 ... wban_pos_stop

IO.popen("unzip -p mshr_enhanced.txt.zip", "r").each_line do |line|
  next if line[wban_range].strip == ""
  puts begin
    col_poses.map do |start, stop|
      line[start-1 ... stop].strip
    end.to_csv
  end
end