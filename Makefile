# # Doesn't have start and end times: don't use.
# storm_data/1950-2016_all_tornadoes.csv:
# 	# Field descriptions here: http://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf
# 	curl http://www.spc.noaa.gov/wcm/data/1950-2016_all_tornadoes.csv > storm_data/1950-2016_all_tornadoes.csv
#
# # Start end times and start end lat lons. Example file. Use `make tornadoes` below for real.
# StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz:
# 	curl https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz > StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz

default:
	cat Makefile

julia:
	julia --project

notebook:
	jupyter lab

# Tornado, wind, and hail events, 2014 through the current year.
# As many as possible from the storm events database, then fill in more recent times with the SPC storm reports.
storm_events:
	ruby storm_data/get_storm_events.rb storm_data/tornadoes_downloaded.csv storm_data/wind_events_downloaded.csv storm_data/hail_events_downloaded.csv --add_spc_storm_reports
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/tornadoes_downloaded.csv > storm_data/tornadoes.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/wind_events_downloaded.csv > storm_data/wind_events.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/hail_events_downloaded.csv > storm_data/hail_events.csv # The merge deduplicates and sorts
	rm storm_data/tornadoes_downloaded.csv storm_data/wind_events_downloaded.csv storm_data/hail_events_downloaded.csv

forecast:
	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project lib/DoPredict.jl

# setup:
# 	julia Setup.jl
# 	# julia Test.jl
#
# get_rap:
# 	# See constants in get_rap.rb for setting date range.
# 	ruby get_rap.rb
#
# test_grib2s/rap_130_20170515_0000_001.grb2:
# 	curl https://nomads.ncdc.noaa.gov/data/rucanl/201705/20170515/rap_130_20170515_0000_001.grb2 > test_grib2s/rap_130_20170515_0000_001.grb2
#
# geo_regions/grid.csv: test_grib2s/rap_130_20170515_0000_001.grb2
# 	wgrib2 test_grib2s/rap_130_20170515_0000_001.grb2 -end -inv /dev/null -gridout - > grid.csv
#
# geo_regions/grid_coords_only.csv: geo_regions/grid.csv
# 	# GRASS GIS point input can't handle extra whitespace
# 	cat geo_regions/grid.csv | ruby -e 'puts STDIN.read.lines.map {|line| line.split(",")[2..3].map(&:strip).join(",")}.join("\n")' > grid_coords_only.csv
