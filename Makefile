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

# Tornado events, 2014 through the current year. Both start and end times.
tornadoes:
	cp storm_data/tornadoes.csv storm_data/tornadoes_old.csv
	ruby storm_data/get_storm_events.rb > storm_data/tornadoes_downloaded.csv
	ruby storm_data/merge_csvs.rb storm_data/tornadoes_downloaded.csv storm_data/tornadoes_downloaded.csv > storm_data/tornadoes.csv # The merge deduplicates and sorts
	rm storm_data/tornadoes_downloaded.csv
	diff storm_data/tornadoes_old.csv storm_data/tornadoes.csv

# In case of gov't shutdown or non-final storm events database, we can use SPC storm reports.
add_2019_spc_tornado_reports:
	cp storm_data/tornadoes.csv storm_data/tornadoes_old.csv
	ruby storm_data/get_spc_storm_reports.rb 2019-01-01 2019-03-17 > storm_data/tornadoes_from_spc_storm_reports.csv
	ruby storm_data/merge_csvs.rb storm_data/tornadoes_old.csv storm_data/tornadoes_from_spc_storm_reports.csv > storm_data/tornadoes.csv
	rm storm_data/tornadoes_from_spc_storm_reports.csv
	diff storm_data/tornadoes_old.csv storm_data/tornadoes.csv

forecast:
	julia --project lib/DoPredict.jl

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
