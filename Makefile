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
	ruby storm_data/get_storm_events.rb > storm_data/tornadoes.csv
	diff storm_data/tornadoes_old.csv storm_data/tornadoes.csv

# In case of gov't shutdown or non-final storm events database, we can use SPC storm reports.
add_2018_spc_tornado_reports:
	rm ~/.julia/compiled/v1.0/StormEvents.ji # Tornadoes are a constant in this file so it will need to be recompiled.
	cd storm_data
	cp tornadoes.csv tornadoes_old.csv
	ruby get_spc_storm_reports.rb 2017-12-31 2019-01-01 > tornadoes_from_spc_storm_reports.csv
	ruby merge_csvs.rb tornadoes_old.csv tornadoes_from_spc_storm_reports.csv > tornadoes.csv
	diff tornadoes_old.csv tornadoes.csv

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
