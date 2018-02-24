1950-2016_all_tornadoes.csv:
	# Field descriptions here: http://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf
	curl http://www.spc.noaa.gov/wcm/data/1950-2016_all_tornadoes.csv > 1950-2016_all_tornadoes.csv

# Start end times and start end lat lons
StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz:
	curl https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz > StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz

setup: 1950-2016_all_tornadoes.csv StormEvents_details-ftp_v1.0_d2017_c20171218.csv.gz
	julia Setup.jl
	# julia Test.jl


get_rap:
	# See constants in get_rap.rb for setting date range.
	ruby get_rap.rb

rap_130_20170515_0000_001.grb2:
	curl https://nomads.ncdc.noaa.gov/data/rucanl/201705/20170515/rap_130_20170515_0000_001.grb2 > rap_130_20170515_0000_001.grb2

grid.txt: rap_130_20170515_0000_001.grb2
	wgrib2 rap_130_20170515_0000_001.grb2 -end -s -gridout - | ruby -e 'puts STDIN.read.strip.split(":").last' > grid.txt

grid_coords_only.txt: grid.txt
	# GRASS GIS point input can't handle extra whitespace
	cat grid.txt | ruby -e 'puts STDIN.read.lines.map {|line| line.split(",")[2..3].map(&:strip).join(",")}.join("\n")' > grid_coords_only.txt