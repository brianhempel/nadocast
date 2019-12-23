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

storm_events_for_climatology:
	# Last NEXRAD of the main installation program was installed in 1997
	START_YEAR=1998 STOP_YEAR=2013 ruby storm_data/get_storm_events.rb storm_data/tornadoes_1998-2013_downloaded.csv storm_data/wind_events_1998-2013_downloaded.csv storm_data/hail_events_1998-2013_downloaded.csv
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/tornadoes_1998-2013_downloaded.csv > storm_data/tornadoes_1998-2013.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/wind_events_1998-2013_downloaded.csv > storm_data/wind_events_1998-2013.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/hail_events_1998-2013_downloaded.csv > storm_data/hail_events_1998-2013.csv # The merge deduplicates and sorts
	rm storm_data/tornadoes_1998-2013_downloaded.csv storm_data/wind_events_1998-2013_downloaded.csv storm_data/hail_events_1998-2013_downloaded.csv


forecast_and_publish:
	TWEET=true make forecast

forecast:
	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project lib/DoPredict.jl

crontab:
	crontab crontab.cron

lib/href_one_field_for_grid.grib2:
	wgrib2 test_grib2s/href.t00z.conus.mean.f07.grib2 -end -grib lib/href_one_field_for_grid.grib2

setup:
	# sudo dpkg-reconfigure tzdata # Choose "Other" to get UTC
	#
	# sudo apt update
	# sudo apt install openssh-server
	# sudo apt install vim
	# vim ~/.ssh/authorized_keys
	# sudo apt install screen
	# sudo apt install exfat-fuse exfat-utils
	# sudo apt install ruby

	# Apparently you need to be logged in graphically in order for the HDs to automount

	# sudo ln -s /media/brian /Volumes
	# sudo apt install git
	# ssh-keygen
	# cat ~/.ssh/id_rsa.pub (upload to https://github.com/settings/keys)
	# git clone git@github.com:brianhempel/nadocast.git
	# cd nadocast
	# sudo apt install curl
	# crontab crontab.cron
	#
	# curl https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz | tar -xvz
	# sudo apt install gcc
	# sudo apt install make
	# sudo apt install gfortran
	# cd grib2/
	# CC=gcc FC=gfortran make  # CC=gcc-8 FC=gfortran-8 make on my Mac
	# mkdir ~/bin
	# ln -s $(pwd)/wgrib2/wgrib2 ~/bin/wgrib2
	# echo 'shell -$SHELL' >> ~/.screenrc
	# echo 'export PATH=$HOME/bin:$PATH' >> ~/.bash_profile
	# source ~/.bash_profile
	# wgrib2 -config
	#
	# cd ~
	# curl https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz | tar -xvz
	# ln -s $(pwd)/julia-1.1.0/bin/julia ~/bin/julia
	#
	# cd ~/nadocast
	# git pull --rebase
	# echo 'import Pkg; Pkg.instantiate()' | julia --project=.
	# echo 'export CORE_COUNT=12' >> ~/.bash_profile
	# source ~/.bash_profile

	# cd models/sref_mid_2018_forward/
	# make train_gradient_boosted_decision_trees

	# cd ~/nadocast/models/href_mid_2018_forward/
	# make train_gradient_boosted_decision_trees

	# cd ~
	# git clone https://github.com/GenericMappingTools/gmt.git
	# cd gmt
	# sudo apt install cmake ninja-build libcurl4-gnutls-dev libnetcdf-dev libgdal-dev libfftw3-dev libpcre3-dev liblapack-dev ghostscript curl
	# mkdir coastlines
	# COASTLINEDIR=$(pwd)/coastlines ./ci/download-coastlines.sh
	# rm -r ~/cache-gshhg-dcw
	# cp cmake/ConfigUserTemplate.cmake cmake/ConfigUser.cmake
	# echo "set (DCW_ROOT \"$(pwd)/coastlines\")" >> cmake/ConfigUser.cmake
	# echo "set (GSHHG_ROOT \"$(pwd)/coastlines\")" >> cmake/ConfigUser.cmake
	# mkdir build
	# cd build
	# cmake -DCMAKE_INSTALL_PREFIX=$HOME -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
	# make -j
	# make -j install

	# sudo apt install ruby-dev
	# sudo gem install twurl
	# twurl authorize --consumer-key key --consumer-secret secret
	# scp ~/.twurlrc nadocaster:~/.twurlrc # if authorization done on foreign machine

	# sudo apt install ffmpeg

	# make forecast
	# scp -r nadocaster:~/nadocast/forecasts remote_forecasts
	# make forecast_and_publish

# Security updates
update:
	sudo apt-get update
	sudo apt-get dist-upgrade
	sudo reboot
	# May need to manually unplug USB drives and physically push restart
	# Also, HD's only auto-mount from the GUI so have to log in and plug them in


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
