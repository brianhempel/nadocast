default:
	cat Makefile

julia:
	julia --project

bench_loading:
	JULIA_NUM_THREADS=${CORE_COUNT} FORECASTS_ROOT=test_grib2s julia --project BenchLoading.jl

notebook:
	jupyter lab

# Tornado, wind, and hail events, 2014 through the current year.
# As many as possible from the storm events database, then fill in more recent times with the SPC storm reports.
storm_events:
	# ruby storm_data/get_storm_events.rb storm_data/tornadoes_downloaded.csv storm_data/wind_events_downloaded.csv storm_data/hail_events_downloaded.csv --add_spc_storm_reports
	ruby storm_data/get_storm_events.rb storm_data/tornadoes_downloaded.csv storm_data/wind_events_downloaded.csv storm_data/hail_events_downloaded.csv
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/tornadoes_downloaded.csv > storm_data/tornadoes.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/wind_events_downloaded.csv > storm_data/wind_events.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/hail_events_downloaded.csv > storm_data/hail_events.csv # The merge deduplicates and sorts
	rm storm_data/tornadoes_downloaded.csv storm_data/wind_events_downloaded.csv storm_data/hail_events_downloaded.csv

storm_events_for_climatology:
	# Last NEXRAD of the main installation program was installed in 1997
	START_YEAR=1998 STOP_YEAR=2013 BAD_LATLON_HANDLING=repair ruby storm_data/get_storm_events.rb storm_data/tornadoes_1998-2013_downloaded.csv storm_data/wind_events_1998-2013_downloaded.csv storm_data/hail_events_1998-2013_downloaded.csv
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/tornadoes_1998-2013_downloaded.csv > storm_data/tornadoes_1998-2013.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/wind_events_1998-2013_downloaded.csv > storm_data/wind_events_1998-2013.csv # The merge deduplicates and sorts
	ruby storm_data/deduplicate_sort_merge_csvs.rb storm_data/hail_events_1998-2013_downloaded.csv > storm_data/hail_events_1998-2013.csv # The merge deduplicates and sorts
	rm storm_data/tornadoes_1998-2013_downloaded.csv storm_data/wind_events_1998-2013_downloaded.csv storm_data/hail_events_1998-2013_downloaded.csv

# EMAIL=asdf@example.com make get_reflectivity_analysis
# get_reflectivity_analysis:
# 	DATASET=RUCANL130 DAYS_PER_ORDER=60 FORECAST_HOURS=0 FIELD_FILTER=REFC OUT_TOP_DIR_NAME=ruc_anl_reflectivity ruby get_data/order_and_get_rap_archived.rb 2008-11-1 2012-5-1
# 	DATASET=RAPANL130 DAYS_PER_ORDER=60 FORECAST_HOURS=0 FIELD_FILTER=REFC OUT_TOP_DIR_NAME=rap_anl_reflectivity ruby get_data/order_and_get_rap_archived.rb 2012-5-9 2022-5-15

# download_forecast_and_publish:
# 	TWEET=true ruby lib/download_and_forecast.rb

# forecast_and_publish:
# 	TWEET=true make forecast

# forecast:
# 	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project lib/DoPredict.jl

crontab:
	crontab crontab.cron

lib/href_one_field_for_grid.grib2:
	wgrib2 test_grib2s/href.t00z.conus.mean.f07.grib2 -end -grib lib/href_one_field_for_grid.grib2

lib/href_one_field_for_grid_cropped_3x_downsampled.grib2: lib/href_one_field_for_grid.grib2
	wgrib2 lib/href_one_field_for_grid.grib2 -new_grid_winds grid -new_grid lambert:265.000000:25.000000:25.000000:25.000000 234.906000:387:15237.000000 19.858000:226:15237.000000 lib/href_one_field_for_grid_cropped_3x_downsampled.grib2


training_setup_schooner:
	mkdir ~/bin

	cd ~
	curl https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz | tar -xvz
	ln -s $(pwd)/julia-1.7.2/bin/julia ~/bin/julia

	curl https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz | tar -xvz
	cd grib2/
	CC=gcc FC=gfortran make
	ln -s $(pwd)/wgrib2/wgrib2 ~/bin/wgrib2

	cd ~
	git clone https://github.com/brianhempel/nadocast.git nadocast_dev
	cd nadocast_dev

	module load zlib
	module load PROJ
	module load OpenMPI
	julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.instantiate()'
	# julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.update("MemoryConstrainedTreeBoosting")'


setup_data_webserver:
	ssh -i ~/.ssh/id_rsa root@data.nadocast.com

	adduser web # do remember the pw
	sudo setcap CAP_NET_BIND_SERVICE=+eip /usr/bin/miniserve # let it bind port 80
	# usermod -aG sudo web
	cp -r .ssh /home/web/
	mkdir /home/web/forecasts
	sudo chown -R web:web /home/web/.ssh /home/web/forecasts

	# Still as root...
	curl -L https://github.com/svenstaro/miniserve/releases/download/v0.19.4/miniserve-v0.19.4-x86_64-unknown-linux-musl > /usr/bin/miniserve
	chmod +x /usr/bin/miniserve
	# curl -L https://github.com/svenstaro/miniserve/raw/26395cd3595db1988fa64d7c8c0bc814c6631548/packaging/miniserve%40.service > /etc/systemd/system/miniserve@.service
	vim /etc/systemd/system/miniserve@.service
	# [Unit]
	# Description=miniserve for %i
	# After=network-online.target
	# Wants=network-online.target systemd-networkd-wait-online.service

	# [Service]
	# ExecStart=/usr/bin/miniserve --enable-tar --enable-tar-gz --enable-zip --show-wget-footer --no-symlinks --port 80 --dirs-first -- %I
	# Restart=on-failure
	# User=web
	# Group=web

	# [Install]
	# WantedBy=multi-user.target
	systemctl enable miniserve@-home-web-forecasts
	systemctl start miniserve@-home-web-forecasts
	systemctl status miniserve@-home-web-forecasts

	ssh -i ~/.ssh/id_rsa web@data.nadocast.com
	echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7ipPlBHZ8OwFc/updNsGshLUsDIgdwYWhXEFnu159NJiYG/gSU9aRw80V0r9MmOGOH8pY6EiTfK+hNTRxZ4pUlWHONglICMjj1d8PTqsr7V53CqjC7gB4vAWVlw1CTdTNN13nyI6aCOOjFy4kQoMp7a8GJQSyrRDBDqeE1zUk8rOBqI+wm70Vb0yq5LzA4v0D6GUJ48gMjG3OX30p/fNaqYbunucLKz0RjKyoleR0JLzqmq36ck1LgOuvYMhUK14IaicDA3Y/JNi4KEHoUDrLZFehJoruPTGNQX+82jnUP7axeQ45yblpzWOZWzcaKrrJATTznkKBN/j9rx+3YxVKXZl+QNbXKkfbClcUjUNGdD7YfiTYOEC1BNjM9MPdArh0jAqFBeDcIbDLlTrHpbbooT95PDPUB3B4vXFmNsWxpgsSHPS+gEPGhwrxFBfNqhwNvDitTkkKYVIqIbivyNyQJN9xu0SNpqFTjWhtjeRsTqOtASC5ad/HVtkayLaBORM= brian@nadocaster2" >> .ssh/authorized_keys

	# rsync -r --perms --chmod=a+rx forecasts/* web@data.nadocast.com:~/forecasts/




setup:
	# sudo dpkg-reconfigure tzdata # Choose "Other" to get UTC
	#
	# sudo apt update
	# sudo apt install openssh-server
	# sudo apt install vim ifstat
	# vim ~/.ssh/authorized_keys
	# sudo apt install screen
	# sudo apt install exfat-fuse exfat-utils
	# sudo apt install ruby

	# cd /media/brian/ssd
	# sudo fallocate -l 20g swapfile
	# sudo chmod 600 swapfile
	# sudo mkswap swapfile
	# sudo swapon swapfile
	# echo '/media/brian/ssd/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

	# Apparently you need to be logged in graphically in order for the HDs to automount

	# sudo ln -s /media/brian /Volumes
	# sudo apt install -y git
	# ssh-keygen
	# cat ~/.ssh/id_rsa.pub (upload to https://github.com/settings/keys)
	# cat ~/.ssh/id_rsa.pub >> authorized_keys # so we can log in to ourselves
	# git clone git@github.com:brianhempel/nadocast.git nadocast_dev
	# git clone git@github.com:brianhempel/nadocast.git nadocast_operational
	# cd nadocast_dev
	# sudo apt install curl
	# crontab crontab.cron
	#
	# curl https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz | tar -xvz
	# sudo apt install -y gcc make gfortran
	# cd grib2/
	# CC=gcc FC=gfortran make # CC=gcc-10 FC=gfortran-10 make on my Mac
	# mkdir ~/bin
	# ln -s $(pwd)/wgrib2/wgrib2 ~/bin/wgrib2
	# echo 'shell -\$SHELL' >> ~/.screenrc
	# echo 'export PATH=$HOME/bin:$PATH' >> ~/.bash_profile
	# echo 'PS1="\e[1;30m\h\e[m \e[1;36m\W\e[m\e[0;33m\$(__git_ps1) \$\e[m "' >> ~/.bash_profile
  # # make **/*.png globs work as expected
	# echo 'shopt -s globstar' >> ~/.bash_profile
	# source ~/.bash_profile
	# wgrib2 -config
	#
	# cd ~
	# curl https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz | tar -xvz
	# ln -s $(pwd)/julia-1.7.2/bin/julia ~/bin/julia

	# sudo apt install inetutils-ftp
	# ln -s `which inetutils-ftp` ~/bin/ftp

	# cd ~/nadocast_dev
	# git pull --rebase
	# sudo apt install -y libtool mpich
	# echo 'ENV["JULIA_MPI_BINARY"]="system"; import Pkg; Pkg.instantiate()' | julia --project=.
	# echo 'export CORE_COUNT=16' >> ~/.bash_profile
	# source ~/.bash_profile

	# cd models/sref_mid_2018_forward/
	# make train_gradient_boosted_decision_trees

	# cd ~/nadocast_dev/models/href_mid_2018_forward/
	# make train_gradient_boosted_decision_trees
	# JULIA_MPI_BINARY=system DISTRIBUTED=true JULIA_NUM_THREADS=$CORE_COUNT EVENT_TYPES=tornado,wind,hail,sig_tornado,sig_wind,sig_hail MUST_LOAD_FROM_DISK=true FORECAST_HOUR_RANGE=2:13 DATA_SUBSET_RATIO=0.26 time mpirun -n 2 -wdir $(pwd) -hosts 192.168.1.112:1,192.168.1.121:1 -bind-to none julia --compiled-modules=no --project=/home/brian/nadocast_dev /home/brian/nadocast_dev/models/sref_mid_2018_forward/TrainGradientBoostedDecisionTrees.jl

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
	# sudo gem install chunky_png
	# twurl authorize --consumer-key key --consumer-secret secret
	# scp ~/.twurlrc nadocaster:~/.twurlrc # if authorization done on foreign machine

	# curl -L https://github.com/shssoichiro/oxipng/releases/download/v5.0.1/oxipng-5.0.1-x86_64-unknown-linux-musl.tar.gz | tar -xvz
	# mv oxipng-5.0.1-x86_64-unknown-linux-musl/oxipng ~/bin/
	# rm -r oxipng-5.0.1-x86_64-unknown-linux-musl
	# sudo apt install pngquant

	# sudo apt install ffmpeg

	# make forecast
	# scp -r nadocaster:~/nadocast_dev/forecasts remote_forecasts
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
