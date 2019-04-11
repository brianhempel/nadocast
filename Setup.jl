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
# crontab get_data/crontab.cron
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

# make forecast
# scp -r nadocaster:~/nadocast/forecasts remote_forecasts





# add DelimitedFiles # For readdlm
# add Plots
# add JSON
# add GDAL
# add ArchGDAL
# add Proj4 # for GeoUtils geodesics
# # add JLD
# # add https://github.com/dmlc/XGBoost.jl
# add TranscodingStreams
# add CodecZstd
# dev ../MemoryConstrainedTreeBoosting.jl


# Pkg.update()
#
# # Pkg.add("DataFrames")
# # Pkg.add("CSV")
# # Pkg.add("DelimitedFiles")
# Pkg.add("TimeZones")
# # Pkg.add("Distances")
# # Pkg.checkout("Distances") # haversine not in registered package as of 2017-12-30
#
# Pkg.add("Flux")
# # Pkg.test("Flux") # Check things installed correctly
#
# Pkg.add("BSON")
#
# Pkg.clone("https://github.com/Allardvm/LightGBM.jl.git")
#
# # try
# #   Pkg.clone("https://github.com/MetServiceDev/ECCodes.jl.git")
# # catch exception
# #   exception isa Base.Pkg.PkgError || rethrow()
# # end
# # Pkg.build("ECCodes")
#
# Pkg.add("Proj4")
# Pkg.build("Proj4")
#
# # try
# #   Pkg.clone("https://github.com/visr/GDAL.jl.git")
# # catch exception
# #   exception isa Base.Pkg.PkgError || rethrow()
# # end
# # Pkg.build("GDAL")
# #
# # try
# #   Pkg.clone("https://github.com/yeesian/ArchGDAL.jl.git")
# # catch exception
# #   exception isa Base.Pkg.PkgError || rethrow()
# # end
