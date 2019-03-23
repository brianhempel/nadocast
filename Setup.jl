# sudo apt update
# sudo apt install openssh-server
# sudo apt install vim
# vim ~/.ssh/authorized_keys
# sudo apt install screen
# sudo apt install exfat-fuse exfat-utils
# sudo apt install ruby
# sudo ln -s /media/brian /Volumes
# sudo apt install git
# ssh-keygen
# cat ~/.ssh/id_rsa.pub (upload to https://github.com/settings/keys)
# git clone git@github.com:brianhempel/nadocast.git
# cd nadocast
# sudo apt install curl
# crontab get_data/crontab.cron



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
