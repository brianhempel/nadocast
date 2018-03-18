Pkg.update()

# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("DelimitedFiles")
Pkg.add("TimeZones")
# Pkg.add("Distances")
# Pkg.checkout("Distances") # haversine not in registered package as of 2017-12-30

Pkg.add("Flux")
# Pkg.test("Flux") # Check things installed correctly

# try
#   Pkg.clone("https://github.com/MetServiceDev/ECCodes.jl.git")
# catch exception
#   exception isa Base.Pkg.PkgError || rethrow()
# end
# Pkg.build("ECCodes")

Pkg.add("Proj4")
Pkg.build("Proj4")

# try
#   Pkg.clone("https://github.com/visr/GDAL.jl.git")
# catch exception
#   exception isa Base.Pkg.PkgError || rethrow()
# end
# Pkg.build("GDAL")
#
# try
#   Pkg.clone("https://github.com/yeesian/ArchGDAL.jl.git")
# catch exception
#   exception isa Base.Pkg.PkgError || rethrow()
# end
