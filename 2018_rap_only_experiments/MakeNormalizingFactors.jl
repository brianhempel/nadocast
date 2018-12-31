# Some normalizing divisors to keep gradients from exploding.
#
# $ julia MakeNormalizingFactors.jl model/something.binfeatures
#
# Outputs absolute values max of each feature: can copy-paste into NormalizingFactors.jl


data_path = ARGS[1]

include("ReadFeatureData.jl")

data, data_file, point_count = open_data_file(data_path)

maxes = abs.(@view data[:,1])

for i = 2:size(data,2)
  maxes = max.(maxes, abs.(@view data[:,i]))
end


# maxes = map(x -> x == 0.0  ? 1.0 : x , findmax(abs.(data),2)[1])
# means = map(x -> x == 0.0 ? 1.0 : x , mean(abs.(data),2))

close(data_file)

println("normalizing_factors = Float32[$(join(maxes, ", "))]")
# println("normalizing_factors = Float32[$(join(means, ", "))]")
