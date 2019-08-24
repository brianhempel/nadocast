using Profile

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

push!(LOAD_PATH, @__DIR__)
import RAP

forecasts = RAP.forecasts()[1:5]

function get_data_and_labels(forecasts)
  print("Loading")

  Xs = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(forecasts)
    data = RAP.get_feature_engineered_data(forecast, data)

    push!(Xs, data)

    print(".")
  end
  println("done.")

  vcat(Xs...) :: Array{<:Number,2}
end


get_data_and_labels(forecasts[1:1]) # Compile first.

# Profile.init(n = 10^7, delay = 0.05)
#
# @profile get_data_and_labels(forecasts[1:2])
#
# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

@time get_data_and_labels(forecasts[1:2])
@time get_data_and_labels(forecasts[1:2])
@time get_data_and_labels(forecasts[1:2])
