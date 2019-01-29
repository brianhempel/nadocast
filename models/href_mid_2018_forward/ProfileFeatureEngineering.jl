using Profile

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

push!(LOAD_PATH, @__DIR__)
import HREF

forecasts = HREF.forecasts()[1:3]

function get_data_and_labels(forecasts)
  print("Loading")

  Xs = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data = HREF.get_feature_engineered_data(forecast, data)

    push!(Xs, data)

    print(".")
  end
  println("done.")

  vcat(Xs...) :: Array{<:Number,2}
end


get_data_and_labels(forecasts[1:1]) # Compile first.

# @profile get_data_and_labels(forecasts)
#
# using ProfileView
# ProfileView.view()
# read(stdin,UInt8)

# Profile.print(format = :flat, combine = true, sortedby = :count, mincount = 2)

@time get_data_and_labels(forecasts)
@time get_data_and_labels(forecasts)
@time get_data_and_labels(forecasts)
