module SREFTrainShared

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Conus
import Forecasts

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import SREF


function get_data_and_labels(grid, conus_grid_bitmask, forecasts; X_transformer = identity)
  Xs = []
  Ys = []

  for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts_no_caching(forecasts)
    data = SREF.get_feature_engineered_data(forecast, data)

    data_in_conus = data[conus_grid_bitmask, :]
    labels        = TrainingShared.forecast_labels(grid, forecast)[conus_grid_bitmask] :: Array{Float32,1}

    push!(Xs, X_transformer(data_in_conus))
    push!(Ys, labels)

    print(".")
  end

  (vcat(Xs...), vcat(Ys...))
end

end # module SREFTrainShared