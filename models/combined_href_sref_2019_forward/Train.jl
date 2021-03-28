import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared

push!(LOAD_PATH, @__DIR__)
import CombinedHREFSREF

forecasts_0z = filter(forecast -> forecast.run_hour == 0, CombinedHREFSREF.forecasts_href_newer())

(train_forecasts_0z, validation_forecasts_0z, _) = TrainingShared.forecasts_train_validation_test(forecasts_0z)

# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

X, y, weights = TrainingShared.get_data_labels_weights(validation_forecasts_0z; save_dir = "validation_data_cache")

function try_combine(combiner)
  ŷ = map(i -> combiner(X[i, :]), 1:length(y))

  sum(logloss.(y, ŷ) .* weights) / sum(weights)
end

println( try_combine(x -> x[1]) )
println( try_combine(x -> x[2]) )
println( try_combine(x -> 0.5f0*(x[1] x[2])) )
println( try_combine(minimum) )
println( try_combine(maximum) )
