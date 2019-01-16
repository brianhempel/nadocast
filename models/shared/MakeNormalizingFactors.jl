module MakeNormalizingFactors

import Random

push!(LOAD_PATH, (@__DIR__) * "/../..")

import Conus
import Forecasts
# import NNTrain
import StormEvents

push!(LOAD_PATH, @__DIR__)

import TrainingShared

# Samples the forecasts, so won't get quite the same values every time.
function calculate_normalizing_factors(all_forecasts)
  (grid, conus_on_grid, feature_count, train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_grid_conus_on_grid_feature_count_train_validation_test(all_forecasts)

  sample_count = 100

  sample_forecasts = Iterators.take(Random.shuffle(train_forecasts), sample_count)

  normalizing_factors = zeros(Float32, feature_count)

  print("Finding max feature values amoung $sample_count random training forecasts")
  for forecast in sample_forecasts
    data =
      try
        Forecasts.get_data(forecast) # dims: grid_count by layer_count
      catch exception
        if isa(exception, EOFError) || isa(exception, ErrorException)
          println("Bad Forecast: $(Forecasts.time_title(forecast))")
          continue
        else
          rethrow(exception)
        end
      end

    for i in 1:feature_count
      conus_layer_abs_values = abs.(data[:,i] .* conus_on_grid)
      max_abs_value          = maximum(conus_layer_abs_values)
      normalizing_factors[i] = max(normalizing_factors[i], max_abs_value)
    end
    print(".")
  end

  print("normalizing 0.0 to 1.0e10...")

  normalizing_factors = map(x -> x == 0.0f0 ? 1.0f10 : x, normalizing_factors)

  println("done.")

  normalizing_factors
end

end # module MakeNormalizingFactors