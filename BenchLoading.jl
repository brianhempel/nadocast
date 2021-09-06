# import Dates

# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ ulimit -n 8192
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=. BenchLoading.jl


push!(LOAD_PATH, (@__DIR__) * "/models/shared")
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/models/rap_march_2014_forward")
import RAP

forecasts = RAP.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1:5:16]

isdir("bench") && rm("bench"; recursive = true)

TrainingShared.load_data_labels_weights_to_disk(
  "bench",
  forecasts[1:1];
  X_and_labels_to_inclusion_probabilities = ((X, labels, is_near_storm_event) -> max.(labels, 0.1f0, 0.5f0 .* is_near_storm_event))
)

rm("bench"; recursive = true)

@time TrainingShared.load_data_labels_weights_to_disk(
  "bench",
  forecasts[2:end];
  X_and_labels_to_inclusion_probabilities = ((X, labels, is_near_storm_event) -> max.(labels, 0.1f0, 0.5f0 .* is_near_storm_event))
)

rm("bench"; recursive = true)

data, labels, weights = TrainingShared.read_data_labels_weights_from_disk(save_dir)

# Ensure not change!
println(hash.((data, labels, weights)))
