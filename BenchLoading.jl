# import Dates

# $ sshfs -o debug,sshfs_debug,loglevel=debug brian@nadocaster2:/Volumes/ ~/nadocaster2/
# $ ulimit -n 8192
# $ FORECASTS_ROOT=/home/brian/nadocaster2/ JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=. BenchLoading.jl


push!(LOAD_PATH, (@__DIR__) * "/lib")
import Forecasts
import Inventories

push!(LOAD_PATH, (@__DIR__) * "/models/shared")
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/models/rap_march_2014_forward")
import RAP

forecasts = RAP.three_hour_window_three_hour_min_mean_max_delta_feature_engineered_forecasts()[1:5:16]

isdir("bench") && rm("bench"; recursive = true)

TrainingShared.load_data_labels_weights_to_disk(
  "bench",
  forecasts[1:1];
  calc_inclusion_probabilities = ((labels, is_near_storm_event) -> max.(labels .> 0f0, 0.1f0, 0.5f0 .* is_near_storm_event))
)

rm("bench"; recursive = true)

# TrainingShared.load_data_labels_weights_to_disk(
#   "bench_ref",
#   forecasts[2:end];
#   calc_inclusion_probabilities = ((labels, is_near_storm_event) -> max.(labels .> 0f0, 0.1f0, 0.5f0 .* is_near_storm_event))
# )

@time TrainingShared.load_data_labels_weights_to_disk(
  "bench",
  forecasts[2:end];
  calc_inclusion_probabilities = ((labels, is_near_storm_event) -> max.(labels .> 0f0, 0.1f0, 0.5f0 .* is_near_storm_event))
)

data, labels, weights = TrainingShared.read_data_labels_weights_from_disk("bench")

# Ensure not changed!
println(hash.((data, labels, weights)))

rm("bench"; recursive = true)

import Statistics

if hash.((data, labels, weights)) != (0xe476246de6fc1e1b, 0x7ad97045e5aa04f0, 0xb73b4d168334c7ee)

  data_ref, labels_ref, weights_ref = TrainingShared.read_data_labels_weights_from_disk("bench_ref")

  if data != data_ref
    # need to calc std dev or something

    # means            = Array{Float64,1}(undef, size(data_ref,2))
    # stddevs          = Array{Float64,1}(undef, size(data_ref,2))
    normalized_diffs      = Array{Float64,2}(undef, size(data_ref))
    normalized_diff_maxes = Array{Float64,1}(undef, size(data_ref,2))

    Threads.@threads for feature_i in 1:size(data_ref,2)
      ref_feature = Float64.(@view data_ref[:,feature_i])
      feature     = Float64.(@view data[:,feature_i])
      # mean        = Statistics.mean(ref_feature)
      stddev      = Statistics.std(ref_feature) + 1e-323

      diff            = abs.(ref_feature .- feature)
      normalized_diff = diff ./ stddev

      # print("$(stddev) $(maximum(diff)) ")

      # means[feature_i]               = mean
      # stddevs[feature_i]             = stddev
      normalized_diffs[:, feature_i]   = normalized_diff
      normalized_diff_maxes[feature_i] = maximum(normalized_diff)
    end

    max_diff, worst_feature_i = findmax(normalized_diff_maxes)

    println("data changed!! mean normalized change $(Statistics.mean(normalized_diffs)) max normalized change $(max_diff)")

    inventory = Forecasts.inventory(forecasts[1])
    println("worst features")
    for feature_i in reverse(sortperm(normalized_diff_maxes))[1:20]
      println("$(normalized_diff_maxes[feature_i])\t$(Inventories.inventory_line_description(inventory[feature_i]))")
    end
  end

  if labels != labels_ref
    println("labels changed!!")
  end

  if weights != weights_ref
    println("weights changed!!")
  end

  error("something changed!!")
end




# 8.642694f0, -8.642695f0), (-12.527906f0, -12.52833f0), (-13.494373f0, -13.494175f0), (-13.499801f0, -13.499439f0), (-13.40384f0, -13.401827f0), (-13.553148f0, -13.551085f0), (-12.903235f0, -12.900507f0), (-13.084454f0, -13.081537f0), (-14.213368f0, -14.210239f0), (-15.300161f0, -15.296888f0), (-10.713873f0, -10.713668f0), (-11.389256f0, -11.389882f0), (-13.587333f0, -13.589408f0), (-14.101154f0, -14.104424f0), (-16.786379f0, -16.789877f0), (-17.047947f0, -17.051294f0), (-17.832739f0, -17.835718f0), (-6.3335886f0, -6.3252926f0), (-6.0753036f0, -6.0719743f0), (-5.539244f0, -5.5438223f0), (-6.7505503f0, -6.7558923f0), (-7.294572f0, -7.298705f0), (-3.7658403f0, -3.765839f0), (-3.6553178f0, -3.6555243f0), (-3.3995957f0, -3.3997014f0), (-3.0399272f0, -3.0395703f0), (-3.6777444f0, -3.677752f0), (-3.655849f0, -3.6558094f0), (-5.9235744f0, -5.923442f0), (-5.0179095f0, -5.0168977f0), (-6.266481f0, -6.2661495f0), (-8.245043f0, -8.245049f0), (-9.244927f0, -9.244914f0), (-9.75494f0, -9.7549f0), (-10.715268f0, -10.715619f0), (-13.342081f0, -13.340622f0), (-13.304461f0, -13.302224f0), (-13.380007f0, -13.377619f0), (-13.045064f0, -13.0419f0), (-14.209668f0, -14.206451f0), (-14.803387f0, -14.800134f0), (-16.101866f0, -16.099237f0), (-16.330849f0, -16.328537f0), (-16.398087f0, -16.397497f0), (-12.1699505f0, -12.170822f0), (-11.385941f0, -11.386234f0), (-10.219652f0, -10.219825f0), (-10.249647f0, -10.249215f0), (-16.162487f0, -16.16525f0), (-17.552164f0, -17.554426f0), (-3.6019979f0, -3.6026287f0), (-3.339561f0, -3.3397357f0), (-3.020655f0, -3.0207996f0), (-3.2372255f0, -3.2374136f0), (-3.5354652f0, -3.53559f0), (-3.6776006f0, -3.677669f0), (-6.509914f0, -6.5099063f0), (-6.4595456f0, -6.4589868f0), (-7.439279f0, -7.439368f0), (-8.432301f0, -8.432323f0), (-13.231062f0, -13.229321f0), (-13.302923f0, -13.300264f0), (-13.037883f0, -13.03451f0), (-16.384192f0, -16.382364f0), (-16.699858f0, -16.698956f0), (-16.15523f0, -16.154787f0), (-11.76235f0, -11.762399f0), (-9.99765f0, -9.997994f0), (-9.762647f0, -9.762608f0), (-10.151565f0, -10.151127f0), (-13.315032f0, -13.318679f0), (-6.416768f0, -6.410009f0), (-6.1012373f0, -6.1097565f0), (-6.7674575f0, -6.773646f0), (-3.457485f0, -3.4585934f0), (-3.0757f0, -3.0756917f0), (-3.1572564f0, -3.1573973f0), (-3.3880754f0, -3.3881867f0), (-3.9435282f0, -3.9433756f0), (-6.057391f0, -6.057263f0), (-6.5482283f0, -6.5476966f0), (-6.0723076f0, -6.071164f0), (-8.882496f0, -8.882512f0), (-11.0160675f0, -11.016579f0), (-11.869229f0, -11.869554f0), (-12.838864f0, -12.839168f0), (-13.253291f0, -13.2530575f0), (-13.189615f0, -13.189257f0), (-13.006914f0, -13.006179f0), (-13.084537f0, -13.081762f0), (-12.985796f0, -12.983128f0), (-13.133522f0, -13.130676f0), (-13.0693245f0, -13.065692f0), (-15.422124f0, -15.422624f0), (-12.181272f0, -12.183165f0), (-11.307917f0, -11.309781f0), (-6.161251f0, -6.161041f0), (-5.803024f0, -5.806358f0), (-7.0444994f0, -7.0475106f0), (-6.2693663f0, -6.269215f0), (-6.141414f0, -6.140883f0), (-5.5951276f0, -5.594508f0), (-5.5910783f0, -5.590188f0), (-8.473027f0, -8.473231f0), (-8.787507f0, -8.787617f0), (-10.03634f0, -10.036563f0), (-9.739187f0, -9.739252f0), (-13.023022f0, -13.020836f0), (-13.167547f0, -13.165068f0), (-12.999292f0, -12.996364f0), (-12.993025f0, -12.990093f0), (-15.108889f0, -15.105487f0), (-15.335428f0, -15.3320875f0), (-16.284971f0, -16.283949f0), (-15.231499f0, -15.232172f0), (-12.12752f0, -12.129844f0), (-12.136741f0, -12.1394615f0), (-11.198179f0, -11.199687f0), (-11.80914f0, -11.809714f0), (-9.355106f0, -9.355086f0), (-9.944278f0, -9.945265f0), (-10.085