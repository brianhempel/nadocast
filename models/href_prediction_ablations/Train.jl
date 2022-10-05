# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.


import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories


(_, validation_forecasts, test_forecasts) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.forecasts(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);
length(validation_forecasts) # 24373

forecast_i = 1
start_time = time_ns()
for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(validation_forecasts)
  elapsed = (time_ns() - start_time) / 1.0e9
  print("\r$forecast_i/~$(length(validation_forecasts)) forecasts loaded.  $(Float32(elapsed / forecast_i))s each.  ~$(Float32((elapsed / forecast_i) * (length(validation_forecasts) - forecast_i) / 60 / 60)) hours left.            ")
  forecast_i += 1
end

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
length(test_forecasts) # 24662

forecast_i = 1
start_time = time_ns()
for (forecast, data) in Forecasts.iterate_data_of_uncorrupted_forecasts(test_forecasts)
  elapsed = (time_ns() - start_time) / 1.0e9
  print("\r$forecast_i/~$(length(test_forecasts)) forecasts loaded.  $(Float32(elapsed / forecast_i))s each.  ~$(Float32((elapsed / forecast_i) * (length(test_forecasts) - forecast_i) / 60 / 60)) hours left.            ")
  forecast_i += 1
end

# disk2_date = Dates.DateTime(2021, 3, 1, 0)
# validation_forecasts = filter(forecast -> Forecasts.run_utc_datetime(forecast) >= disk2_date, validation_forecasts);

# for testing
# validation_forecasts = rand(validation_forecasts, 30);

# validation_forecasts1 = filter(forecast -> isodd(forecast.run_day),  validation_forecasts);
# validation_forecasts2 = filter(forecast -> iseven(forecast.run_day), validation_forecasts);

# # rm("validation_forecasts_with_blurs_and_forecast_hour1"; recursive=true)
# # rm("validation_forecasts_with_blurs_and_forecast_hour2"; recursive=true)

# # To double loading speed, manually run the other one of these in a separate process with USE_ALT_DISK=true
# # When it's done, run it in the main process and it will load from the save_dir

# function dictmap(f, dict)
#   out = Dict()
#   for (k, v) in dict
#     out[k] = f(v)
#   end
#   out
# end

# if get(ENV, "USE_ALT_DISK", "false") != "true"
#   X1, Ys1, weights1 =
#     TrainingShared.get_data_labels_weights(
#       validation_forecasts1;
#       event_name_to_labeler = TrainingShared.event_name_to_labeler,
#       save_dir = "validation_forecasts_with_blurs_and_forecast_hour1"
#     );

#   Ys1 = dictmap(y -> y .> 0.5, Ys1) # Convert to bitarrays. This saves memory
#   # Then wait for the X2, Ys2, weights2 to finish in the other process, then continue.
# end

# # blur_radii = [0; HREFPrediction.blur_radii]
# # X1 = X1[:,1:2*length(blur_radii)]

# GC.gc()

# X2, Ys2, weights2 =
#   TrainingShared.get_data_labels_weights(
#     validation_forecasts2;
#     event_name_to_labeler = TrainingShared.event_name_to_labeler,
#     save_dir = "validation_forecasts_with_blurs_and_forecast_hour2"
#   );
# Ys2 = dictmap(y -> y .> 0.5, Ys2) # Convert to bitarrays. This saves memory

# GC.gc()

# if get(ENV, "USE_ALT_DISK", "false") == "true"
#   exit(0)
# end

# # blur_radii = [0; HREFPrediction.blur_radii]
# # X2 = X2[:,1:2*length(blur_radii)]

# Ys = Dict{String, BitArray}();
# for event_name in keys(Ys1)
#   Ys[event_name] = vcat(Ys1[event_name], Ys2[event_name])
# end

# GC.gc()

# weights = vcat(weights1, weights2);

# # Free
# Ys1, weights1 = (nothing, nothing)
# Ys2, weights2 = (nothing, nothing)

# GC.gc()

# X = vcat(X1, X2);

# # Free
# X1 = nothing
# X2 = nothing

# GC.gc()



# function test_predictive_power(forecasts, X, Ys, weights)
#   inventory = Forecasts.inventory(forecasts[1])

#   for prediction_i in 1:length(HREFPrediction.models)
#     (event_name, _, _) = HREFPrediction.models[prediction_i]
#     y = Ys[event_name]
#     for j in 1:size(X,2)
#       x = @view X[:,j]
#       au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
#       println("$event_name ($(round(sum(y)))) feature $j $(Inventories.inventory_line_description(inventory[j]))\tAU-PR-curve: $au_pr_curve")
#     end
#   end
# end
# test_predictive_power(validation_forecasts, X, Ys, weights)





# CHECKING that the blurred forecasts are correct

# Run this file in a Julia REPL bit by bit.
#
# Poor man's notebook.

# The unblurred prediction forecasts should be in the lib/computation_cache, so don't waste time hitting disk
# FORECAST_DISK_PREFETCH=false make julia

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_blurred, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.regular_forecasts(HREFPredictionAblations.forecasts_blurred()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts_blurred = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_blurred);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_blurred[100]);

# rm("validation_forecasts_blurred"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_blurred; event_name_to_labeler = Dict("tornado" => TrainingShared.event_name_to_labeler["tornado"]), save_dir = "validation_forecasts_blurred");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])
  Σy = Float32(sum(Ys["tornado"]))

  for prediction_i in 1:length(HREFPredictionAblations.models)
    (model_name, _, _) = HREFPredictionAblations.models[prediction_i]
    y = Ys["tornado"]
    x = @view X[:,prediction_i]
    # au_pr_curve = Metrics.area_under_pr_curve_fast(x, y, weights)
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)

    println("$model_name ($Σy) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
  end
end
test_predictive_power(validation_forecasts_blurred, X, Ys, weights)

# tornadao_mean_58                                           (75293.0) feature 1  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.03622471
# tornadao_prob_80                                           (75293.0) feature 2  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.03266264
# tornadao_mean_prob_138                                     (75293.0) feature 3  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.03977368
# tornadao_mean_prob_computed_no_sv_219                      (75293.0) feature 4  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.037974633
# tornadao_mean_prob_computed_220                            (75293.0) feature 5  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.036877435
# tornadao_mean_prob_computed_partial_climatology_227        (75293.0) feature 6  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.04454543
# tornadao_mean_prob_computed_climatology_253                (75293.0) feature 7  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.048470356
# tornadao_mean_prob_computed_climatology_blurs_910          (75293.0) feature 8  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.04281302
# tornadao_mean_prob_computed_climatology_grads_1348         (75293.0) feature 9  TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.04544138
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   (75293.0) feature 10 TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.045882963
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 (75293.0) feature 11 TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.05112551
# tornadao_mean_prob_computed_climatology_3hr_1567           (75293.0) feature 12 TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.056183856
# tornado_full_13831                                         (75293.0) feature 13 TORPROB:calculated:hour fcst:calculated_prob:blurred AU-PR-curve: 0.047563255


# Yay!

function reliability_curves_stairstep(bin_count, X, y, weights, inventory)
  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

  nmodels = size(X, 2)

  bins_Σŷ      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σy      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σweight = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_max     = map(_ -> ones(Float32, bin_count), 1:nmodels)

  for prediction_i in 1:nmodels
    ŷ              = @view X[:,prediction_i]; # HREF prediction for event_name
    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_i = 1
    for i in eachindex(y_sorted)
      if ŷ_sorted[i] > bins_max[prediction_i][bin_i]
        bin_i += 1
      end

      bins_Σŷ[prediction_i][bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
      bins_Σy[prediction_i][bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
      bins_Σweight[prediction_i][bin_i] += Float64(weights_sorted[i])

      if bins_Σy[prediction_i][bin_i] >= per_bin_pos_weight
        bins_max[prediction_i][bin_i] = ŷ_sorted[i]
      end
    end
  end

  for prediction_i in 1:nmodels
    print("ŷ_$(Inventories.inventory_line_description(inventory[prediction_i])),y_$(Inventories.inventory_line_description(inventory[prediction_i])),")
  end
  println()

  for bin_i in 1:bin_count
    for prediction_i in 1:nmodels
      # Σŷ      = bins_Σŷ[prediction_i][bin_i]
      Σy      = bins_Σy[prediction_i][bin_i]
      Σweight = bins_Σweight[prediction_i][bin_i]
      bin_min = bin_i == 1 ? 0f0 : bins_max[prediction_i][bin_i-1]
      bin_max = bins_max[prediction_i][bin_i]

      # mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      print("$(Float32(bin_min)),$(Float32(mean_y)),")
    end
    println()
    for prediction_i in 1:nmodels
      # Σŷ      = bins_Σŷ[prediction_i][bin_i]
      Σy      = bins_Σy[prediction_i][bin_i]
      Σweight = bins_Σweight[prediction_i][bin_i]
      bin_min = bin_i == 1 ? 0f0 : bins_max[prediction_i][bin_i-1]
      bin_max = bins_max[prediction_i][bin_i]

      # mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      print("$(Float32(bin_max)),$(Float32(mean_y)),")
    end
    println()
  end
end

reliability_curves_stairstep(20, X, Ys["tornado"], weights, Forecasts.inventory(validation_forecasts_blurred[1]))

# ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,ŷ_TORPROB:calculated:hour fcst:calculated_prob:blurred,y_TORPROB:calculated:hour fcst:calculated_prob:blurred,
# 0.0,6.120034e-6,0.0,6.1161923e-6,0.0,6.0722255e-6,0.0,6.0644675e-6,0.0,6.0613474e-6,0.0,6.0445045e-6,0.0,6.0376847e-6,0.0,6.004684e-6,0.0,6.0252123e-6,0.0,6.009672e-6,0.0,6.0188836e-6,0.0,6.009018e-6,0.0,6.0045345e-6
# 0.0001360241,6.120034e-6,0.00012906881,6.1161923e-6,0.00013945138,6.0722255e-6,0.00013743874,6.0644675e-6,0.00014626162,6.0613474e-6,0.00014990816,6.0445045e-6,0.00015490079,6.0376847e-6,0.00017161363,6.004684e-6,0.00014289253,6.0252123e-6,0.00015699367,6.009672e-6,0.00017310673,6.0188836e-6,0.00017243491,6.009018e-6,0.00015733525,6.0045345e-6
# 0.0001360241,0.0002137477,0.00012906881,0.00020788572,0.00013945138,0.00023179823,0.00013743874,0.00023936332,0.00014626162,0.0002466926,0.00014990816,0.00026644042,0.00015490079,0.00026808435,0.00017161363,0.0003019938,0.00014289253,0.00027533987,0.00015699367,0.00029678908,0.00017310673,0.000293166,0.00017243491,0.00030945576,0.00015733525,0.00029663634
# 0.00039147332,0.0002137477,0.00040239556,0.00020788572,0.00046305673,0.00023179823,0.00045061333,0.00023936332,0.00045350858,0.0002466926,0.00043706165,0.00026644042,0.0004632729,0.00026808435,0.0005288409,0.0003019938,0.000486511,0.00027533987,0.00049671443,0.00029678908,0.0004906886,0.000293166,0.00048558583,0.00030945576,0.00048746474,0.00029663634
# 0.00039147332,0.0004993167,0.00040239556,0.0004961985,0.00046305673,0.0006096289,0.00045061333,0.00061898335,0.00045350858,0.00060985726,0.00043706165,0.00062225177,0.0004632729,0.00065888464,0.0005288409,0.0007767707,0.000486511,0.00070618634,0.00049671443,0.000700952,0.0004906886,0.0007103708,0.00048558583,0.00070423563,0.00048746474,0.0007362142
# 0.0007924319,0.0004993167,0.00086976564,0.0004961985,0.0009455887,0.0006096289,0.0009184299,0.00061898335,0.00092305953,0.00060985726,0.00091092684,0.00062225177,0.000946135,0.00065888464,0.0010451098,0.0007767707,0.001003469,0.00070618634,0.0010572267,0.000700952,0.00096127833,0.0007103708,0.0009918495,0.00070423563,0.0010206574,0.0007362142
# 0.0007924319,0.0009486528,0.00086976564,0.0010045904,0.0009455887,0.0011671513,0.0009184299,0.0010931459,0.00092305953,0.0010882297,0.00091092684,0.0011748137,0.000946135,0.0012489078,0.0010451098,0.0013720464,0.001003469,0.0013343142,0.0010572267,0.0014923833,0.00096127833,0.0012819029,0.0009918495,0.0013015767,0.0010206574,0.0014777698
# 0.0013110752,0.0009486528,0.0014658585,0.0010045904,0.0015486027,0.0011671513,0.0015691228,0.0010931459,0.0015784178,0.0010882297,0.00155073,0.0011748137,0.001577702,0.0012489078,0.0017479623,0.0013720464,0.001655089,0.0013343142,0.0016931997,0.0014923833,0.0015911746,0.0012819029,0.0016741612,0.0013015767,0.0016885459,0.0014777698
# 0.0013110752,0.0015736056,0.0014658585,0.0016897048,0.0015486027,0.001754813,0.0015691228,0.001828795,0.0015784178,0.0017857623,0.00155073,0.0019575842,0.001577702,0.0019952394,0.0017479623,0.0023284913,0.001655089,0.0020241858,0.0016931997,0.0021273622,0.0015911746,0.0020406519,0.0016741612,0.0022229147,0.0016885459,0.0023369417
# 0.001929298,0.0015736056,0.0021741204,0.0016897048,0.0023376262,0.001754813,0.0023607747,0.001828795,0.0024095038,0.0017857623,0.0023385258,0.0019575842,0.0023776456,0.0019952394,0.0025600037,0.0023284913,0.0024996852,0.0020241858,0.0025590856,0.0021273622,0.0023825644,0.0020406519,0.002472485,0.0022229147,0.0025259834,0.0023369417
# 0.001929298,0.0021070794,0.0021741204,0.002333274,0.0023376262,0.0025598272,0.0023607747,0.0026622622,0.0024095038,0.0026732483,0.0023385258,0.0029058168,0.0023776456,0.002945247,0.0025600037,0.0031297735,0.0024996852,0.0029695428,0.0025590856,0.003202374,0.0023825644,0.0028348353,0.002472485,0.003040552,0.0025259834,0.003159947
# 0.0027584978,0.0021070794,0.003099869,0.002333274,0.0033160716,0.0025598272,0.0033493657,0.0026622622,0.0034447552,0.0026732483,0.0032927012,0.0029058168,0.0033508553,0.002945247,0.003610894,0.0031297735,0.0035281149,0.0029695428,0.0035896488,0.003202374,0.0034171129,0.0028348353,0.0035155292,0.003040552,0.0036488005,0.003159947
# 0.0027584978,0.002869047,0.003099869,0.0032256448,0.0033160716,0.0037251713,0.0033493657,0.0038054243,0.0034447552,0.0040506776,0.0032927012,0.0037498667,0.0033508553,0.0038500745,0.003610894,0.0043680132,0.0035281149,0.0041540083,0.0035896488,0.004322953,0.0034171129,0.0037712664,0.0035155292,0.0042985324,0.0036488005,0.0044003646
# 0.0038070749,0.002869047,0.0042618793,0.0032256448,0.0044450676,0.0037251713,0.004527556,0.0038054243,0.0046047834,0.0040506776,0.004544588,0.0037498667,0.004604337,0.0038500745,0.004865581,0.0043680132,0.004752259,0.0041540083,0.0048664864,0.004322953,0.0047767297,0.0037712664,0.004760961,0.0042985324,0.005043538,0.0044003646
# 0.0038070749,0.0040819156,0.0042618793,0.004549683,0.0044450676,0.004909013,0.004527556,0.0050787707,0.0046047834,0.0053077983,0.004544588,0.0050470857,0.004604337,0.005250627,0.004865581,0.0055256686,0.004752259,0.005372366,0.0048664864,0.005562412,0.0047767297,0.005498165,0.004760961,0.0056530093,0.005043538,0.005752358
# 0.005012274,0.0040819156,0.0056207157,0.004549683,0.005818216,0.004909013,0.005963363,0.0050787707,0.006013583,0.0053077983,0.0060814493,0.0050470857,0.0060889656,0.005250627,0.006445844,0.0055256686,0.00627381,0.005372366,0.0064653032,0.005562412,0.006327279,0.005498165,0.006289906,0.0056530093,0.0068110833,0.005752358
# 0.005012274,0.005528794,0.0056207157,0.005734663,0.005818216,0.0064403014,0.005963363,0.0064446824,0.006013583,0.006915704,0.0060814493,0.006532085,0.0060889656,0.006277618,0.006445844,0.0070996718,0.00627381,0.0066002794,0.0064653032,0.0074426867,0.006327279,0.0066196662,0.006289906,0.0067684185,0.0068110833,0.0072833244
# 0.0064020324,0.005528794,0.007333001,0.005734663,0.00745745,0.0064403014,0.0077767167,0.0064446824,0.0076962207,0.006915704,0.007994232,0.006532085,0.008071004,0.006277618,0.008388008,0.0070996718,0.008252666,0.0066002794,0.008350283,0.0074426867,0.008390352,0.0066196662,0.008357476,0.0067684185,0.009059567,0.0072833244
# 0.0064020324,0.0071754074,0.007333001,0.007484331,0.00745745,0.008291551,0.0077767167,0.008439885,0.0076962207,0.008548667,0.007994232,0.00811078,0.008071004,0.00836146,0.008388008,0.008817589,0.008252666,0.008674709,0.008350283,0.00965409,0.008390352,0.008579258,0.008357476,0.008718449,0.009059567,0.009662776
# 0.0080511365,0.0071754074,0.009378783,0.007484331,0.009406076,0.008291551,0.0099715255,0.008439885,0.009802246,0.008548667,0.010471421,0.00811078,0.010457789,0.00836146,0.010854301,0.008817589,0.010644383,0.008674709,0.010596573,0.00965409,0.010961007,0.008579258,0.010964822,0.008718449,0.011743862,0.009662776
# 0.0080511365,0.009058032,0.009378783,0.010092138,0.009406076,0.0102410745,0.0099715255,0.010897755,0.009802246,0.011152641,0.010471421,0.010435589,0.010457789,0.010493183,0.010854301,0.012680496,0.010644383,0.0119652925,0.010596573,0.01204024,0.010961007,0.011695713,0.010964822,0.012081518,0.011743862,0.013830291
# 0.0100304885,0.009058032,0.011657525,0.010092138,0.011791943,0.0102410745,0.012572419,0.010897755,0.012260336,0.011152641,0.01357872,0.010435589,0.01347677,0.010493183,0.013482881,0.012680496,0.013343003,0.0119652925,0.013339128,0.01204024,0.013954833,0.011695713,0.013961726,0.012081518,0.014591732,0.013830291
# 0.0100304885,0.011412926,0.011657525,0.012649613,0.011791943,0.012527924,0.012572419,0.013562863,0.012260336,0.0139883,0.01357872,0.013793615,0.01347677,0.014133178,0.013482881,0.015928805,0.013343003,0.016252447,0.013339128,0.015799673,0.013954833,0.01569474,0.013961726,0.015716536,0.014591732,0.018398592
# 0.012422994,0.011412926,0.014299933,0.012649613,0.014761484,0.012527924,0.015759228,0.013562863,0.015200004,0.0139883,0.017349394,0.013793615,0.016963195,0.014133178,0.016596135,0.015928805,0.016344376,0.016252447,0.016526751,0.015799673,0.017400928,0.01569474,0.017547589,0.015716536,0.017682679,0.018398592
# 0.012422994,0.013756386,0.014299933,0.015283354,0.014761484,0.016368145,0.015759228,0.018238394,0.015200004,0.017906647,0.017349394,0.018593326,0.016963195,0.017590888,0.016596135,0.018839296,0.016344376,0.020447444,0.016526751,0.020076362,0.017400928,0.01912934,0.017547589,0.01908938,0.017682679,0.021057948
# 0.015511306,0.013756386,0.017460383,0.015283354,0.018183492,0.016368145,0.019341461,0.018238394,0.01864708,0.017906647,0.021770358,0.018593326,0.021281624,0.017590888,0.02052781,0.018839296,0.019860737,0.020447444,0.020355813,0.020076362,0.02175699,0.01912934,0.022154383,0.01908938,0.021562288,0.021057948
# 0.015511306,0.01767274,0.017460383,0.01872703,0.018183492,0.020302251,0.019341461,0.02183127,0.01864708,0.021851875,0.021770358,0.025196848,0.021281624,0.02377781,0.02052781,0.02407947,0.019860737,0.024142258,0.020355813,0.024071993,0.02175699,0.025542254,0.022154383,0.024500115,0.021562288,0.025226573
# 0.019263353,0.01767274,0.021233799,0.01872703,0.02238655,0.020302251,0.023920009,0.02183127,0.02293894,0.021851875,0.026787786,0.025196848,0.026227398,0.02377781,0.025216933,0.02407947,0.024345811,0.024142258,0.025261737,0.024071993,0.026812486,0.025542254,0.027893249,0.024500115,0.026383128,0.025226573
# 0.019263353,0.022769336,0.021233799,0.022678787,0.02238655,0.025278553,0.023920009,0.027705675,0.02293894,0.026689522,0.026787786,0.031097023,0.026227398,0.031130375,0.025216933,0.029027393,0.024345811,0.03014816,0.025261737,0.029200692,0.026812486,0.033160675,0.027893249,0.03196464,0.026383128,0.028913405
# 0.023930725,0.022769336,0.025838397,0.022678787,0.02770695,0.025278553,0.029577652,0.027705675,0.028443903,0.026689522,0.032931134,0.031097023,0.03220051,0.031130375,0.03116849,0.029027393,0.029962398,0.03014816,0.031739525,0.029200692,0.03284496,0.033160675,0.034863174,0.03196464,0.03284894,0.028913405
# 0.023930725,0.027868807,0.025838397,0.02682279,0.02770695,0.030379916,0.029577652,0.03519075,0.028443903,0.03362788,0.032931134,0.038114015,0.03220051,0.03789979,0.03116849,0.036490936,0.029962398,0.035476074,0.031739525,0.039884232,0.03284496,0.03878158,0.034863174,0.038650155,0.03284894,0.03251899
# 0.030413788,0.027868807,0.031970896,0.02682279,0.034911487,0.030379916,0.036694016,0.03519075,0.03549115,0.03362788,0.04083188,0.038114015,0.04025146,0.03789979,0.038545437,0.036490936,0.037895657,0.035476074,0.039461866,0.039884232,0.040913984,0.03878158,0.04410225,0.038650155,0.04242313,0.03251899
# 0.030413788,0.03761668,0.031970896,0.032784514,0.034911487,0.039263822,0.036694016,0.04330684,0.03549115,0.04176445,0.04083188,0.04547895,0.04025146,0.048522286,0.038545437,0.04344247,0.037895657,0.048290495,0.039461866,0.051041424,0.040913984,0.048997726,0.04410225,0.05515053,0.04242313,0.048398882
# 0.038834836,0.03761668,0.040533602,0.032784514,0.04466647,0.039263822,0.046314873,0.04330684,0.045126423,0.04176445,0.05219342,0.04547895,0.051288165,0.048522286,0.048693687,0.04344247,0.048343994,0.048290495,0.049606282,0.051041424,0.051846378,0.048997726,0.055262726,0.05515053,0.054029245,0.048398882
# 0.038834836,0.04849398,0.040533602,0.04680432,0.04466647,0.052853018,0.046314873,0.054101817,0.045126423,0.053171977,0.05219342,0.058713254,0.051288165,0.0653264,0.048693687,0.055935655,0.048343994,0.06379676,0.049606282,0.066310965,0.051846378,0.06413248,0.055262726,0.07354862,0.054029245,0.073836155
# 0.052334726,0.04849398,0.0515375,0.04680432,0.059061613,0.052853018,0.060666546,0.054101817,0.059502676,0.053171977,0.06953868,0.058713254,0.067302816,0.0653264,0.0639076,0.055935655,0.06369451,0.06379676,0.0641242,0.066310965,0.06824921,0.06413248,0.07148566,0.07354862,0.06852642,0.073836155
# 0.052334726,0.070569254,0.0515375,0.0621787,0.059061613,0.078878604,0.060666546,0.07963571,0.059502676,0.07669584,0.06953868,0.090768866,0.067302816,0.09235883,0.0639076,0.08649645,0.06369451,0.08971998,0.0641242,0.09018853,0.06824921,0.09611546,0.07148566,0.10104343,0.06852642,0.10680757
# 0.07702938,0.070569254,0.069507495,0.0621787,0.08159043,0.078878604,0.08293914,0.07963571,0.08312293,0.07669584,0.097221866,0.090768866,0.09576751,0.09235883,0.08801432,0.08649645,0.09116121,0.08971998,0.0892239,0.09018853,0.096545584,0.09611546,0.100995034,0.10104343,0.09044987,0.10680757
# 0.07702938,0.119701274,0.069507495,0.09756376,0.08159043,0.121792644,0.08293914,0.113481656,0.08312293,0.109648675,0.097221866,0.15217575,0.09576751,0.1646441,0.08801432,0.1353726,0.09116121,0.15017122,0.0892239,0.1462048,0.096545584,0.17255092,0.100995034,0.18100062,0.09044987,0.15076944
# 1.0,0.119701274,1.0,0.09756376,1.0,0.121792644,1.0,0.113481656,1.0,0.109648675,1.0,0.15217575,1.0,0.1646441,1.0,0.1353726,1.0,0.15017122,1.0,0.1462048,1.0,0.17255092,1.0,0.18100062,1.0,0.15076944



function reliability_curves_midpoints(bin_count, X, y, weights, inventory)
  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

  nmodels = size(X, 2)

  bins_Σŷ      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σy      = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_Σweight = map(_ -> zeros(Float64, bin_count), 1:nmodels)
  bins_max     = map(_ -> ones(Float32, bin_count), 1:nmodels)

  for prediction_i in 1:nmodels
    ŷ              = @view X[:,prediction_i]; # HREF prediction for event_name
    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_i = 1
    for i in eachindex(y_sorted)
      if ŷ_sorted[i] > bins_max[prediction_i][bin_i]
        bin_i += 1
      end

      bins_Σŷ[prediction_i][bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
      bins_Σy[prediction_i][bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
      bins_Σweight[prediction_i][bin_i] += Float64(weights_sorted[i])

      if bins_Σy[prediction_i][bin_i] >= per_bin_pos_weight
        bins_max[prediction_i][bin_i] = ŷ_sorted[i]
      end
    end
  end

  for prediction_i in 1:nmodels
    print("ŷ_$(Inventories.inventory_line_description(inventory[prediction_i])),y_$(Inventories.inventory_line_description(inventory[prediction_i])),")
  end
  println()

  for bin_i in 1:bin_count
    for prediction_i in 1:nmodels
      Σŷ      = bins_Σŷ[prediction_i][bin_i]
      Σy      = bins_Σy[prediction_i][bin_i]
      Σweight = bins_Σweight[prediction_i][bin_i]

      mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      print("$(Float32(mean_ŷ)),$(Float32(mean_y)),")
    end
    println()
  end
end

reliability_curves_midpoints(20, X, Ys["tornado"], weights, Forecasts.inventory(validation_forecasts_blurred[1]))


import LogisticRegression

σ(x)     = 1f0 / (1f0 + exp(-x))
logit(p) = log(p / (1f0 - p))

function platt_calibrate(x, y, weights)
  # nontrivial = x .>= 0.0001f0
  # X = reshape(logit.(x[nontrivial]), (count(nontrivial),1))
  # a, b = LogisticRegression.fit(X, y[nontrivial], weights[nontrivial])
  X = reshape(logit.(x), (length(x),1))
  a, b = LogisticRegression.fit(X, y, weights)
  a, b
end

function apply_platt_calibrate(x, a, b; out = Vector{eltype(x)}(undef, length(x)))
  Threads.@threads for i in eachindex(out)
    out[i] = σ(logit(x[i]) * a + b)
  end
  out
end

X_platted = Array{Float32}(undef, size(X))
nmodels = size(X, 2)
for prediction_i in 1:nmodels
  ŷ = @view X[:,prediction_i] # HREF prediction for event_name
  y = Ys["tornado"]
  a, b = platt_calibrate(ŷ, y, weights)
  apply_platt_calibrate(ŷ, a, b; out = @view X_platted[:,prediction_i])
  println("$prediction_i, $a, $b")
end

reliability_curves_stairstep(20, X_platted, Ys["tornado"], weights, Forecasts.inventory(validation_forecasts_blurred[1]))

function bss(ŷ, y, weights)
  total_weight = sum(weights)
  ŷ_baseline = sum(y .* weights) / total_weight

  bs_baseline = sum((ŷ_baseline .- y).^2f0) / total_weight
  bs          = sum((ŷ .- y).^2f0)          / total_weight

  1 - bs / bs_baseline
end

for prediction_i in 1:nmodels
  ŷ           = @view X[:,prediction_i]
  ŷ_platted   = @view X_platted[:,prediction_i]
  y           = Ys["tornado"]
  bss_uncalib = bss(ŷ, y, weights)
  bss_platted = bss(ŷ_platted, y, weights)

  println("$prediction_i, $bss_uncalib, $bss_platted")
end


reliability_curves_midpoints(20, X_platted, Ys["tornado"], weights, Forecasts.inventory(validation_forecasts_blurred[1]))


# To make HREF-only day 2 predictions, we need to calibrate the hourlies

const bin_count = 6

function find_ŷ_bin_splits(event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

  ŷ              = @view X[:,prediction_i]; # HREF prediction for event_name
  sort_perm      = Metrics.parallel_sort_perm(ŷ);
  y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
  ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
  weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

  bins_Σŷ      = zeros(Float64, bin_count)
  bins_Σy      = zeros(Float64, bin_count)
  bins_Σweight = zeros(Float64, bin_count)
  bins_max     = ones(Float32, bin_count)

  bin_i = 1
  for i in 1:length(y_sorted)
    if ŷ_sorted[i] > bins_max[bin_i]
      bin_i += 1
    end

    bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
    bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
    bins_Σweight[bin_i] += Float64(weights_sorted[i])

    if bins_Σy[bin_i] >= per_bin_pos_weight
      bins_max[bin_i] = ŷ_sorted[i]
    end
  end

  println("event_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for bin_i in 1:bin_count
    Σŷ      = bins_Σŷ[bin_i]
    Σy      = bins_Σy[bin_i]
    Σweight = bins_Σweight[bin_i]

    mean_ŷ = Σŷ / Σweight
    mean_y = Σy / Σweight

    println("$event_name\t$mean_y\t$mean_ŷ\t$Σweight\t$(bins_max[bin_i])")
  end

  bins_max
end

event_to_bins = Dict{String,Vector{Float32}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  (model_name, _, _, _, _) = HREFPredictionAblations.models[prediction_i]

  event_to_bins[model_name] = find_ŷ_bin_splits("tornado", prediction_i, X, Ys, weights)
end

# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9546283460263947e-5   2.2912671045334642e-5   6.03312368404197e8      0.000958296
# tornado 0.0016891992369654878   0.0017934076373789863   6.981415401887536e6     0.0034187553
# tornado 0.004948037217402474    0.005168478938572997    2.3832744669790864e6    0.008049908
# tornado 0.011424145437805936    0.011350910662165814    1.0322749166604877e6    0.016707344
# tornado 0.024844464814848657    0.02355134374673053     474665.4468984604       0.035788268
# tornado 0.06534440616601209     0.05845152755317162     180423.58301466703      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.952048753995379e-5    2.2570578951849063e-5   6.041016966930331e8     0.0010611907
# tornado 0.0018286788128066543   0.0019994760061401297   6.448962287696362e6     0.0038441075
# tornado 0.005365929945839972    0.005917378442814821    2.19770114856863e6      0.009374168
# tornado 0.012686362642379712    0.013058873879137041    929561.3617351651       0.018635446
# tornado 0.024267615193777164    0.025674283979364002    485961.8814621568       0.037465766
# tornado 0.05878834718031695     0.05659407890493848     200538.84672176838      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9493034602905252e-5   2.1351372578408005e-5   6.0498466493152e8       0.0011344727
# tornado 0.0020002697472507975   0.002128917861310692    5.895731727658629e6     0.0040516052
# tornado 0.005961920977041346    0.006084624440286901    1.9780237764064074e6    0.009403912
# tornado 0.013030905760816591    0.013293301519597801    904971.3572544456       0.019440345
# tornado 0.027329997389006273    0.02727398169725481     431487.2761924863       0.04088651
# tornado 0.06953617201611069     0.06434524655072116     169543.15073412657      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.948746887976373e-5    2.1127070318148625e-5   6.051342292721803e8     0.001112896
# tornado 0.002003938521911659    0.0021148570755995634   5.884751990515292e6     0.0040969327
# tornado 0.006039798968851563    0.006283556698721366    1.9525058577931523e6    0.009968531
# tornado 0.014099923794318948    0.014121696905084219    836369.7399896383       0.020722248
# tornado 0.030328602452665453    0.028818927140861303    388836.2492079139       0.042520937
# tornado 0.07029083024948092     0.06638864093438278     167729.11018127203      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.948467974793234e-5    2.1283149571755178e-5   6.052143890412238e8     0.0011188483
# tornado 0.0020012635063782033   0.002138303087845265    5.892686175814688e6     0.0041876957
# tornado 0.006341460801589016    0.006313960179293873    1.8596584796265364e6    0.009798439
# tornado 0.01433325413152705     0.01372844026739828     822733.639330864        0.019887649
# tornado 0.0291998548763413      0.027854887919160996    403877.41673892736      0.04154461
# tornado 0.06891378924912535     0.06647958599945804     171077.4679558277       1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9472597309671077e-5   2.0779565167775547e-5   6.055963373664137e8     0.0011116201
# tornado 0.00215600289282287     0.0021159211405857793   5.469758782774508e6     0.004092531
# tornado 0.006001807099580369    0.006447203411139815    1.9647856851531267e6    0.010467994
# tornado 0.014063746089748948    0.015312099139766175    838487.3090659976       0.023395522
# tornado 0.03382950981092436     0.032560044893616       348592.064255774        0.047907665
# tornado 0.08050470964413829     0.07440538512599332     146461.01205140352      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9467270263306555e-5   2.1024433282619958e-5   6.057776749521786e8     0.001142257
# tornado 0.0022155410541996974   0.002164200716874335    5.322620295134962e6     0.004164435
# tornado 0.006103299228173797    0.006509063931231617    1.9322305209964514e6    0.0104533415
# tornado 0.013948629631993906    0.0151837540080779      845415.9522585273       0.02289575
# tornado 0.03367731666068475     0.03171801522588309     350182.0657916665       0.0471496
# tornado 0.08649767521065985     0.07344672887148468     136298.43424993753      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9443193384695918e-5   2.056610038730587e-5    6.065345226027207e8     0.001259957
# tornado 0.002472666316966537    0.002345205792965128    4.76918044762063e6      0.004403745
# tornado 0.00657427976029817     0.006821190774966018    1.7937452619232535e6    0.010849744
# tornado 0.015878771893361742    0.015195404705541685    742643.8353065848       0.022011435
# tornado 0.0320217103577533      0.03062640920572916     368281.1955651045       0.044833306
# tornado 0.07555310753219195     0.06903765604997823     156048.87840247154      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.945506542826972e-5    1.9350555019745457e-5   6.061307137429807e8     0.0012080052
# tornado 0.0022965157557417624   0.0022751290210653117   5.135064444976687e6     0.0043216683
# tornado 0.006351641815409172    0.006684114408690142    1.856621015407741e6     0.010642204
# tornado 0.015988982721514763    0.014760814075321925    737552.4464214444       0.021209253
# tornado 0.03234925028949919     0.029633021105958238    364543.9477273226       0.044587743
# tornado 0.08426051260694865     0.07090625525839882     139926.62393420935      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9438735761157884e-5   1.971266591309856e-5    6.06649102850286e8      0.0012621598
# tornado 0.0024826052739416      0.002343514717738839    4.750198672922969e6     0.004405628
# tornado 0.0068254214978517415   0.006743157189626929    1.727754718410015e6     0.01059343
# tornado 0.015805989811441302    0.014914108437707954    746109.162173748        0.021830112
# tornado 0.033306021319186065    0.030687729794326903    354083.55564296246      0.046009377
# tornado 0.08594433047576132     0.07222145106897757     137173.26255828142      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9461630387168764e-5   2.062539200104247e-5    6.059461806390712e8     0.0011541534
# tornado 0.002206239044412362    0.002217468457911261    5.345209844344497e6     0.004339112
# tornado 0.00641272374788433     0.006798643279769065    1.8389220087162852e6    0.010959722
# tornado 0.0154512622355201      0.01567778150879449     763237.6720257998       0.023348574
# tornado 0.035000407070519435    0.032471321015563305    336924.1686872244       0.04785411
# tornado 0.08801861815682034     0.07358855962799962     133947.88783818483      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9447402329091983e-5   2.0623531831039144e-5   6.063973414912629e8     0.001206508
# tornado 0.002379261221684243    0.002267531650985861    4.956321831174552e6     0.004330907
# tornado 0.0065735720776828425   0.0067838507068722345   1.7939356570031643e6    0.0109639205
# tornado 0.015611282233925617    0.015856140511686115    755410.3557307124       0.023913587
# tornado 0.03472199009267542     0.033981682203900854    339633.2417806387       0.051235575
# tornado 0.09681353277533476     0.07893512276464525     121779.64333802462      1.0
# event_name      mean_y  mean_ŷ  Σweight bin_max
# tornado 1.9433767699185833e-5   2.0002769098688164e-5   6.06803381293335e8      0.0012153604
# tornado 0.0025180521817733475   0.0023336320664977006   4.683162616125584e6     0.004538168
# tornado 0.006885780923916211    0.007216856966735946    1.7126045786351562e6    0.011742671
# tornado 0.017711000447631936    0.0162420525903571      665834.9625293612       0.023059275
# tornado 0.03148117549867956     0.03282356196542077     374601.9258365035       0.049979284
# tornado 0.09444591873851253     0.0749646351119608      124836.84526234865      1.0

println(event_to_bins)
# Dict{String, Vector{Float32}}("tornadao_mean_prob_computed_partial_climatology_227" => [0.0011116201, 0.004092531, 0.010467994, 0.023395522, 0.047907665, 1.0], "tornadao_mean_58" => [0.000958296, 0.0034187553, 0.008049908, 0.016707344, 0.035788268, 1.0], "tornadao_mean_prob_computed_climatology_prior_next_hrs_691" => [0.0011541534, 0.004339112, 0.010959722, 0.023348574, 0.04785411, 1.0], "tornadao_prob_80" => [0.0010611907, 0.0038441075, 0.009374168, 0.018635446, 0.037465766, 1.0], "tornado_full_13831" => [0.0012153604, 0.004538168, 0.011742671, 0.023059275, 0.049979284, 1.0], "tornadao_mean_prob_computed_no_sv_219" => [0.001112896, 0.0040969327, 0.009968531, 0.020722248, 0.042520937, 1.0], "tornadao_mean_prob_computed_220" => [0.0011188483, 0.0041876957, 0.009798439, 0.019887649, 0.04154461, 1.0], "tornadao_mean_prob_computed_climatology_253" => [0.001142257, 0.004164435, 0.0104533415, 0.02289575, 0.0471496, 1.0], "tornadao_mean_prob_138" => [0.0011344727, 0.0040516052, 0.009403912, 0.019440345, 0.04088651, 1.0], "tornadao_mean_prob_computed_climatology_grads_1348" => [0.0012080052, 0.0043216683, 0.010642204, 0.021209253, 0.044587743, 1.0], "tornadao_mean_prob_computed_climatology_blurs_910" => [0.001259957, 0.004403745, 0.010849744, 0.022011435, 0.044833306, 1.0], "tornadao_mean_prob_computed_climatology_blurs_grads_2005" => [0.0012621598, 0.004405628, 0.01059343, 0.021830112, 0.046009377, 1.0], "tornadao_mean_prob_computed_climatology_3hr_1567" => [0.001206508, 0.004330907, 0.0109639205, 0.023913587, 0.051235575, 1.0])



# 4. combine bin-pairs (overlapping, 5 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + b)

import LogisticRegression

const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

function find_logistic_coeffs(model_name, event_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,prediction_i]; # HREF prediction for event_name

  bins_max = event_to_bins[model_name]
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_href_x  = X[bin_members, prediction_i]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 1))

    Threads.@threads for i in 1:length(bin_y)
      logit_href = logit(bin_href_x[i])

      bin_X_features[i,1] = logit_href
      # bin_X_features[i,3] = bin_X[i,1]*bin_X[i,2]
      # bin_X_features[i,3] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,4] = logit(bin_X[i,1]*bin_X[i,2])
      # bin_X_features[i,5] = max(logit_href, logit_sref)
      # bin_X_features[i,6] = min(logit_href, logit_sref)
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("model_name", model_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("HREF_ŷ_min", bin_min),
      ("HREF_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_HREF_ŷ", sum(bin_href_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("HREF_logloss", sum(logloss.(bin_y, bin_href_x) .* bin_weights) / bin_weight),
      ("HREF_au_pr", Metrics.area_under_pr_curve(bin_href_x, bin_y, bin_weights)),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights)),
      ("logistic_coeffs", coeffs)
    ]

    headers = map(first, stuff)
    row     = map(last, stuff)

    bin_i == 1 && println(join(headers, "\t"))
    println(join(row, "\t"))

    push!(bins_logistic_coeffs, coeffs)
  end

  bins_logistic_coeffs
end

event_to_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  (model_name, _, _, _, _) = HREFPredictionAblations.models[prediction_i]

  event_to_bins_logistic_coeffs[model_name] = find_logistic_coeffs(model_name, "tornado", prediction_i, X, Ys, weights)
end

# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_58                                           1-2 -1.0         0.0034187553 662236238 25438.0   6.1029376e8 4.3166132e-5 3.864617e-5  0.0003279372802231187  0.0016775245143762633 3.864618e-5     0.000327510748967054   0.0016506126514052018 Float32[1.0608453,  0.34233692]
# tornadao_mean_58                                           2-3 0.000958296  0.008049908  10048574  25222.0   9.36469e6   0.0026523492 0.0025185598 0.01711929991296464    0.005106269049166281  0.0025185589    0.017115796323922187   0.0050992929674585985 Float32[1.0113722,  0.013388185]
# tornadao_mean_58                                           3-4 0.0034187553 0.016707344  3644304   25069.0   3.4155495e6 0.007036983  0.0069052987 0.04044299210900128    0.011468777336170312  0.0069053015    0.04043871606638279    0.011468742248978493  Float32[1.0653579,  0.29739413]
# tornadao_mean_58                                           4-5 0.008049908  0.035788268  1595068   25002.0   1.5069402e6 0.015193881  0.015651362  0.07908171944822744    0.02539244358809829   0.015651366     0.07907204113769895    0.02539244412770883   Float32[1.0437187,  0.20861574]
# tornadao_mean_58                                           5-6 0.016707344  1.0          687698    24786.0   655089.0    0.0331635    0.035998896  0.14872481455202125    0.08134661226969421   0.035998903     0.1485787155962834     0.08134660809672188   Float32[1.0588217,  0.27578637]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_prob_80                                           1-2 -1.0         0.0038441075 662535488 25431.0   6.1055066e8 4.345171e-5  3.8629783e-5 0.00032712932627822704 0.0018008679335613333 3.8629787e-5    0.0003268462923201525  0.0017943022340260058 Float32[1.0096253,  -0.0469623]
# tornadao_prob_80                                           2-3 0.0010611907 0.009374168  9230973   25204.0   8.646664e6  0.0029952796 0.002727733  0.01833380490596757    0.005496938345391945  0.002727733     0.01832131723479828    0.005496934170200845  Float32[0.9850988,  -0.17768726]
# tornadao_prob_80                                           3-4 0.0038441075 0.018635446  3314733   25098.0   3.1272625e6 0.008040148  0.007541888  0.04350844591814723    0.012710821220841086  0.007541886     0.04348910342048436    0.012710812551149883  Float32[1.0664655,  0.24821338]
# tornadao_prob_80                                           4-5 0.009374168  0.037465766  1490204   24995.0   1.4155232e6 0.017389856  0.016662309  0.08360079167813393    0.024440506654733924  0.016662313     0.08358195277532174    0.024440512069968088  Float32[0.9513827,  -0.2362276]
# tornadao_prob_80                                           5-6 0.018635446  1.0          718019    24764.0   686500.7    0.0347065    0.03435173   0.14475956926170394    0.06982947289434904   0.034351725     0.14472474729785914    0.06982943797926351   Float32[1.0854785,  0.2633485]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_138                                     1-2 -1.0         0.0040516052 662875450 25405.0   6.108804e8  4.169193e-5  3.8609916e-5 0.00032127922476961974 0.00204620120910871   3.860992e-5     0.00032113491079358204 0.002046236195476628  Float32[1.0230768,  0.09093034]
# tornadao_mean_prob_138                                     2-3 0.0011344727 0.009403912  8424959   25208.0   7.8737555e6 0.0031226599 0.0029955052 0.0198293190351229     0.006074786807123417  0.0029955066    0.019826231608007568   0.006074768821940664  Float32[1.0292343,  0.12141731]
# tornadao_mean_prob_138                                     3-4 0.0040516052 0.019440345  3063375   25099.0   2.882995e6  0.008347427  0.0081808735 0.04661923923901055    0.01347688187995983   0.008180876     0.0466174231305052     0.013476885694353699  Float32[1.0123438,  0.037362378]
# tornadao_mean_prob_138                                     4-5 0.009403912  0.04088651   1409765   25006.0   1.3364586e6 0.017807085  0.017647492  0.08709269222682331    0.028153102935692034  0.017647494     0.0870900407986672     0.028153097713510856  Float32[1.0350122,  0.1281267]
# tornadao_mean_prob_138                                     5-6 0.019440345  1.0          629415    24789.0   601030.44   0.03773132   0.039235827  0.1588687761418238     0.08830645602859426   0.03923583      0.15878022975297934    0.08830645021128024   Float32[1.0963296,  0.3398288]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_no_sv_219                      1-2 -1.0         0.0040969327 663018933 25375.0   6.1101894e8 4.1291885e-5 3.859981e-5  0.00032138160620669533 0.0020591077398719225 3.8599796e-5    0.0003212230088741455  0.0020497884475417566 Float32[1.0354019,  0.1911897]
# tornadao_mean_prob_computed_no_sv_219                      2-3 0.001112896  0.009968531  8388707   25179.0   7.837258e6  0.0031534103 0.0030093973 0.01987441558611732    0.006166935928254792  0.0030093987    0.019870787016576113   0.006166931432952669  Float32[1.0219095,  0.07485228]
# tornadao_mean_prob_computed_no_sv_219                      3-4 0.0040969327 0.020722248  2966114   25095.0   2.7888755e6 0.008634175  0.00845699   0.04778305138740547    0.014536187893856495  0.008456991     0.0477789745846244     0.014536159414052733  Float32[1.0493485,  0.2075882]
# tornadao_mean_prob_computed_no_sv_219                      4-5 0.009968531  0.042520937  1293709   25053.0   1.225206e6  0.018786069  0.019250322  0.09315758217623905    0.03148938130266943   0.019250324     0.09313699217719332    0.03148937390686688   Float32[1.0940257,  0.38835412]
# tornadao_mean_prob_computed_no_sv_219                      5-6 0.020722248  1.0          583193    24823.0   556565.4    0.04014111   0.042371806  0.16962132024441082    0.08082243157443954   0.042371802     0.169554187020561      0.08082242431253567   Float32[1.0213909,  0.12248391]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_220                            1-2 -1.0         0.0041876957 663113128 25379.0   6.111071e8  4.169681e-5  3.8594262e-5 0.000320793491817166   0.0021321689830207536 3.8594277e-5    0.0003206266801501963  0.002132167275357274  Float32[1.0302787,  0.14280733]
# tornadao_mean_prob_computed_220                            2-3 0.0011188483 0.009798439  8298336   25169.0   7.752345e6  0.0031399736 0.0030424045 0.02001433366050727    0.006341342402958952  0.0030424043    0.0200101704303265     0.006341337002615769  Float32[1.0681692,  0.34719107]
# tornadao_mean_prob_computed_220                            3-4 0.0041876957 0.019887649  2852343   25082.0   2.682392e6  0.008588104  0.008792675  0.049420149387687115   0.01481394516850477   0.008792677     0.04941566824860986    0.014807034184406641  Float32[1.0482181,  0.24779604]
# tornadao_mean_prob_computed_220                            4-5 0.009798439  0.04154461   1295239   25048.0   1.226611e6  0.018379753  0.019228274  0.09332164879805956    0.03015176706069509   0.01922827      0.09330051945124358    0.030151756466691194  Float32[1.0291348,  0.15958983]
# tornadao_mean_prob_computed_220                            5-6 0.019887649  1.0          602769    24832.0   574954.9    0.03934764   0.04101671   0.16559922182415915    0.0776385235850777    0.04101671      0.1655568941154445     0.0776384972997189    Float32[0.9713004,  -0.044102136]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_partial_climatology_227        1-2 -1.0         0.004092531  663074592 25347.0   6.110661e8  3.9533537e-5 3.859705e-5  0.0003191858573527934  0.0021503914286219988 3.859704e-5     0.000319003282364821   0.002150426708202286  Float32[1.0562521,  0.38805947]
# tornadao_mean_prob_computed_partial_climatology_227        2-3 0.0011116201 0.010467994  7944819   25161.0   7.434544e6  0.0032605832 0.003172364  0.02087306482590535    0.006089550707324115  0.0031723639    0.020868106696795983   0.006089474662569881  Float32[0.92319447, -0.45271772]
# tornadao_mean_prob_computed_partial_climatology_227        3-4 0.004092531  0.023395522  2975069   25099.0   2.803273e6  0.009098783  0.008413215  0.04758402922915587    0.014815449229935345  0.008413215     0.04755709479569043    0.014815367454967353  Float32[0.9888554,  -0.13014041]
# tornadao_mean_prob_computed_partial_climatology_227        4-5 0.010467994  0.047907665  1251362   25030.0   1.1870794e6 0.02037705   0.019868065  0.09514342616255923    0.03327751016523551   0.019868063     0.09510241917047474    0.033277507743809366  Float32[1.1344755,  0.48079926]
# tornadao_mean_prob_computed_partial_climatology_227        5-6 0.023395522  1.0          518579    24847.0   495053.06   0.04493995   0.047638327  0.18427817825693765    0.09860787937959017   0.04763833      0.18414848997779926    0.09860786485999973   Float32[1.0812244,  0.30014265]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_253                1-2 -1.0         0.004164435  663102614 25310.0   6.111003e8  3.9691276e-5 3.8594844e-5 0.0003185660095777487  0.0022322787691518267 3.859486e-5     0.00031835276404280503 0.002232270499337158  Float32[1.0602976,  0.41283026]
# tornadao_mean_prob_computed_climatology_253                2-3 0.001142257  0.0104533415 7759419   25139.0   7.254851e6  0.0033213957 0.0032509924 0.021325326655027902   0.006190585896438221  0.0032509915    0.02132063142024078    0.006184367947386807  Float32[0.92125416, -0.4563326]
# tornadao_mean_prob_computed_climatology_253                3-4 0.004164435  0.02289575   2954074   25082.0   2.7776465e6 0.009149328  0.008491136  0.04802154093460711    0.014500650445509702  0.008491136     0.047996153824213045   0.014500652183244804  Float32[0.9710596,  -0.20777129]
# tornadao_mean_prob_computed_climatology_253                4-5 0.0104533415 0.0471496    1264004   25024.0   1.195598e6  0.02002652   0.01972702   0.09459411418912315    0.033822168990673515  0.019727018     0.09454245357329727    0.03382219813444753   Float32[1.1646894,  0.60834074]
# tornadao_mean_prob_computed_climatology_253                5-6 0.02289575   1.0          511552    24901.0   486480.5    0.04340925   0.048476126  0.18573619792367815    0.11020228340571951   0.048476122     0.1852672892608807     0.11020227846286433   Float32[1.1520658,  0.5673795]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_blurs_910          1-2 -1.0         0.004403745  663322993 25309.0   6.113037e8  3.870214e-5  3.858239e-5  0.0003139754669254639  0.0025116080533879597 3.8582388e-5    0.0003137373025382762  0.0025105967549543876 Float32[1.065771,   0.47239247]
# tornadao_mean_prob_computed_climatology_blurs_910          2-3 0.001259957  0.010849744  7010378   25103.0   6.562926e6  0.0035685587 0.003593698  0.02323656750980681    0.006774354735870697  0.0035936977    0.023233112733964632   0.0067554072719462595 Float32[0.9286899,  -0.38221243]
# tornadao_mean_prob_computed_climatology_blurs_910          3-4 0.004403745  0.022011435  2694066   25050.0   2.536389e6  0.009273125  0.009298596  0.051659505162474945   0.01582503506934771   0.009298593     0.05165631029094152    0.01582504688332089   Float32[1.0567164,  0.26145768]
# tornadao_mean_prob_computed_climatology_blurs_910          4-5 0.010849744  0.044833306  1173587   25037.0   1.110925e6  0.020310916  0.021230295  0.10108630517210025    0.0323731470751796    0.021230299     0.1010651618955714     0.032373152975580674  Float32[0.9913568,  0.01256458]
# tornadao_mean_prob_computed_climatology_blurs_910          5-6 0.022011435  1.0          551181    24934.0   524330.1    0.042058196  0.044977333  0.17678725470701953    0.09236300709860684   0.044977333     0.17663808546345494    0.09236300249770739   Float32[1.0834966,  0.3218238]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_grads_1348         1-2 -1.0         0.0043216683 663284893 25327.0   6.112658e8  3.830069e-5  3.8583985e-5 0.0003162771401849483  0.0023208255915797733 3.8583967e-5    0.00031625252010606764 0.0023208366578953655 Float32[1.0203846,  0.15487157]
# tornadao_mean_prob_computed_climatology_grads_1348         2-3 0.0012080052 0.010642204  7466746   25145.0   6.991685e6  0.0034459222 0.0033733423 0.022003040736436177   0.006472538560769513  0.0033733423    0.02199994859359966    0.006472529125985412  Float32[0.93945754, -0.35369343]
# tornadao_mean_prob_computed_climatology_grads_1348         3-4 0.0043216683 0.021209253  2754403   25079.0   2.5941735e6 0.00898041   0.0090916455 0.05060370820833725    0.01625549789516264   0.0090916455    0.05058700888663842    0.016255510930656792  Float32[1.1307863,  0.61256605]
# tornadao_mean_prob_computed_climatology_grads_1348         4-5 0.010642204  0.044587743  1162357   25043.0   1.1020964e6 0.019680142  0.021400522  0.1016040664665611     0.03374149644768925   0.021400519     0.10152589115107861    0.033741488139538395  Float32[1.0428475,  0.24978055]
# tornadao_mean_prob_computed_climatology_grads_1348         5-6 0.021209253  1.0          528944    24887.0   504470.56   0.041081112  0.046748042  0.1812230944176543     0.09975332026672865   0.046748042     0.1807688334589041     0.09975333695804696   Float32[1.0871354,  0.40001607]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   1-2 -1.0         0.004405628  663429355 25357.0   6.113993e8  3.776719e-5  3.8576032e-5 0.00031429785216913837 0.00245820322303698   3.857604e-5     0.0003142251432004988  0.002458236377339989  Float32[1.0343003,  0.2688026]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   2-3 0.0012621598 0.01059343   6916132   25132.0   6.4779535e6 0.0035169567 0.003640891  0.023469581701948038   0.00702280826429791   0.0036408915    0.02346631663109107    0.007002240885820756  Float32[0.95909494, -0.18902206]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   3-4 0.004405628  0.021830112  2624237   25054.0   2.473864e6  0.009207488  0.009533931  0.05273009054914814    0.016264261395901047  0.009533931     0.05271991857452223    0.016264252055453027  Float32[1.0664421,  0.33880964]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   4-5 0.01059343   0.046009377  1159353   25025.0   1.1001928e6 0.019990655  0.02143816   0.10150207994657089    0.03494275906316805   0.02143816      0.10144074785058343    0.034942758034021464  Float32[1.0681071,  0.33059332]
# tornadao_mean_prob_computed_climatology_blurs_grads_2005   5-6 0.021830112  1.0          514648    24882.0   491256.8    0.042285156  0.048004176  0.18501339876697287    0.10012101648176111   0.04800417      0.1845719888923512     0.10012100399008048   Float32[1.078756,   0.37049353]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 1-2 -1.0         0.004339112  663308855 25323.0   6.112914e8  3.9834867e-5 3.858309e-5  0.00031709891054001066 0.002155569640482817  3.8583068e-5    0.0003169695057196112  0.002134572333011802  Float32[1.0457013,  0.29932234]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 2-3 0.0011541534 0.010959722  7679677   25144.0   7.184132e6  0.0033901117 0.0032829726 0.021491509750556744   0.006350251107097701  0.0032829724    0.02148527229346765    0.006329640697765771  Float32[0.91751766, -0.48559842]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 3-4 0.004339112  0.023348574  2765056   25081.0   2.6021595e6 0.009402977  0.009063812  0.050580266014628644   0.01582404656224819   0.009063811     0.05057369763866885    0.015824025158005482  Float32[1.0170314,  0.0401797]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 4-5 0.010959722  0.04785411   1161788   25033.0   1.1001618e6 0.020820798  0.021438183  0.10131920633175762    0.03487588858750456   0.02143818      0.10129126801555675    0.034875879913718204  Float32[1.0982827,  0.39915916]
# tornadao_mean_prob_computed_climatology_prior_next_hrs_691 5-6 0.023348574  1.0          494329    24889.0   470872.06   0.044167846  0.050082374  0.1904964098507854     0.11627820686007802   0.050082378     0.18989239501803587    0.11627819046322736   Float32[1.1686069,  0.6304107]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornadao_mean_prob_computed_climatology_3hr_1567           1-2 -1.0         0.004330907  663373936 25305.0   6.1135366e8 3.88395e-5   3.8578713e-5 0.00031562018403785957 0.0023768518216676767 3.857872e-5     0.00031539361526278486 0.0023768506048695123 Float32[1.0639268,  0.45781243]
# tornadao_mean_prob_computed_climatology_3hr_1567           2-3 0.001206508  0.0109639205 7215556   25106.0   6.7502575e6 0.0034677798 0.0034939332 0.02267022329014175    0.006669357939037312  0.0034939335    0.022666061713047456   0.006641683821688842  Float32[0.9226182,  -0.4165027]
# tornadao_mean_prob_computed_climatology_3hr_1567           3-4 0.004330907  0.023913587  2709237   25044.0   2.549346e6  0.009472109  0.009251584  0.051479204475367864   0.01576807731555346   0.009251584     0.05147614096916684    0.015768073447982145  Float32[0.9799438,  -0.11478417]
# tornadao_mean_prob_computed_climatology_3hr_1567           4-5 0.0109639205 0.051235575  1156696   25017.0   1.0950435e6 0.021477869  0.021538567  0.1016485700571772     0.036430788000440174  0.021538567     0.10164260290653448    0.03643073289400433   Float32[1.0525097,  0.19820513]
# tornadao_mean_prob_computed_climatology_3hr_1567           5-6 0.023913587  1.0          485067    24944.0   461412.88   0.045846142  0.051109668  0.1917515134033739     0.13069280362593708   0.05110967      0.19104354034408572    0.1306927956290262    Float32[1.2263151,  0.7710571]
# model_name                                                 bin HREF_ŷ_min   HREF_ŷ_max   count     pos_count weight      mean_HREF_ŷ  mean_y       HREF_logloss           HREF_au_pr            mean_logistic_ŷ logistic_logloss       logistic_au_pr        logistic_coeffs
# tornado_full_13831                                         1-2 -1.0         0.004538168  663522521 25328.0   6.114865e8  3.7722053e-5 3.8569822e-5 0.0003142306401920053  0.0025057626862979797 3.8569815e-5    0.0003139239661282196  0.0025057810877705806 Float32[1.0710595,  0.5394862]
# tornado_full_13831                                         2-3 0.0012153604 0.011742671  6826598   25086.0   6.395767e6  0.003641221  0.003687606  0.023732974473695958   0.007081899006221959  0.003687606     0.023724181537556593   0.007080886384121605  Float32[0.89414734, -0.5616081]
# tornado_full_13831                                         3-4 0.004538168  0.023059275  2521603   25053.0   2.3784395e6 0.009743426  0.009916259  0.054313734992198      0.01738706205391112   0.009916258     0.05429968233024003    0.01738732278874145   Float32[1.108447,   0.50613326]
# tornado_full_13831                                         4-5 0.011742671  0.049979284  1096364   25059.0   1.0404369e6 0.022212109  0.022668853  0.10695685690139083    0.0329149983068956    0.022668855     0.10690160015370286    0.032915092429173884  Float32[0.83939207, -0.5758451]
# tornado_full_13831                                         5-6 0.023059275  1.0          524116    24912.0   499438.75   0.043356903  0.047219485  0.18174162725040247    0.10422838676781673   0.04721949      0.18125306291455362    0.10422838362397668   Float32[1.2171175,  0.7334359]

print("event_to_bins_logistic_coeffs = $event_to_bins_logistic_coeffs")
# event_to_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("tornadao_mean_prob_computed_partial_climatology_227" => [[1.0562521, 0.38805947], [0.92319447, -0.45271772], [0.9888554, -0.13014041], [1.1344755, 0.48079926], [1.0812244, 0.30014265]], "tornadao_mean_58" => [[1.0608453, 0.34233692], [1.0113722, 0.013388185], [1.0653579, 0.29739413], [1.0437187, 0.20861574], [1.0588217, 0.27578637]], "tornadao_mean_prob_computed_climatology_prior_next_hrs_691" => [[1.0457013, 0.29932234], [0.91751766, -0.48559842], [1.0170314, 0.0401797], [1.0982827, 0.39915916], [1.1686069, 0.6304107]], "tornadao_prob_80" => [[1.0096253, -0.0469623], [0.9850988, -0.17768726], [1.0664655, 0.24821338], [0.9513827, -0.2362276], [1.0854785, 0.2633485]], "tornado_full_13831" => [[1.0710595, 0.5394862], [0.89414734, -0.5616081], [1.108447, 0.50613326], [0.83939207, -0.5758451], [1.2171175, 0.7334359]], "tornadao_mean_prob_computed_no_sv_219" => [[1.0354019, 0.1911897], [1.0219095, 0.07485228], [1.0493485, 0.2075882], [1.0940257, 0.38835412], [1.0213909, 0.12248391]], "tornadao_mean_prob_computed_220" => [[1.0302787, 0.14280733], [1.0681692, 0.34719107], [1.0482181, 0.24779604], [1.0291348, 0.15958983], [0.9713004, -0.044102136]], "tornadao_mean_prob_computed_climatology_253" => [[1.0602976, 0.41283026], [0.92125416, -0.4563326], [0.9710596, -0.20777129], [1.1646894, 0.60834074], [1.1520658, 0.5673795]], "tornadao_mean_prob_138" => [[1.0230768, 0.09093034], [1.0292343, 0.12141731], [1.0123438, 0.037362378], [1.0350122, 0.1281267], [1.0963296, 0.3398288]], "tornadao_mean_prob_computed_climatology_grads_1348" => [[1.0203846, 0.15487157], [0.93945754, -0.35369343], [1.1307863, 0.61256605], [1.0428475, 0.24978055], [1.0871354, 0.40001607]], "tornadao_mean_prob_computed_climatology_blurs_910" => [[1.065771, 0.47239247], [0.9286899, -0.38221243], [1.0567164, 0.26145768], [0.9913568, 0.01256458], [1.0834966, 0.3218238]], "tornadao_mean_prob_computed_climatology_blurs_grads_2005" => [[1.0343003, 0.2688026], [0.95909494, -0.18902206], [1.0664421, 0.33880964], [1.0681071, 0.33059332], [1.078756, 0.37049353]], "tornadao_mean_prob_computed_climatology_3hr_1567" => [[1.0639268, 0.45781243], [0.9226182, -0.4165027], [0.9799438, -0.11478417], [1.0525097, 0.19820513], [1.2263151, 0.7710571]])

# ...


# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)






import Dates

push!(LOAD_PATH, (@__DIR__) * "/../shared")
import TrainingShared
import Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories

(_, validation_forecasts_calibrated_with_sig_gated, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.regular_forecasts(HREFPredictionAblations.forecasts_calibrated_with_sig_gated()); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts_calibrated_with_sig_gated = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts_calibrated_with_sig_gated);

# Make sure a forecast loads
Forecasts.data(validation_forecasts_calibrated_with_sig_gated[100]);

# rm("validation_forecasts_calibrated_with_sig_gated"; recursive=true)

X, Ys, weights = TrainingShared.get_data_labels_weights(validation_forecasts_calibrated_with_sig_gated; event_name_to_labeler = Dict("tornado" => TrainingShared.event_name_to_labeler["tornado"]), save_dir = "validation_forecasts_calibrated_with_sig_gated");

function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for prediction_i in 1:length(HREFPredictionAblations.models)
    (event_name, _, _) = HREFPredictionAblations.models[prediction_i]
    y = Ys[event_name]
    x = @view X[:,prediction_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$event_name ($(round(sum(y)))) feature $prediction_i $(Inventories.inventory_line_description(inventory[prediction_i]))\tAU-PR-curve: $(Float32(au_pr_curve))")
  end
end
test_predictive_power(validation_forecasts_calibrated_with_sig_gated, X, Ys, weights)

# Same as before, because calibration was monotonic
# tornado (68134.0)    feature 1 TORPROB:calculated:hour  fcst:calculated_prob: AU-PR-curve: 0.038589306
# wind (562866.0)      feature 2 WINDPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.11570608
# hail (246689.0)      feature 3 HAILPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.07425418
# sig_tornado (9182.0) feature 4 STORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.033819992
# sig_wind (57701.0)   feature 5 SWINDPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.016239488
# sig_hail (30597.0)   feature 6 SHAILPRO:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.015588201



function test_calibration(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  println("model_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    (event_name, _, model_name) = HREFPredictionAblations.models_with_gated[prediction_i]
    y = Ys[event_name]
    ŷ = @view X[:, feature_i]

    sort_perm      = Metrics.parallel_sort_perm(ŷ);
    y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
    ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
    weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

    bin_count = 20
    per_bin_pos_weight = Float64(sum(y .* weights)) / bin_count

    # bins = map(_ -> Int64[], 1:bin_count)
    bins_Σŷ      = map(_ -> 0.0, 1:bin_count)
    bins_Σy      = map(_ -> 0.0, 1:bin_count)
    bins_Σweight = map(_ -> 0.0, 1:bin_count)
    bins_max     = map(_ -> 1.0f0, 1:bin_count)

    bin_i = 1
    for i in 1:length(y_sorted)
      if ŷ_sorted[i] > bins_max[bin_i]
        bin_i += 1
      end

      bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
      bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
      bins_Σweight[bin_i] += Float64(weights_sorted[i])

      if bins_Σy[bin_i] >= per_bin_pos_weight
        bins_max[bin_i] = ŷ_sorted[i]
      end
    end

    for bin_i in 1:bin_count
      Σŷ      = bins_Σŷ[bin_i]
      Σy      = bins_Σy[bin_i]
      Σweight = bins_Σweight[bin_i]

      mean_ŷ = Σŷ / Σweight
      mean_y = Σy / Σweight

      println("$model_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
    end
  end
end
test_calibration(validation_forecasts_calibrated_with_sig_gated, X, Ys, weights)

# event_name                   mean_y        mean_ŷ        Σweight     bin_max
# tornado                      4.7577023e-6  5.0776057e-6  6.726667e8  8.822641e-5
# tornado                      0.00020302102 0.00016350107 1.5761458e7 0.0003000085
# tornado                      0.00042865236 0.0004785654  7.4647785e6 0.00075306173
# tornado                      0.0009798631  0.0009926242  3.265949e6  0.0013058825
# tornado                      0.0015785117  0.0016265801  2.0273739e6 0.0020298408
# tornado                      0.0024154384  0.0024362206  1.3249922e6 0.0029350107
# tornado                      0.0035027056  0.003438191   913703.3    0.0040292507
# tornado                      0.0048957192  0.004606743   653651.0    0.005281379
# tornado                      0.0059676054  0.006045543   536161.94   0.006945363
# tornado                      0.0075219213  0.0079397755  425421.72   0.009119052
# tornado                      0.009690316   0.010305535   330234.44   0.011710509
# tornado                      0.01346105    0.013127627   237694.03   0.014796797
# tornado                      0.017307408   0.016604794   184907.88   0.018740496
# tornado                      0.02118981    0.021137537   151027.0    0.023900302
# tornado                      0.026993312   0.026827194   118535.86   0.030261038
# tornado                      0.032782637   0.034166776   97617.08    0.038810108
# tornado                      0.043053117   0.043917384   74317.92    0.049964588
# tornado                      0.058244523   0.056626145   54942.633   0.06514058
# tornado                      0.07824215    0.076699294   40898.242   0.09354243
# tornado                      0.1312069     0.12843598    24320.938   1.0
# wind                         3.9025166e-5  3.5552963e-5  6.702e8     0.001023272
# wind                         0.0018041574  0.0018325968  1.4496691e7 0.00312451
# wind                         0.004134872   0.0042761746  6.325239e6  0.0057691615
# wind                         0.006964759   0.0071469685  3.7552998e6 0.008803299
# wind                         0.010242495   0.010400374   2.553491e6  0.012276133
# wind                         0.014020917   0.014157137   1.8653774e6 0.016341235
# wind                         0.018865112   0.01849844    1.3863931e6 0.020964393
# wind                         0.023777591   0.023456464   1.0999704e6 0.026248872
# wind                         0.029288191   0.029076602   892990.7    0.032227375
# wind                         0.03493034    0.035501644   748756.44   0.039143797
# wind                         0.042730443   0.04298102    612086.2    0.047272224
# wind                         0.05136805    0.05173728    509152.84   0.056737307
# wind                         0.06225012    0.061877556   420148.72   0.06760992
# wind                         0.07506528    0.07360504    348415.72   0.08035976
# wind                         0.08970167    0.087473385   291572.2    0.09547654
# wind                         0.10498218    0.10420463    249132.23   0.11412729
# wind                         0.12361988    0.1251774     211568.9    0.1384551
# wind                         0.14966369    0.15507202    174753.56   0.17585532
# wind                         0.19542663    0.20388791    133831.77   0.2444367
# wind                         0.32773158    0.3162817     79774.57    1.0
# hail                         1.6801008e-5  1.5820162e-5  6.7684454e8 0.00054557895
# hail                         0.0009952725  0.0009636982  1.1425841e7 0.0016103927
# hail                         0.0021486226  0.0022045332  5.292855e6  0.0029754357
# hail                         0.003628722   0.0036911208  3.1338678e6 0.0045527727
# hail                         0.0053407014  0.0053622695  2.1292708e6 0.006295701
# hail                         0.007297265   0.0071850354  1.558415e6  0.008185917
# hail                         0.009084847   0.0091959955  1.2517269e6 0.010347905
# hail                         0.011276938   0.0115990015  1.0084563e6 0.013054606
# hail                         0.0145315835  0.014584941   782605.4    0.016366318
# hail                         0.018257974   0.018321542   622864.56   0.020623907
# hail                         0.022885127   0.022993619   496903.44   0.025654215
# hail                         0.028959394   0.028281879   392700.84   0.031198185
# hail                         0.03420509    0.034223042   332455.53   0.03755446
# hail                         0.04172632    0.040982887   272527.97   0.04498506
# hail                         0.04953626    0.049575552   229573.7    0.0549675
# hail                         0.059924424   0.061312187   189779.1    0.069058925
# hail                         0.07762781    0.07809931    146497.45   0.08907306
# hail                         0.10160117    0.10173436    111923.93   0.117873155
# hail                         0.13919899    0.13863233    81698.22    0.16830547
# hail                         0.22663823    0.22620103    50136.42    1.0
# sig_tornado                  6.303477e-7   7.2038415e-7  6.9721e8    3.3574852e-5
# sig_tornado                  8.845426e-5   7.785889e-5   4.9699965e6 0.00016840432
# sig_tornado                  0.0002450834  0.00026580694 1.7936699e6 0.00041199927
# sig_tornado                  0.0005134927  0.0005492066  857060.8    0.0007302659
# sig_tornado                  0.0008319362  0.00093923253 528898.6    0.001231456
# sig_tornado                  0.0016940493  0.0014945535  259441.08   0.0018364261
# sig_tornado                  0.0021714815  0.0022762588  202382.6    0.0027798968
# sig_tornado                  0.003079509   0.0032655848  142670.39   0.003864245
# sig_tornado                  0.004283152   0.0045055184  102679.26   0.005289895
# sig_tornado                  0.0068380344  0.006024709   64276.76    0.006899892
# sig_tornado                  0.008351387   0.007884817   52685.367   0.009023594
# sig_tornado                  0.009828747   0.0103205135  44777.113   0.011838434
# sig_tornado                  0.014543949   0.013168855   30207.785   0.014640002
# sig_tornado                  0.014615556   0.016604831   30074.26    0.01905926
# sig_tornado                  0.020052748   0.021651385   21923.078   0.024841756
# sig_tornado                  0.029392289   0.028060745   14950.955   0.032008216
# sig_tornado                  0.037813835   0.036915112   11618.702   0.043161657
# sig_tornado                  0.050977778   0.05074944    8636.439    0.060440518
# sig_tornado                  0.08372934    0.07116039    5257.4688   0.087302685
# sig_tornado                  0.12550399    0.13317385    3444.568    1.0
# sig_wind                     3.961684e-6   3.698574e-6   6.731528e8  8.695716e-5
# sig_wind                     0.00020420847 0.00015564042 1.3060035e7 0.00026710462
# sig_wind                     0.00038452615 0.00040125483 6.934748e6  0.00059058797
# sig_wind                     0.0007386655  0.0007636923  3.6094055e6 0.0009785821
# sig_wind                     0.0011495483  0.0011788368  2.3194235e6 0.0014143221
# sig_wind                     0.0015365441  0.0016512765  1.7351946e6 0.0019224314
# sig_wind                     0.0021286062  0.002173407   1.2528846e6 0.002466119
# sig_wind                     0.0027563218  0.0027933675  967334.06   0.0031891225
# sig_wind                     0.0037000172  0.003623283   720684.75   0.004152767
# sig_wind                     0.00453524    0.004806697   587936.2    0.005618073
# sig_wind                     0.0060018664  0.006419017   444325.28   0.0073132077
# sig_wind                     0.008196961   0.008096937   325332.03   0.008940162
# sig_wind                     0.010665658   0.00966769    249982.47   0.010425533
# sig_wind                     0.011979158   0.011265674   222626.69   0.012294381
# sig_wind                     0.0140962135  0.013374587   189175.36   0.01455917
# sig_wind                     0.015122527   0.015887551   176309.06   0.017349068
# sig_wind                     0.017545048   0.018884214   151972.52   0.02070126
# sig_wind                     0.023196883   0.02284445    114938.61   0.025555518
# sig_wind                     0.032804325   0.028975844   81290.97    0.03396688
# sig_wind                     0.045624822   0.045747764   58251.93    1.0
# sig_hail                     2.0565428e-6  1.8520311e-6  6.916354e8  0.00012762647
# sig_hail                     0.0002722307  0.00021408104 5.223518e6  0.0003455947
# sig_hail                     0.000487786   0.0004972943  2.9153055e6 0.000705616
# sig_hail                     0.0009044581  0.00089577783 1.572771e6  0.0011281127
# sig_hail                     0.0012851037  0.0013562975  1.1067718e6 0.0016188795
# sig_hail                     0.001778059   0.0018571168  799735.25   0.002116471
# sig_hail                     0.0023178202  0.0023510398  613404.94   0.0026450232
# sig_hail                     0.0030434355  0.00295975    467368.8    0.0033051437
# sig_hail                     0.003672403   0.00364622    387185.9    0.004014159
# sig_hail                     0.0043777893  0.0043652044  324800.38   0.004732903
# sig_hail                     0.0054701753  0.005127069   259978.28   0.005571102
# sig_hail                     0.0058137854  0.006111384   244702.9    0.0067338077
# sig_hail                     0.0073693227  0.0073966407  193047.25   0.00816771
# sig_hail                     0.008702842   0.00905629    163393.56   0.010077691
# sig_hail                     0.011695439   0.011109824   121624.31   0.012307547
# sig_hail                     0.013128468   0.013785124   108296.38   0.015555069
# sig_hail                     0.017100515   0.017555939   83184.47    0.019888023
# sig_hail                     0.024237033   0.022286052   58693.44    0.025278533
# sig_hail                     0.031395745   0.029240599   45288.848   0.034825012
# sig_hail                     0.04681459    0.047757152   30188.738   1.0
# sig_tornado_gated_by_tornado 6.302589e-7   7.182884e-7   6.9730816e8 3.3574852e-5
# sig_tornado_gated_by_tornado 8.967794e-5   7.7806995e-5  4.90218e6   0.00016840432
# sig_tornado_gated_by_tornado 0.00024817706 0.000265865   1.771311e6  0.00041199927
# sig_tornado_gated_by_tornado 0.00051613967 0.00054935203 852665.5    0.0007302659
# sig_tornado_gated_by_tornado 0.0008330432  0.00093932    528195.8    0.001231456
# sig_tornado_gated_by_tornado 0.0017015577  0.0014930425  258320.4    0.0018325506
# sig_tornado_gated_by_tornado 0.0021746664  0.002269171   202131.94   0.0027703033
# sig_tornado_gated_by_tornado 0.0031450652  0.0032401416  139781.14   0.0038170351
# sig_tornado_gated_by_tornado 0.0042994996  0.0044404706  102274.555  0.0052007576
# sig_tornado_gated_by_tornado 0.007062154   0.005880263   62300.543   0.006684648
# sig_tornado_gated_by_tornado 0.0088073695  0.0075628147  49902.71    0.008563295
# sig_tornado_gated_by_tornado 0.009809098   0.009749743   44861.94    0.011120642
# sig_tornado_gated_by_tornado 0.014496204   0.012330667   30325.43    0.013680984
# sig_tornado_gated_by_tornado 0.014378119   0.015405993   30600.826   0.017518433
# sig_tornado_gated_by_tornado 0.017365474   0.020096652   25344.516   0.023232374
# sig_tornado_gated_by_tornado 0.028734975   0.025981048   15294.354   0.029258389
# sig_tornado_gated_by_tornado 0.034875296   0.033482328   12610.21    0.038910735
# sig_tornado_gated_by_tornado 0.043232504   0.046440676   10172.941   0.056332543
# sig_tornado_gated_by_tornado 0.08532565    0.065858305   5154.6875   0.079786226
# sig_tornado_gated_by_tornado 0.14123648    0.10858203    3053.138    1.0
# sig_wind_gated_by_wind       3.9612237e-6  3.662285e-6   6.73231e8   8.695716e-5
# sig_wind_gated_by_wind       0.00020478632 0.00015562537 1.3023182e7 0.00026710462
# sig_wind_gated_by_wind       0.00038580652 0.00040120847 6.9116015e6 0.00059044163
# sig_wind_gated_by_wind       0.0007398425  0.00076363154 3.6037328e6 0.0009785821
# sig_wind_gated_by_wind       0.0011514088  0.0011788331  2.3156758e6 0.0014143221
# sig_wind_gated_by_wind       0.0015394853  0.0016512779  1.7318795e6 0.0019224314
# sig_wind_gated_by_wind       0.002132129   0.0021734545  1.2508146e6 0.002466119
# sig_wind_gated_by_wind       0.0027594941  0.0027933582  966222.0    0.0031891225
# sig_wind_gated_by_wind       0.0037032824  0.0036233203  720049.25   0.004152767
# sig_wind_gated_by_wind       0.00453644    0.0048067686  587780.7    0.005618073
# sig_wind_gated_by_wind       0.006005268   0.006418907   444093.22   0.0073130066
# sig_wind_gated_by_wind       0.008202279   0.008096789   325106.75   0.008940162
# sig_wind_gated_by_wind       0.010677682   0.009667546   249700.97   0.010425533
# sig_wind_gated_by_wind       0.011989084   0.011265771   222442.39   0.012294381
# sig_wind_gated_by_wind       0.014109744   0.013374586   188993.95   0.01455917
# sig_wind_gated_by_wind       0.015136881   0.015887503   176141.88   0.017349068
# sig_wind_gated_by_wind       0.017557878   0.018884273   151861.47   0.02070126
# sig_wind_gated_by_wind       0.023211686   0.022844542   114865.31   0.025555518
# sig_wind_gated_by_wind       0.032814723   0.028976081   81265.2     0.03396688
# sig_wind_gated_by_wind       0.04562977    0.045748796   58245.617   1.0
# sig_hail_gated_by_hail       2.0561513e-6  1.8400787e-6  6.916353e8  0.00012675468
# sig_hail_gated_by_hail       0.00027106964 0.00021342444 5.2468915e6 0.0003455947
# sig_hail_gated_by_hail       0.0004918202  0.0004963276  2.8923572e6 0.0007029206
# sig_hail_gated_by_hail       0.00091127097 0.0008909939  1.5602256e6 0.0011205256
# sig_hail_gated_by_hail       0.0012830608  0.0013479741  1.1087965e6 0.0016097318
# sig_hail_gated_by_hail       0.0017705082  0.0018484303  803382.9    0.0021084556
# sig_hail_gated_by_hail       0.002288989   0.0023458595  621372.44   0.0026427328
# sig_hail_gated_by_hail       0.003050013   0.0029568179  466339.38   0.0033016822
# sig_hail_gated_by_hail       0.0036771693  0.0036425344  386654.47   0.004010433
# sig_hail_gated_by_hail       0.004375538   0.004362241   325012.34   0.0047308113
# sig_hail_gated_by_hail       0.0054804753  0.005124128   259476.84   0.0055672345
# sig_hail_gated_by_hail       0.0057958863  0.006109305   245453.33   0.0067338394
# sig_hail_gated_by_hail       0.0073692654  0.0073973755  193041.38   0.008169025
# sig_hail_gated_by_hail       0.008707519   0.009057851   163305.88   0.010079546
# sig_hail_gated_by_hail       0.011702028   0.011111906   121554.73   0.0123103075
# sig_hail_gated_by_hail       0.013108428   0.013793052   108532.54   0.015570249
# sig_hail_gated_by_hail       0.017164396   0.017568279   82880.484   0.019894622
# sig_hail_gated_by_hail       0.024283849   0.022290794   58578.574   0.025279433
# sig_hail_gated_by_hail       0.031356428   0.029249929   45346.637   0.034850664
# sig_hail_gated_by_hail       0.046860833   0.04778715    30117.734   1.0
