import Dates
import Printf
import Statistics

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import TrainingShared

push!(LOAD_PATH, (@__DIR__) * "/../models/spc_outlooks")
import SPCOutlooks

push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
import CombinedHREFSREF

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction")
import HREFPrediction

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations")
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations2")
import HREFPredictionAblations2

push!(LOAD_PATH, (@__DIR__) * "/../models/href_day_experiment")
import HREFDayExperiment

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
using Grid130
import PlotMap
import StormEvents
import ForecastCombinators

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

GRID                 = Conus.href_cropped_5km_grid();
CONUS_MASK           = Conus.conus_mask_href_cropped_5km_grid();
VERIFIABLE_GRID_MASK = CONUS_MASK .&& TrainingShared.is_verifiable.(GRID.latlons) :: BitVector;



# Run below is 2019-1-7 through 2022-5-31, but we are missing lots of HREFs between Nov 2020 and mid-March 2021

# conus_area = sum(GRID.point_areas_sq_miles[CONUS_MASK))

# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[1,3] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[2,4] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[5,6] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[7,8] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[9,10] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[11,12] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl

TASKS = eval(Meta.parse(get(ENV, "TASKS", "")))
if isnothing(TASKS)
  TASKS = typemin(Int64):typemax(Int64)
end
DRAW_SPC_MAPS = get(ENV, "DRAW_SPC_MAPS", "true") == "true"

const ϵ = eps(1f0)

function do_it(spc_forecasts, forecasts, model_names; run_hour, suffix, cutoff = Dates.DateTime(2022, 6, 1, 12), use_train_validation_too = false, compute_extra_mask = nothing)

  println("************ $(run_hour)z$(suffix) ************")

  println("$(length(spc_forecasts)) SPC forecasts available") #

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  if use_train_validation_too
    test_forecasts = vcat(train_forecasts, validation_forecasts, test_forecasts)
    sort!(test_forecasts, by = Forecasts.run_utc_datetime, alg = MergeSort)
  end

  println("$(length(test_forecasts)) unfiltered test forecasts")
  test_forecasts = filter(forecast -> forecast.run_hour == run_hour, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts")


  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts before the event data cutoff date $cutoff") #

  # If you want to augment the test set with all the days after training

  # training_data_end = Dates.DateTime(2022, 1, 1, 0)
  # other_test_forecasts = vcat(train_forecasts, validation_forecasts);
  # length(other_test_forecasts) # 3754
  # other_test_forecasts = filter(forecast -> forecast.run_hour == run_hour, other_test_forecasts);
  # length(other_test_forecasts) # 939
  # other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, other_test_forecasts);
  # length(other_test_forecasts) # 841
  # other_test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) > training_data_end, other_test_forecasts);
  # length(other_test_forecasts) # 45

  # test_forecasts = vcat(test_forecasts, other_test_forecasts);
  # length(test_forecasts) # 186

  event_name_to_unadj_events = Dict(
    "tornado"     => StormEvents.conus_tornado_events(),
    "wind"        => StormEvents.conus_severe_wind_events(),
    "hail"        => StormEvents.conus_severe_hail_events(),
    "sig_tornado" => StormEvents.conus_sig_tornado_events(),
    "sig_wind"    => StormEvents.conus_sig_wind_events(),
    "sig_hail"    => StormEvents.conus_sig_hail_events(),
  )

  event_name_to_thresholds = Dict(
    "tornado"      => [0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
    "wind"         => [0.02, 0.05, 0.15, 0.3, 0.45, 0.6],
    "wind_adj"     => [0.02, 0.05, 0.15, 0.3, 0.45, 0.6],
    "hail"         => [0.02, 0.05, 0.15, 0.3, 0.45, 0.6],
    "sig_tornado"  => [0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
    "sig_wind"     => [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
    "sig_wind_adj" => [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
    "sig_hail"     => [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6],
  )

  # is_ablation(model_name) = occursin(r"\Atornado_.+_\d+\z", model_name)
  model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1")
  unadj_name(event_name) = replace(event_name, r"_adj\z" => "")

  # Use non-sig colors (i.e. not just 10%)
  model_name_to_colorer(model_name) = PlotMap.event_name_to_colorer_more_sig_colors[model_name_to_event_name(model_name)]

  compute_forecast_labels(model_name, spc_forecast) = begin
    # Annoying that we have to recalculate this.
    start_seconds =
      if spc_forecast.run_hour == 6
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
      elseif spc_forecast.run_hour == 13
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
      elseif spc_forecast.run_hour == 16
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
      end
    end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
    println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
    utc_datetime = Dates.unix2datetime(start_seconds)
    println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
    println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
    window_half_size = (end_seconds - start_seconds) ÷ 2
    window_mid_time  = (end_seconds + start_seconds) ÷ 2
    event_name = model_name_to_event_name(model_name)
    if endswith(event_name, "wind_adj")
      measured_events, estimated_events, gridded_normalization =
        if event_name == "wind_adj"
          StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), TrainingShared.day_estimated_wind_gridded_normalization()
        elseif event_name == "sig_wind_adj"
          StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), TrainingShared.day_estimated_sig_wind_gridded_normalization()
        else
          error("unknown adj event $event_name")
        end
      measured_labels  = StormEvents.grid_to_event_neighborhoods(measured_events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, spc_forecast.grid, Grid130.GRID_130_CROPPED, gridded_normalization, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      max.(measured_labels, estimated_labels)
    else
      events = event_name_to_unadj_events[event_name]
      StormEvents.grid_to_event_neighborhoods(events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
    end
  end

  spc_threshold_painted_areas         = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  test_threshold_painted_areas        = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  spc_threshold_true_positive_areas   = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  test_threshold_true_positive_areas  = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  spc_threshold_false_negative_areas  = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  test_threshold_false_negative_areas = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  spc_threshold_true_negative_areas   = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))
  test_threshold_true_negative_areas  = Dict(map(model_name -> model_name => map(_ -> Float64[], event_name_to_thresholds[model_name_to_event_name(model_name)]), model_names))

  function forecast_stats(data, labels, threshold, extra_mask=nothing)
    painted   = ((@view data[:,1]) .>= threshold*0.9999) .* VERIFIABLE_GRID_MASK
    unpainted = ((@view data[:,1]) .<  threshold*0.9999) .* VERIFIABLE_GRID_MASK
    if !isnothing(extra_mask)
      painted   = painted   .* extra_mask
      unpainted = unpainted .* extra_mask
    end
    painted_area        = sum(GRID.point_areas_sq_miles[painted])
    unpainted_area      = sum(GRID.point_areas_sq_miles[unpainted])
    true_positive_area  = sum(GRID.point_areas_sq_miles .* painted   .* labels)
    false_negative_area = sum(GRID.point_areas_sq_miles .* unpainted .* labels)
    true_negative_area  = unpainted_area - false_negative_area
    (painted_area, true_positive_area, false_negative_area, true_negative_area)
  end

  # event_name_to_thresholds plus another threshold at 1.0
  bin_maxes = Dict(map(model_names) do model_name
    model_name => [event_name_to_thresholds[model_name_to_event_name(model_name)]; 1.0]
  end)

  spc_bin_painted_areas        = Dict(map(model_name -> model_name => map(_ -> Float64[], bin_maxes[model_name]), model_names))
  test_bin_painted_areas       = Dict(map(model_name -> model_name => map(_ -> Float64[], bin_maxes[model_name]), model_names))
  spc_bin_true_positive_areas  = Dict(map(model_name -> model_name => map(_ -> Float64[], bin_maxes[model_name]), model_names))
  test_bin_true_positive_areas = Dict(map(model_name -> model_name => map(_ -> Float64[], bin_maxes[model_name]), model_names))

  function reliability_stats(data, labels, threshold_lo, threshold_hi, extra_mask=nothing)
    painted             = ((@view data[:,1]) .>= threshold_lo*0.9999) .* ((@view data[:,1]) .< threshold_hi*0.9999) .* VERIFIABLE_GRID_MASK
    if !isnothing(extra_mask)
      painted = painted .* extra_mask
    end
    painted_area        = sum(GRID.point_areas_sq_miles[painted])
    true_positive_area  = sum(GRID.point_areas_sq_miles .* painted .* labels)
    (painted_area, true_positive_area)
  end

  open((@__DIR__) * "/test_$(run_hour)z$(suffix).csv", "w") do csv

    headers = ["yymmdd", "spc", "nadocast"]

    for model_name in model_names
      for threshold in event_name_to_thresholds[model_name_to_event_name(model_name)]
        headers = vcat(headers, ["$(model_name)_spc_painted_sq_mi_$threshold",      "$(model_name)_spc_true_positive_sq_mi_$threshold",      "$(model_name)_spc_false_negative_sq_mi_$threshold",      "$(model_name)_spc_true_negative_sq_mi_$threshold"])
        headers = vcat(headers, ["$(model_name)_nadocast_painted_sq_mi_$threshold", "$(model_name)_nadocast_true_positive_sq_mi_$threshold", "$(model_name)_nadocast_false_negative_sq_mi_$threshold", "$(model_name)_nadocast_true_negative_sq_mi_$threshold"])
      end
    end

    println(join(headers, ","))
    println(csv, join(headers, ","))

    for spc_forecast in spc_forecasts
      test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
      if isnothing(test_forecast_i)
        continue
      end
      test_forecast = test_forecasts[test_forecast_i]

      if !isnothing(compute_extra_mask)
        extra_mask = compute_extra_mask(spc_forecast) :: BitVector
        if sum(extra_mask .* VERIFIABLE_GRID_MASK) == 0
          continue
        end
      end

      row = [Forecasts.yyyymmdd(spc_forecast), Forecasts.time_title(spc_forecast), Forecasts.time_title(test_forecast)]

      spc_data  = Forecasts.data(spc_forecast)
      ForecastCombinators.turn_forecast_caching_on()
      test_data = Forecasts.data(test_forecast)
      ForecastCombinators.clear_cached_forecasts()

      for model_name in model_names
        event_name   = model_name_to_event_name(model_name)
        spc_event_i  = findfirst(m -> m[1] == unadj_name(event_name), SPCOutlooks.models)
        test_event_i = findfirst(isequal(model_name), model_names)

        spc_event_probs  = @view spc_data[:, spc_event_i]
        test_event_probs = @view test_data[:, test_event_i]

        forecast_labels = compute_forecast_labels(model_name, spc_forecast) :: Vector{Float32}

        map_root = ((@__DIR__) * "/maps/$(Forecasts.yyyymmdd(spc_forecast))")
        mkpath(map_root)

        post_process(img) =
          if isnothing(extra_mask)
            PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img))
          else
            PlotMap.multiply_image(1 .- (CONUS_MASK .* extra_mask .* 0.5f0), PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK .* extra_mask, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img)))
          end

        make_plot(file_name, data) = begin
          path = map_root * "/" * file_name
          PlotMap.plot_fast(path, GRID, data .* CONUS_MASK; val_to_color=model_name_to_colorer(model_name), post_process=post_process)
          PlotMap.optimize_png(path; wait = false)
        end

        if !isnothing(spc_event_i) && DRAW_SPC_MAPS
          make_plot("spc_day_1_$(event_name)_$(Forecasts.yyyymmdd_thhz(spc_forecast))", spc_event_probs)
        end
        make_plot("nadocast$(suffix)_$(model_name)_$(Forecasts.yyyymmdd_thhz(test_forecast))", test_event_probs)

        for threshold_i in 1:length(event_name_to_thresholds[model_name_to_event_name(model_name)])
          threshold = event_name_to_thresholds[model_name_to_event_name(model_name)][threshold_i]

          (spc_painted_area,  spc_true_positive_area,  spc_false_negative_area,  spc_true_negative_area)  = forecast_stats(spc_event_probs,  forecast_labels, threshold, extra_mask)
          (test_painted_area, test_true_positive_area, test_false_negative_area, test_true_negative_area) = forecast_stats(test_event_probs, forecast_labels, threshold, extra_mask)

          row = vcat(row, [spc_painted_area,  spc_true_positive_area,  spc_false_negative_area,  spc_true_negative_area])
          row = vcat(row, [test_painted_area, test_true_positive_area, test_false_negative_area, test_true_negative_area])

          push!(spc_threshold_painted_areas[model_name][threshold_i],         spc_painted_area)
          push!(spc_threshold_true_positive_areas[model_name][threshold_i],   spc_true_positive_area)
          push!(spc_threshold_false_negative_areas[model_name][threshold_i],  spc_false_negative_area)
          push!(spc_threshold_true_negative_areas[model_name][threshold_i],   spc_true_negative_area)
          push!(test_threshold_painted_areas[model_name][threshold_i],        test_painted_area)
          push!(test_threshold_true_positive_areas[model_name][threshold_i],  test_true_positive_area)
          push!(test_threshold_false_negative_areas[model_name][threshold_i], test_false_negative_area)
          push!(test_threshold_true_negative_areas[model_name][threshold_i],  test_true_negative_area)
        end

        bin_lo = 0.0
        for bin_i in 1:length(bin_maxes[model_name])
          bin_hi = bin_maxes[model_name][bin_i]

          (spc_painted_area,  spc_true_positive_area)  = reliability_stats(spc_event_probs,  forecast_labels, bin_lo, bin_hi, extra_mask)
          push!(spc_bin_painted_areas[model_name][bin_i],        spc_painted_area)
          push!(spc_bin_true_positive_areas[model_name][bin_i],  spc_true_positive_area)

          (test_painted_area, test_true_positive_area) = reliability_stats(test_event_probs, forecast_labels, bin_lo, bin_hi, extra_mask)
          push!(test_bin_painted_areas[model_name][bin_i],       test_painted_area)
          push!(test_bin_true_positive_areas[model_name][bin_i], test_true_positive_area)

          bin_lo = bin_hi
        end
      end

      println(join(row, ","))
      println(csv, join(row, ","))
    end

    open((@__DIR__) * "/stats_$(run_hour)z$(suffix).csv", "w") do stats_csv
      stats_headers = ["event", "days_count", "threshold", "spc_threshold_days", "nadocast_threshold_days", "spc_success_ratio", "spc_pod", "spc_false_positive_rate", "nadocast_success_ratio", "nadocast_pod", "nadocast_false_positive_rate"]

      stats_headers = vcat(stats_headers, ["spc_success_ratio_0.025_bootstrap",      "spc_success_ratio_0.975_bootstrap",      "spc_pod_0.025_bootstrap",      "spc_pod_0.975_bootstrap",      "spc_csi_0.025_bootstrap",      "spc_csi_0.975_bootstrap"])
      stats_headers = vcat(stats_headers, ["nadocast_success_ratio_0.025_bootstrap", "nadocast_success_ratio_0.975_bootstrap", "nadocast_pod_0.025_bootstrap", "nadocast_pod_0.975_bootstrap", "nadocast_csi_0.025_bootstrap", "nadocast_csi_0.975_bootstrap"])
      stats_headers = vcat(stats_headers, ["p_value_nadocast_csi_better"])

      println(join(stats_headers, ","))
      println(stats_csv, join(stats_headers, ","))

      for model_name in model_names
        for threshold_i in 1:length(event_name_to_thresholds[model_name_to_event_name(model_name)])
          threshold = event_name_to_thresholds[model_name_to_event_name(model_name)][threshold_i]

          total_days_count = length(test_threshold_painted_areas[model_name][threshold_i])
          spc_threshold_days_count  = count(spc_threshold_painted_areas[model_name][threshold_i] .> 0)
          test_threshold_days_count = count(test_threshold_painted_areas[model_name][threshold_i] .> 0)

          spc_painted_area         = sum(spc_threshold_painted_areas[model_name][threshold_i])
          test_painted_area        = sum(test_threshold_painted_areas[model_name][threshold_i])
          spc_true_positive_area   = sum(spc_threshold_true_positive_areas[model_name][threshold_i])
          test_true_positive_area  = sum(test_threshold_true_positive_areas[model_name][threshold_i])
          spc_false_negative_area  = sum(spc_threshold_false_negative_areas[model_name][threshold_i])
          test_false_negative_area = sum(test_threshold_false_negative_areas[model_name][threshold_i])
          spc_true_negative_area   = sum(spc_threshold_true_negative_areas[model_name][threshold_i])
          test_true_negative_area  = sum(test_threshold_true_negative_areas[model_name][threshold_i])

          spc_false_positive_area  = spc_painted_area - spc_true_positive_area
          spc_negative_area        = spc_false_positive_area + spc_true_negative_area
          test_false_positive_area = test_painted_area - test_true_positive_area
          test_negative_area       = test_false_positive_area + test_true_negative_area

          spc_sr_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area = Float32(sum(view(spc_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            painted_area  = Float32(sum(view(spc_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))

            true_pos_area / (painted_area + ϵ)
          end

          spc_pod_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area  = Float32(sum(view(spc_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            false_neg_area = Float32(sum(view(spc_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            true_pos_area / (true_pos_area + false_neg_area + ϵ)
          end

          test_sr_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area = Float32(sum(view(test_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            painted_area  = Float32(sum(view(test_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))

            true_pos_area / (painted_area + ϵ)
          end

          test_pod_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area  = Float32(sum(view(test_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            false_neg_area = Float32(sum(view(test_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            true_pos_area / (true_pos_area + false_neg_area + ϵ)
          end

          spc_csi_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area  = Float32(sum(view(spc_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            painted_area   = Float32(sum(view(spc_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))
            false_neg_area = Float32(sum(view(spc_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            # CSI = tp / (tp + fp + fn)
            true_pos_area / (painted_area  + false_neg_area + ϵ)
          end

          test_csi_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area  = Float32(sum(view(test_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            painted_area   = Float32(sum(view(test_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))
            false_neg_area = Float32(sum(view(test_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            # CSI = tp / (tp + fp + fn)
            true_pos_area / (painted_area  + false_neg_area + ϵ)
          end

          csi_diff_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            spc_true_pos  = Float32(sum(view(spc_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            spc_painted   = Float32(sum(view(spc_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))
            spc_false_neg = Float32(sum(view(spc_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            test_true_pos  = Float32(sum(view(test_threshold_true_positive_areas[model_name][threshold_i], bootstrap_is)))
            test_painted   = Float32(sum(view(test_threshold_painted_areas[model_name][threshold_i], bootstrap_is)))
            test_false_neg = Float32(sum(view(test_threshold_false_negative_areas[model_name][threshold_i], bootstrap_is)))

            # CSI = tp / (tp + fp + fn)

            spc_csi  = spc_true_pos  / (spc_painted  + spc_false_neg + ϵ)
            test_csi = test_true_pos / (test_painted + test_false_neg + ϵ)

            test_csi - spc_csi
          end

          row = [
            model_name,
            total_days_count,
            Float32(threshold),
            spc_threshold_days_count,
            test_threshold_days_count,
            Float32(spc_true_positive_area  / spc_painted_area), # sr
            Float32(spc_true_positive_area  / (spc_true_positive_area + spc_false_negative_area)), # pod
            Float32(spc_false_positive_area / spc_negative_area), # fpr
            Float32(test_true_positive_area / test_painted_area), # sr
            Float32(test_true_positive_area / (test_true_positive_area + test_false_negative_area)), # pod
            Float32(test_false_positive_area / test_negative_area), # fpr
            Statistics.quantile(spc_sr_bootstraps, 0.025),
            Statistics.quantile(spc_sr_bootstraps, 0.975),
            Statistics.quantile(spc_pod_bootstraps, 0.025),
            Statistics.quantile(spc_pod_bootstraps, 0.975),
            Statistics.quantile(spc_csi_bootstraps, 0.025),
            Statistics.quantile(spc_csi_bootstraps, 0.975),
            Statistics.quantile(test_sr_bootstraps, 0.025),
            Statistics.quantile(test_sr_bootstraps, 0.975),
            Statistics.quantile(test_pod_bootstraps, 0.025),
            Statistics.quantile(test_pod_bootstraps, 0.975),
            Statistics.quantile(test_csi_bootstraps, 0.025),
            Statistics.quantile(test_csi_bootstraps, 0.975),
            Float32(count(csi_diff_bootstraps .<= 0) / length(csi_diff_bootstraps))
          ]

          println(join(row, ","))
          println(stats_csv, join(row, ","))
        end
      end
    end

    open((@__DIR__) * "/test_reliability_$(run_hour)z$(suffix).csv", "w") do reliability_csv
      reliability_headers = ["event", "days_count", "bin_low", "bin_high_exclusive", "spc_bin_days", "nadocast_bin_days", "spc_observed_rate", "nadocast_observed_rate", "spc_observed_rate_0.025_bootstrap", "spc_observed_rate_0.975_bootstrap", "nadocast_observed_rate_0.025_bootstrap", "nadocast_observed_rate_0.975_bootstrap"]

      println(join(reliability_headers, ","))
      println(reliability_csv, join(reliability_headers, ","))

      for model_name in model_names
        bin_lo = 0.0
        for bin_i in 1:length(bin_maxes[model_name])
          bin_hi = bin_maxes[model_name][bin_i]

          total_days_count    = length(test_bin_painted_areas[model_name][bin_i])
          spc_bin_days_count  = count(spc_bin_painted_areas[model_name][bin_i] .> 0)
          test_bin_days_count = count(test_bin_painted_areas[model_name][bin_i] .> 0)

          spc_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area = Float32(sum(view(spc_bin_true_positive_areas[model_name][bin_i], bootstrap_is)))
            painted_area  = Float32(sum(view(spc_bin_painted_areas[model_name][bin_i], bootstrap_is)))

            true_pos_area / (painted_area + ϵ)
          end

          test_bootstraps = map(1:1_000_000) do _
            bootstrap_is = rand(1:total_days_count, total_days_count)

            true_pos_area = Float32(sum(view(test_bin_true_positive_areas[model_name][bin_i], bootstrap_is)))
            painted_area  = Float32(sum(view(test_bin_painted_areas[model_name][bin_i], bootstrap_is)))

            true_pos_area / (painted_area + ϵ)
          end

          row = [
            model_name,
            total_days_count,
            Float32(bin_lo),
            Float32(bin_hi),
            spc_bin_days_count,
            test_bin_days_count,
            Float32(sum(spc_bin_true_positive_areas[model_name][bin_i])  / sum(spc_bin_painted_areas[model_name][bin_i])),
            Float32(sum(test_bin_true_positive_areas[model_name][bin_i]) / sum(test_bin_painted_areas[model_name][bin_i])),
            Statistics.quantile(spc_bootstraps, 0.025),
            Statistics.quantile(spc_bootstraps, 0.975),
            Statistics.quantile(test_bootstraps, 0.025),
            Statistics.quantile(test_bootstraps, 0.975),
          ]

          println(join(row, ","))
          println(reliability_csv, join(row, ","))

          bin_lo = bin_hi
        end
      end
    end
  end
end

function only_forecasts_with_runtimes(reference_forecasts, forecasts_to_filter)
  reference_runtimes = Set(Forecasts.run_utc_datetime.(reference_forecasts))

  filter(fcst -> Forecasts.run_utc_datetime(fcst) in reference_runtimes, forecasts_to_filter)
end

# HREF-SREF models

model_names = map(m -> m[3], CombinedHREFSREF.models_with_gated)

1 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 0, suffix = "")
# 1433 SPC forecasts available
# 694 unfiltered test forecasts
# 173 0z test forecasts
# 155 0z test forecasts before the event data cutoff date
2 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), CombinedHREFSREF.forecasts_day_with_sig_gated(), model_names; run_hour = 0, suffix = "_absolutely_calibrated")
3 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 12, suffix = "")
4 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), CombinedHREFSREF.forecasts_day_with_sig_gated(), model_names; run_hour = 12, suffix = "_absolutely_calibrated")


# HREF-only models

5 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated()), model_names; run_hour = 0, suffix = "_href_only")
6 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFPrediction.forecasts_day_with_sig_gated()), model_names; run_hour = 0, suffix = "_href_only_absolutely_calibrated")
7 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated()), model_names; run_hour = 12, suffix = "_href_only")
8 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFPrediction.forecasts_day_with_sig_gated()), model_names; run_hour = 12, suffix = "_href_only_absolutely_calibrated")


# HREF-only tornado ablations

ablation_model_names = map(m -> m[1], HREFPredictionAblations.models)

9 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations.forecasts_day_spc_calibrated()), ablation_model_names; run_hour = 0, suffix = "_href_ablations")
10 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations.forecasts_day()), ablation_model_names; run_hour = 0, suffix = "_href_ablations_absolutely_calibrated")
11 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations.forecasts_day_spc_calibrated()), ablation_model_names; run_hour = 12, suffix = "_href_ablations")
# 1433 SPC forecasts available
# 694 unfiltered test forecasts
# 174 12z test forecasts
# 155 12z test forecasts before the event data cutoff date
12 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations.forecasts_day()), ablation_model_names; run_hour = 12, suffix = "_href_ablations_absolutely_calibrated")



# Redrawing the adjusted wind maps with dithering

function draw_only(spc_forecasts, forecasts, model_names; run_hour, suffix, start = Dates.DateTime(2000, 1, 1, 1), cutoff = Dates.DateTime(2022, 6, 1, 12), only_models = model_names, all_days_of_week = false)

  println("************ $(run_hour)z$(suffix) ************")

  println("$(length(spc_forecasts)) SPC forecasts available") #

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  if all_days_of_week
    test_forecasts = vcat(train_forecasts, validation_forecasts, test_forecasts)
  end

  println("$(length(test_forecasts)) unfiltered test forecasts") # 627
  test_forecasts = filter(forecast -> forecast.run_hour == run_hour, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts") # 157

  # We don't have storm events past this time.

  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff && Forecasts.valid_utc_datetime(forecast) >= start, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts before the event data cutoff date") #

  event_name_to_unadj_events = Dict(
    "tornado"     => StormEvents.conus_tornado_events(),
    "wind"        => StormEvents.conus_severe_wind_events(),
    "hail"        => StormEvents.conus_severe_hail_events(),
    "sig_tornado" => StormEvents.conus_sig_tornado_events(),
    "sig_wind"    => StormEvents.conus_sig_wind_events(),
    "sig_hail"    => StormEvents.conus_sig_hail_events(),
  )

  # is_ablation(model_name) = occursin(r"\Atornado_.+_\d+\z", model_name)
  model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1")
  unadj_name(event_name) = replace(event_name, r"_adj\z" => "")

  # Use non-sig colors (i.e. not just 10%)
  model_name_to_colorer(model_name) = PlotMap.event_name_to_colorer_more_sig_colors[model_name_to_event_name(model_name)]

  compute_forecast_labels(model_name, spc_forecast) = begin
    # Annoying that we have to recalculate this.
    start_seconds =
      if spc_forecast.run_hour == 6
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
      elseif spc_forecast.run_hour == 13
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
      elseif spc_forecast.run_hour == 16
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
      end
    end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
    println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
    utc_datetime = Dates.unix2datetime(start_seconds)
    println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
    println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
    window_half_size = (end_seconds - start_seconds) ÷ 2
    window_mid_time  = (end_seconds + start_seconds) ÷ 2
    event_name = model_name_to_event_name(model_name)
    if endswith(event_name, "wind_adj")
      measured_events, estimated_events, gridded_normalization =
        if event_name == "wind_adj"
          StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), TrainingShared.day_estimated_wind_gridded_normalization()
        elseif event_name == "sig_wind_adj"
          StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), TrainingShared.day_estimated_sig_wind_gridded_normalization()
        else
          error("unknown adj event $event_name")
        end
      measured_labels  = StormEvents.grid_to_event_neighborhoods(measured_events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, spc_forecast.grid, Grid130.GRID_130_CROPPED, gridded_normalization, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      max.(measured_labels, estimated_labels)
    else
      events = event_name_to_unadj_events[event_name]
      StormEvents.grid_to_event_neighborhoods(events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
    end
  end


  for spc_forecast in spc_forecasts
    test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
    if isnothing(test_forecast_i)
      continue
    end
    test_forecast = test_forecasts[test_forecast_i]

    spc_data  = Forecasts.data(spc_forecast)
    ForecastCombinators.turn_forecast_caching_on()
    test_data = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    for model_name in model_names
      model_name in only_models || continue
      event_name   = model_name_to_event_name(model_name)
      spc_event_i  = findfirst(m -> m[1] == unadj_name(event_name), SPCOutlooks.models)
      test_event_i = findfirst(isequal(model_name), model_names)

      spc_event_probs  = @view spc_data[:, spc_event_i]
      test_event_probs = @view test_data[:, test_event_i]

      forecast_labels = compute_forecast_labels(model_name, spc_forecast) :: Vector{Float32}

      map_root = ((@__DIR__) * "/maps/$(Forecasts.yyyymmdd(spc_forecast))")
      mkpath(map_root)

      post_process(img) = PlotMap.shade_forecast_labels(forecast_labels .* CONUS_MASK, PlotMap.add_conus_lines_href_5k_native_proj_80_pct(img))

      make_plot(file_name, data) = begin
        path = map_root * "/" * file_name
        println(file_name)
        PlotMap.plot_fast(path, GRID, data .* CONUS_MASK; val_to_color=model_name_to_colorer(model_name), post_process=post_process)
        PlotMap.optimize_png(path; wait = false)
      end

      if !isnothing(spc_event_i) && DRAW_SPC_MAPS
        make_plot("spc_day_1_$(event_name)_$(Forecasts.yyyymmdd_thhz(spc_forecast))", spc_event_probs)
      end
      make_plot("nadocast$(suffix)_$(model_name)_$(Forecasts.yyyymmdd_thhz(test_forecast))", test_event_probs)
    end
  end

end

start  = Dates.DateTime(2022, 6, 1, 12)
cutoff = Dates.DateTime(2022, 8, 1, 12)

13 in TASKS && draw_only(SPCOutlooks.forecasts_day_0600(), CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 0, suffix = "", start = start, cutoff = cutoff, all_days_of_week = true)
14 in TASKS && draw_only(SPCOutlooks.forecasts_day_0600(), CombinedHREFSREF.forecasts_day_with_sig_gated(), model_names; run_hour = 0, suffix = "_absolutely_calibrated", start = start, cutoff = cutoff, all_days_of_week = true)
15 in TASKS && draw_only(SPCOutlooks.forecasts_day_1630(), CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 12, suffix = "", start = start, cutoff = cutoff, all_days_of_week = true)
16 in TASKS && draw_only(SPCOutlooks.forecasts_day_1630(), CombinedHREFSREF.forecasts_day_with_sig_gated(), model_names; run_hour = 12, suffix = "_absolutely_calibrated", start = start, cutoff = cutoff, all_days_of_week = true)


# HREF-only models

17 in TASKS && draw_only(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated()), model_names; run_hour = 0, suffix = "_href_only", start = start, cutoff = cutoff, all_days_of_week = true)
18 in TASKS && draw_only(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFPrediction.forecasts_day_with_sig_gated()), model_names; run_hour = 0, suffix = "_href_only_absolutely_calibrated", start = start, cutoff = cutoff, all_days_of_week = true)
19 in TASKS && draw_only(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated()), model_names; run_hour = 12, suffix = "_href_only", start = start, cutoff = cutoff, all_days_of_week = true)
20 in TASKS && draw_only(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFPrediction.forecasts_day_with_sig_gated()), model_names; run_hour = 12, suffix = "_href_only_absolutely_calibrated", start = start, cutoff = cutoff, all_days_of_week = true)

# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[13] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl; FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[16,19,20] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[15] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl; FORECASTS_ROOT=~/nadocaster2 FORECAST_DISK_PREFETCH=false TASKS=[14,17,18] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false TASKS=[13,14,17,18] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false TASKS=[15,16,19,20] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl


(21 in TASKS || 22 in TASKS || 23 in TASKS || 24 in TASKS) && begin
  push!(LOAD_PATH, (@__DIR__))
  import PublishedForecasts2020Models
  import PublishedForecasts2021Models

# tornado only
  forecasts_2020_14z = PublishedForecasts2020Models.forecasts_14z()
  forecasts_2021_12z = PublishedForecasts2021Models.forecasts_12z()

  forecasts_2020_14z_runtimes = Set(Forecasts.run_utc_datetime.(forecasts_2020_14z))
  forecasts_2021_12z_runtimes = Set(Forecasts.run_utc_datetime.(forecasts_2021_12z))

  filter!(fcst -> Forecasts.run_utc_datetime(fcst) - Dates.Hour(2) in forecasts_2021_12z_runtimes, forecasts_2020_14z)
  filter!(fcst -> Forecasts.run_utc_datetime(fcst) + Dates.Hour(2) in forecasts_2020_14z_runtimes, forecasts_2021_12z)

  21 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), forecasts_2020_14z, ["tornado"]; run_hour = 14, suffix = "_2020_models_tornado_with_hrrr",                   cutoff = Dates.DateTime(2022, 8, 1, 12), use_train_validation_too = true)
  22 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), forecasts_2021_12z, ["tornado"]; run_hour = 12, suffix = "_2021_models_tornado_href_sref",                   cutoff = Dates.DateTime(2022, 8, 1, 12), use_train_validation_too = true)
  23 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), forecasts_2020_14z, ["tornado"]; run_hour = 14, suffix = "_2020_models_tornado_with_hrrr_vs_1630_tornadoes", cutoff = Dates.DateTime(2022, 8, 1, 12), use_train_validation_too = true)
  24 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), forecasts_2021_12z, ["tornado"]; run_hour = 12, suffix = "_2021_models_tornado_href_sref_vs_1630_tornadoes", cutoff = Dates.DateTime(2022, 8, 1, 12), use_train_validation_too = true)

  # scp nadocaster2:~/nadocast_dev/test/stats_12z_2021_models_tornado_href_sref.csv ./
  # scp nadocaster2:~/nadocast_dev/test/stats_12z_2021_models_tornado_href_sref_vs_1630_tornadoes.csv ./
  # scp nadocaster2:~/nadocast_dev/test/stats_14z_2020_models_tornado_with_hrrr.csv ./
  # scp nadocaster2:~/nadocast_dev/test/stats_14z_2020_models_tornado_with_hrrr_vs_1630_tornadoes.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_12z_2021_models_tornado_href_sref.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_12z_2021_models_tornado_href_sref_vs_1630_tornadoes.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_14z_2020_models_tornado_with_hrrr.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_14z_2020_models_tornado_with_hrrr_vs_1630_tornadoes.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_12z_2021_models_tornado_href_sref.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_12z_2021_models_tornado_href_sref_vs_1630_tornadoes.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_14z_2020_models_tornado_with_hrrr.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_14z_2020_models_tornado_with_hrrr_vs_1630_tornadoes.csv ./
end

# TASKS=[21] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# TASKS=[22] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# TASKS=[21,22] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl


# Day experiment models

# FORECAST_DISK_PREFETCH=false TASKS=[25,27] DRAW_SPC_MAPS=true julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false TASKS=[28,26] julia -t 16 --project=.. Test.jl

day_experiment_model_names = first.(HREFDayExperiment.models)

25 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFDayExperiment.blurred_spc_calibrated_day_prediction_forecasts()), day_experiment_model_names; run_hour = 0, suffix = "_href_day_experiment")
26 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFDayExperiment.blurred_calibrated_day_prediction_forecasts()), day_experiment_model_names; run_hour = 0, suffix = "_href_day_experiment_absolutely_calibrated")
27 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFDayExperiment.blurred_spc_calibrated_day_prediction_forecasts()), day_experiment_model_names; run_hour = 12, suffix = "_href_day_experiment")
28 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_with_sig_gated(), HREFDayExperiment.blurred_calibrated_day_prediction_forecasts()), day_experiment_model_names; run_hour = 12, suffix = "_href_day_experiment_absolutely_calibrated")



ablation2_model_names = map(m -> m[1], HREFPredictionAblations2.models)

# FORECAST_DISK_PREFETCH=false TASKS=[29,30] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false USE_ALT_DISK=true TASKS=[31,32] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl

29 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations2.forecasts_day_spc_calibrated()), ablation2_model_names; run_hour = 0, suffix = "_href_ablations2")
30 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations2.forecasts_day()), ablation2_model_names; run_hour = 0, suffix = "_href_ablations2_absolutely_calibrated")
31 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations2.forecasts_day_spc_calibrated()), ablation2_model_names; run_hour = 12, suffix = "_href_ablations2")
32 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), only_forecasts_with_runtimes(CombinedHREFSREF.forecasts_day_spc_calibrated_with_sig_gated(), HREFPredictionAblations2.forecasts_day()), ablation2_model_names; run_hour = 12, suffix = "_href_ablations2_absolutely_calibrated")


function sum_probs(spc_forecasts, forecasts, model_names; run_hour, suffix, start = Dates.DateTime(2000, 1, 1, 1), cutoff = Dates.DateTime(2022, 6, 1, 12), only_models = model_names, all_days_of_week = false)

  println("************ $(run_hour)z$(suffix) ************")

  println("$(length(spc_forecasts)) SPC forecasts available") #

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  if all_days_of_week
    test_forecasts = vcat(train_forecasts, validation_forecasts, test_forecasts)
  end

  println("$(length(test_forecasts)) unfiltered test forecasts") #
  test_forecasts = filter(forecast -> forecast.run_hour == run_hour, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts") #

  # We don't have storm events past this time.

  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff && Forecasts.valid_utc_datetime(forecast) >= start, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts before the event data cutoff date") #

  event_name_to_unadj_events = Dict(
    "tornado"     => StormEvents.conus_tornado_events(),
    "wind"        => StormEvents.conus_severe_wind_events(),
    "hail"        => StormEvents.conus_severe_hail_events(),
    "sig_tornado" => StormEvents.conus_sig_tornado_events(),
    "sig_wind"    => StormEvents.conus_sig_wind_events(),
    "sig_hail"    => StormEvents.conus_sig_hail_events(),
  )

  # is_ablation(model_name) = occursin(r"\Atornado_.+_\d+\z", model_name)
  model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1")
  unadj_name(event_name) = replace(event_name, r"_adj\z" => "")

  compute_forecast_labels(model_name, spc_forecast) = begin
    # Annoying that we have to recalculate this.
    start_seconds =
      if spc_forecast.run_hour == 6
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
      elseif spc_forecast.run_hour == 13
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
      elseif spc_forecast.run_hour == 16
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
      end
    end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
    println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
    utc_datetime = Dates.unix2datetime(start_seconds)
    println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
    println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
    window_half_size = (end_seconds - start_seconds) ÷ 2
    window_mid_time  = (end_seconds + start_seconds) ÷ 2
    event_name = model_name_to_event_name(model_name)
    if endswith(event_name, "wind_adj")
      measured_events, estimated_events, gridded_normalization =
        if event_name == "wind_adj"
          StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), TrainingShared.day_estimated_wind_gridded_normalization()
        elseif event_name == "sig_wind_adj"
          StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), TrainingShared.day_estimated_sig_wind_gridded_normalization()
        else
          error("unknown adj event $event_name")
        end
      measured_labels  = StormEvents.grid_to_event_neighborhoods(measured_events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, spc_forecast.grid, Grid130.GRID_130_CROPPED, gridded_normalization, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
      max.(measured_labels, estimated_labels)
    else
      events = event_name_to_unadj_events[event_name]
      StormEvents.grid_to_event_neighborhoods(events, spc_forecast.grid, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
    end
  end

  ndays               = 0
  spc_total_probs     = Dict()
  test_total_probs    = Dict()
  reports_total_probs = Dict()

  for spc_forecast in spc_forecasts
    test_forecast_i = findfirst(forecast -> (forecast.run_year, forecast.run_month, forecast.run_day) == (spc_forecast.run_year, spc_forecast.run_month, spc_forecast.run_day), test_forecasts)
    if isnothing(test_forecast_i)
      continue
    end
    ndays += 1
    test_forecast = test_forecasts[test_forecast_i]

    spc_data  = Forecasts.data(spc_forecast)
    ForecastCombinators.turn_forecast_caching_on()
    test_data = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    for model_name in model_names
      model_name in only_models || continue
      event_name   = model_name_to_event_name(model_name)
      spc_event_i  = findfirst(m -> m[1] == unadj_name(event_name), SPCOutlooks.models)
      test_event_i = findfirst(isequal(model_name), model_names)

      spc_event_probs  = @view spc_data[:, spc_event_i]
      test_event_probs = @view test_data[:, test_event_i]

      forecast_labels = compute_forecast_labels(model_name, spc_forecast)

      if !(model_name in keys(spc_total_probs))
        spc_total_probs[model_name] = spc_event_probs[:]
      else
        spc_total_probs[model_name] .+= spc_event_probs
      end

      if !(model_name in keys(test_total_probs))
        test_total_probs[model_name] = test_event_probs[:]
      else
        test_total_probs[model_name] .+= test_event_probs
      end

      if !(model_name in keys(reports_total_probs))
        reports_total_probs[model_name] = forecast_labels[:]
      else
        reports_total_probs[model_name] .+= forecast_labels
      end
    end

    print(".")
  end
  println()


  for model_name in model_names
    model_name in only_models || continue

    open((@__DIR__) * "/total_prob_spc_$(model_name)_$(ndays)_days_$(run_hour)z$(suffix).csv", "w") do csv
      headers = ["lat", "lon", "total_prob"]
      println(csv, join(headers, ","))
      for ((lat, lon), total_prob) in zip(spc_forecasts[1].grid.latlons, spc_total_probs[model_name])
        println(csv, join(Float32[lat, lon, total_prob], ","))
      end
    end

    open((@__DIR__) * "/total_prob_nadocast_$(model_name)_$(ndays)_days_$(run_hour)z$(suffix).csv", "w") do csv
      headers = ["lat", "lon", "total_prob"]
      println(csv, join(headers, ","))
      for ((lat, lon), total_prob) in zip(test_forecasts[1].grid.latlons, test_total_probs[model_name])
        println(csv, join(Float32[lat, lon, total_prob], ","))
      end
    end

    open((@__DIR__) * "/total_prob_reports_$(model_name)_$(ndays)_days_$(run_hour)z$(suffix).csv", "w") do csv
      headers = ["lat", "lon", "total_prob"]
      println(csv, join(headers, ","))
      for ((lat, lon), total_prob) in zip(spc_forecasts[1].grid.latlons, reports_total_probs[model_name])
        println(csv, join(Float32[lat, lon, total_prob], ","))
      end
    end
  end
end

# FORECAST_DISK_PREFETCH=false FORECASTS_ROOT=~/nadocaster2 TASKS=[33,34] julia -t 16 --project=.. Test.jl

model_names = map(m -> m[3], HREFPrediction.models_with_gated)
33 in TASKS && sum_probs(SPCOutlooks.forecasts_day_1300(), HREFPrediction.forecasts_day_with_sig_gated(),                model_names; run_hour = 12, suffix = "_absolutely_calibrated", only_models = ["wind", "wind_adj"], all_days_of_week = false)
34 in TASKS && sum_probs(SPCOutlooks.forecasts_day_1300(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 12, suffix = "_spc_calibrated",        only_models = ["wind", "wind_adj"], all_days_of_week = false)


function sum_probs_one(forecasts, model_names; run_hour, suffix, start = Dates.DateTime(2000, 1, 1, 1), cutoff = Dates.DateTime(2022, 6, 1, 12), only_models = model_names, all_days_of_week = false)

  println("************ $(run_hour)z$(suffix) ************")

  println("$(length(forecasts)) forecasts available") #

  (train_forecasts, validation_forecasts, test_forecasts) =
    TrainingShared.forecasts_train_validation_test(
      ForecastCombinators.resample_forecasts(forecasts, Grids.get_upsampler, GRID);
      just_hours_near_storm_events = false
    );

  if all_days_of_week
    test_forecasts = vcat(train_forecasts, validation_forecasts, test_forecasts)
  end

  println("$(length(test_forecasts)) unfiltered test forecasts") #
  test_forecasts = filter(forecast -> forecast.run_hour == run_hour, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts") #

  # We don't have storm events past this time.

  test_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff && Forecasts.valid_utc_datetime(forecast) >= start, test_forecasts);
  println("$(length(test_forecasts)) $(run_hour)z test forecasts before the event data cutoff date") #

  println(join(sort(Forecasts.time_title.(test_forecasts)), ", "))

  event_name_to_unadj_events = Dict(
    "tornado"     => StormEvents.conus_tornado_events(),
    "wind"        => StormEvents.conus_severe_wind_events(),
    "hail"        => StormEvents.conus_severe_hail_events(),
    "sig_tornado" => StormEvents.conus_sig_tornado_events(),
    "sig_wind"    => StormEvents.conus_sig_wind_events(),
    "sig_hail"    => StormEvents.conus_sig_hail_events(),
  )

  ndays            = 0
  test_total_probs = Dict()

  # is_ablation(model_name) = occursin(r"\Atornado_.+_\d+\z", model_name)
  model_name_to_event_name(model_name) = replace(model_name, r"_gated_by_\w+" => "", r"\A(tornado|wind|hail)_.+_\d+\z" => s"\1")
  unadj_name(event_name) = replace(event_name, r"_adj\z" => "")

  for test_forecast in test_forecasts
    ForecastCombinators.turn_forecast_caching_on()
    test_data = Forecasts.data(test_forecast)
    ForecastCombinators.clear_cached_forecasts()

    for model_name in model_names
      model_name in only_models || continue
      test_event_i = findfirst(isequal(model_name), model_names)

      test_event_probs = @view test_data[:, test_event_i]

      if !(model_name in keys(test_total_probs))
        test_total_probs[model_name] = test_event_probs[:]
      else
        test_total_probs[model_name] .+= test_event_probs
      end
    end

    ndays += 1
    print(".")
  end
  println()

  for model_name in model_names
    model_name in only_models || continue

    open((@__DIR__) * "/total_prob_$(model_name)_$(ndays)_days_$(run_hour)z$(suffix).csv", "w") do csv
      headers = ["lat", "lon", "total_prob"]
      println(csv, join(headers, ","))
      for ((lat, lon), total_prob) in zip(test_forecasts[1].grid.latlons, test_total_probs[model_name])
        println(csv, join(Float32[lat, lon, total_prob], ","))
      end
    end
  end
end

# FORECAST_DISK_PREFETCH=false FORECASTS_ROOT=~/nadocaster2 TASKS=[35] julia -t 16 --project=.. Test.jl

model_names = first.(SPCOutlooks.models)
35 in TASKS && sum_probs_one(SPCOutlooks.forecasts_day_1300(), model_names; run_hour = 13, suffix = "_spc_all", only_models = ["wind"], all_days_of_week = true, cutoff = Dates.DateTime(2022, 1, 1, 12))

# HREF-only models vs. 13Z spc

# FORECAST_DISK_PREFETCH=false TASKS=[36] julia -t 16 --project=.. Test.jl

model_names = map(m -> m[3], HREFPrediction.models_with_gated)
36 in TASKS && do_it(SPCOutlooks.forecasts_day_1300(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 12, suffix = "_href_only_more_days_vs_13Z_spc")

# scp nadocaster2:~/nadocast_dev/test/stats_12z_href_only_more_days_vs_13Z_spc.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_12z_href_only_more_days_vs_13Z_spc.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_reliability_12z_href_only_more_days_vs_13Z_spc.csv ./



# HREF-only versus SPC but for 183 days (back to 2018/7 instead of 2019/1)

# FORECAST_DISK_PREFETCH=false TASKS=[37] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false USE_ALT_DISK=true TASKS=[38] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl

37 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 0,  suffix = "_href_only_more_days")
38 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated(), model_names; run_hour = 12, suffix = "_href_only_more_days")

# scp nadocaster2:~/nadocast_dev/test/stats_0z_href_only_more_days.csv ./
# scp nadocaster2:~/nadocast_dev/test/stats_12z_href_only_more_days.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_0z_href_only_more_days.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_12z_href_only_more_days.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_reliability_0z_href_only_more_days.csv ./
# scp nadocaster2:~/nadocast_dev/test/test_reliability_12z_href_only_more_days.csv ./



# Testing more recent HREF-only performance vs SPC, since Nadocast was implemented at SPC

# FORECAST_DISK_PREFETCH=false TASKS=[39] julia -t 16 --project=.. Test.jl
# FORECAST_DISK_PREFETCH=false USE_ALT_DISK=true TASKS=[40] julia -t 16 --project=.. Test.jl

(39 in TASKS || 40 in TASKS) && begin

  start  = Dates.DateTime(2022, 7, 1, 12)
  cutoff = Dates.DateTime(2023, 4, 1, 12)

  # Can't remember which date in "late June" 2022 the implementation was announced
  forecasts_after_nadocast_implemented_at_spc = filter(fcst -> Forecasts.valid_utc_datetime(fcst) > start, HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated())

  39 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), forecasts_after_nadocast_implemented_at_spc, model_names; run_hour = 0,  cutoff = cutoff, suffix = "_href_only_since_spc_implementation", use_train_validation_too = true)
  40 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), forecasts_after_nadocast_implemented_at_spc, model_names; run_hour = 12, cutoff = cutoff, suffix = "_href_only_since_spc_implementation", use_train_validation_too = true)

  # scp nadocaster2:~/nadocast_dev/test/stats_0z_href_only_since_spc_implementation.csv ./
  # scp nadocaster2:~/nadocast_dev/test/stats_12z_href_only_since_spc_implementation.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_0z_href_only_since_spc_implementation.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_12z_href_only_since_spc_implementation.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_0z_href_only_since_spc_implementation.csv ./
  # scp nadocaster2:~/nadocast_dev/test/test_reliability_12z_href_only_since_spc_implementation.csv ./
end


# FORECAST_DISK_PREFETCH=false TASKS=[41] DRAW_SPC_MAPS=false julia -t 16 --project=.. Test.jl
(41 in TASKS || 42 in TASKS || 43 in TASKS|| 44 in TASKS) && begin

  import DelimitedFiles

  training_end  = Dates.DateTime(2022, 6, 1, 12)
  cutoff = Dates.DateTime(2023, 1, 1, 12) # Don't have 2023 TCs yet

  non_training_forecasts = filter(fcst -> TrainingShared.is_test(fcst) || Forecasts.valid_utc_datetime(fcst) > training_end, HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated())

  const tc_segments_path = joinpath(@__DIR__, "..", "tropical_cyclones", "tropical_cyclones_2018-2022.csv")
  tc_rows, tc_headers = DelimitedFiles.readdlm(tc_segments_path, ',', String; header=true)

  # begin_time_str,begin_time_seconds,end_time_str,end_time_seconds,id,name,status,knots,max_radius_34_knot_winds_nmiles,begin_lat,begin_lon,end_lat,end_lon
  tc_headers = tc_headers[1,:]

  start_seconds_col_i = findfirst(isequal("begin_time_seconds"), tc_headers)
  end_seconds_col_i   = findfirst(isequal("end_time_seconds"),   tc_headers)
  start_lat_col_i     = findfirst(isequal("begin_lat"),          tc_headers)
  start_lon_col_i     = findfirst(isequal("begin_lon"),          tc_headers)
  end_lat_col_i       = findfirst(isequal("end_lat"),            tc_headers)
  end_lon_col_i       = findfirst(isequal("end_lon"),            tc_headers)
  status_col_i        = findfirst(isequal("status"),             tc_headers)

  struct TCSegment
    start_seconds_from_epoch_utc :: Int64
    end_seconds_from_epoch_utc   :: Int64
    start_latlon                 :: Tuple{Float64, Float64}
    end_latlon                   :: Tuple{Float64, Float64}
    status                       :: String # TS and HU are the ones we care about
  end

  row_to_tc(row) = TCSegment(
    parse(Int64, row[start_seconds_col_i]),
    parse(Int64, row[end_seconds_col_i]),
    parse.(Float64, (row[start_lat_col_i], row[start_lon_col_i])),
    parse.(Float64, (row[end_lat_col_i],   row[end_lon_col_i])),
    row[status_col_i],
  )

  tc_segs = mapslices(row_to_tc, tc_rows, dims = [2])[:,1]
  const hurricane_and_tropical_storm_segments = filter(seg -> seg.status == "TS" || seg.status == "HU", tc_segs)

  function compute_tc_grid(spc_forecast)
    radius_mi = 500.0

    # Annoying that we have to recalculate this.
    start_seconds =
      if spc_forecast.run_hour == 6
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 6*HOUR
      elseif spc_forecast.run_hour == 13
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast)
      elseif spc_forecast.run_hour == 16
        Forecasts.run_time_in_seconds_since_epoch_utc(spc_forecast) + 30*MINUTE
      end
    end_seconds = Forecasts.valid_time_in_seconds_since_epoch_utc(spc_forecast) + HOUR
    # println(Forecasts.yyyymmdd_thhz_fhh(spc_forecast))
    # utc_datetime = Dates.unix2datetime(start_seconds)
    # println(Printf.@sprintf "%04d%02d%02d_%02dz" Dates.year(utc_datetime) Dates.month(utc_datetime) Dates.day(utc_datetime) Dates.hour(utc_datetime))
    # println(Forecasts.valid_yyyymmdd_hhz(spc_forecast))
    window_half_size = (end_seconds - start_seconds) ÷ 2
    window_mid_time  = (end_seconds + start_seconds) ÷ 2

    0.5f0 .< StormEvents.grid_to_event_neighborhoods(hurricane_and_tropical_storm_segments, spc_forecast.grid, radius_mi, window_mid_time, window_half_size)
  end

  41 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), non_training_forecasts, model_names; run_hour = 0,  cutoff = cutoff, suffix = "_href_only_near_tc", use_train_validation_too = true, compute_extra_mask = compute_tc_grid)
  42 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), non_training_forecasts, model_names; run_hour = 12, cutoff = cutoff, suffix = "_href_only_near_tc", use_train_validation_too = true, compute_extra_mask = compute_tc_grid)
  43 in TASKS && do_it(SPCOutlooks.forecasts_day_0600(), non_training_forecasts, model_names; run_hour = 0,  cutoff = cutoff, suffix = "_href_only_not_near_tc", use_train_validation_too = true, compute_extra_mask = spc_fcst -> (!).(compute_tc_grid(spc_fcst)))
  44 in TASKS && do_it(SPCOutlooks.forecasts_day_1630(), non_training_forecasts, model_names; run_hour = 12, cutoff = cutoff, suffix = "_href_only_not_near_tc", use_train_validation_too = true, compute_extra_mask = spc_fcst -> (!).(compute_tc_grid(spc_fcst)))
end

# now look for hail hotspots
(45 in TASKS) && begin

  training_end  = Dates.DateTime(2022, 6, 1, 12)
  cutoff = Dates.DateTime(2023, 10, 1, 12)

  non_training_forecasts = filter(fcst -> TrainingShared.is_test(fcst) || Forecasts.valid_utc_datetime(fcst) > training_end, HREFPrediction.forecasts_day_spc_calibrated_with_sig_gated())

  45 in TASKS && sum_probs(SPCOutlooks.forecasts_day_0600(), non_training_forecasts, model_names; cutoff = cutoff, run_hour = 0, suffix = "_spc_calibrated", all_days_of_week = true)
end