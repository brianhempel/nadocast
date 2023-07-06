import Dates
import Printf
# import Statistics

push!(LOAD_PATH, (@__DIR__) * "/../models/shared")
import TrainingShared

# push!(LOAD_PATH, (@__DIR__) * "/../models/spc_outlooks")
# import SPCOutlooks

# push!(LOAD_PATH, (@__DIR__) * "/../models/combined_href_sref")
# import CombinedHREFSREF

# push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction")
# import HREFPrediction

# push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations")
# import HREFPredictionAblations

# push!(LOAD_PATH, (@__DIR__) * "/../models/href_prediction_ablations2")
# import HREFPredictionAblations2

# push!(LOAD_PATH, (@__DIR__) * "/../models/href_day_experiment")
# import HREFDayExperiment

push!(LOAD_PATH, (@__DIR__) * "/../lib")
import Conus
import Forecasts
import Grids
import GeoUtils
import Grid130
# import PlotMap
import StormEvents
# import ForecastCombinators
using HREF15KMGrid

const CONUS_ON_HREF_CROPPED_15KM_GRID = Conus.is_in_conus.(HREF_CROPPED_15KM_GRID.latlons)

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

# GRID       = Conus.href_cropped_5km_grid();
# CONUS_MASK = Conus.conus_mask_href_cropped_5km_grid();

# VERIFIABLE_GRID_MASK = CONUS_MASK .&& TrainingShared.is_verifiable.(GRID.latlons) :: BitVector;

function convective_day_window_mid_time_and_half_size(date)
  start_seconds = Forecasts.time_in_seconds_since_epoch_utc(Dates.year(date), Dates.month(date), Dates.day(date), 12)
  end_seconds   = Forecasts.time_in_seconds_since_epoch_utc(Dates.year(date), Dates.month(date), Dates.day(date + Dates.Day(1)), 12)
  window_half_size = (end_seconds - start_seconds) ÷ 2
  window_mid_time  = (end_seconds + start_seconds) ÷ 2
  window_mid_time, window_half_size
end


event_name_to_unadj_events = Dict(
  "tornado"     => StormEvents.conus_tornado_events(),
  "wind"        => StormEvents.conus_severe_wind_events(),
  "hail"        => StormEvents.conus_severe_hail_events(),
  "sig_tornado" => StormEvents.conus_sig_tornado_events(),
  "sig_wind"    => StormEvents.conus_sig_wind_events(),
  "sig_hail"    => StormEvents.conus_sig_hail_events(),
)

function mean(xs)
  sum(xs) / length(xs)
end

compute_day_labels(event_name, date) = begin
  window_mid_time, window_half_size = convective_day_window_mid_time_and_half_size(date)
  if endswith(event_name, "wind_adj")
    measured_events, estimated_events, gridded_normalization =
      if event_name == "wind_adj"
        StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), TrainingShared.day_estimated_wind_gridded_normalization()
      elseif event_name == "sig_wind_adj"
        StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), TrainingShared.day_estimated_sig_wind_gridded_normalization()
      else
        error("unknown adj event $event_name")
      end
    measured_labels  = StormEvents.grid_to_event_neighborhoods(measured_events, HREF_CROPPED_15KM_GRID, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
    estimated_labels = StormEvents.grid_to_adjusted_event_neighborhoods(estimated_events, HREF_CROPPED_15KM_GRID, Grid130.GRID_130_CROPPED, gridded_normalization, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
    max.(measured_labels, estimated_labels)
  else
    events = event_name_to_unadj_events[event_name]
    StormEvents.grid_to_event_neighborhoods(events, HREF_CROPPED_15KM_GRID, TrainingShared.EVENT_SPATIAL_RADIUS_MILES, window_mid_time, window_half_size)
  end
end

day_event_latlonweights(event_name, date) = begin
  window_mid_time, window_half_size = convective_day_window_mid_time_and_half_size(date)
  if endswith(event_name, "wind_adj")
    measured_events, estimated_events, gridded_normalization =
      if event_name == "wind_adj"
        StormEvents.conus_measured_severe_wind_events(), StormEvents.conus_estimated_severe_wind_events(), TrainingShared.day_estimated_wind_gridded_normalization()
      elseif event_name == "sig_wind_adj"
        StormEvents.conus_measured_sig_wind_events(), StormEvents.conus_estimated_sig_wind_events(), TrainingShared.day_estimated_sig_wind_gridded_normalization()
      else
        error("unknown adj event $event_name")
      end
    measured_segments  = StormEvents.event_segments_around_time(measured_events,  window_mid_time, window_half_size)
    estimated_segments = StormEvents.event_segments_around_time(estimated_events, window_mid_time, window_half_size)
    measured_latlonweights = map(measured_segments) do ((lat1, lon1), (lat2, lon2))
      lat, lon = (mean([lat1, lat2]), mean([lon1, lon2]))
      (lat, lon, 1.0)
    end
    estimated_latlonweights = map(estimated_segments) do ((lat1, lon1), (lat2, lon2))
      # use mean reweighting of of start and end latlons
      # (this is how we did it for the climatology maps)
      factor1 = Grids.lookup_nearest(Grid130.GRID_130_CROPPED, gridded_normalization, (lat1, lon1))
      factor2 = Grids.lookup_nearest(Grid130.GRID_130_CROPPED, gridded_normalization, (lat2, lon2))
      factor = mean([factor1, factor2])
      lat, lon = (mean([lat1, lat2]), mean([lon1, lon2]))
      (lat, lon, factor)
    end
    vcat(measured_latlonweights, estimated_latlonweights)
  else
    events = event_name_to_unadj_events[event_name]
    segments = StormEvents.event_segments_around_time(events, window_mid_time, window_half_size)
    latlonweights = map(segments) do ((lat1, lon1), (lat2, lon2))
      lat, lon = (mean([lat1, lat2]), mean([lon1, lon2]))
      (lat, lon, 1.0)
    end
    latlonweights
  end
end

function gaussian_blur(grid, conus_bitmask, σ_km, vals; only_in_conus = false)
  mid_xi = grid.width ÷ 2
  mid_yi = grid.height ÷ 2

  # a box roughly 6*σ_km on each side
  radius_nx = findfirst(mid_xi:grid.width) do east_xi
    GeoUtils.instantish_distance(grid.latlons[Grids.get_grid_i(grid, (mid_yi, mid_xi))], grid.latlons[Grids.get_grid_i(grid, (mid_yi, east_xi))]) / 1000.0 > σ_km*3
  end
  radius_ny = findfirst(mid_yi:grid.height) do north_yi
    GeoUtils.instantish_distance(grid.latlons[Grids.get_grid_i(grid, (mid_yi, mid_xi))], grid.latlons[Grids.get_grid_i(grid, (north_yi, mid_xi))]) / 1000.0 > σ_km*3
  end

  # println(stderr, "σ_km = $(σ_km), radius_nx = $radius_nx, radius_ny = $radius_ny")

  out = zeros(Float64, size(vals))

  if σ_km == 0
    out[conus_bitmask] = vals[conus_bitmask]
    return out
  end

  for y1 in 1:grid.height
    Threads.@threads for x1 in 1:grid.width
      weight = eps(1.0)
      amount = 0.0
      i1 = Grids.get_grid_i(grid, (y1, x1))
      (!only_in_conus || conus_bitmask[i1]) || continue
      val_ll = grid.latlons[i1]
      for y2 in clamp(y1 - radius_ny, 1, grid.height):clamp(y1 + radius_ny, 1, grid.height)
        for x2 in clamp(x1 - radius_nx, 1, grid.width):clamp(x1 + radius_nx, 1, grid.width)
          i2 = Grids.get_grid_i(grid, (y2, x2))
          conus_bitmask[i2] || continue
          ll = grid.latlons[i2]
          meters = GeoUtils.instantish_distance(val_ll, ll)
          w = exp(-(meters/1000)^2 / (2 * σ_km^2)) * grid.point_areas_sq_miles[i2]
          amount += w * vals[i2]
          weight += w
        end
      end
      out[i1] = amount / weight
    end
  end

  out
end

function to_csv(event_name, date)
  data = compute_day_labels(event_name, date)
  σ_km = 120
  data_blurred = gaussian_blur(HREF_CROPPED_15KM_GRID, CONUS_ON_HREF_CROPPED_15KM_GRID, σ_km, data; only_in_conus = true)

  time_title = Printf.@sprintf "%04d-%02d-%02d" Dates.year(date) Dates.month(date) Dates.day(date)

  open("practically_perfect_$(event_name)_$(time_title).csv", "w") do csv
    println(csv, "lat,lon,prob")
    for ((lat,lon), prob) in zip(HREF_CROPPED_15KM_GRID.latlons, data_blurred)
      println(csv, "$lat,$lon,$prob")
    end
  end

  open("ground_truth_gridded_$(event_name)_$(time_title).csv", "w") do csv
    println(csv, "lat,lon,prob")
    for ((lat,lon), prob) in zip(HREF_CROPPED_15KM_GRID.latlons, data)
      println(csv, "$lat,$lon,$prob")
    end
  end

  open("events_$(event_name)_$(time_title).csv", "w") do csv
    println(csv, "lat,lon,weight")
    for (lat, lon, weight) in day_event_latlonweights(event_name, date)
      println(csv, "$lat,$lon,$weight")
    end
  end
end

# # worst = most sq miles within 25mi of a severe report
#
# # 2020-04-12 worst wind
# # 2022-06-20 worst wind_adj, sig_wind_adj
# # 2019-10-20 worst sig_wind

to_csv("wind", Dates.Date(2020,4,12))
to_csv("wind", Dates.Date(2022,6,20))
to_csv("wind", Dates.Date(2019,10,20))
to_csv("sig_wind", Dates.Date(2020,4,12))
to_csv("sig_wind", Dates.Date(2022,6,20))
to_csv("sig_wind", Dates.Date(2019,10,20))
to_csv("wind_adj", Dates.Date(2020,4,12))
to_csv("wind_adj", Dates.Date(2022,6,20))
to_csv("wind_adj", Dates.Date(2019,10,20))
to_csv("sig_wind_adj", Dates.Date(2020,4,12))
to_csv("sig_wind_adj", Dates.Date(2022,6,20))
to_csv("sig_wind_adj", Dates.Date(2019,10,20))
