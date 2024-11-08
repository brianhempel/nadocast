# Look at motion of composite reflectivity within 100mi of severe events.
# Which wind vector does it best follow?

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Conus
import GeoUtils
import Inventories

push!(LOAD_PATH, (@__DIR__) * "/../shared")

import StormEvents


include("HREF.jl")

forecasts = HREF.extra_features_forecasts();

f1_forecasts = filter(fcst -> fcst.forecast_hour == 1, forecasts);
f2_forecasts = filter(fcst -> fcst.forecast_hour == 2, forecasts);

f1_f2_forecasts_paired = []

for f1_fcst in f1_forecasts
  i = findfirst(f2_fcst -> Forecasts.run_year_month_day_hour(f1_fcst) == Forecasts.run_year_month_day_hour(f2_fcst), f2_forecasts)
  if !isnothing(i)
    f2_fcst = f2_forecasts[i]
    push!(f1_f2_forecasts_paired, (f1_fcst, f2_fcst))
  end
end

grid      = HREF.grid()
inventory = f1_forecasts[1]._get_inventory()

feature_key_to_i = Dict{String,Int64}()
feature_keys = map(Inventories.inventory_line_key, inventory)

for i in 1:length(inventory)
  feature_key_to_i[feature_keys[i]] = i
end

function get_layer(data, key)
  @view data[:, feature_key_to_i[key]]
end

const MINUTE = 60

function grid_to_labels(events, forecast)
  StormEvents.grid_to_event_neighborhoods(events, forecast.grid, 100.0, Forecasts.valid_time_in_seconds_since_epoch_utc(forecast), 30*MINUTE)
end

function try_it(factor)
  vector_wind_layers = [
    "GRD:250 mb:hour fcst:wt ens mean",
    "GRD:500 mb:hour fcst:wt ens mean",
    "GRD:700 mb:hour fcst:wt ens mean",
    "GRD:850 mb:hour fcst:wt ens mean",
    "GRD:925 mb:hour fcst:wt ens mean",
    "STM:calculated:hour fcst:", # Our computed Bunkers storm motion.
    "STM½:calculated:hour fcst:", # half as much deviation from the mean wind
    "½STM½500mb:calculated:hour fcst:",
    "SHEAR:calculated:hour fcst:",
    "MEAN:calculated:hour fcst:",
  ]

  layers_abs_dev = zeros(length(vector_wind_layers))
  layers_weight  = zeros(length(vector_wind_layers))

  for forecast_i in eachindex(f1_f2_forecasts_paired)
    print("\r$forecast_i/$(length(f1_f2_forecasts_paired))")
    f1_fcst, f2_fcst = f1_f2_forecasts_paired[forecast_i]
    data1 = Forecasts.data_or_nothing(f1_fcst)
    data2 = Forecasts.data_or_nothing(f2_fcst)
    !isnothing(data1) || continue
    !isnothing(data2) || continue
    refl1 = get_layer(data1, "REFC:entire atmosphere:hour fcst:estimated from probs")
    refl2 = get_layer(data2, "REFC:entire atmosphere:hour fcst:estimated from probs")

    grid_labeled1 = grid_to_labels(StormEvents.conus_severe_events(), f1_fcst)
    grid_labeled2 = grid_to_labels(StormEvents.conus_severe_events(), f2_fcst)

    for wind_layer_i in eachindex(vector_wind_layers)
      wind_layer_key = vector_wind_layers[wind_layer_i]
      u_key = "U" * wind_layer_key
      v_key = "V" * wind_layer_key

      us1    = get_layer(data1, u_key)
      vs1    = get_layer(data1, v_key)
      us2    = get_layer(data2, u_key)
      vs2    = get_layer(data2, v_key)

      width               = grid.width
      height              = grid.height
      latlons             = grid.latlons
      point_weights       = grid.point_weights
      point_widths_miles  = grid.point_widths_miles
      point_heights_miles = grid.point_heights_miles

      abs_devs = zeros(length(latlons))
      weights  = zeros(length(latlons))

      Threads.@threads :static for j in 1:height
        for i in 1:width
          flat_i = width*(j-1) + i

          if Conus.is_in_conus(latlons[flat_i])
            point_width_meters  = Float32(point_widths_miles[flat_i]  * GeoUtils.METERS_PER_MILE)
            point_height_meters = Float32(point_heights_miles[flat_i] * GeoUtils.METERS_PER_MILE)

            if grid_labeled1[flat_i] > 0.1f0
              # forward, f1 to f2
              u, v = us1[flat_i], vs1[flat_i]

              du = round(Int64, factor * u*60f0*60f0 / point_width_meters)
              dv = round(Int64, factor * v*60f0*60f0 / point_height_meters)

              i2 = clamp(i+du, 1, width)
              j2 = clamp(j+dv, 1, height)

              flat_i2 = width*(j2-1) + i2

              abs_dev = abs(refl2[flat_i2] - refl1[flat_i])

              abs_devs[flat_i] += abs_dev * point_weights[flat_i]
              weights[flat_i]  += point_weights[flat_i]
            end

            if grid_labeled2[flat_i] > 0.1f0
              # backward, f2 to f1
              u, v = us2[flat_i], vs2[flat_i]

              du = round(Int64, factor * u*60f0*60f0 / point_width_meters)
              dv = round(Int64, factor * v*60f0*60f0 / point_height_meters)

              i2 = clamp(i-du, 1, width)
              j2 = clamp(j-dv, 1, height)

              flat_i2 = width*(j2-1) + i2

              abs_dev = abs(refl1[flat_i2] - refl2[flat_i])

              abs_devs[flat_i] += abs_dev * point_weights[flat_i]
              weights[flat_i]  += point_weights[flat_i]
            end
          end
        end
      end

      layers_abs_dev[wind_layer_i] += sum(abs_devs)
      layers_weight[wind_layer_i]  += sum(weights)
    end
  end

  println()
  for wind_layer_i in eachindex(vector_wind_layers)
    wind_layer_key = vector_wind_layers[wind_layer_i]
    println("$wind_layer_key\t$(Float32(layers_abs_dev[wind_layer_i] / layers_weight[wind_layer_i]))")
  end
end

try_it(0.95f0);
# GRD:250 mb:hour fcst:wt ens mean  5.3206134
# GRD:500 mb:hour fcst:wt ens mean  4.289878
# GRD:700 mb:hour fcst:wt ens mean  4.3090906
# GRD:850 mb:hour fcst:wt ens mean  5.0702868
# GRD:925 mb:hour fcst:wt ens mean  5.436484
# STM:calculated:hour fcst:         4.47567
# STM½:calculated:hour fcst:        4.3612885
# ½STM½500mb:calculated:hour fcst:  4.1178074 *best*
# SHEAR:calculated:hour fcst:       4.4639482
# MEAN:calculated:hour fcst:        4.5022173

try_it(1f0);
# GRD:250 mb:hour fcst:wt ens mean  5.5074487
# GRD:500 mb:hour fcst:wt ens mean  4.3725104
# GRD:700 mb:hour fcst:wt ens mean  4.338505
# GRD:850 mb:hour fcst:wt ens mean  5.1156535
# GRD:925 mb:hour fcst:wt ens mean  5.4856634
# STM:calculated:hour fcst:         4.5193696
# STM½:calculated:hour fcst:        4.3771033
# ½STM½500mb:calculated:hour fcst:  4.153804 *
# SHEAR:calculated:hour fcst:       4.5594854
# MEAN:calculated:hour fcst:        4.5154295

try_it(1.1f0);
# GRD:250 mb:hour fcst:wt ens mean  5.8886676
# GRD:500 mb:hour fcst:wt ens mean  4.563737
# GRD:700 mb:hour fcst:wt ens mean  4.4184375
# GRD:850 mb:hour fcst:wt ens mean  5.2162657
# GRD:925 mb:hour fcst:wt ens mean  5.587692
# STM:calculated:hour fcst:         4.6195164
# STM½:calculated:hour fcst:        4.4206405
# ½STM½500mb:calculated:hour fcst:  4.2465105*
# SHEAR:calculated:hour fcst:       4.7773128
# MEAN:calculated:hour fcst:        4.554372

