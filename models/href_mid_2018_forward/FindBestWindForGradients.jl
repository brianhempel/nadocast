push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Conus
import GeoUtils
import Inventories

include("HREF.jl")

forecasts = HREF.extra_features_forecasts()

f1_forecasts = filter(fcst -> fcst.forecast_hour == 1, forecasts)
f2_forecasts = filter(fcst -> fcst.forecast_hour == 2, forecasts)

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

function try_it(factor)
  vector_wind_layers = [
    "GRD:250 mb:hour fcst:wt ens mean",
    "GRD:500 mb:hour fcst:wt ens mean",
    "GRD:700 mb:hour fcst:wt ens mean",
    "GRD:850 mb:hour fcst:wt ens mean",
    "GRD:925 mb:hour fcst:wt ens mean",
    "STM:calculated:hour fcst:", # Our computed Bunkers storm motion.
    "STM½:calculated:hour fcst:", # half as much deviation from the mean wind
    "SHEAR:calculated:hour fcst:",
    "MEAN:calculated:hour fcst:",
  ]

  layers_abs_dev = zeros(length(vector_wind_layers))
  layers_weight  = zeros(length(vector_wind_layers))

  for forecast_i in eachindex(f1_f2_forecasts_paired)
    print("\r$forecast_i/$(length(f1_f2_forecasts_paired))")
    f1_fcst, f2_fcst = f1_f2_forecasts_paired[forecast_i]
    data1 = f1_fcst._get_data()
    data2 = f2_fcst._get_data()
    refl1 = get_layer(data1, "REFC:entire atmosphere:hour fcst:prob >30")
    refl2 = get_layer(data2, "REFC:entire atmosphere:hour fcst:prob >30")

    for wind_layer_i in eachindex(vector_wind_layers)
      wind_layer_key = vector_wind_layers[wind_layer_i]
      u_key = "U" * wind_layer_key
      v_key = "V" * wind_layer_key

      us    = get_layer(data1, u_key)
      vs    = get_layer(data2, v_key)

      width               = grid.width
      height              = grid.height
      latlons             = grid.latlons
      point_weights       = grid.point_weights
      point_widths_miles  = grid.point_widths_miles
      point_heights_miles = grid.point_heights_miles

      abs_devs = zeros(length(latlons))
      weights  = zeros(length(latlons))

      Threads.@threads for j in 1:height
        for i in 1:width
          flat_i = width*(j-1) + i

          if Conus.is_in_conus(latlons[flat_i]) && refl1[flat_i] > 10f0
            point_width_meters  = Float32(point_widths_miles[flat_i]  * GeoUtils.METERS_PER_MILE)
            point_height_meters = Float32(point_heights_miles[flat_i] * GeoUtils.METERS_PER_MILE)

            u, v = us[flat_i], vs[flat_i]

            du = round(Int64, factor * u*60f0*60f0 / point_width_meters)
            dv = round(Int64, factor * v*60f0*60f0 / point_height_meters)

            i2 = clamp(i+du, 1, width)
            j2 = clamp(j+dv, 1, height)

            flat_i2 = width*(j2-1) + i2

            abs_dev = abs(refl2[flat_i2] - refl1[flat_i])

            abs_devs[flat_i] = abs_dev * point_weights[flat_i]
            weights[flat_i]  = point_weights[flat_i]
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

try_it(1f0)
