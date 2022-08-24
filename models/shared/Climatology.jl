# Exposes the data generated in MakeClimatologicalBackground.jl

module Climatology

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids
using HREF15KMGrid
using Grid130

# push!(LOAD_PATH, (@__DIR__) * "/../../climatological_background_1998-2013")

climatology_data_dir      = joinpath((@__DIR__), "..", "..", "climatological_background_1998-2013")
asos_climatology_data_dir = joinpath((@__DIR__), "..", "..", "asos_climatology_2003-2021")

# population is basically climatology, since it determines the reporting rate
population_data_dir = joinpath((@__DIR__), "..", "..", "population")

function load_on_grid(dir, file_base_name, in_grid, out_grid)
  resampler = Grids.get_upsampler(in_grid, out_grid) # not always upsampling, but this does nearest neighbor

  resampler(Float32.(reinterpret(Float16, read(joinpath(dir, file_base_name * ".float16.bin")))))
end

function fill_grid(val, grid)
  fill(Float32(val), length(grid.latlons))
end

function spatial_climatology_feature(file_name, grid)
  climatology_on_grid = load_on_grid(climatology_data_dir, file_name, HREF_CROPPED_15KM_GRID, grid)
  feature_name        = replace(file_name, "_probability" => "_spatial_prob")
  (feature_name, _ -> climatology_on_grid)
end

function population_density_feature(grid)
  gridded = load_on_grid(population_data_dir, "pop_density_on_15km_grid", HREF_CROPPED_15KM_GRID, grid)
  ("people_per_sq_km", _ -> gridded)
end

# asos_gust_days_per_year_grid_130_cropped_blurred
# asos_sig_gust_days_per_year_grid_130_cropped_blurred
function asos_spatial_climatology_feature(file_name, grid)
  climatology_on_grid = load_on_grid(asos_climatology_data_dir, file_name, GRID_130_CROPPED, grid)
  feature_name        = replace(file_name, "_grid_130_cropped_blurred" => "")
  (feature_name, _ -> climatology_on_grid)
end


hail_day_spatial_probability_feature(grid)                                         = spatial_climatology_feature("hail_day_climatological_probability",                                         grid)
hail_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)        = spatial_climatology_feature("hail_day_geomean_absolute_and_conditional_climatological_probability",        grid)
hail_day_given_severe_day_spatial_probability_feature(grid)                        = spatial_climatology_feature("hail_day_given_severe_day_climatological_probability",                        grid)
severe_day_spatial_probability_feature(grid)                                       = spatial_climatology_feature("severe_day_climatological_probability",                                       grid)
sig_hail_day_spatial_probability_feature(grid)                                     = spatial_climatology_feature("sig_hail_day_climatological_probability",                                     grid)
sig_hail_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)    = spatial_climatology_feature("sig_hail_day_geomean_absolute_and_conditional_climatological_probability",    grid)
sig_hail_day_given_severe_day_spatial_probability_feature(grid)                    = spatial_climatology_feature("sig_hail_day_given_severe_day_climatological_probability",                    grid)
sig_severe_day_spatial_probability_feature(grid)                                   = spatial_climatology_feature("sig_severe_day_climatological_probability",                                   grid)
sig_severe_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)  = spatial_climatology_feature("sig_severe_day_geomean_absolute_and_conditional_climatological_probability",  grid)
sig_severe_day_given_severe_day_spatial_probability_feature(grid)                  = spatial_climatology_feature("sig_severe_day_given_severe_day_climatological_probability",                  grid)
sig_tornado_day_spatial_probability_feature(grid)                                  = spatial_climatology_feature("sig_tornado_day_climatological_probability",                                  grid)
sig_tornado_day_geomean_absolute_and_conditional_spatial_probability_feature(grid) = spatial_climatology_feature("sig_tornado_day_geomean_absolute_and_conditional_climatological_probability", grid)
sig_tornado_day_given_severe_day_spatial_probability_feature(grid)                 = spatial_climatology_feature("sig_tornado_day_given_severe_day_climatological_probability",                 grid)
sig_wind_day_spatial_probability_feature(grid)                                     = spatial_climatology_feature("sig_wind_day_climatological_probability",                                     grid)
sig_wind_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)    = spatial_climatology_feature("sig_wind_day_geomean_absolute_and_conditional_climatological_probability",    grid)
sig_wind_day_given_severe_day_spatial_probability_feature(grid)                    = spatial_climatology_feature("sig_wind_day_given_severe_day_climatological_probability",                    grid)
tornado_day_spatial_probability_feature(grid)                                      = spatial_climatology_feature("tornado_day_climatological_probability",                                      grid)
tornado_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)     = spatial_climatology_feature("tornado_day_geomean_absolute_and_conditional_climatological_probability",     grid)
tornado_day_given_severe_day_spatial_probability_feature(grid)                     = spatial_climatology_feature("tornado_day_given_severe_day_climatological_probability",                     grid)
wind_day_spatial_probability_feature(grid)                                         = spatial_climatology_feature("wind_day_climatological_probability",                                         grid)
wind_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)        = spatial_climatology_feature("wind_day_geomean_absolute_and_conditional_climatological_probability",        grid)
wind_day_given_severe_day_spatial_probability_feature(grid)                        = spatial_climatology_feature("wind_day_given_severe_day_climatological_probability",                        grid)

asos_gust_days_per_year_feature(grid)                                              = asos_spatial_climatology_feature("asos_gust_days_per_year_grid_130_cropped_blurred",                       grid)
asos_sig_gust_days_per_year_feature(grid)                                          = asos_spatial_climatology_feature("asos_sig_gust_days_per_year_grid_130_cropped_blurred",                   grid)


# I won't say the curves for the different event types are *exactly* the same, but they are rather similar, so we'll just include severe.

hour_i_to_severe_prob = Float32[0.001386736264994684, 0.001181139688679887, 0.000943544458305578, 0.0006980642458361038, 0.0005392923421321292, 0.000438512868137227, 0.00035341094931301244, 0.0003034618886429722, 0.00027125897075474967, 0.00024739765969603527, 0.000250806077033577, 0.0002850269006405788, 0.00036891985131610056, 0.0005498391027982735, 0.0008123599878051271, 0.0010988212247395632, 0.0013756905103398476, 0.0015541354838174903, 0.0016178430876226202, 0.00167347326477131, 0.0016449479168609681, 0.0016522666059494153, 0.00163435775069809, 0.0015394856806544921]

function hour_in_day_severe_probability_feature(grid)
  ( "hour_in_day_climatological_severe_prob"
  , forecast -> fill_grid(hour_i_to_severe_prob[1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

# function forecast_hour_tornado_given_severe_probability_feature(grid)
#   make_feature_data(forecast) = begin
#     hour_i = 1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))
#     fill_grid(hour_i_to_tornado_prob[hour_i] / hour_i_to_severe_prob[hour_i], grid)
#   end
#   ("forecast_hour_climatological_tornado_given_severe_prob", make_feature_data)
# end

# function forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)
#   make_feature_data(forecast) = begin
#     hour_i      = 1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))
#     tor_prob    = hour_i_to_tornado_prob[hour_i]
#     conditional = tor_prob / hour_i_to_severe_prob[hour_i]
#     fill_grid(sqrt(tor_prob*conditional), grid)
#   end
#   ("forecast_hour_geomean_tornado_and_conditional_prob", make_feature_data)
# end



# Eh, these curves are less similar...
# include Hail/Wind/Tor/Sev, and conditioned on severe

month_i_to_tornado_day_prob     = Float32[0.0005912588636110609, 0.0006566448193920728, 0.0011690501952992978, 0.0026897816853817284, 0.0035972006097341825, 0.0031528937801516455, 0.0016045015577131128, 0.0011301638718519643, 0.0010532646436953144, 0.0008074310719645224, 0.0007586494447008108, 0.0004647142381289501]
month_i_to_wind_day_prob        = Float32[0.0021742787126688887, 0.0033437091711604333, 0.004695487907537702, 0.009760425949617445, 0.018254639544583263, 0.030584483887483086, 0.028072308543725295, 0.01955194600286778, 0.006494150753787536, 0.003229257434317082, 0.0026405721266939473, 0.0017064056577864198]
month_i_to_hail_day_prob        = Float32[0.0006473405429670652, 0.0014506713245414835, 0.004449855622614641, 0.009894493501055387, 0.015590465765259015, 0.015924038298705954, 0.009739799417423558, 0.007078603129470401, 0.002971160427583693, 0.0014843636396106874, 0.0006178489530891205, 0.00036391667180627255]
month_i_to_severe_day_prob      = Float32[0.0027906819727792275, 0.004485391403851149, 0.008220348963084304, 0.017084536581618584, 0.029059158188577103, 0.0408017586442409, 0.03414646364574204, 0.024219324303074812, 0.009055027212928201, 0.004581523815163829, 0.0032915877905099764, 0.0021814863760897777]
month_i_to_sig_tornado_day_prob = Float32[0.00014353889071694597, 0.00017995150442440733, 0.00028227656059078135, 0.0006169281846573612, 0.0006334680407315207, 0.00027851419032152346, 0.00011388875288482689, 8.44194127395308e-5, 0.00012614148625710476, 0.00012526349325472208, 0.0002346849759241835, 0.00011381850944044274]
month_i_to_sig_wind_day_prob    = Float32[0.00025011009872090174, 0.0003783928615320516, 0.0006006276086852014, 0.0013452642029077315, 0.0022134991969843935, 0.0031492876293528014, 0.0026848405498328797, 0.001776510227050555, 0.0005593375630399119, 0.0002985255687100561, 0.0002476220613107017, 0.00015382449982362078]
month_i_to_sig_hail_day_prob    = Float32[7.916751542765838e-5, 0.00012174983348249795, 0.0005401873122813536, 0.0013791188222479019, 0.0021999192211872866, 0.0019687284328674657, 0.0010964530592716965, 0.0006832811409030232, 0.000316083101926938, 0.00012612148985031082, 4.4488606753264214e-5, 2.286536189910439e-5]
month_i_to_sig_severe_day_prob  = Float32[0.0004149323498785256, 0.0006245327622484127, 0.0012851295093226094, 0.0030234200809628135, 0.0045878184068599516, 0.005053728349770844, 0.0037185810688890133, 0.0024575037875598322, 0.0009516879949254754, 0.0005207052016221992, 0.00048408068763509054, 0.0002759490252219173]

function month_probability_feature(event_name, month_i_to_prob, grid)
  ( "month_climatological_$(event_name)_day_prob"
  , forecast -> fill_grid(month_i_to_prob[Dates.month(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

function month_probability_given_severe_day_feature(event_name, month_i_to_prob, grid)
  make_feature_data(forecast) = begin
    month_i = Dates.month(Forecasts.valid_utc_datetime(forecast))
    fill_grid(month_i_to_prob[month_i] / month_i_to_severe_day_prob[month_i], grid)
  end
  ( "month_climatological_$(event_name)_day_given_severe_day_prob"
  , make_feature_data
  )
end

# function month_geomean_tornado_and_conditional_probability_feature(grid)
#   make_feature_data(forecast) = begin
#     month_i = Dates.month(Forecasts.valid_utc_datetime(forecast))
#     tor_prob    = month_i_to_tornado_day_prob[month_i]
#     conditional = tor_prob / month_i_to_severe_day_prob[month_i]
#     fill_grid(sqrt(tor_prob*conditional), grid)
#   end
#   ("month_geomean_tornado_and_conditional_prob", make_feature_data)
# end

month_tornado_day_probability_feature(grid)                  = month_probability_feature(                 "tornado", month_i_to_tornado_day_prob, grid)
month_wind_day_probability_feature(grid)                     = month_probability_feature(                 "wind",    month_i_to_wind_day_prob,    grid)
month_hail_day_probability_feature(grid)                     = month_probability_feature(                 "hail",    month_i_to_hail_day_prob,    grid)
month_severe_day_probability_feature(grid)                   = month_probability_feature(                 "severe",  month_i_to_severe_day_prob,  grid)
month_tornado_day_given_severe_day_probability_feature(grid) = month_probability_given_severe_day_feature("tornado", month_i_to_tornado_day_prob, grid)
month_wind_day_given_severe_day_probability_feature(grid)    = month_probability_given_severe_day_feature("wind",    month_i_to_wind_day_prob,    grid)
month_hail_day_given_severe_day_probability_feature(grid)    = month_probability_given_severe_day_feature("hail",    month_i_to_hail_day_prob,    grid)

end # module Climatology