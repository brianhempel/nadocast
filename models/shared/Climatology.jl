# Exposes the data generated in MakeClimatologicalBackground.jl

module Climatology

import Dates

push!(LOAD_PATH, (@__DIR__) * "/../../lib")

import Forecasts
import Grib2
import Grids

# push!(LOAD_PATH, (@__DIR__) * "/../../climatological_background_1998-2013")

climatology_data_dir = joinpath((@__DIR__), "..", "..", "climatological_background_1998-2013")

# Same cropping and 3x downsampling as in HREF.jl
HREF_CROPPED_15KM_GRID =
  Grib2.read_grid(
    (@__DIR__) * "/../../lib/href_one_field_for_grid.grib2",
    crop = ((1+214):(1473 - 99), (1+119):(1025-228)),
    downsample = 3
  ) :: Grids.Grid


function load_climatology_on_grid(climatology_file, grid)
  resampler = Grids.get_upsampler(HREF_CROPPED_15KM_GRID, grid) # not always upsampling, but this does nearest neighbor

  resampler(Float32.(reinterpret(Float16, read(joinpath(climatology_data_dir, climatology_file * ".float16.bin")))))
end

function fill_grid(val, grid)
  fill(Float32(val), length(grid.latlons))
end


function spatial_climatology_feature(file_name, grid)
  climatology_on_grid = load_climatology_on_grid(file_name, grid)
  feature_name        = replace(file_name, "_probability" => "_spatial_prob")
  (feature_name, _ -> climatology_on_grid)
end

hail_day_spatial_probability_feature(grid)                                         = spatial_climatology_feature("hail_day_climatological_probability",                                         grid)
hail_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)        = spatial_climatology_feature("hail_day_geomean_absolute_and_conditional_climatological_probability",        grid)
hail_day_given_severe_day_spatial_probability_feature(grid)                        = spatial_climatology_feature("hail_day_given_severe_day_climatological_probability",                        grid)
severe_day_spatial_probability_feature(grid)                                       = spatial_climatology_feature("severe_day_climatological_probability",                                       grid)
severe_day_geomean_absolute_and_conditional_spatial_probability_feature(grid)      = spatial_climatology_feature("severe_day_geomean_absolute_and_conditional_climatological_probability",      grid)
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


# I won't say the curves for the different event types are *exactly* the same, but they are rather similar, so we'll just include severe.

hour_i_to_severe_prob = Float32[0.0013185286942441264, 0.0011221491513765183, 0.0008926469915730582, 0.0006575419735763914, 0.0005053820531486138, 0.0004097323508702168, 0.0003288946034436646, 0.00028559558511602283, 0.00025563277021592267, 0.00023229462329994906, 0.00023604158405287468, 0.00026788452784516685, 0.0003516411273044998, 0.0005228534478340439, 0.0007735592168252087, 0.0010422597919820245, 0.0012988410806906055, 0.001471330940148913, 0.001529418493911422, 0.0015844557041777712, 0.0015582217500002705, 0.0015652654527321306, 0.0015540827701441612, 0.0014650361255499629]

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

month_i_to_tornado_day_prob     = Float32[0.0005912588636110609, 0.0006566448193920728, 0.0011690501952992978, 0.002689031711471158, 0.0035972006097341825, 0.003151486712053723, 0.0016031769612964244, 0.0011273924818847902, 0.001052397789801193, 0.0008061086130634434, 0.0007586494447008108, 0.0004647142381289501]
month_i_to_wind_day_prob        = Float32[0.0021701267130796137, 0.003332475149959834, 0.004692374485391373, 0.009747500693746249, 0.018235766346102678, 0.030558393430565007, 0.028044016789867364, 0.01953429401281888, 0.0064876948191625134, 0.003222887006037163, 0.002636611573482487, 0.0017016603113496978]
month_i_to_hail_day_prob        = Float32[0.0006473405429670652, 0.0014501671463444299, 0.004448432329871499, 0.00989113163733782, 0.015579595628507062, 0.01591402923807628, 0.009732367012624045, 0.00707567406691841, 0.0029677710902595803, 0.0014797863289096953, 0.000615034855603229, 0.0003636807326750079]
month_i_to_severe_day_prob      = Float32[0.0027865892080668757, 0.0044739155446376095, 0.008216050372064233, 0.01707046600204392, 0.02903485547966859, 0.04076942658887546, 0.03411450888782611, 0.024197575846163104, 0.00904617218782285, 0.004570236749062177, 0.0032848131398126247, 0.002176505090521792]
month_i_to_sig_tornado_day_prob = Float32[0.00014353889071694597, 0.00017995150442440733, 0.00028227656059078135, 0.0006169281846573612, 0.0006334680407315207, 0.00027851419032152346, 0.00011388875288482689, 8.44194127395308e-5, 0.00012614148625710476, 0.00012526349325472208, 0.0002346849759241835, 0.00011381850944044274]
month_i_to_sig_wind_day_prob    = Float32[0.00025011009872090174, 0.0003783928615320516, 0.0006006276086852014, 0.0013437343026582091, 0.002207070698867628, 0.0031492876293528014, 0.002680330531885248, 0.0017762392428364562, 0.0005580218796202924, 0.0002985255687100561, 0.0002476220613107017, 0.0001525428874219422]
month_i_to_sig_hail_day_prob    = Float32[7.916751542765838e-5, 0.00012174983348249795, 0.0005401873122813536, 0.0013791188222479019, 0.0021996989350041045, 0.001967426775966405, 0.001095181515045675, 0.0006832811409030232, 0.000316083101926938, 0.00012612148985031082, 4.4488606753264214e-5, 2.286536189910439e-5]
month_i_to_sig_severe_day_prob  = Float32[0.0004149323498785256, 0.0006245327622484127, 0.0012851295093226094, 0.0030218901807132913, 0.004581169622560005, 0.005052426692869783, 0.0037134880570005324, 0.002457232803345733, 0.0009503723115058557, 0.00052070

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