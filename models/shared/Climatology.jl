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



function tornado_day_spacial_probability_feature(grid)
  climatology_on_grid = load_climatology_on_grid("tornado_day_climatological_probability", grid)
  ("tornado_day_climatological_spacial_prob", _ -> climatology_on_grid)
end

function severe_day_spacial_probability_feature(grid)
  climatology_on_grid = load_climatology_on_grid("severe_day_climatological_probability", grid)
  ("severe_day_climatological_spacial_prob", _ -> climatology_on_grid)
end

function tornado_day_given_severe_day_spacial_probability_feature(grid)
  climatology_on_grid = load_climatology_on_grid("tornado_day_given_severe_day_climatological_probability", grid)
  ("tornado_day_given_severe_day_spacial_prob", _ -> climatology_on_grid)
end

function geomean_tornado_and_conditional_spacial_probability_feature(grid)
  climatology_on_grid = load_climatology_on_grid("geomean_absolute_and_conditional_climatological_probability", grid)
  ("geomean_tornado_and_conditional_spacial_prob", _ -> climatology_on_grid)
end

# Hour in day	Tornado Prob	Severe Prob
# 0	0.00011281105589407664	0.0016023082155621861
# 1	9.542421364076437e-5	0.0013623035856667757
# 2	7.715274555775172e-5	0.0010855538933892944
# 3	4.6671249398663465e-5	0.0008256739049214956
# 4	3.8775537478660835e-5	0.0006386233952509774
# 5	3.5468275746195707e-5	0.0005312684547429962
# 6	2.7773615089433353e-5	0.0004403691211915202
# 7	2.4574173641975926e-5	0.0003915619648709643
# 8	2.0915068390769666e-5	0.00035983595485939704
# 9	2.266193139047085e-5	0.00033027840031592574
# 10	2.79491412942416e-5	0.0003368885868618107
# 11	3.209419466673707e-5	0.0003818199314824443
# 12	4.093512080238398e-5	0.0004920051141764673
# 13	5.63113670579055e-5	0.0007253110513502339
# 14	7.970223364709648e-5	0.0010512935636334706
# 15	0.00010412373680106372	0.0013813556459886616
# 16	0.00012957508827892515	0.001696731311387437
# 17	0.00014456868869499806	0.001894080868249279
# 18	0.00015874903775936556	0.0019672777784898177
# 19	0.00015525202952999288	0.0020145104648122397
# 20	0.00013703402383647353	0.001977639225067383
# 21	0.00012021929890633081	0.001956761225011932
# 22	0.00012540520675259194	0.001927498480735839
# 23	0.00012211326059844427	0.001794125630938225

hour_i_to_tornado_prob = Float32[
  0.00011281105589407664,
  9.542421364076437e-5,
  7.715274555775172e-5,
  4.6671249398663465e-5,
  3.8775537478660835e-5,
  3.5468275746195707e-5,
  2.7773615089433353e-5,
  2.4574173641975926e-5,
  2.0915068390769666e-5,
  2.266193139047085e-5,
  2.79491412942416e-5,
  3.209419466673707e-5,
  4.093512080238398e-5,
  5.63113670579055e-5,
  7.970223364709648e-5,
  0.00010412373680106372,
  0.00012957508827892515,
  0.00014456868869499806,
  0.00015874903775936556,
  0.00015525202952999288,
  0.00013703402383647353,
  0.00012021929890633081,
  0.00012540520675259194,
  0.00012211326059844427,
]

hour_i_to_severe_prob = Float32[
  0.0016023082155621861,
  0.0013623035856667757,
  0.0010855538933892944,
  0.0008256739049214956,
  0.0006386233952509774,
  0.0005312684547429962,
  0.0004403691211915202,
  0.0003915619648709643,
  0.00035983595485939704,
  0.00033027840031592574,
  0.0003368885868618107,
  0.0003818199314824443,
  0.0004920051141764673,
  0.0007253110513502339,
  0.0010512935636334706,
  0.0013813556459886616,
  0.001696731311387437,
  0.001894080868249279,
  0.0019672777784898177,
  0.0020145104648122397,
  0.001977639225067383,
  0.001956761225011932,
  0.001927498480735839,
  0.001794125630938225,
]

function forecast_hour_tornado_probability_feature(grid)
  ( "forecast_hour_climatological_tornado_prob"
  , forecast -> fill_grid(hour_i_to_tornado_prob[1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

function forecast_hour_severe_probability_feature(grid)
  ( "forecast_hour_climatological_severe_prob"
  , forecast -> fill_grid(hour_i_to_severe_prob[1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

function forecast_hour_tornado_given_severe_probability_feature(grid)
  make_feature_data(forecast) = begin
    hour_i = 1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))
    fill_grid(hour_i_to_tornado_prob[hour_i] / hour_i_to_severe_prob[hour_i], grid)
  end
  ("forecast_hour_climatological_tornado_given_severe_prob", make_feature_data)
end

function forecast_hour_geomean_tornado_and_conditional_probability_feature(grid)
  make_feature_data(forecast) = begin
    hour_i      = 1 + Dates.hour(Forecasts.valid_utc_datetime(forecast))
    tor_prob    = hour_i_to_tornado_prob[hour_i]
    conditional = tor_prob / hour_i_to_severe_prob[hour_i]
    fill_grid(sqrt(tor_prob*conditional), grid)
  end
  ("forecast_hour_geomean_tornado_and_conditional_prob", make_feature_data)
end


# Month	Tornado Day Prob	Severe Day Prob
# 1	0.0006040794235708681	0.0033315314012978435
# 2	0.0006566448193920728	0.005525090396417805
# 3	0.0011690501952992978	0.0106942747404591
# 4	0.002689031711471158	0.021670062450983234
# 5	0.003601540094162475	0.0356165042151227
# 6	0.0031561038846052185	0.04797686001522021
# 7	0.0016031769612964244	0.03902905429465588
# 8	0.0011273924818847902	0.02806697548060397
# 9	0.0010541411218898957	0.01074848441613435
# 10	0.0008076681769902537	0.005483847465320244
# 11	0.0007586494447008108	0.0037478221231480326
# 12	0.0004647142381289501	0.0025546124759955094

month_i_to_tornado_day_prob = Float32[
  0.0006040794235708681,
  0.0006566448193920728,
  0.0011690501952992978,
  0.002689031711471158,
  0.003601540094162475,
  0.0031561038846052185,
  0.0016031769612964244,
  0.0011273924818847902,
  0.0010541411218898957,
  0.0008076681769902537,
  0.0007586494447008108,
  0.0004647142381289501,
]

month_i_to_severe_day_prob = Float32[
  0.0033315314012978435,
  0.005525090396417805,
  0.0106942747404591,
  0.021670062450983234,
  0.0356165042151227,
  0.04797686001522021,
  0.03902905429465588,
  0.02806697548060397,
  0.01074848441613435,
  0.005483847465320244,
  0.0037478221231480326,
  0.0025546124759955094,
]

function month_tornado_day_probability_feature(grid)
  ( "month_climatological_tornado_day_prob"
  , forecast -> fill_grid(month_i_to_tornado_day_prob[Dates.month(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

function month_severe_day_probability_feature(grid)
  ( "month_climatological_sever_day_prob"
  , forecast -> fill_grid(month_i_to_severe_day_prob[Dates.month(Forecasts.valid_utc_datetime(forecast))], grid)
  )
end

function month_tornado_day_given_severe_day_probability_feature(grid)
  make_feature_data(forecast) = begin
    month_i = Dates.month(Forecasts.valid_utc_datetime(forecast))
    fill_grid(month_i_to_tornado_day_prob[month_i] / month_i_to_severe_day_prob[month_i], grid)
  end
  ("month_climatological_tornado_given_severe_prob", make_feature_data)
end

function month_geomean_tornado_and_conditional_probability_feature(grid)
  make_feature_data(forecast) = begin
    month_i = Dates.month(Forecasts.valid_utc_datetime(forecast))
    tor_prob    = month_i_to_tornado_day_prob[month_i]
    conditional = tor_prob / month_i_to_severe_day_prob[month_i]
    fill_grid(sqrt(tor_prob*conditional), grid)
  end
  ("month_geomean_tornado_and_conditional_prob", make_feature_data)
end


end # module Climatology