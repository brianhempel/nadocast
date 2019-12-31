push!(LOAD_PATH, (@__DIR__) * "/../lib")

import Dates

import Conus
import Forecasts
import GeoUtils
import Grib2
import Grids
import PlotMap
import StormEvents

MINUTE = 60
HOUR   = StormEvents.HOUR
DAY    = StormEvents.DAY

NEIGHBORHOOD_RADIUS_MILES = 25

# Four evenly divides 16 years
CROSS_VALIDATION_FOLD_COUNT = 4


# Same cropping and 3x downsampling as in HREF.jl
HREF_CROPPED_15KM_GRID =
  Grib2.read_grid(
    (@__DIR__) * "/../lib/href_one_field_for_grid.grib2",
    crop = ((1+214):(1473 - 99), (1+119):(1025-228)),
    downsample = 3
  ) :: Grids.Grid

CONUS_ON_HREF_CROPPED_15KM_GRID = Conus.is_in_conus.(HREF_CROPPED_15KM_GRID.latlons)

function read_and_filter_events_csv(path)
  filter(StormEvents.read_events_csv(path)) do event
    Conus.is_in_conus_bounding_box(event.start_latlon) || Conus.is_in_conus_bounding_box(event.end_latlon)
  end
end

conus_tornadoes = begin
  println("Loading tornadoes...")
  read_and_filter_events_csv((@__DIR__) * "/../storm_data/tornadoes_1998-2013.csv")
end

conus_wind_events = begin
  println("Loading wind events...")
  read_and_filter_events_csv((@__DIR__) * "/../storm_data/wind_events_1998-2013.csv")
end

conus_hail_events = begin
  println("Loading hail events...")
  read_and_filter_events_csv((@__DIR__) * "/../storm_data/hail_events_1998-2013.csv")
end

convective_day_to_tornadoes     = Dict{Int64,Vector{StormEvents.Event}}()
convective_day_to_wind_events   = Dict{Int64,Vector{StormEvents.Event}}()
convective_day_to_hail_events   = Dict{Int64,Vector{StormEvents.Event}}()
hour_since_epoch_to_tornadoes   = Dict{Int64,Vector{StormEvents.Event}}()
hour_since_epoch_to_wind_events = Dict{Int64,Vector{StormEvents.Event}}()
hour_since_epoch_to_hail_events = Dict{Int64,Vector{StormEvents.Event}}()

# Mutates convective_day_to_events dictionary
function make_convective_day_to_events!(convective_day_to_events, events)
  for event in events
    for day_i in StormEvents.start_time_in_convective_days_since_epoch_utc(event):StormEvents.end_time_in_convective_days_since_epoch_utc(event)
      if !haskey(convective_day_to_events, day_i)
        convective_day_to_events[day_i] = StormEvents.Event[]
      end
      push!(convective_day_to_events[day_i], event)
    end
  end
end

# Mutates convective_day_to_events dictionary
function make_hour_since_epoch_to_events!(hour_since_epoch_to_events, events)
  for event in events
    for hour_i in fld(event.start_seconds_from_epoch_utc, HOUR):fld(event.end_seconds_from_epoch_utc, HOUR)
      if !haskey(hour_since_epoch_to_events, hour_i)
        hour_since_epoch_to_events[hour_i] = StormEvents.Event[]
      end
      push!(hour_since_epoch_to_events[hour_i], event)
    end
  end
end

make_convective_day_to_events!(convective_day_to_tornadoes,   conus_tornadoes)
make_convective_day_to_events!(convective_day_to_wind_events, conus_wind_events)
make_convective_day_to_events!(convective_day_to_hail_events, conus_hail_events)
make_hour_since_epoch_to_events!(hour_since_epoch_to_tornadoes,   conus_tornadoes)
make_hour_since_epoch_to_events!(hour_since_epoch_to_wind_events, conus_wind_events)
make_hour_since_epoch_to_events!(hour_since_epoch_to_hail_events, conus_hail_events)

# Depending on the storm events, the overall start/end times here may be off by a bit but the max error is trivial.
start_seconds_from_epoch = Forecasts.time_in_seconds_since_epoch_utc(1998, 1, 1, 0) - 12*HOUR
end_seconds_from_epoch   = Forecasts.time_in_seconds_since_epoch_utc(2014, 1, 1, 0) - 12*HOUR

fold_ranges_in_seconds_from_epoch = map(1:CROSS_VALIDATION_FOLD_COUNT) do fold_i
  seconds_per_fold = (end_seconds_from_epoch - start_seconds_from_epoch) / CROSS_VALIDATION_FOLD_COUNT
  seconds_per_fold = Int64(round(seconds_per_fold/DAY)) * DAY # count_events requires that each fold start precisely on a convective day boundary

  Int64(round(start_seconds_from_epoch + (fold_i-1)*seconds_per_fold)):Int64(round(start_seconds_from_epoch + fold_i*seconds_per_fold)-1)
end

# println(fold_ranges_in_seconds_from_epoch)

@assert first(fold_ranges_in_seconds_from_epoch).start == start_seconds_from_epoch
@assert last(fold_ranges_in_seconds_from_epoch).stop   == end_seconds_from_epoch - 1


# Adds 1 to each grid point that is within miles of any of the event segments. (Mulitple events do not accumulate more than 1.)
#
# Mutates counts_grid.
function count_neighborhoods!(counts_grid, grid, event_segments, miles)

  positive_grid_is = Set{Int64}()

  for (latlon1, latlon2) in event_segments
    event_grid_is = Grids.diamond_search(grid, Grids.latlon_to_closest_grid_i(grid, latlon1)) do candidate_latlon
      meters_away = GeoUtils.instant_meters_to_line(candidate_latlon, latlon1, latlon2)
      meters_away <= miles * GeoUtils.METERS_PER_MILE
    end

    push!(positive_grid_is, event_grid_is...)
  end

  for grid_i in positive_grid_is
    counts_grid[grid_i] += 1f0
  end

  ()
end


# Returns (day_count, tornado_day_counts_grid, event_day_counts_grid)
function count_events_by_day(range_in_seconds_from_epoch, grid, convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)
  day_count = 0

  tornado_day_counts_grid  = zeros(Float32, size(grid.latlons))
  event_day_counts_grid    = zeros(Float32, size(grid.latlons))

  for day_seconds_from_epoch in range_in_seconds_from_epoch.start:DAY:range_in_seconds_from_epoch.stop
    day_i = StormEvents.seconds_to_convective_days_since_epoch_utc(day_seconds_from_epoch)

    print(".")

    tornadoes = get(convective_day_to_tornadoes, day_i, StormEvents.Event[])
    events    = vcat(tornadoes, get(convective_day_to_wind_events, day_i, StormEvents.Event[]), get(convective_day_to_hail_events, day_i, StormEvents.Event[]))

    tornado_segments = StormEvents.event_segments_around_time(tornadoes, day_seconds_from_epoch + 12*HOUR, 12*HOUR)
    event_segments   = StormEvents.event_segments_around_time(events,    day_seconds_from_epoch + 12*HOUR, 12*HOUR)

    count_neighborhoods!(tornado_day_counts_grid, grid, tornado_segments, NEIGHBORHOOD_RADIUS_MILES)
    count_neighborhoods!(event_day_counts_grid,   grid, event_segments,   NEIGHBORHOOD_RADIUS_MILES)
    day_count += 1
  end
  println("")

  (day_count, tornado_day_counts_grid, event_day_counts_grid)
end

fold_day_tornado_and_event_counts =
  map(fold_ranges_in_seconds_from_epoch) do fold_range
    count_events_by_day(fold_range, HREF_CROPPED_15KM_GRID, convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)
  end

const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0

# Copied from Flux.jl.
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

# Returns (loss, weight).
function compute_loss(prediction_grid, counts_grid, day_count)
  prediction_grid = Float64.(prediction_grid)
  counts_grid     = Float64.(counts_grid)
  # Compute negatives and positives separately for a bit more numerical accuracy.
  positives_loss  = 0.0
  negatives_loss  = 0.0
  weight          = 0.0
  for grid_i in 1:length(HREF_CROPPED_15KM_GRID.latlons)
    if CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i]
      positives_loss += logloss(1.0, prediction_grid[grid_i]) * counts_grid[grid_i]               * HREF_CROPPED_15KM_GRID.point_weights[grid_i]
      negatives_loss += logloss(0.0, prediction_grid[grid_i]) * (day_count - counts_grid[grid_i]) * HREF_CROPPED_15KM_GRID.point_weights[grid_i]
      weight         += day_count * HREF_CROPPED_15KM_GRID.point_weights[grid_i]
    end
  end

  (positives_loss + negatives_loss, weight)
end


function meanify(in, mean_is)
  out = zeros(Float32, size(in))

  Threads.@threads for grid_i in 1:length(in)
    val = 0f0

    @inbounds for near_i in mean_is[grid_i]
      val += in[near_i]
    end

    # Further than the blur distance from CONUS
    if length(mean_is[grid_i]) == 0
      out[grid_i] = in[grid_i]
    else
      out[grid_i] = val / Float32(length(mean_is[grid_i]))
    end
  end

  out
end


# Returns loss
function try_parameters(transform_day_count_and_event_counts_to_prediction_grid)
  loss   = 0.0
  weight = 0.0
  for fold_i_to_predict in 1:CROSS_VALIDATION_FOLD_COUNT
    tornado_day_counts_grid = zeros(Float64, size(fold_day_tornado_and_event_counts[1][2]))
    day_count               = 0
    for fold_i in 1:CROSS_VALIDATION_FOLD_COUNT
      if fold_i != fold_i_to_predict
        fold_day_count, fold_tornado_day_counts_grid, _ = fold_day_tornado_and_event_counts[fold_i]
        tornado_day_counts_grid .+= fold_tornado_day_counts_grid
        day_count                += fold_day_count
      end
    end
    prediction_grid = transform_day_count_and_event_counts_to_prediction_grid(day_count, tornado_day_counts_grid)
    fold_to_predict_day_count, fold_to_predict_tornado_day_counts_grid, _ = fold_day_tornado_and_event_counts[fold_i_to_predict]
    fold_loss, fold_weight = compute_loss(prediction_grid, fold_to_predict_tornado_day_counts_grid, fold_to_predict_day_count)
    # println("Loss predicting fold $fold_i_to_predict:\t$(fold_loss/fold_weight)")
    loss   += fold_loss
    weight += fold_weight
  end

  loss / weight
end

function compute_conus_radius_grid_is(miles)
  map(Grids.radius_grid_is(HREF_CROPPED_15KM_GRID, miles)) do grid_mean_is
    filter(grid_i -> CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i], grid_mean_is)
  end
end

# Returns (total, total spacial weight)
function weighted_sum_conus(values)
  total  = 0.0
  weight = 0.0

  for grid_i in 1:length(HREF_CROPPED_15KM_GRID.latlons)
    if CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i]
      total  += values[grid_i] * HREF_CROPPED_15KM_GRID.point_weights[grid_i]
      weight +=                  HREF_CROPPED_15KM_GRID.point_weights[grid_i]
    end
  end

  (total, weight)
end

function compute_daily_mean_tor_prob(day_count, tornado_day_counts_grid)
  # println("day counts $day_count")

  tornado_weighted_spacial_counts, weight_total = weighted_sum_conus(tornado_day_counts_grid)

  tornado_weighted_spacial_counts / weight_total / day_count
end

# best_params = (0.010477507670158212, 50.0, 96.0) # loss, blur_radius, smoothing_strength
best_params = (Inf, nothing, nothing) # loss, blur_radius, smoothing_strength

# Best params: Blur radius 50.0	Smoothing strength 96.0	Loss: 0.010477507670158212

for blur_radius in [0.0, 10.0, 25.0, 35.0, 50.0, 70.0, 100.0]
  global best_params

  if blur_radius == 0.0
    blurrer = identity
  else
    println("Computing $(blur_radius)mi mean is...")
    mean_is = compute_conus_radius_grid_is(blur_radius)

    blurrer(tornado_day_counts_grid) = meanify(tornado_day_counts_grid, mean_is)
  end

  for smoothing_strength in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
    # println("Trying smoothing strength $smoothing_strength")
    transform_day_count_and_event_counts_to_prediction_grid(day_count, tornado_day_counts_grid) = begin
      background_prob = compute_daily_mean_tor_prob(day_count, tornado_day_counts_grid)
      # println("Background prob: $background_prob\texpected ~0.0014756858856650565")
      prediction_grid = tornado_day_counts_grid .+ (background_prob * smoothing_strength)
      prediction_grid ./= smoothing_strength + day_count
      blurrer(prediction_grid)
    end
    parameters_loss = try_parameters(transform_day_count_and_event_counts_to_prediction_grid)
    println("Blur radius $blur_radius\tSmoothing strength $smoothing_strength\tLoss: $parameters_loss")
    if parameters_loss < best_params[1]
      best_params = (parameters_loss, blur_radius, smoothing_strength)
    end
  end
end

best_loss, best_blur_radius, best_smoothing_strength = best_params

println("Best params: Blur radius $best_blur_radius\tSmoothing strength $best_smoothing_strength\tLoss: $best_loss")

tornado_day_counts_grid = zeros(Float64, size(fold_day_tornado_and_event_counts[1][2]))
event_day_counts_grid   = zeros(Float64, size(tornado_day_counts_grid))
day_count               = 0
for fold_i in 1:CROSS_VALIDATION_FOLD_COUNT
  global tornado_day_counts_grid
  global event_day_counts_grid
  global day_count
  fold_day_count, fold_tornado_day_counts_grid, fold_event_day_counts_grid = fold_day_tornado_and_event_counts[fold_i]
  tornado_day_counts_grid .+= fold_tornado_day_counts_grid
  event_day_counts_grid   .+= fold_event_day_counts_grid
  day_count                += fold_day_count
end

mean_is = compute_conus_radius_grid_is(best_blur_radius)

background_tornado_prob = compute_daily_mean_tor_prob(day_count, tornado_day_counts_grid)
background_event_prob   = compute_daily_mean_tor_prob(day_count, event_day_counts_grid)
println("Background tornado day prob: $background_tornado_prob\t(expected ~0.0014756858856650565)")
println("Background severe day prob: $background_event_prob")
tornado_prediction_grid = tornado_day_counts_grid .+ (background_tornado_prob * best_smoothing_strength)
tornado_prediction_grid ./= best_smoothing_strength + day_count
tornado_prediction_grid = meanify(tornado_prediction_grid, mean_is)

event_prediction_grid = event_day_counts_grid .+ (background_event_prob * best_smoothing_strength)
event_prediction_grid ./= best_smoothing_strength + day_count
event_prediction_grid = meanify(event_prediction_grid, mean_is)

tornado_day_given_event_day = tornado_prediction_grid ./ event_prediction_grid

geomean_absolute_and_conditional_probabilty = sqrt.(tornado_prediction_grid .* tornado_day_given_event_day)

println("Plotting Tornado Day Climatological Probability...")
PlotMap.plot_debug_map(
  "tornado_day_climatological_probability",
  HREF_CROPPED_15KM_GRID,
  tornado_prediction_grid;
  title="Tornado Day Probability 1998-2013",
  zlow=0.000,
  zhigh=0.006,
  steps=12
)
write("tornado_day_climatological_probability.float16.bin", Float16.(tornado_prediction_grid))

println("Plotting Severe Day Climatological Probability...")
PlotMap.plot_debug_map(
  "severe_day_climatological_probability",
  HREF_CROPPED_15KM_GRID,
  event_prediction_grid;
  title="Severe Day Probability 1998-2013",
  zlow=0.00,
  zhigh=0.06,
  steps=12
)
write("severe_day_climatological_probability.float16.bin", Float16.(event_prediction_grid))

println("Plotting p(TornadoDay|SevereDay)...")
PlotMap.plot_debug_map(
  "tornado_day_given_severe_day_climatological_probability",
  HREF_CROPPED_15KM_GRID,
  tornado_day_given_event_day;
  title="p(Tornado|SevereDay) 1998-2013",
  zlow=0.0,
  zhigh=0.6,
  steps=12
)
write("tornado_day_given_severe_day_climatological_probability.float16.bin", Float16.(tornado_day_given_event_day))

println("Plotting geomean of tor prob and p(TornadoDay|SevereDay)...")
PlotMap.plot_debug_map(
  "geomean_absolute_and_conditional_climatological_probability",
  HREF_CROPPED_15KM_GRID,
  geomean_absolute_and_conditional_probabilty;
  title="Geomean abs&cond prob 1998-2013",
  zlow=0.0,
  zhigh=0.04,
  steps=8
)
write("geomean_absolute_and_conditional_climatological_probability.float16.bin", Float16.(geomean_absolute_and_conditional_probabilty))


# Counts by hour

# Returns (hour_tornado_probs, hour_severe_probs)
function count_events_by_hour(range_in_seconds_from_epoch, hour_since_epoch_to_tornadoes, hour_since_epoch_to_wind_events, hour_since_epoch_to_hail_events)
  grid = HREF_CROPPED_15KM_GRID

  tornado_hour_counts_grids = map(_ -> zeros(Float32, size(grid.latlons)), 0:23)
  event_hour_counts_grids   = map(_ -> zeros(Float32, size(grid.latlons)), 0:23)
  hour_counts               = map(_ -> 0f0, 0:23)

  for hour_seconds_from_epoch in range_in_seconds_from_epoch.start:HOUR:range_in_seconds_from_epoch.stop
    hour_from_epoch = fld(hour_seconds_from_epoch, HOUR)
    hour_in_day     = mod(hour_from_epoch, 24)
    hour_in_day_i   = hour_in_day + 1
    # day_i = StormEvents.seconds_to_convective_days_since_epoch_utc(day_seconds_from_epoch)

    if hour_in_day == 0
      print(".")
    end

    tornadoes = vcat(
      get(hour_since_epoch_to_tornadoes, hour_from_epoch-1, StormEvents.Event[]),
      get(hour_since_epoch_to_tornadoes, hour_from_epoch, StormEvents.Event[])
    )
    events = vcat(
      tornadoes,
      get(hour_since_epoch_to_wind_events, hour_from_epoch-1, StormEvents.Event[]),
      get(hour_since_epoch_to_wind_events, hour_from_epoch, StormEvents.Event[]),
      get(hour_since_epoch_to_hail_events, hour_from_epoch-1, StormEvents.Event[]),
      get(hour_since_epoch_to_hail_events, hour_from_epoch, StormEvents.Event[])
    )

    tornado_segments = StormEvents.event_segments_around_time(tornadoes, hour_seconds_from_epoch, 30*MINUTE)
    event_segments   = StormEvents.event_segments_around_time(events,    hour_seconds_from_epoch, 30*MINUTE)

    count_neighborhoods!(tornado_hour_counts_grids[hour_in_day_i], grid, tornado_segments, NEIGHBORHOOD_RADIUS_MILES)
    count_neighborhoods!(event_hour_counts_grids[hour_in_day_i],   grid, event_segments,   NEIGHBORHOOD_RADIUS_MILES)
    hour_counts[hour_in_day_i] += 1
  end
  println("")

  hour_tornado_probs = zeros(Float64, 24)
  hour_severe_probs  = zeros(Float64, 24)

  map(1:24) do hour_in_day_i
    tornado_hour_counts_grid = tornado_hour_counts_grids[hour_in_day_i]
    event_hour_counts_grid   = event_hour_counts_grids[hour_in_day_i]
    tornado_counts_for_hour, conus_weight = weighted_sum_conus(tornado_hour_counts_grid)
    severe_counts_for_hour,  conus_weight = weighted_sum_conus(event_hour_counts_grid)

    hour_tornado_probs[hour_in_day_i] = tornado_counts_for_hour / conus_weight / hour_counts[hour_in_day_i]
    hour_severe_probs[hour_in_day_i]  = severe_counts_for_hour  / conus_weight / hour_counts[hour_in_day_i]
  end

  (hour_tornado_probs, hour_severe_probs)
end

println("Computing tornado probability for hours in day")

hour_tornado_probs, hour_severe_probs = count_events_by_hour(start_seconds_from_epoch:(end_seconds_from_epoch - 1), hour_since_epoch_to_tornadoes, hour_since_epoch_to_wind_events, hour_since_epoch_to_hail_events)

println("Hour in day\tTornado Prob\tSevere Prob")

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

for hour_in_day in 0:23
  hour_in_day_i = hour_in_day + 1

  println("$hour_in_day\t$(hour_tornado_probs[hour_in_day_i])\t$(hour_severe_probs[hour_in_day_i])")
end

# Counts by season

# Returns (tornado_day_probs_by_month, event_day_probs_by_month)
function count_events_by_month(range_in_seconds_from_epoch, convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)
  grid = HREF_CROPPED_15KM_GRID

  tornado_day_counts_grids  = map(_ -> zeros(Float32, size(grid.latlons)), 1:12)
  event_day_counts_grids    = map(_ -> zeros(Float32, size(grid.latlons)), 1:12)
  day_counts_by_month       = zeros(Float32, 12)

  for day_seconds_from_epoch in range_in_seconds_from_epoch.start:DAY:range_in_seconds_from_epoch.stop
    day_i = StormEvents.seconds_to_convective_days_since_epoch_utc(day_seconds_from_epoch)

    # Might be off by a day b/c convective day whatevers. No biggie, this is rough stats.
    month_i = Dates.month(Dates.unix2datetime(day_seconds_from_epoch))

    print(".")

    tornadoes = get(convective_day_to_tornadoes, day_i, StormEvents.Event[])
    events    = vcat(tornadoes, get(convective_day_to_wind_events, day_i, StormEvents.Event[]), get(convective_day_to_hail_events, day_i, StormEvents.Event[]))

    tornado_segments = StormEvents.event_segments_around_time(tornadoes, day_seconds_from_epoch + 12*HOUR, 12*HOUR)
    event_segments   = StormEvents.event_segments_around_time(events,    day_seconds_from_epoch + 12*HOUR, 12*HOUR)

    count_neighborhoods!(tornado_day_counts_grids[month_i], grid, tornado_segments, NEIGHBORHOOD_RADIUS_MILES)
    count_neighborhoods!(event_day_counts_grids[month_i],   grid, event_segments,   NEIGHBORHOOD_RADIUS_MILES)
    day_counts_by_month[month_i] += 1
  end
  println("")

  tornado_day_probs_by_month = zeros(Float64, 12)
  event_day_probs_by_month  = zeros(Float64, 12)

  map(1:12) do month_i
    tornado_day_counts_grid = tornado_day_counts_grids[month_i]
    event_day_counts_grid   = event_day_counts_grids[month_i]
    tornado_day_counts_for_month, conus_weight = weighted_sum_conus(tornado_day_counts_grid)
    severe_day_counts_for_month,  conus_weight = weighted_sum_conus(event_day_counts_grid)

    tornado_day_probs_by_month[month_i] = tornado_day_counts_for_month / conus_weight / day_counts_by_month[month_i]
    event_day_probs_by_month[month_i]   = severe_day_counts_for_month  / conus_weight / day_counts_by_month[month_i]
  end

  (tornado_day_probs_by_month, event_day_probs_by_month)
end

tornado_day_probs_by_month, event_day_probs_by_month = count_events_by_month(start_seconds_from_epoch:(end_seconds_from_epoch - 1), convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)

println("Month\tTornado Day Prob\tSevere Day Prob")

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


for month_i in 1:12
  println("$month_i\t$(tornado_day_probs_by_month[month_i])\t$(event_day_probs_by_month[month_i])")
end
