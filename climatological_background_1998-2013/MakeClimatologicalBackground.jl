push!(LOAD_PATH, (@__DIR__) * "/../lib")

import Conus
import Forecasts
import GeoUtils
import Grib2
import Grids
import PlotMap
import StormEvents

HOUR = StormEvents.HOUR
DAY  = StormEvents.DAY

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

  # radius_is[flat_i] =
  #   diamond_search(grid, i, j) do candidate_latlon
  #     GeoUtils.instantish_distance(candidate_latlon, latlon) <= miles * GeoUtils.METERS_PER_MILE
  #   end

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

  # is_near_event(latlon) = begin
  #   is_near = false
  #
  #   for (latlon1, latlon2) in event_segments
  #     meters_away = GeoUtils.instant_meters_to_line(latlon, latlon1, latlon2)
  #     if meters_away <= miles * GeoUtils.METERS_PER_MILE
  #       is_near = true
  #     end
  #   end
  #
  #   is_near
  # end
  #
  # Threads.@threads for grid_i in 1:length(grid.latlons)
  #   latlon = grid.latlons[grid_i]
  #
  #   if is_near_event(latlon)
  #     counts_grid[grid_i] += 1f0
  #   end
  # end

  ()
end


# event_hour_counts_grid   = zeros(Float32, size(HREF_CROPPED_15KM_GRID.latlons))
# tornado_hour_counts_grid = zeros(Float32, size(HREF_CROPPED_15KM_GRID.latlons))

# Returns (day_count, tornado_day_counts_grid, event_day_counts_grid)
function count_events(range_in_seconds_from_epoch, grid, convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)
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
    count_events(fold_range, HREF_CROPPED_15KM_GRID, convective_day_to_tornadoes, convective_day_to_wind_events, convective_day_to_hail_events)
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

function compute_daily_mean_tor_prob(day_count, tornado_day_counts_grid)
  # println("day counts $day_count")

  tornado_weighted_spacial_counts = 0.0
  weight_total                    = 0.0

  for grid_i in 1:length(HREF_CROPPED_15KM_GRID.latlons)
    if CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i]
      tornado_weighted_spacial_counts += tornado_day_counts_grid[grid_i] * HREF_CROPPED_15KM_GRID.point_weights[grid_i]
      weight_total                    +=                                   HREF_CROPPED_15KM_GRID.point_weights[grid_i]
    end
  end

  tornado_weighted_spacial_counts / weight_total / day_count
end

# daily_mean_tor_prob = compute_daily_mean_tor_prob(day_count, tornado_day_counts_grid)
#
#
# println("Overall conus daily mean tor prob (within 25mi): $daily_mean_tor_prob")

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

println("Plotting geomean of tor prob and p(TornadoDay|SevereDay)...")
PlotMap.plot_debug_map(
  "geomean_absolute_and_conditional_climatological_probability",
  HREF_CROPPED_15KM_GRID,
  geomean_absolute_and_conditional_probabilty;
  title="Geomean abs & cond prob 1998-2013",
  zlow=0.0,
  zhigh=0.2,
  steps=10
)



# tornado_day_counts_grid_blurred = meanify(tornado_day_counts_grid, hundred_mi_mean_is)
# event_day_counts_grid_blurred   = meanify(event_day_counts_grid, hundred_mi_mean_is)
# println("Plotting Tornado Day Counts...")
# PlotMap.plot_debug_map("tornado_day_counts",                HREF_CROPPED_15KM_GRID, tornado_day_counts_grid_blurred;                                    title="Tornado Day Counts")
# println("Plotting Severe Event Day Counts...")
# PlotMap.plot_debug_map("event_day_counts",                  HREF_CROPPED_15KM_GRID, event_day_counts_grid_blurred;                                      title="Severe Event Day Counts")
# println("Plotting p(TornadoDay|SevereEventDay)...")
# PlotMap.plot_debug_map("prob_tornado_day_given_severe_day", HREF_CROPPED_15KM_GRID, tornado_day_counts_grid_blurred ./ (event_day_counts_grid_blurred .+ 1f0); title="p(TornadoDay|SevereEventDay)")
