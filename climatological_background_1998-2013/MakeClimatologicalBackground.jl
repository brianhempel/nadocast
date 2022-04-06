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

conus_severe_wind_events = begin
  println("Loading wind events...")
  filter(read_and_filter_events_csv((@__DIR__) * "/../storm_data/wind_events_1998-2013.csv")) do wind_event
    wind_event.severity.knots >= 50.0
  end
end

conus_severe_hail_events = begin
  println("Loading hail events...")
  filter(read_and_filter_events_csv((@__DIR__) * "/../storm_data/hail_events_1998-2013.csv")) do hail_event
    hail_event.severity.inches >= 1.0
  end
end

conus_severe_events = vcat(conus_tornadoes, conus_severe_wind_events, conus_severe_hail_events)

is_sigtor(tornado)     = tornado.severity.ef_rating >= 2
is_sigwind(wind_event) = wind_event.severity.knots  >= 65.0
is_sighail(hail_event) = hail_event.severity.inches >= 2.0

conus_sig_tornadoes          = filter(is_sigtor,  conus_tornadoes)
conus_sig_severe_wind_events = filter(is_sigwind, conus_severe_wind_events)
conus_sig_severe_hail_events = filter(is_sighail, conus_severe_hail_events)
conus_sig_severe_events      = vcat(conus_sig_tornadoes, conus_sig_severe_wind_events, conus_sig_severe_hail_events)


function make_convective_day_to_events(events)
  convective_day_to_events = Dict{Int64,Vector{StormEvents.Event}}()

  for event in events
    for day_i in StormEvents.start_time_in_convective_days_since_epoch_utc(event):StormEvents.end_time_in_convective_days_since_epoch_utc(event)
      if !haskey(convective_day_to_events, day_i)
        convective_day_to_events[day_i] = StormEvents.Event[]
      end
      push!(convective_day_to_events[day_i], event)
    end
  end

  convective_day_to_events
end

function make_hour_since_epoch_to_events(events)
  hour_since_epoch_to_events = Dict{Int64,Vector{StormEvents.Event}}()

  for event in events
    for hour_i in fld(event.start_seconds_from_epoch_utc, HOUR):fld(event.end_seconds_from_epoch_utc, HOUR)
      if !haskey(hour_since_epoch_to_events, hour_i)
        hour_since_epoch_to_events[hour_i] = StormEvents.Event[]
      end
      push!(hour_since_epoch_to_events[hour_i], event)
    end
  end

  hour_since_epoch_to_events
end

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
function count_events_by_day(range_in_seconds_from_epoch, grid, convective_day_to_events_of_interest, convective_day_to_all_events)
  day_count = 0

  event_of_interest_day_counts_grid = zeros(Float32, size(grid.latlons))
  event_day_counts_grid             = zeros(Float32, size(grid.latlons))

  for day_seconds_from_epoch in range_in_seconds_from_epoch.start:DAY:range_in_seconds_from_epoch.stop
    day_i = StormEvents.seconds_to_convective_days_since_epoch_utc(day_seconds_from_epoch)

    print(".")

    events_of_interest = get(convective_day_to_events_of_interest, day_i, StormEvents.Event[])
    events             = get(convective_day_to_all_events,         day_i, StormEvents.Event[])

    event_of_interest_segments = StormEvents.event_segments_around_time(events_of_interest, day_seconds_from_epoch + 12*HOUR, 12*HOUR)
    event_segments             = StormEvents.event_segments_around_time(events,             day_seconds_from_epoch + 12*HOUR, 12*HOUR)

    count_neighborhoods!(event_of_interest_day_counts_grid, grid, event_of_interest_segments, NEIGHBORHOOD_RADIUS_MILES)
    count_neighborhoods!(event_day_counts_grid,             grid, event_segments,             NEIGHBORHOOD_RADIUS_MILES)
    day_count += 1
  end
  println("")

  (day_count, event_of_interest_day_counts_grid, event_day_counts_grid)
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
function try_parameters(transform_day_count_and_event_counts_to_prediction_grid, fold_day_event_and_severe_counts)
  loss   = 0.0
  weight = 0.0
  for fold_i_to_predict in 1:CROSS_VALIDATION_FOLD_COUNT
    tornado_day_counts_grid = zeros(Float64, size(fold_day_event_and_severe_counts[1][2]))
    day_count               = 0
    for fold_i in 1:CROSS_VALIDATION_FOLD_COUNT
      if fold_i != fold_i_to_predict
        fold_day_count, fold_tornado_day_counts_grid, _ = fold_day_event_and_severe_counts[fold_i]
        tornado_day_counts_grid .+= fold_tornado_day_counts_grid
        day_count                += fold_day_count
      end
    end
    prediction_grid = transform_day_count_and_event_counts_to_prediction_grid(day_count, tornado_day_counts_grid)
    fold_to_predict_day_count, fold_to_predict_tornado_day_counts_grid, _ = fold_day_event_and_severe_counts[fold_i_to_predict]
    fold_loss, fold_weight = compute_loss(prediction_grid, fold_to_predict_tornado_day_counts_grid, fold_to_predict_day_count)
    # println("Loss predicting fold $fold_i_to_predict:\t$(fold_loss/fold_weight)")
    loss   += fold_loss
    weight += fold_weight
  end

  loss / weight
end

# Returns (total, total spatial weight)
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

function compute_daily_mean_prob(day_count, event_day_counts_grid)
  # println("day counts $day_count")

  weighted_spatial_counts, weight_total = weighted_sum_conus(event_day_counts_grid)

  weighted_spatial_counts / weight_total / day_count
end


types = [
  ("tornado",     conus_tornadoes),              # Best params: Blur radius 50.0  Smoothing strength 96.0  Loss: 0.010463806248995431
  ("wind",        conus_severe_wind_events),     # Best params: Blur radius 10.0  Smoothing strength 48.0  Loss: 0.05522807765232684
  ("hail",        conus_severe_hail_events),     # Best params: Blur radius 25.0  Smoothing strength 48.0  Loss: 0.03370438382594854
  ("severe",      conus_severe_events),          # Best params: Blur radius 10.0  Smoothing strength 48.0  Loss: 0.07244811816382359
  ("sig_tornado", conus_sig_tornadoes),          # Best params: Blur radius 150.0 Smoothing strength 64.0  Loss: 0.0021216824523653294
  ("sig_wind",    conus_sig_severe_wind_events), # Best params: Blur radius 35.0  Smoothing strength 256.0 Loss: 0.008384086529900121
  ("sig_hail",    conus_sig_severe_hail_events), # Best params: Blur radius 70.0  Smoothing strength 48.0  Loss: 0.005511058500037982
  ("sig_severe",  conus_sig_severe_events),      # Best params: Blur radius 35.0  Smoothing strength 96.0  Loss: 0.013285981224650177
]


# println("Computing smoothing for points outside CONUS...")
# mean_is_outside_conus =
#   map(1:length(HREF_CROPPED_15KM_GRID.latlons)) do grid_i
#     print(".")
#     if !CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i]
#       radius = Inf
#       Grids.diamond_search(HREF_CROPPED_15KM_GRID, grid_i) do candidate_latlon
#         distance = GeoUtils.instantish_distance(candidate_latlon, HREF_CROPPED_15KM_GRID.latlons[grid_i])
#         if Conus.is_in_conus(candidate_latlon) && distance <= radius
#           if radius == Inf
#             radius = distance * 1.2
#           end
#           true
#         else
#           false
#         end
#       end
#     else
#       Int64[]
#     end
#   end

# function compute_conus_radius_grid_is(miles)
#   mean_is = Grids.radius_grid_is(HREF_CROPPED_15KM_GRID, miles)
#   for grid_i in 1:length(mean_is)
#     mean_is[grid_i] = filter(grid_i -> CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i], mean_is[grid_i])
#     if !CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i] && length(mean_is[grid_i]) < length(mean_is_outside_conus[grid_i])
#       mean_is[grid_i] = mean_is_outside_conus[grid_i]
#     end
#   end
#   mean_is
# end

# blur_radii = [0.0, 10.0, 25.0, 35.0, 50.0, 70.0, 100.0, 125.0, 150.0, 200.0, 300.0]

# blurrers = map(blur_radii) do radius
#   if radius == 0.0
#     blurrer = identity
#   else
#     println("Computing $(radius)mi mean is...")
#     mean_is = compute_conus_radius_grid_is(radius)

#     blurrer(event_day_counts_grid) = meanify(event_day_counts_grid, mean_is)
#   end

#   blurrer
# end

# convective_day_to_severe_events   = make_convective_day_to_events(conus_severe_events)

# println()
# for (event_name, events_of_interest) in types
#   println("=== $event_name spatial climatology ===")

#   convective_day_to_events_of_interest = make_convective_day_to_events(events_of_interest)

#   fold_day_event_and_severe_counts =
#     map(fold_ranges_in_seconds_from_epoch) do fold_range
#       count_events_by_day(fold_range, HREF_CROPPED_15KM_GRID, convective_day_to_events_of_interest, convective_day_to_severe_events)
#     end

#   # best_params = (0.010477507670158212, 50.0, 96.0) # loss, blur_radius, smoothing_strength
#   best_params = (Inf, nothing, nothing) # loss, blur_radius, smoothing_strength

#   # Best params: Blur radius 50.0	Smoothing strength 96.0	Loss: 0.010477507670158212

#   for blur_i in 1:length(blur_radii)
#     blur_radius = blur_radii[blur_i]
#     blurrer     = blurrers[blur_i]
#     # global best_params

#     for smoothing_strength in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
#       # println("Trying smoothing strength $smoothing_strength")
#       transform_day_count_and_event_counts_to_prediction_grid(day_count, event_day_counts_grid) = begin
#         background_prob = compute_daily_mean_prob(day_count, event_day_counts_grid)
#         # println("Background prob: $background_prob\texpected ~0.0014756858856650565")
#         prediction_grid = event_day_counts_grid .+ (background_prob * smoothing_strength)
#         prediction_grid ./= smoothing_strength + day_count
#         blurrer(prediction_grid)
#       end
#       parameters_loss = try_parameters(transform_day_count_and_event_counts_to_prediction_grid, fold_day_event_and_severe_counts)
#       # println("Blur radius $blur_radius\tSmoothing strength $smoothing_strength\tLoss: $parameters_loss")
#       if parameters_loss < best_params[1]
#         best_params = (parameters_loss, blur_radius, smoothing_strength)
#       end
#     end
#   end

#   best_loss, best_blur_radius, best_smoothing_strength = best_params

#   println("Best params: Blur radius $best_blur_radius\tSmoothing strength $best_smoothing_strength\tLoss: $best_loss")

#   event_day_counts_grid  = zeros(Float64, size(fold_day_event_and_severe_counts[1][2]))
#   severe_day_counts_grid = zeros(Float64, size(event_day_counts_grid))
#   day_count               = 0
#   for fold_i in 1:CROSS_VALIDATION_FOLD_COUNT
#     # global tornado_day_counts_grid
#     # global event_day_counts_grid
#     # global day_count
#     fold_day_count, fold_event_day_counts_grid, fold_severe_day_counts_grid = fold_day_event_and_severe_counts[fold_i]
#     event_day_counts_grid .+= fold_event_day_counts_grid
#     severe_day_counts_grid   .+= fold_severe_day_counts_grid
#     day_count                += fold_day_count
#   end

#   mean_is = compute_conus_radius_grid_is(best_blur_radius)

#   background_event_prob = compute_daily_mean_prob(day_count, event_day_counts_grid)
#   background_severe_prob   = compute_daily_mean_prob(day_count, severe_day_counts_grid)
#   println("Background event day prob: $background_event_prob\t(expected ~0.0014756858856650565)")
#   println("Background severe day prob: $background_severe_prob")
#   event_prediction_grid = event_day_counts_grid .+ (background_event_prob * best_smoothing_strength)
#   event_prediction_grid ./= best_smoothing_strength + day_count
#   event_prediction_grid = meanify(event_prediction_grid, mean_is)

#   severe_prediction_grid = severe_day_counts_grid .+ (background_severe_prob * best_smoothing_strength)
#   severe_prediction_grid ./= best_smoothing_strength + day_count
#   severe_prediction_grid = meanify(severe_prediction_grid, mean_is)

#   event_day_given_severe_day = event_prediction_grid ./ severe_prediction_grid

#   geomean_absolute_and_conditional_probabilty = sqrt.(event_prediction_grid .* event_day_given_severe_day)

#   println("Plotting $event_name Day Climatological Probability...")
#   PlotMap.plot_debug_map(
#     "$(event_name)_day_climatological_probability",
#     HREF_CROPPED_15KM_GRID,
#     event_prediction_grid;
#     title="$(event_name) Day Probability 1998-2013",
#     zlow=0.000,
#     zhigh=maximum(event_prediction_grid[CONUS_ON_HREF_CROPPED_15KM_GRID]),
#     steps=12
#   )
#   write("$(event_name)_day_climatological_probability.float16.bin", Float16.(event_prediction_grid))

#   if event_name != "severe"
#     println("Plotting p($(event_name)Day|SevereDay)...")
#     PlotMap.plot_debug_map(
#       "$(event_name)_day_given_severe_day_climatological_probability",
#       HREF_CROPPED_15KM_GRID,
#       event_day_given_severe_day;
#       title="p($(event_name)|SevereDay) 1998-2013",
#       zlow=0.0,
#       zhigh=maximum(event_day_given_severe_day[CONUS_ON_HREF_CROPPED_15KM_GRID]),
#       steps=12
#     )
#     write("$(event_name)_day_given_severe_day_climatological_probability.float16.bin", Float16.(event_day_given_severe_day))
#   end

#   println("Plotting geomean of $event_name prob and p($(event_name)Day|SevereDay)...")
#   PlotMap.plot_debug_map(
#     "$(event_name)_day_geomean_absolute_and_conditional_climatological_probability",
#     HREF_CROPPED_15KM_GRID,
#     geomean_absolute_and_conditional_probabilty;
#     title="$(event_name) Geomean abs&cond prob 1998-2013",
#     zlow=0.0,
#     zhigh=maximum(geomean_absolute_and_conditional_probabilty[CONUS_ON_HREF_CROPPED_15KM_GRID]),
#     steps=12
#   )
#   write("$(event_name)_day_geomean_absolute_and_conditional_climatological_probability.float16.bin", Float16.(geomean_absolute_and_conditional_probabilty))

# end




# # Counts by hour
# println("=== hour climatology ===")

# # Returns (hour_tornado_probs, hour_severe_probs)
# function count_events_by_hour(range_in_seconds_from_epoch, hour_since_epoch_to_events_of_interest)
#   grid = HREF_CROPPED_15KM_GRID

#   event_hour_counts_grids         = map(_ -> zeros(Float32, size(grid.latlons)), 0:23)
#   hour_counts                     = map(_ -> 0f0, 0:23)

#   for hour_seconds_from_epoch in range_in_seconds_from_epoch.start:HOUR:range_in_seconds_from_epoch.stop
#     hour_from_epoch = fld(hour_seconds_from_epoch, HOUR)
#     hour_in_day     = mod(hour_from_epoch, 24)
#     hour_in_day_i   = hour_in_day + 1

#     # if hour_in_day == 0
#     #   print(".")
#     # end

#     events_of_interest = vcat(
#       get(hour_since_epoch_to_events_of_interest, hour_from_epoch-1, StormEvents.Event[]),
#       get(hour_since_epoch_to_events_of_interest, hour_from_epoch,   StormEvents.Event[])
#     )

#     event_segments = StormEvents.event_segments_around_time(events_of_interest, hour_seconds_from_epoch, 30*MINUTE)

#     count_neighborhoods!(event_hour_counts_grids[hour_in_day_i], grid, event_segments, NEIGHBORHOOD_RADIUS_MILES)
#     hour_counts[hour_in_day_i] += 1
#   end
#   # println("")

#   hour_event_probs = zeros(Float64, 24)

#   map(1:24) do hour_in_day_i
#     event_hour_counts_grid              = event_hour_counts_grids[hour_in_day_i]
#     event_counts_for_hour, conus_weight = weighted_sum_conus(event_hour_counts_grid)

#     hour_event_probs[hour_in_day_i] = event_counts_for_hour / conus_weight / hour_counts[hour_in_day_i]
#   end

#   hour_event_probs
# end


# println()
# for (event_name, events_of_interest) in types

#   # println("Computing $event_name probability for hours in day")

#   hour_since_epoch_to_events = make_hour_since_epoch_to_events(events_of_interest)

#   hour_event_probs = count_events_by_hour(start_seconds_from_epoch:(end_seconds_from_epoch - 1), hour_since_epoch_to_events)

#   println("Hour in day\t$event_name prob")

#   for hour_in_day in 0:23
#     hour_in_day_i = hour_in_day + 1

#     println("$hour_in_day\t$(hour_event_probs[hour_in_day_i])")
#   end

#   println("hour_i_to_$(event_name)_prob = $hour_event_probs")
# end

# hour_i_to_tornado_prob     = [0.00011129670539161479, 9.38202928035434e-5, 7.554882472053065e-5, 4.506732856144229e-5, 3.7171616641439754e-5, 3.3920325564553303e-5, 2.6174652595109117e-5, 2.2975211147651646e-5, 1.9316105896445397e-5, 2.1067907043886953e-5, 2.6355116947657688e-5, 3.0500170320153166e-5, 3.934109645580011e-5, 5.4712401228771426e-5, 7.810326781796235e-5, 0.0001025200727213134, 0.00012786740530530984, 0.000142843733729483, 0.00015703643460658612, 0.0001534503061897477, 0.00013531649365335443, 0.00011850176872321167, 0.0001238013663200843, 0.00012037984473310072]
# hour_i_to_wind_prob        = [0.0008633908985919635, 0.0007366986617443046, 0.0005951624839093926, 0.0004562635095165616, 0.00036014503539837046, 0.00029217315831413465, 0.0002360174543649597, 0.00020449999726284446, 0.00017980798813168404, 0.0001586411458048938, 0.00016164817390954743, 0.00017664099113015713, 0.00022822245523106672, 0.0003444102869851599, 0.0005065886675066152, 0.0006744633615789415, 0.0008310590568744863, 0.000938921838745861, 0.0009850906866351416, 0.0010388969370322767, 0.001048979085091771, 0.001057213768015918, 0.001035649270971638, 0.0009592686130077239]
# hour_i_to_hail_prob        = [0.0005233713092750962, 0.00044817886057142866, 0.00033958843571957656, 0.00022839764241073068, 0.00016423001714810723, 0.00012613332199399894, 9.610642523244228e-5, 8.280646672289787e-5, 7.66645365198274e-5, 7.175355703656904e-5, 6.961093930269191e-5, 8.467978687305689e-5, 0.0001172065823407831, 0.00017335863624106483, 0.0002719099809688329, 0.00038313878938291785, 0.0004986045006191108, 0.0005778386932548916, 0.0005956984824827397, 0.0006097130739288307, 0.0005783867174166265, 0.0005925437612127855, 0.0006012629877916027, 0.0005762189888610501]
# hour_i_to_severe_prob      = [0.0013185286942441264, 0.0011221491513765183, 0.0008926469915730582, 0.0006575419735763914, 0.0005053820531486138, 0.0004097323508702168, 0.0003288946034436646, 0.00028559558511602283, 0.00025563277021592267, 0.00023229462329994906, 0.00023604158405287468, 0.00026788452784516685, 0.0003516411273044998, 0.0005228534478340439, 0.0007735592168252087, 0.0010422597919820245, 0.0012988410806906055, 0.001471330940148913, 0.001529418493911422, 0.0015844557041777712, 0.0015582217500002705, 0.0015652654527321306, 0.0015540827701441612, 0.0014650361255499629]
# hour_i_to_sig_tornado_prob = [2.0204267374007864e-5, 1.6483179744648044e-5, 1.9828965487783052e-5, 1.227257408871806e-5, 8.991101390618407e-6, 7.420427093425188e-6, 6.063931955567028e-6, 5.374721310510674e-6, 4.442737613941213e-6, 4.366186475981335e-6, 4.566673524769823e-6, 5.014560663968468e-6, 5.007149605154559e-6, 6.659298615190264e-6, 9.249450774722642e-6, 1.3236540217490568e-5, 1.686617015077223e-5, 2.1023662389142084e-5, 2.4391014593128085e-5, 2.7955555470838846e-5, 2.335316564545225e-5, 2.2093884582046064e-5, 2.2851638277852296e-5, 2.344853440421122e-5]
# hour_i_to_sig_wind_prob    = [9.029705995199781e-5, 7.636025579254354e-5, 6.262330789092903e-5, 4.98926293585469e-5, 3.668399332503053e-5, 3.523300858657056e-5, 2.9110753775460693e-5, 2.3587496987802423e-5, 2.1232370001213908e-5, 1.645180995561152e-5, 1.6104369929470466e-5, 1.9707565530330088e-5, 2.1742012044131422e-5, 2.7030035093373642e-5, 3.6623393717190304e-5, 5.29461637628618e-5, 6.572232581028526e-5, 7.389756698545703e-5, 7.580026576043009e-5, 8.844151898343568e-5, 8.603962748501033e-5, 8.442108173792976e-5, 8.204180406200946e-5, 7.781096244238919e-5]
# hour_i_to_sig_hail_prob    = [6.093163144469527e-5, 5.054194882559355e-5, 3.729363459075629e-5, 1.9050842884638393e-5, 1.4558258797997798e-5, 1.1855207637106955e-5, 1.014363432227151e-5, 5.806495674899298e-6, 5.561520012063525e-6, 6.272954409987843e-6, 6.378957777550435e-6, 7.770346429269897e-6, 8.301491181449269e-6, 1.4454809934808583e-5, 2.0809292519545706e-5, 3.764703602096664e-5, 5.407710212593903e-5, 6.870875054556627e-5, 7.557910191168755e-5, 6.716775540063365e-5, 6.20262751428271e-5, 6.148156922907246e-5, 6.539665468681525e-5, 6.694607605596366e-5]
# hour_i_to_sig_severe_prob  = [0.00016373441553649458, 0.0001362747742499125, 0.00011290382863872936, 7.79501984150932e-5, 5.7784273633235266e-5, 5.299013168953906e-5, 4.3572943574462124e-5, 3.322037471272543e-5, 3.0316792241538725e-5, 2.6185188706639833e-5, 2.5634125191824903e-5, 3.152117627844046e-5, 3.3847382951443264e-5, 4.6502450299912195e-5, 6.409427699969845e-5, 0.00010084421757960914, 0.000132262649672685, 0.0001557266055294015, 0.00016737697725801898, 0.00017709324628305533, 0.000164226133844577, 0.00015958048781038308, 0.00016177705939201268, 0.00016011201360300388]



# Counts by season

println("=== month climatology ===")


# Returns (tornado_day_probs_by_month, event_day_probs_by_month)
function count_events_by_month(range_in_seconds_from_epoch, convective_day_to_events)
  grid = HREF_CROPPED_15KM_GRID

  event_day_counts_grids    = map(_ -> zeros(Float32, size(grid.latlons)), 1:12)
  day_counts_by_month       = zeros(Float32, 12)

  for day_seconds_from_epoch in range_in_seconds_from_epoch.start:DAY:range_in_seconds_from_epoch.stop
    day_i = StormEvents.seconds_to_convective_days_since_epoch_utc(day_seconds_from_epoch)

    # Might be off by a day b/c convective day whatevers. No biggie, this is rough stats.
    month_i = Dates.month(Dates.unix2datetime(day_seconds_from_epoch))

    # print(".")

    events = get(convective_day_to_events, day_i, StormEvents.Event[])

    event_segments = StormEvents.event_segments_around_time(events, day_seconds_from_epoch + 12*HOUR, 12*HOUR)

    count_neighborhoods!(event_day_counts_grids[month_i], grid, event_segments, NEIGHBORHOOD_RADIUS_MILES)
    day_counts_by_month[month_i] += 1
  end
  # println("")

  event_day_probs_by_month  = zeros(Float64, 12)

  map(1:12) do month_i
    event_day_counts_grid   = event_day_counts_grids[month_i]
    severe_day_counts_for_month,  conus_weight = weighted_sum_conus(event_day_counts_grid)

    event_day_probs_by_month[month_i]   = severe_day_counts_for_month  / conus_weight / day_counts_by_month[month_i]
  end

  event_day_probs_by_month
end

println()
for (event_name, events_of_interest) in types

  # println("Computing $event_name probability by month")

  convective_day_to_events_of_interest = make_convective_day_to_events(events_of_interest)

  event_day_probs_by_month = count_events_by_month(start_seconds_from_epoch:(end_seconds_from_epoch - 1), convective_day_to_events_of_interest)

  println("Month\t$event_name Day Prob")

  for month_i in 1:12
    println("$month_i\t$(event_day_probs_by_month[month_i])")
  end

  println("month_i_to_$(event_name)_day_prob = $event_day_probs_by_month")
end

# Month	tornado Day Prob
# 1	0.0005912588636110609
# 2	0.0006566448193920728
# 3	0.0011690501952992978
# 4	0.002689031711471158
# 5	0.0035972006097341825
# 6	0.003151486712053723
# 7	0.0016031769612964244
# 8	0.0011273924818847902
# 9	0.001052397789801193
# 10	0.0008061086130634434
# 11	0.0007586494447008108
# 12	0.0004647142381289501
# month_i_to_tornado_day_prob = [0.0005912588636110609, 0.0006566448193920728, 0.0011690501952992978, 0.002689031711471158, 0.0035972006097341825, 0.003151486712053723, 0.0016031769612964244, 0.0011273924818847902, 0.001052397789801193, 0.0008061086130634434, 0.0007586494447008108, 0.0004647142381289501]
# Month	wind Day Prob
# 1	0.0021701267130796137
# 2	0.003332475149959834
# 3	0.004692374485391373
# 4	0.009747500693746249
# 5	0.018235766346102678
# 6	0.030558393430565007
# 7	0.028044016789867364
# 8	0.01953429401281888
# 9	0.0064876948191625134
# 10	0.003222887006037163
# 11	0.002636611573482487
# 12	0.0017016603113496978
# month_i_to_wind_day_prob = [0.0021701267130796137, 0.003332475149959834, 0.004692374485391373, 0.009747500693746249, 0.018235766346102678, 0.030558393430565007, 0.028044016789867364, 0.01953429401281888, 0.0064876948191625134, 0.003222887006037163, 0.002636611573482487, 0.0017016603113496978]
# Month	hail Day Prob
# 1	0.0006473405429670652
# 2	0.0014501671463444299
# 3	0.004448432329871499
# 4	0.00989113163733782
# 5	0.015579595628507062
# 6	0.01591402923807628
# 7	0.009732367012624045
# 8	0.00707567406691841
# 9	0.0029677710902595803
# 10	0.0014797863289096953
# 11	0.000615034855603229
# 12	0.0003636807326750079
# month_i_to_hail_day_prob = [0.0006473405429670652, 0.0014501671463444299, 0.004448432329871499, 0.00989113163733782, 0.015579595628507062, 0.01591402923807628, 0.009732367012624045, 0.00707567406691841, 0.0029677710902595803, 0.0014797863289096953, 0.000615034855603229, 0.0003636807326750079]
# Month	severe Day Prob
# 1	0.0027865892080668757
# 2	0.0044739155446376095
# 3	0.008216050372064233
# 4	0.01707046600204392
# 5	0.02903485547966859
# 6	0.04076942658887546
# 7	0.03411450888782611
# 8	0.024197575846163104
# 9	0.00904617218782285
# 10	0.004570236749062177
# 11	0.0032848131398126247
# 12	0.002176505090521792
# month_i_to_severe_day_prob = [0.0027865892080668757, 0.0044739155446376095, 0.008216050372064233, 0.01707046600204392, 0.02903485547966859, 0.04076942658887546, 0.03411450888782611, 0.024197575846163104, 0.00904617218782285, 0.004570236749062177, 0.0032848131398126247, 0.002176505090521792]
# Month	sig_tornado Day Prob
# 1	0.00014353889071694597
# 2	0.00017995150442440733
# 3	0.00028227656059078135
# 4	0.0006169281846573612
# 5	0.0006334680407315207
# 6	0.00027851419032152346
# 7	0.00011388875288482689
# 8	8.44194127395308e-5
# 9	0.00012614148625710476
# 10	0.00012526349325472208
# 11	0.0002346849759241835
# 12	0.00011381850944044274
# month_i_to_sig_tornado_day_prob = [0.00014353889071694597, 0.00017995150442440733, 0.00028227656059078135, 0.0006169281846573612, 0.0006334680407315207, 0.00027851419032152346, 0.00011388875288482689, 8.44194127395308e-5, 0.00012614148625710476, 0.00012526349325472208, 0.0002346849759241835, 0.00011381850944044274]
# Month	sig_wind Day Prob
# 1	0.00025011009872090174
# 2	0.0003783928615320516
# 3	0.0006006276086852014
# 4	0.0013437343026582091
# 5	0.002207070698867628
# 6	0.0031492876293528014
# 7	0.002680330531885248
# 8	0.0017762392428364562
# 9	0.0005580218796202924
# 10	0.0002985255687100561
# 11	0.0002476220613107017
# 12	0.0001525428874219422
# month_i_to_sig_wind_day_prob = [0.00025011009872090174, 0.0003783928615320516, 0.0006006276086852014, 0.0013437343026582091, 0.002207070698867628, 0.0031492876293528014, 0.002680330531885248, 0.0017762392428364562, 0.0005580218796202924, 0.0002985255687100561, 0.0002476220613107017, 0.0001525428874219422]
# Month	sig_hail Day Prob
# 1	7.916751542765838e-5
# 2	0.00012174983348249795
# 3	0.0005401873122813536
# 4	0.0013791188222479019
# 5	0.0021996989350041045
# 6	0.001967426775966405
# 7	0.001095181515045675
# 8	0.0006832811409030232
# 9	0.000316083101926938
# 10	0.00012612148985031082
# 11	4.4488606753264214e-5
# 12	2.286536189910439e-5
# month_i_to_sig_hail_day_prob = [7.916751542765838e-5, 0.00012174983348249795, 0.0005401873122813536, 0.0013791188222479019, 0.0021996989350041045, 0.001967426775966405, 0.001095181515045675, 0.0006832811409030232, 0.000316083101926938, 0.00012612148985031082, 4.4488606753264214e-5, 2.286536189910439e-5]
# Month	sig_severe Day Prob
# 1	0.0004149323498785256
# 2	0.0006245327622484127
# 3	0.0012851295093226094
# 4	0.0030218901807132913
# 5	0.004581169622560005
# 6	0.005052426692869783
# 7	0.0037134880570005324
# 8	0.002457232803345733
# 9	0.0009503723115058557
# 10	0.0005207052016221992
# 11	0.00048408068763509054
# 12	0.00027466741282023875
# month_i_to_sig_severe_day_prob = [0.0004149323498785256, 0.0006245327622484127, 0.0012851295093226094, 0.0030218901807132913, 0.004581169622560005, 0.005052426692869783, 0.0037134880570005324, 0.002457232803345733, 0.0009503723115058557, 0.00052070