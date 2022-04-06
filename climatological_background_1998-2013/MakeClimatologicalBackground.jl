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

conus_sig_tornadoes          = filter(StormEvents.is_sig_tornado,  conus_tornadoes)
conus_sig_severe_wind_events = filter(StormEvents.is_sig_wind, conus_severe_wind_events)
conus_sig_severe_hail_events = filter(StormEvents.is_sig_hail, conus_severe_hail_events)
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
  println()

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
  ("tornado",     conus_tornadoes),              # Best params: Blur radius 50.0  Smoothing strength 96.0  Loss: 0.01046915792229855
  ("wind",        conus_severe_wind_events),     # Best params: Blur radius 10.0  Smoothing strength 48.0  Loss: 0.05527591966960997
  ("hail",        conus_severe_hail_events),     # Best params: Blur radius 25.0  Smoothing strength 48.0  Loss: 0.03372312626826926
  ("severe",      conus_severe_events),          # Best params: Blur radius 10.0  Smoothing strength 48.0  Loss: 0.07250396517030827
  ("sig_tornado", conus_sig_tornadoes),          # Best params: Blur radius 150.0 Smoothing strength 64.0  Loss: 0.0021216824523653294
  ("sig_wind",    conus_sig_severe_wind_events), # Best params: Blur radius 35.0  Smoothing strength 256.0 Loss: 0.008392463171617015
  ("sig_hail",    conus_sig_severe_hail_events), # Best params: Blur radius 70.0  Smoothing strength 48.0  Loss: 0.005512757617995458
  ("sig_severe",  conus_sig_severe_events),      # Best params: Blur radius 35.0  Smoothing strength 96.0  Loss: 0.013294755694834527
]


println("Computing smoothing for points outside CONUS...")
mean_is_outside_conus =
  map(1:length(HREF_CROPPED_15KM_GRID.latlons)) do grid_i
    print(".")
    if !CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i]
      radius = Inf
      Grids.diamond_search(HREF_CROPPED_15KM_GRID, grid_i) do candidate_latlon
        distance = GeoUtils.instantish_distance(candidate_latlon, HREF_CROPPED_15KM_GRID.latlons[grid_i])
        if Conus.is_in_conus(candidate_latlon) && distance <= radius
          if radius == Inf
            radius = distance * 1.2
          end
          true
        else
          false
        end
      end
    else
      Int64[]
    end
  end

function compute_conus_radius_grid_is(miles)
  mean_is = Grids.radius_grid_is(HREF_CROPPED_15KM_GRID, miles)
  for grid_i in 1:length(mean_is)
    mean_is[grid_i] = filter(grid_i -> CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i], mean_is[grid_i])
    if !CONUS_ON_HREF_CROPPED_15KM_GRID[grid_i] && length(mean_is[grid_i]) < length(mean_is_outside_conus[grid_i])
      mean_is[grid_i] = mean_is_outside_conus[grid_i]
    end
  end
  mean_is
end

blur_radii = [0.0, 10.0, 25.0, 35.0, 50.0, 70.0, 100.0, 125.0, 150.0, 200.0, 300.0]

blurrers = map(blur_radii) do radius
  if radius == 0.0
    blurrer = identity
  else
    println("Computing $(radius)mi mean is...")
    mean_is = compute_conus_radius_grid_is(radius)

    blurrer(event_day_counts_grid) = meanify(event_day_counts_grid, mean_is)
  end

  blurrer
end

convective_day_to_severe_events   = make_convective_day_to_events(conus_severe_events)

println()
for (event_name, events_of_interest) in types
  println("=== $event_name spatial climatology ===")

  convective_day_to_events_of_interest = make_convective_day_to_events(events_of_interest)

  fold_day_event_and_severe_counts =
    map(fold_ranges_in_seconds_from_epoch) do fold_range
      count_events_by_day(fold_range, HREF_CROPPED_15KM_GRID, convective_day_to_events_of_interest, convective_day_to_severe_events)
    end

  # best_params = (0.010477507670158212, 50.0, 96.0) # loss, blur_radius, smoothing_strength
  best_params = (Inf, nothing, nothing) # loss, blur_radius, smoothing_strength

  # Best params: Blur radius 50.0	Smoothing strength 96.0	Loss: 0.010477507670158212

  for blur_i in 1:length(blur_radii)
    blur_radius = blur_radii[blur_i]
    blurrer     = blurrers[blur_i]
    # global best_params

    for smoothing_strength in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
      # println("Trying smoothing strength $smoothing_strength")
      transform_day_count_and_event_counts_to_prediction_grid(day_count, event_day_counts_grid) = begin
        background_prob = compute_daily_mean_prob(day_count, event_day_counts_grid)
        # println("Background prob: $background_prob\texpected ~0.0014756858856650565")
        prediction_grid = event_day_counts_grid .+ (background_prob * smoothing_strength)
        prediction_grid ./= smoothing_strength + day_count
        blurrer(prediction_grid)
      end
      parameters_loss = try_parameters(transform_day_count_and_event_counts_to_prediction_grid, fold_day_event_and_severe_counts)
      # println("Blur radius $blur_radius\tSmoothing strength $smoothing_strength\tLoss: $parameters_loss")
      if parameters_loss < best_params[1]
        best_params = (parameters_loss, blur_radius, smoothing_strength)
      end
    end
  end

  best_loss, best_blur_radius, best_smoothing_strength = best_params

  println("Best params: Blur radius $best_blur_radius\tSmoothing strength $best_smoothing_strength\tLoss: $best_loss")

  event_day_counts_grid  = zeros(Float64, size(fold_day_event_and_severe_counts[1][2]))
  severe_day_counts_grid = zeros(Float64, size(event_day_counts_grid))
  day_count               = 0
  for fold_i in 1:CROSS_VALIDATION_FOLD_COUNT
    # global tornado_day_counts_grid
    # global event_day_counts_grid
    # global day_count
    fold_day_count, fold_event_day_counts_grid, fold_severe_day_counts_grid = fold_day_event_and_severe_counts[fold_i]
    event_day_counts_grid .+= fold_event_day_counts_grid
    severe_day_counts_grid   .+= fold_severe_day_counts_grid
    day_count                += fold_day_count
  end

  mean_is = compute_conus_radius_grid_is(best_blur_radius)

  background_event_prob = compute_daily_mean_prob(day_count, event_day_counts_grid)
  background_severe_prob   = compute_daily_mean_prob(day_count, severe_day_counts_grid)
  println("Background event day prob: $background_event_prob\t(expected ~0.0014756858856650565)")
  println("Background severe day prob: $background_severe_prob")
  event_prediction_grid = event_day_counts_grid .+ (background_event_prob * best_smoothing_strength)
  event_prediction_grid ./= best_smoothing_strength + day_count
  event_prediction_grid = meanify(event_prediction_grid, mean_is)

  severe_prediction_grid = severe_day_counts_grid .+ (background_severe_prob * best_smoothing_strength)
  severe_prediction_grid ./= best_smoothing_strength + day_count
  severe_prediction_grid = meanify(severe_prediction_grid, mean_is)

  event_day_given_severe_day = event_prediction_grid ./ severe_prediction_grid

  geomean_absolute_and_conditional_probabilty = sqrt.(event_prediction_grid .* event_day_given_severe_day)

  println("Plotting $event_name Day Climatological Probability...")
  PlotMap.plot_debug_map(
    "$(event_name)_day_climatological_probability",
    HREF_CROPPED_15KM_GRID,
    event_prediction_grid;
    title="$(event_name) Day Probability 1998-2013",
    zlow=0.000,
    zhigh=maximum(event_prediction_grid[CONUS_ON_HREF_CROPPED_15KM_GRID]),
    steps=12
  )
  write("$(event_name)_day_climatological_probability.float16.bin", Float16.(event_prediction_grid))

  if event_name != "severe"
    println("Plotting p($(event_name)Day|SevereDay)...")
    PlotMap.plot_debug_map(
      "$(event_name)_day_given_severe_day_climatological_probability",
      HREF_CROPPED_15KM_GRID,
      event_day_given_severe_day;
      title="p($(event_name)|SevereDay) 1998-2013",
      zlow=0.0,
      zhigh=maximum(event_day_given_severe_day[CONUS_ON_HREF_CROPPED_15KM_GRID]),
      steps=12
    )
    write("$(event_name)_day_given_severe_day_climatological_probability.float16.bin", Float16.(event_day_given_severe_day))
  end

  println("Plotting geomean of $event_name prob and p($(event_name)Day|SevereDay)...")
  PlotMap.plot_debug_map(
    "$(event_name)_day_geomean_absolute_and_conditional_climatological_probability",
    HREF_CROPPED_15KM_GRID,
    geomean_absolute_and_conditional_probabilty;
    title="$(event_name) Geomean abs&cond prob 1998-2013",
    zlow=0.0,
    zhigh=maximum(geomean_absolute_and_conditional_probabilty[CONUS_ON_HREF_CROPPED_15KM_GRID]),
    steps=12
  )
  write("$(event_name)_day_geomean_absolute_and_conditional_climatological_probability.float16.bin", Float16.(geomean_absolute_and_conditional_probabilty))

end




# Counts by hour
println("=== hour climatology ===")

# Returns (hour_tornado_probs, hour_severe_probs)
function count_events_by_hour(range_in_seconds_from_epoch, hour_since_epoch_to_events_of_interest)
  grid = HREF_CROPPED_15KM_GRID

  event_hour_counts_grids         = map(_ -> zeros(Float32, size(grid.latlons)), 0:23)
  hour_counts                     = map(_ -> 0f0, 0:23)

  for hour_seconds_from_epoch in range_in_seconds_from_epoch.start:HOUR:range_in_seconds_from_epoch.stop
    hour_from_epoch = fld(hour_seconds_from_epoch, HOUR)
    hour_in_day     = mod(hour_from_epoch, 24)
    hour_in_day_i   = hour_in_day + 1

    # if hour_in_day == 0
    #   print(".")
    # end

    events_of_interest = vcat(
      get(hour_since_epoch_to_events_of_interest, hour_from_epoch-1, StormEvents.Event[]),
      get(hour_since_epoch_to_events_of_interest, hour_from_epoch,   StormEvents.Event[])
    )

    event_segments = StormEvents.event_segments_around_time(events_of_interest, hour_seconds_from_epoch, 30*MINUTE)

    count_neighborhoods!(event_hour_counts_grids[hour_in_day_i], grid, event_segments, NEIGHBORHOOD_RADIUS_MILES)
    hour_counts[hour_in_day_i] += 1
  end
  # println()

  hour_event_probs = zeros(Float64, 24)

  map(1:24) do hour_in_day_i
    event_hour_counts_grid              = event_hour_counts_grids[hour_in_day_i]
    event_counts_for_hour, conus_weight = weighted_sum_conus(event_hour_counts_grid)

    hour_event_probs[hour_in_day_i] = event_counts_for_hour / conus_weight / hour_counts[hour_in_day_i]
  end

  hour_event_probs
end


println()
for (event_name, events_of_interest) in types

  # println("Computing $event_name probability for hours in day")

  hour_since_epoch_to_events = make_hour_since_epoch_to_events(events_of_interest)

  hour_event_probs = count_events_by_hour(start_seconds_from_epoch:(end_seconds_from_epoch - 1), hour_since_epoch_to_events)

  println("Hour in day\t$event_name prob")

  for hour_in_day in 0:23
    hour_in_day_i = hour_in_day + 1

    println("$hour_in_day\t$(hour_event_probs[hour_in_day_i])")
  end

  println("hour_i_to_$(event_name)_prob = $hour_event_probs")
end

# Hour in day	tornado prob
# 0	0.00011208100863164152
# 1	9.398953039251033e-5
# 2	7.615198696835708e-5
# 3	4.523942537179877e-5
# 4	3.7220979121385484e-5
# 5	3.4327180816387543e-5
# 6	2.6561651320771587e-5
# 7	2.3144851631010755e-5
# 8	1.9605944731661573e-5
# 9	2.1345620519246734e-5
# 10	2.666518858724128e-5
# 11	3.1266679137308404e-5
# 12	4.0029641178710914e-5
# 13	5.6085737277403675e-5
# 14	7.94691596247683e-5
# 15	0.00010496333719016243
# 16	0.00013024784693458098
# 17	0.00014591869216771128
# 18	0.00015936778571329406
# 19	0.0001562006014341579
# 20	0.00013690598873123164
# 21	0.00011990963352668346
# 22	0.00012519750236180316
# 23	0.00012124047353870332
# hour_i_to_tornado_prob = [0.00011208100863164152, 9.398953039251033e-5, 7.615198696835708e-5, 4.523942537179877e-5, 3.7220979121385484e-5, 3.4327180816387543e-5, 2.6561651320771587e-5, 2.3144851631010755e-5, 1.9605944731661573e-5, 2.1345620519246734e-5, 2.666518858724128e-5, 3.1266679137308404e-5, 4.0029641178710914e-5, 5.6085737277403675e-5, 7.94691596247683e-5, 0.00010496333719016243, 0.00013024784693458098, 0.00014591869216771128, 0.00015936778571329406, 0.0001562006014341579, 0.00013690598873123164, 0.00011990963352668346, 0.00012519750236180316, 0.00012124047353870332]
# Hour in day	wind prob
# 0	0.0009166247576253776
# 1	0.0007836079136567992
# 2	0.00063593005772162
# 3	0.0004867899307008593
# 4	0.00038798876687535686
# 5	0.00031525587105295873
# 6	0.0002552253627277109
# 7	0.0002185130048344488
# 8	0.00019247556624977299
# 9	0.0001706729904910728
# 10	0.00017340165897479975
# 11	0.00018910816147448578
# 12	0.00024069964302520056
# 13	0.00036666933767190543
# 14	0.000538784688856574
# 15	0.000719816757272354
# 16	0.0008931601972293983
# 17	0.001003979990686207
# 18	0.0010584134691574705
# 19	0.0011118986606950332
# 20	0.0011191869930757207
# 21	0.001128213892608944
# 22	0.0010981407083641916
# 23	0.0010203605170103437
# hour_i_to_wind_prob = [0.0009166247576253776, 0.0007836079136567992, 0.00063593005772162, 0.0004867899307008593, 0.00038798876687535686, 0.00031525587105295873, 0.0002552253627277109, 0.0002185130048344488, 0.00019247556624977299, 0.0001706729904910728, 0.00017340165897479975, 0.00018910816147448578, 0.00024069964302520056, 0.00036666933767190543, 0.000538784688856574, 0.000719816757272354, 0.0008931601972293983, 0.001003979990686207, 0.0010584134691574705, 0.0011118986606950332, 0.0011191869930757207, 0.001128213892608944, 0.0010981407083641916, 0.0010203605170103437]
# Hour in day	hail prob
# 0	0.0005476373941258122
# 1	0.00046801091486749604
# 2	0.0003546337747837403
# 3	0.00024212184157443322
# 4	0.00017443737463029152
# 5	0.00013404430012160947
# 6	0.00010271775777014372
# 7	8.823772496719175e-5
# 8	8.02802829466184e-5
# 9	7.581651103161307e-5
# 10	7.367960486676418e-5
# 11	8.989639534707708e-5
# 12	0.0001227185701075357
# 13	0.00017930916278610466
# 14	0.0002824123138822773
# 15	0.00039885876349847086
# 16	0.000519831694593571
# 17	0.0006052023637998322
# 18	0.0006223383693610371
# 19	0.0006381329390176494
# 20	0.000606887961325876
# 21	0.0006194544999011227
# 22	0.0006270133545383444
# 23	0.0006001956654302055
# hour_i_to_hail_prob = [0.0005476373941258122, 0.00046801091486749604, 0.0003546337747837403, 0.00024212184157443322, 0.00017443737463029152, 0.00013404430012160947, 0.00010271775777014372, 8.823772496719175e-5, 8.02802829466184e-5, 7.581651103161307e-5, 7.367960486676418e-5, 8.989639534707708e-5, 0.0001227185701075357, 0.00017930916278610466, 0.0002824123138822773, 0.00039885876349847086, 0.000519831694593571, 0.0006052023637998322, 0.0006223383693610371, 0.0006381329390176494, 0.000606887961325876, 0.0006194544999011227, 0.0006270133545383444, 0.0006001956654302055]
# Hour in day	severe prob
# 0	0.001386736264994684
# 1	0.001181139688679887
# 2	0.000943544458305578
# 3	0.0006980642458361038
# 4	0.0005392923421321292
# 5	0.000438512868137227
# 6	0.00035341094931301244
# 7	0.0003034618886429722
# 8	0.00027125897075474967
# 9	0.00024739765969603527
# 10	0.000250806077033577
# 11	0.0002850269006405788
# 12	0.00036891985131610056
# 13	0.0005498391027982735
# 14	0.0008123599878051271
# 15	0.0010988212247395632
# 16	0.0013756905103398476
# 17	0.0015541354838174903
# 18	0.0016178430876226202
# 19	0.00167347326477131
# 20	0.0016449479168609681
# 21	0.0016522666059494153
# 22	0.00163435775069809
# 23	0.0015394856806544921
# hour_i_to_severe_prob = [0.001386736264994684, 0.001181139688679887, 0.000943544458305578, 0.0006980642458361038, 0.0005392923421321292, 0.000438512868137227, 0.00035341094931301244, 0.0003034618886429722, 0.00027125897075474967, 0.00024739765969603527, 0.000250806077033577, 0.0002850269006405788, 0.00036891985131610056, 0.0005498391027982735, 0.0008123599878051271, 0.0010988212247395632, 0.0013756905103398476, 0.0015541354838174903, 0.0016178430876226202, 0.00167347326477131, 0.0016449479168609681, 0.0016522666059494153, 0.00163435775069809, 0.0015394856806544921]
# Hour in day	sig_tornado prob
# 0	2.0315842186049887e-5
# 1	1.6483179744648044e-5
# 2	1.9828965487783052e-5
# 3	1.227257408871806e-5
# 4	8.991101390618407e-6
# 5	7.420427093425188e-6
# 6	6.063931955567028e-6
# 7	5.374721310510674e-6
# 8	4.442737613941213e-6
# 9	4.366186475981335e-6
# 10	4.566673524769823e-6
# 11	5.014560663968468e-6
# 12	5.007149605154559e-6
# 13	6.659298615190264e-6
# 14	9.249450774722642e-6
# 15	1.3236540217490568e-5
# 16	1.7091773481262868e-5
# 17	2.1023662389142084e-5
# 18	2.4454198800414143e-5
# 19	2.8072850609920412e-5
# 20	2.335316564545225e-5
# 21	2.2093884582046064e-5
# 22	2.2851638277852296e-5
# 23	2.344853440421122e-5
# hour_i_to_sig_tornado_prob = [2.0315842186049887e-5, 1.6483179744648044e-5, 1.9828965487783052e-5, 1.227257408871806e-5, 8.991101390618407e-6, 7.420427093425188e-6, 6.063931955567028e-6, 5.374721310510674e-6, 4.442737613941213e-6, 4.366186475981335e-6, 4.566673524769823e-6, 5.014560663968468e-6, 5.007149605154559e-6, 6.659298615190264e-6, 9.249450774722642e-6, 1.3236540217490568e-5, 1.7091773481262868e-5, 2.1023662389142084e-5, 2.4454198800414143e-5, 2.8072850609920412e-5, 2.335316564545225e-5, 2.2093884582046064e-5, 2.2851638277852296e-5, 2.344853440421122e-5]
# Hour in day	sig_wind prob
# 0	9.461214789245994e-5
# 1	8.174936916893805e-5
# 2	6.593600066125565e-5
# 3	5.2685464070342675e-5
# 4	3.965508618240307e-5
# 5	3.700789604889291e-5
# 6	3.086584146294729e-5
# 7	2.489569495837653e-5
# 8	2.2240387237854667e-5
# 9	1.778022310643805e-5
# 10	1.6363270077873185e-5
# 11	2.0699265413263e-5
# 12	2.2797588531625434e-5
# 13	2.8180161062544515e-5
# 14	3.851241205648254e-5
# 15	5.557727563364827e-5
# 16	7.042736637359784e-5
# 17	7.917822557184195e-5
# 18	7.962576750978568e-5
# 19	9.32877306741423e-5
# 20	9.177451711288632e-5
# 21	8.984958528711286e-5
# 22	8.712728371794457e-5
# 23	8.352331514218602e-5
# hour_i_to_sig_wind_prob = [9.461214789245994e-5, 8.174936916893805e-5, 6.593600066125565e-5, 5.2685464070342675e-5, 3.965508618240307e-5, 3.700789604889291e-5, 3.086584146294729e-5, 2.489569495837653e-5, 2.2240387237854667e-5, 1.778022310643805e-5, 1.6363270077873185e-5, 2.0699265413263e-5, 2.2797588531625434e-5, 2.8180161062544515e-5, 3.851241205648254e-5, 5.557727563364827e-5, 7.042736637359784e-5, 7.917822557184195e-5, 7.962576750978568e-5, 9.32877306741423e-5, 9.177451711288632e-5, 8.984958528711286e-5, 8.712728371794457e-5, 8.352331514218602e-5]
# Hour in day	sig_hail prob
# 0	6.316252724973134e-5
# 1	5.430602762677378e-5
# 2	3.918881158288507e-5
# 3	2.067907440814021e-5
# 4	1.535670342440398e-5
# 5	1.2675341028526707e-5
# 6	1.0598601956153157e-5
# 7	6.5657389899864634e-6
# 8	5.8856520483232565e-6
# 9	6.3905337510749745e-6
# 10	6.825985426534838e-6
# 11	7.836853631385986e-6
# 12	8.65819923412163e-6
# 13	1.4839353935123578e-5
# 14	2.1750622643932243e-5
# 15	3.861477949995283e-5
# 16	5.569325663605293e-5
# 17	7.250816679088989e-5
# 18	7.873938691475598e-5
# 19	7.042898590372579e-5
# 20	6.53138485423595e-5
# 21	6.401042935517331e-5
# 22	6.818112750943166e-5
# 23	6.997007108879904e-5
# hour_i_to_sig_hail_prob = [6.316252724973134e-5, 5.430602762677378e-5, 3.918881158288507e-5, 2.067907440814021e-5, 1.535670342440398e-5, 1.2675341028526707e-5, 1.0598601956153157e-5, 6.5657389899864634e-6, 5.8856520483232565e-6, 6.3905337510749745e-6, 6.825985426534838e-6, 7.836853631385986e-6, 8.65819923412163e-6, 1.4839353935123578e-5, 2.1750622643932243e-5, 3.861477949995283e-5, 5.569325663605293e-5, 7.250816679088989e-5, 7.873938691475598e-5, 7.042898590372579e-5, 6.53138485423595e-5, 6.401042935517331e-5, 6.818112750943166e-5, 6.997007108879904e-5]
# Hour in day	sig_severe prob
# 0	0.00017031378685785893
# 1	0.00014511432537476837
# 2	0.00011758204914335199
# 3	8.216765740341353e-5
# 4	6.151016402949586e-5
# 5	5.558515254328111e-5
# 6	4.5762592333366357e-5
# 7	3.52878159983866e-5
# 8	3.158533747897978e-5
# 9	2.7601589453085662e-5
# 10	2.634005298921204e-5
# 11	3.257938336348941e-5
# 12	3.525966749160955e-5
# 13	4.791608269461841e-5
# 14	6.692462546337728e-5
# 15	0.00010431331531113154
# 16	0.00013860471518229055
# 17	0.00016440283338529522
# 18	0.00017402140560148933
# 19	0.00018485967928068918
# 20	0.00017305971291019
# 21	0.00016706199945322573
# 22	0.00016936057562499978
# 23	0.00016841162424405075
# hour_i_to_sig_severe_prob = [0.00017031378685785893, 0.00014511432537476837, 0.00011758204914335199, 8.216765740341353e-5, 6.151016402949586e-5, 5.558515254328111e-5, 4.5762592333366357e-5, 3.52878159983866e-5, 3.158533747897978e-5, 2.7601589453085662e-5, 2.634005298921204e-5, 3.257938336348941e-5, 3.525966749160955e-5, 4.791608269461841e-5, 6.692462546337728e-5, 0.00010431331531113154, 0.00013860471518229055, 0.00016440283338529522, 0.00017402140560148933, 0.00018485967928068918, 0.00017305971291019, 0.00016706199945322573, 0.00016936057562499978, 0.00016841162424405075]


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
  # println()

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
# 4	0.0026897816853817284
# 5	0.0035972006097341825
# 6	0.0031528937801516455
# 7	0.0016045015577131128
# 8	0.0011301638718519643
# 9	0.0010532646436953144
# 10	0.0008074310719645224
# 11	0.0007586494447008108
# 12	0.0004647142381289501
# month_i_to_tornado_day_prob = [0.0005912588636110609, 0.0006566448193920728, 0.0011690501952992978, 0.0026897816853817284, 0.0035972006097341825, 0.0031528937801516455, 0.0016045015577131128, 0.0011301638718519643, 0.0010532646436953144, 0.0008074310719645224, 0.0007586494447008108, 0.0004647142381289501]
# Month	wind Day Prob
# 1	0.0021742787126688887
# 2	0.0033437091711604333
# 3	0.004695487907537702
# 4	0.009760425949617445
# 5	0.018254639544583263
# 6	0.030584483887483086
# 7	0.028072308543725295
# 8	0.01955194600286778
# 9	0.006494150753787536
# 10	0.003229257434317082
# 11	0.0026405721266939473
# 12	0.0017064056577864198
# month_i_to_wind_day_prob = [0.0021742787126688887, 0.0033437091711604333, 0.004695487907537702, 0.009760425949617445, 0.018254639544583263, 0.030584483887483086, 0.028072308543725295, 0.01955194600286778, 0.006494150753787536, 0.003229257434317082, 0.0026405721266939473, 0.0017064056577864198]
# Month	hail Day Prob
# 1	0.0006473405429670652
# 2	0.0014506713245414835
# 3	0.004449855622614641
# 4	0.009894493501055387
# 5	0.015590465765259015
# 6	0.015924038298705954
# 7	0.009739799417423558
# 8	0.007078603129470401
# 9	0.002971160427583693
# 10	0.0014843636396106874
# 11	0.0006178489530891205
# 12	0.00036391667180627255
# month_i_to_hail_day_prob = [0.0006473405429670652, 0.0014506713245414835, 0.004449855622614641, 0.009894493501055387, 0.015590465765259015, 0.015924038298705954, 0.009739799417423558, 0.007078603129470401, 0.002971160427583693, 0.0014843636396106874, 0.0006178489530891205, 0.00036391667180627255]
# Month	severe Day Prob
# 1	0.0027906819727792275
# 2	0.004485391403851149
# 3	0.008220348963084304
# 4	0.017084536581618584
# 5	0.029059158188577103
# 6	0.0408017586442409
# 7	0.03414646364574204
# 8	0.024219324303074812
# 9	0.009055027212928201
# 10	0.004581523815163829
# 11	0.0032915877905099764
# 12	0.0021814863760897777
# month_i_to_severe_day_prob = [0.0027906819727792275, 0.004485391403851149, 0.008220348963084304, 0.017084536581618584, 0.029059158188577103, 0.0408017586442409, 0.03414646364574204, 0.024219324303074812, 0.009055027212928201, 0.004581523815163829, 0.0032915877905099764, 0.0021814863760897777]
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
# 4	0.0013452642029077315
# 5	0.0022134991969843935
# 6	0.0031492876293528014
# 7	0.0026848405498328797
# 8	0.001776510227050555
# 9	0.0005593375630399119
# 10	0.0002985255687100561
# 11	0.0002476220613107017
# 12	0.00015382449982362078
# month_i_to_sig_wind_day_prob = [0.00025011009872090174, 0.0003783928615320516, 0.0006006276086852014, 0.0013452642029077315, 0.0022134991969843935, 0.0031492876293528014, 0.0026848405498328797, 0.001776510227050555, 0.0005593375630399119, 0.0002985255687100561, 0.0002476220613107017, 0.00015382449982362078]
# Month	sig_hail Day Prob
# 1	7.916751542765838e-5
# 2	0.00012174983348249795
# 3	0.0005401873122813536
# 4	0.0013791188222479019
# 5	0.0021999192211872866
# 6	0.0019687284328674657
# 7	0.0010964530592716965
# 8	0.0006832811409030232
# 9	0.000316083101926938
# 10	0.00012612148985031082
# 11	4.4488606753264214e-5
# 12	2.286536189910439e-5
# month_i_to_sig_hail_day_prob = [7.916751542765838e-5, 0.00012174983348249795, 0.0005401873122813536, 0.0013791188222479019, 0.0021999192211872866, 0.0019687284328674657, 0.0010964530592716965, 0.0006832811409030232, 0.000316083101926938, 0.00012612148985031082, 4.4488606753264214e-5, 2.286536189910439e-5]
# Month	sig_severe Day Prob
# 1	0.0004149323498785256
# 2	0.0006245327622484127
# 3	0.0012851295093226094
# 4	0.0030234200809628135
# 5	0.0045878184068599516
# 6	0.005053728349770844
# 7	0.0037185810688890133
# 8	0.0024575037875598322
# 9	0.0009516879949254754
# 10	0.0005207052016221992
# 11	0.00048408068763509054
# 12	0.0002759490252219173
# month_i_to_sig_severe_day_prob = [0.0004149323498785256, 0.0006245327622484127, 0.0012851295093226094, 0.0030234200809628135, 0.0045878184068599516, 0.005053728349770844, 0.0037185810688890133, 0.0024575037875598322, 0.0009516879949254754, 0.0005207052016221992, 0.00048408068763509054, 0.0002759490252219173]