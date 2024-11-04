module PlotMap

import Dates
import TimeZones
import DelimitedFiles
import Statistics
using Printf

import PNGFiles
import PNGFiles.ImageCore.ColorTypes

push!(LOAD_PATH, @__DIR__)
import Conus
import Grids
import GeoUtils

tornado_colors_path   = (@__DIR__) * "/tornado_colors.cpt"
wind_hail_colors_path = (@__DIR__) * "/wind_hail_colors.cpt"
sig_colors_path       = (@__DIR__) * "/sig_colors.cpt"
sig_tornado_more_colors_path   = (@__DIR__) * "/sig_tornado_more_colors.cpt" # 10% contour is blackened
sig_wind_hail_more_colors_path = (@__DIR__) * "/sig_wind_hail_more_colors.cpt" # adds a 10% contour
tornado_life_risk_colors_path  = (@__DIR__) * "/tornado_life_risk_colors.cpt"

HOUR = 60*60

# "12am" "3pm"
function hour_12_hour_ampm_str(datetime)
  hour = Dates.hour(datetime)
  twelve_hour = hour % 12 == 0 ? 12 : hour % 12
  ampm_str = hour <= 11 ? "am" : "pm"
  "$(twelve_hour)$(ampm_str)"
end

function writeout_medians(latlons, vals, out_path; miles=150)
  resolution = 0.5 # degrees
  radius     = miles*GeoUtils.METERS_PER_MILE  # miles
  lats = map(latlon -> latlon[1], latlons)
  lons = map(latlon -> latlon[2], latlons)
  lat_range = round(minimum(lats)/resolution, RoundDown)*resolution : resolution : round(maximum(lats)/resolution, RoundUp)*resolution
  lon_range = round(minimum(lons)/resolution, RoundDown)*resolution : resolution : round(maximum(lons)/resolution, RoundUp)*resolution

  out_rows = []

  for lat in lat_range
    for lon in lon_range
      latlon = (lat, lon)
      is = findall(latlons) do val_latlon
        GeoUtils.instantish_distance(val_latlon, latlon) <= radius
      end

      if length(is) >= 1
        median = Statistics.median(vals[is])
        push!(out_rows, (lon, lat, median))
      end
    end
  end

  open(out_path, "w") do f
    # lon lat val
    DelimitedFiles.writedlm(f, out_rows, '\t')
  end

  ()
end


function plot_debug_map(base_path, grid, vals; args...)
  plot_debug_map_latlons(base_path, grid.latlons, vals; args...)
end

# Plots the vals
# Color scheme choices: http://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#built-in-color-palette-tables-cpt
function plot_debug_map_latlons(base_path, latlons, vals; title=nothing, zlow=minimum(vals), zhigh=maximum(vals), steps=10, sparse=false, miles=150, color_scheme="tofino")
  open(base_path * ".xyz", "w") do f
    # lon lat val
    DelimitedFiles.writedlm(f, map(i -> (latlons[i][2], latlons[i][1], vals[i]), 1:length(vals)), '\t')
  end

  # GMT sets internal working directory based on parent pid, so run in sh so each has a diff parent pid.
  open(base_path * ".sh", "w") do f

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    if !sparse
      println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I2k -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")
    else
      # Compute 150mi radius medians
      writeout_medians(latlons, vals, "$(base_path)_medians.xyz", miles = miles)
      println(f, "gmt sphinterpolate $(base_path)_medians.xyz -R-134/-61/21.2/52.5 -I2k -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")
    end

    range     = zhigh - zlow
    step_size = range / steps

    # println(f, "gmt grd2cpt $base_path.nc -C$color_scheme -T$zlow/$zhigh/$step_size > $base_path.cpt")
    println(f, "gmt makecpt -C$color_scheme -T$zlow/$zhigh/$step_size > $base_path.cpt")

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")

    println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$base_path.cpt # draw the numbers using the projection")

    # println(f, "gmt sphtriangulate $base_path.xyz -Qv > triangulated")
    # println(f, "gmt triangulate $base_path.xyz -M -Qn \$region \$projection > triangulated")
    # println(f, "gmt plot triangulated \$region \$projection -L -C$base_path.cpt")

    if sparse
      # Plot individual obs
      println(f, "gmt plot $base_path.xyz -Sc0.07 -W0.25p \$region \$projection -C$base_path.cpt")
    end
    println(f, "gmt coast -Q # stop clipping")

    println(f, "gmt coast \$region \$projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent")
    println(f, "gmt coast \$region \$projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent")

    # Draw legend box.
    if !isnothing(title)
      println(f, "gmt legend -Dx0.04i/0.04i+w1.7i+l1.3 -C0.03i/0.03i -F+gwhite+pthin << EOF")
      println(f, "L 7pt,Helvetica-Bold C $title")
      println(f, "L 7pt,Helvetica-Bold C ") # blank line
      println(f, "L 7pt,Helvetica-Bold C ") # blank line
      println(f, "EOF")
    end

    println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=4p,Helvetica --MAP_FRAME_PEN=0.5p --MAP_TICK_PEN_PRIMARY=0.5p -Dx0.25i/0.25i+w1.3i+h -S -Ba$(range/2) -Np -C$base_path.cpt")

    println(f, "gmt end")

    # println(f, "pdftoppm $base_path.pdf $base_path -png -r 300 -singlefile")
    # # reduce png size
    # println(f, "which pngquant && pngquant 64 --nofs --ext -quantized.png $base_path.png && rm $base_path.png && mv $base_path-quantized.png $base_path.png")
    # println(f, "which oxipng && oxipng  -o max --strip safe --libdeflater $base_path.png")

    println(f, "rm $base_path.nc")
    if sparse
      println(f, "rm $(base_path)_medians.xyz")
    end
    println(f, "rm $base_path.xyz")
    println(f, "rm $base_path.cpt")
  end

  plot() = begin
    run(`sh $base_path.sh`)
    rm(base_path * ".sh")
  end

  # @async plot()
  plot()

  ()
end

# Requires GMT >=6
#
# For daily forecast, provide a multi-hour range for forecast_hour_range
function plot_map(base_path, grid, vals; pdf=true, sig_vals=nothing, run_time_utc=nothing, forecast_hour_range=nothing, hrrr_run_hours=Int64[], rap_run_hours=Int64[], href_run_hours=Int64[], sref_run_hours=Int64[], event_title="Tor", models_str="2021 Models")
  open(base_path * ".xyz", "w") do f
    # lon lat val
    DelimitedFiles.writedlm(f, map(i -> (grid.latlons[i][2], grid.latlons[i][1], vals[i]), 1:length(vals)), '\t')
  end
  if !isnothing(sig_vals)
    open(base_path * "-sig.xyz", "w") do f
      # lon lat val
      DelimitedFiles.writedlm(f, map(i -> (grid.latlons[i][2], grid.latlons[i][1], sig_vals[i]), 1:length(sig_vals)), '\t')
    end
  end
  # gmt nearneighbor cape.xyz -R-139/-58/17/58 -I1k -S15k -Gcape.nc
  # gmt surface cape.xyz -R-139/-58/17/58 -I1k -Gcape.nc
  # gmt triangulate cape.xyz -R-139/-58/17/58 -I1k -Gcape.nc
  # gmt sphinterpolate cape.xyz -R-139/-58/17/58 -I1k -Q0 -Gcape.nc
  # gmt begin map pdf
  # gmt grdimage cape.nc -Jl-100/35/33/45/0.3
  # gmt coast -R-139/-58/17/58 -Jl-100/35/33/45/0.3 -N1 -N2/thinnest -A500 -Wthinnest
  # gmt end
  # open map.pdf

  # GMT sets internal working directory based on parent pid, so run in sh so each has a diff parent pid.
  open(base_path * ".sh", "w") do f

    # projection=-Jl-100/35/33/45/0.3
    # region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped
    #
    # gmt sphinterpolate href_20190602_t18z_w0.75_sref15z_20190602_19z-11z.xyz -R-134/-61/21.2/52.5 -I2k -Q0 -Ghref_20190602_t18z_w0.75_sref15z_20190602_19z-11z.nc # interpolate the xyz coordinates to a grid covering roughly the HRRR's area
    #
    # gmt begin href_20190602_t18z_w0.75_sref15z_20190602_19z-11z pdf
    #
    # gmt coast $region $projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america
    # gmt grdimage href_20190602_t18z_w0.75_sref15z_20190602_19z-11z.nc -nn $region $projection -Clib/tornado_colors.cpt # draw the predictions using the projection
    # gmt coast -Q # stop clipping
    #
    # gmt coast $region $projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent
    # gmt coast $region $projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent
    #
    # gmt legend -Dx0.04i/0.04i+w1.7i+l1.3 -C0.03i/0.03i -F+gwhite+pthin << EOF
    # # L 7pt,Helvetica-Bold C Nadocast Day 2019-7-2 13Z
    # # L 6pt,Helvetica C Valid 2019-7-2 19:00 UTC
    # # L 6pt,Helvetica C Through 2019-7-3 11:00 UTC
    # L 7pt,Helvetica-Bold C Nadocast 2019-7-2 13Z +6
    # L 6pt,Helvetica C Valid 2019-7-2 19:00 UTC
    # L 5pt,Helvetica C 12pm PDT / 1pm MDT / 2pm CDT / 3pm EDT
    # L 4pt,Helvetica C HRRR 11/12/13Z, RAP 11/12/13Z, HREF 6Z, SREF 9Z
    # L 5pt,Helvetica C PROBABILITIES UNCALIBRATED
    # L 4pt,Helvetica,gray C @_nadocast.com@_
    # EOF
    #
    # gmt end

    colors_path = Dict(
      "Tor"              => tornado_colors_path,
      "Wind"             => wind_hail_colors_path,
      "Wind Adjusted"    => wind_hail_colors_path,
      "Hail"             => wind_hail_colors_path,
      "Sigtor"           => sig_tornado_more_colors_path,
      "Sigwind"          => sig_wind_hail_more_colors_path,
      "Sigwind Adjusted" => sig_wind_hail_more_colors_path,
      "Sighail"          => sig_wind_hail_more_colors_path,
      "Tor Life Risk"    => tornado_life_risk_colors_path,
    )[event_title]

    dark_colors_path = replace(colors_path, ".cpt" => "-dark.cpt")

    hazard_str = Dict(
      "Tor"              => "a tornado",
      "Wind"             => "50+ knot t-storm wind",
      "Wind Adjusted"    => "50+ knot t-storm wind",
      "Hail"             => "1+ inch hail",
      "Sigtor"           => "an EF2+ tornado",
      "Sigwind"          => "65+ knot t-storm wind",
      "Sigwind Adjusted" => "65+ knot t-storm wind",
      "Sighail"          => "2+ inch hail",
      "Tor Life Risk"    => "a tornado death",
    )[event_title]

    sig_hazard_str = Dict(
      "Tor"              => "EF2+",
      "Wind"             => "65kt+",
      "Wind Adjusted"    => "65kt+",
      "Hail"             => "2in+",
      "Sigtor"           => "",
      "Sigwind"          => "",
      "Sigwind Adjusted" => "",
      "Sighail"          => "",
      "Tor Life Risk"    => "",
    )[event_title]

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I2k -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")
    println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$colors_path  # draw the predictions using the projection")

    println(f, "gmt contour $base_path.xyz \$region \$projection -An -W0.4p+cl -C$dark_colors_path # draw outlines around each contour")

    # I would use the following dump contour/draw poly method for all plotting because it's fast and pretty, but it doesn't work fill correctly when there are holes in the region.
    if !isnothing(sig_vals)
      println(f, "# Dump contour to file, then draw it as a polygon")
      println(f, "gmt contour $base_path-sig.xyz \$region \$projection -C$sig_colors_path -D$base_path-sig_contour.xy")
      println(f, "gmt psxy $base_path-sig_contour.xy \$region \$projection -Gp26+b-+r800 # pattern fill")
      println(f, "gmt psxy $base_path-sig_contour.xy \$region \$projection -W0.5p # pen, width 0.5pt")
    end

    println(f, "gmt coast -Q # stop clipping")

    println(f, "gmt coast \$region \$projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent")
    println(f, "gmt coast \$region \$projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent")

    # Draw legend box.
    if !isnothing(run_time_utc) && !isnothing(forecast_hour_range)
      is_hourly_forecast     = forecast_hour_range.start == forecast_hour_range.stop
      is_fourhourly_forecast = forecast_hour_range.start + 3 == forecast_hour_range.stop
      is_day_forecast        = !is_hourly_forecast && !is_fourhourly_forecast

      println(f, "gmt legend -Dx0.04i/0.04i+w1.7i+l1.3 -C0.03i/0.03i -F+gwhite+pthin << EOF")

      valid_start = run_time_utc + Dates.Hour(forecast_hour_range.start)
      valid_stop  = run_time_utc + Dates.Hour(forecast_hour_range.stop)

      function legend_title_command(title)
        size =
          if length(title) > 53
            3
          elseif length(title) > 42
            4
          elseif length(title) > 35
            5
          elseif length(title) > 30
            6
          else
            7
          end
        "L $(size)pt,Helvetica-Bold C $title"
      end

      if is_day_forecast
        # because we train ±30min to each hour...
        valid_start -= Dates.Minute(30)
        valid_stop  += Dates.Minute(30)

        is_day2_forecast = forecast_hour_range.start > 12

        if is_day2_forecast
          println(f, legend_title_command("Nadocast $(event_title) $(Dates.format(run_time_utc, "H"))Z Day 2 for $(Dates.format(valid_start, "yyyy-m-d"))"))
        else
          println(f, legend_title_command("Nadocast $(event_title) Day $(Dates.format(run_time_utc, "yyyy-m-d H"))Z"))
        end
        println(f, "L 6pt,Helvetica C Valid $(Dates.format(valid_start, "yyyy-m-d H:MM")) UTC")
        println(f, "L 6pt,Helvetica C Through $(Dates.format(valid_stop, "yyyy-m-d H:MM")) UTC")
      elseif is_fourhourly_forecast
        valid_start -= Dates.Minute(30)
        valid_stop  += Dates.Minute(30)

        println(f, legend_title_command("Nadocast $(event_title) $(Dates.format(run_time_utc, "yyyy-m-d H"))Z +$(forecast_hour_range.start)-$(forecast_hour_range.stop)"))
        println(f, "L 6pt,Helvetica C Valid $(Dates.format(valid_start, "yyyy-m-d H:MM")) UTC")
        println(f, "L 6pt,Helvetica C Through $(Dates.format(valid_stop, "yyyy-m-d H:MM")) UTC")
      else
        forecast_hour = forecast_hour_range.start
        valid_time    = valid_start

        println(f, legend_title_command("Nadocast $(event_title) $(Dates.format(run_time_utc, "yyyy-m-d H"))Z +$forecast_hour"))
        println(f, "L 6pt,Helvetica C Valid $(Dates.format(valid_time, "yyyy-m-d H:MM")) UTC")

        valid_pt = TimeZones.ZonedDateTime(valid_time, TimeZones.tz"America/Los_Angeles", from_utc = true)
        valid_mt = TimeZones.ZonedDateTime(valid_time, TimeZones.tz"America/Denver",      from_utc = true)
        valid_ct = TimeZones.ZonedDateTime(valid_time, TimeZones.tz"America/Chicago",     from_utc = true)
        valid_et = TimeZones.ZonedDateTime(valid_time, TimeZones.tz"America/New_York",    from_utc = true)

        println(f, "L 5pt,Helvetica C $(hour_12_hour_ampm_str(valid_pt)) PT / $(hour_12_hour_ampm_str(valid_mt)) MT / $(hour_12_hour_ampm_str(valid_ct)) CT / $(hour_12_hour_ampm_str(valid_et)) ET")
      end
      if !isempty(vcat(hrrr_run_hours, rap_run_hours, href_run_hours, sref_run_hours))
        sources_str = ""
        if !isempty(hrrr_run_hours)
          sources_str = sources_str * "HRRR $(join(hrrr_run_hours, "/"))Z "
        end
        if !isempty(rap_run_hours)
          sources_str = sources_str * "RAP $(join(rap_run_hours, "/"))Z "
        end
        if !isempty(href_run_hours)
          sources_str = sources_str * "HREF $(join(href_run_hours, "/"))Z "
        end
        if !isempty(sref_run_hours)
          sources_str = sources_str * "SREF $(join(sref_run_hours, "/"))Z"
        end
        println(f, "L 4pt,Helvetica C $sources_str")
      end
      println(f, "L 4pt,Helvetica C $models_str")
      println(f, "L 4pt,Helvetica,gray C @_nadocast.com@_")
      println(f, "EOF")
    end

    println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=4p,Helvetica --MAP_FRAME_PEN=0i --MAP_TICK_LENGTH_PRIMARY=0i --MAP_TICK_PEN_PRIMARY=0 -Dn0.54/0.0695+w1.4i/0.1i+h -S -L0i -Np -C$colors_path")
    println(f, "echo '-95.55 27.93 Chance of $hazard_str within 25 miles of a point.' | gmt text \$region \$projection -F+f4p,Helvetica+jLB")
    if !isnothing(sig_vals)
      println(f, "echo '-95.65 25.35 Black hatched = 10%+ chance of $sig_hazard_str' | gmt text \$region \$projection -F+f4p,Helvetica+jLB")
    end
    if event_title == "Tor Life Risk"
      expected_deaths = 0.0
      for i in 1:length(vals)
        if Conus.is_in_conus(grid.latlons[i])
          expected_deaths += vals[i] * grid.point_areas_sq_miles[i] / (25*π^2)
        end
      end
      println(f, "echo '-95.65 25.35 Naive expected deaths in CONUS: $(@sprintf("%.1f", expected_deaths))' | gmt text \$region \$projection -F+f4p,Helvetica+jLB")
    end
    println(f, "gmt end")

    println(f, "pdftoppm $base_path.pdf $base_path -png -r 300 -singlefile")

    # reduce png size
    println(f, "which pngquant && pngquant 64 --nofs --ext -quantized.png $base_path.png && rm $base_path.png && mv $base_path-quantized.png $base_path.png")
    println(f, "which oxipng && oxipng  -o max --strip safe --libdeflater $base_path.png")

    println(f, "rm $base_path.nc")
    println(f, "rm $base_path.xyz")
    if !isnothing(sig_vals)
      println(f, "rm $base_path-sig.xyz")
      println(f, "rm $base_path-sig_contour.xy")
    end
    if !pdf
      println(f, "rm $base_path.pdf")
    end
  end

  plot() = begin
    run(`sh $base_path.sh`)
    rm(base_path * ".sh")
  end

  @async plot()

  # try
  #   run(`gmt sphinterpolate $(base_path * ".xyz") -R-139/-58/17/58 -I2k -Q0 -G$(base_path * ".nc")`)
  #   # GMT sets internal working directory based on parent pid, so this is fine to run in parallel (by process).
  #   run(`gmt begin $base_path.pdf pdf`)
  #   run(`gmt grdimage $(base_path * ".nc") -nn -Jl-100/35/33/45/0.3 -C$(colors_path)`)
  #   run(`gmt coast -R-139/-58/17/58 -Jl-100/35/33/45/0.3 -N1 -N2/thinnest -A500 -Wthinnest`)
  #   # run(`gmt colorbar -DjCB -Ctornado_colors.cpt`) # Skip legend for uncalibrated images.
  #   run(`gmt end`)
  #   # run(`convert -density 250 $(base_path * ".pdf") $(base_path * ".png")`) # `convert` utility from ImageMagick
  #   # run(`optipng -o2 -strip all $base_path.png`)                            # `optipng` utility
  #
  #   rm(base_path * ".nc")
  #   rm(base_path * ".xyz")
  #   # run(`rm $(base_path * ".nc")`)
  #   # run(`rm $(base_path * ".xyz")`)
  # catch
  # end
  # try
  #   run(`open $(base_path * ".pdf")`)
  #   run(`rm $(base_path * ".xyz")`)
  #   run(`rm $(base_path * ".nc")`)
  # catch
  # end
  ()
end

# Plots the vals
# Color scheme choices: http://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#built-in-color-palette-tables-cpt
function plot_map_for_paper(base_path, latlons, vals; title=nothing, zlow=minimum(vals), zhigh=maximum(vals), steps=10, sparse_vals=nothing, sparse_val_symbol="c", sparse_val_size="0.07", sparse_val_pen="0.25p", colors="tofino", pdf=false, label_contours=true, nearest_neighbor=false)
  # GMT sets internal working directory based on parent pid, so run in sh so each has a diff parent pid.
  open(base_path * ".sh", "w") do f

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    if vals != []
      open(base_path * ".xyz", "w") do f
        # lon lat val
        DelimitedFiles.writedlm(f, map(i -> (latlons[i][2], latlons[i][1], vals[i]), 1:length(vals)), '\t')
      end

      if nearest_neighbor
        println(f, "gmt nearneighbor $base_path.xyz -R-134/-61/21.2/52.5 -I2k -Nn -G$base_path.nc  # interpolate the xyz coordinates to a grid")
      else
        println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I2k -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid")
      end
    end

    range     = zhigh - zlow
    step_size = range / (isnothing(steps) ? 0 : steps)

    cpt_path =
      if endswith(colors, ".cpt")
        colors
      else
        println(f, "gmt makecpt -C$colors -T$zlow/$zhigh/$step_size > $base_path.cpt")
        "$base_path.cpt"
      end

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")

    if vals != []
      println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$cpt_path # draw the vals using the projection")
      label_contours && println(f, "gmt grdcontour $(base_path).nc \$region \$projection -A+f3p+i -W0.2p,70/70/70 -C$cpt_path -Gd2i # draw labels on each contour")
    end

    # println(f, "gmt sphtriangulate $base_path.xyz -Qv > triangulated")
    # println(f, "gmt triangulate $base_path.xyz -M -Qn \$region \$projection > triangulated")
    # println(f, "gmt plot triangulated \$region \$projection -L -C$cpt_path")

    # sparsevals should be a vec of (latlon, val)
    if !isnothing(sparse_vals)
      # Plot individual obs
      open(base_path * "_sparse.xyz", "w") do f
        # lon lat val
        DelimitedFiles.writedlm(f, map(i -> (sparse_vals[i][1][2], sparse_vals[i][1][1], sparse_vals[i][2]), 1:length(sparse_vals)), '\t')
      end
      # Possible symbols: https://docs.generic-mapping-tools.org/6.4/plot.html#s
      println(f, "gmt plot $(base_path)_sparse.xyz -S$(sparse_val_symbol)$(sparse_val_size) -W$(sparse_val_pen) \$region \$projection -C$cpt_path")
    end
    println(f, "gmt coast -Q # stop clipping")

    println(f, "gmt coast \$region \$projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent")
    println(f, "gmt coast \$region \$projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent")

    # Draw legend box.
    if !isnothing(title)
      println(f, "gmt legend -DjCT+o0i/0.07i -C0.03i/0.03i -F+gwhite+pthin << EOF")
      println(f, "L 10pt,Helvetica-Bold C $title")
      println(f, "EOF")
    end

    if endswith(colors, ".cpt")
      # println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=7p,Helvetica --MAP_FRAME_PEN=0.5p --MAP_TICK_PEN_PRIMARY=0.5p -Dx0.16i/0.3i+w1.6i/0.12i+h -F+c0.08i+p0.5p+gwhite -S -L0i -Np -C$cpt_path")
      if isnothing(steps)
        println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=7p,Helvetica --MAP_FRAME_PEN=0.5p --MAP_TICK_PEN_PRIMARY=0.5p -Dx0.16i/0.3i+w1.6i/0.12i+h -F+c0.08i+p0.5p+gwhite -S+c -Np -C$cpt_path")
      else
        println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=7p,Helvetica --MAP_FRAME_PEN=0.5p --MAP_TICK_PEN_PRIMARY=0.5p -Dx0.16i/0.3i+w1.6i/0.12i+h -F+c0.08i+p0.5p+gwhite -S -Bx$(step_size) -Np -C$cpt_path")
      end
    else
      println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=7p,Helvetica --MAP_FRAME_PEN=0.5p --MAP_TICK_PEN_PRIMARY=0.5p -Dx0.16i/0.3i+w1.6i/0.12i+h -F+c0.08i+p0.5p+gwhite -S -Ba$(range/2) -Np -C$cpt_path")
    end

    println(f, "gmt end")

    if !pdf
      println(f, "pdftoppm $base_path.pdf $base_path -png -r 300 -singlefile")
      # reduce png size
      println(f, "which pngquant && pngquant 128 --nofs --ext -quantized.png $base_path.png && rm $base_path.png && mv $base_path-quantized.png $base_path.png")
      println(f, "which oxipng && oxipng  -o max --strip safe --libdeflater $base_path.png")
      # println(f, "rm $base_path.pdf")
    end
    if vals != []
      println(f, "rm $base_path.nc")
      println(f, "rm $base_path.xyz")
    end
    !isnothing(sparse_vals)   && println(f, "rm $(base_path)_sparse.xyz")
    !endswith(colors, ".cpt") && println(f, "rm $base_path.cpt")
  end

  run(`sh $base_path.sh`)
  rm(base_path * ".sh")

  ()
end


using PNGFiles.ImageCore.ColorTypes
using PNGFiles.ImageCore.ColorTypes.FixedPointNumbers
using ColorVectorSpace

spc_tornado_colors =
  [ (0.0,  RGB{N0f8}(1,1,1))
  , (0.02, RGB{N0f8}(0.067,0.541,0.078))
  , (0.05, RGB{N0f8}(0.541,0.278,0.165))
  , (0.1,  RGB{N0f8}(0.996,0.780,0.180))
  , (0.15, RGB{N0f8}(0.988,0.051,0.106))
  , (0.30, RGB{N0f8}(0.988,0.157,0.988))
  , (0.45, RGB{N0f8}(0.565,0.224,0.918))
  , (0.60, RGB{N0f8}(0.082,0.310,0.537))
  ]

spc_wind_hail_colors =
  [ (0.0,  RGB{N0f8}(1,1,1))
  , (0.05, RGB{N0f8}(0.541,0.278,0.165))
  , (0.15, RGB{N0f8}(0.996,0.780,0.180))
  , (0.30, RGB{N0f8}(0.988,0.051,0.106))
  , (0.45, RGB{N0f8}(0.988,0.157,0.988))
  , (0.60, RGB{N0f8}(0.565,0.224,0.918))
  ]

spc_sig_colors =
  [ (0.0,  RGB{N0f8}(1,1,1))
  , (0.1,  RGB{N0f8}(0.3,0.3,0.3))
  ]

# Regular tornado colors, but dark gray 10%
spc_sig_tornado_more_colors =
  [ (0.0,  RGB{N0f8}(1,1,1))
  , (0.02, RGB{N0f8}(0.067,0.541,0.078))
  , (0.05, RGB{N0f8}(0.541,0.278,0.165))
  , (0.1,  RGB{N0f8}(0.3,0.3,0.3))
  , (0.15, RGB{N0f8}(0.988,0.051,0.106))
  , (0.30, RGB{N0f8}(0.988,0.157,0.988))
  , (0.45, RGB{N0f8}(0.565,0.224,0.918))
  , (0.60, RGB{N0f8}(0.082,0.310,0.537))
  ]

# Regular wind and hail colors, but additional dark gray 10%
spc_sig_wind_hail_more_colors =
  [ (0.0,  RGB{N0f8}(1,1,1))
  , (0.05, RGB{N0f8}(0.541,0.278,0.165))
  , (0.1,  RGB{N0f8}(0.3,0.3,0.3))
  , (0.15, RGB{N0f8}(0.996,0.780,0.180))
  , (0.30, RGB{N0f8}(0.988,0.051,0.106))
  , (0.45, RGB{N0f8}(0.988,0.157,0.988))
  , (0.60, RGB{N0f8}(0.565,0.224,0.918))
  ]

function threshold_colorer(threshold_colors)
  function prob_to_color(p)
    _, prior_color = threshold_colors[1]
    for (threshold, color) in threshold_colors
      if p < threshold*0.9999
        return prior_color
      end
      prior_color = color
    end
    return prior_color
  end

  prob_to_color
end

event_name_to_colorer = Dict(
  "tornado"      => threshold_colorer(spc_tornado_colors),
  "wind"         => threshold_colorer(spc_wind_hail_colors),
  "wind_adj"     => threshold_colorer(spc_wind_hail_colors),
  "hail"         => threshold_colorer(spc_wind_hail_colors),
  "sig_tornado"  => threshold_colorer(spc_sig_colors),
  "sig_wind"     => threshold_colorer(spc_sig_colors),
  "sig_wind_adj" => threshold_colorer(spc_sig_colors),
  "sig_hail"     => threshold_colorer(spc_sig_colors),
)

event_name_to_colorer_more_sig_colors = Dict(
  "tornado"      => threshold_colorer(spc_tornado_colors),
  "wind"         => threshold_colorer(spc_wind_hail_colors),
  "wind_adj"     => threshold_colorer(spc_wind_hail_colors),
  "hail"         => threshold_colorer(spc_wind_hail_colors),
  "sig_tornado"  => threshold_colorer(spc_sig_tornado_more_colors),
  "sig_wind"     => threshold_colorer(spc_sig_wind_hail_more_colors),
  "sig_wind_adj" => threshold_colorer(spc_sig_wind_hail_more_colors),
  "sig_hail"     => threshold_colorer(spc_sig_wind_hail_more_colors),
)

function conus_lines_href_5k_native_proj()
  Float32.(Gray.(PNGFiles.load((@__DIR__) * "/conus_lines_href_5k_native_cropped_proj.png")))
end

# Hashed
function shade_forecast_labels(labels, img)
  h, w = size(img)
  labels = permutedims(reshape(labels, (w, h)))
  # Now flip vertically
  for j in 1:(h ÷ 2)
    row = labels[j,:]
    labels[j,:] = labels[h - j + 1,:]
    labels[h - j + 1,:] = row
  end

  out = deepcopy(img)

  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
  label_dithering = deepcopy(labels)
  for j in 1:h
    for i in 1:w
      if mod(i + j, 2) == 0
        if label_dithering[j,i] > 0.75 && labels[j,i] > 0
          out[j,i] = RGB{N0f8}(0.0, 0.0, 0.0)
          error = label_dithering[j,i] - 1f0
        elseif label_dithering[j,i] > 0.25 && labels[j,i] > 0
          out[j,i] = out[j,i] * N0f8(0.5)
          error = label_dithering[j,i] - 0.5f0
        else
          error = label_dithering[j,i] - 0f0
        end
        # The original Floyd-Steinberg coeffs don't make sense since we are already cross-hatching.
        i+2 <= w &&             (label_dithering[j,  i+2] += error * 1/3f0)
        j+1 <= h && i-1 >= 1 && (label_dithering[j+1,i-1] += error * 1/3f0)
        j+1 <= h && i+1 <= w && (label_dithering[j+1,i+1] += error * 1/3f0)
      end
    end
  end

  # for i in 1:size(img,1)
  #   for j in 1:size(img,2)
  #     if mod(i + j, 2) == 0 && labels[i,j] > 0
  #       gray = 1.0 - labels[i,j]
  #       out[i,j] = RGB{N0f8}(gray, gray, gray)
  #     end
  #   end
  # end

  out
end

function multiply_image(grid_layer, img)
  h, w = size(img)
  grid_layer = permutedims(reshape(grid_layer, (w, h)))
  # Now flip vertically
  for j in 1:(h ÷ 2)
    row = grid_layer[j,:]
    grid_layer[j,:] = grid_layer[h - j + 1,:]
    grid_layer[h - j + 1,:] = row
  end

  img .* grid_layer
end

function add_conus_lines_href_5k_native_proj_80_pct(img)
  lines = 0.2f0 .+ 0.8f0 .* conus_lines_href_5k_native_proj()
  img .* lines
end

# PlotMap.plot_fast(base_path, grid, vals; val_to_color=PlotMap.prob_to_spc_color, post_process=PlotMap.add_conus_lines_href_5k_native_proj_80_pct)

function plot_fast_no_resample(base_path, grid, vals; val_to_color=Gray, post_process=add_conus_lines_href_5k_native_proj_80_pct)
  # Awww yeah rotation.
  vals = permutedims(reshape(vals, (grid.width, grid.height)))[:,:]
  # Now flip vertically
  for j in 1:(grid.height ÷ 2)
    row = vals[j,:]
    vals[j,:] = vals[grid.height - j + 1,:]
    vals[grid.height - j + 1,:] = row
  end
  PNGFiles.save("$base_path.png", post_process(val_to_color.(vals)); compression_level = 9)
end

function plot_fast(base_path, grid, vals; val_to_color=Gray, post_process=add_conus_lines_href_5k_native_proj_80_pct)
  href_cropped_5km_grid = Conus.href_cropped_5km_grid()
  resampler = Grids.get_upsampler(grid, href_cropped_5km_grid) # not always upsampling, but this does nearest neighbor

  plot_fast_no_resample(base_path, href_cropped_5km_grid, resampler(vals); val_to_color=val_to_color, post_process=post_process)
end

function optimize_png(base_path; wait = true, quantization_levels = nothing)
  if isnothing(quantization_levels)
    run(`oxipng -o max --strip safe --libdeflater $(base_path * ".png")`; wait = wait)
  else
    run(`sh -c "pngquant $quantization_levels --nofs --ext -quantized.png $base_path.png && rm $base_path.png && mv $base_path-quantized.png $base_path.png && oxipng -o max --strip safe --libdeflater $(base_path * ".png")"`; wait = wait)
  end
end

end # module PlotMap
