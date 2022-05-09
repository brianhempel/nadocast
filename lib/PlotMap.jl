module PlotMap

import Dates
import TimeZones
import DelimitedFiles

import PNGFiles
import PNGFiles.ImageCore.ColorTypes

push!(LOAD_PATH, @__DIR__)
import Grids

tornado_colors_path   = (@__DIR__) * "/tornado_colors.cpt"
wind_hail_colors_path = (@__DIR__) * "/wind_hail_colors.cpt"
sig_colors_path       = (@__DIR__) * "/sig_colors.cpt"

HOUR = 60*60

# "12am" "3pm"
function hour_12_hour_ampm_str(datetime)
  hour = Dates.hour(datetime)
  twelve_hour = hour % 12 == 0 ? 12 : hour % 12
  ampm_str = hour <= 11 ? "am" : "pm"
  "$(twelve_hour)$(ampm_str)"
end


# Plots the vals using a default color scheme
function plot_debug_map(base_path, grid, vals; title=nothing, zlow=minimum(vals), zhigh=maximum(vals), steps=10)
  color_scheme = "tofino" # Choices: http://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#built-in-color-palette-tables-cpt

  open(base_path * ".xyz", "w") do f
    # lon lat val
    DelimitedFiles.writedlm(f, map(i -> (grid.latlons[i][2], grid.latlons[i][1], vals[i]), 1:length(vals)), '\t')
  end

  # GMT sets internal working directory based on parent pid, so run in sh so each has a diff parent pid.
  open(base_path * ".sh", "w") do f

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I1M -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")

    range     = zhigh - zlow
    step_size = range / steps

    # println(f, "gmt grd2cpt $base_path.nc -C$color_scheme -T$zlow/$zhigh/$step_size > $base_path.cpt")
    println(f, "gmt makecpt -C$color_scheme -T$zlow/$zhigh/$step_size > $base_path.cpt")

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")
    println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$base_path.cpt # draw the numbers using the projection")
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

    println(f, "rm $base_path.nc")
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
function plot_map(base_path, grid, vals; sig_vals=nothing, run_time_utc=nothing, forecast_hour_range=nothing, hrrr_run_hours=Int64[], rap_run_hours=Int64[], href_run_hours=Int64[], sref_run_hours=Int64[], event_title="Tor")
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
    # gmt sphinterpolate href_20190602_t18z_w0.75_sref15z_20190602_19z-11z.xyz -R-134/-61/21.2/52.5 -I1M -Q0 -Ghref_20190602_t18z_w0.75_sref15z_20190602_19z-11z.nc # interpolate the xyz coordinates to a grid covering roughly the HRRR's area
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
      "Tor"     => tornado_colors_path,
      "Wind"    => wind_hail_colors_path,
      "Hail"    => wind_hail_colors_path,
      "Sigtor"  => tornado_colors_path,
      "Sigwind" => wind_hail_colors_path,
      "Sighail" => wind_hail_colors_path,
    )[event_title]

    hazard_str = Dict(
      "Tor"     => "a tornado",
      "Wind"    => "50+ knot t-storm wind",
      "Hail"    => "1+ inch hail",
      "Sigtor"  => "an EF2+ tornado",
      "Sigwind" => "65+ knot t-storm wind",
      "Sighail" => "2+ inch hail",
    )[event_title]

    sig_hazard_str = Dict(
      "Tor"     => "EF2+",
      "Wind"    => "65kt+",
      "Hail"    => "2in+",
      "Sigtor"  => "",
      "Sigwind" => "",
      "Sighail" => "",
    )[event_title]

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I1M -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")

    if !isnothing(sig_vals)
      println(f, "gmt sphinterpolate $base_path-sig.xyz -R-134/-61/21.2/52.5 -I1M -Q0 -G$base_path-sig.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")
    end

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")
    println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$colors_path  # draw the predictions using the projection")

    if !isnothing(sig_vals)
      println(f, "# Dump contour to file, then draw it as a polygon")
      println(f, "gmt contour $base_path-sig.xyz \$region \$projection -C$sig_colors_path -D$base_path-sig_contour.xy")
      println(f, "gmt psxy $base_path-sig_contour.xy \$region \$projection -Gp26+b-+r800 # pattern fill")
      println(f, "gmt psxy $base_path-sig_contour.xy \$region \$projection -W0.4p # pen, width 0.4pt")
    end

    println(f, "gmt coast -Q # stop clipping")

    println(f, "gmt coast \$region \$projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent")
    println(f, "gmt coast \$region \$projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent")

    # Draw legend box.
    if !isnothing(run_time_utc) && !isnothing(forecast_hour_range)
      is_day_forecast = forecast_hour_range.start != forecast_hour_range.stop

      println(f, "gmt legend -Dx0.04i/0.04i+w1.7i+l1.3 -C0.03i/0.03i -F+gwhite+pthin << EOF")

      valid_start = run_time_utc + Dates.Hour(forecast_hour_range.start)
      valid_stop  = run_time_utc + Dates.Hour(forecast_hour_range.stop)

      if is_day_forecast
        println(f, "L 7pt,Helvetica-Bold C Nadocast $(event_title) Day $(Dates.format(run_time_utc, "yyyy-m-d H"))Z")
        println(f, "L 6pt,Helvetica C Valid $(Dates.format(valid_start, "yyyy-m-d H:MM")) UTC")
        println(f, "L 6pt,Helvetica C Through $(Dates.format(valid_stop, "yyyy-m-d H:MM")) UTC")
      else
        forecast_hour = forecast_hour_range.start
        valid_time    = valid_start

        println(f, "L 7pt,Helvetica-Bold C Nadocast $(Dates.format(run_time_utc, "yyyy-m-d H"))Z +$forecast_hour")
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
      println(f, "L 4pt,Helvetica C 2021 Models")
      println(f, "L 4pt,Helvetica,gray C @_nadocast.com@_")
      println(f, "EOF")
    end

    println(f, "gmt colorbar --FONT_ANNOT_PRIMARY=4p,Helvetica --MAP_FRAME_PEN=0i --MAP_TICK_LENGTH_PRIMARY=0i --MAP_TICK_PEN_PRIMARY=0 -Dn0.54/0.0695+w1.4i/0.1i+h -S -L0i -Np -C$colors_path")
    println(f, "echo '-95.55 27.93 Chance of $hazard_str within 25 miles of a point.' | gmt text \$region \$projection -F+f4p,Helvetica+jLB")
    if !isnothing(sig_vals)
      println(f, "echo '-95.65 25.35 Black hatched = 10%+ chance of $sig_hazard_str' | gmt text \$region \$projection -F+f4p,Helvetica+jLB")
    end
    println(f, "gmt end")

    println(f, "pdftoppm $base_path.pdf $base_path -png -r 300 -singlefile")

    println(f, "rm $base_path.nc")
    println(f, "rm $base_path.xyz")
    if !isnothing(sig_vals)
      println(f, "rm $base_path-sig.nc")
      println(f, "rm $base_path-sig.xyz")
      println(f, "rm $base_path-sig_contour.xy")
    end
  end

  plot() = begin
    run(`sh $base_path.sh`)
    rm(base_path * ".sh")
  end

  @async plot()

  # try
  #   run(`gmt sphinterpolate $(base_path * ".xyz") -R-139/-58/17/58 -I1M -Q0 -G$(base_path * ".nc")`)
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
  "tornado"     => threshold_colorer(spc_tornado_colors),
  "wind"        => threshold_colorer(spc_wind_hail_colors),
  "hail"        => threshold_colorer(spc_wind_hail_colors),
  "sig_tornado" => threshold_colorer(spc_sig_colors),
  "sig_wind"    => threshold_colorer(spc_sig_colors),
  "sig_hail"    => threshold_colorer(spc_sig_colors),
)

function conus_lines_href_5k_native_proj()
  Float32.(Gray.(PNGFiles.load((@__DIR__) * "/conus_lines_href_5k_native_cropped_proj.png")))
end

# Hashed
function shade_forecast_labels(labels, img)
  orig_h, orig_w = size(img)
  labels = permutedims(reshape(labels, (orig_w, orig_h)))
  # Now flip vertically
  for j in 1:(orig_h ÷ 2)
    row = labels[j,:]
    labels[j,:] = labels[orig_h - j + 1,:]
    labels[orig_h - j + 1,:] = row
  end

  out = deepcopy(img)

  for i in 1:size(img,1)
    for j in 1:size(img,2)
      if mod(i + j, 2) == 0 && labels[i,j] > 0.5
        out[i,j] *= 0f0
      end
    end
  end

  out
end

function add_conus_lines_href_5k_native_proj_80_pct(img)
  lines = 0.2f0 .+ 0.8f0 .* conus_lines_href_5k_native_proj()
  img .* lines
end

# PlotMap.plot_fast(base_path, grid, vals; val_to_color=PlotMap.prob_to_spc_color, post_process=PlotMap.add_conus_lines_href_5k_native_proj_80_pct)

function plot_fast(base_path, grid, vals; val_to_color=Gray, post_process=add_conus_lines_href_5k_native_proj_80_pct)
  # Awww yeah rotation.
  vals = permutedims(reshape(vals, (grid.width, grid.height)))
  # Now flip vertically
  for j in 1:(grid.height ÷ 2)
    row = vals[j,:]
    vals[j,:] = vals[grid.height - j + 1,:]
    vals[grid.height - j + 1,:] = row
  end
  PNGFiles.save("$base_path.png", post_process(val_to_color.(vals)); compression_level = 9)
end

end # module PlotMap
