module PlotMap

import Dates
import TimeZones
import DelimitedFiles

push!(LOAD_PATH, @__DIR__)
import Grids

colors_path = (@__DIR__) * "/prob_colors.cpt"

HOUR = 60*60

# "12am" "3pm"
function hour_12_hour_ampm_str(datetime)
  hour = Dates.hour(datetime)
  twelve_hour = hour % 12 == 0 ? 12 : hour % 12
  ampm_str = hour <= 11 ? "am" : "pm"
  "$(twelve_hour)$(ampm_str)"
end

# Requires GMT >=6
#
# For daily forecast, provide a multi-hour range for forecast_hour_range
function plot_map(base_path, grid, vals; run_time_utc=nothing, forecast_hour_range=nothing, hrrr_run_hours=Int64[], rap_run_hours=Int64[], href_run_hours=Int64[], sref_run_hours=Int64[])
  open(base_path * ".xyz", "w") do f
    # lon lat val
    DelimitedFiles.writedlm(f, map(i -> (grid.latlons[i][2], grid.latlons[i][1], vals[i]), 1:length(vals)), '\t')
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
    # gmt grdimage href_20190602_t18z_w0.75_sref15z_20190602_19z-11z.nc -nn $region $projection -Clib/prob_colors.cpt # draw the predictions using the projection
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

    println(f, "projection=-Jl-100/35/33/45/0.3")
    println(f, "region=-R-120.7/22.8/-63.7/47.7+r # +r makes the map square rather than weird shaped")

    println(f, "gmt sphinterpolate $base_path.xyz -R-134/-61/21.2/52.5 -I1M -Q0 -G$base_path.nc  # interpolate the xyz coordinates to a grid covering roughly the HRRR's area")

    println(f, "gmt begin $base_path pdf")

    println(f, "gmt coast \$region \$projection -B+g240/245/255+n -ENA -Gc # Use the color of water for the background and begin clipping to north america")
    println(f, "gmt grdimage $base_path.nc -nn \$region \$projection -C$colors_path  # draw the predictions using the projection")
    println(f, "gmt coast -Q # stop clipping")

    println(f, "gmt coast \$region \$projection -A500 -N2/thinnest -t65 # draw state borders 65% transparent")
    println(f, "gmt coast \$region \$projection -A500 -N1 -Wthinnest -t45 # draw country borders and coastlines 45% transparent")

    if !isnothing(run_time_utc) && !isnothing(forecast_hour_range)
      is_day_forecast = forecast_hour_range.start != forecast_hour_range.stop

      println(f, "gmt legend -Dx0.04i/0.04i+w1.7i+l1.3 -C0.03i/0.03i -F+gwhite+pthin << EOF")

      valid_start = run_time_utc + Dates.Hour(forecast_hour_range.start)
      valid_stop  = run_time_utc + Dates.Hour(forecast_hour_range.stop)

      if is_day_forecast
        println(f, "L 7pt,Helvetica-Bold C Nadocast Day $(Dates.format(run_time_utc, "yyyy-m-d H"))Z")
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
      println(f, "L 5pt,Helvetica C PROBABILITIES UNCALIBRATED")
      println(f, "L 4pt,Helvetica,gray C @_nadocast.com@_")
      println(f, "EOF")
    end

    println(f, "gmt end")

    println(f, "pdftoppm $base_path.pdf $base_path -png -r 300 -singlefile")

    println(f, "rm $base_path.nc")
    println(f, "rm $base_path.xyz")
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
  #   # run(`gmt colorbar -DjCB -Cprob_colors.cpt`) # Skip legend for uncalibrated images.
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

end # module PlotMap