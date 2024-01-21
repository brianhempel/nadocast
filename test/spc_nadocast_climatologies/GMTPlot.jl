module GMTPlot

import DelimitedFiles

# Plots the vals
# Color scheme choices: http://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#built-in-color-palette-tables-cpt
function plot_map(base_path, latlons, vals; title=nothing, zlow=minimum(vals), zhigh=maximum(vals), steps=10, sparse_vals=nothing, sparse_val_symbol="c", sparse_val_size="0.07", sparse_val_pen="0.25p", colors="tofino", pdf=false, label_contours=true, nearest_neighbor=false)
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

end