# Requires GMT >=6
function plot_map(base_path, lats, lons, vals)
  open(base_path * ".xyz", "w") do f
    # lon lat val
    writedlm(f, map(i -> (lons[i], lats[i], vals[i]), 1:length(vals)), '\t')
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
  try
    run(`gmt sphinterpolate $(base_path * ".xyz") -R-139/-58/17/58 -I1M -Q0 -G$(base_path * ".nc")`)
    # GMT sets internal working directory based on parent pid, so this is fine to run in parallel (by process).
    run(`gmt begin $base_path.pdf pdf`)
    run(`gmt grdimage $(base_path * ".nc") -nn -Jl-100/35/33/45/0.3 -Cprob_colors.cpt`)
    run(`gmt coast -R-139/-58/17/58 -Jl-100/35/33/45/0.3 -N1 -N2/thinnest -A500 -Wthinnest`)
    # run(`gmt colorbar -DjCB -Cprob_colors.cpt`) # Skip legend for uncalibrated images.
    run(`gmt end`)
    # run(`convert -density 250 $(base_path * ".pdf") $(base_path * ".png")`) # `convert` utility from ImageMagick
    # run(`optipng -o2 -strip all $base_path.png`)                            # `optipng` utility
    run(`rm $(base_path * ".nc")`)
    run(`rm $(base_path * ".xyz")`)
  end
  try
    # run(`open $(base_path * ".pdf")`)
    # run(`rm $(base_path * ".xyz")`)
    # run(`rm $(base_path * ".nc")`)
  end
  ()
end
