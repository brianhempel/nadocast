#!/usr/bin/env ruby [a.grib2 ...]

# If given no arguments, prints (does not run) the wgrib2 -new_grid commands to make 15km, 12km, and 9km
# versions of the REFS 3km CONUS grid. If given args, makes 15km 12km and 9km versions of each.
#
# Each coarse point sits at the center of an NxN block of original 3km points, starting
# with the SW-most complete block and ending with the NE-most complete block. Same
# convention as Grib2.read_grid's `downsample` (lib/Grib2.jl): the SW point is at source
# index div(N+1,2), i.e. 3 for 5x5, 2.5 for 4x4 (between points, as in read_grid's 4-point
# mean), and 2 for 3x3.
#
# The grids are hard-coded below. They were derived from sample REFS and HREF files, but
# nothing depends on those files being around anymore -- the derivation is kept, commented
# out, next to the numbers it produced.

require "shellwords"
require "tmpdir"

WGRIB2      = ENV["WGRIB2"] || "wgrib2"
DOWNSAMPLES = [5, 4, 3]

# The categorical fields are stored 0/1 with a binary scale factor of -1, i.e. the only
# values the packing can represent are multiples of 0.5. -set_grib_type same carries that
# over to the output, so budget interpolation's area fractions get rounded right back to
# 0/0.5/1. Repack just these at 2^-8 (255 distinct values). Deliberately not a global
# -set_scaling/-set_bin_prec: that would coarsen the fields that are finely packed already
# (APCP is 22 bits at a 1e-5 step). The -else resets the scaling, since -set_scaling is
# sticky state that would otherwise leak into every later message.
#
# The match is on the 'wt ens mean' suffix, NOT on the field names alone: the prob files carry
# records with these same four names as 'prob >=1 <0:prob fcst 0/14' probabilities. Those are
# 0-100 and want to stay at their 0.5 step (201 levels, plenty); repacking them at 2^-8 takes
# prob CRAIN from 4 bits to 16 for no gain. So this block is a no-op on a prob file.
CATEGORICAL = %w[CRAIN CSNOW CICEP CFRZR]

CATEGORICAL_SCALING = "-if ':(#{CATEGORICAL.join('|')}):surface:.*wt ens mean' -set_scaling same -8 -else -set_scaling same same -endif"

def wgrib2(*args)
  out = `#{WGRIB2} #{args.shelljoin} 2>&1`
  raise "wgrib2 #{args.shelljoin} failed:\n#{out}" unless $?.success?
  out
end

def lambert_arg(g)
  "lambert:#{format('%g', g[:lov])}:#{format('%g', g[:latin1])}:#{format('%g', g[:latin2])}"
end

# The REFS 3km CONUS source grid, read off refs.t00z.mean.f01.conus.grib2 with the commented-out
# grid_info below. Lat1/Lon1 are the SW corner at the full micro-degree precision of the header.
G = {
  nx: 1799, ny: 1059,
  lat1: 21.138123, lon1: 237.280472,
  lov: 262.5, latin1: 38.5, latin2: 38.5,
  dx: 3000.0, dy: 3000.0,
  winds: "grid",
}.freeze

# The three output grids, keyed by downsample factor: 5 -> 15km, 4 -> 12km, 3 -> 9km.
# lat1/lon1 are the center of the SW-most complete NxN block of source points; nx/ny count the
# complete blocks, less `trim` columns off the east edge (see east_columns_to_trim below).
CORNERS = {
  5 => { lat1: 21.201847, lon1: 237.318839, nx: 358, ny: 211, full_nx: 359, trim: 1 },
  4 => { lat1: 21.185915, lon1: 237.309242, nx: 448, ny: 264, full_nx: 449, trim: 1 },
  3 => { lat1: 21.169984, lon1: 237.299648, nx: 597, ny: 353, full_nx: 599, trim: 2 },
}.freeze

# How the two tables above were derived, kept for the record. It needed sample REFS and HREF
# files in test_grib2s/, which nothing depends on anymore; restore the files and un-comment to
# re-derive (e.g. if the REFS grid ever changes).
#
# REFS = File.expand_path("../test_grib2s/refs.t00z.mean.f01.conus.grib2", __dir__)
#
# These grids are meant to hold HREF as well as REFS, so they get trimmed to HREF's coverage.
# HREF fields come in two bitmap variants (809032 and 809035 undefined points); CRAIN carries
# the larger one, so sizing against it covers every field in both the mean and prob files.
#
# HREF = File.expand_path("../test_grib2s/href.t00z.conus.mean.f01.grib2", __dir__)
# HREF_WORST_BITMAP_FIELD = "CRAIN"
#
# Parse `wgrib2 -grid` for the first message.
#
# def grid_info(path)
#   out = wgrib2(path, "-grid", "-end")
#   raise "not a Lambert conformal grid:\n#{out}" unless out =~ /Lambert Conformal:\s+\((\d+) x (\d+)\)/
#
#   num = ->(name) { out[/\b#{name}\s+(-?[\d.]+)/, 1]&.to_f or raise("no #{name} in:\n#{out}") }
#
#   {
#     nx: $1.to_i, ny: $2.to_i,
#     lat1: num["Lat1"], lon1: num["Lon1"],
#     lov: num["LoV"], latin1: num["Latin1"], latin2: num["Latin2"],
#     dx: num["Dx"], dy: num["Dy"],
#     winds: out.include?("winds(grid)") ? "grid" : "earth",
#   }
# end
#
# HREF's grid is much larger than its members' coverage, so its fields carry a bitmap with a
# ragged undefined margin. The SE corner of the REFS-derived grids pokes out past it (into the
# Atlantic near Turks and Caicos), leaving a few undefined points. Regrid HREF's worst-case
# bitmap onto a candidate grid and report how many east columns have to go for a clean result.
# REFS itself produces no undefined points; this trim exists solely to stay inside HREF.
#
# def east_columns_to_trim(dir, g, lat1, lon1, nx, ny, dx)
#   return 0 unless File.exist?(HREF)
#
#   out = File.join(dir, "href_probe.grb2")
#   txt = File.join(dir, "href_probe.txt")
#   wgrib2(HREF, "-match", ":#{HREF_WORST_BITMAP_FIELD}:", "-inv", "/dev/null",
#          "-new_grid_winds", g[:winds], "-new_grid_interpolation", "budget",
#          "-new_grid", lambert_arg(g), "#{lon1}:#{nx}:#{format('%g', dx)}", "#{lat1}:#{ny}:#{format('%g', dx)}", out)
#   wgrib2(out, "-inv", "/dev/null", "-no_header", "-text", txt)
#
#   # Values come back in the grid's own WE:SN order, so index k is (i, j) = (k % nx, k / nx).
#   undefined_is = File.foreach(txt).with_index.filter_map { |v, k| k % nx + 1 if v.to_f > 1e20 }
#   undefined_is.empty? ? 0 : nx - (undefined_is.min - 1)
# end
#
# G = grid_info(REFS)
#
# To find the coarse SW corner we let wgrib2 do all the geolocation: build a probe grid with the
# same projection and corner as REFS but half the spacing, where the center of the SW NxN
# block lands exactly on index (N,N), then crop there and read the corner back out of the
# grib header at full micro-degree precision. (-gridout only prints 3 decimals, ~100m,
# which is too coarse to define a grid with, and can't represent the 4x4 center at all.)
#
# CORNERS = Dir.mktmpdir do |dir|
#   probe = File.join(dir, "probe.grb2")
#   wgrib2(REFS, "-end", "-inv", "/dev/null", "-new_grid_winds", G[:winds],
#          "-new_grid", lambert_arg(G),
#          "#{G[:lon1]}:#{2 * DOWNSAMPLES.max}:#{G[:dx] / 2}",
#          "#{G[:lat1]}:#{2 * DOWNSAMPLES.max}:#{G[:dy] / 2}", probe)
#
#   # If the projection args were wrong the probe wouldn't round-trip its own corner.
#   p_info = grid_info(probe)
#   unless [p_info[:lat1], p_info[:lon1]] == [G[:lat1], G[:lon1]]
#     raise "probe grid corner #{p_info[:lat1]},#{p_info[:lon1]} != REFS corner #{G[:lat1]},#{G[:lon1]}"
#   end
#
#   DOWNSAMPLES.to_h do |n|
#     crop = File.join(dir, "crop_#{n}.grb2")
#     wgrib2(probe, "-end", "-inv", "/dev/null", "-ijsmall_grib", "#{n}:#{n + 1}", "#{n}:#{n + 1}", crop)
#     lat1, lon1 = grid_info(crop).values_at(:lat1, :lon1)
#
#     full_nx, ny = G[:nx] / n, G[:ny] / n # complete blocks only
#     trim = east_columns_to_trim(dir, G, lat1, lon1, full_nx, ny, n * G[:dx])
#     [n, { lat1: lat1, lon1: lon1, nx: full_nx - trim, ny: ny, full_nx: full_nx, trim: trim }]
#   end
# end

# foo.grib2 -> foo.12km.grib2, foo.grb2 -> foo.12km.grb2
def downsampled_path(path, km, suffix = "")
  return "#{path}.#{km}km#{suffix}" unless path =~ /\.gri?b2\z/i
  path.sub(/\.(gri?b2)\z/i) { ".#{km}km#{suffix}.#{$1}" }
end

# The one place the command is built, so what gets printed is exactly what gets run.
def pipeline(input, output, unpaired, g, c, n, km)
  stage1 = "#{WGRIB2} #{input} -inv /dev/null -new_grid_order - #{unpaired}"
  stage2 = [
    WGRIB2, "-",
    "-inv /dev/null",
    "-set_grib_type same",
    CATEGORICAL_SCALING,
    "-new_grid_winds #{g[:winds]}",
    "-new_grid_interpolation budget",
    "-new_grid #{lambert_arg(g)}",
    "#{c[:lon1]}:#{c[:nx]}:#{format('%g', n * g[:dx])}",
    "#{c[:lat1]}:#{c[:ny]}:#{format('%g', n * g[:dy])}",
    output,
  ].join(" ")
  "#{stage1} | \\\n  #{stage2}"
end

def describe(g, c, n, km)
  center = (n + 1) / 2.0 # SW block center, in source grid indices
  [
    "# #{km}km: #{n}x#{n} blocks of the #{g[:nx]}x#{g[:ny]} source -> #{c[:nx]}x#{c[:ny]}",
    "#   SW point at source (i,j) = (#{center}, #{center}); NE point at (#{center + (c[:nx] - 1) * n}, #{center + (c[:ny] - 1) * n})",
    "#   Drops #{g[:nx] - c[:nx] * n} source column(s) off the E edge and #{g[:ny] - c[:ny] * n} row(s) off the N edge:",
    "#     #{(c[:full_nx] - c[:nx]) * n} of those are the #{c[:trim]} block(s) trimmed to stay inside HREF's bitmap; the rest had no complete block.",
  ]
end

def record_count(path)
  wgrib2(path).lines.size
end

def undef_totals(path)
  wgrib2(path, "-stats").scan(/undef=(\d+)/).flatten.map(&:to_i).uniq.sort
end

# [downsample factor, corner/size info, km] for each output grid. get_data/downsample_refs.rb
# requires this file for these, so everything below here is CLI-only.
GRIDS = DOWNSAMPLES.map { |n| [n, CORNERS[n], (n * G[:dx] / 1000.0).round] }

return unless __FILE__ == $PROGRAM_NAME

g, grids = G, GRIDS

if ARGV.empty?
  puts "# REFS source grid: #{g[:nx]}x#{g[:ny]}, Dx=Dy=#{format('%g', g[:dx])}m, #{lambert_arg(g)}"
  puts "# Corners found by cropping a half-spacing probe grid, so wgrib2 does all the geolocation."
  puts "# Source winds are #{g[:winds]}-relative; -new_grid_winds keeps them that way."
  puts "#"
  puts "# The -new_grid_order stage is required: -new_grid only vector-interpolates a U field"
  puts "# that is immediately followed by its V, and REFS stores all 6 pressure-level UGRDs in a"
  puts "# row and then all 6 VGRDs. Unpaired U/V are DROPPED (a stderr warning, exit status 0),"
  puts "# which silently costs 10 of the 70 fields. -new_grid_order pairs them up first."
  puts "# Check the unpaired file is empty -- anything in it did not make it to the output."
  puts "#"
  puts "# Pass grib2 files as arguments to run these instead of printing them."
  puts
  grids.each do |n, c, km|
    puts describe(g, c, n, km)
    puts pipeline("in.grib2", "in.#{km}km.grib2", "in.#{km}km.unpaired.grib2", g, c, n, km)
    puts
  end
  exit
end

missing = ARGV.reject { |p| File.exist?(p) }
abort "no such file: #{missing.join(', ')}" unless missing.empty?

ARGV.each do |input|
  in_records = record_count(input)
  ok = true

  grids.each do |n, c, km|
    output   = downsampled_path(input, km)
    unpaired = downsampled_path(input, km, ".unpaired")
    cmd      = pipeline(input.shellescape, output.shellescape, unpaired.shellescape, g, c, n, km)

    puts cmd
    unless system(cmd)
      warn "  FAILED: #{input} -> #{output}"
      ok = false
      next
    end

    # -new_grid drops a U or V it cannot pair, warning only on stderr and still exiting 0, so
    # the record count is the check that actually catches it.
    out_records = record_count(output)
    if out_records != in_records
      warn "  WARNING: #{output} has #{out_records} records, input had #{in_records}"
      ok = false
    end
    if File.size(unpaired) > 0
      warn "  WARNING: #{unpaired} is not empty -- those fields are missing from #{output}"
      ok = false
    else
      File.delete(unpaired)
    end

    undefs = undef_totals(output)
    note = undefs == [0] ? "" : "  UNDEFINED POINTS: #{undefs.join(', ')} per field"
    warn "  WARNING:#{note}" unless note.empty?
    puts "  -> #{output}  #{c[:nx]}x#{c[:ny]}  #{out_records} records#{note}"
    puts
  end

  warn "#{input}: finished with warnings" unless ok
end
