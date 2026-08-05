#!/usr/bin/env ruby

# Downsamples a tree of grib2 files onto the 15km/12km/9km grids defined in
# lib/refs_downsample_grid_commands.rb, mirroring the directory structure into a
# sibling tree per grid:
#
#   /Volumes/SREF_HREF_3/refs/202608/20260801/refs_conus_20260801_t00z_mean_f01.grib2
#   -> /Volumes/SREF_HREF_3/refs_12km/202608/20260801/refs_conus_20260801_t00z_mean_f01.grib2
#
# File names are preserved, so downstream code only has to swap the root path
# (pass --suffix-names for foo.12km.grib2 names instead).
#
# Made to be run intermittently (e.g. cron) after get_data/get_refs.rb. Re-running is
# cheap: a file is skipped unless its output is missing, empty, or older than the input.
# Outputs are written to a temp file in the destination directory and renamed only after
# the record-count/unpaired/undefined checks pass, so an interrupted or failed run leaves
# nothing that a later run would mistake for finished work.
#
#   downsample_refs.rb /Volumes/SREF_HREF_3/refs
#   downsample_refs.rb --km 12 -j 8 /Volumes/SREF_HREF_3/refs /Volumes/SREF_HREF_1/refs

require "digest"
require "fileutils"
require "open3"
require "optparse"
require "shellwords"
require "tmpdir"

options = {
  km:           nil, # nil = every grid the grid-definition file knows about
  jobs:         1, # wgrib2 parallelizes the conversion
  force:        false,
  dry_run:      false,
  min_age:      120, # seconds; a file still being downloaded is not a file to convert
  suffix_names: true,
  quiet:        false,
}

parser = OptionParser.new do |o|
  o.banner = "Usage: #{File.basename($PROGRAM_NAME)} [options] SRC_DIR [SRC_DIR...]\n" \
             "Mirrors each SRC_DIR to SRC_DIR_15km, SRC_DIR_12km, SRC_DIR_9km."
  o.on("--km LIST", "Only these grids, e.g. --km 12 or --km 12,9 (default: all)") { |v| options[:km] = v.split(",").map { |s| Integer(s.strip) } }
  o.on("-j", "--jobs N", Integer, "Files converted in parallel (default #{options[:jobs]})") { |v| options[:jobs] = v }
  o.on("-f", "--force", "Redo files that already look up to date") { options[:force] = true }
  o.on("-n", "--dry-run", "List what would be converted") { options[:dry_run] = true }
  o.on("--min-age SECONDS", Integer, "Skip inputs modified this recently (default #{options[:min_age]})") { |v| options[:min_age] = v }
  o.on("--no-suffix-names", "Name outputs foo.grib2 instead of foo.12km.grib2") { options[:suffix_names] = false }
  o.on("-q", "--quiet", "Only print warnings and the summary") { options[:quiet] = true }
  o.on("-h", "--help") { puts o; exit }
end
parser.parse!

SRC_ROOTS = ARGV.map { |dir| File.expand_path(dir).chomp("/") }
if SRC_ROOTS.empty?
  warn parser
  exit 1
end
missing = SRC_ROOTS.reject { |dir| Dir.exist?(dir) }
abort "no such directory: #{missing.join(', ')}" unless missing.empty?

# Only one run at a time: these are minutes-long wgrib2 pipelines over a network volume,
# and two runs would just duplicate the work. Later cron firings drop out silently.
LOCK = File.open(File.join(Dir.tmpdir, "downsample_refs-#{Digest::MD5.hexdigest(SRC_ROOTS.sort.join("\n"))}.lock"), File::RDWR | File::CREAT, 0644)
unless LOCK.flock(File::LOCK_EX | File::LOCK_NB)
  puts "another downsample_refs.rb is already working on #{SRC_ROOTS.join(', ')}; exiting"
  exit 0
end

# Brings in the grid definitions (GRIDS) and the wgrib2 plumbing: pipeline, downsampled_path,
# record_count, undef_totals.
require_relative "../lib/refs_downsample_grid_commands"

GRIDS_TO_MAKE = GRIDS.select { |_n, _c, km| options[:km].nil? || options[:km].include?(km) }
if GRIDS_TO_MAKE.empty?
  abort "--km #{options[:km].join(',')}: no such grid (have #{GRIDS.map { |_n, _c, km| km }.join(', ')})"
end

def out_root(src_root, km)
  "#{src_root}_#{km}km"
end

def out_path(src_root, rel_path, km, suffix_names)
  path = File.join(out_root(src_root, km), rel_path)
  suffix_names ? downsampled_path(path, km) : path
end

# Skip work already done. mtime rather than a record count so the common case (nothing new)
# costs no wgrib2 runs at all; the rename-after-validation means an existing output is
# always a complete one.
def up_to_date?(input, output)
  File.exist?(output) && File.size(output) > 0 && File.mtime(output) >= File.mtime(input)
end

STDOUT.sync = true
MUTEX  = Mutex.new
COUNTS = Hash.new(0)
def say(line, quiet: false)
  MUTEX.synchronize { puts line unless quiet }
end

def count(key, by = 1)
  MUTEX.synchronize { COUNTS[key] += by }
end

# Temp output lives in the destination directory so the rename is atomic (same filesystem).
# Leading dot keeps it out of the way; the pattern is what the stale sweep below matches.
TMP_GLOB = ".*.downsample_refs_tmp*"
def tmp_path(output, tag)
  File.join(File.dirname(output), ".#{File.basename(output)}.downsample_refs_tmp#{tag}.#{Process.pid}.#{Thread.current.object_id}")
end

# A killed run (reboot, SIGKILL) can strand temp files. Nothing else cleans them up, and this
# script runs forever on a full disk otherwise.
def sweep_stale_temps(dirs, dry_run)
  dirs.each do |dir|
    Dir.glob(File.join(dir, "**", TMP_GLOB)).each do |path|
      next unless (File.mtime(path) rescue Time.now) < Time.now - 3600
      warn "removing stale temp file #{path}"
      File.delete(path) unless dry_run
    end
  end
end

start_time = Time.now
sweep_stale_temps(SRC_ROOTS.product(GRIDS_TO_MAKE.map { |_n, _c, km| km }).map { |root, km| out_root(root, km) }.select { |d| Dir.exist?(d) }, options[:dry_run])

# [src_root, relative path] for every grib2 file under the source roots.
inputs = SRC_ROOTS.flat_map do |root|
  Dir.glob("**/*.{grib2,grb2}", base: root).sort.reject { |rel| rel.include?(".unpaired.") }.map { |rel| [root, rel] }
end

queue = Queue.new # Thread::Queue only got that name in Ruby 3.0
inputs.each { |job| queue << job }
queue.close

MKDIR_MUTEX = Mutex.new

def convert(src_root, rel, grids, options)
  input = File.join(src_root, rel)

  todo = grids.reject { |_n, _c, km| !options[:force] && up_to_date?(input, out_path(src_root, rel, km, options[:suffix_names])) }
  count(:up_to_date, grids.size - todo.size)
  return if todo.empty?

  age = Time.now - File.mtime(input)
  if age < options[:min_age]
    say("skipping #{rel}: modified #{age.round}s ago, may still be downloading", quiet: options[:quiet])
    return count(:too_fresh, todo.size)
  end

  if options[:dry_run]
    todo.each { |_n, _c, km| say("would write #{out_path(src_root, rel, km, options[:suffix_names])}") }
    return count(:would_convert, todo.size)
  end

  in_records =
    begin
      record_count(input)
    rescue => e
      warn "FAILED to read #{input}: #{e.message.lines.first}"
      return count(:failed, todo.size)
    end

  todo.each do |n, c, km|
    output   = out_path(src_root, rel, km, options[:suffix_names])
    tmp_out  = tmp_path(output, "")
    unpaired = tmp_path(output, ".unpaired")
    MKDIR_MUTEX.synchronize { FileUtils.mkdir_p(File.dirname(output)) }

    begin
      cmd = pipeline(input.shellescape, tmp_out.shellescape, unpaired.shellescape, G, c, n, km)
      # Via bash with pipefail: a failure in the -new_grid_order stage would otherwise be
      # hidden by the second wgrib2 exiting 0 on its truncated input.
      out, status = Open3.capture2e("bash", "-o", "pipefail", "-c", cmd)
      unless status.success?
        warn "FAILED #{input} -> #{output}\n#{cmd}\n#{out}"
        count(:failed)
        next
      end

      # -new_grid drops a U or V it cannot pair, warning only on stderr and still exiting 0,
      # so the record count is the check that actually catches it.
      out_records = record_count(tmp_out)
      if out_records != in_records
        warn "FAILED #{output}: #{out_records} records, input had #{in_records}"
        count(:failed)
        next
      end
      if File.size(unpaired) > 0
        warn "FAILED #{output}: unpaired fields did not make it into the output"
        count(:failed)
        next
      end
      undefs = undef_totals(tmp_out)
      unless undefs == [0]
        warn "FAILED #{output}: undefined points: #{undefs.join(', ')} per field"
        count(:failed)
        next
      end

      File.rename(tmp_out, output)
      say("#{rel} -> #{output}  #{c[:nx]}x#{c[:ny]}  #{out_records} records", quiet: options[:quiet])
      count(:converted)
    rescue => e
      warn "FAILED #{input} -> #{output}: #{e.message.lines.first}"
      count(:failed)
    ensure
      [tmp_out, unpaired].each { |path| File.delete(path) if File.exist?(path) }
    end
  end
end

threads = options[:jobs].times.map do
  Thread.new do
    while job = queue.pop
      convert(job[0], job[1], GRIDS_TO_MAKE, options)
    end
  end
end
threads.each(&:join)

tally = COUNTS.reject { |_key, n| n.zero? }.map { |key, n| "#{n} #{key}" }.join(", ")
puts "#{inputs.size} grib2 files, #{GRIDS_TO_MAKE.map { |_n, _c, km| "#{km}km" }.join(' ')}: " \
     "#{tally.empty? ? 'nothing to do' : tally} in #{(Time.now - start_time).round}s"

exit 1 if COUNTS[:failed] > 0
