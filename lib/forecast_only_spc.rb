#!/usr/bin/env ruby
#
# Default forecast date / run hour is based on current UTC time.
#
# Usage: RUN_DATE=2021-12-10 RUN_HOUR=12 JULIA_NUM_THREADS=16 ruby forecast_only_spc.rb
#
# Defaults to today and current UTC hour

require "date"
require "fileutils"

lib_dir = File.expand_path("..", __FILE__)

RUN_DATE = ENV["RUN_DATE"] ? Date.parse(ENV["RUN_DATE"]) : Time.now.utc.to_date
RUN_HOUR = ENV["RUN_HOUR"] ? Integer(ENV["RUN_HOUR"])    : Time.now.utc.hour
JULIA    = ENV["JULIA"]    ? ENV["JULIA"]                : "julia"

MINUTE     = 60
HOUR       = 60*MINUTE
start_time = Time.now

loop do
  2.times do
    FileUtils.cd lib_dir

    # DoPredict.jl will fail if the forecasts are not all downloaded.
    if system("RUN_HOUR=#{RUN_HOUR} RUN_DATE=#{RUN_DATE} time #{JULIA} --project DoPredictSPC.jl")
      exit 0
    end
  end

  if Time.now - start_time > 6*HOUR
    STDERR.puts("RUN_HOUR=#{RUN_HOUR} RUN_DATE=#{RUN_DATE} 6hr timeout")
    exit(1)
  end

  sleep(2*MINUTE)
end
