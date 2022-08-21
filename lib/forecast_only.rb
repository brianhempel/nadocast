#!/usr/bin/env ruby
#
# Default forecast date / run hour is based on current UTC time.
#
# Will continuously re-attempt downloads until they exist. Not currently written to look in the archives if you want a historic forecast and didn't download the HRRR/RAP.
#
# Usage: RUN_DATE=2021-12-10 RUN_HOUR=10 HRRR_RAP=true TWEET=true ruby download_and_forecast.rb
#
# Defaults to today and current UTC hour with HRRR_RAP=true (and TWEET=false for DoPredict.jl)

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
    if system("JULIA_NUM_THREADS=#{ENV["CORE_COUNT"]} RUN_HOUR=#{RUN_HOUR} RUN_DATE=#{RUN_DATE} time #{JULIA} --project DoPredict.jl")
      exit 0
    end
  end

  if Time.now - start_time > 10*HOUR
    STDERR.puts("RUN_HOUR=#{RUN_HOUR} RUN_DATE=#{RUN_DATE} 10hr timeout")
    exit(1)
  end

  sleep(5*MINUTE)
end
