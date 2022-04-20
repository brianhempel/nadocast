#!/usr/bin/env ruby
#
# Default forecast date / run hour is based on current UTC time.
#
# Will continuously re-attempt downloads until they exist. Not currently written to look in the archives if you want a historic forecast and didn't download the HRRR/RAP.
#
# Usage: FORECAST_DATE=2021-12-10 RUN_HOUR=10 HRRR_RAP=true TWEET=true ruby download_and_forecast.rb
#
# Defaults to today and current UTC hour with HRRR_RAP=true (and TWEET=false for DoPredict.jl)

require "date"
require "fileutils"


FORECAST_DATE = ENV["FORECAST_DATE"]       ? Date.parse(ENV["FORECAST_DATE"]) : Time.now.utc.to_date
RUN_HOUR      = ENV["RUN_HOUR"]            ? Integer(ENV["RUN_HOUR"])         : Time.now.utc.hour
HRRR_RAP      = ENV["HRRR_RAP"].to_s != "" ? ENV["HRRR_RAP"] == "true"        : true
JULIA         = ENV["JULIA"]               ? ENV["JULIA"]                     : "julia"


loop do
  2.times do
    FileUtils.cd File.expand_path("../../get_data", __FILE__)

    download_pids = []
    if HRRR_RAP
      run_hours_str = "#{(RUN_HOUR-2)%24},#{(RUN_HOUR-1)%24},#{RUN_HOUR}" # mod 24, but the date might be off...no worries we don't use HRRR/RAP for 0Z
      hrrr_pid      = Process.spawn({ "RUN_HOURS" => run_hours_str, "FORECAST_HOURS" => "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"}, "ruby get_hrrr.rb", out: "get_hrrr.log", err: "get_hrrr.log")
      rap_pid       = Process.spawn({ "RUN_HOURS" => run_hours_str },                                                                    "ruby get_rap.rb",  out: "get_rap.log",  err: "get_rap.log")
      download_pids += [hrrr_pid, rap_pid]
    end
    sref_pid = Process.spawn("ruby get_sref.rb", out: "get_sref.log", err: "get_sref.log")
    href_pid = Process.spawn("ruby get_href.rb", out: "get_href.log", err: "get_href.log")
    download_pids += [sref_pid, href_pid]

    while download_pids != []
      download_pids.select! do |pid|
        Process.wait2(pid, Process::WNOHANG) == nil
      end
      sleep 2
    end

    FileUtils.cd File.expand_path("..", __FILE__)

    # DoPredict.jl will fail if the forecasts are not all downloaded.
    if system("JULIA_NUM_THREADS=#{ENV["CORE_COUNT"]} RUN_HOUR=#{RUN_HOUR} FORECAST_DATE=#{FORECAST_DATE} time #{JULIA} --project DoPredict.jl")
      exit 0
    end
  end

  sleep(60*5)
end
