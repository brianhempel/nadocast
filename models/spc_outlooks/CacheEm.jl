# I don't trust ArchGDAL to work in threads, so do this as separate processes
#
# $ ruby -e '16.times.map {|n| Thread.new { system("julia --project=../.. CacheEm.jl #{n+1} 16") } }.map(&:join)'

push!(LOAD_PATH, (@__DIR__))
push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import SPCOutlooks
import Forecasts

process_i, process_count = parse.(Int64, ARGS)

forecasts = vcat(SPCOutlooks.forecasts_day_0600(), SPCOutlooks.forecasts_day_1300(), SPCOutlooks.forecasts_day_1630());

for forecast in filter(f -> hash(Forecasts.time_title(f)) % process_count == process_i-1, forecasts)
  Forecasts.data(forecast)
end

