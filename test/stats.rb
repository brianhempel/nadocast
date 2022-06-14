# $ ruby stats.rb test_0z.csv > stats_0z.csv

require "csv"

rows = CSV.read(ARGV[0], headers: true)

headers = rows.first.headers

hazards = headers.map { |header| header[/\A(\w+)_spc_painted_sq_mi_.*/, 1]}.compact.uniq

puts ["hazard", "days_count", "threshold", "spc_success_ratio", "spc_pod", "nadocast_success_ratio", "nadocast_pod"].to_csv
hazards.each do |hazard|
  thresholds = headers.grep(/\A#{hazard}_spc_painted_sq_mi_.*/).map { |header| header[/\A#{hazard}_spc_painted_sq_mi_(.+)/, 1] }.uniq
  thresholds.each do |threshold|
    spc_painted_sq_mi             = rows.map { |r| Float(r["#{hazard}_spc_painted_sq_mi_#{threshold}"])             }.sum
    spc_true_positive_sq_mi       = rows.map { |r| Float(r["#{hazard}_spc_true_positive_sq_mi_#{threshold}"])       }.sum
    spc_false_negative_sq_mi      = rows.map { |r| Float(r["#{hazard}_spc_false_negative_sq_mi_#{threshold}"])      }.sum
    nadocast_painted_sq_mi        = rows.map { |r| Float(r["#{hazard}_nadocast_painted_sq_mi_#{threshold}"])        }.sum
    nadocast_true_positive_sq_mi  = rows.map { |r| Float(r["#{hazard}_nadocast_true_positive_sq_mi_#{threshold}"])  }.sum
    nadocast_false_negative_sq_mi = rows.map { |r| Float(r["#{hazard}_nadocast_false_negative_sq_mi_#{threshold}"]) }.sum

    spc_success_ratio      = spc_true_positive_sq_mi      / spc_painted_sq_mi
    spc_pod                = spc_true_positive_sq_mi      / (spc_true_positive_sq_mi + spc_false_negative_sq_mi)
    nadocast_success_ratio = nadocast_true_positive_sq_mi / nadocast_painted_sq_mi
    nadocast_pod           = nadocast_true_positive_sq_mi / (nadocast_true_positive_sq_mi + nadocast_false_negative_sq_mi)

    puts [hazard, rows.size, threshold, spc_success_ratio, spc_pod, nadocast_success_ratio, nadocast_pod].to_csv
  end
end
