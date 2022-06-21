require "csv"

csis = (0.1..0.9).step(0.1)

puts (["sr"] + csis.map {|csi| "pod_for_csi_%.1f" % csi}).to_csv

(0.01..0.99).step(0.01).each do |sr|
  # CSI = tp / (painted + fn)
  # SR  = tp / painted
  # POD = tp / (tp + fn)

  # Worked this out on paper:
  # CSI = SR*POD / (SR + POD - SR*POD)
  #     = 1 / (1/SR + 1/POD - 1)
  # POD = 1 / (1/CSI - 1/SR + 1)

  pods_for_csis = csis.map do |csi|
    pod = 1.0 / (1.0/csi - 1.0/sr + 1.0)
  end

  puts (["%.2f" % sr] + pods_for_csis.map {|x| "%.4f" % x}).to_csv
end