require "csv"

csi_step = 0.1
sr_step  = 0.0025

csis = (csi_step..1-csi_step).step(csi_step)

puts csis.flat_map {|csi| ["sr", "pod_for_csi_%.2f" % csi]}.to_csv

def format_pod(pod)
  pod >= 0 ? ("%.4f" % pod) : nil
end

(sr_step..1-sr_step).step(sr_step).each do |sr|
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

  puts pods_for_csis.flat_map {|pod| ["%.4f" % sr, format_pod(pod)]}.to_csv
end