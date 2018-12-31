# LightGBM counts features important by number of times a feature is chosen to split on (no weighting).
#
# But it also computes it before the trimming we have to do manually, and don't hanve nice header names, so here we do it ourselves.
#
# Usage: ruby read_lightgbm_feature_importance.rb model/trained.model [n|--all|--grouped]

path = ARGV[0] || (raise "Usage: ruby read_lightgbm_feature_importance.rb model/trained.model")

model_dir  = File.dirname(path)
model_text = File.read(path)

feature_uses = model_text.scan(/split_feature=.*/).flat_map { |splits_line| splits_line.scan(/\d+/) }.map(&:to_i)

module Enumerable
  def counts
    counts = {}
    each do |item|
      counts[item] ||= 0
      counts[item]  += 1
    end
    counts.sort_by {|_, count| -count}.to_h
  end
end

headers = File.read("#{model_dir}/headers.txt").split("\n").drop(10)

counts = feature_uses.counts


if ARGV.include?("--all")
  headers.each_with_index do |header, header_i|
    puts [counts[header_i] || "", header].join("\t")
  end
elsif ARGV.include?("--grouped")
  headers.each_with_index.group_by { |header, header_i| header[/\A[^:]+:[^:]+/] }.each do |header_group_prefix, header_is_pairs|
    count = header_is_pairs.map(&:last).map { |header_i| counts[header_i] || 0 }.sum
    puts [count, header_group_prefix].join("\t")
  end
else
  n = Integer(ARGV[1] || "30")

  counts.take(n).each do |header_i, count|
    puts [count, headers[header_i]].join("\t")
  end
end
