group_tokens = [
  /\bUpstream\w*2hr/,
  /\bUpstream\w*1hr/,
  /\bUpstreamCRAIN/,
  /\bUpstream\w*CAPE/,
  /\bUpstream\w*DPT/,
  /\bUpstream/,
  /StormUpstream/,
  /3hrGated/,
  /6hrGated/,
  /9hrGated/,
  /-1hr:/,
  /\+1hr:/,
  /:\n/,
  /:25mi/,
  /:50mi/,
  /:100mi/,
  /:prob /,
  /ens mean:/,
  /mi mean/,
  /3hr delta/,
  /3hr max/,
  /3hr mean/,
  /3hr min/,
  /forward grad/,
  /leftward grad/,
  /linestraddling grad/,
  /month/,
  /hour_in_day/,
  /spatial/,
  /\*/,
  /surface/,
  /entire atmosphere/,
  /calculated/,
  /:10 m/,
  /:80 m/,
  /925 mb/,
  /850 mb/,
  /700 mb/,
  /500 mb/,
  /250 mb/,
  /90-0 mb/,
  /180-0 mb/,
  /SCP/,
  /cloud base/,
  /CAPE/,
  /climatol/,
  /Convergence/,
  /DifferentialDivergence/,
  /Divergence/,
  /850mb/,
  /925mb/,
  /REF/,
  /sqrt/,
  /Vorticity/,
]

ALL_FEATURES = File.read("features2021models.txt").lines.to_a

simple_features             = ALL_FEATURES.map { |line| Regexp.new(Regexp.escape(line.split(":").first + ":")) }.uniq
simple_features_with_levels = ALL_FEATURES.map { |line| Regexp.new(Regexp.escape(line.split(":")[0..1].join(":") + ":")) }.uniq

USED_FEATURES =
  File.read("f2-13_feature_importance_2021_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f2-13_feature_importance_2021_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f2-13_feature_importance_2021_hail_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_hail_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_hail_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f2-13_feature_importance_2021_sig_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_sig_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_sig_tornado_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f2-13_feature_importance_2021_sig_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_sig_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_sig_wind_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f2-13_feature_importance_2021_sig_hail_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f13-24_feature_importance_2021_sig_hail_models.txt").lines.grep(/\A\d+\s\d+\s/) +
  File.read("f24-35_feature_importance_2021_sig_hail_models.txt").lines.grep(/\A\d+\s\d+\s/)

TOTAL_COUNT = USED_FEATURES.map(&:split).map(&:first).map(&:to_i).sum

def do_it(tokens)
  tokens.map do |token|
    frac_of_all  = ALL_FEATURES.grep(token).size.to_f / ALL_FEATURES.size
    frac_of_used = USED_FEATURES.grep(token).map(&:split).map(&:first).map(&:to_i).sum.to_f / TOTAL_COUNT

    [frac_of_used / frac_of_all, token.source]
  end.sort.reverse.each do |relative_use, str|
    puts("%.3f\t#{str}" % relative_use)
  end
end

do_it(simple_features)
do_it(simple_features_with_levels)
do_it(group_tokens)