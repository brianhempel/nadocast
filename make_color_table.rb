def hex_to_rgb(hex_color)
  [
    hex_color[1..2].to_i(16),
    hex_color[3..4].to_i(16),
    hex_color[5..6].to_i(16),
  ]
end

def rgb_to_cpt(rgb)
  rgb.join("/")
end

def blend(rgb1, rgb2, ratio)
  r1, g1, b1 = rgb1
  r2, g2, b2 = rgb2

  [
    r1 + ((r2 - r1)*ratio).round,
    g1 + ((g2 - g1)*ratio).round,
    b1 + ((b2 - b1)*ratio).round,
  ]
end

colors_and_stops = [
  [0.00, hex_to_rgb("#FFFFFF")],
  [0.02, hex_to_rgb("#118A14")],
  [0.05, hex_to_rgb("#8A472A")],
  [0.10, hex_to_rgb("#FEC72E")],
  [0.15, hex_to_rgb("#FC0D1B")],
  [0.30, hex_to_rgb("#FC28FC")],
  [0.45, hex_to_rgb("#9039EA")],
  [0.60, hex_to_rgb("#154F89")],
  [0.75, hex_to_rgb("#000000")],
]

nan_color = hex_to_rgb("#FF00FF")

(0...colors_and_stops.size-1).each do |i|
  stop1, rgb1 = colors_and_stops[i]
  stop2, rgb2 = colors_and_stops[i+1]
  rgb_low  = rgb1
  rgb_high = blend(rgb1, rgb2, 0.0)
  puts [
    stop1, rgb_to_cpt(rgb_low),
    stop2, rgb_to_cpt(rgb_high),
  ].join("\t")
end

puts "B\t#{rgb_to_cpt(colors_and_stops[0][1])}"
puts "F\t#{rgb_to_cpt(colors_and_stops[-1][1])}"
puts "N\t#{rgb_to_cpt(nan_color)}"
