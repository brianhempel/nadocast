



# function bagged_weights(bagging_temperature, n)
#   out = zeros(Float32, n)
#   seed = rand(Int)

#   # logs and powers are slow; sample from pre-computed distribution for non-extreme values.

#   weight_lookup = map(r -> (-log(r))^bagging_temperature, 0.005f0:0.01f0:0.995f0)
#   # Temperatures between 0 and 1 shift the expected effective total weight
#   # downward slightly (up to ~11.5%) which effectively changes the
#   # min_data_in_leaf and L2 regularization. Compensate.
#   temperature_compensation = length(weight_lookup) / sum(weight_lookup)
#   weight_lookup .*= temperature_compensation

#   rng = Random.MersenneTwister(abs(seed + 1 * 1234))
#   @inbounds for i in 1:length(out)
#     bin_i = rand(rng, 1:100)
#     count =
#       if bin_i in 2:99
#         weight_lookup[bin_i]
#       else
#         r = rand(rng, Float32) * 0.01f0 + (bin_i == 100 ? 0.99f0 : 1f-7)
#         (-log(r))^bagging_temperature * temperature_compensation
#       end

#     out[i] = count
#   end

#   out
# end

# maximum(bagged_weights(0.25, 28000000))


# function print_tree(tree, level = 0; feature_i_to_name = nothing, bin_splits = nothing)
#   indentation = repeat("    ", level)
#   if isa(tree, MemoryConstrainedTreeBoosting.Node)
#     feature_name = isnothing(feature_i_to_name) ? "feature $(tree.feature_i)" : feature_i_to_name(tree.feature_i)
#     split_str = isnothing(bin_splits) ? "$(tree.split_i)" : "$(bin_splits[tree.feature_i][tree.split_i])"
#     println(indentation * "$feature_name\tsplit at $split_str")
#     print_tree(tree.left,  level + 1, feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
#     print_tree(tree.right, level + 1, feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
#   else
#     println(indentation * "Δscore $(tree.Δscore)")
#   end
# end

function dot_tree(tree, level = 0; feature_i_to_name = nothing, bin_splits = nothing)
  if level == 0
    println("strict digraph {")
  end
  id(tree) = Int64(pointer_from_objref(tree))
  # indentation = repeat("    ", level)
  if isa(tree, MemoryConstrainedTreeBoosting.Node)
    feature_name = isnothing(feature_i_to_name) ? "feature $(tree.feature_i)" : feature_i_to_name(tree.feature_i)
    split_str = isnothing(bin_splits) ? "$(tree.split_i)" : "$(round(bin_splits[tree.feature_i][tree.split_i]; sigdigits=2))"

    name1, name2, _blah, threshold_and_window, mean_or_grad = split(feature_name, ":")
    name = name1 * ":" * name2

    window = match(r"[\-+]?\dhr(?: \w+)?", threshold_and_window)
    window = isnothing(window) ? "" : window.match

    threshold = strip(replace(threshold_and_window, window => ""))
    threshold = replace(threshold, "wt ens mean" => "")

    name = replace(name, "REFC:entire atmosphere" => "Composite reflectivity")
    name = replace(name, "REFD:1000 m above ground" => "1km reflectivity")
    name = replace(name, "sqrtSBCAPE" => "sqrt(SBCAPE)")
    name = replace(name, "sqrtMLCAPE" => "sqrt(MLCAPE)")
    name = replace(name, ":calculated" => "")
    name = replace(name, "CSNOW:surface" => "Snow (0 or 1)")
    name = replace(name, "FLGHT:surface" => "Flight category")

    threshold_and_mean_or_grad = "\\n" * join(filter(str -> str != "", [threshold, mean_or_grad]), ", ")

    window = window == "" ? "" : "\\n$window"

    # println(indentation * "$feature_name\tsplit at $split_str")
    println("_$(id(tree)) [label=\"$name$threshold_and_mean_or_grad$window\" shape=\"box\"]")
    println("_$(id(tree)) -> _$(id(tree.left)) [label=\"<$split_str\"]")
    println("_$(id(tree)) -> _$(id(tree.right)) [label=\"≥$split_str\"]")
    dot_tree(tree.left,  level + 1, feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
    dot_tree(tree.right, level + 1, feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
  else
    Δscore = round(tree.Δscore; sigdigits=2)
    perhaps_plus = Δscore > 0 ? "+" : ""
    println("_$(id(tree)) [label=\"$(perhaps_plus)$(Δscore)\"]")
    # println(indentation * "Δscore $(tree.Δscore)")
  end
  if level == 0
    println("}")
  end
end

import MemoryConstrainedTreeBoosting


bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-09-23T02.26.17.492_tornado_climatology_all/676_trees_loss_0.0012007512.model")

length(trees)

feature_names = readlines("../href_mid_2018_forward/features_2021v2.txt");

feature_i_to_name(i) = feature_names[i]

# # print_tree(trees[2]; feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
dot_tree(trees[2]; feature_i_to_name = feature_i_to_name, bin_splits = bin_splits)
# strict digraph {
# _5472539152 [label="MXUPHL:5000-2000 m above ground\nestimated from probs, 50mi mean\n+1hr" shape="box"]
# _5472539152 -> _5472536848 [label="<1.3"]
# _5472539152 -> _5472539008 [label="≥1.3"]
# _5472536848 [label="sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m*(200+SBCIN)\n25mi mean\n3hr mean" shape="box"]
# _5472536848 -> _5472474816 [label="<8.8e6"]
# _5472536848 -> _5472536704 [label="≥8.8e6"]
# _5472474816 [label="MAXREF:1000 m above ground\nestimated from probs, 25mi mean\n3hr mean" shape="box"]
# _5472474816 -> _5472397072 [label="<23.0"]
# _5472474816 -> _5472397792 [label="≥23.0"]
# _5472397072 [label="sqrt(SBCAPE)*BWD0-6km\n50mi forward grad\n3hr delta" shape="box"]
# _5472397072 -> _5472396784 [label="<220.0"]
# _5472397072 -> _5472385024 [label="≥220.0"]
# _5472396784 [label="MXUPHL:5000-2000 m above ground\nestimated from probs, 25mi mean" shape="box"]
# _5472396784 -> _5472335904 [label="<2.2"]
# _5472396784 -> _5472366000 [label="≥2.2"]
# _5472335904 [label="RETOP:entire atmosphere\nprob >9144, 25mi mean\n+1hr" shape="box"]
# _5472335904 -> _5472264896 [label="<42.0"]
# _5472335904 -> _5472264976 [label="≥42.0"]
# _5472264896 [label="-0.035"]
# _5472264976 [label="-0.00045"]
# _5472366000 [label="-1.5e-5"]
# _5472385024 [label="+0.00075"]
# _5472397792 [label="sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m*(200+SBCIN)\n25mi mean\n3hr mean" shape="box"]
# _5472397792 -> _5472397504 [label="<3.4e6"]
# _5472397792 -> _5472385264 [label="≥3.4e6"]
# _5472397504 [label="TMP:925 mb\n25mi mean\n3hr min" shape="box"]
# _5472397504 -> _5472385104 [label="<300.0"]
# _5472397504 -> _5472385184 [label="≥300.0"]
# _5472385104 [label="-0.0066"]
# _5472385184 [label="+0.00066"]
# _5472385264 [label="+0.0058"]
# _5472536704 [label="1km reflectivity\nestimated from probs, 100mi mean\n+1hr" shape="box"]
# _5472536704 -> _5472486272 [label="<6.9"]
# _5472536704 -> _5472536560 [label="≥6.9"]
# _5472486272 [label="-0.0057"]
# _5472536560 [label="Wind700mb\n25mi mean\n+1hr" shape="box"]
# _5472536560 -> _5472486352 [label="<8.7"]
# _5472536560 -> _5472536416 [label="≥8.7"]
# _5472486352 [label="-0.00043"]
# _5472536416 [label="MUCAPE*700-500mbLapseRate*BWD0-6km\n\n3hr mean" shape="box"]
# _5472536416 -> _5472486432 [label="<8700.0"]
# _5472536416 -> _5472475680 [label="≥8700.0"]
# _5472486432 [label="-0.00034"]
# _5472475680 [label="CRAIN:surface\n100mi leftward grad\n-1hr" shape="box"]
# _5472475680 -> _5472486512 [label="<0.73"]
# _5472475680 -> _5472486592 [label="≥0.73"]
# _5472486512 [label="+0.034"]
# _5472486592 [label="-0.0001"]
# _5472539008 [label="sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m\n100mi mean" shape="box"]
# _5472539008 -> _5472537568 [label="<24000.0"]
# _5472539008 -> _5472538864 [label="≥24000.0"]
# _5472537568 [label="AbsVorticity850mb*10^5\n100mi mean\n3hr max" shape="box"]
# _5472537568 -> _5472537280 [label="<4.0"]
# _5472537568 -> _5472505776 [label="≥4.0"]
# _5472537280 [label="SOILW:0-0.1 m below ground\n100mi mean\n3hr max" shape="box"]
# _5472537280 -> _5472505616 [label="<0.41"]
# _5472537280 -> _5472505696 [label="≥0.41"]
# _5472505616 [label="-0.0031"]
# _5472505696 [label="+0.0015"]
# _5472505776 [label="+0.0062"]
# _5472538864 [label="Flight category\nprob >=4 <0, 100mi mean\n+1hr" shape="box"]
# _5472538864 -> _5472538576 [label="<98.0"]
# _5472538864 -> _5472506176 [label="≥98.0"]
# _5472538576 [label="WIND:850-300 mb\nprob <5, 100mi leftward grad" shape="box"]
# _5472538576 -> _5472505856 [label="<-14.0"]
# _5472538576 -> _5472538432 [label="≥-14.0"]
# _5472505856 [label="+0.0003"]
# _5472538432 [label="HGT:500 mb\n50mi leftward grad\n3hr min" shape="box"]
# _5472538432 -> _5472538144 [label="<-6.8"]
# _5472538432 -> _5472506096 [label="≥-6.8"]
# _5472538144 [label="CIN:180-0 mb above ground\n100mi mean\n3hr mean" shape="box"]
# _5472538144 -> _5472505936 [label="<-130.0"]
# _5472538144 -> _5472506016 [label="≥-130.0"]
# _5472505936 [label="-0.00014"]
# _5472506016 [label="+0.13"]
# _5472506096 [label="+0.00025"]
# _5472506176 [label="+0.0016"]
# }

# Pop it into an online renderer: https://dreampuf.github.io/GraphvizOnline/#strict%20digraph%20%7B%0A_5472539152%20%5Blabel%3D%22MXUPHL%3A5000-2000%20m%20above%20ground%5Cnestimated%20from%20probs%2C%2050mi%20mean%5Cn%2B1hr%22%20shape%3D%22box%22%5D%0A_5472539152%20-%3E%20_5472536848%20%5Blabel%3D%22%3C1.3%22%5D%0A_5472539152%20-%3E%20_5472539008%20%5Blabel%3D%22%E2%89%A51.3%22%5D%0A_5472536848%20%5Blabel%3D%22sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m*(200%2BSBCIN)%5Cn25mi%20mean%5Cn3hr%20mean%22%20shape%3D%22box%22%5D%0A_5472536848%20-%3E%20_5472474816%20%5Blabel%3D%22%3C8.8e6%22%5D%0A_5472536848%20-%3E%20_5472536704%20%5Blabel%3D%22%E2%89%A58.8e6%22%5D%0A_5472474816%20%5Blabel%3D%22MAXREF%3A1000%20m%20above%20ground%5Cnestimated%20from%20probs%2C%2025mi%20mean%5Cn3hr%20mean%22%20shape%3D%22box%22%5D%0A_5472474816%20-%3E%20_5472397072%20%5Blabel%3D%22%3C23.0%22%5D%0A_5472474816%20-%3E%20_5472397792%20%5Blabel%3D%22%E2%89%A523.0%22%5D%0A_5472397072%20%5Blabel%3D%22sqrt(SBCAPE)*BWD0-6km%5Cn50mi%20forward%20grad%5Cn3hr%20delta%22%20shape%3D%22box%22%5D%0A_5472397072%20-%3E%20_5472396784%20%5Blabel%3D%22%3C220.0%22%5D%0A_5472397072%20-%3E%20_5472385024%20%5Blabel%3D%22%E2%89%A5220.0%22%5D%0A_5472396784%20%5Blabel%3D%22MXUPHL%3A5000-2000%20m%20above%20ground%5Cnestimated%20from%20probs%2C%2025mi%20mean%22%20shape%3D%22box%22%5D%0A_5472396784%20-%3E%20_5472335904%20%5Blabel%3D%22%3C2.2%22%5D%0A_5472396784%20-%3E%20_5472366000%20%5Blabel%3D%22%E2%89%A52.2%22%5D%0A_5472335904%20%5Blabel%3D%22RETOP%3Aentire%20atmosphere%5Cnprob%20%3E9144%2C%2025mi%20mean%5Cn%2B1hr%22%20shape%3D%22box%22%5D%0A_5472335904%20-%3E%20_5472264896%20%5Blabel%3D%22%3C42.0%22%5D%0A_5472335904%20-%3E%20_5472264976%20%5Blabel%3D%22%E2%89%A542.0%22%5D%0A_5472264896%20%5Blabel%3D%22-0.035%22%5D%0A_5472264976%20%5Blabel%3D%22-0.00045%22%5D%0A_5472366000%20%5Blabel%3D%22-1.5e-5%22%5D%0A_5472385024%20%5Blabel%3D%220.00075%22%5D%0A_5472397792%20%5Blabel%3D%22sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m*(200%2BSBCIN)%5Cn25mi%20mean%5Cn3hr%20mean%22%20shape%3D%22box%22%5D%0A_5472397792%20-%3E%20_5472397504%20%5Blabel%3D%22%3C3.4e6%22%5D%0A_5472397792%20-%3E%20_5472385264%20%5Blabel%3D%22%E2%89%A53.4e6%22%5D%0A_5472397504%20%5Blabel%3D%22TMP%3A925%20mb%5Cn25mi%20mean%5Cn3hr%20min%22%20shape%3D%22box%22%5D%0A_5472397504%20-%3E%20_5472385104%20%5Blabel%3D%22%3C300.0%22%5D%0A_5472397504%20-%3E%20_5472385184%20%5Blabel%3D%22%E2%89%A5300.0%22%5D%0A_5472385104%20%5Blabel%3D%22-0.0066%22%5D%0A_5472385184%20%5Blabel%3D%220.00066%22%5D%0A_5472385264%20%5Blabel%3D%220.0058%22%5D%0A_5472536704%20%5Blabel%3D%221km%20reflectivity%5Cnestimated%20from%20probs%2C%20100mi%20mean%5Cn%2B1hr%22%20shape%3D%22box%22%5D%0A_5472536704%20-%3E%20_5472486272%20%5Blabel%3D%22%3C6.9%22%5D%0A_5472536704%20-%3E%20_5472536560%20%5Blabel%3D%22%E2%89%A56.9%22%5D%0A_5472486272%20%5Blabel%3D%22-0.0057%22%5D%0A_5472536560%20%5Blabel%3D%22Wind700mb%5Cn25mi%20mean%5Cn%2B1hr%22%20shape%3D%22box%22%5D%0A_5472536560%20-%3E%20_5472486352%20%5Blabel%3D%22%3C8.7%22%5D%0A_5472536560%20-%3E%20_5472536416%20%5Blabel%3D%22%E2%89%A58.7%22%5D%0A_5472486352%20%5Blabel%3D%22-0.00043%22%5D%0A_5472536416%20%5Blabel%3D%22MUCAPE*700-500mbLapseRate*BWD0-6km%5Cn%5Cn3hr%20mean%22%20shape%3D%22box%22%5D%0A_5472536416%20-%3E%20_5472486432%20%5Blabel%3D%22%3C8700.0%22%5D%0A_5472536416%20-%3E%20_5472475680%20%5Blabel%3D%22%E2%89%A58700.0%22%5D%0A_5472486432%20%5Blabel%3D%22-0.00034%22%5D%0A_5472475680%20%5Blabel%3D%22CRAIN%3Asurface%5Cn100mi%20leftward%20grad%5Cn-1hr%22%20shape%3D%22box%22%5D%0A_5472475680%20-%3E%20_5472486512%20%5Blabel%3D%22%3C0.73%22%5D%0A_5472475680%20-%3E%20_5472486592%20%5Blabel%3D%22%E2%89%A50.73%22%5D%0A_5472486512%20%5Blabel%3D%220.034%22%5D%0A_5472486592%20%5Blabel%3D%22-0.0001%22%5D%0A_5472539008%20%5Blabel%3D%22sqrt(SBCAPE)*BWD0-6km*HLCY3000-0m%5Cn100mi%20mean%22%20shape%3D%22box%22%5D%0A_5472539008%20-%3E%20_5472537568%20%5Blabel%3D%22%3C24000.0%22%5D%0A_5472539008%20-%3E%20_5472538864%20%5Blabel%3D%22%E2%89%A524000.0%22%5D%0A_5472537568%20%5Blabel%3D%22AbsVorticity850mb*10%5E5%5Cn100mi%20mean%5Cn3hr%20max%22%20shape%3D%22box%22%5D%0A_5472537568%20-%3E%20_5472537280%20%5Blabel%3D%22%3C4.0%22%5D%0A_5472537568%20-%3E%20_5472505776%20%5Blabel%3D%22%E2%89%A54.0%22%5D%0A_5472537280%20%5Blabel%3D%22SOILW%3A0-0.1%20m%20below%20ground%5Cn100mi%20mean%5Cn3hr%20max%22%20shape%3D%22box%22%5D%0A_5472537280%20-%3E%20_5472505616%20%5Blabel%3D%22%3C0.41%22%5D%0A_5472537280%20-%3E%20_5472505696%20%5Blabel%3D%22%E2%89%A50.41%22%5D%0A_5472505616%20%5Blabel%3D%22-0.0031%22%5D%0A_5472505696%20%5Blabel%3D%220.0015%22%5D%0A_5472505776%20%5Blabel%3D%220.0062%22%5D%0A_5472538864%20%5Blabel%3D%22Flight%20category%5Cnprob%20%3E%3D4%20%3C0%2C%20100mi%20mean%5Cn%2B1hr%22%20shape%3D%22box%22%5D%0A_5472538864%20-%3E%20_5472538576%20%5Blabel%3D%22%3C98.0%22%5D%0A_5472538864%20-%3E%20_5472506176%20%5Blabel%3D%22%E2%89%A598.0%22%5D%0A_5472538576%20%5Blabel%3D%22WIND%3A850-300%20mb%5Cnprob%20%3C5%2C%20100mi%20leftward%20grad%22%20shape%3D%22box%22%5D%0A_5472538576%20-%3E%20_5472505856%20%5Blabel%3D%22%3C-14.0%22%5D%0A_5472538576%20-%3E%20_5472538432%20%5Blabel%3D%22%E2%89%A5-14.0%22%5D%0A_5472505856%20%5Blabel%3D%220.0003%22%5D%0A_5472538432%20%5Blabel%3D%22HGT%3A500%20mb%5Cn50mi%20leftward%20grad%5Cn3hr%20min%22%20shape%3D%22box%22%5D%0A_5472538432%20-%3E%20_5472538144%20%5Blabel%3D%22%3C-6.8%22%5D%0A_5472538432%20-%3E%20_5472506096%20%5Blabel%3D%22%E2%89%A5-6.8%22%5D%0A_5472538144%20%5Blabel%3D%22CIN%3A180-0%20mb%20above%20ground%5Cn100mi%20mean%5Cn3hr%20mean%22%20shape%3D%22box%22%5D%0A_5472538144%20-%3E%20_5472505936%20%5Blabel%3D%22%3C-130.0%22%5D%0A_5472538144%20-%3E%20_5472506016%20%5Blabel%3D%22%E2%89%A5-130.0%22%5D%0A_5472505936%20%5Blabel%3D%22-0.00014%22%5D%0A_5472506016%20%5Blabel%3D%220.13%22%5D%0A_5472506096%20%5Blabel%3D%220.00025%22%5D%0A_5472506176%20%5Blabel%3D%220.0016%22%5D%0A%7D


import MemoryConstrainedTreeBoosting

# Most 2020 models seem not to have the repeated split problem:
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2021-05-25T11.12.14.168/416_trees_loss_0.0009960941.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2021-05-20T16.41.52.952/164_trees_loss_0.0010514505.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2021-05-26T15.55.17.39/305_trees_loss_0.0010976213.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12-23_Dates.DateTime(\"2021-04-22T01.22.58.76\")/175_trees_loss_0.0012076722.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2_2021-05-21T22.37.26.544/463_trees_loss_0.0008549077.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f6_2021-05-27T17.12.43.406/465_trees_loss_0.00092996453.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12_2021-05-10T20.00.43.881/471_trees_loss_0.0010017317.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../hrrr_late_aug_2016_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f17_2021-04-30T10.49.47.495/341_trees_loss_0.0010138382.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../rap_march_2014_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f6_2021-06-04T14.25.56.451/297_trees_loss_0.0009202203.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../rap_march_2014_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f12_2021-06-07T05.23.57.904/477_trees_loss_0.0009632701.model");


# Some 2020 models do:
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_Dates.DateTime(\"2021-04-20T04.17.36.114\")/173_trees_loss_0.0011276418.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../sref_mid_2018_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f21-38_2021-04-25T00.37.19.274/198_trees_loss_0.001240725.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../rap_march_2014_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f2_2021-06-11T14.38.30.5/445_trees_loss_0.00081564416.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("../rap_march_2014_forward/gbdt_3hr_window_3hr_min_mean_max_delta_f17_2021-06-03T15.44.45.541/243_trees_loss_0.0009980382.model");




# Most 2021 models do:
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_tornado/391_trees_loss_0.0010360148.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_tornado/317_trees_loss_0.001094988.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_wind/754_trees_loss_0.0062351814.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_wind/581_trees_loss_0.00660574.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_wind/414_trees_loss_0.006970079.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-16T10.56.27.856_hail/460_trees_loss_0.003063131.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_hail/560_trees_loss_0.003272809.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_hail/485_trees_loss_0.0034841662.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-18T17.13.12.938_sig_tornado/368_trees_loss_0.00015682736.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-21T23.06.01.396_sig_wind/269_trees_loss_0.00094322924.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-21T23.05.58.511_sig_wind/176_trees_loss_0.0009875718.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-22T04.03.41.799_sig_wind/184_trees_loss_0.0010262702.model");

# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f2-13_2022-04-21T23.06.01.396_sig_hail/274_trees_loss_0.00049601856.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-21T23.05.58.511_sig_hail/358_trees_loss_0.00052869593.model");


# Some 2021 models do not:
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-16T14.36.46.241_tornado/308_trees_loss_0.0011393429.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f13-24_2022-04-19T11.41.57.211_sig_tornado/158_trees_loss_0.0001665851.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-21T05.00.10.408_sig_tornado/310_trees_loss_0.00017311782.model");
# bin_splits, trees = MemoryConstrainedTreeBoosting.load("gbdt_3hr_window_3hr_min_mean_max_delta_f24-35_2022-04-22T04.03.41.799_sig_hail/485_trees_loss_0.0005760971.model");





length(trees)

# feature_names = readlines("../href_mid_2018_forward/features2021models.txt");

feature_i_to_name(i) = "feat$i" # feature_names[i]

function repeated_splits(node, higher_split_nodes=[])
  if isa(node, MemoryConstrainedTreeBoosting.Leaf)
    []
  else
    is_repeated = any(higher_split_nodes) do higher_node
      node.feature_i == higher_node.feature_i && node.split_i == higher_node.split_i
    end

    nodes_through_here = [higher_split_nodes; node]

    [
      (is_repeated ? [node] : []);
      repeated_splits(node.left, nodes_through_here);
      repeated_splits(node.right, nodes_through_here);
    ]
  end
end

function node_str(node, bin_splits, feature_i_to_name)
  if isa(node, MemoryConstrainedTreeBoosting.Leaf)
    "leaf"
  else
    "$(feature_i_to_name(node.feature_i))@$(bin_splits[node.feature_i][node.split_i])"
  end
end

for tree_i in 1:length(trees)
  tree = trees[tree_i]
  repeats = repeated_splits(tree)
  if repeats != []
    repeat_names = map(node -> node_str(node, bin_splits, feature_i_to_name), repeats)
    println("Tree $tree_i:\t$(join(repeat_names, ", "))")
  end
end
