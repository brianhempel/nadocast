# julia --project=../.. GBDTFeatureImportance.jl

import MagicTreeBoosting

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts

push!(LOAD_PATH, @__DIR__)
import SREF
import FeatureEngineering


# gbdt_path = "gbdt_2019-01-26T17:22:12.466/epoch_71_356_trees_loss_0.004774756557004337.model"
gbdt_path = "gbdt_2019-01-28T09:31:28.654/epoch_42_211_trees_loss_0.004799674883322649.model"

(bin_splits, trees) = MagicTreeBoosting.load(gbdt_path)

# println("$(length(trees)) trees")

inventory = Forecasts.inventory(SREF.example_forecast())


println("Feature importance by appearance count:")
for (feature_i, appearance_count) in MagicTreeBoosting.feature_importance_by_appearance_count(trees)

  feature_name = FeatureEngineering.feature_i_to_name(inventory, feature_i)

  println("$appearance_count\t$feature_i\t$feature_name")
end
println()

println("Feature importance by absolute Δscores in descendant leaves:")
for (feature_i, abs_Δscore) in MagicTreeBoosting.feature_importance_by_absolute_delta_score(trees)

  feature_name = FeatureEngineering.feature_i_to_name(inventory, feature_i)

  println("$(abs_Δscore)\t$feature_i\t$feature_name")
end
println()