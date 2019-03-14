module GBDTFeatureImportance

import MemoryConstrainedTreeBoosting

function print_feature_importances(gbdt_path, feature_i_to_name)
  (bin_splits, trees) = MemoryConstrainedTreeBoosting.load(gbdt_path)

  println("$(length(trees)) trees")

  println("Feature importance by appearance count:")
  for (feature_i, appearance_count) in MemoryConstrainedTreeBoosting.feature_importance_by_appearance_count(trees)

    feature_name = feature_i_to_name(feature_i)

    println("$appearance_count\t$feature_i\t$feature_name")
  end
  println()

  println("Feature importance by absolute Δscores in descendant leaves:")
  for (feature_i, abs_Δscore) in MemoryConstrainedTreeBoosting.feature_importance_by_absolute_delta_score(trees)

    feature_name = feature_i_to_name(feature_i)

    println("$(abs_Δscore)\t$feature_i\t$feature_name")
  end
  println()
end

end # module GBDTFeatureImportance
