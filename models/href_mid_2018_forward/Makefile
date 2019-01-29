remake_common_layers:
	rm common_layers_mean.txt common_layers_prob.txt; make common_layers_mean.txt common_layers_prob.txt

common_layers_mean.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/href "path =~ /_mean_/" > common_layers_mean.txt

common_layers_prob.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/href "path =~ /_prob_/" > common_layers_prob.txt

profile_feature_engineering:
	julia --project=../.. ProfileFeatureEngineering.jl

train_gradient_boosted_decision_trees:
	JULIA_NUM_THREADS=4 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
