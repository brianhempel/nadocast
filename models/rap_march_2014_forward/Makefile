remake_common_layers:
	rm common_layers.txt; make common_layers.txt

common_layers.txt:
	# Don't include x25 mb and x75 mb layers, just x00 mb and x50 mb.
	ruby ../../lib/find_common_layers.rb /Volumes/RAP_1/rap "path[/\\d\\d\\d\\d\\d\\d\\d\\d\\//].to_i >= 20140226" "!(desc[/\\A(\d+) mb\\z/, 1] && desc[/\\A(\d+) mb\\z/, 1].to_i % 50 != 0)" > common_layers.txt

profile_feature_engineering:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=../.. ProfileFeatureEngineering.jl

train_+12_gradient_boosted_decision_trees:
	FORECAST_HOUR=12 make train_gradient_boosted_decision_trees

train_gradient_boosted_decision_trees:
	ulimit -n 8192; JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
