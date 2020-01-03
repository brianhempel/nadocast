remake_common_layers:
	rm common_layers.txt; make common_layers.txt

common_layers.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/HRRR_1/hrrr > common_layers_hrrr_1.txt
	ruby ../../lib/find_common_layers.rb /Volumes/HRRR_2/hrrr > common_layers_hrrr_2.txt
	diff common_layers_hrrr_1.txt common_layers_hrrr_2.txt
	# Make sure they match and then manually merge into common_layers.txt

train_+2_gradient_boosted_decision_trees:
	FORECAST_HOUR=2 make train_gradient_boosted_decision_trees

train_+6_gradient_boosted_decision_trees:
	FORECAST_HOUR=6 make train_gradient_boosted_decision_trees

train_+12_gradient_boosted_decision_trees:
	FORECAST_HOUR=12 make train_gradient_boosted_decision_trees

train_+17_gradient_boosted_decision_trees:
	FORECAST_HOUR=17 make train_gradient_boosted_decision_trees

train_gradient_boosted_decision_trees:
	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
