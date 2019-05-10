remake_common_layers:
	rm common_layers.txt; make common_layers.txt

common_layers.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/HRRR_1/hrrr > common_layers.txt

train_+6_gradient_boosted_decision_trees:
	FORECAST_HOUR=6 make train_gradient_boosted_decision_trees

train_+12_gradient_boosted_decision_trees:
	FORECAST_HOUR=12 make train_gradient_boosted_decision_trees

train_gradient_boosted_decision_trees:
	ulimit -n 8192; JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
