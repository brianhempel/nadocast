default:
	cat Makefile

train_+1_gradient_boosted_decision_trees:
	FORECAST_HOUR=1 make train_gradient_boosted_decision_trees

train_+5_gradient_boosted_decision_trees:
	FORECAST_HOUR=5 make train_gradient_boosted_decision_trees

train_+11_gradient_boosted_decision_trees:
	FORECAST_HOUR=11 make train_gradient_boosted_decision_trees

train_+16_gradient_boosted_decision_trees:
	FORECAST_HOUR=16 make train_gradient_boosted_decision_trees

train_gradient_boosted_decision_trees:
	ulimit -n 8192; JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
