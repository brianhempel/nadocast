default:
	cat Makefile

julia:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=../..

train_gradient_boosted_decision_trees:
	ulimit -n 8192; JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
