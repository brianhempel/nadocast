default:
	cat Makefile

train_gbdt_stack_for_href_newer_than_sref:
	HREF_NEWER_THAN_SREF=true make train_gradient_boosted_decision_trees

train_gbdt_stack_for_sref_newer_than_href:
	HREF_NEWER_THAN_SREF=false make train_gradient_boosted_decision_trees

train_gradient_boosted_decision_trees:
	ulimit -n 8192; JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
