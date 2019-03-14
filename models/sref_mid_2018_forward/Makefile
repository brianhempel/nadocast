default:
	cat makefile

julia:
	julia --project=../..

remake_common_layers:
	rm common_layers_mean.txt common_layers_prob.txt; make common_layers_mean.txt common_layers_prob.txt

common_layers_mean.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/sref "path =~ /_mean_/" > common_layers_mean.txt

common_layers_prob.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/sref "path =~ /_prob_/" > common_layers_prob.txt

profile_feature_engineering:
	julia --project=../.. ProfileFeatureEngineering.jl

train_logistic:
	# Some weirdo bug in OpenBLAS "OpenBLAS : Program will terminate because you tried to start too many threads."
	# https://github.com/xianyi/OpenBLAS/issues/1735
	# All our slowness is loading time, not BLAS.
	OPENBLAS_NUM_THREADS=1 julia --project=../.. TrainLogistic.jl

train_double_logistic:
	# Some weirdo bug in OpenBLAS "OpenBLAS : Program will terminate because you tried to start too many threads."
	# https://github.com/xianyi/OpenBLAS/issues/1735
	# All our slowness is loading time, not BLAS.
	OPENBLAS_NUM_THREADS=1 julia --project=../.. TrainDoubleLogistic.jl

train_gradient_boosted_decision_trees:
	# time julia --project=../.. TrainGradientBoostedDecisionTrees.jl
	JULIA_NUM_THREADS=4 time julia --project=../.. TrainGradientBoostedDecisionTrees.jl

train_elastic_nets:
	JULIA_NUM_THREADS=4 time julia --project=../.. TrainElasticNets.jl
