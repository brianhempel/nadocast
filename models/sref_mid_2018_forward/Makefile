default:
	cat makefile

julia:
	julia --project=../..

remake_common_layers:
	rm common_layers_mean.txt common_layers_prob.txt; make common_layers_mean.txt common_layers_prob.txt

common_layers_mean.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/Tornadoes/sref "path =~ /_mean_/" > common_layers_mean.txt

common_layers_prob.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/Tornadoes/sref "path =~ /_prob_/" > common_layers_prob.txt

normalizing_factors.txt:
	julia --project=../.. -e 'push!(LOAD_PATH, @__DIR__); import SREF; push!(LOAD_PATH, (@__DIR__) * "/../shared"); import MakeNormalizingFactors; using DelimitedFiles; writedlm("normalizing_factors.txt", MakeNormalizingFactors.calculate_normalizing_factors(SREF.forecasts()))'

train_logistic:
	# Some weirdo bug in OpenBLAS "OpenBLAS : Program will terminate because you tried to start too many threads."
	# https://github.com/xianyi/OpenBLAS/issues/1735
	# All our slowness is loading time, not BLAS.
	OPENBLAS_NUM_THREADS=1 julia --project=../.. TrainLogistic.jl

train_gradient_boosted_decision_trees:
	julia --project=../.. TrainGradientBoostedDecisionTrees.jl
