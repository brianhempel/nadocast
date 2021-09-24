default:
	cat Makefile

julia:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=..

confidence_shapes.txt: test_0z.csv BootstrapConfidenceIntervals.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. BootstrapConfidenceIntervals.jl > confidence_shapes.txt
