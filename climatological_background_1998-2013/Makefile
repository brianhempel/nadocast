default:
	cat makefile

make_climatological_background:
	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=.. MakeClimatologicalBackground.jl
