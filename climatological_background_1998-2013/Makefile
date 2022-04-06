default:
	cat makefile

make_climatological_background:
	# Produces several .float16.bin spatial output files.
	# Probability by hour, and by month, are produced on standard output for copy/paste.
	# (Actually they're also in the file directly; also in the .numbers spreadsheets.)
	JULIA_NUM_THREADS=${CORE_COUNT} time julia --project=.. MakeClimatologicalBackground.jl
