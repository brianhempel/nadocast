default:
	cat Makefile

julia:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=..

confidence_shapes.txt: test_0z.csv BootstrapConfidenceIntervals.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. BootstrapConfidenceIntervals.jl > confidence_shapes.txt

index.html: index.html.erb maps/*/*.png
	erb index.html.erb > index.html

.PHONY: day2_vs_day1.html
day2_vs_day1.html:
	erb day2_vs_day1.html.erb > day2_vs_day1.html

publish: index.html
	cp -r maps *.html 2021_models_objective_plots.pdf ../../test.nadocast.com/2021_models/
	cd ../../test.nadocast.com && git add .; git commit -m "Update"; git pull --rebase && git push
