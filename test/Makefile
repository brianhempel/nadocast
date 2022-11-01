default:
	cat Makefile

julia:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=..

confidence_shapes.txt: test_0z.csv BootstrapConfidenceIntervals.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. BootstrapConfidenceIntervals.jl > confidence_shapes.txt

index.html: index.html.erb maps/*/*.png
	erb index.html.erb > index.html

plots.html: plots.html.erb plots/*.png
	erb plots.html.erb > plots.html

ablations.html: ablations.html.erb maps/*/*ablation*.png
	erb ablations.html.erb > ablations.html

.PHONY: day2_vs_day1.html
day2_vs_day1.html:
	erb day2_vs_day1.html.erb > day2_vs_day1.html

publish: index.html
	cp -r maps maps_2021 plots plots_2021 *.html ../../test.nadocast.com/
	cd ../../test.nadocast.com && git add .; git commit -m "Update"; git pull --rebase && git push

notebook:
	jupyter notebook
