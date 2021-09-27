default:
	cat Makefile

julia:
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=..

confidence_shapes.txt: test_0z.csv BootstrapConfidenceIntervals.jl
	JULIA_NUM_THREADS=${CORE_COUNT} julia --project=.. BootstrapConfidenceIntervals.jl > confidence_shapes.txt

index.html: index.html.erb
	erb index.html.erb > index.html

publish: index.html
	cp -r maps *.html ../../test.nadocast.com/
	cd ../../test.nadocast.com && git add .; git commit -m "Update"; git pull --rebase && git push