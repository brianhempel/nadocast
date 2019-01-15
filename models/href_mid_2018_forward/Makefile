remake_common_layers:
	rm common_layers_mean.txt common_layers_prob.txt; make common_layers_mean.txt common_layers_prob.txt

common_layers_mean.txt:
	ruby ../../find_common_layers.rb /Volumes/Tornadoes/href "path =~ /_mean_/" > common_layers_mean.txt

common_layers_prob.txt:
	ruby ../../find_common_layers.rb /Volumes/Tornadoes/href "path =~ /_prob_/" > common_layers_prob.txt
