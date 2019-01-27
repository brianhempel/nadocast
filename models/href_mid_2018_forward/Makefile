remake_common_layers:
	rm common_layers_mean.txt common_layers_prob.txt; make common_layers_mean.txt common_layers_prob.txt

common_layers_mean.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/href "path =~ /_mean_/" > common_layers_mean.txt

common_layers_prob.txt:
	ruby ../../lib/find_common_layers.rb /Volumes/SREF_HREF_1/href "path =~ /_prob_/" > common_layers_prob.txt
