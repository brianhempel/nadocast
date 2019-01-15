remake_common_layers:
	rm common_layers.txt; make common_layers.txt

common_layers.txt:
	ruby ../../find_common_layers.rb /Volumes/Tornadoes/rap "path[/\\A\\d\\d\\d\\d\\d\\d\\//].to_i >= 201403" > common_layers.txt