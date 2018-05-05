# Usage:
#
# $ julia MakeTornadoNeighborhoodsData.jl train.txt
#
# Outputs neighborhoods_train.binfeatures neighborhoods_train.binlabels neighborhoods_train.binweights

if length(ARGS) != 1
  error("Need to provide an argument: a file that contains a list of grib2 to consolidate")
end

files_to_process_at_a_time = 5

files_to_process = split(strip(String(read(ARGS[1]))), "\n")
out_path_base    = replace(ARGS[1], ".txt", "")

println("Consolidating $(length(files_to_process)) files...")

in_flight = []

headers = split(strip(String(read("headers.txt"))), "\n")
VALUES_PER_POINT = length(headers)

out_path     = "$out_path_base.binfeatures"
labels_path  = "$out_path_base.binlabels"  # Float32's
weights_path = "$out_path_base.binweights" # Float32's

out_file     = open(out_path, "a")
labels_file  = open(labels_path, "a")
weights_file = open(weights_path, "a")

while length(files_to_process) > 0

  for _ = 1:files_to_process_at_a_time
    if length(files_to_process) > 0
      file = pop!(files_to_process)
      println(file)
      # println(files_to_process)
      push!(in_flight, open(`julia read_grib.jl $file --250mi_tornado_neighborhoods`))
      # println(in_flight)
    end
  end

  while length(in_flight) > 0
    data_stream, proc = pop!(in_flight)
    # success = wait(proc)
    tmp_data_path = last(split(strip(String(read(data_stream))),"\n")) # last line is the tmp file with the data

    # If reading grib file okay...
    if endswith(tmp_data_path, ".bindata")
      # println(tmp_data_path)
      data_chunk_byte_size = filesize(tmp_data_path)
      data_chunk_float_size = div(data_chunk_byte_size, 4)
      data_chunk_point_count = div(data_chunk_float_size, VALUES_PER_POINT)

      data_chunk = read(open(tmp_data_path), Float32, (VALUES_PER_POINT, data_chunk_point_count))

      if length(data_chunk) > 0
        open(`rm $tmp_data_path`)

        println(size(data_chunk,2))
        if any(isnan, data_chunk)
          error("contains nans")
        end
        if any(isinf, data_chunk)
          error("contains infs")
        end
        # Data has one column per point, so with Julia's column-major ordering
        # we can simply append the data to add more columns (points)
        write(out_file, @view data_chunk[11:VALUES_PER_POINT,:])
        write(labels_file, data_chunk[9,:])
        write(weights_file, data_chunk[10,:])
      else
        println(0)
      end
    end
    # println(in_flight)
  end

  flush(out_file)
end

close(out_file)
