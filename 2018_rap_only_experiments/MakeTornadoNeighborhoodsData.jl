# Usage:
#
# $ julia MakeTornadoNeighborhoodsData.jl model/train.txt
#
# Outputs model/neighborhoods_train.binfeatures model/neighborhoods_train.binlabels model/neighborhoods_train.binweights

if length(ARGS) != 1
  error("Need to provide an argument: a file that contains a list of grib2 pairs to consolidate")
end

rows_to_process_at_a_time = 5

rows_to_process = map(split, split(strip(String(read(ARGS[1]))), "\n"))

out_path_base   = replace(ARGS[1], ".txt", "")
out_path_dir    = dirname(ARGS[1])

println("Consolidating $(length(rows_to_process)) feature snapshots...")

in_flight = []

headers = split(strip(String(read("$out_path_dir/headers.txt"))), "\n")
VALUES_PER_POINT = length(headers)

out_path     = "$out_path_base.binfeatures"
labels_path  = "$out_path_base.binlabels"  # Float32's
weights_path = "$out_path_base.binweights" # Float32's

out_file     = open(out_path, "a")
labels_file  = open(labels_path, "a")
weights_file = open(weights_path, "a")

while length(rows_to_process) > 0

  for _ = 1:rows_to_process_at_a_time
    if length(rows_to_process) > 0
      prior_hour_forecast_file, file = pop!(rows_to_process)
      println((prior_hour_forecast_file, file))
      # println(files_to_process)
      push!(in_flight, open(`julia $out_path_dir/AdjacentGribsToFeatures.jl $prior_hour_forecast_file $file --250mi_tornado_neighborhoods`))
      # println(in_flight)
    end
  end

  while length(in_flight) > 0
    data_stream, proc = pop!(in_flight)
    # success = wait(proc)
    reader_log_out = String(read(data_stream))
    # print(reader_log_out)
    tmp_data_path = last(split(strip(reader_log_out),"\n")) # last line is the tmp file with the data

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
          nan_is = find(isnan, data_chunk)
          count = length(nan_is)
          first_nan_i = nan_is[1]
          header_i = mod(first_nan_i-1,VALUES_PER_POINT) + 1
          error("contains $count nans; some in $(headers[header_i])")
        end
        if any(isinf, data_chunk)
          inf_is = find(isnan, data_chunk)
          count = length(inf_is)
          first_inf_i = inf_is[1]
          header_i = mod(first_inf_i-1,VALUES_PER_POINT) + 1
          error("contains $count infs; some in $(headers[header_i])")
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
