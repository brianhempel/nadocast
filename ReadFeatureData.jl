const HEADERS          = split(strip(String(read("headers.txt"))), "\n")
const FEATURE_HEADERS  = HEADERS[11:length(HEADERS)] # First 10 "features" are actually not.
const VALUES_PER_POINT = length(HEADERS)
const FEATURE_COUNT    = length(FEATURE_HEADERS)


function open_data_file(path, width=FEATURE_COUNT)
  data_byte_size = filesize(path)
  data_float_size = div(data_byte_size, 4)
  data_point_count = div(data_float_size, width)

  data_file = open(path)
  data = Mmap.mmap(data_file, Array{Float32,2}, (width, data_point_count))

  (data, data_file, data_point_count)
end

function read_data_file(path, width=VALUES_PER_POINT)
  data_byte_size = filesize(path)
  data_float_size = div(data_byte_size, 4)
  data_point_count = div(data_float_size, width)

  read(path, Float32, (width, data_point_count)) :: Array{Float32, 2}
end
