module Cache

import Serialization
using TranscodingStreams, CodecZstd


### Caching in the local computation_cache folder

function cached(f, item_key_parts, function_name, more_keying...)
  cached = Cache.get_cached(item_key_parts, function_name, more_keying...)
  if !isnothing(cached)
    return cached
  end

  result = f()

  Cache.cache_and_return(result, item_key_parts, function_name, more_keying...)
end

function cache_path(item_key_parts, function_name, more_keying...)
  item_key_str    = replace(replace(join(item_key_parts, "/"), r"\.gri?b2$" => ""), ":" => "|")
  more_keying_str = isempty(more_keying) ? "" : string(hash(more_keying))
  (@__DIR__) * "/computation_cache/" * item_key_str * "/" * function_name * "_" * more_keying_str
end

function clear_all(item_key_parts)
  item_key_str = replace(replace(join(item_key_parts, "/"), r"\.gri?b2$" => ""), ":" => "|")
  dir_path = (@__DIR__) * "/computation_cache/" * item_key_str
  rm(dir_path; recursive = true)
end

function get_cached(item_key_parts, function_name, more_keying...)
  path = cache_path(item_key_parts, function_name, more_keying...)
  if isfile(path)
    open(CodecZstd.ZstdDecompressorStream, path, "r") do stream
      Serialization.deserialize(stream)
    end
  else
    nothing
  end
end

function cache_and_return(item, item_key_parts, function_name, more_keying...)
  path = cache_path(item_key_parts, function_name, more_keying...)

  mkpath(dirname(path))

  open(CodecZstd.ZstdCompressorStream, path, "w") do stream
    Serialization.serialize(stream, item)
  end

  item
end

end # module Cache
