#!/usr/bin/env ruby

# For downloading from the long term storage request system at https://www.ncdc.noaa.gov/has/HAS.FileAppRouter?datasetname=RAP130&subqueryby=STATION&applname=&outdest=FILE
# You still have to request the data manually.

# Usage: ruby get_rap_archived.rb https://www1.ncdc.noaa.gov/pub/has/model/HAS011081869/ [thread_count] [base_directory]

base_url = ARGV[0] || raise("Usage: ruby get_rap_archived.rb https://www1.ncdc.noaa.gov/pub/has/model/HAS011081869/ [thread_count] [base_directory]")

THREAD_COUNT = Integer(ARGV[1] || "6")
BASE_DIRECTORY = ARGV[2] || "/Volumes/RAP_1/rap"
MIN_FILE_BYTES = 10_000_000
FORECAST_HOURS = [1,2,3,5,6,7,11,12,13,16,17,18]
FILE_NAME_REGEXP = /\Arap_130_\w+_(#{FORECAST_HOURS.map {|i| "%03d" % i}.join("|")}).grb2/

html = `curl -s #{base_url}`

urls_to_get = html.scan(/\w+\.g2\.tar\b/).uniq

if urls_to_get.empty?
  puts html
  exit 1
end

threads = THREAD_COUNT.times.map do
  Thread.new do
    while file_url = urls_to_get.shift
      file_url =~ /_130_(\d\d\d\d)(\d\d)(\d\d)(\d\d)\./
      year_str, month_str, day_str, run_hour_str = $1, $2, $3, $4

      # Gotta walk the tar file ourselves.
      # https://en.wikipedia.org/wiki/Tar_(computing)#File_format

      tar_url = base_url + file_url

      at_byte = 0

      # There's extra weirdo "./PaxHeaders.6294/rap_130_20170101_0400_000.grb2" files that we want to skip.
      loop do
        header = `curl -s -H"Range: bytes=#{at_byte}-#{at_byte + 512 - 1}" #{tar_url}`
        break unless $?.success?
        file_name = header[0...100][/\A[^\x0]+/]
        break unless file_name
        # puts file_name
        file_size = header[124...135].to_i(8) # Wikipedia says 12 bytes, but last byte appears to always be a null

        at_byte += 512
        if file_name =~ FILE_NAME_REGEXP
          directory = "#{BASE_DIRECTORY}/#{year_str}#{month_str}/#{year_str}#{month_str}#{day_str}"
          system("mkdir -p #{directory} 2> /dev/null")
          path = "#{directory}/#{file_name}"
          if (File.size(path) rescue 0) < MIN_FILE_BYTES
            puts "#{file_name} -> #{path}"
            data = `curl -f -s --show-error -H"Range: bytes=#{at_byte}-#{at_byte + file_size - 1}" #{tar_url}`
            if data.bytesize == file_size
              File.write(path, data)
            else
              STDERR.puts "Asked for #{file_size} bytes of #{file_name} but only got #{data.size}!!! Not saved."
            end
          end
        end
        at_byte += (file_size / 512.0).ceil * 512
      end
    end
  end
end

threads.each(&:join)




