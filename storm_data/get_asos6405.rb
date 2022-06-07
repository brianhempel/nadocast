# https://data.eol.ucar.edu/datafile/nph-get/113.029/td6405.pdf

# ftp ftp@ftp.ncdc.noaa.gov
# brianhempel@ou.edu
# cd /pub/data/asos-onemin/6405-2020
# prompt
# mget *.dat

require "fileutils"
require "csv"

start_year, start_month = (ENV["START_YEAR_MONTH"] || "2000-1").split("-").map(&:to_i)

here                 = File.expand_path("..", __FILE__)
process_script       = File.expand_path("../process_asos6405.rb", __FILE__)
out_path             = File.expand_path("../gusts.csv", __FILE__)
good_row_counts_path = File.expand_path("../good_row_counts.csv", __FILE__)

if !ENV["START_YEAR_MONTH"]
  # Initialize new gusts.csv and good_row_counts.csv
  # These need to match process_asos6405.rb
  File.write(out_path, %w[
    time_str
    time_seconds
    wban_id
    name
    state
    county
    knots
    gust_knots
    lat
    lon
  ].to_csv)

  File.write(good_row_counts_path, %w[
    convective_date
    convective_day_index
    wban_id
    lat
    lon
    good_row_count
  ].to_csv)
end

FileUtils.cd here

# This doesn't have as many stations
# system("curl https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv > isd-history.csv")

# mshr_enhanced.txt is 15mb zipped, 500mb unzipped...lol...fixed-width columns filled with spaces
# Much larger database of stations than isd-history.csv above
system("curl https://www.ncei.noaa.gov/access/homr/file/mshr_enhanced.txt.zip > mshr_enhanced.txt.zip")
system("ruby mshr_enhanced_to_csv.rb > mshr_enhanced_wban_only.csv")

def wait_for_prompt(ftp)
  buf = ""
  loop do
    buf = ftp.readpartial(1_000_000)
    print buf
    loop do
      sleep(0.5) # Data might still be coming
      break unless more_buf = (fswatch.read_nonblock(1_000_000) rescue nil)
      buf << more_buf
      print more_buf
      STDOUT.flush
    end
    break if buf.end_with?("ftp> ")
  end
end

def ftp_do(ftp, command)
  ftp.puts command # "password"
  puts command
  wait_for_prompt(ftp)
end

puts "What's your email address?"
email = gets.strip

exit(1) unless email =~ /\S+@\S+\.\S+/

process_thread = nil
(2000..2022).each do |year|
  (1..12).each do |month|
    next if ([year, month] <=> [start_year, start_month]) < 0

    puts "#{year}-#{month}"

    FileUtils.cd here

    month_dir = "asos/6405-#{year}#{month}"
    FileUtils.mkdir_p month_dir
    FileUtils.cd month_dir

    IO.popen("ftp --no-prompt --prompt -n ftp.ncdc.noaa.gov", "r+") do |ftp|
      wait_for_prompt(ftp)
      ftp_do(ftp, "user ftp #{email}") # password is email address
      ftp_do(ftp, "cd /pub/data/asos-onemin/6405-#{year}")
      ftp_do(ftp, "mget *%02d.dat" % month)
      ftp.puts "quit"
      puts "quit"
    end
    process_thread.join if process_thread

    # Generate CSV gust lines from all files, sort them, and write them to gusts.csv
    # But do it while we download more data
    dat_glob = File.join(here, month_dir, "*.dat")
    process_thread = Thread.new do
      cmd = "GOOD_ROW_COUNTS_PATH=#{good_row_counts_path} ruby #{process_script} #{dat_glob}"
      puts cmd
      buf = `#{cmd}`.lines.sort.join
      File.open(out_path, "a") { |out| out.print(buf) }
      system("rm #{dat_glob}")
    end
  end
end

process_thread.join if process_thread
