#!/usr/bin/env ruby

# Usage:
#
# $ ruby order_and_get_rap_archived.rb 2014-2-1 2018-12-31
#
# Or, if there are still rap_archive_dates_to_order or outstanding_orders
#
# $ ruby order_and_get_rap_archived.rb

# Automates and downloading from the long term storage request system at https://www.ncdc.noaa.gov/has/HAS.FileAppRouter?datasetname=RAP130&subqueryby=STATION&applname=&outdest=FILE
#
# Relys on get_rap_archived.rb to download once the order is processed.

require 'net/http'
require 'date'
require 'fileutils'


MIN_FILE_BYTES = 10_000_000
FORECAST_HOURS = [1,2,3,5,6,7,11,12,13,16,17,18]
RUN_HOURS      = (0..23).to_a
FILE_NAME_REGEXP = /\Arap_130_\w+_(#{FORECAST_HOURS.map {|i| "%03d" % i}.join("|")}).grb2/

MAX_SIMULTANEOUS_ORDERS = 2
DAYS_PER_ORDER          = 15

class PersistentHash
  include Enumerable

  def initialize(name)
    @dir = File.expand_path("../#{name}", __FILE__)
    system("mkdir '#{@dir}' 2> /dev/null")
  end

  def key_path(key)
    "#{@dir}/#{key}"
  end

  def count
    keys.count
  end

  def keys
    keys_iter.to_a
  end

  def keys_iter
    Dir.each_child(@dir)
  end

  def [](key)
    path = key_path(key)
    File.read(path) if File.exists?(path)
  end

  def []=(key, value)
    File.write(key_path(key), value)
  end

  def delete(key)
    value = self[key]
    path  = key_path(key)
    FileUtils.rm(path) if File.exists?(path)
    value
  end

  def each
    keys_iter.each do |file_name|
      yield(file_name, self[file_name])
    end
  end
end

# "2019-1-10" to date
def str_to_date(str)
  year, month, day = str.split("-").map(&:to_i)
  Date.new(year, month, day)
end

dates_to_order     = PersistentHash.new("rap_archive_dates_to_order")
outstanding_orders = PersistentHash.new("rap_archive_outstanding_orders")
orders_downloading = PersistentHash.new("rap_archive_orders_downloading")

if dates_to_order.count == 0 && ARGV.size == 2
  start_date = str_to_date(ARGV[0])
  end_date   = str_to_date(ARGV[1])

  (start_date..end_date).each do |date|
    dates_to_order[date.to_s] = ""
  end
end

def make_order(dates_to_order, outstanding_orders)
  all_dates_to_order = dates_to_order.keys.map { |date_str|  str_to_date(date_str) }.sort

  order_start_date = all_dates_to_order.first
  order_end_date   = all_dates_to_order.select { |date| date < order_start_date + DAYS_PER_ORDER }.last

  order_date_range = order_start_date..order_end_date

  order_dates = all_dates_to_order.select { |date| order_date_range.cover?(date) }

  uri = URI('https://www.ncdc.noaa.gov/has/HAS.FileSelect')
  request = Net::HTTP::Post.new(uri)
  request.set_form_data(
    'satdisptype'   => "N/A",
    'stations'      => RUN_HOURS.map { |run_hour| "%02d" % run_hour },
    'station_lst'   => "",
    'typeofdata'    => "MODEL",
    'dtypelist'     => "",
    'begdatestring' => "",
    'enddatestring' => "",
    'begyear'       => "%04d" % order_start_date.year,
    'begmonth'      => "%02d" % order_start_date.month,
    'begday'        => "%02d" % order_start_date.day,
    'beghour'       => "",
    'begmin'        => "",
    'endyear'       => "%04d" % order_end_date.year,
    'endmonth'      => "%02d" % order_end_date.month,
    'endday'        => "%02d" % order_end_date.day,
    'endhour'       => "",
    'endmin'        => "",
    'outmed'        => "FTP",
    'outpath'       => "",
    'pri'           => "500",
    'datasetname'   => "RAP130",
    'directsub'     => "Y",
    'emailadd'      => "brianhempel@uchicago.edu",
    'outdest'       => "FILE",
    'applname'      => "",
    'subqueryby'    => "STATION",
    'tmeth'         => "Awaiting-Data-Transfer",
  )

  print "Ordering RAP from #{order_start_date} to #{order_end_date}..."

  response =
    begin
      Net::HTTP.start(uri.hostname, uri.port, use_ssl: true, read_timeout: 3*60 ) do |http|
        http.request(request)
      end
    rescue Net::ReadTimeout
      STDERR.puts "read timeout (3 minutes)"
      return
    end

  case response
  when Net::HTTPSuccess
    html_str = response.body
    order_name       = html_str[/Order Number:.*>(HAS\d+)</, 1]
    order_status_url = "https://www.ncdc.noaa.gov/has/" + html_str[/href="(HAS.CheckOrderStatus\?[^"]+)"/i, 1]
    outstanding_orders[order_name] = order_status_url
    puts "ordered. (#{order_status_url})"
    order_dates.each do |date|
      dates_to_order.delete(date.to_s)
    end
  else
    STDERR.puts response.inspect
    STDERR.puts response.body
  end
end

def check_outstanding_orders(outstanding_orders, orders_downloading)
  outstanding_orders.each do |order_name, order_status_url|
    html_str = `curl -s '#{order_status_url.strip}'`
    if html_str =~ /Queued For Processing/
      puts "#{order_name}\tQueued"
    elsif html_str =~ /In Progress/
      progress = html_str.scan(/In Progress.*\s(\d+)%/).flatten.map(&:to_i).max
      puts "#{order_name}\t#{progress}%"
    elsif html_str =~ /Complete.*100%/
      order_url = html_str[/href="([^"]+#{order_name}[^"]*)"/i, 1].sub("http:", "https:")
      outstanding_orders.delete(order_name)
      orders_downloading[order_name] = order_url
      Thread.new do
        cmd = "ruby #{File.expand_path("../get_rap_archived.rb", __FILE__)} #{order_url}"
        puts cmd
        system(cmd) && orders_downloading.delete(order_name)
      end
    else
      STDERR.puts "Unsuccessful check of #{order_name} #{order_status_url}"
      STDERR.puts html_str
    end
  end
end

loop do
  # "if" not "while" to avoid request flooding if we have some bug
  if dates_to_order.count > 0 && outstanding_orders.count < MAX_SIMULTANEOUS_ORDERS
    make_order(dates_to_order, outstanding_orders)
  end
  check_outstanding_orders(outstanding_orders, orders_downloading)
  puts
  sleep(15*60)
end





# Example request from Chrome:
#
# POST /has/HAS.FileSelect HTTP/1.1
# Host: www.ncdc.noaa.gov
# Connection: keep-alive
# Content-Length: 394
# Pragma: no-cache
# Cache-Control: no-cache
# Origin: https://www.ncdc.noaa.gov
# Upgrade-Insecure-Requests: 1
# DNT: 1
# Content-Type: application/x-www-form-urlencoded
# User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36
# Sec-Fetch-User: ?1
# Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
# Sec-Fetch-Site: same-origin
# Sec-Fetch-Mode: navigate
# Referer: https://www.ncdc.noaa.gov/has/HAS.FileAppRouter?datasetname=RAP130&subqueryby=STATION&applname=&outdest=FILE
# Accept-Encoding: gzip, deflate, br
# Accept-Language: en-US,en;q=0.9
# satdisptype=N%2FA&stations=00&stations=01&stations=03&station_lst=&typeofdata=MODEL&dtypelist=&begdatestring=&enddatestring=&begyear=2014&begmonth=03&begday=01&beghour=&begmin=&endyear=2014&endmonth=03&endday=02&endhour=&endmin=&outmed=FTP&outpath=&pri=500&datasetname=RAP130&directsub=N&emailadd=brianhempel%40uchicago.edu&outdest=FILE&applname=&subqueryby=STATION&tmeth=Awaiting-Data-Transfer
#
# satdisptype: N/A
# stations: 00
# stations: 01
# stations: 03
# station_lst:
# typeofdata: MODEL
# dtypelist:
# begdatestring:
# enddatestring:
# begyear: 2014
# begmonth: 03
# begday: 01
# beghour:
# begmin:
# endyear: 2014
# endmonth: 03
# endday: 02
# endhour:
# endmin:
# outmed: FTP
# outpath:
# pri: 500
# datasetname: RAP130
# directsub: N
# emailadd: brianhempel@uchicago.edu
# outdest: FILE
# applname:
# subqueryby: STATION
# tmeth: Awaiting-Data-Transfer
