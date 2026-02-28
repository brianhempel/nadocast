# $ ruby report_run_failure.rb SUBJECT LINE

require 'json'
require 'date'
require 'open3'

if ENV["POSTMARK_SERVER_TOKEN"].to_s == ""
  STDERR.puts "POSTMARK_SERVER_TOKEN env var required"
  exit(1)
end

if ENV["EMAIL_ADDRESS"].to_s !~ /@/
  STDERR.puts "EMAIL_ADDRESS env var required"
end

body = "Nadocast run failure: #{ARGV.join(" ")}\n"
body << "\n"

def report_cmd_result(body, cmd)
  body << "$ #{cmd}\n"
  body << `#{cmd} 2>&1`.gsub("\u0000", '')
  body << "\n"
  body << "\n"
end

def report_mtime(body, fname)
  cmd = "stat -c '%y' #{fname} # modification time"
  body << "$ #{cmd}\n"
  mtime_str = `#{cmd} 2>&1`
  body << mtime_str
  begin
    mtime_date = DateTime.parse(mtime_str)
    mins_ago_int = ((DateTime.now - mtime_date) * 24 * 60).round
    hrs_ago_int = mins_ago_int / 60
    body << (hrs_ago_int > 0 ? "#{hrs_ago_int}hr #{mins_ago_int % 60}min ago\n" : "#{mins_ago_int}min ago\n")
  rescue Date::Error => e
    body << e.to_s
  end
  body << "\n"
  body << "\n"
end

report_cmd_result(body, "df -h")
report_cmd_result(body, "top -c -b -n 1 -o %CPU | head -30")
body << "\n"
body << "\n"
report_mtime(body, "/home/brian/nadocast_dev/forecaster.log")
report_cmd_result(body, "tail -60 /home/brian/nadocast_dev/forecaster.log")
report_mtime(body, "/home/brian/nadocast_dev/get_data/get_href.log")
report_cmd_result(body, "tail -10 /home/brian/nadocast_dev/get_data/get_href.log")
report_mtime(body, "/home/brian/nadocast_dev/get_data/get_sref.log")
report_cmd_result(body, "tail -10 /home/brian/nadocast_dev/get_data/get_sref.log")
body << "\n"
body << "\n"
report_mtime(body, "/home/brian/nadocast_operational_2020/forecaster.log")
report_cmd_result(body, "tail -60 /home/brian/nadocast_operational_2020/forecaster.log")
report_mtime(body, "/home/brian/nadocast_operational_2020/get_data/get_href.log")
report_cmd_result(body, "tail -10 /home/brian/nadocast_operational_2020/get_data/get_href.log")
report_mtime(body, "/home/brian/nadocast_operational_2020/get_data/get_sref.log")
report_cmd_result(body, "tail -10 /home/brian/nadocast_operational_2020/get_data/get_sref.log")

json_str = {
  "From" => "Nadocast Bot <#{ENV["EMAIL_ADDRESS"]}>",
  "To"   => ENV["EMAIL_ADDRESS"],
  "Subject" => "Nadocast: #{ARGV.join(" ")}",
  "TextBody" => body,
  "MessageStream" => "outbound"
}.to_json

# puts json_str

cmd = ["curl", "https://api.postmarkapp.com/email", "-X", "POST", "-H", "Accept: application/json", "-H", "Content-Type: application/json", "-H", "X-Postmark-Server-Token: #{ENV['POSTMARK_SERVER_TOKEN']}", "--data-binary", "@-"]

IO.popen(cmd, "r+") do |io|
  io.write(json_str)
  io.close_write
  puts io.read
end