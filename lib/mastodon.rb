# ruby mastodon.rb "Toot text" [attachment_path]

require "json"
require "shellwords"
require 'mastodon'

toot_str, attachment_path = ARGV

def crash_if_error(json)
  if json["error"]
    STDERR.puts json["error"]
    exit 1
  end
end

if attachment_path
  extention = attachment_path.split(".")[-1]
  file_size = File.size(attachment_path)
  media_type =
    case extention.downcase
    when "mp4"
      "video/mp4"
    when "png"
      "image/png"
    when /^jpe?g$/
      "image/jpeg"
    else
      STDERR.puts "Unknown attachment type #{extention}!"
      exit 1
    end

  if haskey(ENV, "MASTODON_SERVER")
    mastodon_server = ENV["MASTODON_SERVER"]
  else
    STDERR.puts "No Mastodon server has been set!"
    
  if haskey(ENV, "MASTODON_TOKEN")
    mastodon_token = ENV["MASTODON_TOKEN"]
  else
    STDERR.puts "No Mastodon access token has been set!"
    
  client = Mastodon::REST::Client.new(base_url: mastodon_server, bearer_token: mastodon_token)
  media_id = client.upload_media(attachment_path, params={description: toot_str})
    
end
    
if media_id
  client.create_status(toot_str, params = {media_ids: [media_id]})
end
    

puts("Tooted " + toot_str)
