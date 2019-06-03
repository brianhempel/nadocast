# ruby tweet.rb "Tweet text" [attachment_path]

require "json"
require "shellwords"

tweet_str, attachment_path = ARGV

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

  # https://developer.twitter.com/en/docs/media/upload-media/uploading-media/chunked-media-upload
  init_json    = JSON.parse(`twurl -H upload.twitter.com "/1.1/media/upload.json" -d "command=INIT&media_type=#{media_type}&total_bytes=#{file_size}"`)
  media_id_str = init_json["media_id_string"]
  crash_if_error(init_json)
  # Docs say "HTTP 2XX will be returned with an empty response body on successful upload."
  append_json_str = `twurl -H upload.twitter.com "/1.1/media/upload.json" -d "command=APPEND&media_id=#{media_id_str}&segment_index=0" --file #{Shellwords.escape(attachment_path)} --file-field "media"`
  if append_json_str.size > 1
    crash_if_error(JSON.parse(append_json_str))
  end
  finalize_json = JSON.parse(`twurl -H upload.twitter.com "/1.1/media/upload.json" -d "command=FINALIZE&media_id=#{media_id_str}"`)
  crash_if_error(finalize_json)
end

if media_id_str
  update_json = JSON.parse(`twurl -d #{Shellwords.escape("status=#{tweet_str}&media_ids=#{media_id_str}")} /1.1/statuses/update.json`)
else
  update_json = JSON.parse(`twurl -d #{Shellwords.escape("status=#{tweet_str}")} /1.1/statuses/update.json`)
end
crash_if_error(update_json)
puts("Tweeted " + update_json["text"].inspect)
