require "date"
require "fileutils"

MINUTE = 60
HOUR   = 60*MINUTE

def ymd_to_date(ymd_str)
  Date.new(Integer(ymd_str[0...4]), ymd_str[4...6].to_i, ymd_str[6...8].to_i)
end

class Forecast < Struct.new(:run_date, :run_hour, :forecast_hour)
  def year_month_day
    "%04d%02d%02d" % [run_date.year, run_date.month, run_date.day]
  end

  def year_month
    "%04d%02d" % [run_date.year, run_date.month]
  end

  def run_hour_str
    "%02d" % [run_hour]
  end

  def forecast_hour_str
    "%02d" % [forecast_hour]
  end

  def valid_time
    Time.utc(run_date.year, run_date.month, run_date.day, run_hour) + forecast_hour*HOUR
  end

  def file_name
    raise "unimplemented"
  end

  def archive_url
    raise "unimplemented"
  end

  def ncep_url
    raise "unimplemented"
  end

  def base_directory
    raise "unimplemented"
  end

  def directory
    "#{base_directory}/#{year_month}/#{year_month_day}"
  end

  def path
    "#{directory}/#{file_name}"
  end

  def alt_directory
    raise "unimplemented"
  end

  def alt_path
    raise "unimplemented"
  end

  def make_directories!
    return if DRY_RUN
    system("mkdir -p #{directory} 2> /dev/null")
    if alt_path
      system("mkdir -p #{alt_directory} 2> /dev/null")
    end
  end

  def min_file_bytes
    raise "unimplemented"
  end

  def downloaded?
    (File.size(path) rescue 0) >= min_file_bytes
  end

  def ensure_downloaded!(from_archive: false)
    url_to_get = from_archive ? archive_url : ncep_url
    make_directories!
    unless downloaded?
      puts "#{url_to_get} -> #{path}"
      return if DRY_RUN
      # The pando HRRR archive server doesn't like it when TLS 1.3 is tried before TLS 1.2
      data = `curl -f -s --show-error --tlsv1.2 --tls-max 1.2 #{url_to_get}`
      if $?.success? && data.size >= min_file_bytes
        File.write(path, data)
        if alt_path
          File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < min_file_bytes
        end
      end
    end
  end

  def remove!
    if File.exists?(path)
      puts "REMOVE #{path}"
      FileUtils.rm(path) unless DRY_RUN
    end
    if alt_path && File.exists?(alt_path)
      puts "REMOVE #{alt_path}"
      FileUtils.rm(alt_path) unless DRY_RUN
    end
  end
end
