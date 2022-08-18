
require 'date'

(Date.new(2020,10,1)..Date.today).each do |date|
  url = "https://mrms.nssl.noaa.gov/qvs/product_viewer/local/render_multi_domain_product_layer.php?mode=run&cpp_exec_dir=/home/metop/web/specific/opv/&web_resources_dir=/var/www/html/qvs/product_viewer/resources/&prod_root=MESHMAX1440M&qperate_pal_option=0&qpe_pal_option=0&year=#{date.year}&month=#{date.month}&day=#{date.day}&hour=12&minute=0&clon=-98&clat=38&zoom=4&width=920&height=630"
  convective_date = date - 1

  dir = "%04d%02d" % [convective_date.year, convective_date.month]
  Dir.exist?(dir) || system("mkdir #{dir}")
  puts url
  system("curl -s '#{url}' > #{dir}/#{"mesh_%04d-%02d-%02d.png" % [convective_date.year, convective_date.month, convective_date.day]}")
end
