# https://data.eol.ucar.edu/datafile/nph-get/113.029/td6405.pdf

# ftp ftp@ftp.ncdc.noaa.gov
# brianhempel@ou.edu
# cd /pub/data/asos-onemin/6405-2020
# prompt
# mget *.dat

require "fileutils"

start_year, start_month = (ENV["START_YEAR_MONTH"] || "2000-1").split("-").map(&:to_i)

here           = File.expand_path("..", __FILE__)
process_script = File.expand_path("../process_asos6405.rb", __FILE__)
out_path       = File.expand_path("../gusts.csv", __FILE__)

FileUtils.rm(out_path) rescue nil

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

(2000..2022).each do |year|
  (1..12).each do |month|
    next if ([year, month] <=> [start_year, start_month]) < 0

    puts "#{year}-#{month}"

    FileUtils.cd here
    FileUtils.mkdir_p "asos/6405-#{year}"
    FileUtils.cd "asos/6405-#{year}"

    IO.popen("ftp --no-prompt --prompt -n ftp.ncdc.noaa.gov", "r+") do |ftp|
      wait_for_prompt(ftp)
      ftp_do(ftp, "user ftp #{email}") # password is email address
      ftp_do(ftp, "cd /pub/data/asos-onemin/6405-#{year}")
      ftp_do(ftp, "mget *%02d.dat" % month)
      ftp.puts "quit"
      puts "quit"
    end

    # Generate CSV gust lines from all files, sort them, and write them to gusts.csv
    File.open(out_path, "a") do |out|
      puts "ruby #{process_script} *.dat"
      out.print(`ruby #{process_script} *.dat`.lines.sort.join)
    end

    # exit 1
    system("rm *.dat")
  end
end



# $ ruby storm_data/get_asos6405.rb
#   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
#                                  Dload  Upload   Total   Spent    Left  Speed
# 100 14.8M  100 14.8M    0     0  10.6M      0  0:00:01  0:00:01 --:--:-- 10.6M
# What's your email address?
# brianhempel@ou.edu
# 2000-1
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *01.dat
# mget *01.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 93805:
# 64050KTLH200001.dat
# 93805KTLH TLH2000011913231823   0.098 D                            12                123      6   129
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200001.dat
# 91212PGUM GUM2000011219270927   0.060 N                             41      7    31    9
# 250 stations
# 53 stations with gusts
# 2000-2
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *02.dat
# mget *02.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 14898:
# 64050KGRB200002.dat
# 14898KGRB GRB2000020420560256   0.072 N                          0511   0.081 N              4     12
# USM00072645,IGRA2,19400101,20151231,"",30106204,KGRB,14898,"","",72645,"","","",GREEN BAY/A.-STRAUBEL,GREEN BAY/A.-STRAUBEL,"","","","","","",WI,"","",US,UNITED STATES,"","",686.7,FEET,"","","","","","","","",44.4986,-88.1119,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072645,""

# 21599,MMS,19960701,20030910,"",20021126,KGRB,14898,GRB,GRB,72645,473269,"",USW00014898,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,"",ASHWAUBENON,06,EAST CENTRAL,WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,687,FEET,685.84,FEET,694,FEET,"","","","",44.4794,-88.1378,DDMMSS,1.7 MI SSW,-6,LANDSFC,"ASOS,COOP,WXSVC",USW00014898,55009,"","",ASOS SITE SURVEY,"",""

# 32995,MMS,19950726,99991231,"",30001854,KGRB,14898,"","","","","","",GREEN BAY,GREEN BAY,"","","","","","",WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,822.506588,FEET,"","","","","","","","",44.498633,-88.111111,DDMMSSss,"",-6,RADAR,NEXRAD,"",55009,"","","","",""
# Conflicting station locations for WBAN 93805:
# 64050KTLH200002.dat
# 93805KTLH TLH2000021911311631   0.140 D     D                            248     12   248   16
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200002.dat
# 91212PGUM GUM2000020115140514   0.050 D                             70     13    59   17
# 250 stations
# 66 stations with gusts
# 2000-3
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *03.dat
# mget *03.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 93805:
# 64050KTLH200003.dat
# 93805KTLH TLH2000030917192219   0.142 D                            19    2221   0.129 D
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200003.dat
# 91212PGUM GUM2000030100001400   0.097 N                             80     13    82   15
# 250 stations
# 68 stations with gusts
# 2000-4
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *04.dat
# mget *04.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 93805:
# 64050KTLH200004.dat
# 93805KTLH TLH2000040104120912    0914   0.309 N                             81      5    92    6
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200004.dat
# 91212PGUM GUM2000040113100310   0.070 D                             78     17    71   23
# 250 stations
# 68 stations with gusts
# 2000-5
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *05.dat
# mget *05.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 93805:
# 64050KTLH200005.dat
# 93805KTLH TLH2000050100260526    0528   0.165 N                            120      6   124    6
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200005.dat
# 91212PGUM GUM2000050100001400   0.082 N                            106      5   100    6
# 250 stations
# 69 stations with gusts
# 2000-6
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *06.dat
# mget *06.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200006.dat
# 94823KPIT PIT2000060608241324    1421   0.160 D     0.101 D                328     13   329   15  [
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 93805:
# 64050KTLH200006.dat
# 93805KTLH TLH2000060103410841   0.166 N                          0843   0.167 N
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200006.dat
# 12842KTPA TPA2000062214581958   0.109    1959   0.113 D                 0.105 D    340      8   349
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# Conflicting station locations for WBAN 23160:
# 64050KTUS200006.dat
# 23160KTUS TUS2000061207151415    1417   0.060 D                            175      4   181    5
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200006.dat
# 91212PGUM GUM2000060113190319   0.060 D                            110      8   112   11
# 250 stations
# 132 stations with gusts
# 2000-7
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *07.dat
# mget *07.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 12842:
# 64050KTPA200007.dat
# 12842KTPA TPA2000070404510951   0.177 N                 0.259 N     6    0953   0.177 N
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200007.dat
# 91212PGUM GUM2000070100001400   0.050 N                            106      5   125    6
# 250 stations
# 96 stations with gusts
# 2000-8
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *08.dat
# mget *08.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 23160:
# 64050KTUS200008.dat
# 23160KTUS TUS2000081316252325   1.628 D                            115     40   120   52
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200008.dat
# 91212PGUM GUM2000080117540754   0.050 D                            168      4   174    5
# 250 stations
# 75 stations with gusts
# 2000-9
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *09.dat
# mget *09.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 14898:
# 64050KGRB200009.dat
# 14898KGRB GRB2000090823000500   0.131 N                             5    0501   0.133 N
# USM00072645,IGRA2,19400101,20151231,"",30106204,KGRB,14898,"","",72645,"","","",GREEN BAY/A.-STRAUBEL,GREEN BAY/A.-STRAUBEL,"","","","","","",WI,"","",US,UNITED STATES,"","",686.7,FEET,"","","","","","","","",44.4986,-88.1119,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072645,""

# 21599,MMS,19960701,20030910,"",20021126,KGRB,14898,GRB,GRB,72645,473269,"",USW00014898,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,"",ASHWAUBENON,06,EAST CENTRAL,WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,687,FEET,685.84,FEET,694,FEET,"","","","",44.4794,-88.1378,DDMMSS,1.7 MI SSW,-6,LANDSFC,"ASOS,COOP,WXSVC",USW00014898,55009,"","",ASOS SITE SURVEY,"",""

# 32995,MMS,19950726,99991231,"",30001854,KGRB,14898,"","","","","","",GREEN BAY,GREEN BAY,"","","","","","",WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,822.506588,FEET,"","","","","","","","",44.498633,-88.111111,DDMMSSss,"",-6,RADAR,NEXRAD,"",55009,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200009.dat
# 12842KTPA TPA2000090502270727   0.174 N                 0.231    0729   0.157 N                 0.200
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200009.dat
# 91212PGUM GUM2000090110490049   0.050 D                            155      5   154    6
# 250 stations
# 68 stations with gusts
# 2000-10
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *10.dat
# mget *10.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 14898:
# 64050KGRB200010.dat
# 14898KGRB GRB2000103015112111 [  M    M]                          [180     17   181   51]
# USM00072645,IGRA2,19400101,20151231,"",30106204,KGRB,14898,"","",72645,"","","",GREEN BAY/A.-STRAUBEL,GREEN BAY/A.-STRAUBEL,"","","","","","",WI,"","",US,UNITED STATES,"","",686.7,FEET,"","","","","","","","",44.4986,-88.1119,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072645,""

# 21599,MMS,19960701,20030910,"",20021126,KGRB,14898,GRB,GRB,72645,473269,"",USW00014898,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,GREEN BAY AUSTIN STRAUBEL INTL AP,GREEN BAY AUSTIN STRAUBEL INTL,"",ASHWAUBENON,06,EAST CENTRAL,WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,687,FEET,685.84,FEET,694,FEET,"","","","",44.4794,-88.1378,DDMMSS,1.7 MI SSW,-6,LANDSFC,"ASOS,COOP,WXSVC",USW00014898,55009,"","",ASOS SITE SURVEY,"",""

# 32995,MMS,19950726,99991231,"",30001854,KGRB,14898,"","","","","","",GREEN BAY,GREEN BAY,"","","","","","",WI,BROWN,47,US,UNITED STATES,CENTRAL,GRB,822.506588,FEET,"","","","","","","","",44.498633,-88.111111,DDMMSSss,"",-6,RADAR,NEXRAD,"",55009,"","","","",""
# Conflicting station locations for WBAN 93805:
# 64050KTLH200010.dat
# 93805KTLH TLH2000102623070407    0409   0.194 N                            318      3   324    3
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# Conflicting station locations for WBAN 23160:
# 64050KTUS200010.dat
# 23160KTUS TUS2000100606561356    1357   0.050 D                            142      7   141    9
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200010.dat
# 91212PGUM GUM2000100100001400   0.054 N                             96      5    92    6
# 251 stations
# 81 stations with gusts
# 2000-11
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *11.dat
# mget *11.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 23160:
# 64050KTUS200011.dat
# 23160KTUS TUS2000110111011801    1803   0.050 D                            220      2   202    2
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200011.dat
# 91212PGUM GUM2000110111240124   0.060 D                             81     15    81   18
# 251 stations
# 61 stations with gusts
# 2000-12
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2000
# cd /pub/data/asos-onemin/6405-2000
# ftp> mget *12.dat
# mget *12.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200012.dat
# 94823KPIT PIT2000120121190219   0.104    0221   0.112 N     0.116 N                360      9   346
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200012.dat
# 12842KTPA TPA2000120121370237   0.147 N                 0.153 N     1    0239   0.150 N
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# Conflicting station locations for WBAN 23160:
# 64050KTUS200012.dat
# 23160KTUS TUS2000120101020802    0804   0.050 N                            151      5   153    5
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200012.dat
# 91212PGUM GUM2000120100001400   0.614 N                             68     21    73   23
# 251 stations
# 91 stations with gusts
# 2001-1
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *01.dat
# mget *01.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200101.dat
# 94823KPIT PIT2001010111371637   0.261 D     0.302 D              1639   0.340 D     0.311 D
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200101.dat
# 12842KTPA TPA2001010101450645    0646   0.424 N                 0.606 N     70      3    53    3
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200101.dat
# 91212PGUM GUM2001010111230123   0.059 D                            173     11   156   12
# 252 stations
# 60 stations with gusts
# 2001-2
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *02.dat
# mget *02.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200102.dat
# 94823KPIT PIT2001020109341434   0.243 D     0.254 D                244      6R60+    1436   0.239 D
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200102.dat
# 12842KTPA TPA2001020100090509   0.487    0510   0.481 N                 0.558 N    194      7   190
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200102.dat
# 91212PGUM GUM2001020116180618   0.050 D                             90      7    79    9
# 252 stations
# 62 stations with gusts
# 2001-3
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *03.dat
# mget *03.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200103.dat
# 94823KPIT PIT2001030121530253    0255   0.090 N     0.095 N                239      7   236    9   28
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200103.dat
# 12842KTPA TPA2001030117492249    2250   0.108 D                 0.098 D    259      9   260   10
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200103.dat
# 91212PGUM GUM2001030100001400   0.076 N                             88     14    76   21
# 251 stations
# 69 stations with gusts
# 2001-4
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *04.dat
# mget *04.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Bad line in 64050KLSE200104.dat: "14920KLS- LSE2001040711551755   0.050 D                            238     42   238   53                        \n"
# Bad line in 64050KLSE200104.dat: "14920KLS- LSE2001040712181818   0.050 D                            237     33   242   53                        \n"
# Bad line in 64050KLSE200104.dat: "14920KLS- LSE2001040712321832   0.050 D                            231     39   231   50                        \n"
# Conflicting station locations for WBAN 94823:
# 64050KPIT200104.dat
# 94823KPIT PIT2001040100410541   0.245 N     0R60+    0542   0.249 N     0.342 N                348
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200104.dat
# 12842KTPA TPA2001040100010501   0.158    0502   0.164 N                 0.262 N    267      6   275
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200104.dat
# 91212PGUM GUM2001040119170917   0.092 N                             99      5   123    6
# 251 stations
# 82 stations with gusts
# 2001-5
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *05.dat
# mget *05.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200105.dat
# 94823KPIT PIT2001050100330533    0534   0.120 N     0.122 N                280      0   290    0   28
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 93805:
# 64050KTLH200105.dat
# 93805KTLH TLH2001052823040404    0406   0.170 N                            205      3   202    4
# USM00072214,IGRA2,19400101,20151231,"",30105651,KTLH,93805,"","",72214,"","","",TALLAHASSEE/MUN.,TALLAHASSEE/MUN.,"","","","","","",FL,"","",US,UNITED STATES,"","",173.2,FEET,"","","","","","","","",30.4461,-84.2994,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072214,""

# 33065,MMS,19950426,99991231,"",30001924,KTLH,93805,"","","","","","",TALLAHASSEE,TALLAHASSEE,"","","","","","",FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,176.837276,FEET,"","","","","","","","",30.397583,-84.328944,DDMMSSss,"",-5,RADAR,NEXRAD,"",12073,"","","","",""

# 4856,MMS,19960401,20110415,"",10002302,KTLH,93805,TLH,TLH,72214,088758,"",USW00093805,TALLAHASSEE REGIONAL AP,TALLAHASSEE RGNL AP,TALLAHASSEE WSO AP,TALLAHASSEE WSO AP,"","",01,NORTHWEST,FL,LEON,08,US,UNITED STATES,SOUTHERN,TAE,55,FEET,55,FEET,82,FEET,"","","","",30.39306,-84.35333,DDMMSS,"",-5,LANDSFC,"ASOS,COOP,USHCN",USW00093805,12073,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200105.dat
# 12842KTPA TPA2001050105061006   0.225 N                 0.275    1007   0.227 N                 0.271
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200105.dat
# 91212PGUM GUM2001050100001400   0.099 N                             85      5    89    5
# 252 stations
# 73 stations with gusts
# 2001-6
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *06.dat
# mget *06.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200106.dat
# 94823KPIT PIT2001060100550555   0.088 N     0.096 N                109      5R60+    0557   0.091 N
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200106.dat
# 12842KTPA TPA2001060100090509   0.157    0510   0.156 N                 0.168 N    160      5   160
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200106.dat
# 91212PGUM GUM2001060120311031   0.058 N                             82      4    79    6
# 251 stations
# 70 stations with gusts
# 2001-7
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *07.dat
# mget *07.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200107.dat
# 94823KPIT PIT2001070102160716   0.288 N     0.324 N              0718   0.297 N     0.315 N
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200107.dat
# 12842KTPA TPA2001070105141014   0.197    1015   0.197 N                 0.240 N     95      4    90
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# Conflicting station locations for WBAN 23160:
# 64050KTUS200107.dat
# 23160KTUS TUS2001070218260126   0.482 D                             71     28    74   52
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200107.dat
# 91212PGUM GUM2001070100001400   0.127 N                             94     15    83   16
# 251 stations
# 78 stations with gusts
# 2001-8
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *08.dat
# mget *08.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200108.dat
# 94823KPIT PIT2001080115032003   0.348R60+      186      9   181   10   28R60+
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# No station records with WBAN 99901
# 64050KST1200108.dat
# 99901KST1 ST12001082306571157   0.314 D                            170      4   154    4
# Conflicting station locations for WBAN 12842:
# 64050KTPA200108.dat
# 12842KTPA TPA2001080119220022   0.087 N                 0.119    0024   0.089 N                 0.120
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200108.dat
# 91212PGUM GUM2001080121321132   0.059 N                             65      4    64    5
# 252 stations
# 66 stations with gusts
# 2001-9
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *09.dat
# mget *09.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200109.dat
# 94823KPIT PIT2001090104240924   0.073 N     0R60+    0925   0.075 N     0.076 N                327
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# No station records with WBAN 99901
# 64050KST1200109.dat
# 99901KST1 ST12001090100000500   0.356 N                            321      1   313    2
# Conflicting station locations for WBAN 12842:
# 64050KTPA200109.dat
# 12842KTPA TPA2001090102540754   0.151 N                 0.143 N     8    0756   0.149 N
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200109.dat
# 91212PGUM GUM2001090111130113   0.055 D                            105      8   104    9
# 252 stations
# 75 stations with gusts
# 2001-10
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *10.dat
# mget *10.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200110.dat
# 94823KPIT PIT2001100115362036   0.067 D     0.065 D              2037   0.076 D     0.068 D
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200110.dat
# 12842KTPA TPA2001100115522052    2054   0.075 D                 0.090 D    357      7   343    8   36
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200110.dat
# 91212PGUM GUM2001100100001400   0.062 N                             96      5    95    6
# 251 stations
# 72 stations with gusts
# 2001-11
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *11.dat
# mget *11.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Bad line in 64050KFNT200111.dat: "14826KFNT FNT2001113012321732  10737418240.00 \x9C                            232     10   233   12                \n"
# Conflicting station locations for WBAN 94823:
# 64050KPIT200111.dat
# 94823KPIT PIT2001110100290529   0.099    0530   0.099 N     0.102 N                167      5   161
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200111.dat
# 12842KTPA TPA2001110100020502   0.115    0504   0.114 N                 0.123 N     50      5    57
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200111.dat
# 91212PGUM GUM2001110117050705   0.093 D                             59      6    61    7
# 251 stations
# 62 stations with gusts
# 2001-12
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2001
# cd /pub/data/asos-onemin/6405-2001
# ftp> mget *12.dat
# mget *12.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Conflicting station locations for WBAN 94823:
# 64050KPIT200112.dat
# 94823KPIT PIT2001120100380538   0.083 N     0R60+5      9   234   11   28R60+
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200112.dat
# 12842KTPA TPA2001120101560656    0658   0.202 N                 0.197 N     58      3    45    3   36
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200112.dat
# 91212PGUM GUM2001120103531753   0.143 N                             75     10    75   10
# 252 stations
# 69 stations with gusts
# 2002-1
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2002
# cd /pub/data/asos-onemin/6405-2002
# ftp> mget *01.dat
# mget *01.dat
# ftp> quit
# ruby /Users/brian/Documents/open_source/nadocast/storm_data/process_asos6405.rb *.dat
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013107381338   0.446 N                            35    1339   0.506  5                        \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013107411341   0.447    1342   0.417            1343   0.405            1344   0.363           \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013107531353   0.272 D          1354                    1355      338      6    1356           \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013108081408   0.322    1409   0.320            1410   0.321      34    1411   0.286           \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013109041504   0.677 D                             1    1505   0.656 D                         \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013109581558    1602   0.434 D                            360      6   351    7                \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013110051605    1606   0.378 D                            353      7   351    8                \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013110091609   0.303    1613   0.350 D                            358      8   357             \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013111571757   0.065 D                          1758   0.065 D                                 \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013115272127   1.024 D                             3   1.060 D                                 \n"
# Bad line in 64050KLSE200201.dat: "14920KLS- LSE2002013116472247   1.169 D                          2251   1.177 D                                 \n"
# Conflicting station locations for WBAN 94823:
# 64050KPIT200201.dat
# 94823KPIT PIT2002010210161516    1518   0.397 D     0.383 D                229      3   202    4   28
# USM00072520,IGRA2,19340101,20151231,"",30106170,KPIT,94823,"","",72520,"","","",PITTSBURGH,PITTSBURGH,"","","","","","",PA,"","",US,UNITED STATES,"","",1182.7,FEET,"","","","","","","","",40.5317,-80.2172,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072520,""

# 17263,MMS,19960701,20080201,"",20016809,KPIT,94823,PIT,PIT,72520,366993,"",USW00094823,PITTSBURGH INTL AP,PITTSBURGH INTL AP,PITTSBURGH WSCOM 2 AP,PITTSBURGH WSCOM 2 AP,"","",09,SOUTHWEST PLATEAU,PA,ALLEGHENY,36,US,UNITED STATES,EASTERN,PBZ,1150,FEET,1213,FEET,1204,FEET,"","","","",40.50139,-80.23111,DDMMSS,"",-5,LANDSFC,"ASOS,COOP",USW00094823,42003,"","","","",""
# Conflicting station locations for WBAN 12842:
# 64050KTPA200201.dat
# 12842KTPA TPA2002010210081508   0.131 D                 0.143 D     69      8L60+    1509   0.124 D
# USM00072210,IGRA2,19310101,20151231,"",30105649,KTPA,12842,"","",72210,"","","",TAMPA BAY AREA,TAMPA BAY AREA,"","","","","","",FL,"","",US,UNITED STATES,"","",43,FEET,"","","","","","","","",27.7053,-82.4006,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072210,""

# 4643,MMS,19951101,20101110,"",20004414,KTPA,12842,TPA,TPA,72211,088788,"",USW00012842,TAMPA INTL AP,TAMPA INTL AP,TAMPA WSCMO AP,TAMPA WSCMO AP,"",DREW FIELD,04,SOUTH CENTRAL,FL,HILLSBOROUGH,08,US,UNITED STATES,SOUTHERN,TBW,19,FEET,40,FEET,26,FEET,"","","","",27.96194,-82.54028,DDMMSS,999 UN UN,-5,LANDSFC,"ASOS,COOP",USW00012842,12057,"","","","",""
# Conflicting station locations for WBAN 23160:
# 64050KTUS200201.dat
# 23160KTUS TUS2002010512411941   0.050 D                              7      5    39    65   0.050 D
# USM00072274,IGRA2,19320101,20151231,"",30105691,KTUS,23160,"","",72274,"","","",TUCSON,TUCSON,"","","","","","",AZ,"","",US,UNITED STATES,"","",2432.4,FEET,"","","","","","","","",32.2278,-110.9558,"","","",UPPERAIR,"UPPERAIR,BALLOON","","","","","",USM00072274,""

# 930,MMS,19960101,20031210,"",20000885,KTUS,23160,TUS,TUS,72274,028820,"",USW00023160,TUCSON INTL AP,TUCSON INTL AP,TUCSON AP,TUCSON AP,"","",07,SOUTHEAST,AZ,PIMA,02,US,UNITED STATES,WESTERN,TWC,2549,FEET,2580,FEET,2630,FEET,"","","","",32.13139,-110.95528,DDMMSS,"",-7,LANDSFC,"ASOS,COOP",USW00023160,04019,"","","","",""
# No station records with WBAN 91212
# 64050PGUM200201.dat
# 91212PGUM GUM2002010203571757   0.068 N                             87      5    87    6
# 252 stations
# 75 stations with gusts
# 2002-2
# ftp> user ftp brianhempel@ou.edu
# user ftp brianhempel@ou.edu
# ftp> cd /pub/data/asos-onemin/6405-2002
# cd /pub/data/asos-onemin/6405-2002
# ftp> mget *02.dat
