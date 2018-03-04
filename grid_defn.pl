#!/usr/bin/perl -w
# public domain 10/2012 Wesley Ebisuzaki
#
#   finds the grid definition of the 1st record of a grib2 file 
#   grid defintion is compatible with wgrib2's -new_grid
#
#   requires wgrib2 and Perl5
#
#   uses: you want to interpolate to a grid as determined by a grib2 file
#
#   ex. wgrib2 IN.grib -new_grid_winds earth -new_grid `grid_defn.pl output_grid.grb` OUT.grib
#
#
#   usage: grid_defn.pl [grib file]
#
#   limitations: only supports lambert conformal, lat-lon, (global) gaussian, polar stereographic
#     more grids will come later
#

$version="0.2";

# ***** if wgrib2 is not on path, add it here
$wgrib2='wgrib2';

if ($#ARGV != 0) {
   print "grid_defn.pl version$version\n";
   print "argument:  grib2 file\noutput: grid definiton that is compatible with wgrib2 -new_grid\n";
   exit 8;
}

$grid=`$wgrib2 -d 1 -grid $ARGV[0]`;
# print $grid;
$_ = $grid;
if (/\s\sLambert Conformal:/) {
    /Lambert Conformal:\s*\((\d*) x (\d*)\) input (\S*) / or die "parse problem #1";
    $nx = $1;
    $ny = $2;
    $scan = $3;
    if ($scan ne "WE:SN") { print "grid scan is $scan, unsupported by -new_grid\n"; }

    /Lat1 (\S*) Lon1 (\S*) LoV (\S*)/ or die "parse problem #2";
    $lat1 = $1;
    $lon1 = $2;
    $lov = $3;

    /Dx (\S*) m Dy (\S*) m/ or die "parse problem #3";
    $dx = $1;
    $dy = $2;

    /LatD (\S*) Latin1 (\S*) Latin2 (\S*)/ or die "parse problem #4";
    $latd = $1;
    $latin1 = $2;
    $latin2 = $3;

    print "lambert:$lov:$latin1:$latin2:$latd $lon1:$nx:$dx $lat1:$ny:$dy";
    exit 0;
}
if (/\s\slat-lon grid:/) {
    /lat-lon grid:\s*\((\d*) x (\d*)\) units \S* input (\S*) / or die "parse problem #5";
    $nx = $1;
    $ny = $2;
    $scan = $3;
    if ($scan ne "WE:SN" && $scan ne "WE:NS") { print "grid scan is $scan, unsupported by -new_grid\n"; }

    /lat (\S*) to (\S*) by (\S*)/ or die "parse problem #6";
    $lat0 = $1;
    $lat1 = $2;
    $dlat = abs($3);
    if ($lat1 < $lat0) { $dlat = -$dlat; }

    /lon (\S*) to \S* by (\S*)/ or die "parse problem #7";
    $lon0 = $1;
    $dlon = $2;

    print "latlon $lon0:$nx:$dlon $lat0:$ny:$dlat";
    exit 0;
}

if (/\s\sGaussian grid:/) {
    /Gaussian grid:\s*\((\d*) x (\d*)\) units \S* input (\S*) / or die "parse problem #8";
    $nx = $1;
    $ny = $2;
    $scan = $3;
    if ($scan ne "WE:SN" && $scan ne "WE:NS") { print "grid scan is $scan, unsupported by -new_grid\n"; }

    /lon (\S*) to \S* by (\S*)/ or die "parse problem #9";
    $lon0 = $1;
    $dlon = $2;

    /lat (\S*) to \S*/ or die "parse problem #10";
    $lat0 = $1;

    print "gaussian $lon0:$nx:$dlon $lat0:$ny";
    exit 0;
}

if (/\s\spolar stereographic grid:/) {
    /polar stereographic grid:\s*\((\d*) x (\d*)\) input (\S*) / or die "parse problem #11";
    $nx = $1;
    $ny = $2;
    $scan = $3;
    if ($scan ne "WE:SN") { print "grid scan is $scan, unsupported by -new_grid\n"; }

    $type='unknown';
    if (/North pole/) { $type='nps'; }
    if (/South pole/) { $type='sps'; }

    /pole lat1 (\S*) lon1 (\S*) latD (\S*) lonV (\S*) dx (\S*) m dy (\S*) m/ or die "parse problem #12";
    $lat1 = $1;
    $lon1 = $2;
    $latd = $3;
    $lov= $4;
    $dx = $5;
    $dy = $6;

    print "$type:$lov:$latd $lon1:$nx:$dx $lat1:$ny:$dx";
    exit 0;
}

if (/\s\sMercator grid:/) {
    /Mercator grid:\s*\((\d*) x (\d*)\) LatD (\S*) input (\S*) / or die "parse problem #13";
    $nx = $1;
    $ny = $2;
    $latd = $3;
    $scan = $4;
    if ($scan ne "WE:SN") { print "grid scan is $scan, unsupported by -new_grid\n"; }

    /lon (\S*) to (\S*) by (\S*) m/ or die "parse problem #14";
    $lon1 = $1;
    $lon2 = $2;
    $dx = $3;

    /lat (\S*) to (\S*) by (\S*) m/ or die "parse problem #15";
    $lat1 = $1;
    $lat2 = $2;
    $dy = abs($3);
    if ($lat2 < $lat1) { $dy = -$dy; }

    print "mercator:$latd $lon1:$nx:$dx:$lon2 $lat1:$ny:$dy:$lat2";
    exit 0;
}

print "unknown grid sorry";
exit 8;
