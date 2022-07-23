#!/usr/bin/env ruby

require "date"
require "set"

# Service Change Notice 17-106
#
# National Weather Service Headquarters Silver Spring, MD
#
# 1140 AM EDT Fri Sep 22 2017
#
#
#
# To:       Subscribers:
#
#           -NOAA Weather Wire Service
#
#           -Emergency Managers Weather Information Network
#
#           -NOAAPORT
#
#           Other NWS Partners, Users and Employees
#
#
#
# From:     Dave Myrick
#
#           NWS Office of Science and Technology Integration
#
#
#
# Subject:  Upgrade to High Resolution Ensemble Forecast (HREF)
#
#           and High-Resolution Window (HIRESW) Effective
#
#           November 1, 2017
#
#
#
# Effective on or about November 1, 2017, beginning with the 1200
#
# Coordinated Universal Time (UTC) run, the National Centers for
#
# Environmental Prediction (NCEP) will upgrade the HREF and HIRESW
#
# systems to makes changes to:
#
#
#
#   - HREF model membership
#
#   - HIRESW model runs, including changes to initialization and
#
# earlier product availability
#
#   - Output products in HREF and HIRESW, including NOAAPORT
#
#
#
# 1) Changes to the HREF model membership
#
#
#
# For the CONUS domain:
#
#
#
#   - Adds a second Weather Research and Forecasting (WRF)
#
# Advanced Research WRF (ARW) member (current cycle run plus the
#
# 12 h old run).
#
#
#
#   - Reduces the number of North American Mesoscale (NAM) 3 km
#
# CONUS nest members in HREF from five to two (previously included
#
# the current cycle plus 6, 12, 18, and 24 hour old runs; now uses
#
# just the current cycle plus the six hour old run).
#
#   - Reduces the number of Nonhydrostatic Multiscale Model on
#
# B-grid (NMMB) members from HIRESW from three to two (previously
#
# included the current cycle plus the 12 and 24 hour old runs; now
#
# uses just the current cycle plus the 12 h old run).
#
#
#
#   - Reduces the number of HIRESW WRF-ARW members from three to
#
# two (previously included the current cycle plus the 12 and 24
#
# hour old runs; now uses just the current cycle plus the 12 h old
#
# run).
#
#
#
# For the Alaska, Hawaii, and Puerto Rico domains:
#
#
#
#   - Generates new HREF products from a six-member ensemble (two
#
# cycles each of HIRESW NMMB, HIRESW ARW, and HIRESW ARW mem2)
#
#
#
# 2) Changes to HIRESW model runs:
#
#
#
# - Unifies the horizontal grid spacing at 3 km for both the ARW
#
# and NMMB models (3.2 km for both over CONUS).
#
#
#
# - Changes horizontal grid spacing (new km/old km)
#
#   For the WRF-ARW runs over the five domains:
#
#    - CONUS (3.2/4.2)
#
#    - Alaska (3.0/3.5)
#
#    - Hawaii (3.0/3.8)
#
#    - Guam (3.0/3.8)
#
#    - Puerto Rico (3.0/3.8)
#
#   For the NMMB run over one domain only
#
#    - CONUS (3.6 to 3.2)
#
#
#
# - Adds a second ARW member for all domains, except for Guam,
#
# that is initialized from the NAM. This second member also uses
#
# the Mellor Yamada Janjic (MYJ) planetary boundary layer (PBL)
#
# and MYJ surface layer physics, so differs from primary HIRESW
#
# ARW member in terms of the physics used in addition to having a
#
# different source of initial and lateral boundary conditions.  It
#
# also has just 40 vertical levels, while the primary ARW member
#
# has 50 vertical levels.
#
#
#
# The NMMB run increases the call frequency for PBL/surface
#
# physics and microphysics from every fourth model time-step to
#
# every other time-step.
#
#
#
# - Changes the vertical level structure within the primary 50
#
# level ARW runs for all domains except Alaska. Counting from the
#
# surface upward, layers 7-18 (roughly 970 to 760 hPa for a
#
# surface pressure of 1010 hPa) are made somewhat thicker, and
#
# layers 19-30 (roughly 690 to 330 hPa for a surface pressure of
#
# 1010 hPa) are made somewhat thinner.  This change was made for
#
# Alaska only in a previous fix to improve numerical stability,
#
# and is extended to all domains with this upgrade.
#
#
#
# - Runs considerably earlier within production suite by using
#
# 6 hr old GFS (or NAM for the second ARW member) files to
#
# generate lateral boundary conditions.  Runs that use the GFS for
#
# initial conditions now utilize a 6 hr forecast from the 6 hr old
#
# GFS for initialization.
#
#
#
# - Changes from 0.5 degree to 0.25 degree GFS data for use in
#
# producing initial and lateral boundary conditions.
#
#
#
# - Initializes the Puerto Rico domain runs from the RAP
#
# (previously initialized from the GFS); the new second ARW member
#
# for Puerto Rico is NAM initialized.
#
#
#
# 3) Changes to Output
#
#
#
# The following changes are on the NCEP Web Services:
#
#   http://nomads.ncep.noaa.gov/pub/data/nccf/com
#
#   http://www.ftp.ncep.noaa.gov/data/nccf/com
#
#   ftp://ftp.ncep.noaa.gov/pub/data/nccf/com
#
#
#
# Where the HIRESW is available under:
#
#   hiresw/prod/hiresw.YYYYMMDD
#
# Where the HREF is available under:
#
#   hiresw/prod/href.YYYYMMDD
#
# Where YYYYMMDD is year, month and day
#
#
#
# HREF Product Changes:
#
#
#
# - The HREF directory will be changing to include subdirectory
#
# hiresw/prod/href.YYYYMMDD/file ->
#
# hiresw/prod/href.YYYYMMDD/ensprod/file
#
# Where file is the href GRIB2 output
#
#
#
# - HREF includes two new output file types: probability matched
#
# mean (pmmn), and an "avrg" type which averages the conventional
#
# mean and probability matched mean.
#
#
#
# - The HREF file naming structure is changed due to introduction
#
# of non-CONUS output:
#
# href.tCCz.TYPE.fFF.grib2 > href.tCCz.GRID.TYPE.fFF.grib2
#
#   Where CC is cycle, GRID is domain (conus, ak, pr or hi), TYPE
#
# is either mean, sprd, prob, pmmn or avrg; and FF is forecast
#
# hour from 00-36.
#
#
#
# - All HREF Output frequency changed from three hourly to hourly,
#
# still extending to 36 hr from the initialization time.
#
#
#
# - Cycle output for the "conus" domain is generated at
#
# 00/06/12/18 UTC. For the Alaska (ak) and Puerto Rico (pr)
#
# domains output is produced at 06/18 UTC. For Hawaii (hi) output
#
# is generated at 00/12 UTC
#
#
#
# - Add neighborhood probability output for more localized fields;
#
# which fields are neighborhood probabilities is specified later
#
# in this subsection in descriptions of "prob" file type changes.
#
#
#
# - Adding new NOMADS Grib Filter functionality.
#
#    HREF Conus - filter_hrefconus.pl
#
#    HREF Alaska - filter_hrefak.pl
#
#    HREF Hawaii - filer_hrefhi.pl
#
#    HREF Puerto Rico - filter_hrefpr.pl
#
#
#
# - Eliminated fields for HREF "mean" and "sprd" file types:
#
#   500 hPa absolute vorticity (ABSV)
#
#
#
# - Added the following fields for HREF "sprd" and "mean" file
#
# types:
#
#
#
# Convective Available Potential Energy (CAPE): 90-0 hPa AGL mixed
#
# layer, 180-0 hPa AGL most unstable and surface based
#
# Convective Inhibition (CIN): 90-0 hPa AGL mixed layer, 180-0 hPa
#
# AGL most unstable and surface based
#
# 3000-0 m AGL helicity
#
# Geopotential Height: 250, 700 and 925 hPa
#
# Soil temperature in 0-10 cm layer
#
# Soil moisture in 0-10 cm layer
#
# U wind component: 250, 500, 700 and 925 hPa.
#
# V wind component: 250, 500, 700 and 925 hPa.
#
# Temperature: 250, 500, 700, and 925 hPa
#
# Dewpoint temperature: 500, 700, 850, and 925 hPa
#
# Wind speed at 250 and 925 hPa
#
# Cloud base height
#
# Low, middle, high and total cloud fraction
#
# 2 m AGL temperature and dewpoint
#
# 10 and 80 m AGL wind speed
#
# Precipitation type (as rain, freezing rain, ice pellets, and
#
# snow)
#
# 1 h precipitation accumulation
#
# 700-500 hPa mean vertical velocity
#
# Haines Index
#
# Wind shear over 0-6000 m layer
#
# 1 h accumulated snowfall liquid equivalent
#
# 3 h accumulated snowfall liquid equivalent [only for forecast
#
# hours dividing evenly into 3]
#
#
#
# - Added fields for HREF "sprd" file type:
#
#
#
# 1000 m AGL simulated reflectivity
#
# 1000 m AGL hourly maximum simulated reflectivity
#
# Composite simulated reflectivity
#
# Echo top height
#
# Updraft helicity over 5000-2000 m AGL layer
#
# Hourly maximum updraft helicity over 5000-2000 m AGL layer
#
# Surface height (topography)
#
#
#
# - This upgrade also corrects the derived forecast type labeling
#
# for "sprd" GRIB2 output to label it as the spread of all members
#
# rather than the weighted mean of all members. In wgrib2
#
# inventory form:
#
#
#
# xx hour fcst:wt mean all members  --->  xx hour fcst:spread all
#
# members
#
#
#
# - Eliminated fields for HREF "prob" file type:
#
#
#
# Hourly maximum updraft velocity over 400-1000 hPa layer (> 5
#
# m/s)
#
# Hourly maximum downdraft velocity over 400-1000 hPa layer (> 1
#
# m/s, > 5 m/s, > 10 m/s)
#
# U and V components of hourly maximum 10 m AGL wind (> 15.4 m/s)
#
# Echo Top height (> 1000 m, > 3000 m, > 5000 m, > 7600 m, >
#
# 10000m)
#
# 3 h accumulated precipitation (> 0.24 mm, > 6.34 mm, > 12.4 mm,
#
# > 25.1 mm, > 50 mm, > 75 mm)
#
# Wind speed at 500 and 250 hPa (> 10.3 m/s, > 20.6 m/s , > 30.9
#
# m/s, > 41.2 m/s, > 51.5 m/s)
#
# Low-level (sfc) wind shear > 20 kts
#
#
#
# - Added fields for “prob” file type:
#
#
#
# ** = a neighborhood probability computed field
#
#
#
# **1000 m AGL simulated reflectivity (> 30 dBZ, > 50 dBZ)
#
# **1000 m AGL hourly maximum simulated reflectivity (> 50 dBZ)
#
# **Echo Top height (> 6096 m, > 9144 m, > 10668 m, > 12192 m, >
#
# 15240 m)
#
# **Updraft Helicity (> 25 m^2/s^2, >100 m^2/s^2)
#
#
#
# **Hourly maximum updraft helicity (> 100 m^2/s^2)
#
# **Hourly maximum updraft velocity over 400-1000 hPa layer (> 20
#
# m/s)
#
# 90-0 m AGL mixed layer CAPE (> 500 J/kg, > 1000 J/kg, > 1500
#
# J/kg, > 2000 J/kg, > 3000 J/kg)
#
# 90-0 m AGL mixed layer CIN (< 0 J/kg, < -50 J/kg, < -100 J/kg, <
#
# -400 J/kg)
#
# 3000-0 m AGL helicity (> 100 m^2/s^2, > 200 m^2/s^2, > 400
#
# m^2/s^2)
#
# 2 m temperature (< 273.15 K)
#
# 2 m dewpoint temperature (> 283.15 K, > 285.93 K, > 288.71 K, >
#
# 291.48 K, > 294.26 K)
#
# Wind shear over 0-6000 m AGL layer (> 10.3 m/s , > 15.4 m/s , >
#
# 20.6 m/s, > 25.7 m/s)
#
# Low-level (sfc) wind shear > 10.3 m/s
#
#
#
# **1 h accumulated precipitation (> 0.25 mm, > 6.35 mm, > 12.7
#
# mm, > 25.4 mm, > 50.8 mm, > 76.2 mm)
#
# **3 h accumulated precipitation (> 0.25 mm, > 6.35 mm, > 12.7
#
# mm, > 25.4 mm, > 50.8 mm, > 76.2 mm) [only for forecast hours
#
# dividing evenly into 3]
#
# ** 6 h accumulated precipitation (> 0.25 mm, > 6.35 mm, > 12.7
#
# mm, > 25.4 mm, > 50.8 mm, > 76.2 mm) [for forecast hours >=6
#
# that divide evenly into 3]
#
# ** 12 h accumulated precipitation (> 2.54 mm, > 6.35 mm, > 12.7
#
# mm, > 25.4 mm > 50.8 mm,  > 76.2 mm,  > 127.0 mm) [for forecast
#
# hours >= 12 that divide evenly into 3]
#
# ** 24 h accumulated precipitation (> 2.54 mm, > 6.35 mm, > 12.7
#
# mm, > 25.4 mm > 50.8 mm,  > 76.2 mm,  > 127.0 mm) [for forecast
#
# hours >= 24 that divide evenly into 3]
#
#
#
# **1 h snowfall liquid equivalent (> 2.54 mm, > 7.62 mm)
#
# **3 h snowfall liquid equivalent (> 2.54 mm, > 7.62 mm, > 15.24
#
# mm) [only for forecast hours dividing evenly into 3]
#
# **6 h snowfall liquid equivalent (> 2.54 mm, > 7.62 mm, > 15.24
#
# mm, > 30.48 mm) [for forecast hours >=6 that divide evenly into
#
# 3]
#
#
#
# - The labeling for the probability of the mean wind over the
#
# 850-300 mb layer has been corrected.  It previously was labeled
#
# as an isobaric layer relative to ground surface, but now is
#
# labeled purely as an isobaric layer.  In wgrib2 inventory form:
#
# WIND:850-300 mb above ground:xx hour fcst:prob <5    --->
#
# WIND:850-300 mb:xx hour fcst:prob <5
#
#
#
# - The new probability matched (PM) mean “pmmn” file type and the
#
# new “avrg” file type, which contains an average of the PM mean
#
# and conventional arithmetic mean, both contain:
#
#
#
# 1000 m AGL simulated reflectivity
#
# 1000 m AGL hourly maximum simulated reflectivity
#
# Composite simulated reflectivity
#
# Echo top height
#
# Updraft helicity over 5000-2000 m AGL layer
#
# Hourly maximum updraft helicity over 5000-2000 m AGL layer
#
# Surface height (topography)
#
# 1 h accumulated precipitation
#
# 3 h accumulated precipitation [only for forecast hours dividing
#
# evenly into 3]
#
#
#
# Changes HIRESW products:
#
#
#
# - For the HIRESW *_5km.*.grib2 and *subset.grib2 output grids,
#
# these new products are added:
#
#   2000-5000 m AGL hourly minimum updraft helicity
#
#   0-3000 m AGL hourly maximum updraft helicity
#
#   0-3000 m AGL hourly minimum updraft helicity
#
#   100-1000 hPa hourly maximum updraft velocity (replaces a 400-
#
# 1000 hPa hourly maximum updraft velocity field)
#
#   100-1000 hPa hourly maximum downdraft velocity (replaces a
#
# 400-1000 hPa hourly maximum downdraft velocity field)
#
#
#
# -For the HIRESW *_5km.*.grib2 (non-subset) output grids, these
#
# new products are added:
#
# VUCSH Vertical U-Component Shear [1/s]:0-6000 m above ground
#
# VVCSH Vertical V-Component Shear [1/s]:0-6000 m above ground
#
#
#
# -For the HIRESW *_5km.*.grib2 Conus only (non-subset) output
#
# grids, these new products are added:
#
#    CPOFP Percent frozen precipitation
#
#    APCP Total Precipitation [kg/m^2]:surface:3-6 hour acc fcst
#
#    WEASD Water Equivalent of Accumulated Snow Depth
#
# [kg/m^2]:surface:3-6 hour acc fcst
#
#
#
# - For the HIRESW *_2p5km*.grib2 and *_3km*.grib2 (non-subset)
#
# grids, two new products are added:
#
#   HGT Cloud ceiling
#
#   HGT Cloud base
#
# 100-1000 hPa hourly maximum updraft velocity (replaces a 400-
#
# 1000 hPa hourly maximum updraft velocity field)
#
#   100-1000 hPa hourly maximum downdraft velocity (replaces a
#
# 400-1000 hPa hourly maximum downdraft velocity field)
#
#
#
# - For the HIRESW *_2p5km*conus.grib2 (non-subset) grids,
#
# following product is removed:
#
#    MAXDVV Hourly Maximum of Downward Vertical Velocity
#
#
#
# - The CONUS “subset” grid with a reduced number of products now
#
# is output on a 1799 x 1059 point 3 km grid (the same grid
#
# utilized for HRRR and NAM CONUS nest output).
#
#
#
# - The file name structure for the CONUS subset grid is changed
#
# to reflect the change in output grid spacing:
#
# hiresw.tCCz.*_5km.fFF.conus.subset.grib2 -->
#
# hiresw.tCCz.*_3km.fFF.conus.subset.grib2
#
#
#
# - A new member will be available for the CONUS domain
#
#  hiresw.tCCz.*_3km.fFF.conusmem2.subset.grib2
#
#
#
# - HIRESW Time labeling for time-averaged surface fluxes
#
# (sensible and latent heat) in NMMB model output have been
#
# corrected for forecast hours not divisible by 3. As an example,
#
# a wgrib2 inventory of a 20 hour forecast of time-averaged latent
#
# heat flux would change in this way:
#
# LHTFL:surface:  --->  LHTFL:surface:18-20 hour ave fcst:
#
#
#
# - Some BUFR output points have been eliminated or redefined:
#
#
#
# All eliminated points are fictitious stations over the ocean
#
# previously added to fill in around the limited stations on land
#
# for Hawaii (HI), Puerto Rico (PR), and Guam.  Details are
#
# available at:
#
# http://www.emc.ncep.noaa.gov/mmb/mpyle/hiresw/bufr_changes/hireswv7.txt
#
#
#
# And while not documented, all domains that had a change in
#
# resolution (all ARW domains and the CONUS NMMB) will have some
#
# slight changes in the exact location of individual BUFR output
#
# stations.  Changes to the grid resolution modifies the grid
#
# dimensions and grid point locations, and because the BUFR output
#
# is taken from values at the nearest model grid point it may be
#
# shifted by up to a few kilometers.
#
#
#
# - The HIRESW 5km data will be discontinued from the NWS web
#
# services, and can instead be found on the NCEP web services.
#
# Please see below for what is being removed, and the exact
#
# replacement file name on NCEP web services. Users are encouraged
#
# to migrate over to the NCEP web files at any time.
#
#
#
# http://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/MT.hires_MR.arw_CY
#
# .CC/
#
# ftp://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/MT.hires_MR.arw_CY.
#
# CC/
#
# http://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/MT.hires_MR.nmm_CY
#
# .CC/
#
# ftp://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/MT.hires_MR.nmm_CY.
#
# CC/
#
# Subdirectories
#
# RD.YYYYMMDD/PT.grid_DF.gr2_AR.conus05/
#
# RD.YYYYMMDD/PT.grid_DF.gr2_AR.alaska05/
#
# RD.YYYYMMDD/PT.grid_DF.gr2_AR.hi05/
#
# RD.YYYYMMDD/PT.grid_DF.gr2_AR.pr05/
#
# Where CC is cycle and YYYYMMDD is year, month and day
#
#
#
# PT.grid_DF.gr2_AR.conus05 fh.00FF_tl.press_gr.awpreg ->
#
# hiresw.tCCz.arw_5km.fFF.conus.grib2
#
# hiresw.tCCz.nmmb_5km.fFF.conus.grib2
#
#
#
# PT.grid_DF.gr2_AR.alaska05 fh.00FF_tl.press_gr.awpreg  ->
#
# hiresw.tCCz.arw_5km.fFF.ak.grib2
#
# hiresw.tCCz.nmmb_5km.fFF.ak.grib2
#
#
#
# PT.grid_DF.gr2_AR.hi05 fh.00FF_tl.press_gr.awpreg ->
#
# hiresw.tCCz.arw_5km.fFF.hi.grib2
#
# hiresw.tCCz.nmmb_5km.fFF.hi.grib2
#
#
#
# PT.grid_DF.gr2_AR.pr05 fh.00FF_tl.press_gr.awpreg ->
#
# hiresw.tCCz.arw_5km.fFF.pr.grib2
#
# hiresw.tCCz.nmmb_5km.fFF.pr.grib2
#
# Where FF is forecast hour
#
#
#
# NOAAPORT/SBN Additions:
#
#
#
# - With this upgrade the HREF will become available over
#
# NOAAPORT/SBN across the CONUS, Alaska, Hawaii and Puerto Rico
#
# domains. The total volume increase with be 8.6GB/day. Please see
#
# this document below for the description of product WMO headers:
#
# http://www.nco.ncep.noaa.gov/pmb/changes/docs/HREF_WMO_description.pdf
#
#
#
# Please see this document for every WMO header available:
#
#
#
# http://www.emc.ncep.noaa.gov/mmb/mpyle/hiresw/wmo/wmo_headers.txt
#
#
#
# Data Availability and Schedule Changes:
#
#
#
# With the change in initialization procedures described earlier,
#
# HIRESW output will be made available much earlier:
#
#
#
# ~115 minutes earlier for CONUS domain
#
# ~115 minutes earlier for Alaska domain
#
# ~95 minutes earlier for Hawaii domain
#
# ~110 minutes earlier for Puerto Rico domain
#
# ~110 minutes earlier for Guam domain
#
#
#
# CONUS HREF output will be available about 110 minutes earlier
#
# than in current operations.
#
#
#
# New HREF domain output will be available approximately this long
#
# after the nominal cycle (CC) time:
#
#
#
# Alaska: CC + 02:55
#
# Hawaii: CC + 02:45
#
# Puerto Rico:  CC + 02:45
#
#
#
# A consistent parallel feed of data is currently available on the
#
# NCEP HTTP server at the following URL:
#
#
#
# http://para.nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/para/
#
# http://para.nomads.ncep.noaa.gov/pub/data/nccf/noaaport/
#
#
#
# NCEP urges all users to ensure their decoders can handle changes
#
# in content order, changes in the scaling factor component within
#
# the product definition section (PDS) of the GRIB files, and
#
# volume changes. These elements may change with future NCEP model
#
# implementations. NCEP will make every attempt to alert users to
#
# these changes before implementation.
#
#
#
# Any questions, comments or requests regarding this
#
# implementation should be directed to the contacts below.  We
#
# will review any feedback and decide whether to proceed.
#
#
#
# For questions regarding these changes, please contact:
#
#
#
# Matthew Pyle
#
# NCEP/EMC Engineering and Implementation Branch
#
# College Park, MD
#
# 3016833687
#
# Matthew.Pyle@noaa.gov
#
#
#
# For questions regarding the data flow aspects, please contact:
#
#
#
# Carissa Klemmer
#
# NCEP/NCO Dataflow Team Lead
#
# College Park, MD
#
# 3016830567
#
# ncep.list.pmb-dataflow@noaa.gov
#
#
#
# NWS Service Change Notices are online at:
#
#
#
#     http://www.weather.gov/om/notif.htm
#
#


# Files available 2.5hrs after run time

DOMAIN = ENV["USE_ALT_DOMAIN"] == "true" ? "ftpprd.ncep.noaa.gov" : "nomads.ncep.noaa.gov/pub"

TYPES          = ["mean", "prob"]
YMDS           = `curl -s https://#{DOMAIN}/data/nccf/com/href/prod/`.scan(/\bhref\.(\d{8})\//).flatten.uniq
HOURS_OF_DAY   = [00, 06, 12, 18]
FORECAST_HOURS = (01..48).to_a
BASE_DIRECTORY_1 = "/Volumes/SREF_HREF_1/href"
BASE_DIRECTORY_2 = "/Volumes/SREF_HREF_3/href"
MIN_FILE_BYTES = 20_000_000
THREAD_COUNT   = Integer(ENV["THREAD_COUNT"] || "4")

AVAILABLE_FOR_DOWNLOAD = YMDS.flat_map do |ymd|
  remote_files = `curl -s https://#{DOMAIN}/data/nccf/com/href/prod/href.#{ymd}/ensprod/`.scan(/\bhref\.t[\.0-9a-z_]+/).grep(/conus.*grib2$/)
  puts `curl -s https://#{DOMAIN}/data/nccf/com/href/prod/href.#{ymd}/ensprod/`
  remote_files.map { |name| "https://#{DOMAIN}/data/nccf/com/href/prod/href.#{ymd}/ensprod/#{name}" }
end.to_set

def alt_location(directory)
  directory.sub(/^\/Volumes\/SREF_HREF_1\//, "/Volumes/SREF_HREF_2/").sub(/^\/Volumes\/SREF_HREF_3\//, "/Volumes/SREF_HREF_4/")
end

loop { break if Dir.exists?("/Volumes/SREF_HREF_3/"); puts "Waiting for SREF_HREF_3 to mount..."; sleep 4 }
loop { break if Dir.exists?("/Volumes/SREF_HREF_4/"); puts "Waiting for SREF_HREF_4 to mount..."; sleep 4 }


# https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.20180629/ensprod/href.t00z.conus.prob.f01.grib2

forecasts_to_get = YMDS.product(HOURS_OF_DAY, FORECAST_HOURS, TYPES)


threads = THREAD_COUNT.times.map do
  Thread.new do
    while forecast_to_get = forecasts_to_get.shift
      year_month_day, run_hour, forecast_hour, type = forecast_to_get
      year_month        = year_month_day[0...6]
      run_hour_str      = "%02d" % [run_hour]
      forecast_hour_str = "%02d" % [forecast_hour]

      file_name         = "href_conus_#{year_month_day}_t#{run_hour_str}z_#{type}_f#{forecast_hour_str}.grib2"
      url_to_get        = "https://#{DOMAIN}/data/nccf/com/href/prod/href.#{year_month_day}/ensprod/href.t#{run_hour_str}z.conus.#{type}.f#{forecast_hour_str}.grib2"
      if AVAILABLE_FOR_DOWNLOAD.include?(url_to_get)
        base_directory    = year_month[0...4].to_i < 2021 ? BASE_DIRECTORY_1 : BASE_DIRECTORY_2
        directory         = "#{base_directory}/#{year_month}/#{year_month_day}"
        path              = "#{directory}/#{file_name}"
        alt_directory     = alt_location(directory)
        alt_path          = alt_location(path)

        system("mkdir -p #{directory} 2> /dev/null")
        system("mkdir -p #{alt_location(alt_directory)} 2> /dev/null")
        if (File.size(path) rescue 0) < MIN_FILE_BYTES
          puts "#{url_to_get} -> #{path}"
          data = `curl -f -s --show-error #{url_to_get}`
          if $?.success? && data.size >= MIN_FILE_BYTES
            File.write(path, data)
            File.write(alt_path, data) if Dir.exists?(alt_directory) && (File.size(alt_path) rescue 0) < MIN_FILE_BYTES
          end
        end
      end
    end
  end
end

threads.each(&:join)
