#!/usr/bin/env bash

# This is what the crontab actually calls.

# https://stackoverflow.com/a/24112741
this_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$this_dir"
ruby get_rap.rb >> get_rap.log 2>&1
FORECAST_HOURS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 ruby get_hrrr.rb >> get_hrrr.log 2>&1
