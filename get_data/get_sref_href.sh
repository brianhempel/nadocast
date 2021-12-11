#!/usr/bin/env bash

# This is what the crontab actually calls.

# https://stackoverflow.com/a/24112741
this_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$this_dir"
ruby get_sref.rb >> get_sref.log 2>&1
ruby get_href.rb >> get_href.log 2>&1
