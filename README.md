# NadoCast.com — Intraday Tornado Prediction

Planning to use the [Rapid Refresh weather model](https://rapidrefresh.noaa.gov/) to generate hourly tornado probabilities.

If all goes well, the expected possible advantages over Storm Prediction Center's [Day 1 Outlook tornado probabilities](http://www.spc.noaa.gov/products/outlook/day1otlk.html) are:

1. Finer resolution:
    1. Temporally: indicate _when_ during the day tornadoes are expected.
    2. Spatially
    3. Probability levels: probabilities not binned to 2%, 5%... etc as they are in the SPC Day 1 Outlook.
2. Hopefully more skill by using machine learning approaches.

## Setup

Get the required packages:

```
$ make setup
```

For training, you need some RAP files.

Make sure you have a lot of hard drive space. Hourly forecasts for a single year of a single forecast hour (e.g. +2 hours ahead) is ~140GB of data. Modify the top of `get_rap.rb` then run:

```
$ make get_rap
```

The script is idempotent and so will not download files that already exist.
