# NadoCast.com — Intraday Tornado Prediction

Planning to use the [HREF](http://nomads.ncep.noaa.gov/txt_descriptions/HREF_doc.shtml), [SREF](https://nomads.ncep.noaa.gov/txt_descriptions/SREF_doc.shtml), [HRRR](https://rapidrefresh.noaa.gov/hrrr/), and [Rapid Refresh](https://rapidrefresh.noaa.gov/) weather models to generate hourly tornado probabilities.

If all goes well, the expected possible advantages over Storm Prediction Center's [Day 1 Outlook tornado probabilities](http://www.spc.noaa.gov/products/outlook/day1otlk.html) are:

1. Finer temporal resolution: indicating _when_ during the day tornadoes are expected.
2. Hopefully more skill by using machine learning approaches.

## Setup

## Status

- [ ] Get Some Data
  - [ ] HREF
  - [ ] SREF
  - [ ] HRRR
  - [ ] RAP
- [ ] Get some storm events
- [ ] Read the weather data

## Model

After a bunch of RAP-only experiments with various gradient descent methods (logistic regression, multiplying multiple logistic regressions together, and lots of other variations), good old gradient boosted decision trees seem to work best but unfortunately require manual feature engineering.

I'm pretty sure I don't have the computation resources to train a convnet that will out-perform the boosted decision trees.

In particular, I would like to include forecasting insight about supercell probability, which based on the angle between the storm motion vector and the initiating front—the supercells move off the front, so that front angle may not be immediately local either temporally or spacially. A dumb convnet that could capture that might have to have a computationally heavy architecture, so some amount of feature engineering may be in order regardless.

The first goal, however, is simply to get the forecast data into an amendable format with some degree of generality between the weather models.

