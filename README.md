# [nadocast.com](http://nadocast.com) — Intraday Tornado Prediction

Planning to use the [HREF](http://nomads.ncep.noaa.gov/txt_descriptions/HREF_doc.shtml), [SREF](https://nomads.ncep.noaa.gov/txt_descriptions/SREF_doc.shtml), [HRRR](https://rapidrefresh.noaa.gov/hrrr/), and [Rapid Refresh](https://rapidrefresh.noaa.gov/) weather models to generate hourly tornado probabilities.

If all goes well, the expected possible advantages over Storm Prediction Center's [Day 1 Outlook tornado probabilities](http://www.spc.noaa.gov/products/outlook/day1otlk.html) are:

1. Finer temporal resolution: indicating _when_ during the day tornadoes are expected.
2. Hopefully more skill by using machine learning approaches.

## Status

- [x] Get Some Data
  - [x] HREF
  - [x] SREF
  - [x] HRRR
  - [x] RAP
- [x] Get some storm events
- [x] Read the weather data
- [x] Build background climatology (spacial, diurnal, annual)
- [x] Storm mode? (some parameters for that, sans estimates from reflectivity)
- [x] Add some interaction terms
- [x] Faster training
- [ ] Faster loading
- [x] ~~Tree refitting~~ didn't help, slow
- [ ] Loss-based tree pruning
- [ ] Set up process:
  - [ ] Retrain SREF with 3-hour chunks
  - [ ] Retrain HREF with 3-hour chunks
  - [ ] Retrain RAP with 3-hour chunks
  - [ ] Retrain HRRR with 3-hour chunks
  - [ ] Optimize hourly weighted combo
  - [ ] Combine into daily

## Model

After a bunch of RAP-only experiments with various gradient descent methods (logistic regression, multiplying multiple logistic regressions together, and lots of other variations), good old gradient boosted decision trees seem to work best but unfortunately require manual feature engineering.

I'm pretty sure I don't have the computation resources to train a convnet that will out-perform the boosted decision trees.

In particular, I would like to include forecasting insight about supercell probability.

The first goal, however, is simply to get the forecast data into an amendable format with some degree of generality between the weather models.

