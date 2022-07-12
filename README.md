# Nadocast - CONUS Severe Weather Probabilities via Feature Engineering and Gradient Boosted Decision Trees

Day tornado outlooks are published on [nadocast.com](http://nadocast.com). All severe hazards and hourly and 4-hourly probabilities are published on [data.nadocast.com](http://data.nadocast.com).

Two models are currently running:

1. **The "2020" models** predict tornadoes based on training data through 2020. 0Z forecasts use the [HREF](http://nomads.ncep.noaa.gov/txt_descriptions/HREF_doc.shtml) and [SREF](https://nomads.ncep.noaa.gov/txt_descriptions/SREF_doc.shtml). 10Z, 14Z, and 20Z intraday updates additionally incorporate the [HRRR](https://rapidrefresh.noaa.gov/hrrr/), and [RAP](https://rapidrefresh.noaa.gov/).
2. **The "2021" models** predict tornadoes, severe wind, severe hail, and significant severe hazards based on training data through 2021. Only the HREF and SREF are used. 0Z, 12Z, and 18Z forecasts are published. The 0Z and 12Z guidance is available internally to SPC forecasters as one of several experimental ML products.

The below describes the "2021" models, but the feature engineering and overall setup for the "2020" models is essentially the same.

## Background

Nadocast began because I (Brian Hempel) wanted to go tornado chasing but knew nothing about meteorology. Back in 2018, the SPC did not produce day 2 tornado probabilities, so I had to stay up until 1am waiting for the day 1 outlook to know if it was worth storm chasing the next day. Additionally, weather forecasting was rarely utilizing state-of-the-art machine learning, but in many domains ML had met or exceeding human performance—why not tornado prediction?

I also wanted hourly predictions so I could know when in the day to chase, so Nadocast was conceived as an hourly predictor.

I was pretty naive at the time about how precise the numerical weather models were, as well as the computation requirements of the problem, but after about three attempts Nadocast started producing outlooks that were competitive with the SPC. That was around the summer of 2021. The "trick" was to add more and more and more features, with more negative data (non-events), as described below.

Because the goal of Nadocast was to maximize prediction performance above that of human-generated forecasts, some tradeoffs were made at the cost of simplicity and elegance. Nadocast is not simple. Some of the feature engineering choices described below might be considered "sloppy" or unscientific, but the expectation was that the learning algorithm would choose helpful features and ignore unhelpful ones.

## Methodology

Input data for training is sourced from an archive of the CONUS HREF and SREF ensemble mean and probability summary files that I have been collecting for several years. The earliest date in the HREF archive is 2018-6-29, and the earliest date in the SREF archive is 2019-1-9 (before this date I didn't grab all the right files). The archive ran out of free storage space on 2020-10-18 and is missing data from then until 2021-3-16 when more space was made available.

The goal is to predict the three types of convective severe weather hazards for which the [Storm Prediction Center (SPC)](https://www.spc.noaa.gov/) issues hazard-specific outlooks: tornadoes, wind, hail. [NOAA's Storm Events Database (Storm Data)](https://www.ncdc.noaa.gov/stormevents/) provide the historical record from which we derive six training targets, shown in the table below. For each of the three hazard types, there are two separate training targets: "severe" events, and more damaging "significant severe" events, based on [the definitions used by the National Weather Service and the SPC](https://www.spc.noaa.gov/misc/about.html).

| Storm Data Event Type | Severe Threshold | Significant Severe Threshold |
| --------------------- | ---------------- | ---------------------------- |
| Tornado               | ≥EF0             | ≥EF2                         |
| Thunderstorm Wind     | ≥50 knots        | ≥65 knots                    |
| Hail                  | ≥1 inch          | ≥2 inches                    |

For the hourly forecasts, a point is given a positive label if it is within ±30 minutes of the hour and within 25 miles (40km) of a point (the distance used by the SPC for their convective outlooks).

The spatial domain of the HREF CONUS grid is larger than the grid of its smallest member, so there are large regions on the edge of the domain with no data. On input, the HREF files are first cropped to remove these regions and to better fit the CONUS. The cropping also considerably speeds processing in later steps. The original 5km grid is resampled to a 15km grid by taking the mean of 3x3 blocks. The SREF input files are cropped to match, but are not resampled and retain their 40km grid.

(In the descriptions below, we give somewhat less detail about the SREF pipeline because the final predictions are derived mostly from the HREF in the final weighting.)

Nadocast performs several steps of feature engineering on the input files: additional parameters are computed from the ensemble mean fields, then spatial means of various radii are computed for all fields, then spatial gradients over various scales. All these parameters from the prior and following forecast hour are concatenated, as well as the min, mean, max, and change over the three hour window. Finally, several fields describing the spatial climatology of storm events are added. For the HREF, this feature engineering expands the initial 58 ensemble mean and 80 ensemble probability fields into 17412 fields for learning (summarized in [Table 1](#table-1)). Similarly, the SREF's 72 mean and 96 probability fields are expanded into 18777 fields for learning. The initial fields and these steps are described in more detail below.

**Initial Fields.** A random sample of 500 forecast hours from the training data was used to determine which fields the input files have in common over the time period. For the HREF, 58 ensemble mean and 80 ensemble probability fields are present over the time period of the dataset ([Table 2](#table-2) and [Table 3](#table-3)), for the SREF, 72 mean and 96 probability field are available for use (not listed). Most notably, only one updraft helicity (UH) field is available to the HREF models. The convection allowing models (CAMs) that make up the HREF explicitly simulate storm updrafts. UH indicates the amount of spin of these updrafts and is a key parameter inspected by human forecasters when predicting severe weather. The only UH parameter available across the whole dataset is the probability of UH >25m²/s², which is a low threshold. When the third version of the HREF was operationalized in May 2021, a >100m²/s² threshold in the HREFv2 was removed and replaced with >75m²/s² and >150m²/s² thresholds, leaving only the >25m²/s² in common throughout the period of the training data. Other automated guidance relies heavily on UH[^stpcalcircle][^srefsseo][^lokenrf], but Nadocast has less access to that information.

**Computed Parameters.** The initial HREF fields include information about wind shear, thermodynamics, and simulated storm attributes, but do not include composite environmental parameters such as the significant tornado parameter (STP) or supercell composite parameter (SCP)[^stpscp]. To give the learning a head-start on some of this environmental information, a number of STP-inspired sub-terms are computed ([Table 4](#table-4)). The computed terms differ slightly from their definitions in the literature because not all the information is directly available in the HREF mean fields, e.g. only 0-3km storm relative helicity (SRH) is available, whereas STP calls for 0-1km SRH. Additionally, alternate versions of the terms were computed with both mixed-layer CAPE (MLCAPE) and surface-based CAPE (SBCAPE), as well as with the square root of each, inspired by the finding of Togstad et al.[^togstad] that sqrt(MLCAPE) * (0-6 bulk wind difference) was better than MLCAPE * (0-6 bulk wind difference) at discriminating between non-tornadic supercells and supercells producing significant tornadoes.

...

For the SREF, to the initial 72 ensemble mean and 96 ensemble probability fields an additional 38 computed fields are added.

## Tables

#### Table 1.

| HREF Feature Engineering Summary                          |
| --------------------------------------------------------- |
| 58 initial ensemble mean fields                           |
| 80 initial ensemble probability fields                    |
| 53 added computed fields                                  |
| 191 25mi spatial mean fields                              |
| 191 50mi spatial mean fields                              |
| 191 100mi spatial mean fields                             |
| 573 forward gradient fields (191 each for 25/50/100mi)    |
| 573 leftward gradient fields (191 each for 25/50/100mi)   |
| 573 straddling gradient fields (191 each for 25/50/100mi) |
| 2483 fields from prior hour                               |
| 2483 fields from following hour                           |
| 2483 3hr min fields                                       |
| 2483 3hr mean fields                                      |
| 2483 3hr max fields                                       |
| 2483 3hr delta fields: $$x_{+1h} - x_{-1h}$$              |
| 31 climatology fields                                     |
| **17412 fields total**                                    |

### Table 2.

| HREF Ensemble Mean Fields Used |
| ------------------------------ |
| CAPE:90-0 mb above ground      |
| CAPE:180-0 mb above ground     |
| CAPE:surface                   |
| CIN:90-0 mb above ground       |
| CIN:180-0 mb above ground      |
| CIN:surface                    |
| HLCY:3000-0 m above ground     |
| HGT:250 mb                     |
| HGT:500 mb                     |
| HGT:700 mb                     |
| HGT:850 mb                     |
| HGT:925 mb                     |
| SOILW:0-0.1 m below ground     |
| TSOIL:0-0.1 m below ground     |
| UGRD:250 mb                    |
| UGRD:500 mb                    |
| UGRD:700 mb                    |
| UGRD:850 mb                    |
| UGRD:925 mb                    |
| VGRD:250 mb                    |
| VGRD:500 mb                    |
| VGRD:700 mb                    |
| VGRD:850 mb                    |
| VGRD:925 mb                    |
| VVEL:700 mb                    |
| TMP:250 mb                     |
| TMP:500 mb                     |
| TMP:700 mb                     |
| TMP:850 mb                     |
| TMP:925 mb                     |
| TMP:2 m above ground           |
| DPT:500 mb                     |
| DPT:700 mb                     |
| DPT:850 mb                     |
| DPT:925 mb                     |
| DPT:2 m above ground           |
| RH:700 mb                      |
| PWAT:entire atmosphere         |
| VIS:surface                    |
| HGT:cloud base                 |
| LCDC:low cloud layer           |
| MCDC:middle cloud layer        |
| HCDC:high cloud layer          |
| TCDC:entire atmosphere         |
| CRAIN:surface                  |
| CFRZR:surface                  |
| CICEP:surface                  |
| CSNOW:surface                  |
| WIND:10 m above ground         |
| WIND:850 mb                    |
| WIND:80 m above ground         |
| WIND:925 mb                    |
| WIND:250 mb                    |
| VVEL:700-500 mb                |
| VWSH:surface                   |
| HINDEX:surface                 |
| VWSH:6000-0 m above ground     |
| HGT:cloud ceiling              |

## Table 3.

| HREF Ensemble Probability Fields Used | Probability Threshold |
| ---- | ---- |
|REFD:1000 m above ground|>30|
|REFD:1000 m above ground|>40|
|REFD:1000 m above ground|>50|
|MAXREF:1000 m above ground|>40|
|MAXREF:1000 m above ground|>50|
|REFC:entire atmosphere|>10|
|REFC:entire atmosphere|>20|
|REFC:entire atmosphere|>30|
|REFC:entire atmosphere|>40|
|REFC:entire atmosphere|>50|
|RETOP:entire atmosphere|>6096|
|RETOP:entire atmosphere|>9144|
|RETOP:entire atmosphere|>10668|
|RETOP:entire atmosphere|>12192|
|RETOP:entire atmosphere|>15240|
|MAXUVV:400-1000 mb|>1|
|MAXUVV:400-1000 mb|>10|
|MAXUVV:400-1000 mb|>20|
|MXUPHL:5000-2000 m above ground|>25|
|CAPE:90-0 mb above ground|>500|
|CAPE:90-0 mb above ground|>1000|
|CAPE:90-0 mb above ground|>1500|
|CAPE:90-0 mb above ground|>2000|
|CAPE:90-0 mb above ground|>3000|
|CIN:90-0 mb above ground|<0|
|CIN:90-0 mb above ground|<-50|
|CIN:90-0 mb above ground|<-100|
|CIN:90-0 mb above ground|<-400|
|HLCY:3000-0 m above ground|>100|
|HLCY:3000-0 m above ground|>200|
|HLCY:3000-0 m above ground|>400|
|TMP:2 m above ground|<273.15|
|DPT:2 m above ground|>283.15|
|DPT:2 m above ground|>285.93|
|DPT:2 m above ground|>288.71|
|DPT:2 m above ground|>291.48|
|DPT:2 m above ground|>294.26|
|PWAT:entire atmosphere|>25|
|PWAT:entire atmosphere|>37.5|
|PWAT:entire atmosphere|>50|
|VIS:surface|<400|
|VIS:surface|<800|
|VIS:surface|<1600|
|VIS:surface|<3200|
|VIS:surface|<6400|
|CRAIN:surface|>=1|
|CFRZR:surface|>=1|
|CICEP:surface|>=1|
|CSNOW:surface|>=1|
|WIND:10 m above ground|>10.3|
|WIND:10 m above ground|>15.4|
|WIND:10 m above ground|>20.6|
|WIND:80 m above ground|>10.3|
|WIND:80 m above ground|>15.4|
|WIND:80 m above ground|>20.6|
|WIND:850 mb|>10.3|
|WIND:850 mb|>20.6|
|WIND:850 mb|>30.9|
|WIND:850 mb|>41.2|
|WIND:850 mb|>51.5|
|WIND:850-300 mb|<5|
|FLGHT:surface|>=1 <2|
|FLGHT:surface|>=2 <3|
|FLGHT:surface|>=3 <4|
|FLGHT:surface|>=4|
|VWSH:surface|>10.3|
|JFWPRB:10 m above ground|>=9 <20|
|HINDEX:surface|>=2 <5|
|HINDEX:surface|>=5 <6|
|HINDEX:surface|>=6 <7|
|VWSH:6000-0 m above ground|>10.3|
|VWSH:6000-0 m above ground|>15.4|
|VWSH:6000-0 m above ground|>20.6|
|VWSH:6000-0 m above ground|>25.7|
|HGT:cloud ceiling|<305|
|HGT:cloud ceiling|<610|
|HGT:cloud ceiling|<915|
|HGT:cloud ceiling|<1372|
|HGT:cloud ceiling|<1830|
|HGT:cloud ceiling|<3050|

### Table 4.

| Computed Parameters from HREF Ensemble Mean Fields        |
| --------------------------------------------------------- |
| SBCAPE\*HLCY3000-0m                                       |
| MLCAPE\*HLCY3000-0m                                       |
| sqrtSBCAPE\*HLCY3000-0m                                   |
| sqrtMLCAPE\*HLCY3000-0m                                   |
| SBCAPE\*BWD0-6km                                          |
| MLCAPE\*BWD0-6km                                          |
| sqrtSBCAPE\*BWD0-6km                                      |
| sqrtMLCAPE\*BWD0-6km                                      |
| SBCAPE\*(200+SBCIN)                                       |
| MLCAPE\*(200+MLCIN)                                       |
| sqrtSBCAPE\*(200+SBCIN)                                   |
| sqrtMLCAPE\*(200+MLCIN)                                   |
| SBCAPE\*HLCY3000-0m\*(200+SBCIN)                          |
| MLCAPE\*HLCY3000-0m\*(200+MLCIN)                          |
| sqrtSBCAPE\*HLCY3000-0m\*(200+SBCIN)                      |
| sqrtMLCAPE\*HLCY3000-0m\*(200+MLCIN)                      |
| SBCAPE\*BWD0-6km\*HLCY3000-0m                             |
| MLCAPE\*BWD0-6km\*HLCY3000-0m                             |
| sqrtSBCAPE\*BWD0-6km\*HLCY3000-0m                         |
| sqrtMLCAPE\*BWD0-6km\*HLCY3000-0m                         |
| SBCAPE\*BWD0-6km\*HLCY3000-0m\*(200+SBCIN)                |
| MLCAPE\*BWD0-6km\*HLCY3000-0m\*(200+MLCIN)                |
| sqrtSBCAPE\*BWD0-6km\*HLCY3000-0m\*(200+SBCIN)            |
| sqrtMLCAPE\*BWD0-6km\*HLCY3000-0m\*(200+MLCIN)            |
| SCPish(RM) = MLCAPE/1000 \* HLCY3000-0m/50 \* BWD0-6km/20 |
| Divergence925mb                                           |
| Divergence850mb                                           |
| Divergence250mb                                           |
| DifferentialDivergence250-850mb                           |
| ConvergenceOnly925mb                                      |
| ConvergenceOnly850mb                                      |
| AbsVorticity925mb                                         |
| AbsVorticity850mb                                         |
| AbsVorticity500mb                                         |
| AbsVorticity250mb                                         |
| SCPish(RM)>1                                              |
| StormUpstream925mbConvergence3hrGatedBySCP                |
| StormUpstream925mbConvergence6hrGatedBySCP                |
| StormUpstream925mbConvergence9hrGatedBySCP                |
| StormUpstream850mbConvergence3hrGatedBySCP                |
| StormUpstream850mbConvergence6hrGatedBySCP                |
| StormUpstream850mbConvergence9hrGatedBySCP                |
| StormUpstreamDifferentialDivergence250-850mb3hrGatedBySCP |
| StormUpstreamDifferentialDivergence250-850mb6hrGatedBySCP |
| StormUpstreamDifferentialDivergence250-850mb9hrGatedBySCP |
| UpstreamSBCAPE1hr                                         |
| UpstreamSBCAPE2hr                                         |
| UpstreamMLCAPE1hr                                         |
| UpstreamMLCAPE2hr                                         |
| Upstream2mDPT1hr                                          |
| Upstream2mDPT2hr                                          |
| UpstreamCRAIN1hr                                          |
| UpstreamCRAIN2hr                                          |


[^stpcalcircle]: Gallo et al. Incorporating UH Occurrence Time to Ensemble-Derived Tornado Probabilities. WAF 2019. https://journals.ametsoc.org/view/journals/wefo/34/1/waf-d-18-0108_1.xml

[^srefsseo]: Jirak et al. Combining probabilistic ensemble information from the environment with simulated storm attributes to generate calibrated probabilities of severe weather hazards. SLS 2014. https://www.spc.noaa.gov/publications/jirak/calprob.pdf

[^lokenrf]: Loken et al. Comparing and Interpreting Differently-Designed Random Forests for Next-Day Severe Weather Hazard Prediction. WAF 2022. https://journals.ametsoc.org/view/journals/wefo/37/6/WAF-D-21-0138.1.xml

[^stpscp]: Thompson et al. Close Proximity Soundings within Supercell Environments Obtained from the Rapid Update Cycle. WAF 2003. https://www.spc.noaa.gov/publications/thompson/ruc_waf.pdf
[^togstad]: Togstad et al. Conditional Probability Estimation for Significant Tornadoes Based on Rapid Update Cycle (RUC) Profiles. WAF 2011. https://journals.ametsoc.org/downloadpdf/journals/wefo/26/5/2011waf2222440_1.xml

