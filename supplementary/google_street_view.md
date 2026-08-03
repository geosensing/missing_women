# Supplementary Information: Google Street View triangulation

The Google Street View exercise closely reproduces the wearable-camera estimate in Delhi and Mumbai, but yields a meaningfully higher estimate in Navi Mumbai. Street View should therefore enter the manuscript as a complementary measurement exercise, not as a validation sample. It uses the same primary estimand, the person-weighted share of classified pedestrian sightings coded as women, but observes different streets, dates, times, and image-generation processes.

## Comparison with the wearable-camera estimates

The Street View estimate is 20.0% in Delhi, 20.5% in Mumbai, and 24.8% in Navi Mumbai. The corresponding wearable-camera estimates are 19.8%, 19.1%, and 18.1%. The difference is 0.2 percentage points in Delhi and 1.4 points in Mumbai. The 6.7-point difference in Navi Mumbai is much larger, and the two descriptive 95% confidence intervals do not overlap.

| City | Street View female share | Wearable-camera female share | Difference, percentage points | 95% intervals overlap |
|---|---:|---:|---:|:---:|
| Delhi | 0.200 [0.171, 0.229] | 0.198 [0.173, 0.222] | +0.2 | Yes |
| Mumbai | 0.205 [0.185, 0.225] | 0.191 [0.174, 0.208] | +1.4 | Yes |
| Navi Mumbai | 0.248 [0.211, 0.284] | 0.181 [0.160, 0.202] | +6.7 | No |

Table S1. Women's share among classified pedestrian sightings in two image sources. Street View intervals cluster annotated images by sampled location. Wearable-camera intervals cluster frames by collection day. The intervals describe variability within the observed image collections. Neither image collection is a probability sample of city residents, so interval overlap is not a population-level equivalence test. Source: `notebooks/outputs/v2/streetscope_comparison.csv` in the Street View repository.

Agreement in Delhi and Mumbai is informative because the image sources differ sharply. Street View uses panoramas acquired by Google or public contributors at dates and times selected by those providers. The wearable-camera study uses images collected by the research team along planned routes during known fieldwork windows. Similar point estimates under those different processes make it less likely that the Delhi and Mumbai results are artifacts of one camera platform alone. They do not show that either source is unbiased.

The Navi Mumbai discrepancy should remain visible. It could reflect differences in covered streets, panorama vintage, unknown capture time, annotation completion, or real temporal and spatial heterogeneity. The current data do not identify how much each mechanism contributes. Reweighting cannot solve this problem because the panorama acquisition and annotation completion probabilities are unknown.

## Sampling frame and location selection

The sampling frame came from OpenStreetMap road ways tagged `primary`, `secondary`, `tertiary`, `residential`, or `unclassified`. Mumbai and Delhi were queried within OpenStreetMap relations 7888990 and 1942586. Navi Mumbai was queried within the bounding box 18.95 to 19.25 latitude and 72.95 to 73.15 longitude. The different boundary rule for Navi Mumbai is part of the design and can change which peripheral roads enter the frame.

Each OpenStreetMap way was broken at consecutive node pairs. A pair of length \(d\) meters contributed \(\max(1, \lfloor d/500 \rfloor)\) equal pieces, and the midpoint of each piece became an eligible road point. Thus, a pair shorter than 1,000 meters contributed one point, while a pair from 1,000 to less than 1,500 meters contributed two. This produced 126,827 eligible segments in Mumbai, 580,608 in Delhi, and 125,804 in Navi Mumbai. The design sampled 2,500 points in Mumbai with seed 42, 2,500 in Delhi with seed 43, and 2,000 in Navi Mumbai with seed 44. It then selected 1,500 points per city with seed 42, giving 4,500 locations for the Street View coverage query. Both stages reproduce the frozen samples exactly.

Sampling was uniform over constructed segment rows. It was not explicitly proportional to physical road length. OpenStreetMap ways with denser node placement can contribute more adjacent pairs, and therefore more eligible rows, even when their physical lengths are similar. Roads outside the five retained tags, pedestrian paths, alleys, private roads, unmapped roads, and places outside the specified relation or bounding box had no chance of selection. Equal allocation of 1,500 queried locations per city was useful for city comparisons but was not population, area, or road-length weighting.

The frozen OpenStreetMap road files do not contain an Overpass query timestamp or the original response payload. Their exact local rows are protected by the input manifest, but the historical state of OpenStreetMap from which they were constructed cannot be independently re-queried. This is a provenance limitation that should be corrected in any future spatial sampling exercise.

## Street View coverage and image construction

Coverage metadata were frozen on February 7, 2026. The pipeline queried the Google Street View Static API using each sampled latitude and longitude. The request did not set a search radius, so the API's default 50-meter radius applied. Google documents that a coordinate request returns the closest panorama found within that radius.[^google-request] Coverage therefore means that the API found a panorama near the sampled point, not necessarily at the point.

The frozen metadata retain the sampled coordinates, panorama ID, capture year and month when available, and API status. They do not retain the panorama coordinates returned by Google. The distance from the sampled road point to the selected panorama cannot be recovered for the existing data. The collection code now stores the returned coordinates and snap distance for any future query, but no new collection is required for this analysis. Google notes that imagery is periodically refreshed and that panorama IDs or positions can change.[^google-metadata]

Of the 4,500 queried locations, 3,551 had coverage. The city coverage rates were 78.1% in Delhi, 83.1% in Mumbai, and 75.6% in Navi Mumbai. Four square views were intended at every covered location, with headings 0, 90, 180, and 270 degrees, pitch 0, and a nominal 90-degree field of view. The high-resolution annotation images were cropped from panorama tiles using `streetlevel` 0.12.5. There were 14,203 available views from 3,551 locations. One intended Navi Mumbai view failed, leaving three rather than four views at that location.

| City | Eligible OSM segments | Initial road-point sample | Locations queried | Locations with coverage | Available views | Annotated views | Annotated locations | Views with pedestrians | Pedestrian sightings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Delhi | 580,608 | 2,500 | 1,500 | 1,171 | 4,684 | 769 | 500 | 375 | 1,453 |
| Mumbai | 126,827 | 2,500 | 1,500 | 1,246 | 4,984 | 636 | 498 | 427 | 2,270 |
| Navi Mumbai | 125,804 | 2,000 | 1,500 | 1,134 | 4,535 | 537 | 437 | 247 | 1,014 |

Table S2. Street View sampling and annotation funnel. Coverage and annotation are separate selection stages. The analysis includes 1,942 annotated views from 1,435 locations. Source: `notebooks/outputs/v2/sampling_funnel.csv`.

## Annotation and estimand

The frozen version 2 export contains 1,947 annotation records for 1,942 image tasks. Five tasks had a second annotation. The primary analysis selects the annotation from the annotator who completed 1,942 tasks, yielding one record per image and no fallback records. Five duplicate annotations are too few to support a meaningful inter-rater reliability estimate. This is one reason to keep the Street View evidence supplementary.

Annotators counted pedestrians perceived as women and men and separately counted two-wheeler riders. Empty count fields are encoded as zero under the annotation protocol. The `10+` category is interval-censored and is represented by its known minimum, 11, in the primary analysis. The primary estimate is

\[
\widehat{p}_c = \frac{\sum_i W_{ic}}{\sum_i (W_{ic}+M_{ic})},
\]

where \(W_{ic}\) and \(M_{ic}\) are the coded pedestrian counts in annotated view \(i\) for city \(c\). Thus, a person sighting is the unit of weighting. Images without pedestrians add zero to both sums. The 95% intervals use location-clustered estimating-equation scores so that headings from the same panorama location are not treated as independent.

This estimand describes classified pedestrian sightings in the annotated Street View images. It is not the proportion of residents who are women, the proportion of unique pedestrians who are women, or a design-weighted estimate for all streets. A person visible across adjacent crop boundaries could be counted more than once, while a person hidden by traffic, vegetation, blur, or the camera's field of view is not counted.

## Panorama vintage and capture timing

Street View capture dates were not randomized. Across all covered locations, panorama years range from 2013 to 2026 in Delhi, 2013 to 2026 in Mumbai, and 2012 to 2025 in Navi Mumbai. The annotated subset is narrower but still mixes years. Delhi is dominated by 2022 imagery, Mumbai by 2021 and 2024 imagery, and Navi Mumbai by 2021, 2024, and 2025 imagery.

| City and capture year | Annotated views | Pedestrian sightings | Pedestrian female share |
|---|---:|---:|---:|
| Delhi, 2022 | 552 | 966 | 0.205 |
| Delhi, 2023 | 16 | 44 | 0.182 |
| Delhi, 2025 | 201 | 443 | 0.190 |
| Mumbai, 2021 | 368 | 1,334 | 0.193 |
| Mumbai, 2024 | 211 | 759 | 0.216 |
| Mumbai, 2025 | 41 | 110 | 0.309 |
| Navi Mumbai, 2021 | 125 | 219 | 0.183 |
| Navi Mumbai, 2022 | 63 | 63 | 0.270 |
| Navi Mumbai, 2024 | 190 | 374 | 0.262 |
| Navi Mumbai, 2025 | 159 | 358 | 0.268 |

Table S3. Street View pedestrian female share by panorama capture year. Cells with fewer than 20 pedestrian sightings are omitted. These are descriptive strata, not time trends, because the streets and panorama acquisition process differ across years. Source: `notebooks/outputs/v2/capture_year_estimates.csv`.

The year pattern helps contextualize the comparison without resolving it. Mumbai's 2021 Street View estimate is 19.3%, almost identical to the 2025 wearable-camera estimate of 19.1%. Navi Mumbai's 2021 Street View estimate is 18.3%, close to the 2025 wearable-camera estimate of 18.1%, while the 2022, 2024, and 2025 Street View strata are about 26% to 27%. These differences cannot be interpreted as temporal change because year is confounded with the streets Google covered, the panorama selected near each query, and the subset completed by annotators.

The metadata provide capture year and usually month, but not time of day, day of week, weather, traffic conditions, or the operational reason a street was captured. Google and public contributors do not visit sampled points at random times. Pedestrian composition varies by commuting period, school schedules, market hours, season, heat, rainfall, and local events. The wearable-camera data have known collection dates and times. The Street View data do not permit matching those windows.

## Selection and measurement limitations

| Stage | Observed issue | Consequence for interpretation |
|---|---|---|
| OpenStreetMap frame | Only five vehicular road tags and specified boundaries enter the frame. Node density affects the number of constructed segments. | The sample does not represent all public space or all road length with known probabilities. |
| Equal city allocation | Each city contributes 1,500 coverage queries regardless of population, area, or road length. | Pooled estimates require an external weighting target and are not reported as city-population estimates. |
| Street View availability | Coverage is 75.6% to 83.1% by city and ranges from 58.3% to 100% across city and road-class cells. Residential and unclassified roads generally have lower coverage. | Covered roads may be more accessible, central, or frequently driven than uncovered roads. |
| Panorama snapping | The API searches up to 50 meters and returns the closest panorama. Returned coordinates were not frozen in the existing metadata. | Some images can represent a nearby road position rather than the exact sampled midpoint. Snap distance cannot be used for sensitivity analysis. |
| Imagery source | The request used Google's default source setting. Google states that default Street View can include Google imagery and public user-generated content.[^google-overview] Source type was not stored. | Camera system, contributor behavior, and acquisition protocols may vary across panoramas. |
| Capture date and time | Capture year and month vary substantially, and time of day is unavailable. | Cross-city and cross-platform differences combine place, vintage, season, and time-of-day composition. |
| Annotation completion | Only 1,942 of 14,203 available views were annotated. No committed random selection rule explains this subset. Annotated locations received 1.23 to 1.54 headings on average. | Missing annotation is not known to be random. Standard errors do not correct this selection. |
| Road-class composition | The share of selected locations that became annotated ranges from 27.2% to 44.3% across city and road-class cells. | Differences in road mix can shift city estimates when pedestrian composition varies by road class. |
| Visibility | Occlusion, distance, resolution, and Google's face and license-plate blurring affect what annotators can see. Thirty-eight annotated views were marked heavily blurred. | Some people and perceived-gender cues are differentially unobservable. |
| Image projection | The high-resolution routine takes direct square crops from an equirectangular panorama rather than constructing perspective-correct rectilinear views. | Geometry and apparent scale vary across each crop and can affect visibility near its edges. |
| Repeated fields of view | Four adjacent 90-degree headings cover a panorama, but people can straddle crop boundaries or move during panorama construction. | Counts are sightings, not deduplicated individuals. |
| Appearance-based coding | Categories record perceived gender from visible appearance. | The measure can misclassify gender and does not establish gender identity or biological sex. |

Coverage selection is empirically nonuniform. Residential-road coverage is 75.8% in Delhi, 77.3% in Mumbai, and 70.8% in Navi Mumbai, compared with 92.9%, 96.2%, and 88.0% on tertiary roads. Annotation completion is another distinct stage. Among covered locations, road-class-specific annotation rates range from 36.1% to 57.6%. The analysis reports these rates rather than assigning inverse-probability weights because the relevant inclusion probabilities are not known.

## Sensitivity analyses

Replacing `10+` by larger values lowers the person-weighted female share because top-coded pedestrian counts are concentrated among men. The Street View estimates at replacements 11, 15, 20, and 30 are 20.0%, 19.2%, 18.4%, and 16.9% in Delhi; 20.5%, 19.0%, 17.4%, and 14.9% in Mumbai; and 24.8%, 24.2%, 23.6%, and 22.4% in Navi Mumbai. This sensitivity is most important in Mumbai. It does not erase the need to discuss the Navi Mumbai discrepancy.

Excluding the 38 views marked heavily blurred changes the primary estimate from 20.0% to 19.8% in Delhi, from 20.5% to 20.6% in Mumbai, and from 24.8% to 24.8% in Navi Mumbai. The blur flag therefore has little influence on the city aggregates. It does not address unflagged occlusion or uncertainty in appearance-based coding.

## Reproducibility and archiving

The authoritative Street View analysis is script based. With the two repositories cloned as siblings, run:

```bash
cd missing_women_gsview
uv sync --all-extras
make analysis
make ci
```

`scripts/04_analyze_annotations.py` reads the frozen road frames, both location samples, coverage metadata, download manifest, Label Studio export, and the sibling Streetscope Parquet files. It regenerates the sampling funnel, city estimates, capture-year tables, sensitivity analyses, maps, and cross-platform comparison in `notebooks/outputs/v2/`. It verifies that the 7,000-location and 4,500-location samples reproduce exactly from their recorded seeds. `input_manifest.json` records byte counts and SHA-256 hashes for every frozen Street View input used by the build.

The current local checkout is exactly reproducible, but a public replication package needs an explicit archive step. The road frames, sample CSVs, coverage CSV, and download manifest are currently excluded by `.gitignore`; they must be deposited with the replication materials or in a stable data archive. The raw Street View images should not be placed in the repository. Google's current policy generally prohibits prefetching, storing, or caching Street View content, while specifically allowing panorama IDs to be stored indefinitely.[^google-policy] The package should distribute code, permitted non-image metadata, annotations, input hashes, and derived tables subject to the applicable Google Maps terms and OpenStreetMap attribution requirements.

The high-resolution image path used `streetlevel`, a third-party library that accesses panorama tiles rather than the official Street View Static API image endpoint. Before public release, the authors should confirm that the original acquisition and any proposed archival or display of those crops comply with the Google Maps terms that applied to the account and acquisition date. This compliance question does not alter the frozen numerical analysis, but it determines which image artifacts can be shared.

A future API query is a reconstruction of the design, not an exact replication of the frozen image sample. Google refreshes imagery, panorama IDs can change, and the nearest available panorama can move.[^google-metadata] Exact computational replication therefore depends on preserving the frozen permitted metadata and annotation export. Reacquisition tests a related sample under a later Street View state.

[^google-request]: Google Maps Platform, [Street View Static API request parameters](https://developers.google.com/maps/documentation/streetview/request-streetview), documenting the default 50-meter radius and nearest-panorama behavior.
[^google-metadata]: Google Maps Platform, [Street View image metadata](https://developers.google.com/maps/documentation/streetview/metadata), documenting returned location and date fields, refreshes, and changing panorama IDs.
[^google-overview]: Google Maps Platform, [Street View Static API overview](https://developers.google.com/maps/documentation/streetview/overview), noting Google and public user-generated imagery sources.
[^google-policy]: Google Maps Platform, [Street View Static API policies](https://developers.google.com/maps/documentation/streetview/policies), describing caching restrictions and the panorama-ID exception.
