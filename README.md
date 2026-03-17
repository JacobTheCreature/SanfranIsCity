# San Francisco

Data Mining course project based on the data sets from https://data.sfgov.org/

## Datasets

- [311 Cases: Needle-related cases after January 1, 2017](https://data.sfgov.org/City-Infrastructure/311-Cases-Needle-related-cases-after-January-1-201/7cgr-479f/about_data)
- [Quarterly count of tents, structures, and lived-in vehicles](https://data.sfgov.org/Housing-and-Buildings/Quarterly-count-of-tents-structures-and-lived-in-v/w9ip-yrij/about_data)
- [San Francisco Public Bathrooms and Water Fountains](https://data.sfgov.org/City-Infrastructure/San-Francisco-Public-Bathrooms-and-Water-Fountains/wfq4-upmv/about_data)

## Dependencies

```bash
pip install pandas numpy scikit-learn geopandas shapely scipy matplotlib seaborn folium plotly
```

## Introduction

This data mining project aims to discover spatial and temporal relationship between homeless encampments, public bathrooms, and exposed needles in San Francisco.

## Motivation

Having an understanding of the relationship between the aformentioned datasets could reveal helpful insights for identifying patterns, identifying concerning locations, concerning relationships, and more.

The homelessness and drug epidemic in San Francisco is real and has a devistating impact on its residents, visitors, and infastructure. So, I want to give light on a few questions that might help with understanding the problem:

- Is the concentration of exposed needles more concentrated around homeless encampments?
- Is the concentration of exposed needles more concentrated around public bathrooms?
- Whats the avverage distance between exposed needles and homeless encampments/public bathrooms and vice versa?
- How has the amount monthly amount of exposed needles changed alongside the monthly amount of homeless encampments?
- Is there an visually identifiable trend/pattern between the three data sets?
- etc

## Data Mining Techniques Applied

### 1. Preprocessing

- Removing invalid coordinates
- Standardizing time formats
- Standardizing column names
- Removing missing rows

### 2. Spatial Analysis

- Convert lat/lng to geometric points
- KDTree nearst neighbor searches
- Counting features within a 500 meter radius of facilities

### 3. Clustering

- K-means clustering
- DBSCAN to identify hotspots and outliers

### 4. Spatial and Temporal Relationship Analysis

- Time series decomposition into weekly, daily, and monthly aggregations
- Identify trends
- Identify patterns
- Year over year analysis
- Interactive maps and timeseries

### 5. Visualization

- Bar charts
- Line charts
- Distance histograms
- Heatmaps
- Seasonal plots

### 6. Correlation

- Distance to facilities vs needle density
- Proximity counts vs clustering patterns
- Temporal patterns vs spatial distribution

## Conclusions

Here are some interesting insights that emerge when viewing the statistical charts and visualizations:

- Many exposed needles are 60-200m from public bathrooms, bathrooms may attact needle incidents.
- Areas greater than 800m from pulbic bathrooms have fewer incidents than areas with public bathrooms.
- The time of day when exposed needles are reported are during the buisness hours, so when people are up and notice them more frequently
- Most exposed needles are left "open" and not resolved, the city is not removing them?
- Cluster 2 (Tenderloin/SoMa) has the highest density of exposed needles with 5,500+ exposed needles in the area
- Outer neighborhoods show significantly less needle incidents and have less public bathrooms

### Key Takeaways

- Areas with more public bathrooms have more needle incidents (less public bathrooms the lower the exposed needle rate)
- Most exposed needles are less than 100m from encampments
- However, some encampments have 0 exposed needles, perhaps location, population, avalible public bathrooms have an effect on homeless encampment drug use
- Only 3 neighborhoods account for the vast majority of exposed needles (Tenderloin, SoMa, Mission)
- The dense urban part of San Francisco has the majority of needle incidents while the residential areas havev significantly less
- Encampments with a high density of needles indicate inadequate disposal services, negative culture, and could be a sign of malicious individuals.