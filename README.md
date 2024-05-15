# GNSS-Raw-Mesurments
This repository contains  code for amateur  (Global Navigation Satellite System) data analysis. The code is designed to decode raw satellite  data collected from a smartphone, calculate satellite positions and user location, and export the results for further analysis or visualization.

# Requirements
Python 3.x (Anaconda or miniconda recommended)
Python Libraries: pandas, numpy, matplotlib, navpy, gnssutils, simplekml

# Run command
python task_0.py

# Usage
Clone or download the code repository to your local machine using git clone or directly.
place GNSS data file in the same directory as the code.
Navigate to the project directory in your terminal.
enter the path to the Gnss data
Run the main script: python task_0.py

# Features
## Data Parsing: 
Reads GNSS data from CSV files, extracting relevant information like Android fixes and GNSS measurements.

## Satellite Positioning: 
Calculates the positions of orbiting satellites based on ephemeris data, timestamps, and GNSS measurements for each epoch (group of measurements).

## User Location Estimation:
Utilizes a least squares estimation algorithm to determine the user's position and clock bias using a weighted set of satellite pseudo-ranges.
## Coordinate Conversion:
Transforms Earth-centered, Earth-fixed (ECEF) coordinates into intuitive latitude, longitude, and altitude (LLA) formats.

## KML Generation: 
Creates KML files containing the user's location path, allowing for visualization on mapping applications like Google Earth.

# Output

## CSV Files:
These files contain the original GNSS data with additional columns for:
Pos.X, Pos.Y, Pos.Z (X, Y, Z coordinates)
Lat, Lon, Alt (Latitude, Longitude, Altitude).

## KML Files:
These files represent the calculated path on a map. You can open them with a KML viewer like Google Earth.

