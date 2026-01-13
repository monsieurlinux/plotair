#!/usr/bin/env python3

"""
Generate CO₂, humidity and temperature plots from VisiblAir sensor CSV files.

This script processes one or more CSV files containing VisiblAir sensor data.
For each file, it reads the data into a pandas DataFrame, ignores incorrectly
formatted lines, keeps only the most recent data sequence, and generates a
Seaborn plot saved as a PNG file with the same base name as the input CSV.

Copyright (c) 2026 Monsieur Linux

Licensed under the MIT License. See the LICENSE file for details.
"""

# Standard library imports
from datetime import datetime
import logging
import os
import re
import shutil
import sys

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration constants
MIN_CSV_COLUMNS = 5  # Most rows have 5 columns
MAX_CSV_COLUMNS = 6  # Some rows have 6 columns
MAX_MISSING_SAMPLES = 4
PLOT_FONT_SCALE = 1.4
PLOT_WIDTH = 11
PLOT_HEIGHT = 8.5
CO2_LABEL = 'CO₂ (ppm)'
HUMIDITY_LABEL = 'Humidité (%)'
TEMP_LABEL = 'Température (°C)'
Y1_AXIS_LABEL = 'CO₂ (ppm)'
Y2_AXIS_LABEL_1 = 'Température (°C)' + '  '  # Spaces separate from humidity
Y2_AXIS_LABEL_2 = '  ' + 'Humidité (%)'      # Spaces separate from temperature
X_AXIS_ROTATION = 30
HUMIDITY_ZONE_MIN = 40
HUMIDITY_ZONE_MAX = 60
HUMIDITY_ZONE_ALPHA = 0.075

# 8000/4000/2000/1600 span for y1 axis work well with 80 span for y2 axis
# 6000/3000/    /1200 span for y1 axis work well with 60 span for y2 axis
Y1_AXIS_MIN_VALUE = 0
Y1_AXIS_MAX_VALUE = 1200
Y2_AXIS_MIN_VALUE = 10
Y2_AXIS_MAX_VALUE = 70

# See Matplotlib documentation for valid colors:
# https://matplotlib.org/stable/gallery/color/named_colors.html
CO2_COLOR = 'tab:blue'
HUMIDITY_COLOR = 'tab:green'
TEMP_COLOR = 'tab:orange'

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    # sys.argv[0] is the script name, so arguments start from index 1
    if len(sys.argv) < 2:
        logger.error("No files were provided")
        print(f"Usage: [python] {sys.argv[0]} <file1> <file2> ...")
    else:
        for filename in sys.argv[1:]:
            logger.info(f"Processing {filename}")
            try:
                df, valid, invalid = read_csv_data(filename)
                if invalid > 0:
                    logger.info(f"{invalid} invalid row(s) ignored")
                df = delete_old_data(df)
                generate_plot(df, filename)
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")


def read_csv_data(filename):
    valid_rows = []
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the file line by line instead of using pandas read_csv function.
    # This is less concise but allows for more control over data validation.
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split(',')
            
            if len(fields) < MIN_CSV_COLUMNS or len(fields) > MAX_CSV_COLUMNS:
                # Skip lines with an invalid number of columns
                logger.debug(f"Skipping line (number of columns): {line}")
                num_invalid_rows += 1
                continue
                
            try:
                # Convert each field to its target data type
                parsed_row = {
                    'date': pd.to_datetime(fields[0], format='%Y-%m-%d %H:%M:%S'),
                    'co2': np.uint16(fields[1]),           # 0 to 10,000 ppm
                    'temperature': np.float32(fields[2]),  # -40 to 70 °C
                    'humidity': np.uint8(fields[3])        # 0 to 100% RH
                }
                # If conversion succeeds, add the parsed row to the list
                num_valid_rows += 1
                valid_rows.append(parsed_row)
                
            except (ValueError, TypeError) as e:
                # Skip lines with conversion errors
                logger.debug(f"Skipping line (conversion error): {line}")
                num_invalid_rows += 1
                continue

        # Create the DataFrame from the valid rows
        df = pd.DataFrame(valid_rows)
        df = df.set_index('date')
        df = df.sort_index()  # Sort in case some dates are not in order

    return df, num_valid_rows, num_invalid_rows


def delete_old_data(df):
    """
    Iterate backwards through the samples to find the first time gap larger
    than the sampling interval. Then return only the latest data sequence.
    """
    sampling_interval = None
    next_date = df.index[-1]

    for date in reversed(list(df.index)):
        current_date = date

        if current_date != next_date:
            if sampling_interval is None:
                sampling_interval = next_date - current_date
            else:
                current_interval = next_date - current_date

                if (current_interval / sampling_interval) > MAX_MISSING_SAMPLES:
                    # This sample is from older sequence, keep only more recent
                    df = df[df.index >= next_date]
                    break
        
        next_date = current_date
        
    return df
    

def generate_plot(df, filename):
    # The dates must be in a non-index column
    df = df.reset_index()

    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=PLOT_FONT_SCALE)

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot the data series
    sns.lineplot(data=df, x='date', y='co2', ax=ax1, color=CO2_COLOR,
                 label=CO2_LABEL, legend=False)
    sns.lineplot(data=df, x='date', y='humidity', ax=ax2, color=HUMIDITY_COLOR,
                 label=HUMIDITY_LABEL, legend=False)
    sns.lineplot(data=df, x='date', y='temperature', ax=ax2, color=TEMP_COLOR,
                 label=TEMP_LABEL, legend=False)

    # Set the ranges for both y axes
    ax1.set_ylim(Y1_AXIS_MIN_VALUE, Y1_AXIS_MAX_VALUE)  # df['co2'].max() * 1.05
    ax2.set_ylim(Y2_AXIS_MIN_VALUE, Y2_AXIS_MAX_VALUE)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=0.7)  
    #ax1.grid(axis='y', alpha=0.7)
    ax2.grid(axis='y', alpha=0.7, linestyle='dashed')

    # Set the background color of the humidity comfort zone
    ax2.axhspan(ymin=HUMIDITY_ZONE_MIN, ymax=HUMIDITY_ZONE_MAX,
                facecolor=HUMIDITY_COLOR, alpha=HUMIDITY_ZONE_ALPHA)

    # Customize the plot title, labels and ticks
    ax1.set_title(get_plot_title(filename))
    ax1.tick_params(axis='x', rotation=X_AXIS_ROTATION)
    ax1.tick_params(axis='y', labelcolor=CO2_COLOR)
    ax1.set_xlabel('')
    ax1.set_ylabel(Y1_AXIS_LABEL, color=CO2_COLOR)
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    x = 1.07  # Slightly to the right of the axis
    y = 0.5   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, Y2_AXIS_LABEL_1, transform=ax2.transAxes,
             color=TEMP_COLOR, rotation='vertical', ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, Y2_AXIS_LABEL_2, transform=ax2.transAxes,
            color=HUMIDITY_COLOR, rotation='vertical', ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(get_png_filename(filename))
    plt.close()


def get_plot_title(filename):
    match = re.search(r'^(\d+\s*-\s*)?(.*)\.[a-zA-Z]+$', filename)
    plot_title = match.group(2) if match else filename
    plot_title = plot_title.capitalize()
    return plot_title


def get_png_filename(filename):
    root, ext = os.path.splitext(filename)
    return f"{root}.png"


if __name__ == '__main__':
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s - %(message)s')
    
    # Configure this script's logger
    logger.setLevel(logging.INFO)

    main()
