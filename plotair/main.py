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
import argparse
import csv
import glob
import logging
import os
import re
import shutil
import sys
import tomllib
from datetime import datetime
from pathlib import Path

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path so script can be called directly w/o 'python3 -m'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from plotair import __version__

CONFIG = {}

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    try:
        load_config()
    except FileNotFoundError as e:
        logger.error(f'Failed to load config: {e}')
        return

    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='+', metavar='FILE', 
                        help='sensor data file to process')
    parser.add_argument('-a', '--all-dates', action='store_true',
                        help='plot all dates, not only latest sequence')
    parser.add_argument('-v', '--version', action='version', 
                        version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Create a list containing all files from all patterns like '*.csv',
    # because under Windows the terminal doesn't expand wildcard arguments.
    all_files = []
    for pattern in args.filenames:
        all_files.extend(glob.glob(pattern))

    for filename in all_files:
        logger.info(f"Processing {filename}")
        try:
            sensor_type = detect_sensor_type(filename)

            if sensor_type == 'visiblair_d':
                df, valid, invalid = read_data_visiblair_d(filename)
            elif sensor_type == 'voc_co_form':
                df, valid, invalid = read_data_voc_co_form(filename)
            else:
                logger.error("Unsupported file format")
                return

            if invalid > 0:
                logger.info(f"{invalid} invalid row(s) ignored")

            if not args.all_dates:
                #log_data_frame(df, description = 'before deleting old data')
                df = delete_old_data(df)
                #log_data_frame(df, description = 'after deleting old data')

            if sensor_type == 'visiblair_d':
                generate_plot_co2_hum_tmp(df, filename)
            elif sensor_type == 'voc_co_form':
                generate_plot_hum_tmp(df, filename)
                generate_plot_voc_co_form(df, filename)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")


def detect_sensor_type(filename):
    sensor_type = None

    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        first_line = next(reader)
        num_fields = len(first_line)
        vis_min, vis_max = CONFIG['sensors']['visible_air_d_num_col']
        voc_min, voc_max = CONFIG['sensors']['voc_co_form_num_col']
        
        if voc_min <= num_fields <= voc_max:
            sensor_type = 'voc_co_form'
        elif vis_min <= num_fields <= vis_max:
            sensor_type = 'visiblair_d'
        
    #logger.debug(f"First line: {first_line}")
    #logger.debug(f"First line type: {type(first_line)}")
    logger.debug(f"Number of fields: {num_fields}")
    logger.debug(f"Sensor type: {sensor_type}")
    #sys.exit()
    
    return sensor_type


def read_data_visiblair_d(filename):
    valid_rows = []
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the file line by line instead of using pandas read_csv function.
    # This is less concise but allows for more control over data validation.
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split(',')
            vis_min, vis_max = CONFIG['sensors']['visible_air_d_num_col']
            
            if not (vis_min <= len(fields) <= vis_max):
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


def read_data_voc_co_form(filename):
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Rename the columns
    df.columns = ['date', 'tvoc', 'co', 'form', 'humidity', 'temperature', 'filename']

    # Convert the 'date' column to pandas datetime objects
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y %I:%M:%S %p')

    # Convert 'form' column to string, replace '< LOD' with '0', and then convert to integer
    df['form'] = df['form'].astype(str).str.replace('< LOD', '0').astype(int)

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

                if (current_interval / sampling_interval) > CONFIG['data']['max_missing_samples']:
                    # This sample is from older sequence, keep only more recent
                    df = df[df.index >= next_date]
                    break
        
        next_date = current_date
        
    return df
    

def generate_plot_co2_hum_tmp(df, filename):
    # The dates must be in a non-index column
    df = df.reset_index()

    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=CONFIG['plot']['font_scale'])

    # Set the font to override the OS-dependent default. Noto is installed by
    # default on both Linux and Windows, but not on macOS.
    plt.rcParams['font.family'] = 'Noto Sans'

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=CONFIG['plot']['size'])
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot the data series
    sns.lineplot(data=df, x='date', y='co2', ax=ax1, color=CONFIG['colors']['co2'],
                 label=CONFIG['labels']['co2'], legend=False)
    sns.lineplot(data=df, x='date', y='humidity', ax=ax2, color=CONFIG['colors']['humidity'],
                 label=CONFIG['labels']['humidity'], legend=False)
    sns.lineplot(data=df, x='date', y='temperature', ax=ax2, color=CONFIG['colors']['temp'],
                 label=CONFIG['labels']['temp'], legend=False)

    # Set the ranges for both y axes
    cmin, cmax = CONFIG['axis_ranges']['co2']
    tmin, tmax = CONFIG['axis_ranges']['temp_humidity']
    ax1.set_ylim(cmin, cmax)  # df['co2'].max() * 1.05
    ax2.set_ylim(tmin, tmax)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=0.7)  
    #ax1.grid(axis='y', alpha=0.7)
    ax2.grid(axis='y', alpha=0.7, linestyle='dashed')

    # Set the background color of the humidity comfort zone
    hmin, hmax = CONFIG['humidity_zone']['range']
    ax2.axhspan(ymin=hmin, ymax=hmax,
                facecolor=CONFIG['colors']['humidity'], alpha=CONFIG['humidity_zone']['opacity'])

    # Customize the plot title, labels and ticks
    co2_label_font = ax1.yaxis.label.get_fontproperties().get_name()
    logger.debug(f"Font used for CO₂ label: {co2_label_font}")

    ax1.set_title(get_plot_title(filename))
    ax1.tick_params(axis='x', rotation=CONFIG['labels']['date_rotation'])
    ax1.tick_params(axis='y', labelcolor=CONFIG['colors']['co2'])
    ax1.set_xlabel('')
    ax1.set_ylabel(CONFIG['labels']['co2'], color=CONFIG['colors']['co2'])
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    x = 1.07  # Slightly to the right of the axis
    y = 0.5   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, CONFIG['labels']['temp'] + '  ', transform=ax2.transAxes,
             color=CONFIG['colors']['temp'], rotation='vertical', ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, '  ' + CONFIG['labels']['humidity'], transform=ax2.transAxes,
            color=CONFIG['colors']['humidity'], rotation='vertical', ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(get_png_filename(filename, '-cht'))
    plt.close()


def generate_plot_hum_tmp(df, filename):
    # The dates must be in a non-index column
    df = df.reset_index()

    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=CONFIG['plot']['font_scale'])

    # Set the font to override the OS-dependent default. Noto is installed by
    # default on both Linux and Windows, but not on macOS.
    plt.rcParams['font.family'] = 'Noto Sans'

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=CONFIG['plot']['size'])
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot the data series
    #sns.lineplot(data=df, x='date', y='co2', ax=ax1, color=CONFIG['colors']['co2'],
    #             label=CONFIG['labels']['co2'], legend=False)
    sns.lineplot(data=df, x='date', y='humidity', ax=ax2, color=CONFIG['colors']['humidity'],
                 label=CONFIG['labels']['humidity'], legend=False)
    sns.lineplot(data=df, x='date', y='temperature', ax=ax2, color=CONFIG['colors']['temp'],
                 label=CONFIG['labels']['temp'], legend=False)

    # Set the ranges for both y axes
    cmin, cmax = CONFIG['axis_ranges']['co2']
    tmin, tmax = CONFIG['axis_ranges']['temp_humidity']
    ax1.set_ylim(cmin, cmax)  # df['co2'].max() * 1.05
    ax2.set_ylim(tmin, tmax)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=0.7)  
    #ax1.grid(axis='y', alpha=0.7)
    ax2.grid(axis='y', alpha=0.7, linestyle='dashed')

    # Set the background color of the humidity comfort zone
    hmin, hmax = CONFIG['humidity_zone']['range']
    ax2.axhspan(ymin=hmin, ymax=hmax,
                facecolor=CONFIG['colors']['humidity'], alpha=CONFIG['humidity_zone']['opacity'])

    # Customize the plot title, labels and ticks
    ax1.set_title(get_plot_title(filename))
    ax1.tick_params(axis='x', rotation=CONFIG['labels']['date_rotation'])
    #ax1.tick_params(axis='y', labelcolor=CONFIG['colors']['co2'])
    ax1.set_xlabel('')
    #ax1.set_ylabel(CONFIG['labels']['co2'], color=CONFIG['colors']['co2'])
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    x = 1.07  # Slightly to the right of the axis
    y = 0.5   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, CONFIG['labels']['temp'] + '  ', transform=ax2.transAxes,
             color=CONFIG['colors']['temp'], rotation='vertical', ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, '  ' + CONFIG['labels']['humidity'], transform=ax2.transAxes,
            color=CONFIG['colors']['humidity'], rotation='vertical', ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Remove the left y-axis elements from ax1
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', left=False, labelleft=False)

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(get_png_filename(filename, '-ht'))
    plt.close()


def generate_plot_voc_co_form(df, filename):
    # The dates must be in a non-index column
    df = df.reset_index()
    df['co_scaled'] = df['co'] * 10

    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=CONFIG['plot']['font_scale'])

    # Set the font to override the OS-dependent default. Noto is installed by
    # default on both Linux and Windows, but not on macOS.
    plt.rcParams['font.family'] = 'Noto Sans'

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=CONFIG['plot']['size'])
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot the data series
    # Filter the DataFrame to only include rows where 'form' is not zero
    df_filtered = df[df['form'] != 0]

    sns.lineplot(data=df, x='date', y='tvoc', ax=ax1, color=CONFIG['colors']['tvoc'],
                 label=CONFIG['labels']['tvoc'], legend=False)
    sns.lineplot(data=df, x='date', y='co_scaled', ax=ax2, color=CONFIG['colors']['co'],
                 label=CONFIG['labels']['co'], legend=False)
    sns.lineplot(data=df_filtered, x='date', y='form', ax=ax2, color=CONFIG['colors']['form'],
                 label=CONFIG['labels']['form'], legend=False)

    # Set the ranges for both y axes
    tmin, tmax = CONFIG['axis_ranges']['tvoc']
    cmin, cmax = CONFIG['axis_ranges']['co_form']
    ax1.set_ylim(tmin, tmax)
    ax2.set_ylim(cmin, cmax)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=0.7)  
    #ax1.grid(axis='y', alpha=0.7)
    ax2.grid(axis='y', alpha=0.7, linestyle='dashed')

    # Set the background color of the humidity comfort zone
    #hmin, hmax = CONFIG['humidity_zone']['range']
    #ax2.axhspan(ymin=hmin, ymax=hmax,
    #            facecolor=CONFIG['colors']['humidity'], alpha=CONFIG['humidity_zone']['opacity'])

    # Customize the plot title, labels and ticks
    ax1.set_title(get_plot_title(filename))
    ax1.tick_params(axis='x', rotation=CONFIG['labels']['date_rotation'])
    ax1.tick_params(axis='y', labelcolor=CONFIG['colors']['tvoc'])
    ax1.set_xlabel('')
    ax1.set_ylabel(CONFIG['labels']['tvoc'], color=CONFIG['colors']['tvoc'])
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    x = 1.07  # Slightly to the right of the axis
    y = 0.5   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, CONFIG['labels']['co'] + '  ', transform=ax2.transAxes,
             color=CONFIG['colors']['co'], rotation='vertical', ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, '  ' + CONFIG['labels']['form'], transform=ax2.transAxes,
            color=CONFIG['colors']['form'], rotation='vertical', ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(get_png_filename(filename, '-vcf'))
    plt.close()


def load_config():
    global CONFIG

    app_name = 'plotair'
    config_file = 'config.toml'

    config_dir = Path.home() / f'.{app_name}'
    config_dir.mkdir(parents=True, exist_ok=True)

    user_config_file = config_dir / config_file
    default_config_file = PROJECT_ROOT / config_file

    if not user_config_file.exists():
        logger.info(f'Config file not found at {user_config_file}')

        if default_config_file.exists():
            shutil.copy2(default_config_file, user_config_file)
            logger.info(f'Config initialized at {user_config_file}')
        else:
            raise FileNotFoundError(f'Default config missing at {default_config_file}')
    else:
        logger.info(f'Found config file at {user_config_file}')

    with open(user_config_file, "rb") as f:
        CONFIG = tomllib.load(f)


def get_plot_title(filename):
    match = re.search(r'^(\d+\s*-\s*)?(.*)\.[a-zA-Z]+$', filename)
    plot_title = match.group(2) if match else filename

    # Capitalize only the first character
    if plot_title: plot_title = plot_title[0].upper() + plot_title[1:]

    return plot_title


def get_png_filename(filename, suffix = ''):
    root, ext = os.path.splitext(filename)
    return f"{root}{suffix}.png"


def log_data_frame(df, description = ''):
    """ This function is used only for debugging. """
    logger.debug(f'DataFrame {description}\n{df}')
    #logger.debug(f'DataFrame index data type: {df.index.dtype}')
    #logger.debug(f'DataFrame index class: {type(df.index)}')
    #logger.debug(f'DataFrame columns data types\n{df.dtypes}')
    logger.debug(f'DataFrame statistics\n{df.describe()}')  # Mean, min, max...
    #sys.exit()


if __name__ == '__main__':
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s - %(message)s')
    
    # Configure this script's logger
    logger.setLevel(logging.DEBUG)

    main()
