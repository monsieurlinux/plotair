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
    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='+', metavar='FILE',
                        help='sensor data file to process')
    parser.add_argument('-a', '--all-dates', action='store_true',
                        help='plot all dates (otherwise only latest sequence)')
    parser.add_argument('-m', '--merge', metavar='FIELD',
                        help='merge field from file1 to file2, and output to file3')
    parser.add_argument('-r', '--reset-config', action='store_true',
                        help='reset configuration file to default')
    parser.add_argument('-s', '--start-date', metavar='DATE',
                        help='date at which to start the plot (YYYY-MM-DD)')
    parser.add_argument('-S', '--stop-date', metavar='DATE',
                        help='date at which to stop the plot (YYYY-MM-DD)')
    parser.add_argument('-t', '--title',
                        help='set the plot title')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    try:
        load_config(args.reset_config)
    except FileNotFoundError as e:
        logger.error(f'Failed to load config: {e}')
        return

    if args.merge:
        field = args.merge
        num_files = len(args.filenames)

        if num_files != 3:
            logger.error('argument -m/--merge requires three file arguments')
            return

        file_format, df1, num_valid_rows1, num_invalid_rows = read_data(args.filenames[0])
        file_format, df2, num_valid_rows2, num_invalid_rows = read_data(args.filenames[1])

        if num_valid_rows1 <= 0 or num_valid_rows2 <= 0:
            logger.error('At least one of the input files is unsupported')
            return

        temp_df = df1[['co2']]
        df2 = pd.concat([df2, temp_df]).sort_index()
        df2.to_csv(args.filenames[2], index=True)

    else:
        # Create a list containing all files from all patterns like '*.csv',
        # because under Windows the terminal doesn't expand wildcard arguments.
        all_files = []
        for pattern in args.filenames:
            all_files.extend(glob.glob(pattern))

        for filename in all_files:
            logger.info(f'Processing {filename}')
            try:
                file_format, df, num_valid_rows, num_invalid_rows = read_data(filename)

                if num_valid_rows > 0:
                    logger.debug(f'{num_valid_rows} row(s) read')
                else:
                    logger.error('Unsupported file format')
                    return

                if num_invalid_rows > 0:
                    logger.info(f'{num_invalid_rows} invalid row(s) ignored')

                if not args.all_dates:
                    df = delete_old_data(df, args.start_date, args.stop_date)

                generate_stats(df, filename)

                if file_format == 'plotair':
                    generate_plot_co2_hum_tmp(df, filename, args.title)
                elif file_format == 'visiblair_d':
                    # TODO: we could send only the name and let generate_plot() get the other variables
                    ds1 = DataSeries(name='co2',
                                     label=CONFIG['labels']['co2'],
                                     color=CONFIG['colors']['co2'],
                                     y_range=CONFIG['axis_ranges']['co2'],
                                     limit=None)
                    ds2 = DataSeries(name='humidity',
                                     label=CONFIG['labels']['humidity'],
                                     color=CONFIG['colors']['humidity'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=CONFIG['limits']['humidity'])
                    ds3 = DataSeries(name='temp',
                                     label=CONFIG['labels']['temp'],
                                     color=CONFIG['colors']['temp'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=None)
                    generate_plot(df, filename, args.title, 'cht', ds1, ds2, ds3)
                elif file_format == 'visiblair_e':
                    ds1 = DataSeries(name='co2',
                                     label=CONFIG['labels']['co2'],
                                     color=CONFIG['colors']['co2'],
                                     y_range=CONFIG['axis_ranges']['co2'],
                                     limit=None)
                    ds2 = DataSeries(name='humidity',
                                     label=CONFIG['labels']['humidity'],
                                     color=CONFIG['colors']['humidity'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=CONFIG['limits']['humidity'])
                    ds3 = DataSeries(name='temp',
                                     label=CONFIG['labels']['temp'],
                                     color=CONFIG['colors']['temp'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=None)
                    generate_plot(df, filename, args.title, 'cht', ds1, ds2, ds3)

                    ds2 = DataSeries(name='pm2.5',
                                     label=CONFIG['labels']['pm2.5'],
                                     color=CONFIG['colors']['pm2.5'],
                                     y_range=CONFIG['axis_ranges']['pm2.5_10'],
                                     limit=CONFIG['limits']['pm2.5'],
                                     limit_label=CONFIG['labels']['pm2.5_limit'],
                                     linewidth=CONFIG['plot']['pm2.5_line_width'],
                                     linestyle=CONFIG['plot']['pm2.5_line_style'])
                    ds3 = DataSeries(name='pm10',
                                     label=CONFIG['labels']['pm10'],
                                     color=CONFIG['colors']['pm10'],
                                     y_range=CONFIG['axis_ranges']['pm2.5_10'],
                                     limit=CONFIG['limits']['pm10'],
                                     limit_label=CONFIG['labels']['pm10_limit'],
                                     linewidth=CONFIG['plot']['pm10_line_width'],
                                     linestyle=CONFIG['plot']['pm10_line_style'])
                    generate_plot(df, filename, args.title, 'pm', ds1=None, ds2=ds2, ds3=ds3)
                elif file_format == 'graywolf_ds':
                    ds2 = DataSeries(name='humidity',
                                     label=CONFIG['labels']['humidity'],
                                     color=CONFIG['colors']['humidity'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=CONFIG['limits']['humidity'])
                    ds3 = DataSeries(name='temp',
                                     label=CONFIG['labels']['temp'],
                                     color=CONFIG['colors']['temp'],
                                     y_range=CONFIG['axis_ranges']['temp_h'],
                                     limit=None)
                    generate_plot(df, filename, args.title, 'ht', ds1=None, ds2=ds2, ds3=ds3)
                    #generate_plot_voc_co_form(df, filename, args.title)
            except Exception as e:
                logger.exception(f'Unexpected error: {e}')


def detect_file_format(filename):
    file_format = None
    visiblair_d_num_col = (5, 6)  # Most rows have 5 columns but some have 6
    visiblair_e_num_col = (21, 21)
    graywolf_ds_num_col = (7, 7)

    # Some files begin with the '\ufeff' character (Byte Order Mark / BOM).
    # This breaks the first field detection. Use 'utf-8-sig' instead of 'utf-8'
    # to automatically handle BOM.
    with open(filename, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        first_line = next(reader)
        num_fields = len(first_line)

        if first_line[0] == 'date':
            file_format = 'plotair'
        elif visiblair_d_num_col[0] <= num_fields <= visiblair_d_num_col[1]:
            file_format = 'visiblair_d'
        elif (visiblair_e_num_col[0] <= num_fields <= visiblair_e_num_col[1] and
              first_line[1] == 'Timestamp'):
            file_format = 'visiblair_e'
        elif (graywolf_ds_num_col[0] <= num_fields <= graywolf_ds_num_col[1] and
              first_line[0] == 'Date Time'):
            file_format = 'graywolf_ds'
        
    logger.debug(f'File format: {file_format}')
    
    return file_format


def read_data(filename):
    file_format = detect_file_format(filename)
    df = pd.DataFrame()
    num_valid_rows = 0
    num_invalid_rows = 0

    if file_format == 'plotair':
        df, num_valid_rows, num_invalid_rows = read_data_plotair(filename)
    elif file_format == 'visiblair_d':
        df, num_valid_rows, num_invalid_rows = read_data_visiblair_d(filename)
    elif file_format == 'visiblair_e':
        df, num_valid_rows, num_invalid_rows = read_data_visiblair_e(filename)
    elif file_format == 'graywolf_ds':
        df, num_valid_rows, num_invalid_rows = read_data_graywolf_ds(filename)

    return file_format, df, num_valid_rows, num_invalid_rows


def read_data_plotair(filename):
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Convert the 'date' column to pandas datetime objects
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    df = df.set_index('date')
    df = df.sort_index()  # Sort in case some dates are not in order
    num_valid_rows = len(df)

    return df, num_valid_rows, num_invalid_rows


def read_data_visiblair_d(filename):
    df = pd.DataFrame()
    num_valid_rows = 0
    num_invalid_rows = 0
    valid_rows = []

    # Read the file line by line instead of using pandas read_csv function.
    # This is less concise but allows for more control over data validation.
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split(',')
            
            if not (5 <= len(fields) <= 6):
                # Skip lines with an invalid number of columns
                logger.debug(f'Skipping line (number of columns): {line}')
                num_invalid_rows += 1
                continue
                
            try:
                # Convert each field to its target data type
                parsed_row = {
                    'date': pd.to_datetime(fields[0], format='%Y-%m-%d %H:%M:%S'),
                    'co2': np.uint16(fields[1]),     # 0 to 10,000 ppm
                    'temp': np.float32(fields[2]),   # -40 to 70 °C
                    'humidity': np.uint8(fields[3])  # 0 to 100% RH
                }
                # If conversion succeeds, add the parsed row to the list
                valid_rows.append(parsed_row)
                
            except (ValueError, TypeError) as e:
                # Skip lines with conversion errors
                logger.debug(f'Skipping line (conversion error): {line}')
                num_invalid_rows += 1
                continue

        # Create the DataFrame from the valid rows
        df = pd.DataFrame(valid_rows)
        df = df.set_index('date')
        df = df.sort_index()  # Sort in case some dates are not in order
        num_valid_rows = len(df)

    return df, num_valid_rows, num_invalid_rows


def read_data_visiblair_e(filename):
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Rename the columns
    df.columns = ['uuid', 'date', 'co2', 'humidity', 'temp', 'pm0.1',
                  'pm0.3', 'pm0.5', 'pm1', 'pm2.5', 'pm5', 'pm10', 'pressure',
                  'voc_index', 'firmware', 'model', 'pcb', 'display_rate',
                  'is_charging', 'is_ac_in', 'batt_voltage']

    # Convert the 'date' column to pandas datetime objects
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    df = df.set_index('date')
    df = df.sort_index()  # Sort in case some dates are not in order
    num_valid_rows = len(df)

    return df, num_valid_rows, num_invalid_rows


def read_data_graywolf_ds(filename):
    num_valid_rows = 0
    num_invalid_rows = 0

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Rename the columns
    df.columns = ['date', 'tvoc', 'co', 'form', 'humidity', 'temp', 'filename']

    # Convert the 'date' column to pandas datetime objects
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y %I:%M:%S %p')

    # Convert 'form' column to string, replace '< LOD' with '0', and then convert to integer
    df['form'] = df['form'].astype(str).str.replace('< LOD', '0').astype(int)

    df = df.set_index('date')
    df = df.sort_index()  # Sort in case some dates are not in order
    num_valid_rows = len(df)

    return df, num_valid_rows, num_invalid_rows


def delete_old_data(df, start_date = None, stop_date = None):
    if not start_date and not stop_date:
        # Iterate backwards through the samples to find the first time gap larger
        # than the sampling interval. Then return only the latest data sequence.
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

    else:
        # Keep only the data range to be plotted (use pandas dates types)
        if start_date:
            sd = pd.Timestamp(start_date)
            df = df[df.index >= sd]
        if stop_date:
            sd = pd.Timestamp(stop_date)
            df = df[df.index <= sd]

    return df
    

class DataSeries:
    def __init__(self, name='', label='', color='tab:blue', y_range=None, limit=None, limit_label=None, linewidth=None, linestyle=None):
        # y_range could be replaced by y_min and y_max
        self.name = name
        self.label = label
        self.color = color
        self.y_range = y_range  # min/max tuple, e.g. (0, 100)
        self.limit = limit      # single value or min/max tuple
        self.limit_label = limit_label
        self.linewidth = linewidth
        self.linestyle = linestyle


def generate_plot(df, filename, title, suffix, ds1=None, ds2=None, ds3=None):
    # The dates must be in a non-index column
    df = df.reset_index()

    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=CONFIG['plot']['font_scale'])

    ff = CONFIG['plot']['font_family']
    if ff != '': plt.rcParams['font.family'] = ff

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=CONFIG['plot']['size'])
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot series #1 main line
    if ds1:
        sns.lineplot(data=df, x='date', y=ds1.name, ax=ax1, color=ds1.color,
                     label=ds1.label, legend=False)

    # Plot series #2 main line
    if ds2.linewidth:
        linewidth = ds2.linewidth
    else:
        linewidth = CONFIG['plot']['default_line_width']

    if ds2.linestyle:
        linestyle = ds2.linestyle
    else:
        linestyle = CONFIG['plot']['default_line_style']

    sns.lineplot(data=df, x='date', y=ds2.name, ax=ax2, color=ds2.color,
                 label=ds2.label, legend=False,
                 linewidth=linewidth, linestyle=linestyle)

    # Display series #2 limit line or zone
    if ds2.limit and not isinstance(ds2.limit, list):
        # Plot the limit line
        line = ax2.axhline(y=ds2.limit, color=ds2.color, label=ds2.limit_label,
                           linewidth=CONFIG['plot']['limit_line_width'],
                           linestyle=CONFIG['plot']['limit_line_style'])
        line.set_alpha(CONFIG['plot']['limit_line_opacity'])

    if ds2.limit and isinstance(ds2.limit, list):
        # Set the background color of the limit zone
        hmin, hmax = ds2.limit
        ax2.axhspan(ymin=hmin, ymax=hmax, facecolor=ds2.color,
                    alpha=CONFIG['plot']['limit_zone_opacity'])

    # Plot series #3 main line
    if ds3.linewidth:
        linewidth = ds3.linewidth
    else:
        linewidth = CONFIG['plot']['default_line_width']

    if ds3.linestyle:
        linestyle = ds3.linestyle
    else:
        linestyle = CONFIG['plot']['default_line_style']

    sns.lineplot(data=df, x='date', y=ds3.name, ax=ax2, color=ds3.color,
                 label=ds3.label, legend=False,
                 linewidth=linewidth, linestyle=linestyle)

    # Plot series #3 limit line
    if ds3.limit and not isinstance(ds3.limit, list):
        # Plot the limit line
        line = ax2.axhline(y=ds3.limit, color=ds3.color, label=ds3.limit_label,
                           linewidth=CONFIG['plot']['limit_line_width'],
                           linestyle=CONFIG['plot']['limit_line_style'])
        line.set_alpha(CONFIG['plot']['limit_line_opacity'])

    if ds3.limit and isinstance(ds3.limit, list):
        # Set the background color of the limit zone
        hmin, hmax = ds3.limit
        ax2.axhspan(ymin=hmin, ymax=hmax, facecolor=ds3.color,
                    alpha=CONFIG['plot']['limit_zone_opacity'])

    # Set the ranges for both y axes
    if ds1:
        cmin, cmax = ds1.y_range  # TODO: can we feed this directly to set_ylim()?
        ax1.set_ylim(cmin, cmax)  # df['co2'].max() * 1.05

    tmin, tmax = ds2.y_range
    ax2.set_ylim(tmin, tmax)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=CONFIG['plot']['grid_opacity'])
    #ax1.grid(axis='y', alpha=CONFIG['plot']['grid_opacity'])
    ax2.grid(axis='y', alpha=CONFIG['plot']['grid2_opacity'], linestyle=CONFIG['plot']['grid2_line_style'])

    # Customize the plot title, labels and ticks
    ax1.set_title(get_plot_title(title, filename))
    ax1.tick_params(axis='x', rotation=CONFIG['plot']['date_rotation'])
    if ds1:
        ax1.tick_params(axis='y', labelcolor=ds1.color)
        ax1.set_ylabel(ds1.label, color=ds1.color)
    ax1.set_xlabel('')
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    bottom_label = ds3.label + '  '
    top_label = '  ' + ds2.label
    x = 1.07  # Slightly to the right of the axis
    y = get_label_center(bottom_label, top_label)   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, bottom_label, transform=ax2.transAxes,
             color=ds3.color, rotation='vertical',
             ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, top_label, transform=ax2.transAxes,
            color=ds2.color, rotation='vertical',
            ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc=CONFIG['plot']['legend_location'])

    if not ds1:
        # Remove the left y-axis elements from ax1
        ax1.grid(axis='y', visible=False)
        ax1.spines['left'].set_visible(False)
        ax1.tick_params(axis='y', left=False, labelleft=False)

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    # TODO: build to plot suffix from the 1st char of each series
    plt.savefig(get_plot_filename(filename, f'-{suffix}'))
    plt.close()


def generate_plot_voc_co_form(df, filename, title):
    # The dates must be in a non-index column
    df = df.reset_index()
    
    # Set a theme and scale all fonts
    sns.set_theme(style='whitegrid', font_scale=CONFIG['plot']['font_scale'])

    ff = CONFIG['plot']['font_family']
    if ff != '': plt.rcParams['font.family'] = ff

    # Set up the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=CONFIG['plot']['size'])
    ax2 = ax1.twinx()  # Secondary y axis

    # Plot the TVOC main line
    sns.lineplot(data=df, x='date', y='tvoc', ax=ax1, legend=False,
                 color=CONFIG['colors']['tvoc'], label=CONFIG['labels']['tvoc'])

    # Plot the TVOC limit line
    line = ax1.axhline(y=CONFIG['limits']['tvoc'], color=CONFIG['colors']['tvoc'],
                linestyle=CONFIG['plot']['limit_line_style'], linewidth=CONFIG['plot']['limit_line_width'],
                label=CONFIG['labels']['tvoc_limit'])
    line.set_alpha(CONFIG['plot']['limit_line_opacity'])

    # Plot the formaldehyde main line
    df_filtered = df[df['form'] != 0]  # Filter out rows where 'form' is zero
    sns.lineplot(data=df_filtered, x='date', y='form', ax=ax2, legend=False,
                 color=CONFIG['colors']['form'], label=CONFIG['labels']['form'])

    # Plot the formaldehyde limit line
    line = ax2.axhline(y=CONFIG['limits']['form'], color=CONFIG['colors']['form'],
                linestyle=CONFIG['plot']['limit_line_style'], linewidth=CONFIG['plot']['limit_line_width'],
                label=CONFIG['labels']['form_limit'])
    line.set_alpha(CONFIG['plot']['limit_line_opacity'])

    # Plot the CO main line
    co_scale = 10
    df['co_scaled'] = df['co'] * co_scale
    sns.lineplot(data=df, x='date', y='co_scaled', ax=ax2, legend=False,
                 color=CONFIG['colors']['co'], label=CONFIG['labels']['co'])

    # Plot the CO limit line
    line = ax2.axhline(y=CONFIG['limits']['co'] * co_scale, color=CONFIG['colors']['co'],
                linestyle=CONFIG['plot']['limit_line_style'], linewidth=CONFIG['plot']['limit_line_width'],
                label=CONFIG['labels']['co_limit'])
    line.set_alpha(CONFIG['plot']['limit_line_opacity'])

    # Set the ranges for both y axes
    tmin, tmax = CONFIG['axis_ranges']['tvoc']
    cmin, cmax = CONFIG['axis_ranges']['co_form']
    ax1.set_ylim(tmin, tmax)
    ax2.set_ylim(cmin, cmax)

    # Add a grid for the x axis and the y axes
    # This is already done if using the whitegrid theme
    #ax1.grid(axis='x', alpha=CONFIG['plot']['grid_opacity'])
    #ax1.grid(axis='y', alpha=CONFIG['plot']['grid_opacity'])
    ax2.grid(axis='y', alpha=CONFIG['plot']['grid2_opacity'], linestyle=CONFIG['plot']['grid2_line_style'])

    # Customize the plot title, labels and ticks
    ax1.set_title(get_plot_title(title, filename))
    ax1.tick_params(axis='x', rotation=CONFIG['plot']['date_rotation'])
    ax1.tick_params(axis='y', labelcolor=CONFIG['colors']['tvoc'])
    ax1.set_xlabel('')
    ax1.set_ylabel(CONFIG['labels']['tvoc'], color=CONFIG['colors']['tvoc'])
    ax2.set_ylabel('')  # We will manually place the 2 parts in different colors

    # Define the position for the center of the right y axis label
    bottom_label = CONFIG['labels']['co'] + '  '
    top_label = '  ' + CONFIG['labels']['form']
    x = 1.07  # Slightly to the right of the axis
    y = get_label_center(bottom_label, top_label)   # Vertically centered

    # Place the first (bottom) part of the label
    ax2.text(x, y, bottom_label, transform=ax2.transAxes,
             color=CONFIG['colors']['co'], rotation='vertical',
             ha='center', va='top')

    # Place the second (top) part of the label
    ax2.text(x, y, top_label, transform=ax2.transAxes,
             color=CONFIG['colors']['form'], rotation='vertical',
             ha='center', va='bottom')

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc=CONFIG['plot']['legend_location'])

    # Adjust the plot margins to make room for the labels
    plt.tight_layout()

    # Save the plot as a PNG image
    plt.savefig(get_plot_filename(filename, '-vcf'))
    plt.close()


def get_label_center(bottom_label, top_label):
    # Return a value between 0 and 1 to estimate where to center the label
    # Divider optimized for 11x8.5 plot size, but not as good for 15x10
    fs = CONFIG['plot']['font_scale']
    divider = 72 * fs**2 - 316 * fs + 414  # Tested for fs between 0.8 and 2
    center = 0.5 + ((len(bottom_label) - len(top_label)) / divider)
    return center


def generate_stats(df, filename):
    summary = df.describe()

    with open(get_stats_filename(filename), 'w') as file:
        file.write(summary.to_string())

    #for column in summary.columns.tolist():
    #    box = sns.boxplot(data=df, y=column)
    #    plt.savefig(get_boxplot_filename(filename, f'-{column}'))
    #    plt.close()


def load_config(reset_config = False):
    global CONFIG

    app_name = 'plotair'
    config_file = 'config.toml'

    config_dir = get_config_dir(app_name)
    user_config_file = config_dir / config_file
    default_config_file = PROJECT_ROOT / app_name / config_file

    if not user_config_file.exists() or reset_config:
        if default_config_file.exists():
            shutil.copy2(default_config_file, user_config_file)
            logger.debug(f'Config initialized at {user_config_file}')
        else:
            raise FileNotFoundError(f'Default config missing at {default_config_file}')
    else:
        logger.debug(f'Found config file at {user_config_file}')

    with open(user_config_file, 'rb') as f:
        CONFIG = tomllib.load(f)


def get_config_dir(app_name):
    if sys.platform == "win32":
        # Windows: Use %APPDATA% (%USERPROFILE%\AppData\Roaming)
        config_dir = Path(os.environ.get("APPDATA", "")) / app_name
    elif sys.platform == "darwin":
        # macOS: Use ~/Library/Preferences
        config_dir = Path.home() / "Library" / "Preferences" / app_name
    else:
        # Linux and other Unix-like: Use ~/.config or XDG_CONFIG_HOME if set
        config_home = os.environ.get("XDG_CONFIG_HOME", "")
        if config_home:
            config_dir = Path(config_home) / app_name
        else:
            config_dir = Path.home() / ".config" / app_name
    
    # Create the directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    return config_dir


def get_plot_title(title, filename):
    if title:
        plot_title = title
    else:
        stem = Path(filename).stem
        match = re.search(r'^(\d+\s*-\s*)?(.*)$', stem)
        plot_title = match.group(2) if match else stem

    # Capitalize only the first character
    if plot_title: plot_title = plot_title[0].upper() + plot_title[1:]

    return plot_title


def get_plot_filename(filename, suffix = ''):
    p = Path(filename)
    return f'{p.parent}/{p.stem}{suffix}.png'


#def get_boxplot_filename(filename, suffix = ''):
#    p = Path(filename)
#    return f'{p.parent}/{p.stem}-boxplot{suffix}.png'


def get_stats_filename(filename):
    p = Path(filename)
    return f'{p.parent}/{p.stem}-stats.txt'


def log_data_frame(df, description = ''):
    """ This function is used only for debugging. """
    logger.debug(f'DataFrame {description}\n{df}')
    #logger.debug(f'DataFrame index data type: {df.index.dtype}')
    #logger.debug(f'DataFrame index class: {type(df.index)}')
    #logger.debug(f'DataFrame columns data types\n{df.dtypes}')
    #logger.debug(f'DataFrame statistics\n{df.describe()}')  # Mean, min, max...
    #sys.exit()


if __name__ == '__main__':
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s - %(message)s')
    
    # Configure this script's logger
    logger.setLevel(logging.DEBUG)

    main()
