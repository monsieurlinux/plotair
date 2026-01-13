![Air quality plot](https://github.com/monsieurlinux/plotair/raw/main/img/cuisine.png)

# PlotAir

PlotAir is a Python script that processes one or more CSV files containing [VisiblAir](https://visiblair.com/) sensor data. For each file, it reads the data into a [pandas](https://github.com/pandas-dev/pandas) DataFrame, ignores incorrectly formatted lines, keeps only the most recent data sequence, and generates a [Seaborn](https://github.com/mwaskom/seaborn) plot saved as a PNG file with the same base name as the input CSV.

## Dependencies

PlotAir requires the following external libraries:

* **[pandas](https://github.com/pandas-dev/pandas)**: Used for data manipulation and analysis.
* **[seaborn](https://github.com/mwaskom/seaborn)**: Used for creating plots.

These libraries and their sub-dependencies will be installed automatically when you install PlotAir.

## Installation

It is recommended to install PlotAir within a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid conflicts with system packages. Some Linux distributions enforce this. You can use `pipx` to handle the virtual environment automatically, or create one manually and use `pip`.

### Installation with `pipx`

`pipx` installs PlotAir in an isolated environment and makes it available globally.

**1. Install `pipx`:**

*   **Linux (Debian / Ubuntu / Mint):**
    
    ```bash
    sudo apt update && sudo apt install pipx
    ```
*   **Linux (Other) / macOS:**
    
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    # Note: Close and restart your terminal after running ensurepath
    ```
*   **Windows:**
    
    ```bash
    python -m pip install --user pipx
    ```

**2. Install PlotAir:**

```bash
pipx install plotair
```

### Installation with `pip`

If you prefer to manage the virtual environment manually, you can create and activate it by following this [tutorial](https://docs.python.org/3/tutorial/venv.html). Then install PlotAir:

```bash
pip install plotair
```

## Usage

### Basic Usage

```bash
plotair <file1> <file2> ...
```

### Command-Line Arguments

None for now.

## License

Copyright (c) 2026 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the creators and contributors of the [pandas](https://github.com/pandas-dev/pandas) and [seaborn](https://github.com/mwaskom/seaborn) libraries, and to the developer of the great [VisiblAir](https://visiblair.com/) air quality monitors and CO₂ sensors. Thanks also to the founder of [Bâtiments vivants](https://batimentsvivants.ca/) for the idea of this script.
