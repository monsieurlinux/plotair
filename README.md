![Air quality plot](https://github.com/monsieurlinux/plotair/raw/main/img/cuisine.png)

# PlotAir

[![PyPI][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]

PlotAir is a Python script that processes one or more CSV files containing [VisiblAir][visiblair-link] sensor data. For each file, it reads the data into a [pandas][pandas-link] DataFrame, ignores incorrectly formatted lines, keeps only the most recent data sequence, and generates a [Seaborn][seaborn-link] plot saved as a PNG file with the same base name as the input CSV.

## Dependencies

PlotAir requires the following external libraries:

* **[pandas][pandas-link]**: Used for data manipulation and analysis.
* **[seaborn][seaborn-link]**: Used for creating plots.

These libraries and their sub-dependencies will be installed automatically when you install PlotAir.

## Installation

It is recommended to install PlotAir within a [virtual environment][venv-link] to avoid conflicts with system packages. Some Linux distributions enforce this. You can use `pipx` to handle the virtual environment automatically, or create one manually and use `pip`.

### Installation with `pipx`

`pipx` installs PlotAir in an isolated environment and makes it available globally.

**1. Install `pipx`:**

*   **Linux (Debian / Ubuntu / Mint):**
    
    ```bash
    sudo apt install pipx
    pipx ensurepath
    ```
*   **Linux (Other) / macOS:**
    
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```
*   **Windows:**
    
    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

You may need to reopen your terminal for the PATH changes to take effect. If you encounter a problem, please refer to the official [pipx documentation][pipx-link].

**2. Install PlotAir:**

```bash
pipx install plotair
```

### Installation with `pip`

If you prefer to manage the virtual environment manually, you can create and activate it by following this [tutorial][venv-link]. Then install PlotAir:

```bash
pip install plotair
```

## Deployments

View all releases on:

- **[PyPI Releases][pypi-releases]**
- **[GitHub Releases][github-releases]**

## Usage

### Basic Usage

```bash
plotair [arguments] FILE [FILE ...]
```

### Command-Line Arguments

None for now.

## Configuration

When you run PlotAir for the first time, a `config.toml` file is automatically created. Its location depends on your operating system (typical paths are listed below):

*   **Linux:** `~/.config/plotair`
*   **macOS:** `~/Library/Preferences/plotair`
*   **Windows:** `C:/Users/YourUsername/AppData/Roaming/plotair`

You can edit this file to customize various settings. Common customizations include translating plot labels into different languages or modifying the line colors.

## License

Copyright (c) 2026 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the creators and contributors of the [pandas][pandas-link] and [seaborn][seaborn-link] libraries, and to the developer of the great [VisiblAir][visiblair-link] air quality monitors and CO₂ sensors. Thanks also to the founder of [Bâtiments vivants][batiments-link] for the idea of this script.

[batiments-link]: https://batimentsvivants.ca/
[github-releases]: https://github.com/monsieurlinux/plotair/releases
[license-badge]: https://img.shields.io/pypi/l/plotair.svg
[license-link]: https://github.com/monsieurlinux/plotair/blob/main/LICENSE
[pandas-link]: https://github.com/pandas-dev/pandas
[pipx-link]: https://github.com/pypa/pipx
[pypi-badge]: https://img.shields.io/pypi/v/plotair.svg
[pypi-link]: https://pypi.org/project/plotair/
[pypi-releases]: https://pypi.org/project/plotair/#history
[seaborn-link]: https://github.com/mwaskom/seaborn
[venv-link]: https://docs.python.org/3/tutorial/venv.html
[visiblair-link]: https://visiblair.com/
