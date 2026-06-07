"""
io.py - Data loading utilities for NumCompute-Stream.

Supports loading CSV files into NumPy arrays, with optional header parsing,
custom delimiters, and configurable missing-value fill.
"""

import os
import numpy as np


def load_csv(filepath, delimiter=",", fill_value=0.0):
    """
    Load a CSV file into a 2D NumPy float array.

    Parameters
    ----------
    filepath  : str   Path to the CSV file.
    delimiter : str   Column separator (default ',').
    fill_value: float Value used to replace missing/empty cells (default 0.0).

    Returns
    -------
    np.ndarray  Shape (n_rows, n_cols), dtype float.

    Raises
    ------
    FileNotFoundError  If the file does not exist.
    ValueError         If the file cannot be parsed or is empty.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
    try:
        data = np.genfromtxt(filepath, delimiter=delimiter,
                             filling_values=fill_value, dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse '{filepath}': {exc}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError(f"File '{filepath}' is empty.")
    return data


def load_csv_with_header(filepath, delimiter=",", fill_value=0.0):
    """
    Load a CSV file that contains a header row.

    Parameters
    ----------
    filepath  : str   Path to the CSV file.
    delimiter : str   Column separator (default ',').
    fill_value: float Value used to replace missing/empty cells.

    Returns
    -------
    headers : list[str]   Column names from the first row.
    data    : np.ndarray  Shape (n_rows, n_cols), dtype float.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
    with open(filepath, "r") as fh:
        headers = fh.readline().strip().split(delimiter)
    try:
        data = np.genfromtxt(filepath, delimiter=delimiter,
                             filling_values=fill_value,
                             dtype=float, skip_header=1)
    except Exception as exc:
        raise ValueError(f"Could not parse '{filepath}': {exc}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError(f"No data rows found after header in '{filepath}'.")
    return headers, data


def stream_csv(filepath, chunk_size=100, delimiter=",", fill_value=0.0,
               skip_header=True):
    """
    Generator that yields successive chunks of rows from a CSV file.

    Parameters
    ----------
    filepath   : str   Path to the CSV file.
    chunk_size : int   Number of rows per chunk.
    delimiter  : str   Column separator.
    fill_value : float Fill value for missing cells.
    skip_header: bool  Whether to skip the first row.

    Yields
    ------
    np.ndarray  Shape (≤chunk_size, n_cols), dtype float.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    with open(filepath, "r") as fh:
        if skip_header:
            fh.readline()
        buf = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = []
            for cell in line.split(delimiter):
                cell = cell.strip()
                try:
                    row.append(float(cell) if cell else fill_value)
                except ValueError:
                    row.append(fill_value)
            buf.append(row)
            if len(buf) == chunk_size:
                yield np.array(buf, dtype=float)
                buf = []
        if buf:
            yield np.array(buf, dtype=float)
