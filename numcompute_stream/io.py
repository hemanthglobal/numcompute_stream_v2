import os
import numpy as np

def load_csv(filepath, delimiter=",", fill_value=0.0):
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
