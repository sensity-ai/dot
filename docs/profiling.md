# Profiling

Profiling should be carried out whenever significant changes are made to the pipeline. Profiling results are saved as `.txt` and `.prof` files.

## Scripts

### Profile SimSwap - `profile_simswap.py`

This script profiles SimSwap pipeline on a single image pair.

#### Basic Usage

```bash
python profile_simswap.py
```

## Visualisation Tools

Apart from analysing the `.txt` profiling data, we visualise and explore the `.prof` profiling data with:

* [snakeviz](#snakviz): <https://jiffyclub.github.io/snakeviz/>
* [gprof2dot](#gprof2dot): <https://github.com/jrfonseca/gprof2dot>
* [flameprof](#flameprof): <https://github.com/baverman/flameprof>

### SnakeViz

#### Conda Installation

```bash
conda install -c conda-forge snakeviz
```

#### Basic Usage

```bash
snakeviz <path/to/profiling_data>.prof --server
```

### GProf2Dot

#### Conda Installation

```bash
conda install graphviz
conda install -c conda-forge gprof2dot
```

#### Basic Usage

```bash
python -m gprof2dot -f pstats <path/to/profiling_data>.prof | dot -Tpng -o <path/to/profiling_data>.png
```

### FlameProf

#### Pip Installation

```bash
pip install flameprof
```

#### Basic Usage

```bash
python -m flameprof <path/to/profiling_data>.prof > <path/to/profiling_data>.svg
```
