**How to use GBsearch in Geemoo**



**Step 1**

Clone GBsearch from:

https://github.com/stefanstrub/GlobalGB/tree/master

**Step 2**

Replace the following files in GBsearch with those in this repository:

* GB_runner.py
* congig.py
* data_loader.py
* match_GB.py
* pyproject.toml

**Step 3**

Install GBsearch with python 3.13 to be compatible with GeeMoo:

        conda create -n global_gb3_13 python=3.13
        conda activate global_gb3_13
        cd GlobalGB
        pip install -e .

**Step 4**

Test GBsearch with a tdi_combined input file:

* copy the tdi_combined.5 file to data/
* copy the orbits.5h file to data/
* copy the search_params.json file to /data

where search_params.json contains:

    {
        "data_set": "tdi_combined",
        "dt": 10,
        "snr_threshold": 9.0,
        "tdi_generation": 2,
        "max_signals_per_window": 10,
        "max_signals_per_window_first_run": 3,
        "channel_combination": "AET",
        "frequency_range": [
            2e-05,
            0.024
        ],
        "seed": 1,
        "batch_size": 1,
        "data_path": "data/tdi_combined.h5",
        "orbits_path": "data/orbits.h5",
        "save_path": "gfrun/0/gbsearch",
        "catalog_path": "../",
        "match_criteria": "overlap",
        "overlap_threshold": 0.9,
        "scaled_error_threshold": 0.3
    }
    

Run with:

    python3 runScript.py


*Note*

Adjust the number of CPUs  in runScript.py.

The script uses 1 CPU per window, starting from the largest frequency window to improve loadbalancing.

