"""Search for GB (Strub et al)."""

import argparse
from copy import deepcopy
from os.path import join as pjoin
import logging
from shutil import rmtree

import numpy as np
import pandas as pd

from jaxgb.jaxgb import JaxGB

from globalfit.gb.generator import CENTRAL_DTYPE, GBGenerator
from globalfit.base import BBLK_ATTRS_DEF
from globalfit.base.block import BaseBlock
from globalfit.gb.likelihood import LDC_NAMES as GB_NAMES


#############
import json
import os
import subprocess
import multiprocessing

logger = logging.getLogger(__name__)
   
#run a gbsearch pass (even1st, odd, even) using 1 cpu core per batch
#the search code could be called here directly, but to achieve a better
#loadbalancing subprocesses are used insted

Ncpus = 16
def search(i, which_run):
    try:
        core = (multiprocessing.current_process()._identity[0] - 1) % Ncpus
        command = "taskset -c %d ldc-gb-search %s %d" % (core, which_run, i)
        subprocess.run([command], shell=True)
        pass
    except Exception as e:
        print(f"Exception in worker with args {args}:\n{traceback.format_exc()}")
        raise

def merge(which_run):
    command = "ldc-gb-merge %s" % which_run
    subprocess.run([command], shell=True)

# fmt: off
GBSBLK_ATTRS_DEF = BBLK_ATTRS_DEF | {

    # Block name. Meant to identify one block type in the pipeline
    # multiple instance of the same block should be identified by
    # the iteration index and, if needed, other suitably defined indexes
    "name": "gbsearch",

    "inputs": [
        {"file": pjoin("{data_dir}", "tdi_combined.h5"), "add": [{"name": "tdi"}]},
        {
            "sub": [
                {
                    "name": "cat",
                    "generator": "load:globalfit.mbhb.generator.MBHBGenerator",
                    "list": None,
                }
            ],
            "file": pjoin(
                "{base_dir}", "{iteration - 1}", "mbhb_search", "mbhb_cat.h5"
            ),
        },
    ],

    # Data / fastwaveform configuration
    "domain": "Frequency",
    "AET": False,
    "ignore_T_channel": False,
    "window": {
        "window": ('planck', 0.05)
    },
    "orbits": pjoin("{data_dir}", "orbits.h5"),
    "tdi_generation": 2,
    "dt": 10,
    "duration": [0, 365.25*24*3600],
    "fband": [2.0e-05, 0.024],
    "fft_win": {},
    "bsize": 1024,

    # Search parameters
    "data_set": "tdi_combined",
    #"dt": 10.0,				#the dt value is already set
    "snr_threshold": 9.0,
    #"tdi_generation": 2,			#the tdi_generation value is already set
    "max_signals_per_window": 10,
    "max_signals_per_window_first_run": 3,
    "channel_combination": "AET",
    "frequency_range": [],			#use the fband value instead
    "seed": 1,
    "batch_size": 1,
    "data_path": "",				#value is set to data_dir
    "orbits_path": "",				#value is set later to orbtis
    "save_path": "",				#value is set to output_dir
    "catalog_path": "../",
    "match_criteria": "overlap",
    "overlap_threshold": 0.9,
    "scaled_error_threshold": 0.3,
    #End of Search parameters 

    # Outfile names
    "output_gb_cat": pjoin("{output_dir}", "gb_catalog.h5"),

}

class GBSearchBlock(BaseBlock):
    """
    Identify detectable GBs

    :param dict defaults: dictionary with configuration defaults (for child class \
                          implementation). Default :py:data:`GBCBLK_ATTRS_DEF`.
    :param str config_file: YAML file with the block configuration. Default: `config_file` \
                            configuration key. If the file is missing it will be ignored.
    :param dict config: dictionary with the block configuration. Overwrites parameters in \
                        YAML file.
    :param LDCData tdi: a LDCData instance to be loaded as base :py:attr:`tdi` property. \
                        Default None.
    """

    def __init__(self, defaults=None, **kwargs):


        if defaults is None:
            defaults = deepcopy(GBSBLK_ATTRS_DEF)

        super().__init__(defaults=defaults, **kwargs)

        self.fgb = None
    def _get_fgb(self, size=None):
        """ Not using GBGenerator to go faster
        """
        if size is None:
            size = self.config["bsize"]
        return JaxGB(orbits=self.orbits,
                     t_obs=self.config["duration"][-1] - self.config["duration"][0],
                     t0=self.config["duration"][0],
                     n=size)

    def load_tdi(self, **kwargs):
        """ Override the base block
        """
        super().load_tdi(domain="Time") # force preprocessing domain
        self.to_domain(self.config["domain"])




    def exec(self):

        # search
        #self.load_orbits()
        #self.load_tdi()
        #self.fgb = self._get_fgb()

        # Specify the GBsearch config file
        search_keys = [
            "data_set", "dt", "snr_threshold", "tdi_generation",
            "max_signals_per_window", "max_signals_per_window_first_run",
            "channel_combination", "frequency_range", "seed", "batch_size",
            "data_path", "orbits_path", "save_path", "catalog_path",
            "match_criteria", "overlap_threshold", "scaled_error_threshold",
        ]
	#Fill in missing parameters for the GBsearch code
        search_params = {k: self.config[k] for k in search_keys}
        search_params["frequency_range"] = list(self.config["fband"])
        search_params["orbits_path"] = self.config["orbits"]
        search_params["data_path"] = self.config["inputs"][0]["file"]
        search_params["save_path"] = self.config["output_dir"]


        config_file = os.path.join(self.config["data_dir"], "search_params.json")
        with open(config_file, "w") as f:
            json.dump(search_params, f, indent=4)
        print(f"config file successfully written to {config_file}")

        os.environ["LDC_CONFIG"] = config_file


        #do a first run to identify the number of needed windows
        Nbatches = 0
        command = "taskset -c 0 ldc-gb-search even1st 0"
        result = subprocess.run([command], capture_output=True, text=True, check=True, shell=True)

        for line in result.stdout.splitlines():
            l = line.split("total number of windows ")
            if(len(l) > 1):
                Nbatches = int(l[1])

        print("Nbatches: ", Nbatches)


        iterations = list(range(Nbatches, -1, -1))

        for which_run in ["even1st", "odd", "even"]:
            with multiprocessing.Pool(processes=Ncpus) as pool:
                args_list = [(i, which_run) for i in iterations]
                pool.starmap(search, args_list, chunksize=1)
            merge(which_run)

        merge("global")

        self.write_gb_catalog(params)
        logger.info("Saved GB catalog")
        return 0


    def write_gb_catalog(self, params):
        """Save gb catalog in the geemoo format

        :param dataframe params: found source parameters

        TODO: feel free to change the input format to what is the most
        convenient.

        """

        dtype = [d for d in CENTRAL_DTYPE if d[0]!="index"]
        geemoo_cat = np.zeros((len(params)), dtype=dtype)

        #physical parameter from search and snr
        for key in params.columns:
            if key in geemoo_cat.dtype.names:
                geemoo_cat[key] = params[key]

        #extra parameters
        geemoo_cat["confidence"] = np.ones((len(params))) # leave one for now
        GBGenerator.save_catalog(self.config['output_gb_cat'],
                                 pd.DataFrame(geemoo_cat))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Command line interface to the GBClusterBlock class"
    )
    PARSER.add_argument("--config-file", type=str, help="yml configuration file.")
    PARSER.add_argument(
        "--iterations", type=int, help="Number of iterations. Ignored.", default=1
    )
    PARSER.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help="dumps raw config to yml file at the end of run. Default: no dump",
    )
    PARSER.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="location of the logfile if any",
    )
    PARSER.add_argument("--debug", action="store_true", help="Set debug logging.")
    PARSER.add_argument(
        "--clear",
        action="store_true",
        help="Clear all data after running (for testing).",
    )

    OPTIONS = PARSER.parse_args()
    OPTIONS_D = vars(OPTIONS)
    LOG_LEVEL = logging.DEBUG if OPTIONS.debug else logging.INFO
    logging.basicConfig(
        level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG = logging.getLogger("GBSearchBlock")

    block = GBSearchBlock(config_file=OPTIONS.config_file, log_file=OPTIONS.log_file)
    block.exec()

    LOG.info("Block config: %s", block.config)

    block.run(maxiter=1, dump_config=OPTIONS.dump_config)

    if OPTIONS.clear:
        rmtree(block.config["basedir"])
