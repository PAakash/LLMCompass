# INFO: What is new?
# - param_sweeps
#   > Creating arbitrary sweeps for sensitivity and overall parameters
#   > Next updated should be with more systematic configurations
#   > Saperate Sweep for memory datarate, capacity, and interconnect
# - param_sweeps_updated:
#   > Proper memory and interconnect configurations for sweeps
#   > Removed sensitivity simulations since it will be in whole sweep
#   > Changed results directory which was storing results in different simulation directory
# - param_sweep_new
#   >Running workload simulations:
#   > Sweep for batch size, input seq length and output seq length
#   > Adding mkdir for results :(


from pathlib import Path
import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

from design_space_exploration.dse import template_to_system, read_architecture_template
from multiprocessing import Process, Lock
import time
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2

import sys

resultsDir = "ae/simRuns/results_workload"

# TODO: Add as args?
input_seq_length = 2048
batch_size = 8
output_seq_length = 1024

arch_specs = read_architecture_template("configs/template.json")
device_count = arch_specs["device_count"]

# Model for prefill or summarization
model_init = TransformerBlockInitComputationTP(
    d_model=12288,
    n_heads=96,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)

# Model for decoding or generation
model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model=12288,
    n_heads=96,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
# _ = model_init(
#     Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
# )
#
# _ = model_auto_regression(
#     Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
#     input_seq_length + output_seq_length,
# )


def sweep_params(tag, workLoadSpec, memBW, interconnectBW, lock):
    arch_specs["device"]["io"]["memory_channel_physical_count"] = memBW["Stacks"]
    arch_specs["device"]["io"]["memory_channel_active_count"] = memBW["Stacks"]
    arch_specs["device"]["memory"]["total_capacity_GB"] = (
        memBW["Stacks"] * memBW["CapacityPerStack"]
    )
    arch_specs["device"]["io"]["pin_count_per_channel"] = memBW["IO"]
    arch_specs["device"]["io"]["bandwidth_per_pin_bit"] = memBW["Datarate"] * 1e9
    arch_specs["interconnect"]["link"]["bandwidth_both_directions_byte"] = (
        interconnectBW["PerLaneBW"] * 1e9
    )
    arch_specs["interconnect"]["link"]["bandwidth_per_directions_byte"] = (
        interconnectBW["PerLaneBW"] / 2
    ) * 1e9
    arch_specs["interconnect"]["link_count_per_device"] = interconnectBW["NrLanes"]

    memory_bandwidth = memBW["IO"] * memBW["Datarate"] * memBW["Stacks"]

    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)

    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")

    with lock:
        _stacks = memBW["Stacks"]
        _memName = memBW["Name"]
        _interConnectName = interconnectBW["Name"]

        directory = f"{resultsDir}/{tag}/{workLoadSpec}"
        # Check if directory exists
        Path(directory).mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        with open(
            f"{resultsDir}/{tag}/{workLoadSpec}/memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs_init.csv",
            "a",
        ) as f:
            f.write(
                f"{memory_bandwidth},{workLoadSpec}, {compute_area_mm2 + io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(
            f"{resultsDir}/{tag}/{workLoadSpec}/memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs_ar.csv",
            "a",
        ) as f:
            f.write(
                f"{memory_bandwidth},{workLoadSpec}, {compute_area_mm2 + io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )

        with open(f"{resultsDir}/{tag}/simRuns.txt", "a") as _file:
            _file.write(
                f"{resultsDir}/{tag}/{workLoadSpec}/memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs\n"
            )


from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}: {self.desc}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


if __name__ == "__main__":
    # TODO: Add argparser here

    lock = Lock()
    processes = []

    # with open(
    #     "./ae/simRuns/memoryConfigs.json", mode="r", encoding="utf-8"
    # ) as read_file:
    #     _memoryType = json.load(read_file)
    # memoryType = _memoryType["memoryType"]
    inSeqLen = [
        1 * 1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        128 * 1024,
        512 * 1024,
        1024 * 1024,
    ]
    outSeqLen = [
        1 * 1024,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        128 * 1024,
        512 * 1024,
        1024 * 1024,
    ]
    batchSize = [1, 2, 4, 8, 16, 32, 64, 128]

    memoryType = [
        {
            "Name": "HBM2e",
            "Datarate": 4,
            "IO": 1024,
            "CapacityPerStack": 16,
            "Stacks": 4,
        },
        {
            "Name": "HBM3",
            "Datarate": 6.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 4,
        },
        {
            "Name": "HBM4",
            "Datarate": 8,
            "IO": 2048,
            "CapacityPerStack": 36,
            "Stacks": 4,
        },
        {
            "Name": "HBM5",
            "Datarate": 9.4,
            "IO": 2048,
            "CapacityPerStack": 48,
            "Stacks": 4,
        },
        # {"Name": "UCIe", "PerLaneBW": 4, "NrLanes": 1024}
        {
            "Name": "HBM2e",
            "Datarate": 4,
            "IO": 1024,
            "CapacityPerStack": 16,
            "Stacks": 8,
        },
        {
            "Name": "HBM3",
            "Datarate": 6.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 8,
        },
        {
            "Name": "HBM4",
            "Datarate": 8,
            "IO": 2048,
            "CapacityPerStack": 36,
            "Stacks": 8,
        },
        {
            "Name": "HBM5",
            "Datarate": 9.4,
            "IO": 2048,
            "CapacityPerStack": 48,
            "Stacks": 8,
        },
        # {"Name": "UCIe", "PerLaneBW": 4, "NrLanes": 1024}
        {
            "Name": "HBM2e",
            "Datarate": 4,
            "IO": 1024,
            "CapacityPerStack": 16,
            "Stacks": 12,
        },
        {
            "Name": "HBM3",
            "Datarate": 6.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 12,
        },
        {
            "Name": "HBM4",
            "Datarate": 8,
            "IO": 2048,
            "CapacityPerStack": 36,
            "Stacks": 12,
        },
        {
            "Name": "HBM5",
            "Datarate": 9.4,
            "IO": 2048,
            "CapacityPerStack": 48,
            "Stacks": 12,
        },
        # {"Name": "UCIe", "PerLaneBW": 4, "NrLanes": 1024}
        {
            "Name": "HBM2e",
            "Datarate": 4,
            "IO": 1024,
            "CapacityPerStack": 16,
            "Stacks": 16,
        },
        {
            "Name": "HBM3",
            "Datarate": 6.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 16,
        },
        {
            "Name": "HBM4",
            "Datarate": 8,
            "IO": 2048,
            "CapacityPerStack": 36,
            "Stacks": 16,
        },
        {
            "Name": "HBM5",
            "Datarate": 9.4,
            "IO": 2048,
            "CapacityPerStack": 48,
            "Stacks": 16,
        },
    ]
    interconnectType = [
        {"Name": "NVLink3", "PerLaneBW": 50, "NrLanes": 12},
        {"Name": "NVLink4", "PerLaneBW": 50, "NrLanes": 18},
        {"Name": "NVLink5", "PerLaneBW": 100, "NrLanes": 18},
        {"Name": "NVLink6", "PerLaneBW": 100, "NrLanes": 18},
        {"Name": "NVLink-C2C", "PerLaneBW": 200, "NrLanes": 10},
        {"Name": "NVLink-HBI", "PerLaneBW": 16384, "NrLanes": 6},
    ]

    # ucie_datarates = [4, 8, 16, 32, 48, 64, 128]
    ucie_datarates = [16, 32, 64, 128]
    ucie_modules = [1, 2, 4, 8, 16, 32, 64]
    for m in ucie_modules:
        for i in ucie_datarates:
            interconnectType.append(
                {"Name": f"UCIe-{m}x{i}", "PerLaneBW": i, "NrLanes": m * 128},
            )

    # INFO: Defaults: total mem capacity per GPU chiplet GB, mem IO datarate
    # (Gbps), mem IOs, interconnect BW (GBps)
    defaults = [
        80,
        {
            "Name": "HBM2e",
            "Datarate": 4,
            "IO": 1024,
            "CapacityPerStack": 16,
            "Stacks": 4,
        },
        {"Name": "NVLink3", "PerLaneBW": 50, "NrLanes": 12},
    ]

    sweepConfis = []
    for _cInSeqLen in inSeqLen:
        for _cOutSeqLen in outSeqLen:
            for _cBatchSize in batchSize:
                # System config
                _ = model_init(
                    Tensor(
                        [batch_size, input_seq_length, model_init.d_model],
                        data_type_dict["fp16"],
                    )
                )

                _ = model_auto_regression(
                    Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
                    input_seq_length + output_seq_length,
                )

                for cInterconnect in interconnectType:
                    for cMem in memoryType:
                        _memName = cMem["Name"]
                        _stacks = cMem["Stacks"]
                        _interConnectName = cInterconnect["Name"]

                        # sweepConfis.append(
                        #     f"memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs{batch_size}"
                        # )
                        processes.append(
                            Process(
                                target=sweep_params,
                                args=(
                                    "sweep",
                                    f"{_cBatchSize}_{_cInSeqLen}_{_cOutSeqLen}",
                                    cMem,
                                    cInterconnect,
                                    lock,
                                ),
                            )
                        )

    print(f"Running {len(processes)} simulations ...")

    totalSims = len(processes)
    loader = Loader(f"[x/{totalSims}]Runnig sims ...", "Done!", 0.05).start()

    jobsToRun = 16
    cSim = 0

    try:
        while len(processes) > 0:
            tmpList = []

            for x in range(jobsToRun):
                if len(processes) > 0:
                    cCmd = processes.pop(0)
                    tmpList.append(cCmd)
                    cSim += 1
            loader.stop()
            loader = Loader(
                f"[{cSim - jobsToRun}-{cSim}]/[{totalSims}] Running sims ...",
                "[Done!]",
                0.05,
            ).start()
            # loader.desc = f"Running sims ] ..."

            for i in tmpList:
                i.start()
                # print(i)
            # print("Process pushed...\n\n")
            while any(p.is_alive() for p in tmpList):
                time.sleep(1)

    except KeyboardInterrupt:
        print("Terminating processes...")
        for p in processes:
            p.terminate()
            p.join()

    loader.stop()

    loader = Loader("Saving sinRuns list ...", "Done!", 0.05).start()

    loader.stop()

    print("All processes have finished.")
