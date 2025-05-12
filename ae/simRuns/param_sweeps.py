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
_ = model_init(
    Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
)

_ = model_auto_regression(
    Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
    input_seq_length + output_seq_length,
)


def test_memory_bandwidth(memory_bandwidth, lock):
    resultsDir = "ae/memoryBW"
    arch_specs["device"]["io"]["memory_channel_physical_count"] = memory_bandwidth
    arch_specs["device"]["io"]["memory_channel_active_count"] = memory_bandwidth
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)
    print(
        f"{memory_bandwidth}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2 + io_area_mm2}"
    )
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
    print(
        f"{memory_bandwidth}, {init_latency_simulated}, {auto_regression_latency_simulated}"
    )
    with lock:
        with open(f"{resultsDir}/memory_bw_results_bs{batch_size}_init.csv", "a") as f:
            f.write(
                f"{memory_bandwidth * 400}, {compute_area_mm2 + io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(f"{resultsDir}/memory_bw_results_bs{batch_size}_ar.csv", "a") as f:
            f.write(
                f"{memory_bandwidth * 400}, {compute_area_mm2 + io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


def sweep_params(memory_capacity, memBW, interconnectBW, lock):
    # FIXME: Really? Running in different dir and saving results in some other?
    # FIXME: REALLY?
    resultsDir = "ae/memoryBW"

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
    # print(memVW["Name"])

    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)

    # print(
    #     f"Sim: {memory_bandwidth}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2 + io_area_mm2}"
    # )
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
    # print(
    #     f"{memory_bandwidth}, {init_latency_simulated}, {auto_regression_latency_simulated}"
    # )

    with lock:
        _stacks = memBW["Stacks"]
        _memName = memBW["Name"]
        _interConnectName = interconnectBW["Name"]
        with open(
            f"{resultsDir}/memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs{batch_size}_init.csv",
            "a",
        ) as f:
            f.write(
                f"{memory_bandwidth}, {compute_area_mm2 + io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(
            f"{resultsDir}/memory_{_memName}_{_stacks}_bw_{_interConnectName}_results_bs{batch_size}_ar.csv",
            "a",
        ) as f:
            f.write(
                f"{memory_bandwidth}, {compute_area_mm2 + io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
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
    lock = Lock()
    processes = []

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
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 4,
        },
        {
            "Name": "HBM4",
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 36,
            "Stacks": 4,
        },
        {
            "Name": "HBM5",
            "Datarate": 6.6,
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
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 8,
        },
        {
            "Name": "HBM4",
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 36,
            "Stacks": 8,
        },
        {
            "Name": "HBM5",
            "Datarate": 6.6,
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
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 12,
        },
        {
            "Name": "HBM4",
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 36,
            "Stacks": 12,
        },
        {
            "Name": "HBM5",
            "Datarate": 6.6,
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
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 24,
            "Stacks": 16,
        },
        {
            "Name": "HBM4",
            "Datarate": 9.4,
            "IO": 1024,
            "CapacityPerStack": 36,
            "Stacks": 16,
        },
        {
            "Name": "HBM5",
            "Datarate": 6.6,
            "IO": 2048,
            "CapacityPerStack": 48,
            "Stacks": 16,
        },
        # {"Name": "UCIe", "PerLaneBW": 4, "NrLanes": 1024}
    ]
    # memDatarate =[4, 9.4, 9.4, 6.6]
    # memIO = [1024, 1024, 1024, 2048]
    # This is for a single LVlink, and there are 12 in A100, 18 in H100
    # NVLink3, 4, 5, 6, C2C, HBI divding 10 TB into 6 lanes for FC
    interconnectType = [
        {"Name": "NVLink3", "PerLaneBW": 50, "NrLanes": 12},
        {"Name": "NVLink4", "PerLaneBW": 50, "NrLanes": 18},
        {"Name": "NVLink5", "PerLaneBW": 100, "NrLanes": 18},
        {"Name": "NVLink6", "PerLaneBW": 100, "NrLanes": 18},
        {"Name": "NVLink-C2C", "PerLaneBW": 200, "NrLanes": 10},
        {"Name": "NVLink-HBI", "PerLaneBW": 90, "NrLanes": 6},
        # {"Name": "UCIe", "PerLaneBW": 4, "NrLanes": 1024}
    ]
    # interconnectBW = [50, 50, 100, 200, 90, 1667]
    # interconnectLanes = [12, 18, 18, 18, 10, 6]

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

    # WARN: First parameter capacity does nothing, moved it as a memory config
    # Interconnect sensitivity
    for cCap in interconnectType:
        processes.append(
            Process(target=sweep_params, args=(defaults[0], defaults[1], cCap, lock))
        )

    # Memory sensitivity
    for cMem in memoryType:
        processes.append(
            Process(target=sweep_params, args=(defaults[0], cMem, defaults[2], lock))
        )

    for cInterconnect in interconnectType:
        for cMem in memoryType:
            processes.append(
                Process(
                    target=sweep_params, args=(defaults[0], cMem, cInterconnect, lock)
                )
            )

    # processes = [
    #     Process(target=test_memory_bandwidth, args=(i, lock))
    #     for i in [1, 2, 3, 4, 5, 6, 7, 8]
    # ]

    loader = Loader("Runnig sims ...", "Done!", 0.05).start()
    try:
        for p in processes:
            p.start()

        while any(p.is_alive() for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating processes...")
        for p in processes:
            p.terminate()
            p.join()

    loader.stop()

    print("All processes have finished.")
