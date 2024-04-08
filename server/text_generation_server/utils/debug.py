# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
import glob
import time

from optimum.habana.utils import to_gb_rounded
import habana_frameworks.torch as htorch

START_TS = None
DBG_TRACE_FILENAME = os.environ.get('DBG_TRACE_FILENAME')
if 'GRAPH_VISUALIZATION' in os.environ:
    for f in glob.glob('.graph_dumps/*'):
        os.remove(f)


def count_hpu_graphs():
    return len(glob.glob('.graph_dumps/*PreGraph*'))


def dbg_trace(tag, txt):
    global START_TS
    if DBG_TRACE_FILENAME is not None and int(os.getenv("RANK", 0)) == 0:
        if START_TS is None:
            START_TS = time.perf_counter()
        time_offset = time.perf_counter() - START_TS
        mem_stats = htorch.hpu.memory.memory_stats()
        mem_used = to_gb_rounded(mem_stats['InUse'])
        max_mem_used = to_gb_rounded(mem_stats['MaxInUse'])
        print(f'ts:{time_offset:.3f}s g:{count_hpu_graphs()} mu:{mem_used:.1f}GB '
              f'mmu:{max_mem_used:.1f}GB | {tag} | {txt}', flush=True, file=open(DBG_TRACE_FILENAME, 'a'))
