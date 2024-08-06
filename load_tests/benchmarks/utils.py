from itertools import chain

import psutil


class SweepParameter:
    def __init__(self, name, start, end, step):
        self.name = name
        self.start = start
        self.end = end
        self.step = step

    def __str__(self):
        return f"{self.name} from {self.start} to {self.end} in steps of {self.step}"

    def __iter__(self):
        for value in chain(range(self.start, self.end, self.step), [self.end]):
            yield value


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
