#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult
import numpy as np


class TeamfierExecutor(ExecutorBase):
    _name = 'Teamfier'

    def run(self, frame: np.ndarray, msg: SharedResult, cache: dict) -> SharedResult:
        return msg

    def post_loop(self, cache):
        self.logger.info(f'{self._name} post processing..')
