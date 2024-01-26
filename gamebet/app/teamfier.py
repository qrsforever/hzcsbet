#!/usr/bin/env python3

from aux.executor import ExecutorBase
from aux.message import SharedResult

class TeamfierExecutor(ExecutorBase):
    def run(self, sr: SharedResult, logger) -> SharedResult:
        logger.info(f'{self._name} get a message {sr._shmid}.')
        return sr
