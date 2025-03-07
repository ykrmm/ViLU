from typing import Any, Dict
from lightning import Fabric

from vilu.lib.logger import logger

Kwargs = Dict[str, Any]


class LightningFabric(Fabric):
    def __init__(
        self,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def info(self, message: str) -> None:
        if self.is_global_zero:
            logger.info(message)

    def warning(self, message: str) -> None:
        if self.is_global_zero:
            logger.warning(message)
        self.barrier()
