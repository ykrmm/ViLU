import os
from omegaconf import OmegaConf

import numpy as np

OmegaConf.register_new_resolver("join", lambda *pth: os.path.join(*pth))
OmegaConf.register_new_resolver("mult", lambda *numbers: float(np.prod([float(x) for x in numbers])))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
OmegaConf.register_new_resolver("in", lambda a, b: a in b)
