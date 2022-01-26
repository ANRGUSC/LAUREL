REGISTRY = {}

from .rnn_agent import RNNAgent
from .tar_rnn_agent import TarRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["tar-rnn"] = TarRNNAgent