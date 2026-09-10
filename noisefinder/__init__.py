from importlib.metadata import PackageNotFoundError, version as _version

from ._dataset import DataSet
from . import cpsd
from .freqscheme import FreqScheme
from . import freqscheme_presets
from . import noiseproj
from . import specwindows

try:
    __version__ = _version("noisefinder")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "DataSet",
    "FreqScheme",
    "cpsd",
    "freqscheme_presets",
    "noiseproj",
    "specwindows",
]
