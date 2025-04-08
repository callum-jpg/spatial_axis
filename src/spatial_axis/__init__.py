# Import with a redundant alias
# Avoids having to: from SpatialAxis.spatial_axis import spatial_axis
from .spatial_axis import spatial_axis as spatial_axis

import logging
from ._logging import configure_logger

log = logging.getLogger("spatial_axis")
configure_logger(log)
