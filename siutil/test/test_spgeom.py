import numpy as np
import sisl as si
from siutil.spgeomutil import spgeom_wrap_uc, spgeom_transfer_to_sub, spgeom_transfer_outeridx, spgeom_transfer_periodic, spgeom_tile_from_matrix, spgeom_lrsub
import pytest
from scipy.spatial.distance import cdist

# TODO
