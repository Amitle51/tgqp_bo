# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# import rpy2.robjects.numpy2ri
#
# rpy2.robjects.numpy2ri.activate()

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion
from rpy2.robjects.conversion import Converter
import rpy2.robjects.numpy2ri as numpy2ri


# --- Create a local converter for numpy <-> R ---
local_converter = Converter('local numpy converter')
local_converter += numpy2ri.converter

# Import the hdtg package and get direct reference to the function
hdtg = importr('hdtg')
harmonicHMC = hdtg.harmonicHMC  # Direct function reference


def run_harmonic_hmc(nSample, mean, choleskyFactor, constrainDirec, constrainBound, init, precFlg=True):
    """
    Wrapper for the harmonicHMC function from the hdtg package.

    Direct function call - no global environment pollution, no string evaluation.

    Parameters:
    - nSample: Number of samples to draw
    - mean: Mean vector (R FloatVector)
    - choleskyFactor: Cholesky factor of the covariance matrix (R matrix)
    - constrainDirec: Linear constraint matrix (R matrix)
    - constrainBound: Linear constraint vector (R FloatVector)
    - init: Initial point (R FloatVector)
    - precFlg: Precision flag (default True)

    Returns:
    - samples: R matrix of samples
    """
    # Call R function directly with named arguments
    samples = harmonicHMC(
        nSample=nSample,
        mean=mean,
        choleskyFactor=choleskyFactor,
        constrainDirec=constrainDirec,
        constrainBound=constrainBound,
        init=init,
        precFlg=precFlg
    )

    return samples