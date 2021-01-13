__all__ = ['tseries', 'filtering', 'hfreq']

#from . import tseries

import matplotlib.colors as colors
import matplotlib.cm as cmx

def get_cmap_colors(Nc, cmap='plasma'):
    """ load colors from a colormap to plot lines

    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc),
                                   cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]
