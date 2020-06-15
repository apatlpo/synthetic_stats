# Internal-Tide-Obs

Gathers various notebooks and tools to assess internal tide observability

## Features

Note on install [fastparquet](https://pypi.org/project/fastparquet/):

```
conda install -c conda-forge fastparquet
```

Note about installing [h3-py](https://github.com/uber/h3-py) on datarmor:

```
# unload all intel modules
module unload mpt intel-fc-16 intel-cmkl-16 cmake
# reset CC environment variable
setenv CC cc
# install cmake
conda install -c conda-forge cmake
# install h3
pip install h3
```

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue)](https://binder.pangeo.io/v2/gh/apatlpo/itide_ops/master?urlpath=lab)

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=AWS+us-west-2&color=orange)](https://aws-uswest2-binder.pangeo.io/v2/gh/apatlpo/itide_ops/master?urlpath=lab)

Try these notebooks on pangeo.binder.io_ : |Binder|

See http://pangeo.io for more information.

Features
--------