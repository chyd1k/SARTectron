import os
import glob
import sys
sys.setrecursionlimit(10000)

block_cipher = None
site_packages = "/home/bogdan/miniconda3/envs/sartectron/lib/python3.10/site-packages"
SARTTECTRON_SRC = "/home/bogdan/SARTectron"
conda_lib = "/home/bogdan/miniconda3/envs/sartectron/lib"

# Fix for scipy.linalg.blas: LD_LIBRARY_PATH + runtime hook
binaries = [
    (os.path.join(conda_lib, "libopenblasp-r0.3.32.so"), "."),
    (os.path.join(conda_lib, "libcblas.so"), "."),
    (os.path.join(conda_lib, "libcblas.so.3"), "."),
    (os.path.join(conda_lib, "libblas.so"), "."),
    (os.path.join(conda_lib, "libblas.so.3"), ".")
]

a = Analysis(['SARTectron.py'],
    pathex=[site_packages, SARTTECTRON_SRC],
    binaries=binaries,
    datas=[
        (os.path.join(SARTTECTRON_SRC, "detectron2"), "detectron2"),
        (os.path.join(SARTTECTRON_SRC, "configs/Includes"), "configs/Includes"),
        (os.path.join(site_packages, "fvcore"), "fvcore"),
        (os.path.join(site_packages, "sympy"), "sympy"),
        (os.path.join(site_packages, "pycocotools"), "pycocotools"),
        (os.path.join(site_packages, "scipy"), "scipy"),
        (os.path.join(site_packages, "skimage"), "skimage")
    ],
    hiddenimports=[
        'detectron2.engine.hooks', 'detectron2.modeling', 'detectron2.utils',
        'detectron2.config', 'detectron2.structures', 'detectron2.data',
        'detectron2.evaluation', 'detectron2.layers', 'detectron2.checkpoint',
        'torchvision.ops', 'torchvision.models', 'torchvision.transforms',
        'lzma', '_lzma',
        'sympy', 'sympy.utilities.timeutils', 'sympy.core',
        'timeit', 'pstats', 'profile', 'cProfile',
        'pycocotools', 'pycocotools._mask', 'pycocotools.coco',
        'pycocotools.cocoeval', 'pycocotools.mas_score_lib',
        'scipy', 'scipy.linalg', 'scipy.linalg.blas',
        'skimage', 'skimage.io', 'skimage.color'
    ],
    # runtime hook for BLAS
    runtime_hooks=[os.path.join(SARTTECTRON_SRC, "custom_trainer/blas_runtime_hook.py")],
    excludes=[
        'libstdc++', 'libgcc_s', 'libgomp', 'libquadmath',
        'libLerc', 'Lerc', 'pillow.libs', '_ssl', 'cryptography',
        'tkinter', 'setuptools.msvc', 'distutils',
        'test', 'tests', 'pytest', 'doctest',
        'trace', 'faulthandler', 'tracemalloc'
    ],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='SARTectron', debug=False, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='SARTectron')
