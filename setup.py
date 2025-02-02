from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

SRC_FOLDER = Path(__file__).parent / "src"

setup(
    name="tawrch",
    ext_modules=[
        CUDAExtension(
            "_tawrch",
            [str(SRC_FOLDER / "augment" / "binding.cpp"),
             str(SRC_FOLDER / "augment" / "augment.cu")],
        )
    ],
    packages=["tawrch"],
    cmdclass={"build_ext": BuildExtension},
)
