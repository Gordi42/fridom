from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import glob

# Function to recursively find all .pyx files
def find_pyx_files(base_dir):
    return [y for x in os.walk(base_dir) 
            for y in glob.glob(os.path.join(x[0], '*.pyx'))]

# Define the extension modules
ext_modules = cythonize([Extension(
    name=os.path.splitext(os.path.relpath(pyx, 'src'))[0].replace(os.sep, '.'),
    sources=[pyx]
) for pyx in find_pyx_files("src/fridom")])

setup(    
    ext_modules=ext_modules,
    package_dir={'': 'src'},
    include_package_data=True,
)
