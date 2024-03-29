"""dautil: Script Generator that organizes cli args by YAML

See:
https://github.com/ickc/dautil-py
"""

# To use a consistent encoding
from io import open

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# Get the long description from the README file
try:
    with open('docs/README.rst', encoding='utf-8') as f:
        long_description = f.read()
# in case the doc wasn't generated by pandoc
except Exception:  # FileNotFoundError: Python 2 doesn't has this error
    long_description = ''

version = '0.2.1'

setup(
    name='dautilpy',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='Data Analysis Utilities',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ickc/dautil-py',

    # Author details
    author='Kolen Cheung',
    author_email='christian.kolen@gmail.com',

    # Choose your license
    license='BSD-3-Clause',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    # What does your project relate to?
    keywords='data analysis',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=['pyyaml',],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['pep8', 'pylint', 'pytest', 'pytest-cov', 'coverage', 'coveralls', 'future'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'h5link_recursive = dautil.IO.h5:h5link_recursive_cli',
            'tree_html_at_level = dautil.script.tree_html_at_level:cli',
            'detect_missing_track_type = dautil.script.detect_missing_track_type:cli',
            'glom_yaml = dautil.script.glom_yaml:cli',
            'texcount_processing = dautil.script.texcount_processing:cli',
            'h5finite = dautil.script.h5finite:cli',
            'h5delete = dautil.script.h5delete:cli',
            'pandas_concat = dautil.script.pandas_concat:cli',
        ],
    },
)
