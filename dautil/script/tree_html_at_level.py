#!/usr/bin/env python

from pathlib import Path
from collections import OrderedDict
from typing import List
import sys

from subprocess import getoutput

import pandas as pd
import defopt

def main(*, outpath: Path = None, level: int = 3, names: List[str] = ['Year', 'Author', 'Name']):
    '''Create an HTML page of the directory tree at a certain level

    :param Path outpath: the output path of the HTML file
    :param int level: the level at which the files is located at
    :param list[str] names: names of each level, must of of length `level`
    '''
    # using find in the shell is quick and dirty, can be improved to use Python glob
    # ignore all first level dir starts with . e.g. .git
    paths = getoutput(f"find . -maxdepth {level} -mindepth {level} -type f \\! -path './.*'").split('\n')
    paths = [Path(path) for path in paths]

    df = pd.DataFrame(
        (str(path) for path in paths),
        index=pd.MultiIndex.from_tuples(
            [
                # replace underscore by space for pretty print
                [' '.join(subdir.split('_')) for subdir in path.parts]
                for path in paths
            ],
            names=names,
        ),
    )

    df.sort_index(inplace=True)

    # last level from the index becomes the text in the link
    df['link'] = [f'<a href="{row[1]}">{row[0]}</a>' for row in df.reset_index(level=-1).values]

    if outpath is None:
        f = sys.stdout
        df[['link']].to_html(f, escape=False)
    else:
        with open(outpath, 'w') as f:
            df[['link']].to_html(f, escape=False)

def cli():
    defopt.run(main)

if __name__ == '__main__':
    cli()
