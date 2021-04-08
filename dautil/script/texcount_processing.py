#!/usr/bin/env python

from pathlib import Path
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import defopt

REGEX = re.compile(r'([0-9]+)\+([0-9]+)\+([0-9]+) \(.*\) ([^\\]*)')


def predict(word, const=-0.021111467468646117, slope=0.00858736952056475):
    return const + word * slope


def main(
    path: Path,
    *,
    split_idxs: Optional[List[int]] = None,
):
    res = []
    with open(path) as f:
        for line in f.readlines():
            if (temp := REGEX.findall(line)):
                res.append(temp[0])

    df = pd.DataFrame(res, columns=['text', 'headers', 'captions', 'title'], dtype=int)
    for col in ('text', 'headers', 'captions'):
        df[col] = df[col].astype(np.int32)
    df['predict'] = df.text.map(predict)
    if split_idxs:
        dfs = np.split(df, split_idxs)
        for _df in dfs:
            print(_df)
            print(f'Word count: {_df.text.sum()}')
            print(f'Time predicted (min.): {_df.predict.sum()}')
    else:
        print(df)
    print(f'Total word count: {df.text.sum()}')
    print(f'Total time predicted (min.): {df.predict.sum()}')

def cli():
    defopt.run(main)


if __name__ == '__main__':
    cli()
