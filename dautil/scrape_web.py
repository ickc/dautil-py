import sys
from functools import partial
from typing import List, Callable, Optional, Union, Tuple
from collections.abc import Iterable
from pathlib import Path

import requests
from readability import Document
from bs4 import BeautifulSoup
import pypandoc

HEADERS = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
}


def requests_readable(
    urls: List[str],
    texts: List[str] = None,
    idxs: Optional[Union[List[Union[int, str]], Callable]] = None,
    titles: Optional[List[str]] = None,
    headers: dict = HEADERS,
    encoding: Optional[str] = None,
    is_path: bool = False,
) -> List[Tuple[Union[int, str], str, str]]:
    '''request web and scrape for content by readability algorithm

    :param list urls: list of urls
    :param list texts: list of html texts of urls if not None. If None, request from web
    :param idxs: can be None, list, or function of url
    :param titles: can be None, list
    :param HEADERS: pass to requests.get
    :param bool is_path: if True, assume urls are local paths instead
    '''
    res = []
    for i, url in enumerate(urls):
        # index
        if idxs is None:
            idx = None
        else:
            if isinstance(idxs, Iterable):
                idx = idxs[i]
            else:
                idx = idxs(url)
        # text
        if texts is None:
            if is_path:
                if encoding is None:
                    with open(url, 'r') as f:
                        text = f.read()
                else:
                    with open(url, 'rb') as f:
                        try:
                            text = f.read().decode(encoding)
                        except UnicodeDecodeError as e:
                            print((f'Cannot decode text in {url} with encoding {encoding}.'), file=sys.stderr)
                            raise e
            else:
                req = requests.get(url, headers=headers)
                if encoding is None:
                    apparent_encoding = req.apparent_encoding
                    print(f'Setting {url} encoding to {apparent_encoding}...', file=sys.stderr)
                    req.encoding = apparent_encoding
                else:
                    req.encoding = encoding
                text = req.text
        else:
            text = texts[i]
        # readability
        doc = Document(text)
        if titles is None:
            title = doc.title()
        else:
            title = titles[i]
        summary = doc.summary()

        res.append((idx, title, summary))
    return res


def to_markdown(
    res: List[Tuple[Union[int, str], str, str]],
    outpath: Union[Path, str],
    logos: bool = False,
    meta: str = None,
    heading_level: int = 1,
):
    heading_prefix = '#' * heading_level
    with open(outpath, 'w') as f:
        _print = partial(print, file=f, end='\n\n')
        if meta:
            _print(meta)
        for idx, title, summary in res:
            if logos:
                _print(f'[[@Headword+en:{idx}]]')
            if idx:
                _print(f'{heading_prefix} {idx} {title}')
            else:
                _print(f'{heading_prefix} {title}')
            _print(pypandoc.convert_text(summary, 'md', format='html'))
