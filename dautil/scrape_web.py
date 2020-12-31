from __future__ import annotations

import sys
from functools import partial
from typing import List, Callable, Optional, Union, Tuple
from collections.abc import Iterable
from pathlib import Path

import requests
from readability import Document
import pypandoc

from map_parallel import starmap_parallel

HEADERS = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
}


def _requests_readable_each(
    i: int,
    url: str,
    texts: List[str] = None,
    idxs: Optional[Union[List[Union[int, str]], Callable]] = None,
    titles: Optional[List[str]] = None,
    headers: dict = HEADERS,
    encoding: Union[None, str, Callable] = None,
    is_path: bool = False,
) -> Tuple[Union[int, str], str, str]:
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
                        temp = f.read()
                        if callable(encoding):
                            text = temp.decode(encoding(temp))
                        else:
                            text = temp.decode(encoding)
                    except UnicodeDecodeError as e:
                        print((f'Cannot decode text in {url} with encoding {encoding}.'), file=sys.stderr)
                        raise e
        else:
            req = requests.get(url, headers=headers)
            if callable(encoding):
                req.encoding = encoding(req)
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

    return idx, title, summary


def requests_readable(
    urls: List[str],
    texts: List[str] = None,
    idxs: Optional[Union[List[Union[int, str]], Callable]] = None,
    titles: Optional[List[str]] = None,
    headers: dict = HEADERS,
    encoding: Union[None, str, Callable] = None,
    is_path: bool = False,
) -> List[Tuple[Union[int, str], str, str]]:
    '''request web and scrape for content by readability algorithm

    :param list urls: list of urls
    :param list texts: list of html texts of urls if not None. If None, request from web
    :param idxs: can be None, list, or function of url
    :param titles: can be None, list
    :param headers: pass to requests.get
    :param encoding: can be None, str, or function of the requests object
    :param bool is_path: if True, assume urls are local paths instead

    Example encoding func:

        def regex_charset_func(req, regex_charset=re.compile(r'charset=([^"]+)')):
            charset = regex_charset.findall(req.text)
            if len(charset) == 1:
                return charset[0]
            else:
                print(f"Cannot parse charset from text: {charset}.", file=sys.stderr)
                return None
    '''
    func = partial(
        _requests_readable_each,
        texts=texts,
        idxs=idxs,
        titles=titles,
        headers=headers,
        encoding=encoding,
        is_path=is_path,
    )
    return starmap_parallel(func, enumerate(urls), mode='multithreading')


def _to_markdown_each(
    heading_prefix: str,
    idx: Union[int, str],
    title: str,
    summary: str,
    logos: bool = False,
) -> str:
    res = []
    if idx:
        if logos:
            res.append(f'{heading_prefix} [[@Headword+en:{idx}]]{idx} {title}')
        else:
            res.append(f'{heading_prefix} {idx} {title}')
    else:
        res.append(f'{heading_prefix} {title}')
    res.append(pypandoc.convert_text(summary, 'md', format='html'))
    return '\n\n'.join(res)


def to_markdown(
    res: List[Tuple[Union[int, str], str, str]],
    outpath: Union[Path, str],
    logos: bool = False,
    meta: str = None,
    heading_level: int = 1,
):
    heading_prefix = '#' * heading_level
    func = partial(_to_markdown_each, heading_prefix, logos=logos)
    strings = starmap_parallel(func, res)
    with open(outpath, 'w') as f:
        _print = partial(print, file=f, end='\n\n')
        if meta:
            _print(meta)
        for string in strings:
            _print(string)
