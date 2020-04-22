from pathlib import Path
import sys
from typing import List

from pymediainfo import MediaInfo
import defopt


def main(*paths: Path, track_types: List[str] = ['Video', 'Audio'], dry_run: bool = False):
    '''detect track type not exists in paths and delete the file

    :param Path paths: media file paths
    :param list[str] track_types: track type understood by mediainfo, e.g. Video, Audio
    :param bool dry_run: if specified, do not delete the file
    '''
    for path in paths:
        media_info = MediaInfo.parse(path)
        for track_type in track_types:
            track_found = None
            for track in media_info.tracks:
                if track.track_type == track_type:
                    track_found = track
                    break
            if track_found is None:
                print(f'{path}')
                if not dry_run:
                    print(f'deleting {path} due to missing {track_type}...', file=sys.stderr)
                    path.unlink()
                else:
                    print(f'(dry run) deleting {path} due to missing {track_type}...', file=sys.stderr)
                break


def cli():
    defopt.run(main)

if __name__ == "__main__":
    cli()
