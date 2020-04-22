from pathlib import Path
import sys

from pymediainfo import MediaInfo
import defopt


def main(*paths: Path, track_type: str = 'Video', dry_run: bool = False):
    '''detect track type not exists in paths and delete the file

    :param Path paths: media file paths
    :param str track_type: track type understood by mediainfo, e.g. Video, Audio
    :param bool dry_run: if specified, do not delete the file
    '''
    for path in paths:
        media_info = MediaInfo.parse(path)
        track_found = None
        for track in media_info.tracks:
            if track.track_type == track_type:
                track_found = track
                break
        if track_found is None:
            print(f'{path}')
            if not dry_run:
                print(f'deleting {path}...', file=sys.stderr)
                path.unlink()
            else:
                print(f'(dry run) deleting {path}...', file=sys.stderr)


def cli():
    defopt.run(main)

if __name__ == "__main__":
    cli()
