#!/usr/bin/env python3

import itertools
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import defopt
import pandas as pd
import pymediainfo


@dataclass(frozen=True)
class ParseMediaInfo:
    """Class to parse media information from a file using pymediainfo."""

    path: Path
    general_attrs: ClassVar[list[str]] = [
        "format",
        "format_version",
        "file_size",
        "overall_bit_rate_mode",
        "overall_bit_rate",
        "frame_rate",
        "title",
        "movie_name",
        "encoded_date",
        "writing_application",
        "encoded_application_name",
        "encoded_application_version",
        "writing_library",
        "encoded_library_name",
        "encoded_library_version",
    ]
    video_attrs: ClassVar[list[str]] = [
        "format",
        "format_info",
        "commercial_name",
        "format_profile",
        "hdr_format",
        "hdr_format_commercial",
        "hdr_format_version",
        "hdr_format_version",
        "hdr_format_profile",
        "hdr_format_level",
        "hdr_format_settings",
        "hdr_format_compression",
        "hdr_format_compatibility",
        "internet_media_type",
        "codec_id",
        "bit_rate",
        "width",
        "height",
        "pixel_aspect_ratio",
        "frame_rate_mode",
        "frame_rate",
        "color_space",
        "chroma_subsampling",
        "bit_depth",
        "bits__pixel_frame",
        "stream_size",
        "language",
        "color_range",
        "color_primaries",
        "matrix_coefficients",
        "mastering_display_color_primaries",
        "original_source_medium",
    ]

    @cached_property
    def meta(self) -> pymediainfo.MediaInfo:
        return pymediainfo.MediaInfo.parse(self.path)

    @property
    def general_track(self) -> pymediainfo.Track:
        for track in self.meta.tracks:
            if track.track_type == "General":
                return track
        raise ValueError(f"No general track found in {self.path}")

    @property
    def first_video_track(self) -> pymediainfo.Track | None:
        for track in self.meta.tracks:
            if track.track_type == "Video":
                return track
        return None

    @property
    def track_type_dict(self) -> dict[str, int]:
        return {
            f"track_type_{key}": value
            for key, value in Counter(
                track.track_type for track in self.meta.tracks
            ).items()
        }

    @property
    def general_info_dict(self) -> dict[str, str | None]:
        return {
            f"general_{attr}": getattr(self.general_track, attr, None)
            for attr in self.general_attrs
        }

    @property
    def video_info_dict(self) -> dict[str, str | None]:
        if not (video := self.first_video_track):
            return {f"video_{attr}": None for attr in self.video_attrs}
        return {
            f"video_{attr}": getattr(video, attr, None) for attr in self.video_attrs
        }

    @property
    def info_dict(self) -> dict[str, str | int | None]:
        return {
            "path": str(self.path),
            **self.track_type_dict,
            **self.general_info_dict,
            **self.video_info_dict,
        }

    @property
    def info_series(self) -> pd.Series:
        return pd.Series(self.info_dict)


def main(
    *glob_patterns: str,
    output: Path,
    base_dir: Path = Path.cwd(),
) -> None:
    """Main function to run the media processing pipeline.

    Args:
        glob_patterns: One or more glob patterns to match files.
        output: Path to the output CSV file.
        base_dir: Base directory to search for files. Defaults to the current working directory.
    """
    glob_iterators = (base_dir.glob(pattern) for pattern in glob_patterns)
    paths_to_process = itertools.chain.from_iterable(glob_iterators)
    dict_generator = (ParseMediaInfo(path).info_dict for path in paths_to_process)
    media_df = pd.DataFrame.from_records(dict_generator)
    media_df.to_csv(output, index=False)
    print(f"✅ Success! Media information saved to '{output}'")


if __name__ == "__main__":
    defopt.run(main)
