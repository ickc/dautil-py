#!/usr/bin/env python

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from logging import getLogger

import defopt
import pandas as pd

logger = getLogger(__name__)
version = "0.1.0"

USER = os.environ["USER"]
TIME_COLS = ["accrue_time", "eligible_time", "end_time", "start_time", "submit_time"]
TIME_DELTA_COLS = ["time_limit"]


def squeue_data() -> pd.DataFrame:
    res = subprocess.run(["squeue", "--json"], capture_output=True, check=True)
    stdout = res.stdout.decode("utf-8")

    data = json.loads(stdout)
    df = pd.DataFrame(data["jobs"])
    df.sort_values("job_id", inplace=True)

    # to datatime
    for col in TIME_COLS:
        df[col] = pd.Index(
            pd.to_datetime(
                df[col],
                unit="s",
                utc=True,
            )
        ).tz_convert("America/Los_Angeles")

    # to timedelta
    for col in TIME_DELTA_COLS:
        df[col] = pd.to_timedelta(df[col], unit="m")

    # replicate the JOBID as shown by squeue
    job_ids = []
    for _, row in df.iterrows():
        if pd.isna(row.array_task_id):
            # is a job-array
            if row.array_task_string:
                job_id = f"{row.array_job_id}_[{row.array_task_string}]"
            # is a simple job
            else:
                job_id = str(row.job_id)
        # is a job that belongs to a job-array
        else:
            job_id = f"{row.array_job_id}_{int(row.array_task_id)}"
        job_ids.append(job_id)
    df["JOBID"] = job_ids
    df.set_index("JOBID", inplace=True)
    return df


@dataclass
class Squeue:
    """Pandas-formatted squeue.

    :param list_columns: If True, list all columns.
    :param columns_containing: If specified, only show columns containing this string.
    :param columns: List of columns to display.
    :param user: Show jobs that belong to this user.
    :param me: If True, only show jobs that belong to the current user.
    :param features: If specified, only show jobs that have these features.
    :param qos: If specified, only show jobs that have this qos.
    :param qos_containing: If specified, only show jobs that have qos containing this string.
    :param partition: If specified, only show jobs that have this partition.
    """

    list_columns: bool = False
    columns_containing: str = ""
    columns: tuple[str] = (
        "job_state",
        "user_name",
        "name",
        "node_count",
        "time_limit",
        "submit_time",
        "qos",
        "start_time",
        "features",
        "state_reason",
    )
    user: str = ""
    me: bool = False
    features: str = ""
    qos: str = ""
    qos_containing: str = ""
    partition: str = ""

    def __post_init__(self) -> None:
        pd.options.display.max_rows = None
        pd.options.display.max_columns = 0
        pd.options.display.width = None

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        return squeue_data()

    def __str__(self) -> str:
        if self.list_columns:
            return "\n".join(self.data_frame.columns)

        cols = (
            [
                col
                for col in self.data_frame.columns
                if self.columns_containing in col.lower()
            ]
            if self.columns_containing
            else list(self.columns)
        )

        df = self.data_frame
        if self.user:
            df = df[df.user_name == self.user]
        elif self.me:
            df = df[df.user_name == USER]

        if self.features:
            df = df[df.features == self.features]

        if self.qos:
            df = df[df.qos == self.qos]
        elif self.qos_containing:
            df = df[df.qos.str.contains(self.qos_containing)]

        if self.partition:
            df = df[df.partition == self.partition]

        return str(df[cols])


if __name__ == "__main__":
    print(
        defopt.run(
            Squeue,
            strict_kwonly=False,
            show_defaults=True,
            show_types=False,
            no_negated_flags=True,
            version=version,
        )
    )
