"""Persistent, cancellable job execution."""

from vast_agent.jobs.cancellation import CancellationToken
from vast_agent.jobs.models import Job, JobStatus

__all__ = ["CancellationToken", "Job", "JobManager", "JobStatus"]


def __getattr__(name: str):
    if name == "JobManager":
        from vast_agent.jobs.manager import JobManager

        return JobManager
    raise AttributeError(name)
