"""Tests for fridom.framework2.io.slurm (mocked environment)."""
import pytest

from fridom.framework2.io import slurm

SCONTROL_OUTPUT = (
    "JobId=123 JobName=run\n"
    "   UserId=me(1000) GroupId=me(1000)\n"
    "   Command=/work/jobs/run.sh --resume\n"
    "   WorkDir=/work/jobs\n")


@pytest.fixture
def calls(monkeypatch):
    """Record subprocess seam calls; fake scontrol/sbatch."""
    recorded = []

    def fake_run(args):
        recorded.append(list(args))
        if args[0].endswith("scontrol"):
            return SCONTROL_OUTPUT
        return "Submitted batch job 124\n"

    monkeypatch.setattr(slurm, "_run", fake_run)
    monkeypatch.setattr(slurm, "_which",
                        lambda name: f"/usr/bin/{name}")
    return recorded


# ================================================================
#  Environment queries
# ================================================================
def test_in_job_and_job_id_inside_an_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert slurm.in_job() is True
    assert slurm.job_id() == "123"


def test_in_job_and_job_id_outside(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert slurm.in_job() is False
    assert slurm.job_id() is None


# ================================================================
#  resubmit_current
# ================================================================
def test_resubmit_outside_an_allocation_raises(monkeypatch, calls):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        slurm.resubmit_current()
    assert calls == []


def test_resubmit_scontrol_to_sbatch(monkeypatch, calls):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    slurm.resubmit_current()
    assert calls == [
        ["/usr/bin/scontrol", "show", "job", "123"],
        ["/usr/bin/sbatch", "/work/jobs/run.sh", "--resume"],
    ]


def test_resubmit_without_procid_still_submits(monkeypatch, calls):
    # single-process launches have no SLURM_PROCID
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    slurm.resubmit_current()
    assert len(calls) == 2


def test_resubmit_rank_guard(monkeypatch, calls):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "3")
    slurm.resubmit_current()  # returns without submitting
    assert calls == []


@pytest.mark.usefixtures("calls")
def test_resubmit_without_command_field(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setattr(slurm, "_run",
                        lambda _args: "JobId=123 JobName=run\n")
    with pytest.raises(RuntimeError, match=r"batch script"):
        slurm.resubmit_current()


def test_resubmit_without_slurm_binaries(monkeypatch, calls):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setattr(slurm, "_which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not found"):
        slurm.resubmit_current()
    assert calls == []


# ================================================================
#  The resubmit() factory (the on_walltime action)
# ================================================================
def test_resubmit_factory_returns_the_action(monkeypatch, calls):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    action = slurm.resubmit()
    assert callable(action)
    assert calls == []  # the factory itself never submits
    action()
    assert len(calls) == 2
