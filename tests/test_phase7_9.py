import asyncio
import logging
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from vast_agent.agent.models import InvestigationResult
from vast_agent.config import HostRegistry
from vast_agent.conversation.state import ConversationStore
from vast_agent.discord_app.auth import DiscordGuard
from vast_agent.discord_app.formatting import split_messages
from vast_agent.discord_app.gateway import DiscordGateway
from vast_agent.execution.ssh import SSHExecutor
from vast_agent.jobs import CancellationToken, JobManager, JobStatus
from vast_agent.jobs.locks import LockClass
from vast_agent.logging_utils import SecretRedactingFormatter
from vast_agent.models.host import Capabilities, Host
from vast_agent.models.tool_result import ErrorCode
from vast_agent.paths import RuntimePaths
from vast_agent.runtime import RuntimeManager
from vast_agent.services.agent_service import AgentService, ServiceReply
from vast_agent.storage.database import Database


class FakeChannel:
    def __init__(self, channel_id=20):
        self.id = channel_id; self.sent = []

    async def send(self, text): self.sent.append(text)


def message(owner=10, channel=20, bot=False, text="hello"):
    return SimpleNamespace(author=SimpleNamespace(id=owner, bot=bot),
                           channel=FakeChannel(channel), content=text)


def test_owner_channel_and_bot_guards():
    guard = DiscordGuard(10, 20)
    assert guard.accepts_message(message())
    assert not guard.accepts_message(message(owner=11))
    assert not guard.accepts_message(message(channel=21))
    assert not guard.accepts_message(message(bot=True))
    assert guard.accepts_interaction(SimpleNamespace(user=SimpleNamespace(id=10)))
    assert not guard.accepts_interaction(SimpleNamespace(user=SimpleNamespace(id=11)))


def test_gateway_ignores_unauthorized_without_service_call():
    class Service:
        calls = 0
        async def handle_question(self, *args): self.calls += 1; return ServiceReply("ok")
    service = Service(); gateway = DiscordGateway(DiscordGuard(10, 20), service)
    async def run():
        for item in (message(owner=11), message(channel=21), message(bot=True)):
            await gateway.on_message(item)
        await gateway.drain()
    asyncio.run(run())
    assert service.calls == 0


def test_gateway_owner_and_message_splitting():
    class Service:
        async def handle_question(self, *args): return ServiceReply("x" * 2001)
    item = message(); gateway = DiscordGateway(DiscordGuard(10, 20), Service())
    async def run(): await gateway.on_message(item); await gateway.drain()
    asyncio.run(run())
    assert len(item.channel.sent) == 3  # one progress message and two bounded result chunks
    assert all(len(chunk) <= 1900 for chunk in split_messages("x" * 2001))


def test_job_persistence_cancel_and_reconcile(tmp_path):
    db = Database(tmp_path / "db.sqlite"); manager = JobManager(db)
    job, token = manager.create("investigate", "one", "request"); manager.start(job)
    assert manager.cancel(job.id) == "cancellation requested"; assert token.cancelled
    manager.finish(job, "cancelled"); assert db.get_job(job.id)["status"] == JobStatus.CANCELLED
    assert manager.cancel(job.id) == "already finished"
    stale, _ = manager.create("gpu", "two", "request")
    another = JobManager(db)
    assert db.get_job(stale.id)["status"] == JobStatus.QUEUED
    another.reconcile_stale()
    assert db.get_job(stale.id)["status"] == JobStatus.INTERRUPTED


def test_bare_stop_cancels_last_job(tmp_path):
    db = Database(tmp_path / "db"); manager = JobManager(db)
    job, token = manager.create("investigate", "torrent", "調査"); manager.start(job)
    store = ConversationStore(db)
    state = store.get(1, 2); state.last_job_id = job.id; state.last_host = "torrent"; store.save(1, 2, state)
    service = AgentService(HostRegistry(hosts={}), object(), object(), manager, store)
    reply = asyncio.run(service.handle_question("止めて", 1, 2))
    assert reply.job_id == job.id and "cancellation requested" in reply.text
    assert token.cancelled
    assert db.get_job(job.id)["status"] == JobStatus.CANCELLING


def test_write_stop_does_not_cancel_last_job(tmp_path):
    for command in ("vastai止めて", "docker止めて"):
        root = tmp_path / command
        db = Database(root / "db"); manager = JobManager(db)
        job, token = manager.create("investigate", "torrent", "調査"); manager.start(job)
        store = ConversationStore(db)
        state = store.get(1, 2); state.last_job_id = job.id; state.last_host = "torrent"; store.save(1, 2, state)
        service = AgentService(HostRegistry(hosts={}), object(), object(), manager, store)
        reply = asyncio.run(service.handle_question(command, 1, 2))
        assert "WRITE/Approval" in reply.text
        assert not token.cancelled
        assert db.get_job(job.id)["status"] == JobStatus.RUNNING


def test_host_coordination_serializes_heavy_but_not_different_hosts(tmp_path):
    locks = JobManager(Database(tmp_path / "db")).locks
    active = 0; peak = 0; guard = threading.Lock()
    def run(host):
        nonlocal active, peak
        with locks.acquire(host, LockClass.HEAVY_READ):
            with guard: active += 1; peak = max(peak, active)
            time.sleep(.05)
            with guard: active -= 1
    threads = [threading.Thread(target=run, args=("same",)) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert peak == 1
    threads = [threading.Thread(target=run, args=(host,)) for host in ("one", "two")]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert peak == 2


def test_ssh_cancellation_terminates_real_child(tmp_path, monkeypatch, host):
    real_popen = subprocess.Popen
    def sleeping_popen(*args, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", sleeping_popen)
    token = CancellationToken(); result = {}
    thread = threading.Thread(target=lambda: result.setdefault(
        "value", SSHExecutor(tmp_path / "known").execute(host, ("true",), 20, token)))
    started = time.monotonic(); thread.start(); time.sleep(.15); token.cancel(); thread.join(3)
    assert not thread.is_alive() and time.monotonic() - started < 3
    assert result["value"].error_code == ErrorCode.CANCELLED


def _service(tmp_path, executor, hosts):
    db = Database(tmp_path / "db"); jobs = JobManager(db)
    class Inspection:
        active = 0; peak = 0; calls = 0
        def inspect_and_record(self, host, remote, scope, token):
            self.calls += 1
            self.active += 1; self.peak = max(self.peak, self.active); time.sleep(.03); self.active -= 1
            gpu = SimpleNamespace(nvml_ok=True, devices=[SimpleNamespace(model="GPU", temperature_c=42,
                                  utilization_percent=0)], pci_count=1)
            obs = SimpleNamespace(host=host.name, ssh_ok=True, signatures=[], gpu=gpu,
                                  system=SimpleNamespace(filesystem_max_percent=10, failed_units=0))
            return SimpleNamespace(observation=obs)
    inspection = Inspection()
    service = AgentService(HostRegistry(hosts={h.name: h for h in hosts}), inspection, executor,
                           jobs, ConversationStore(db), max_parallel_hosts=2)
    return service, inspection


def test_followup_last_host_and_no_arbitrary_host(tmp_path):
    host = Host(name="torrent", address="192.0.2.1", ssh_user="u",
                capabilities=Capabilities(nvidia=True))
    service, _ = _service(tmp_path, object(), [host])
    async def run():
        first = await service.handle_question("torrentのGPU温度", 1, 2)
        second = await service.handle_question("ついでにPCIも", 1, 2)
        return first, second
    first, second = asyncio.run(run())
    assert first.job_id and second.job_id
    assert service.jobs.database.get_job(second.job_id)["host"] == "torrent"
    other, _ = _service(tmp_path / "other", object(), [host])
    reply = asyncio.run(other.handle_question("それも見て", 1, 2))
    assert reply.job_id is None and "host" in reply.text


def test_remember_last_host_false_disables_followup(tmp_path):
    host = Host(name="torrent", address="192.0.2.1", ssh_user="u",
                capabilities=Capabilities(nvidia=True))
    service, _ = _service(tmp_path, object(), [host])
    service.remember_last_host = False
    async def run():
        await service.handle_question("torrentのGPU温度", 1, 2)
        return await service.handle_question("ついでにPCIも", 1, 2)
    reply = asyncio.run(run())
    assert reply.job_id is None and "host" in reply.text
    assert service.conversations.get(1, 2).last_host is None


def test_write_is_refused_without_remote_and_fleet_is_bounded(tmp_path):
    hosts = [Host(name=f"host-{i}", address=f"192.0.2.{i}", ssh_user="u",
                  capabilities=Capabilities(nvidia=True)) for i in range(5)]
    service, inspection = _service(tmp_path, object(), hosts)
    refused = asyncio.run(service.handle_question("host-1をrebootして", 1, 2))
    assert refused.job_id is None and "WRITE/Approval" in refused.text
    fleet = asyncio.run(service.handle_question("全台GPU状態見て", 1, 2))
    assert fleet.job_id and inspection.peak <= 2


def test_runtime_duplicate_and_stale_pid_safety(tmp_path, monkeypatch):
    runtime = RuntimeManager(RuntimePaths(tmp_path))
    monkeypatch.setattr(runtime, "status", lambda: ("RUNNING", {"pid": 123}))
    assert runtime.start() == (False, "already running")
    monkeypatch.setattr(runtime, "status", lambda: ("STOPPED / stale pid", {"pid": 123}))
    ok, detail = runtime.stop()
    assert not ok and "refusing" in detail


def test_fleet_cancellation_does_not_start_remaining_hosts(tmp_path):
    hosts = [Host(name=f"host-{i}", address=f"192.0.2.{i}", ssh_user="u",
                  capabilities=Capabilities(nvidia=True)) for i in range(8)]
    service, inspection = _service(tmp_path, object(), hosts)
    async def run():
        task = asyncio.create_task(service.handle_question("全台GPU状態見て", 1, 2))
        while not service.jobs.running(): await asyncio.sleep(0)
        service.jobs.cancel(int(service.jobs.running()[0]["id"]))
        return await task
    result = asyncio.run(run())
    assert service.jobs.database.get_job(result.job_id)["status"] == JobStatus.CANCELLED
    assert inspection.calls < len(hosts)


def test_fleet_host_failure_continues_and_finishes_parent(tmp_path):
    hosts = [Host(name=name, address="192.0.2.1", ssh_user="u",
                  capabilities=Capabilities(nvidia=True)) for name in ("bad", "good")]
    service, inspection = _service(tmp_path, object(), hosts)
    original = inspection.inspect_and_record
    def inspect(host, *args):
        if host.name == "bad": raise RuntimeError("boom")
        return original(host, *args)
    inspection.inspect_and_record = inspect
    reply = asyncio.run(service.handle_question("全台GPU状態見て", 1, 2))
    row = service.jobs.database.get_job(reply.job_id)
    assert "bad" in reply.text and "ERROR" in reply.text and "good" in reply.text
    assert row["status"] == JobStatus.FAILED


def test_fleet_and_investigation_same_host_do_not_overlap(tmp_path):
    host = Host(name="torrent", address="192.0.2.1", ssh_user="u",
                capabilities=Capabilities(nvidia=True))
    service, inspection = _service(tmp_path, object(), [host])
    active = 0; peak = 0
    original = inspection.inspect_and_record
    def enter_and_wait(callback):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        try: time.sleep(.06); return callback()
        finally: active -= 1
    def inspect(*args): return enter_and_wait(lambda: original(*args))
    inspection.inspect_and_record = inspect
    class Agent:
        def investigate(self, *args):
            return enter_and_wait(lambda: InvestigationResult(summary="ok"))
    service.agent = Agent()
    async def run():
        await asyncio.gather(
            service.handle_question("torrentの原因調べて", 1, 2),
            service.handle_question("全台GPU状態見て", 1, 2),
        )
    asyncio.run(run())
    assert peak == 1


def test_secret_redaction_at_service_database_gemini_and_logging(tmp_path):
    secret = "super-secret-value"
    host = Host(name="torrent", address="192.0.2.1", ssh_user="u",
                capabilities=Capabilities(nvidia=True))
    db = Database(tmp_path / "db"); jobs = JobManager(db); seen = {}
    class Agent:
        def investigate(self, host, question):
            seen["question"] = question
            return InvestigationResult(summary=f"model echoed {secret}")
    service = AgentService(HostRegistry(hosts={host.name: host}), object(), object(), jobs,
                           ConversationStore(db), Agent(), secrets=(secret,))
    reply = asyncio.run(service.handle_question(f"torrentの原因調べて {secret}", 1, 2))
    row = db.get_job(reply.job_id)
    assert secret not in seen["question"]
    assert secret not in row["request_summary"] and secret not in row["result_summary"]
    assert secret not in reply.text

    formatter = SecretRedactingFormatter("%(message)s", (secret,))
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, f"failed {secret}", (),
                                   sys.exc_info())
    assert secret not in formatter.format(record)


def test_observation_severity_is_not_green_for_failures():
    def observation(ssh_ok=True, nvml_ok=True, signatures=(), devices=True):
        gpu = SimpleNamespace(nvml_ok=nvml_ok, devices=(
            [SimpleNamespace(model="GPU", temperature_c=42, utilization_percent=0)] if devices else []))
        return SimpleNamespace(host="host", ssh_ok=ssh_ok, signatures=list(signatures), gpu=gpu)
    assert AgentService._format_observation(observation()).startswith("🟢")
    assert AgentService._format_observation(observation(ssh_ok=False)).startswith("🔴")
    assert not AgentService._format_observation(observation(nvml_ok=False)).startswith("🟢")
    assert not AgentService._format_observation(observation(signatures=("NVML_UNAVAILABLE",))).startswith("🟢")
