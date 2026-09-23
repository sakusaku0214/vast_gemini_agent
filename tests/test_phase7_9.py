import asyncio
import logging
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from vast_agent.approval import ProposalStore
from vast_agent.config import HostRegistry
from vast_agent.conversation.state import ConversationStore
from vast_agent.discord_app.auth import DiscordGuard
from vast_agent.discord_app.formatting import split_messages
from vast_agent.discord_app.gateway import DiscordGateway
from vast_agent.execution.ssh import SSHExecutor
from vast_agent.jobs import CancellationToken, JobManager, JobStatus
from vast_agent.logging_utils import SecretRedactingFormatter
from vast_agent.models.host import Host
from vast_agent.models.tool_result import ErrorCode, ToolResult
from vast_agent.paths import RuntimePaths
from vast_agent.runtime import RuntimeManager
from vast_agent.services.agent_service import AgentService, ServiceReply
from vast_agent.storage.database import Database


class FakeChannel:
    def __init__(self, channel_id=20):
        self.id = channel_id
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


def message(owner=10, channel=20, bot=False, text="hello"):
    return SimpleNamespace(
        author=SimpleNamespace(id=owner, bot=bot),
        channel=FakeChannel(channel),
        content=text,
    )


def test_owner_channel_and_bot_guards():
    guard = DiscordGuard(10, 20)
    assert guard.accepts_message(message())
    assert not guard.accepts_message(message(owner=11))
    assert not guard.accepts_message(message(channel=21))
    assert not guard.accepts_message(message(bot=True))


def test_gateway_ignores_unauthorized_without_service_call():
    class Service:
        calls = 0

        async def handle_question(self, *args):
            self.calls += 1
            return ServiceReply("ok")

    service = Service()
    gateway = DiscordGateway(DiscordGuard(10, 20), service)

    async def run():
        for item in (message(owner=11), message(channel=21), message(bot=True)):
            await gateway.on_message(item)
        await gateway.drain()

    asyncio.run(run())
    assert service.calls == 0


def test_gateway_owner_and_message_splitting():
    class Service:
        async def handle_question(self, *args):
            return ServiceReply("x" * 2001)

    item = message()
    gateway = DiscordGateway(DiscordGuard(10, 20), Service())

    async def run():
        await gateway.on_message(item)
        await gateway.drain()

    asyncio.run(run())
    assert len(item.channel.sent) == 3
    assert all(len(chunk) <= 1900 for chunk in split_messages("x" * 2001))


def test_job_persistence_cancel_and_reconcile(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    manager = JobManager(db)
    job, token = manager.create("agent", None, "request")
    manager.start(job)
    assert manager.cancel(job.id) == "cancellation requested"
    assert token.cancelled
    manager.finish(job, "cancelled")
    assert db.get_job(job.id)["status"] == JobStatus.CANCELLED

    stale, _ = manager.create("agent", None, "request")
    another = JobManager(db)
    assert db.get_job(stale.id)["status"] == JobStatus.QUEUED
    another.reconcile_stale()
    assert db.get_job(stale.id)["status"] == JobStatus.INTERRUPTED


def test_bare_stop_cancels_last_job(tmp_path):
    db = Database(tmp_path / "db")
    manager = JobManager(db)
    job, token = manager.create("agent", None, "調査")
    manager.start(job)
    store = ConversationStore(db)
    state = store.get(1, 2)
    state.last_job_id = job.id
    store.save(1, 2, state)

    service = AgentService(HostRegistry(hosts={}), object(), object(), manager, store)
    reply = asyncio.run(service.handle_question("止めて", 1, 2))

    assert reply.job_id == job.id
    assert "cancellation requested" in reply.text
    assert token.cancelled


def test_natural_language_is_passed_to_agent_without_routing(tmp_path):
    db = Database(tmp_path / "db")
    manager = JobManager(db)
    store = ConversationStore(db)
    seen = {}

    class Agent:
        def investigate(self, host, question, **kwargs):
            seen["host"] = host
            seen["question"] = question
            return "Gemini answer"

    service = AgentService(
        HostRegistry(hosts={}), object(), object(), manager, store, Agent()
    )
    request = "全台の今日の通信量をランキング付けて一覧にして"
    reply = asyncio.run(service.handle_question(request, 1, 2))

    assert seen == {"host": None, "question": request}
    assert "Gemini answer" in reply.text
    row = db.get_job(reply.job_id)
    assert row["host"] is None
    assert row["kind"] == "agent"


def test_host_words_do_not_trigger_code_side_semantics(tmp_path):
    db = Database(tmp_path / "db")
    seen = []

    class Agent:
        def investigate(self, host, question, **kwargs):
            seen.append(question)
            return "ok"

    service = AgentService(
        HostRegistry(hosts={}),
        object(),
        object(),
        JobManager(db),
        ConversationStore(db),
        Agent(),
    )

    for text in (
        "garage-torrentのGPU温度を見て",
        "全台GPU状態見て",
        "vastai再起動して",
        "ついでにPCIも",
    ):
        asyncio.run(service.handle_question(text, 1, 2))

    assert seen == [
        "garage-torrentのGPU温度を見て",
        "全台GPU状態見て",
        "vastai再起動して",
        "ついでにPCIも",
    ]


def test_ssh_cancellation_terminates_real_child(tmp_path, monkeypatch, host):
    real_popen = subprocess.Popen

    def sleeping_popen(*args, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", sleeping_popen)
    token = CancellationToken()
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "value", SSHExecutor(tmp_path / "known").execute(host, ("true",), 20, token)
        )
    )
    started = time.monotonic()
    thread.start()
    time.sleep(.15)
    token.cancel()
    thread.join(3)
    assert not thread.is_alive()
    assert time.monotonic() - started < 3
    assert result["value"].error_code == ErrorCode.CANCELLED


def test_runtime_duplicate_and_stale_pid_safety(tmp_path, monkeypatch):
    paths = RuntimePaths(tmp_path)
    paths.create()
    paths.process_state.write_text('{"pid":123}', encoding="utf-8")
    runtime = RuntimeManager(paths)

    monkeypatch.setattr(runtime, "status", lambda: ("RUNNING", {"pid": 123}))
    assert runtime.start() == (False, "already running")

    monkeypatch.setattr(runtime, "status", lambda: ("STOPPED / stale pid", {"pid": 123}))
    ok, detail = runtime.stop()
    assert ok
    assert detail == "stale PID state cleared"
    assert not paths.process_state.exists()


def test_secret_redaction_at_service_database_and_logging(tmp_path):
    secret = "super-secret-value"
    db = Database(tmp_path / "db")
    seen = {}

    class Agent:
        def investigate(self, host, question, **kwargs):
            seen["question"] = question
            return f"model echoed {secret}"

    service = AgentService(
        HostRegistry(hosts={}),
        object(),
        object(),
        JobManager(db),
        ConversationStore(db),
        Agent(),
        secrets=(secret,),
    )
    reply = asyncio.run(service.handle_question(f"調べて {secret}", 1, 2))
    row = db.get_job(reply.job_id)

    assert secret not in seen["question"]
    assert secret not in row["request_summary"]
    assert secret not in row["result_summary"]
    assert secret not in reply.text

    formatter = SecretRedactingFormatter("%(message)s", (secret,))
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, f"failed {secret}", (), sys.exc_info()
        )
    assert secret not in formatter.format(record)



class WriteRecordingExecutor:
    def __init__(self, result=None):
        self.result = result or ToolResult(
            success=True, exit_code=0, stdout="restarted", duration_ms=1,
        )
        self.calls = []

    def execute(self, host, command, timeout, cancellation=None):
        self.calls.append((host.name, tuple(command), timeout, cancellation))
        return self.result


def make_write_service(tmp_path, owner="10", channel="20"):
    db = Database(tmp_path / "db")
    db.migrate()
    host = Host(name="torrent", address="192.0.2.10", ssh_user="tester")
    registry = HostRegistry(hosts={host.name: host})
    executor = WriteRecordingExecutor()
    proposals = ProposalStore(db)
    service = AgentService(
        registry,
        object(),
        executor,
        JobManager(db),
        ConversationStore(db),
        agent=None,
        proposals=proposals,
    )
    proposal = proposals.create(
        owner_id=owner,
        channel_id=channel,
        host=host.name,
        argv=("sudo", "-n", "systemctl", "restart", "vastai"),
        reason="restart requested",
        timeout=30,
    )
    return service, executor, proposals, proposal


def test_write_is_not_executed_before_owner_approval(tmp_path):
    service, executor, proposals, proposal = make_write_service(tmp_path)
    assert executor.calls == []
    assert proposals.get(proposal.id).status == "PENDING"


def test_owner_approval_executes_exact_stored_argv_once(tmp_path):
    service, executor, proposals, proposal = make_write_service(tmp_path)

    first = asyncio.run(service.handle_question(f"承認 #{proposal.id}", 10, 20))
    second = asyncio.run(service.handle_question(f"承認 #{proposal.id}", 10, 20))

    assert "EXECUTED" in first.text
    assert len(executor.calls) == 1
    assert executor.calls[0][1] == (
        "sudo", "-n", "systemctl", "restart", "vastai",
    )
    assert proposals.get(proposal.id).status == "EXECUTED"
    assert "承認待ちではありません" in second.text


def test_wrong_owner_cannot_approve_proposal(tmp_path):
    service, executor, proposals, proposal = make_write_service(tmp_path)

    reply = asyncio.run(service.handle_question(f"承認 #{proposal.id}", 11, 20))

    assert executor.calls == []
    assert proposals.get(proposal.id).status == "PENDING"
    assert "承認できません" in reply.text


def test_bare_approval_works_only_with_one_pending_proposal(tmp_path):
    service, executor, proposals, proposal = make_write_service(tmp_path)

    reply = asyncio.run(service.handle_question("承認", 10, 20))

    assert "EXECUTED" in reply.text
    assert len(executor.calls) == 1
    assert proposals.get(proposal.id).status == "EXECUTED"


def test_bare_approval_requires_id_when_multiple_pending(tmp_path):
    service, executor, proposals, _ = make_write_service(tmp_path)
    proposals.create(
        owner_id="10",
        channel_id="20",
        host="torrent",
        argv=("sudo", "-n", "systemctl", "restart", "docker"),
        reason="second proposal",
        timeout=30,
    )

    reply = asyncio.run(service.handle_question("承認", 10, 20))

    assert executor.calls == []
    assert "承認対象を指定" in reply.text
    assert "#1" in reply.text and "#2" in reply.text


def test_failed_write_is_single_use(tmp_path):
    service, executor, proposals, proposal = make_write_service(tmp_path)
    executor.result = ToolResult(
        success=False, exit_code=1, stderr="failed", duration_ms=1,
    )

    first = asyncio.run(service.handle_question(f"承認 #{proposal.id}", 10, 20))
    second = asyncio.run(service.handle_question(f"承認 #{proposal.id}", 10, 20))

    assert "FAILED" in first.text
    assert len(executor.calls) == 1
    assert proposals.get(proposal.id).status == "FAILED"
    assert "承認待ちではありません" in second.text



def test_ssh_known_hosts_option_has_no_embedded_quotes(tmp_path, monkeypatch, host):
    known = tmp_path / "folder with spaces" / "known_hosts"
    known.parent.mkdir()
    known.write_text("", encoding="utf-8")
    captured = {}

    class Proc:
        returncode = 0

        def communicate(self, timeout=None):
            return b"ok", b""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 1

        def kill(self):
            self.returncode = 1

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = SSHExecutor(known).execute(host, ("nvidia-smi",), 5)

    assert result.success
    option = next(
        captured["args"][i + 1]
        for i, value in enumerate(captured["args"][:-1])
        if value == "-o" and captured["args"][i + 1].startswith("UserKnownHostsFile=")
    )
    expected = known.resolve().as_posix().replace("\\", "\\\\").replace(" ", "\\ ")
    assert option == f"UserKnownHostsFile={expected}"
    assert '"' not in option
    assert "\\ " in option



def test_windows_runtime_stop_uses_taskkill(tmp_path, monkeypatch):
    paths = RuntimePaths(tmp_path)
    paths.create()
    runtime = RuntimeManager(paths)
    paths.process_state.write_text(
        '{"pid":321,"started_at":"2026-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )

    monkeypatch.setattr("vast_agent.runtime.os.name", "nt")
    monkeypatch.setattr(
        runtime,
        "status",
        lambda: ("RUNNING", {"pid": 321}),
    )

    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    seen = iter(["python vast-agent-run-discord", None])
    monkeypatch.setattr(runtime, "_command_line", lambda pid: next(seen))
    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, detail = runtime.stop(grace=0.2)

    assert ok
    assert detail == "stopped"
    assert ("taskkill", "/PID", "321") in calls
    assert not paths.process_state.exists()
