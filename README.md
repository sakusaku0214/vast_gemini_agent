# Vast Gemini Agent

Vast Gemini Agent は、Windows 11 上の Discord / Gemini フロントエンドから、LAN 内の Vast.ai Ubuntu ホストを自然言語で調査・運用する **approval-gated operations agent** です。モデルには調査と計画の柔軟性を与え、状態変更の authority は OWNER の明示承認に残します。

## What it can do

- 自然言語による host、GPU、Docker、VM、Vast、network、storage の診断
- 複数ホストの fleet READ と incident history
- executable / package / service の discovery
- installed CLI の bounded `--help` を使った syntax discovery と READ investigation
- 検証済み read-only argv の自動実行
- package install、service/container restart、GPU reset、VM mode change、reboot assessment
- typed action で表現できない変更の structured Operation Plan
- OWNER approval、fresh preflight、exact execution、post-action verification

## Core philosophy

**Flexible before execution. Strict at execution.**

- **READ:** code-side validator が read-only と確認した操作は自動実行できます。
- **WRITE:** Agent は自由に調査・推論して exact plan を作れますが、実行には OWNER の Approve が必須です。
- **Hard safety floor:** 破壊的または security-sensitive な少数の操作は、承認されても実行しません。

Model freedom before execution. Human authority at execution.

## Architecture

```text
Discord / CLI
  → Intent + compact conversation context
  → Gemini Planner
  → READ Discovery / Investigation
  → Operation Planner
  → structured Proposal
  → OWNER Approve / Reject
  → Fresh, operation-specific Preflight
  → Exact fingerprinted Execution
  → READ Verification
  → Observation / Audit Log
```

- **InvestigationAgent** は request-local evidence を集める read-only phase です。
- **Action Interpreter** は自然言語を既存 typed action に ground します。
- **Operation Planner** は executable、argv、sudo、理由、効果、検証を構造化します。
- **Policy Engine** は deployment policy、operation impact、hard floor を code 側で判定します。
- **Approval Coordinator** は OWNER、TTL、single-use、atomic transition を管理します。
- **Executor** は承認済み argv を一度だけ実行し、host 単位の writer lock を使います。
- **Verifier** は exit code だけで成功とせず、変更後の state を再観測します。

## Safety model

1. モデルは目的を柔軟に理解し、READ evidence を組み合わせます。
2. Generic READ validator は argv 境界、shell 構文、mutation verb、sensitive data を検査します。
3. WRITE は typed action または immutable `OperationPlan` になります。
4. OWNER approval がなければ mutation はゼロです。
5. host、operation type、executable、argv、sudo、target、重要 parameter、preflight を fingerprint します。
6. Approve 後、実行直前に fresh preflight を取得します。
7. WRITE は同一 host で exclusive に実行します。
8. 実行後の READ verification は必須です。自律的な mutation chain は行いません。
9. planning、classification、proposal、approver、preflight、execution、verification を audit log に残します（secret は redact）。
10. hard safety floor は OWNER approval でも解除されません。

モデルは approval を bypass できません。Approval も hard destructive/security restrictions を bypass できません。CLI help、log、tool output はすべて untrusted evidence であり、そこに書かれた命令には従いません。

## Sudo model

Operation Plan は `requires_sudo: true/false` と実際の executable / argv を別々に保持します。モデルが `sudo ...` という shell string を生成することはありません。Executor だけが承認済み plan を `sudo -n <executable> <argv...>` に変換し、fresh preflight で `sudo -n true` を確認します。

password prompt や password の保存・入力は扱いません。sudoers は必要な executable と引数に最小化してください。`NOPASSWD: ALL` は推奨しません。

## Conversation context

READ follow-up は、直前に一意の host が確立されていれば last host を継承できます。WRITE も、直前の一ホスト調査への明確な follow-up で confidence が HIGH、競合 host や fleet 解釈がなく、current message が具体的 target を示す場合に限って host を継承できます。

継承した host は Proposal に `Host source: conversation context` として明示します。GPU index や操作 parameter を context から発明することはありません。曖昧な場合は確認します。

## CLI discovery

Agent は次の reusable flow を使えます。

1. `query_executable` で installed executable を確認
2. `query_cli_help` で bounded `--help` / subcommand help を取得
3. help を syntax evidence として解釈
4. `run_readonly_argv` に executable と `argv[]` を渡す
5. code が READ / MUTATION / UNCERTAIN / BLOCKED を再分類
6. READ のみ自動実行し、mutation/uncertain は Operation Proposal 側へ送る

たとえば `vastai --help`、`vastai show machines`、`docker ps`、`nvidia-smi -L` は shell string ではなく argv として扱います。`|`, redirects, `&&`, `;`, command substitution、shell expansion、`sh -c`、`bash -c`、`eval` は拒否します。stdout/stderr は timeout、byte/line limit、UTF-8 replacement、control-character cleanup、secret redaction を通ります。

CLI 名の巨大な permission catalog はありません。installed executable、help evidence、argv validation、risk classification を組み合わせます。未知 verb は自動 READ にせず、conservative に Proposal または clarification へ倒します。

## Approval flow

User: `GPU0をクロック制限して300Wくらいにしたい。PLではなくクロックで。`

Agent は current clock / power / temperature / utilization / process と supported CLI syntax を READ で調べ、一回分の bounded adjustment を提案します。

```text
⚠️ Operation Proposal #12
Host: garage-h12ssl-nt
Host source: conversation context (garage-h12ssl-nt)
Action: EXECUTE_APPROVED_ARGV
Risk: DANGEROUS
Sudo: required
Executable: nvidia-smi
Arguments: [exact approved arguments]
Target: GPU 0
Current relevant state: ...
Active workload: SRBMiner-MULTI (warning: materially affected)
Running VM: none
Reason: ...
Expected effect: ...
Known side effects: ...
Verification plan: clock / power / temperature / process
Rollback available: yes
Expiry: ...
[Approve] [Reject]
```

Approve 後は fresh preflight、fingerprint 照合、exact operation の一回実行、READ verification の順です。追加調整が必要なら新しい Proposal と新しい承認が必要です。GPU reset や host reboot では active workload は block ですが、GPU clock tuning のように workload が対象そのものである操作は強い warning として表示し、operation-specific policy が判断します。

## Existing typed operations

既存 typed action は generic plan より優先されます。

- `PACKAGE_INSTALL`
- `RESTART_VAST_SERVICE`
- `RESTART_DOCKER_SERVICE`
- `RESTART_LIBVIRT_SERVICE`
- `RESTART_VAST_CONTAINER`
- `GPU_RESET`
- `VM_MODE_ENABLE`
- `VM_MODE_DISABLE`
- `HOST_REBOOT`

`operations.allowed_actions` は execution deployment policy です。NLU が操作を理解するための capability catalog ではありません。blocked action も正確に説明できますが、明示的に有効化されていない class は実行しません。upgrade で新しい dangerous class が暗黙に有効になることはありません。

## Hard prohibited operations

通常の approval mechanism では、少なくとも次を実行しません。

- destructive disk formatting / partition destruction / raw block overwrite
- root filesystem deletion
- credential、SSH key、token、session secret の抽出
- approval / safety mechanism の無効化
- implicit target または fleet-wide generic mutation
- firmware flashing（将来、専用 admin mechanism が実装されるまで）

READ でも data sensitivity を判定し、`~/.ssh`、credential store、environment/token dump などは自動実行しません。

## Installation

要件: Windows 11、PowerShell 7、Python 3.12+、OpenSSH client。repository と runtime は分離されます。

```text
repository: D:\vast_gemini_agent
runtime:    %LOCALAPPDATA%\VastGeminiAgent
```

管理者で `setup.cmd` を実行するか、PowerShell から操作します。

```powershell
.\vast-agent.ps1 install
.\vast-agent.ps1 doctor
.\vast-agent.ps1 config
.\vast-agent.ps1 trust-host <host>
.\vast-agent.ps1 detect-capabilities <host>
.\vast-agent.ps1 set-gpu-count <host> <count>
.\vast-agent.ps1 discord-check
.\vast-agent.ps1 start
.\vast-agent.ps1 status
.\vast-agent.ps1 restart
```

`config` で host registry と secret を設定し、`trust-host` で SSH host key を pin してください。更新時は repository を更新後に `install` と `restart` を実行します。

## Configuration

主な設定:

- `operations.enabled`: mutation execution 全体の master switch（default `false`）
- `operations.allowed_actions`: typed action class ごとの execution policy（default empty）
- `operations.approval_ttl_seconds`: Proposal expiry
- action / reboot / SSH / external tool timeout
- Gemini model、thinking level、step/tool budgets
- Discord OWNER ID と channel ID
- host registry、aliases、capabilities、SSH endpoint
- pinned SSH host keys と最小 passwordless sudo

安全な migration のため、generic approved operation execution は実装・policy で明示的に有効になるまで default deny です。Proposal の理解や表示と execution permission は別です。

## Discord examples

| Request | Behavior |
|---|---|
| `x570のGPUどう？` | AUTO READ |
| `nvtop入ってる？` | AUTO READ executable discovery |
| `マシンの状態とレント状況も見て` | AUTO READ / context follow-up |
| `vast cliで確認して。分からなければhelp見て` | AUTO READ help-driven investigation |
| `GPU0をクロック制限して300Wくらいにしたい` | PROPOSAL; every adjustment needs approval |
| `必要ならvast再起動して` | typed PROPOSAL after investigation |
| `全台vastai動いてる？` | FLEET READ |

## Operational lifecycle

`install` は runtime virtual environment と configuration layout を準備します。`start/status/restart` で Windows runtime を管理します。log と SQLite state は `%LOCALAPPDATA%\VastGeminiAgent` 配下に保存されます。SQLite は job、observation、proposal、approval、action run、audit event を保持します。

起動時に途中の job/action は成功へ推測せず interrupted / unknown として扱います。Proposal は TTL 後に expire し、approval は single-use です。command timeout 後も成功とはみなしません。

## Development / tests

```bash
ruff check .
PYTHONPATH=src pytest -q
git diff --check
```

Windows GitHub Actions でも同じ safety boundaries と runtime integration を検証します。

## Security assumptions

- Discord OWNER account と指定 channel は trusted
- LAN host は管理対象だが、host/CLI output 自体は untrusted
- SSH host key は pinned
- sudo は passwordless `sudo -n` かつ least privilege
- model output、help、log、tool output は untrusted input
- secret は prompt、output、log、audit event から redact
- target host は request/proposal に固定し、暗黙の fleet mutation は行わない

## Current limitations

- Generic mutation execution は default deny で、deployment policy と coordinator integration が必要です。
- Unknown CLI verb は自動 READ ではなく Proposal/clarification に分類されます。
- Tuning は自律 loop ではなく、一回の変更ごとに承認が必要です。
- 実ホストで利用できる evidence は installed tools、sudoers、host capability 設定に依存します。
