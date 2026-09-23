# Vast Gemini Agent

Vast Gemini Agent は、Windows 11 上で動作し、Discord から LAN 内の Vast.ai Ubuntu
ホストを調査・運用するエージェントです。ユーザーは「何を知りたいか」「どうしたいか」を普段の
言葉で伝えます。Agent は安全な READ 手段を選んで調べ、変更が必要な場合だけ OWNER に承認を
求めます。

## What it can do

- 自然言語による host、GPU、Docker、VM、Vast、network、storage の診断
- 複数ホストの fleet READ と incident history
- executable / package / service の discovery
- installed CLI の bounded `--help` を使った syntax discovery と READ investigation
- 検証済み read-only argv の自動実行
- package install、service/container restart、GPU reset、VM mode change、reboot assessment
- typed action で表現できない変更の structured Operation Plan
- OWNER approval、fresh preflight、exact execution、post-action verification

## 基本方針

**Think freely. Execute only with approval. — 「考えるのは自由、実行は承認制」**

- **READ:** code-side validator が read-only と確認した操作は自動実行できます。
- **WRITE:** Agent は自由に調査・推論して exact plan を作れますが、実行には OWNER の Approve が必須です。
- **Hard safety floor:** 破壊的または security-sensitive な少数の操作は、承認されても実行しません。

安全性は「理解できる言葉を制限する」ことでなく、実行境界で保証します。Agent が自由に調査・
比較・計画しても、承認済みの正確な操作以外は実行できません。

## 処理の流れ

```text
Discord / CLI
  → application-owned host / immutable host-set resolution
  → adaptive READ Discovery / Investigation Agent
  → answer or explicit mutation conclusion
  → Operation Planner
  → structured Proposal
  → OWNER Approve / Reject
  → Fresh, operation-specific Preflight
  → Exact fingerprinted Execution
  → READ Verification
  → Observation / Audit Log
```

- **InvestigationAgent** は目的を理解し、必要な READ evidence を集めます。
- **Router** は明白で安全な READ の latency optimization にすぎず、理解の gate ではありません。
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

## 会話の引き継ぎ

通常の READ は一意の host context を確立します。調査の目的、結論、重要な findings、未解決の
evidence もサイズを制限して保存します（生ログは保存しません）。以後の `二枚あるでしょう？`、
`NVTOP入ってる？`、`nvidia-smi見せて`、`GPU0ちょっと抑えて` は intent keyword を要求せず、
`つまり？`、`ログを見れば理由まで分からない？` も直前の証拠を引き継いで同じ host の Agent に
渡ります。fleet context は単一 host の WRITE authority にはなりません。

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

### Catalogs and registries

**Catalogs are metadata, not permission or intelligence boundaries.** いずれの registry も Gemini が理解してよい概念を制限しません。

- `HOST_READ_CAPABILITIES` は typed arguments、host binding、bounded output を持つ安全な callable READ API surface です。
- acquisition metadata registry は `traffic_history → vnstat`、interface detail → `ethtool`、NVMe health → `nvme-cli` のような、選択された自動取得候補だけを保持します。一般的な capability catalog ではありません。
- generic CLI discovery は acquisition metadata にない installed executable も `query_executable → query_cli_help → run_readonly_argv` で調査できます。
- `operations.allowed_actions` は既存 typed action の execution deployment policy です。
- `operations.generic_operations_enabled` は generic mutation execution の独立 policy です。

`SCOPES` / `WRITE_WORDS` / `AGENT_WORDS` は明白な要求を低 latency で処理する fast path にすぎません。
該当しない host-context message もそのまま Investigation Agent に届きます。mandatory semantic
classifier や `UNCERTAIN` refusal gate はありません。Agent は READ evidence を集めた同じ reasoning
loop で answer / advice / mutation request を区別し、mutation のときだけ planner に引き渡します。

## Fleet は capability ではなく scope

Fleet は capability ではなく、application が固定する immutable host set です。`全台` は enabled
host 全体、複数の alias はその host だけを選び、Gemini が address を追加することはありません。
同じ Investigation Agent と validated READ を host ごとに `max_parallel_hosts` 以下で実行するため、
uptime、kernel version、Docker container 数、`nvidia-smi` などを専用 fleet handler なしで比較できます。
一台が失敗しても、ほかの結果は捨てません。比較、filter、ranking、sum/average/min/max は構造化した
結果から合成します。昨日の vnStat RX/TX/total のように数値の正確さが重要なものは application 側で
sort する高速経路もあります。fleet mutation と batch approval は提供しません。

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
- `operations.generic_operations_enabled`: generic approved argv execution の独立 opt-in（default `false`）
- `operations.allowed_actions`: typed action class ごとの execution policy（default empty）
- `operations.approval_ttl_seconds`: Proposal expiry
- action / reboot / SSH / external tool timeout
- Gemini model、thinking level、step/tool budgets
- Discord OWNER ID と channel ID
- host registry、aliases、capabilities、SSH endpoint
- pinned SSH host keys と最小 passwordless sudo

安全な migration のため、generic approved operation execution は `operations.enabled` と
`operations.generic_operations_enabled` の両方を明示的に有効化するまで default deny です。
Proposal の理解や表示と execution permission は別です。

### Secret の優先順位と Bot の共存

Vast Gemini Agent は、同じ Windows ユーザーで動く旧 Bot のグローバル環境変数に runtime secret を
上書きされません。次の順で、最初に見つかった値を使います。

1. Agent 専用環境変数 `VAST_AGENT_*`（明示的な一時上書き）
2. `%LOCALAPPDATA%\VastGeminiAgent\secrets\secrets.env`
3. 従来の汎用環境変数（互換用 fallback）

| 用途 | Agent 専用（最優先） | secrets.env / 従来変数名 |
|---|---|---|
| Discord Bot token | `VAST_AGENT_DISCORD_BOT_TOKEN` | `DISCORD_BOT_TOKEN` |
| Discord channel | `VAST_AGENT_DISCORD_CHANNEL_ID` | `DISCORD_CHANNEL_ID` |
| Discord OWNER | `VAST_AGENT_DISCORD_OWNER_USER_ID` | `DISCORD_OWNER_USER_ID` |
| Gemini API key | `VAST_AGENT_GEMINI_API_KEY` | `GEMINI_API_KEY` |
| Search API key | `VAST_AGENT_SEARCH_API_KEY` | `SEARCH_API_KEY` |

たとえば旧 Bot の `DISCORD_BOT_TOKEN` がユーザー環境変数に残っていても、Vast Gemini Agent の
`secrets.env` に書いた token が優先されます。環境変数を削除する必要はありません。Agent 専用変数も
runtime file もない既存環境では、従来の変数をそのまま利用できます。secret の値は log に出しません。

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

- Generic mutation execution は approval pipeline に統合済みですが、deployment ごとの明示 opt-in が必要です。
- Generic operation を code-owned verification へ安全に対応付けられない場合は `SUCCEEDED_UNVERIFIED` として記録し、fully verified success とは表示しません。
- Unknown CLI verb は自動 READ ではなく Proposal/clarification に分類されます。
- Tuning は自律 loop ではなく、一回の変更ごとに承認が必要です。
- 実ホストで利用できる evidence は installed tools、sudoers、host capability 設定に依存します。
