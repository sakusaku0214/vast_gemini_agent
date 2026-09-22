# Vast Gemini Agent v2

Windows 11からUbuntu/Vast.aiホストを安全に観測する、Phase 0〜9の **READ ONLY** 実装です。
Geminiは調査にだけ使用し、任意shell、再起動、GPU reset、自動修復は含みません。

## Windows install

前提: Windows 11、Python 3.12、Git、Windows OpenSSH Client。

```powershell
setup.cmd
.\vast-agent.ps1 doctor
```

セットアップはリポジトリ内 `.venv` を作り、実データは
`%LOCALAPPDATA%\VastGeminiAgent` に作成します。再実行しても既存設定は上書きしません。

`config/examples/hosts.example.yaml` を参考に runtime の `config/hosts.yaml` を編集してください。
example host は予約された文書用IPであり、初期runtimeにはコピーされません。

## Discord を日常利用する

基本的な導入順序は `setup.cmd` → secrets設定 → `trust-host` → `gemini-check` →
`start` です。その後は指定したDiscord channelから自然文で利用できます。
Discord Developer PortalのBot設定で **MESSAGE CONTENT INTENT** を有効にしてください。

`%LOCALAPPDATA%\VastGeminiAgent\secrets\secrets.env` に次を設定してください。値はログ、
status、Discord応答に表示されません。OWNERは1ユーザーだけで、指定channel内のOWNER本人による
messageだけを受理します。BOT、webhook、他ユーザー、他channelは無視します。

```dotenv
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_CHANNEL_ID=your-channel-id
DISCORD_OWNER_USER_ID=your-owner-user-id
GEMINI_API_KEY=your-gemini-key
```

```powershell
.\vast-agent.ps1 discord-check
.\vast-agent.ps1 start
.\vast-agent.ps1 status
.\vast-agent.ps1 stop
.\vast-agent.ps1 restart
.\vast-agent.ps1 run-discord  # foreground / Ctrl+Cで停止
```

`start` は外部service managerなしでbackground processを起動し、agent logを
`%LOCALAPPDATA%\VastGeminiAgent\logs\agent\agent.log` に保存します。PID stateに加えてprocessの
command lineを検証するため、stale PIDや無関係なprocessは停止しません。

自然文の例は `torrentのGPU温度`、`torrentなんかおかしくない？`、`ついでにPCIも`、
`全台GPU状態見て`、`今の調査止めて`、`#184止めて`、`ジョブ見せて` です。明白な照会とfleetは
Geminiを呼ばず、fleetは設定された上限（既定3 host）で並列実行します。conversation stateとjobは
SQLiteへ小さなsummaryだけを保存し、再起動前に実行中だったjobは`INTERRUPTED`へ移行します。
Cancel Buttonは現Phaseではgatewayへ接続しておらず、cancelは上記のテキスト入力で行います。

## CLI

```powershell
.\vast-agent.ps1 install
.\vast-agent.ps1 doctor
.\vast-agent.ps1 config
.\vast-agent.ps1 hosts
.\vast-agent.ps1 trust-host HOST
.\vast-agent.ps1 test-host HOST
.\vast-agent.ps1 inspect HOST [--gpu|--pci|--vast|--docker|--vm] [--json]
.\vast-agent.ps1 ask "garage-torrentのGPU温度"
.\vast-agent.ps1 ask "garage-torrentなんかおかしくない？"
.\vast-agent.ps1 investigate garage-torrent "GPUが消えた原因を調べて"
.\vast-agent.ps1 gemini-check
.\vast-agent.ps1 migrate-v1 D:\path\vast_gemini.py
```

`trust-host` は `ssh-keyscan` の候補fingerprintを表示し、利用者が別経路で照合して `YES` と
明示するまで専用known_hostsへ書きません。鍵変更は自動承認しません。

`migrate-v1` は対象ファイルを実行せず、ASTのliteral `MACHINES` だけを読みます。同名hostは
上書きせず、tokenや環境変数を取り込みません。

## Architecture and security

* logical host nameを内部IDとし、IP addressをIDにしません。
* SSHは専用known_hosts、`BatchMode=yes`、`StrictHostKeyChecking=yes`、timeout/keepaliveを強制します。
* Tool registryに固定されたcommand tokenのみ実行し、呼出側からshell文字列を受けません。
* `read_config` は固定allowlistの1ファイルだけを対象とし、credential/private key/environmentを読みません。
* raw出力はtyped Observationへ変換し、DB payloadはサイズを抑えます。大きなraw log用にはruntime logsとDB path列を用意しています。
* SQLite migration、incident dedup key、FakeExecutorにより将来の拡張と実機不要テストを支えます。

## Gemini investigation

`%LOCALAPPDATA%\VastGeminiAgent\secrets\secrets.env` に次の1行を保存します（実際のkeyを
READMEやログへ貼らないでください）。環境変数 `GEMINI_API_KEY` はこのファイルをoverrideします。

```dotenv
GEMINI_API_KEY=your-key-here
```

明白なGPU/PCI/Vast/Docker/VM/ディスク照会とホスト一覧はローカルで決定し、Geminiを0回で
実行します。原因調査だけがInteractions APIのbounded function-calling loopへ進みます。
公開functionは `inspect_host`、`collect_evidence`、`get_recent_incidents` のREAD ONLY 3個で、
任意commandは受け付けません。interactionは `store=false` で、履歴はローカル側が保持します。

通常判断はthinking `low`、調査は `medium` です。`high` は設定で許可できますが現Phaseでは
自動昇格しません。各API callのpurpose/model/thinking levelと、APIが返したtoken count（nullable）を
SQLiteの `token_usage` に保存します。`doctor` は設定/key有無だけを確認しAPIを呼びません。
疎通確認はtoken消費を抑えた `gemini-check` を明示的に実行してください。

## Job cancellation semantics

cancelはローカルの待機処理を止め、実行中SSH subprocessへterminateを送り、timeout時にはkillします。
すでにremote側で開始した処理を巻き戻す意味ではありません。Geminiへ公開するremote toolは引き続きすべてREAD ONLYです。Phase 10–12のrestart、reboot、
GPU reset、VM modeは、下記の独立したtyped proposal/OWNER Approval経路だけから実行できます。

## Development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

## Phase 10–12: approval-gated operations

State-changing operations are **disabled by default**. They are separate from Gemini's read-only
function registry: Gemini can analyze and recommend, but it cannot invoke an action. Enable the
master switch explicitly in `agent.yaml` only after reviewing the host-specific access policy:

```yaml
operations:
  enabled: false
  approval_ttl_seconds: 600
  action_timeout_seconds: 60
  reboot_recovery_timeout_seconds: 300
  reboot_poll_seconds: 10
  enable_vms_script: null
```

A typed proposal and read-only preflight precede every operation. The configured OWNER must use the
Approve button in the configured channel before a fresh preflight is compared with the proposal.
Approvals expire, are atomic/single-use, and mean exactly one state-changing command. There is no
retry, force/no-approval option, fleet write, action chaining, or arbitrary shell interface. A
changed or newly unsafe preflight invalidates the approval. Restart, Vast `C.<digits>` container,
GPU reset, and VM-mode targets are strictly typed and validated. `enable_vms_script` must be an
explicit absolute remote POSIX path whose basename is `enable_vms.py`; null blocks VM actions.

Writes require non-interactive `sudo -n`. Configure only the minimum action-specific `NOPASSWD`
permissions needed by the deployment; **do not grant `NOPASSWD: ALL`**. This project does not edit
sudoers. Reboot uses a separate down/up monitor. If SSH does not return before the configured
recovery timeout, the agent stops and requests physical handling. It never attempts BMC, IPMI,
smart-plug/power-cycle, shutdown, a second reboot, or another automatic repair. Command timeout is
also never retried because the remote state may already have changed.
