# Vast Gemini Agent v2

## Adaptive investigation architecture

**Prefer composition over capability proliferation.** Ambiguous host diagnostics use a bounded,
request-local investigation session. Gemini selects the smallest useful set of registry-generated
READ functions, evaluates compact untrusted evidence, reuses exact-call results, and stops when the
goal is answerable or a tool, round, evidence, or time bound is reached. The application validates
every typed argument and fixes the target host; the planner has no shell, arbitrary executable,
package-install, restart, reset, reboot, or other mutation function.

READ budget exhaustion does not discard evidence already gathered: when possible, the final
reserved Gemini call synthesizes that evidence with no tools available. Narrow questions should
stop as soon as the smallest relevant evidence set makes a full or explicitly partial answer possible.
Specialized READs are preferred for each explicit sub-goal, while broad evidence gathering requires a
concrete unresolved question. Final-result schema validation failures are logged with bounded structural
diagnostics and typed locations only, never with raw model output.

A new high-level capability is appropriate only when dedicated backend semantics are required,
state or acquisition semantics are special, or safe composition of existing READ tools is
insufficient. Tool failure is `EVIDENCE_MISSING`, a registered backend unavailable in context is
`TOOL_UNAVAILABLE`, and only an absent ability is `CAPABILITY_GAP`. Package acquisition remains in
the existing application-owned, approval-gated bridge and is never performed by the planner.
Model-reported software status is non-authoritative: the bridge can propose acquisition only after
the same target-bound investigation session contains successful, typed package and executable READ
evidence matching code-owned acquisition metadata. Missing, failed, cross-host, or contradictory
evidence fails closed as unknown; an inactive service is degraded rather than package-missing.

## Safe capability acquisition

`PACKAGE_INSTALL` is a dangerous, approval-gated typed action. Catalogs describe known capabilities
and their automatic acquisition metadata; they are not the global execution permission system.
An explicitly user-named Debian/Ubuntu package may instead be dynamically validated against the
target host. Package targets must come verbatim from the current message and pass the strict Debian
package-name schema. The action is not exposed as a Gemini tool and does not accept shell commands,
repositories, URLs, or arbitrary package-manager input.
The default remains fail-closed: `operations.enabled` is `false` and `allowed_actions` is empty.

To opt a deployment in, add `PACKAGE_INSTALL` to `operations.allowed_actions` and enable
operations. Before a proposal and again after approval, the agent checks SSH and passwordless
sudo, package status and candidate, package-manager locks, active containers, and running VMs.
Any active workload, running VM, missing candidate, or busy package manager blocks installation.
Thus every install still follows typed `PACKAGE_INSTALL` → policy → OWNER approval → fresh preflight
→ fixed argv. Approved installs use the fixed `apt-get install -y --no-install-recommends -- <package>` argv once,
then verify the installed version and observe known executables/services without starting or
enabling a service.

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
Windowsユーザープロファイル名と専用known_hostsのパスに空白が含まれていても利用できます。

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
processが既に消滅したstale stateは安全に削除します。Windowsで`CTRL_BREAK_EVENT`が
`OSError`または`SystemError`になる場合は、markerで検証済みのprocessに限り、まず
`taskkill /PID` (forceなし) へfallbackします。

Interactions APIは`store: false`で使用し、初回入力の`user_input`とSDKが返した全step、
READ-only toolの`function_result`をその順で次のrequestへreplayします。過去に初回入力を
bare `text`で送った場合、初回が200でも2回目は`input[0]` = `UNKNOWN`として400になる
問題がありました。現在は明示的な`user_input` / `content: [{type: text, ...}]`形式で送信します。

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
.\vast-agent.ps1 detect-capabilities HOST [--apply]
.\vast-agent.ps1 set-gpu-count HOST COUNT
.\vast-agent.ps1 inspect HOST [--gpu|--pci|--vast|--docker|--vm] [--json]
.\vast-agent.ps1 ask "garage-torrentのGPU温度"
.\vast-agent.ps1 ask "garage-torrentなんかおかしくない？"
.\vast-agent.ps1 investigate garage-torrent "GPUが消えた原因を調べて"
.\vast-agent.ps1 gemini-check
.\vast-agent.ps1 migrate-v1 D:\path\vast_gemini.py
```

`trust-host` は `ssh-keyscan` の候補fingerprintを表示し、利用者が別経路で照合して `YES` と
明示するまで専用known_hostsへ書きません。Windows標準`ssh-keyscan`のKEX互換問題時は、認証を
無効にした`ssh`と一時known_hostsで候補鍵だけを安全に取得します。鍵変更は自動承認しません。

`detect-capabilities` は登録済みhostへ固定のREAD-only probeだけを実行します。`--apply` を付けると、
検出した`nvidia`、`docker`、`libvirt`、`vast`の値だけをlocal runtimeの`hosts.yaml`へatomicに保存し、
address、user、aliases、enabledやoperations設定は変更しません。

`set-gpu-count` はhost名またはaliasを解決し、そのhostの`expected_gpu_count`だけをlocal runtimeの
`hosts.yaml`へatomicに保存します。`COUNT`には1以上の整数が必要で、SSH接続は行いません。

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
公開functionはmetadata-drivenなHost READ capability registryから生成されます。既存の固定診断に加え、
package、executable、service、network、interface、process、OS factsの安全なprimitiveを、Geminiが結果に
応じて複数stepで組み合わせられます。ユーザーがtool名やLinux commandを覚える必要はありません。
hostごとに現在利用可能なregistry capabilityを照会することもできます。

各primitiveは厳格に型付け・検証された値を、code-owned argv templateの単一要素としてだけ渡します。
任意shell、任意SSH、任意argv、generic file read、environment/full command line、mutationは公開しません。
結果はtimeoutと文字数上限を持つuntrusted evidenceであり、不足するcapabilityは推測せず明示します。
interactionは `store=false` で、履歴はローカル側が保持します。

通常判断はthinking `low`、調査は `medium` です。`high` は設定で許可できますが現Phaseでは
自動昇格しません。各API callのpurpose/model/thinking levelと、APIが返したtoken count（nullable）を
SQLiteの `token_usage` に保存します。`doctor` は設定/key有無だけを確認しAPIを呼びません。
疎通確認はtoken消費を抑えた `gemini-check` を明示的に実行してください。

## General READ capabilities

一般会話ではhost調査用functionとは分離されたregistryから、Geminiが必要なREAD toolだけを選びます。
現在のcapabilityは **Weather**（Open-Meteo）、**FX**（Frankfurter/ECB）、**Web search**
（Brave Search）、およびregistry由来の **Capability listing** です。安定した一般知識にはtoolを強制せず、
「今日」「現在」「最新」の情報には該当toolを使います。外部結果はすべてuntrusted evidenceとして扱います。

```yaml
external_tools:
  default_weather_location: null  # nullなら場所の明示が必要。IP位置推定はしません
  request_timeout_seconds: 8
  max_response_bytes: 262144
  search_provider: disabled       # brave にする場合のみSEARCH_API_KEYが必要
```

Weather/FXはkey不要の公開HTTPS providerへ、Web searchは設定時だけallowlist済みproviderへ接続します。
HTTPS、provider host、timeout、response size、query/argument sizeを制限し、redirect、private/local address、
任意URL fetch、cookie、user-controlled headerを許可しません。provider障害時は過去知識で現在値を補いません。
検索はsnippetを資料として要約するもので、リンク先本文の正しさを保証するものではありません。

これらはすべて一般用途の **READ-only** capabilityです。shell/SSH、install、file/config write、restart、
reset、VM/Docker/GPU mutation、ActionRequestは一切公開されません。下記のapproval-gated WRITE経路とは
完全に別で、general toolからWRITE proposalを作ることもできません。

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
