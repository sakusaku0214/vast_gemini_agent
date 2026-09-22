# Vast Gemini Agent v2

Windows 11からUbuntu/Vast.aiホストを安全に観測する、Phase 0〜6の **READ ONLY** 実装です。
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

## Development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```
