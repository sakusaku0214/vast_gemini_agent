# Vast Gemini Agent v2

Windows 11からUbuntu/Vast.aiホストを安全に観測する、Phase 0〜3の **READ ONLY** 実装です。
Gemini、Discord、任意shell、再起動、GPU reset、自動修復は含みません。

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

## Development

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```
