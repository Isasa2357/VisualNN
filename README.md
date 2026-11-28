# VisualNN

## セットアップ

Python 3.12 と依存パッケージは uv で管理します。Windows ではリポジトリのルートで次を実行します。

```powershell
.\setup.ps1
```

初回は Enter で CPU 版を選びます。`auto` と入力すると uv がこの PC のハードウェアに合う PyTorch の配布元を調べます。`cu130` のようなビルド名も直接指定できます。再実行時は Enter で前回の選択を使います。対話を省く場合は `.\setup.ps1 -TorchBackend auto` のように指定します。

セットアップは選んだ配布元を `pyproject.toml` に保存し、`uv.lock` と `.venv` を同期します。以後は通常の `uv add`、`uv sync`、`uv run` が同じ配布元を使います。CPU に戻すには `.\setup.ps1 -TorchBackend cpu` を実行してください。CUDA を選んだ PC で変更された `pyproject.toml` と `uv.lock` をコミットすると、その選択がほかの利用者にも共有されます。

たとえば PrefetchDataLoader をこの環境に追加するには、セットアップ後に次を実行します。

```powershell
uv add "git+https://github.com/Isasa2357/PrefetchDataLoader.git@main"
```
