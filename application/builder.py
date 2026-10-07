"""Build a runnable source application, not a second Agent implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import yaml

from configuration import ConfigurationError, read_configuration_source
from easyagent.version import __version__

from .assets import AssetCollector
from .source import environment_names, protect_credentials, validate_source


LAUNCHER = '''#!/usr/bin/env python3
"""Generated EasyAgent application. Configure agent.yaml and local .env."""
from pathlib import Path

def main():
    try:
        from easyagent.application import run_application
    except ImportError:
        raise SystemExit("Install the EasyAgent checkout with application support; see README.md.")
    return run_application(Path(__file__).resolve().parent)

if __name__ == "__main__":
    raise SystemExit(main())
'''


@dataclass
class ApplicationPlan:
    source: Path
    name: str
    files: dict[Path, bytes] = field(repr=False)
    directories: set[Path] = field(repr=False)
    executable: set[Path] = field(default_factory=set, repr=False)
    environment: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def plan_application(path: str | Path) -> ApplicationPlan:
    """Read/validate source and assets, without credentials, clients or writes."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(str(source), "buildagent accepts .yaml / .yml configuration files")
    raw = read_configuration_source(source)
    raw, protected_fields = protect_credentials(raw)
    document = validate_source(raw)
    assets = AssetCollector(source.parent)
    assets.relocate(raw)
    validate_source(raw)
    names = tuple(sorted(environment_names(raw)))
    extras = []
    if document.ui is None or document.ui.enabled:
        extras.append("ui")
    for module in ("memory", "mcp"):
        spec = getattr(document, module)
        if spec is not None and spec.enabled:
            extras.append(module)
    dependency = "easyagent" + ("[" + ",".join(extras) + "]" if extras else "") + "==" + __version__
    notes = list(dict.fromkeys(assets.notes))
    if protected_fields:
        notes.append("Literal credential fields were replaced with env references; fill .env locally")
    notes.append("Prompt/Skill/policy contents are copied as trusted assets; review them for embedded secrets before sharing")
    suffix = "[" + ",".join(extras) + "]" if extras else ""
    readme = f'''# Generated EasyAgent application

这个目录包含启动入口和声明式配置，使用现有 EasyAgent Runtime，不是独立二进制。

## 运行

在构建时使用的 Python/conda 环境中：

```bash
python app.py --check
python app.py
python app.py --query "介绍当前项目"
python app.py --workspace /path/to/project --query "分析代码结构"
```

`--check` 不调用模型；无参数启动默认交互 UI。单次请求遇到 ASK 不会自动批准。
`--workspace` 只覆盖 Agent 默认工作区，显式工具路径和 allowed_roots 保持配置中的值。
从其他目录也可运行 `python /path/to/application/app.py`。

## 环境与依赖

启动时仅加载此目录的 `.env`，已设置的进程环境变量优先。参照 `.env.example`
填写本地密钥；不要提交 `.env`。生成时没有读取环境变量的值。
`.env` 只为配置中的显式 env 引用提供值，不修改进程全局环境。

在其他环境中，先安装包含 `easyagent.application` 的同一版 EasyAgent 源码：

```bash
python -m pip install -e "/path/to/EasyAgent{suffix}"
```

requirements.txt 记录运行依赖；它不是已发布版本或完整部署可用性的保证。
额外的 MCP Server、命令行程序、语言服务器、OS 沙箱需要自行安装。

## 文件与状态

- `agent.yaml`：可继续编辑的配置；框架更新不需要重新生成执行循环。
- `assets/`：配置直接引用的 Prompt、Skill 和权限文件。
- `state/`：原相对路径的会话/观测/协作存储使用新的应用本地路径，不拷贝旧数据。
- `build.json`：所需环境变量、构建版本及外部资源说明，不含密钥值。

工作区及显式外部路径保留原指向，不复制整个代码仓库。迁移机器时修改配置或使用
`--workspace`；文件内容中的路径、MCP 命令参数不做猜测性改写。
常见密钥字段被转换成环境变量引用，但 Prompt/Skill/权限文件仍需人工检查敏感内容。
'''
    files = dict(assets.files)
    texts = {
        Path("app.py"): LAUNCHER,
        Path("agent.yaml"): yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        Path(".env.example"): "# Fill locally; existing process environment takes precedence.\n" + "".join(name + "=\n" for name in names),
        Path(".gitignore"): ".env\n.env.*\n!.env.example\nstate/\n__pycache__/\n*.pyc\n",
        Path("requirements.txt"): "# Requires the EasyAgent build with easyagent.application support.\n" + dependency + "\n",
        Path("README.md"): readme,
        Path("build.json"): json.dumps({
            "format_version": 1, "easyagent_version": __version__,
            "environment_variables": names, "credential_fields_externalized": protected_fields,
            "notes": notes, "configuration": "agent.yaml", "entrypoint": "app.py",
        }, ensure_ascii=False, indent=2) + "\n",
    }
    files.update({p: text.encode("utf-8") for p, text in texts.items()})
    return ApplicationPlan(source, document.agent.name, files, assets.directories | {Path("state")},
                           assets.executable | {Path("app.py")}, names, tuple(notes))


def write_application(plan: ApplicationPlan, output: str | Path) -> Path:
    """Create an exclusively owned new directory. Never overwrite existing output."""
    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ConfigurationError(str(target), "Output already exists; choose a new directory")
    for relative in (*plan.files, *plan.directories):
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigurationError("output", "Application paths must stay inside the output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()  # Exclusive reservation, including concurrent builds.
    except FileExistsError:
        raise ConfigurationError(str(target), "Output already exists; choose a new directory") from None
    created_files, created_dirs = [], [target]
    try:
        directories = set(plan.directories)
        for relative in (*plan.files, *plan.directories):
            directories.update(p for p in relative.parents if p != Path("."))
        for relative in sorted(directories, key=lambda p: (len(p.parts), str(p))):
            directory = target / relative
            directory.mkdir(exist_ok=False)
            created_dirs.append(directory)
        # Completion manifest is written last. All inputs were checked before mkdir.
        for relative in sorted(plan.files, key=lambda p: (p.name == "build.json", str(p))):
            path = target / relative
            with path.open("xb") as stream:
                created_files.append(path)
                stream.write(plan.files[relative])
            if relative in plan.executable:
                path.chmod(0o755)
        return target
    except BaseException:
        # Remove only artifacts we created; do not recursively delete user files.
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


def build_application(path: str | Path, *, output: str | Path | None = None) -> Path:
    plan = plan_application(path)
    return write_application(plan, output if output is not None else Path.cwd() / (plan.source.stem + "-app"))
