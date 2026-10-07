"""Copy only explicit assets; preserve external execution boundaries and references."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import stat
from typing import Any

from configuration import ConfigurationError

from .source import is_env


@dataclass
class AssetCollector:
    directory: Path
    files: dict[Path, bytes] = field(default_factory=dict, repr=False)
    directories: set[Path] = field(default_factory=set)
    executable: set[Path] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    size: int = 0

    def path(self, value: Any, location: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ConfigurationError(location, "Packaged asset paths must be literal strings, not env references")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.directory / path
        if path.is_symlink():
            raise ConfigurationError(location, "Symlinked assets are not packaged")
        return path

    def copy_file(self, source: Path, destination: Path, location: str) -> str:
        try:
            mode = source.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ConfigurationError(location, "Assets must be regular files, not symlinks or devices")
            if len(self.files) >= 10000:
                raise ConfigurationError(location, "Asset count exceeds 10000 files")
            # Bounded read, including files that change after stat().
            with source.open("rb") as stream:
                data = stream.read(50 * 1024 * 1024 - self.size + 1)
        except (OSError, ValueError):
            raise ConfigurationError(location, "Cannot read asset") from None
        self.size += len(data)
        if self.size > 50 * 1024 * 1024:
            raise ConfigurationError(location, "Packaged assets exceed 50 MiB")
        self.files[destination] = data
        if mode & 0o111:
            self.executable.add(destination)
        self.directories.add(destination.parent)
        return destination.as_posix()

    def copy_tree(self, source: Path, destination: Path, location: str) -> str:
        if source.is_symlink() or not source.is_dir():
            raise ConfigurationError(location, "Skill directory does not exist or is a symlink")
        self.directories.add(destination)
        for child in sorted(source.iterdir()):
            if child.name in {".git", ".venv", "venv", "__pycache__", "node_modules", ".state", "state"} or child.name.startswith(".env"):
                self.notes.append(f"Excluded private/cache asset under {location}")
                continue
            if child.suffix.lower() in {".pem", ".key", ".p12", ".sqlite", ".sqlite3", ".db", ".pyc"}:
                self.notes.append(f"Excluded private/state asset under {location}")
                continue
            if child.is_symlink():
                raise ConfigurationError(location, "Symlinked Skill assets are not packaged")
            if child.is_dir():
                self.copy_tree(child, destination / child.name, location)
            else:
                self.copy_file(child, destination / child.name, location)
        return destination.as_posix()

    def external(self, value: Any, location: str) -> Any:
        if value is None:
            return None
        if is_env(value):
            self.notes.append(f"{location}: supply an absolute path through its environment variable")
            return value
        if not isinstance(value, str):
            raise ConfigurationError(location, "Expected a path string")
        path = Path(value).expanduser()
        self.notes.append(f"{location}: external path retained; not copied into the application")
        return str((path if path.is_absolute() else self.directory / path).resolve())

    def relocate(self, raw: dict[str, Any]) -> None:
        agent = raw.setdefault("agent", {})
        agent["workspace"] = self.external(agent.get("workspace", "."), "agent.workspace")
        if "allowed_roots" in agent:
            agent["allowed_roots"] = [self.external(p, "agent.allowed_roots") for p in agent["allowed_roots"]]
        prompt = raw.get("prompt") or {}
        if prompt.get("file") is not None:
            path = self.path(prompt["file"], "prompt.file")
            prompt["file"] = self.copy_file(path, Path("assets/prompt") / path.name, "prompt.file")
        skills = raw.get("skills") or {}
        if skills.get("enabled", True) and raw.get("skills") is not None:
            paths = []
            seen_directories, skill_names = set(), set()
            for index, value in enumerate(skills.get("directories", [])):
                source = self.path(value, "skills.directories")
                if source.resolve() in seen_directories:
                    continue
                seen_directories.add(source.resolve())
                from skill.folder_loader import discover_skill_files, load_skill_manifest
                try:
                    manifests = list(discover_skill_files(source))
                    if not manifests:
                        raise ValueError("No skills")
                    for manifest in manifests:
                        loaded = load_skill_manifest(manifest)
                        if loaded.name != manifest.parent.name or loaded.name in skill_names:
                            raise ValueError("Skill names must be unique and match directories")
                        skill_names.add(loaded.name)
                except Exception:
                    raise ConfigurationError("skills.directories", "Invalid or unreadable Skill directory") from None
                paths.append(self.copy_tree(source, Path("assets/skills") / str(index) / source.name, "skills.directories"))
            if not paths:
                raise ConfigurationError("skills.directories", "At least one Skill directory is required")
            skills["directories"] = paths
        permissions = raw.get("permissions") or {}
        if permissions.get("enabled", True) and raw.get("permissions") is not None:
            config = permissions.setdefault("config", {})
            from permissions.default.policy import load_policy_file

            def policy(path: Path, destination: Path) -> str:
                if path.is_symlink() or not path.is_file():
                    raise ConfigurationError("permissions", "Missing or symlinked policy file")
                try:
                    load_policy_file(path)
                except Exception:
                    raise ConfigurationError("permissions", "Invalid policy JSON") from None
                return self.copy_file(path, destination, "permissions")

            if config.get("policy_directory") is not None:
                directory = self.path(config["policy_directory"], "permissions.policy_directory")
                if not directory.is_dir():
                    raise ConfigurationError("permissions.policy_directory", "Policy directory does not exist")
                destination = Path("assets/permissions")
                self.directories.add(destination)
                for name in ("settings.json", "settings.local.json"):
                    path = directory / name
                    if path.exists() or path.is_symlink():
                        policy(path, destination / name)
                config["policy_directory"] = destination.as_posix()
            if "policy_paths" in config:
                config["policy_paths"] = [policy(self.path(p, "permissions.policy_paths"), Path("assets/policies") / f"{i}.json")
                                          for i, p in enumerate(config["policy_paths"])]
        for kind, field_name in (("session", "path"), ("observability", "path"), ("multi_agent", "storage_dir"), ("worktree", "storage_dir")):
            spec = raw.get(kind)
            if spec is None or not spec.get("enabled", True):
                continue
            config = spec.setdefault("config", {})
            value = config.get(field_name)
            if value is None and field_name == "path":
                continue
            if is_env(value) or (isinstance(value, str) and Path(value).expanduser().is_absolute()):
                config[field_name] = self.external(value, f"{kind}.{field_name}")
            else:
                basename = Path(value).name if isinstance(value, str) else "data"
                config[field_name] = (Path("state") / kind / basename).as_posix()
                self.notes.append(f"{kind}.{field_name}: fresh application-local state; existing state is not copied")
        for index, tool in enumerate(raw.get("tools", [])):
            if not tool.get("enabled", True):
                continue
            config = tool.get("config", {})
            for key in ("workspace_root", "cwd"):
                if config.get(key) is not None:
                    config[key] = self.external(config[key], f"tools[{index}].{key}")
            for key in ("allowed_roots",):
                if config.get(key) is not None:
                    config[key] = [self.external(p, f"tools[{index}].{key}") for p in config[key]]
            sandbox = config.get("sandbox") or {}
            for key in ("allow_write", "deny_write", "allow_read", "deny_read"):
                if key in sandbox:
                    sandbox[key] = [self.external(p, f"tools[{index}].sandbox.{key}") for p in sandbox[key]]
            if sandbox.get("bubblewrap_path"):
                sandbox["bubblewrap_path"] = self.external(sandbox["bubblewrap_path"], "sandbox.bubblewrap_path")
        mcp = raw.get("mcp") or {}
        if mcp.get("enabled", True):
            for server in mcp.get("servers", []):
                if not server.get("enabled", True):
                    continue
                if server.get("source_type") == "file":
                    server["source"] = self.external(server["source"], "mcp.source")
                self.notes.append("MCP servers, command arguments and their dependencies must be provisioned separately")
