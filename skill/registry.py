"""
SkillRegistry — 全局 Skill 发现注册中心

提供 Skill 类的全局注册、工厂创建和目录自动发现功能。
类似插件注册表，允许通过名称创建 Skill 实例。
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseSkill, SkillConfig, SkillManifest

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    全局 Skill 发现注册中心（单例）

    职责:
    1. 注册 Skill 类或工厂函数
    2. 按名称创建 Skill 实例
    3. 从目录自动发现 Skill（.py / .yaml / .md 文件）
    4. 列出所有可用 Skill

    Example::

        registry = SkillRegistry.instance()
        registry.register_class(WebSearchSkill)
        skill = registry.create("web_search", api_key="xxx")

        # 使用装饰器
        @registry.skill("my_skill")
        class MySkill(BaseSkill):
            ...

        # 自动发现
        registry.discover_from_directory("./skills/")
    """

    _instance: Optional["SkillRegistry"] = None

    def __init__(self):
        self._skill_classes: Dict[str, Type[BaseSkill]] = {}
        self._skill_factories: Dict[str, Callable[..., BaseSkill]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}  # 名称 → 元信息
        self._skill_descriptions: Dict[str, str] = {}  # 名称 → 描述
        self._manifests: Dict[str, SkillManifest] = {}

    @classmethod
    def instance(cls) -> "SkillRegistry":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（主要用于测试）"""
        cls._instance = None

    # ==================== 注册 ====================

    def register_class(
        self,
        skill_class: Type[BaseSkill],
        name: Optional[str] = None,
    ) -> None:
        """
        注册一个 Skill 类

        Args:
            skill_class: BaseSkill 的子类
            name: 注册名称，默认使用类名的 snake_case 形式

        Raises:
            TypeError: 参数不是 BaseSkill 子类
        """
        if not (isinstance(skill_class, type) and issubclass(skill_class, BaseSkill)):
            raise TypeError(f"skill_class 必须是 BaseSkill 的子类，收到: {skill_class}")

        reg_name = name or self._class_to_name(skill_class)
        self._skill_classes[reg_name] = skill_class
        self._metadata[reg_name] = {
            "type": "class",
            "class": skill_class.__name__,
            "module": skill_class.__module__,
        }
        # 从类 docstring 获取默认描述
        desc = skill_class.__doc__.strip() if skill_class.__doc__ else ""
        self._skill_descriptions[reg_name] = desc
        self._manifests[reg_name] = SkillManifest(
            name=reg_name,
            description=desc,
            listing_description=desc or reg_name,
            source_type="python",
            metadata=dict(self._metadata[reg_name]),
        )

        logger.debug("SkillRegistry: 注册类 '%s' → %s", reg_name, skill_class.__name__)

    def register_factory(
        self,
        name: str,
        factory: Callable[..., BaseSkill],
    ) -> None:
        """
        注册一个 Skill 工厂函数

        Args:
            name: 注册名称
            factory: 工厂函数，调用后返回 BaseSkill 实例
        """
        if not callable(factory):
            raise TypeError("factory 必须是可调用对象")

        self._skill_factories[name] = factory
        self._metadata[name] = {"type": "factory", "callable": repr(factory)}
        self._manifests.setdefault(
            name,
            SkillManifest(
                name=name,
                listing_description=name,
                source_type="factory",
                metadata=dict(self._metadata[name]),
            ),
        )
        logger.debug("SkillRegistry: 注册工厂 '%s'", name)
    def skill(self, name: Optional[str] = None, **default_kwargs):
        """
        装饰器：将类注册为 Skill

        Args:
            name: 注册名称（默认使用类名）
            **default_kwargs: 创建实例时的默认参数

        Example::

            @registry.skill("web_search")
            class WebSearchSkill(BaseSkill):
                ...
        """
        def decorator(cls):
            reg_name = name or self._class_to_name(cls)
            self.register_class(cls, reg_name)
            if default_kwargs:
                self._metadata[reg_name]["default_kwargs"] = default_kwargs
            return cls
        return decorator

    # ==================== 创建 ====================

    def create(self, skill_name: str, **kwargs) -> BaseSkill:
        """
        根据注册名称创建 Skill 实例

        优先使用工厂函数，其次使用类。

        Args:
            skill_name: 注册名称
            **kwargs: 传递给构造函数或工厂的参数

        Returns:
            BaseSkill 实例

        Raises:
            KeyError: 名称未注册
        """
        # 工厂优先
        if skill_name in self._skill_factories:
            return self._skill_factories[skill_name](**kwargs)

        if skill_name in self._skill_classes:
            cls = self._skill_classes[skill_name]
            # 合并默认 kwargs
            meta = self._metadata.get(skill_name, {})
            merged_kwargs = {**meta.get("default_kwargs", {}), **kwargs}
            return cls(**merged_kwargs)

        available = self.list_available_names()
        raise KeyError(f"Skill '{skill_name}' 未注册。可用: {available}")

    # ==================== 发现 ====================

    def discover_from_directory(self, path: str) -> List[str]:
        """
        从目录自动发现并注册 Skill

        支持的文件类型:
        - .py — Python 模块，导入后扫描 BaseSkill 子类
        - .yaml / .yml — YAML Skill 定义文件
        - .md — Markdown Skill 定义文件

        Args:
            path: 目录路径

        Returns:
            成功注册的 Skill 名称列表
        """
        if not os.path.isdir(path):
            logger.warning("Skill 发现目录不存在: %s", path)
            return []

        registered = []

        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)

            # if not os.path.isfile(filepath):
            #     continue

            try:
                if filename.endswith(".py") and not filename.startswith("_"):
                    names = self._discover_from_python(filepath)
                    registered.extend(names)

                elif filename.endswith((".yaml", ".yml")):
                    names = self._discover_from_yaml(filepath)
                    registered.extend(names)

                elif filename.endswith(".md"):
                    names = self._discover_from_markdown(filepath)
                    registered.extend(names)

            except Exception as e:
                logger.warning("发现 Skill 文件 '%s' 失败: %s", filepath, e)

            if os.path.isdir(filepath):
                # 检查是否是一个有效 Folder Skill 目录
                if os.path.isfile(os.path.join(filepath, "README.md")) or \
                   os.path.isfile(os.path.join(filepath, "skill.md")):
                    try:
                        names = self._discover_from_folder(filepath)
                        registered.extend(names)
                    except Exception as e:
                        logger.warning("加载 Folder Skill 目录 '%s' 失败: %s", filepath, e)

        if registered:
            logger.info(
                "从目录 '%s' 发现并注册 %d 个 Skill: %s",
                path, len(registered), registered,
            )
        return registered

    # ==================== 查询 ====================

    def list_available_names(self) -> List[str]:
        """列出所有已注册的 Skill 名称"""
        names = set(self._skill_classes.keys()) | set(self._skill_factories.keys())
        return sorted(names)

    def list_available(self) -> List[Dict[str, Any]]:
        """列出所有可用的 Skill 信息"""
        return [manifest.model_dump(mode="json") for manifest in self.list_manifests()]

    def list_manifests(self) -> List[SkillManifest]:
        """列出所有已注册 Skill 的 manifest。"""
        manifests = [self._manifests[name] for name in self.list_available_names() if name in self._manifests]
        manifests.sort(key=lambda item: (item.priority, item.name), reverse=True)
        return manifests

    def get_manifest(self, name: str) -> SkillManifest:
        """获取指定 Skill 的 manifest。"""
        if name not in self._manifests:
            raise KeyError(f"Skill '{name}' 的 manifest 不存在")
        return self._manifests[name].model_copy(deep=True)

    def has(self, name: str) -> bool:
        """检查名称是否已注册"""
        return name in self._skill_classes or name in self._skill_factories

    def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        按关键词和标签搜索可用 Skill

        搜索逻辑：
        - query 会与 Skill 的 name、description、tags 进行模糊匹配（大小写不敏感）
        - tags 过滤：如果指定，Skill 必须包含至少一个给定 tag
        - query 和 tags 同时指定时取交集

        Args:
            query: 搜索关键词（空字符串匹配全部）
            tags: 标签过滤列表（可选）

        Returns:
            匹配的 Skill 信息列表
        """
        query_lower = query.lower().strip()
        query_keywords = query_lower.split() if query_lower else []
        tag_set = {t.lower() for t in tags} if tags else set()

        scored_results: list[tuple[int, int, dict[str, Any]]] = []
        for manifest in self.list_manifests():
            searchable_parts = [
                manifest.name,
                manifest.description,
                manifest.listing_description,
                manifest.when_to_use,
                " ".join(manifest.tags),
                " ".join(manifest.tool_names),
                " ".join(f"{key} {value}" for key, value in manifest.metadata.items()),
            ]
            searchable = " ".join(part for part in searchable_parts if part).lower()
            manifest_tags = {tag.lower() for tag in manifest.tags}

            if query_keywords and not all(keyword in searchable for keyword in query_keywords):
                continue

            if tag_set and not tag_set.intersection(manifest_tags):
                continue

            score = 0
            if query_lower:
                if manifest.name.lower() == query_lower:
                    score += 100
                if query_lower in manifest.name.lower():
                    score += 40
                if query_lower in manifest.listing_description.lower():
                    score += 30
                if query_lower in manifest.when_to_use.lower():
                    score += 20
                if query_lower in manifest.description.lower():
                    score += 10
                score += sum(5 for keyword in query_keywords if keyword in searchable)
            score += manifest.priority

            scored_results.append((score, manifest.priority, manifest.model_dump(mode="json")))

        scored_results.sort(key=lambda item: (item[0], item[1], item[2]["name"]), reverse=True)
        return [item[2] for item in scored_results]

    def update_metadata(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        **extra: Any,
    ) -> None:
        """
        更新已注册 Skill 的元信息（description、tags 等）

        Args:
            name: Skill 注册名称
            description: 描述
            tags: 标签列表
            **extra: 其他元信息
        """
        if name not in self._metadata:
            self._metadata[name] = {}
        if description:
            self._metadata[name]["description"] = description
            self._skill_descriptions[name] = description
            manifest = self._manifests.get(name)
            if manifest is not None:
                manifest.description = description
                if not manifest.listing_description:
                    manifest.listing_description = description
        if tags is not None:
            self._metadata[name]["tags"] = tags
            manifest = self._manifests.get(name)
            if manifest is not None:
                manifest.tags = list(tags)
        self._metadata[name].update(extra)
        manifest = self._manifests.get(name)
        if manifest is not None:
            for key, value in extra.items():
                if hasattr(manifest, key):
                    setattr(manifest, key, value)
                else:
                    manifest.metadata[key] = value

    # ==================== 内部 ====================

    def _discover_from_python(self, filepath: str) -> List[str]:
        """从 Python 文件发现 BaseSkill 子类"""
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(f"skill_discover.{module_name}", filepath)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore

        registered = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSkill)
                and attr is not BaseSkill
            ):
                name = self._class_to_name(attr)
                if not self.has(name):
                    self.register_class(attr, name)
                    registered.append(name)

        return registered

    def _discover_from_yaml(self, filepath: str) -> List[str]:
        """从 YAML 文件发现 Skill"""
        from .yaml_loader import YAMLSkillLoader

        try:
            skill = YAMLSkillLoader.load(filepath)
            name = skill.name
            manifest = skill.build_manifest()
            desc = manifest.description
            if not self.has(name):
                self._skill_descriptions[name] = desc
                self.register_factory(name, lambda fp=filepath: YAMLSkillLoader.load(fp))
                manifest.source_path = filepath
                self._manifests[name] = manifest
                self.update_metadata(
                    name,
                    description=manifest.description,
                    tags=manifest.tags,
                    listing_description=manifest.listing_description,
                    when_to_use=manifest.when_to_use,
                    exposure_mode=manifest.exposure_mode,
                    execution_mode=manifest.execution_mode,
                    source_type=manifest.source_type,
                    source_path=manifest.source_path,
                )
                return [name]
        except Exception as e:
            logger.warning("加载 YAML Skill '%s' 失败: %s", filepath, e)

        return []

    def _discover_from_markdown(self, filepath: str) -> List[str]:
        """从 Markdown 文件发现 Skill"""
        from .yaml_loader import MarkdownSkillLoader

        try:
            skill = MarkdownSkillLoader.load(filepath)
            name = skill.name
            manifest = skill.build_manifest()
            desc = manifest.description
            if not self.has(name):
                self._skill_descriptions[name] = desc
                self.register_factory(name, lambda fp=filepath: MarkdownSkillLoader.load(fp))
                manifest.source_path = filepath
                self._manifests[name] = manifest
                self.update_metadata(
                    name,
                    description=manifest.description,
                    tags=manifest.tags,
                    listing_description=manifest.listing_description,
                    when_to_use=manifest.when_to_use,
                    exposure_mode=manifest.exposure_mode,
                    execution_mode=manifest.execution_mode,
                    source_type=manifest.source_type,
                    source_path=manifest.source_path,
                )
                return [name]
        except Exception as e:
            logger.warning("加载 Markdown Skill '%s' 失败: %s", filepath, e)

        return []

    def _discover_from_folder(self, dirpath: str) -> List[str]:
        """从特定格式的文件夹发现 Skill"""
        from .folder_loader import FolderSkillLoader

        try:
            skill = FolderSkillLoader.load(dirpath)
            name = skill.name
            manifest = skill.build_manifest()
            desc = manifest.description
            if not self.has(name):
                self._skill_descriptions[name] = desc
                self.register_factory(name, lambda dp=dirpath: FolderSkillLoader.load(dp))
                manifest.source_path = dirpath
                self._manifests[name] = manifest
                self.update_metadata(
                    name,
                    description=manifest.description,
                    tags=manifest.tags,
                    listing_description=manifest.listing_description,
                    when_to_use=manifest.when_to_use,
                    exposure_mode=manifest.exposure_mode,
                    execution_mode=manifest.execution_mode,
                    source_type=manifest.source_type,
                    source_path=manifest.source_path,
                )
                return [name]
        except Exception as e:
            logger.warning("加载 Folder Skill '%s' 失败: %s", dirpath, e)

        return []

    @staticmethod
    def _class_to_name(skill_cls: type) -> str:
        """将类名转换为 snake_case 名称"""
        name = skill_cls.__name__
        # 移除 Skill 后缀
        if name.endswith("Skill"):
            name = name[:-5]
        # CamelCase → snake_case
        result = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                result.append("_")
            result.append(ch.lower())
        return "".join(result) or "skill"

    def get_skills_description(self) -> dict:
        """获取 Skill 的描述"""
        return self._skill_descriptions

    def load_body(self, name: str, **kwargs: Any) -> str:
        """加载指定 Skill 的正文 prompt。"""
        return self.create(name, **kwargs).get_body_prompt()
