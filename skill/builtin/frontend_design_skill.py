"""
FrontendDesignSkill — 前端设计与审美技能

提供高质量前端界面设计方法，不依赖额外工具。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class FrontendDesignSkill(BaseSkill):
    """
    前端设计技能

    强调视觉方向、排版、层次、色彩、动效与响应式质量，而不是套模板。
    """

    def __init__(self):
        config = SkillConfig(
            name="frontend_design",
            description="前端设计与审美技能，帮助模型产出更有方向感、层次和完成度的界面方案。",
            listing_description="Design stronger, more intentional frontend UI with better aesthetic judgment.",
            when_to_use="当你要设计页面、组件、落地页、仪表盘、品牌站或任何前端视觉界面时使用。",
            version="1.0.0",
            tags=["frontend", "design", "ui", "ux", "aesthetics", "css"],
            priority=9,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 前端设计与审美能力
你需要以设计师而不是模板拼装器的标准来产出前端界面。

核心要求：
- 先定义视觉方向，再展开布局和组件，不要直接堆常见 UI 模板
- 设计要有明确气质，例如冷静专业、编辑感、展览感、数据感、实验感，而不是“普通 SaaS 风”
- 页面必须在桌面和移动端都成立，不能只在单一视口下看起来正常

视觉判断标准：
- Typography：优先建立清晰字号层级、字重对比、行长与留白节奏，不要只有“标题大一点”
- Color：先确定主色、辅助色、背景层级和强调色的职责，避免廉价渐变和默认紫色方案
- Depth：通过阴影、边框、模糊、分区、材质感建立层次，不要整页平铺
- Layout：使用有意图的栅格、分组和密度控制，让信息结构一眼可读
- Motion：只加入有意义的进入、切换、反馈动效，不要到处加微动画

前端实现要求：
- 优先定义设计 token 或 CSS 变量，保证颜色、圆角、间距、阴影一致
- 组件风格要统一，同一页面不要出现多套视觉语言
- 交互状态要完整：hover、active、focus、disabled、loading 都要考虑
- 不要只给“好看”方案，还要兼顾可实现性和可维护性

避免的问题：
- 千篇一律的 hero + 三栏卡片 + 渐变按钮
- 只靠大圆角和投影制造“现代感”
- 排版、留白、颜色、动效之间互相打架
- 完全忽略移动端与长内容场景
"""
