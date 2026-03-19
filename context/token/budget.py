"""
Token 预算管理

按来源分配 token 配额，支持动态调整。
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenBudget:
    """Token 预算分配器"""

    max_tokens: int = 8000
    allocations: Dict[str, float] = field(default_factory=lambda: {
        "system": 0.10,
        "history": 0.30,
        "rag": 0.35,
        "memory": 0.15,
        "tool": 0.10,
    })

    def get_budget(self, source: str) -> int:
        """获取指定来源的 token 预算"""
        ratio = self.allocations.get(source, 0.0)
        return int(self.max_tokens * ratio)

    def set_allocation(self, source: str, ratio: float) -> None:
        """设置指定来源的分配比例"""
        self.allocations[source] = ratio

    def remaining(self, used: Dict[str, int]) -> int:
        """计算剩余可用 token"""
        return self.max_tokens - sum(used.values())

    def redistribute(self, used: Dict[str, int]) -> Dict[str, int]:
        """将各来源未用完的配额按比例重分配给超额来源

        Returns:
            各来源最终可用 token 数
        """
        budgets = {s: self.get_budget(s) for s in self.allocations}
        surplus = 0
        deficit_sources = []

        for source, budget in budgets.items():
            actual = used.get(source, 0)
            if actual < budget:
                surplus += budget - actual
                budgets[source] = actual  # 只给实际量
            elif actual > budget:
                deficit_sources.append(source)

        # 按原比例把 surplus 分配给 deficit sources
        if deficit_sources and surplus > 0:
            total_deficit_ratio = sum(
                self.allocations[s] for s in deficit_sources
            )
            for source in deficit_sources:
                if total_deficit_ratio > 0:
                    share = int(surplus * self.allocations[source] / total_deficit_ratio)
                    budgets[source] += share

        return budgets
