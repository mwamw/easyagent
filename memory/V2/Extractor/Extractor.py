"""
LLM 实体关系提取器实现
基于结构化输出和解析
复用StructuredOutputAgent

包含两阶段流程：
1. 提取 Agent：从文本中提取实体和关系
2. 验证 Agent：检查提取结果的质量和一致性，修正后返回
"""
from typing import Optional, Any
from agent import StructuredOutputAgent
from core.llm import EasyLLM
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
import logging
from Store.GraphStore import Entity, Relation

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """你是一个实体关系提取器。请从文本中提取所有实体和关系。
规则：
1. entity_type 使用简洁的类别名（如：人物、组织、地点、概念、技术、职业）
2. 每个实体必须有清晰的 name 和 description
3. relation 的 from_entity 和 to_entity 必须是已提取实体的 name
4. strength 范围 0.0-1.0，表示关系的确定程度
5. evidence 引用原文中支持该关系的片段
6. properties 中放置额外的属性信息（如年龄、地址等）
"""

VERIFIER_SYSTEM_PROMPT = """你是一个实体关系验证和修正专家。你需要检查从文本中提取的实体和关系，确保结果的质量和一致性。

验证规则：
1. 每个关系的 from_entity 和 to_entity 必须是实体列表中某个实体的 name，如果不存在则补充缺失的实体
2. 检查是否有遗漏的重要实体或关系，如有则补充
3. 去除重复的实体（同一概念不同表述应合并）
4. 关系的 strength 应合理反映文本中的确定程度
5. evidence 必须来自原文，不能编造
6. entity_type 应保持一致（相同类型使用相同的类别名）
7. 如果提取结果已经完全正确，则原样返回，不要做不必要的修改

请输出修正后的完整实体和关系列表。
"""


class ExtractedEntity(BaseModel):
    """提取出的实体"""
    entity_type: str = Field(description="实体类型")
    name: str = Field(description="实体名称")
    description: str = Field(default="", description="实体描述")
    properties: dict = Field(default_factory=dict, description="实体属性")


class ExtractedRelation(BaseModel):
    """提取出的关系"""
    from_entity: str = Field(description="起始实体名称")
    to_entity: str = Field(description="目标实体名称")
    relation_type: str = Field(description="关系类型")
    strength: float = Field(default=1.0, description="关系强度，0.0-1.0")
    evidence: str = Field(default="", description="关系证据")
    properties: dict = Field(default_factory=dict, description="关系属性")


class ExtractionOutput(BaseModel):
    """提取输出"""
    entities: list[ExtractedEntity] = Field(description="实体列表")
    relations: list[ExtractedRelation] = Field(description="关系列表")


class Extractor:
    def __init__(self, llm: EasyLLM, enable_verification: bool = True):
        """
        初始化提取器
        Args:
            llm: LLM 实例
            enable_verification: 是否启用验证 Agent（默认启用）
        """
        self.llm = llm
        self.enable_verification = enable_verification

        # 提取 Agent
        self.extract_agent = StructuredOutputAgent(
            name="extractor",
            llm=llm,
            output_model=ExtractionOutput,
            system_prompt=EXTRACTOR_SYSTEM_PROMPT
        )

        # 验证 Agent
        if enable_verification:
            self.verify_agent = StructuredOutputAgent(
                name="verifier",
                llm=llm,
                output_model=ExtractionOutput,
                system_prompt=VERIFIER_SYSTEM_PROMPT
            )

    def extract(self, content: str) -> dict[str, Any]:
        """
        提取实体和关系（可选验证）
        Args:
            content: 待提取的内容
        Returns:
            提取的实体和关系
        """
        try:
            # 第一阶段：提取
            extracted_data = self.extract_agent.invoke(query=content)
            logger.info(f"提取完成: {len(extracted_data.entities)} 个实体, "
                        f"{len(extracted_data.relations)} 个关系")

            # 第二阶段：验证和修正
            if self.enable_verification:
                extracted_data = self._verify(content, extracted_data)

            return {
                "entities": extracted_data.entities,
                "relations": extracted_data.relations
            }
        except Exception as e:
            logger.error(f"提取实体和关系失败: {e}")
            return {
                "entities": [],
                "relations": []
            }

    def _verify(self, original_content: str, extracted: ExtractionOutput) -> ExtractionOutput:
        """
        验证和修正提取结果
        Args:
            original_content: 原始文本
            extracted: 提取的结果
        Returns:
            验证修正后的结果
        """
        # 构造验证请求：包含原文和提取结果
        entities_str = "\n".join(
            f"  - name: {e.name}, type: {e.entity_type}, description: {e.description}"
            for e in extracted.entities
        )
        relations_str = "\n".join(
            f"  - {r.from_entity} -[{r.relation_type}]-> {r.to_entity} "
            f"(strength: {r.strength}, evidence: {r.evidence})"
            for r in extracted.relations
        )

        verify_query = f"""原始文本：
{original_content}

提取结果：
实体：
{entities_str}

关系：
{relations_str}

请验证并修正以上提取结果。"""

        try:
            verified_data = self.verify_agent.invoke(query=verify_query)
            logger.info(f"验证完成: {len(verified_data.entities)} 个实体, "
                        f"{len(verified_data.relations)} 个关系")
            return verified_data
        except Exception as e:
            logger.warning(f"验证失败，使用原始提取结果: {e}")
            return extracted

    def extract_for_graph(self, content: str) -> dict[str, list]:
        """
        提取实体和关系，并转换为可直接写入 GraphStore 的格式
        Args:
            content: 待提取的内容
        Returns:
            包含 Entity 和 Relation 对象的字典，格式与 GraphStore 对齐
        """
        result = self.extract(content)
        graph_entities = []
        # 记录 name -> entity_id 的映射，用于关系中引用
        name_to_id: dict[str, str] = {}

        for e in result["entities"]:
            entity_id = str(uuid.uuid4())
            name_to_id[e.name] = entity_id
            graph_entities.append(Entity(
                entity_id=entity_id,
                name=e.name,
                entity_type=e.entity_type,
                properties=e.properties,
                description=e.description
            ))

        graph_relations = []
        for r in result["relations"]:
            from_id: str = name_to_id.get(r.from_entity) or r.from_entity
            to_id: str = name_to_id.get(r.to_entity) or r.to_entity
            graph_relations.append(Relation(
                from_entity=from_id,
                to_entity=to_id,
                relation_type=r.relation_type,
                strength=r.strength,
                evidence=r.evidence,
                properties=r.properties
            ))

        return {
            "entities": graph_entities,
            "relations": graph_relations
        }
