import logging
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from .Store.VectorStore import VectorStore
from .BaseMemory import MemoryType,MemoryItem
from typing import Optional,Any,List
from .BaseMemory import MemoryConfig,BaseMemory,ForgetType
from .Embedding.BaseEmbeddingModel import BaseEmbeddingModel
from datetime import datetime
from .Store.GraphStore import GraphStore
import numpy as np
from .Store.GraphStore import Entity,Relation
from .Extractor.Extractor import Extractor

class SemanticMemory(BaseMemory):
    def __init__(self,memory_config:MemoryConfig,
        vector_store:VectorStore,
        graph_store:GraphStore,
        extractor:Extractor,
        embedding_model:BaseEmbeddingModel):
        
        super().__init__(memory_config)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.extractor = extractor
        self.embedding_model = embedding_model
    
        # 图数据缓存
        self.entities:dict[str,Entity] = {}
        self.entities_name_to_id:dict[str,str] = {}
        self.relations:list[Relation] = []

        # 记忆缓存
        self.memories:list[MemoryItem] = []
        self.id_to_memory:dict[str,MemoryItem] = {}
        self.memory_embeddings:dict[str,np.ndarray] = {}
    # ==================== 添加记忆 ====================

    def add_memory(self, memory: MemoryItem) -> str:
        if memory.id in self.memory_embeddings:
            logger.warning(f"记忆 {memory.id} 已存在")
            return ""
        try:
            # 向量生成
            memory_content = memory.content
            embedding = self.embedding_model.embed([memory_content])

            # 实体、关系提取（使用 extract_for_graph 返回 Entity/Relation 对象）
            extracted_data = self.extractor.extract_for_graph(memory_content)
            entities = extracted_data.get("entities", [])
            relations = extracted_data.get("relations", [])

            # 向量存储
            metadata = {
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "content": memory.content,
                "timestamp": int(memory.timestamp.timestamp()),
                "importance": memory.importance,
                "memory_type": memory.type,
                "entities": [entity.entity_id for entity in entities],
                "entity_count": len(entities),
                "relation_count": len(relations),
            }
            self.vector_store.add_vectors(embedding, [metadata], [memory.id])

            # 图存储
            for entity in entities:
                self._add_entity_to_graph(entity, memory)
            for relation in relations:
                self._add_relation_to_graph(relation, memory)

            # 记录元数据
            memory.metadata["entities"] = [e.entity_id for e in entities]
            memory.metadata["relations"] = [
                f"{r.from_entity}|||{r.relation_type}|||{r.to_entity}" for r in relations
            ]
            memory.metadata["entity_count"] = len(entities)
            memory.metadata["relation_count"] = len(relations)

            # 缓存处理
            self.memories.append(memory)
            self.id_to_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = np.array(embedding[0])
            logger.info(f"✅ 添加语义记忆: {len(entities)}个实体, {len(relations)}个关系")
            return memory.id
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return ""

    # ==================== 删除记忆 ====================

    def remove_memory(self, memory_id: str) -> bool:
        removed_memory = self.id_to_memory.get(memory_id, None)
        if not removed_memory:
            logger.warning(f"记忆 {memory_id} 在缓存中不存在")
            return False

        # 1. 处理向量数据库
        try:
            self.vector_store.remove_vectors([memory_id])
            logger.info(f"✅ 向量数据库成功删除记忆: {memory_id}")
        except Exception as e:
            logger.error(f"向量数据库删除记忆失败: {e}")
            return False

        # 2. 处理图数据库
        try:
            self.graph_store.delete_entity_by_memoryid(memory_id)
            self.graph_store.delete_relation_by_memoryid(memory_id)
            logger.info(f"✅ 图数据库成功删除记忆: {memory_id}")
        except Exception as e:
            logger.error(f"图数据库删除记忆失败: {e}")
            return False

        # 3. 处理缓存
        try:
            self.id_to_memory.pop(memory_id, None)
            self.memory_embeddings.pop(memory_id, None)
            self.memories.remove(removed_memory)

            # 清除实体缓存
            for entity_id in removed_memory.metadata.get("entities", []):
                removed_entity = self.entities.pop(entity_id, None)
                if removed_entity:
                    self.entities_name_to_id.pop(removed_entity.name, None)

            # 清除关系缓存
            removed_relation_keys = set()
            for rel_str in removed_memory.metadata.get("relations", []):
                parts = rel_str.split("|||")
                if len(parts) == 3:
                    removed_relation_keys.add(tuple(parts))

            self.relations = [
                r for r in self.relations
                if (r.from_entity, r.relation_type, r.to_entity) not in removed_relation_keys
            ]
            logger.info(f"✅ 缓存成功删除记忆: {memory_id}")
        except Exception as e:
            logger.error(f"缓存删除记忆失败: {e}")
            return False

        return True

    # ==================== 更新记忆 ====================

    def update_memory(self, id: str, content: str,
                      importance: Optional[float] = None,
                      metadata: Optional[dict[str, Any]] = None) -> bool:
        """
        更新记忆内容：删除旧记忆，用新内容重新提取并存储。
        Args:
            id: 记忆 ID
            content: 新的内容
            importance: 新的重要性（可选，None 则保留原值）
            metadata: 新的元数据（可选，None 则保留原值）
        Returns:
            是否更新成功
        """
        try:
            old_memory = self.id_to_memory.get(id, None)
            if not old_memory:
                logger.warning(f"记忆 {id} 在缓存中不存在，无法更新")
                return False

            # 保留旧信息
            new_importance = importance if importance is not None else old_memory.importance
            new_metadata = metadata if metadata is not None else {}
            user_id = old_memory.user_id
            memory_type = old_memory.type

            # 删除旧记忆
            removed = self.remove_memory(id)
            if not removed:
                logger.error(f"更新记忆失败: 无法删除旧记忆 {id}")
                return False

            # 用新内容重新创建记忆（保持同一 ID）
            new_memory = MemoryItem(
                id=id,
                user_id=user_id,
                type=memory_type,
                content=content,
                timestamp=datetime.now(),
                importance=new_importance,
                metadata=new_metadata,
            )
            result = self.add_memory(new_memory)
            if result:
                logger.info(f"✅ 更新记忆成功: {id}")
                return True
            else:
                logger.error(f"更新记忆失败: 重新添加记忆 {id} 失败")
                return False
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            return False


    def get_memory(self,ids:list[str])->list[MemoryItem]:
        results=[]
        for id in ids:
            memory=self.id_to_memory.get(id)
            if memory:
                results.append(memory)
        return results
    # ==================== 搜索记忆 ====================

    def search_memory(self, query: str, limit: int = 5,
                      user_id: Optional[str] = None, **kwargs) -> List[MemoryItem]:
        """
        语义搜索记忆，结合向量相似度和图谱上下文。
        Args:
            query: 搜索查询文本
            limit: 返回数量限制
            user_id: 可选的用户 ID 过滤
            **kwargs:
        Returns:
            匹配的记忆列表（按相关性排序）
        """
        try:
            user_id =user_id if user_id else "default"

            # 1. 向量搜索
            query_embedding=self.embedding_model.embed([query])[0]
            where={"user_id":user_id,"memory_type":"semantic"}
            vector_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit*2,
                where=where
            )

            # 2. 图谱搜索
            logger.info(f"🔍 [DEBUG] 向量搜索返回 {len(vector_results)} 条结果")
            try:
                graph_query=self.extractor.extract_for_graph(query)
            except Exception as e:
                logger.warning(f"生成图谱查询失败: {e}")
                graph_query={}
            graph_entities=graph_query.get("entities",[])
            logger.info(f"🔍 [DEBUG] 提取到 {len(graph_entities)} 个查询实体: {[e.name if hasattr(e,'name') else e for e in graph_entities]}")
            logger.info(f"🔍 [DEBUG] entities_name_to_id 缓存: {list(self.entities_name_to_id.keys())}")
            related_memory_ids=set()
            for entity in graph_entities:
                entity_name = entity.name if hasattr(entity, 'name') else entity.get('name', '')
                try:
                    related_entities=self.graph_store.get_related_entities(
                        entity_name=entity_name,
                        user_id=user_id,
                        limit=20,
                        max_depth=2
                    )
                    logger.info(f"🔍 [DEBUG] 实体 '{entity_name}' get_related_entities 返回 {len(related_entities)} 个")
                    for rel_entity in related_entities:
                        mid = rel_entity.properties.get("memory_id")
                        logger.info(f"🔍 [DEBUG]   相关实体: {rel_entity.name}, memory_id={mid}, properties={rel_entity.properties}")
                        if mid:
                            related_memory_ids.add(mid)

                    entity_rels=self.graph_store.get_entity_relations(
                        entity_name=entity_name,
                        user_id=user_id,
                    )
                    logger.info(f"🔍 [DEBUG] 实体 '{entity_name}' get_entity_relations 返回 {len(entity_rels)} 个")
                    for rel in entity_rels:
                        mid = rel.properties.get("memory_id")
                        logger.info(f"🔍 [DEBUG]   关系实体: {rel.name}, memory_id={mid}, properties={rel.properties}")
                        if mid:
                            related_memory_ids.add(mid)
                except Exception as e:
                    logger.warning(f"获取实体 {entity_name} 的相关记忆失败: {e}")
            logger.info(f"🔍 [DEBUG] 图谱搜索找到 {len(related_memory_ids)} 个 related_memory_ids: {related_memory_ids}")
            ##缓存中得到完整记忆
            graph_results=[]
            for memory_id in related_memory_ids:
                memory=self.id_to_memory.get(memory_id)
                if memory:
                    metadata = {
                        "content": memory.content,
                        "user_id": memory.user_id,
                        "memory_type": memory.type,
                        "importance": memory.importance,
                        "timestamp": int(memory.timestamp.timestamp()),
                        "entities": memory.metadata.get("entities", []),
                        "entity_count": memory.metadata.get("entity_count", 0),
                        "relation_count": memory.metadata.get("relation_count", 0),
                    }

                    # 计算图相关性分数
                    graph_score = self._calculate_graph_relevance(metadata, graph_entities)
                    logger.info(f"🔍 [DEBUG] memory_id={memory_id}, graph_score={graph_score:.4f}, memory_entities={metadata.get('entities', [])}")

                    graph_results.append({
                        "id": memory_id,
                        "memory_id": memory_id,
                        "content": metadata.get("content", ""),
                        "similarity": graph_score,
                        "user_id": metadata.get("user_id"),
                        "memory_type": metadata.get("memory_type"),
                        "importance": metadata.get("importance", 0.5),
                        "timestamp": metadata.get("timestamp"),
                        "entities": metadata.get("entities", []),
                        "entity_count": metadata.get("entity_count", 0),
                        "relation_count": metadata.get("relation_count", 0),
                    })
            logger.info(f"🔍 [DEBUG] graph_results 共 {len(graph_results)} 条: {[(g['memory_id'][:8], g['similarity']) for g in graph_results]}")
            # 3. 标准化向量结果格式
            normalized_vector_results = []
            for vr in vector_results:
                payload = vr.get("metadata", {})
                normalized_vector_results.append({
                    "memory_id": vr.get("memory_id", ""),
                    "content": payload.get("content", ""),
                    "score": vr.get("similarity", 0.0),
                    "user_id": payload.get("user_id", ""),
                    "memory_type": payload.get("memory_type", "semantic"),
                    "importance": payload.get("importance", 0.5),
                    "timestamp": payload.get("timestamp", 0),
                    "entities": payload.get("entities", []),
                    "entity_count": payload.get("entity_count", 0),
                    "relation_count": payload.get("relation_count", 0),
                })

            # 4. 混合排序
            ranked_results = self._combine_and_rank_results(
                vector_results=normalized_vector_results,
                graph_results=graph_results,
                limit=limit
            )

            # 5. 转化为 MemoryItem
            results: List[MemoryItem] = []
            for item in ranked_results:
                memory_id = item.get("memory_id", "")
                # 优先从缓存获取
                cached = self.id_to_memory.get(memory_id)
                if cached:
                    memory_copy = cached.model_copy()
                else:
                    # 从结果重建
                    memory_copy = MemoryItem(
                        id=memory_id,
                        user_id=item.get("user_id", ""),
                        type=item.get("memory_type", "semantic"),
                        content=item.get("content", ""),
                        timestamp=datetime.fromtimestamp(item.get("timestamp", 0)),
                        importance=item.get("importance", 0.5),
                        metadata={},
                    )
                # 附加搜索评分信息
                memory_copy.metadata["combined_score"] = item.get("combined_score", 0.0)
                memory_copy.metadata["vector_score"] = item.get("vector_score", 0.0)
                memory_copy.metadata["graph_score"] = item.get("graph_score", 0.0)
                results.append(memory_copy)

            logger.info(f"✅ 搜索到 {len(results)} 条相关记忆")
            return results
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []

    def _combine_and_rank_results(
        self,
        vector_results: List[dict],
        graph_results: List[dict],
        limit: int
    ) -> List[dict]:
        """混合排序结果 — 基于向量分数与图谱分数的加权融合"""
        combined: dict[str, dict] = {}
        content_seen: set[int] = set()

        # 添加向量结果
        for result in vector_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            content_hash = hash(content.strip())
            if content_hash in content_seen:
                continue
            content_seen.add(content_hash)
            combined[memory_id] = {
                **result,
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0,
                "content_hash": content_hash,
            }

        # 添加 / 合并图结果
        for result in graph_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            content_hash = hash(content.strip())
            if memory_id in combined:
                # 已有向量结果，补充图分数
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            elif content_hash not in content_seen:
                content_seen.add(content_hash)
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0),
                    "content_hash": content_hash,
                }

        # 计算混合分数
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)

            # 基础相关性 = 向量 70% + 图谱 30%
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            # 重要性加权: importance ∈ [0,1] → weight ∈ [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight

            result["combined_score"] = combined_score

        # 过滤低分结果
        min_threshold = 0.1
        filtered = [r for r in combined.values() if r["combined_score"] >= min_threshold]

        # 按综合分数降序排序
        filtered.sort(key=lambda x: x["combined_score"], reverse=True)

        logger.debug(f"🔍 向量结果: {len(vector_results)}, 图结果: {len(graph_results)}, "
                     f"去重后: {len(combined)}, 过滤后: {len(filtered)}")
        return filtered[:limit]

    def _calculate_graph_relevance(self, memory: dict, query_entities: list[Entity]) -> float:
        """
        计算记忆与查询的图谱相关性
        """
        try:
            memory_entity_ids=memory.get("entities",[])
            query_entity_ids=[]
            for e in query_entities:
                if e.name in self.entities_name_to_id:
                    query_entity_ids.append(self.entities_name_to_id[e.name])
                else:
                    query_entity_ids.append(e.entity_id)
            matching_entities=len(set(memory_entity_ids).intersection(set(query_entity_ids)))
            entity_score = matching_entities / len(query_entity_ids) if query_entity_ids else 0
            logger.info(f"🔍 [DEBUG] _calculate_graph_relevance:")
            logger.info(f"  memory_entity_ids={memory_entity_ids}")
            logger.info(f"  query_entity_ids={query_entity_ids}")
            logger.info(f"  matching={matching_entities}, entity_score={entity_score:.4f}")
            logger.info(f"  entity_count={memory.get('entity_count',0)}, relation_count={memory.get('relation_count',0)}")
            # 实体数量加权
            entity_count = memory.get("entity_count", 0)
            entity_density = min(entity_count / 10, 1.0)  # 归一化到[0,1]
            
            # 关系数量加权
            relation_count = memory.get("relation_count", 0)
            relation_density = min(relation_count / 5, 1.0)  # 归一化到[0,1]
            
            # 综合分数
            relevance_score = (
                entity_score * 0.6 +           # 实体匹配权重60%
                entity_density * 0.2 +         # 实体密度权重20%
                relation_density * 0.2         # 关系密度权重20%
            )
            
            return min(relevance_score, 1.0)
        except Exception as e:
            logger.error(f"计算图谱相关性失败: {e}")
            return 0.0
    def _get_graph_context(self, memory: MemoryItem) -> dict[str, Any]:
        """
        获取记忆相关的图谱上下文信息（关联实体和关系）。
        """
        context: dict[str, Any] = {"entities": [], "relations": []}
        try:
            entity_ids = memory.metadata.get("entities", [])
            for entity_id in entity_ids:
                entity = self.entities.get(entity_id)
                if entity:
                    context["entities"].append({
                        "name": entity.name,
                        "type": entity.entity_type,
                        "description": entity.description,
                    })

            relation_strs = memory.metadata.get("relations", [])
            for rel_str in relation_strs:
                parts = rel_str.split("|||")
                if len(parts) == 3:
                    context["relations"].append({
                        "from": parts[0],
                        "type": parts[1],
                        "to": parts[2]
                    })
        except Exception as e:
            logger.warning(f"获取图谱上下文失败: {e}")
        return context

    # ==================== 查找记忆 ====================

    def find_memory(self, id: str) -> bool:
        """检查记忆是否存在"""
        return id in self.id_to_memory

    # ==================== 清除记忆 ====================

    def clear_memory(self) -> None:
        """清空所有记忆（向量库、图数据库、缓存）"""
        try:
            # 1. 清空向量数据库
            all_ids = list(self.id_to_memory.keys())
            if all_ids:
                self.vector_store.remove_vectors(all_ids)
            logger.info("✅ 向量数据库已清空")
        except Exception as e:
            logger.error(f"清空向量数据库失败: {e}")

        try:
            # 2. 清空图数据库
            self.graph_store.clear()
            logger.info("✅ 图数据库已清空")
        except Exception as e:
            logger.error(f"清空图数据库失败: {e}")

        # 3. 清空缓存
        self.memories.clear()
        self.id_to_memory.clear()
        self.memory_embeddings.clear()
        self.entities.clear()
        self.entities_name_to_id.clear()
        self.relations.clear()
        logger.info("✅ 所有语义记忆已清空")

    # ==================== 统计信息 ====================

    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "count": len(self.memories),
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "memory_type": "semantic",
            "max_capacity": self.config.max_capacity,
            "user_ids": list(set(m.user_id for m in self.memories)),
            "avg_importance": (
                sum(m.importance for m in self.memories) / len(self.memories)
                if self.memories else 0.0
            ),
        }

    # ==================== 内部方法 ====================

    def _add_entity_to_graph(self, entity: Entity, memory: MemoryItem) -> bool:
        """添加实体到图数据库"""
        try:
            existing_entity = self.graph_store.get_entity(entity.entity_id)
            if existing_entity:
                existing_entity.frequency += 1
                existing_entity.updated_at = datetime.now()
                self.graph_store.update_entity(existing_entity)
                logger.info(f"更新实体: {entity.entity_id}")
            else:
                properties = {
                    "name": entity.name,
                    "description": entity.description,
                    "frequency": entity.frequency,
                    "memory_id": memory.id,
                    "user_id": memory.user_id,
                    "importance": memory.importance,
                    **entity.properties
                }
                self.graph_store.add_entity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    properties=properties
                )
                logger.info(f"添加新实体: {entity.entity_id}")

            # 更新实体缓存
            if entity.entity_id in self.entities:
                self.entities[entity.entity_id].frequency += 1
                self.entities[entity.entity_id].updated_at = datetime.now()
            else:
                self.entities[entity.entity_id] = entity
                self.entities_name_to_id[entity.name] = entity.entity_id
            return True
        except Exception as e:
            logger.error(f"添加实体到图数据库失败: {e}")
            return False

    def _add_relation_to_graph(self, relation: Relation, memory: MemoryItem) -> bool:
        """添加关系到图数据库"""
        try:
            properties = {
                "strength": relation.strength,
                "evidence": relation.evidence,
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "importance": memory.importance,
                **relation.properties
            }
            success = self.graph_store.add_relation(
                from_entity=relation.from_entity,
                to_entity=relation.to_entity,
                relation_type=relation.relation_type,
                properties=properties
            )
            if success:
                self.relations.append(relation)
                logger.info(f"添加新关系: {relation.from_entity}-{relation.relation_type}-{relation.to_entity}")
            return success
        except Exception as e:
            logger.error(f"添加关系到图数据库失败: {e}")
            return False

    def _remove_entity_from_graph(self, entity_id: str) -> bool:
        """从图数据库删除实体"""
        try:
            self.graph_store.delete_entity(entity_id)
            logger.info(f"✅ 图数据库成功删除实体: {entity_id}")
            return True
        except Exception as e:
            logger.error(f"图数据库删除实体失败: {e}")
            return False

    def _remove_relation_from_graph(self, relation: Relation) -> bool:
        """从图数据库删除关系"""
        try:
            self.graph_store.delete_relation(
                relation.from_entity, relation.to_entity, relation.relation_type
            )
            logger.info(f"✅ 图数据库成功删除关系: "
                        f"{relation.from_entity}-{relation.relation_type}-{relation.to_entity}")
            return True
        except Exception as e:
            logger.error(f"图数据库删除关系失败: {e}")
            return False

    # ==================== 启动加载 / 同步 ====================

    def load_from_store(self):
        """从数据库加载缓存（程序启动时调用）

        从向量存储加载所有记忆 → 重建 memories、id_to_memory、memory_embeddings
        从图存储加载所有实体和关系 → 重建 entities、relations
        """
        # 清空现有缓存
        self.memories.clear()
        self.id_to_memory.clear()
        self.memory_embeddings.clear()
        self.entities.clear()
        self.entities_name_to_id.clear()
        self.relations.clear()

        # 1. 从向量存储加载记忆
        try:
            all_vectors = self.vector_store.get_all_vectors(with_vector=True)
            for item in all_vectors:
                memory_id = item.get("memory_id", "")
                vector = item.get("vector")
                payload = item.get("metadata", {})

                if not memory_id or not payload:
                    continue

                memory = MemoryItem(
                    id=memory_id,
                    user_id=payload.get("user_id", ""),
                    type=payload.get("memory_type", "semantic"),
                    content=payload.get("content", ""),
                    timestamp=datetime.fromtimestamp(payload.get("timestamp", 0)),
                    importance=payload.get("importance", 0.5),
                    metadata={
                        "entities": payload.get("entities", []),
                        "entity_count": payload.get("entity_count", 0),
                        "relation_count": payload.get("relation_count", 0),
                    },
                )
                self.memories.append(memory)
                self.id_to_memory[memory_id] = memory
                if vector is not None:
                    self.memory_embeddings[memory_id] = np.array(vector)

            logger.info(f"✅ 从向量存储加载 {len(self.memories)} 条记忆")
        except Exception as e:
            logger.error(f"从向量存储加载记忆失败: {e}")

        # 2. 从图存储加载实体
        try:
            all_entities = self.graph_store.get_all_entities()
            for entity in all_entities:
                self.entities[entity.entity_id] = entity
                self.entities_name_to_id[entity.name] = entity.entity_id
            logger.info(f"✅ 从图存储加载 {len(self.entities)} 个实体")
        except Exception as e:
            logger.error(f"从图存储加载实体失败: {e}")

        # 3. 从图存储加载关系
        try:
            all_relations = self.graph_store.get_all_relations()
            self.relations = all_relations
            logger.info(f"✅ 从图存储加载 {len(self.relations)} 个关系")
        except Exception as e:
            logger.error(f"从图存储加载关系失败: {e}")

        # 4. 补充记忆的 relations 元数据（从图数据反推）
        self._rebuild_relation_metadata()

        logger.info(f"✅ 缓存加载完成: {len(self.memories)} 条记忆, "
                     f"{len(self.entities)} 个实体, {len(self.relations)} 个关系")

    def _rebuild_relation_metadata(self):
        """根据实体的 memory_id 属性，将关系元数据补充到对应的记忆中"""
        # 建立 entity_id → memory_id 的映射
        entity_to_memory: dict[str, str] = {}
        for entity_id, entity in self.entities.items():
            mid = entity.properties.get("memory_id", "")
            if mid:
                entity_to_memory[entity_id] = mid

        for relation in self.relations:
            # 关系的 from_entity 对应的 memory_id
            mid = entity_to_memory.get(relation.from_entity, "")
            if mid and mid in self.id_to_memory:
                memory = self.id_to_memory[mid]
                rel_str = f"{relation.from_entity}|||{relation.relation_type}|||{relation.to_entity}"
                existing = memory.metadata.get("relations", [])
                if rel_str not in existing:
                    existing.append(rel_str)
                    memory.metadata["relations"] = existing

    def sync_stores(self) -> dict[str, Any]:
        """检测并修复缓存、向量存储与图存储之间的数据不一致

        Returns:
            同步结果统计信息
        """
        stats = {
            "cache_count": len(self.memories),
            "synced_to_vector": 0,
            "synced_from_graph": 0,
            "removed_orphan_vectors": 0,
            "errors": [],
        }

        # 1. 从图存储获取所有实体和关系
        try:
            graph_entities = self.graph_store.get_all_entities()
            graph_entity_ids = {e.entity_id for e in graph_entities}
        except Exception as e:
            stats["errors"].append(f"无法读取图存储实体: {e}")
            graph_entities = []
            graph_entity_ids = set()

        try:
            graph_relations = self.graph_store.get_all_relations()
        except Exception as e:
            stats["errors"].append(f"无法读取图存储关系: {e}")
            graph_relations = []

        # 2. 同步实体缓存 — 补充缓存中缺失的实体
        cache_entity_ids = set(self.entities.keys())
        missing_entities = graph_entity_ids - cache_entity_ids
        for entity in graph_entities:
            if entity.entity_id in missing_entities:
                self.entities[entity.entity_id] = entity
                self.entities_name_to_id[entity.name] = entity.entity_id
                stats["synced_from_graph"] += 1

        # 3. 同步关系缓存
        existing_rel_keys = {
            (r.from_entity, r.relation_type, r.to_entity) for r in self.relations
        }
        for relation in graph_relations:
            key = (relation.from_entity, relation.relation_type, relation.to_entity)
            if key not in existing_rel_keys:
                self.relations.append(relation)
                existing_rel_keys.add(key)
                stats["synced_from_graph"] += 1

        # 4. 获取向量存储中的所有记忆
        try:
            vector_items = self.vector_store.get_all_vectors(with_vector=True)
            vector_ids = {item.get("memory_id", "") for item in vector_items}
        except Exception as e:
            stats["errors"].append(f"无法读取向量存储: {e}")
            vector_items = []
            vector_ids = set()

        # 5. 缓存中有但向量存储中缺失的记忆 → 重新嵌入
        cache_ids = set(self.id_to_memory.keys())
        missing_in_vector = cache_ids - vector_ids
        for memory_id in missing_in_vector:
            memory = self.id_to_memory[memory_id]
            try:
                embedding = self.embedding_model.embed([memory.content])
                metadata = {
                    "memory_id": memory.id,
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "timestamp": int(memory.timestamp.timestamp()),
                    "importance": memory.importance,
                    "memory_type": memory.type,
                    "entities": memory.metadata.get("entities", []),
                    "entity_count": memory.metadata.get("entity_count", 0),
                    "relation_count": memory.metadata.get("relation_count", 0),
                }
                self.vector_store.add_vectors(
                    vectors=embedding,
                    metadata=[metadata],
                    ids=[memory_id]
                )
                self.memory_embeddings[memory_id] = np.array(embedding[0])
                stats["synced_to_vector"] += 1
            except Exception as e:
                stats["errors"].append(f"同步向量失败 {memory_id[:8]}...: {e}")

        # 6. 向量存储中有但缓存中没有的 → 加载到缓存
        missing_in_cache = vector_ids - cache_ids
        for item in vector_items:
            mid = item.get("memory_id", "")
            if mid not in missing_in_cache:
                continue
            payload = item.get("metadata", {})
            vector = item.get("vector")
            try:
                memory = MemoryItem(
                    id=mid,
                    user_id=payload.get("user_id", ""),
                    type=payload.get("memory_type", "semantic"),
                    content=payload.get("content", ""),
                    timestamp=datetime.fromtimestamp(payload.get("timestamp", 0)),
                    importance=payload.get("importance", 0.5),
                    metadata={
                        "entities": payload.get("entities", []),
                        "entity_count": payload.get("entity_count", 0),
                        "relation_count": payload.get("relation_count", 0),
                    },
                )
                self.memories.append(memory)
                self.id_to_memory[mid] = memory
                if vector is not None:
                    self.memory_embeddings[mid] = np.array(vector)
            except Exception as e:
                stats["errors"].append(f"加载缓存失败 {mid[:8]}...: {e}")

        logger.info(f"✅ 同步完成: 从图存储补充 {stats['synced_from_graph']} 条, "
                     f"向量存储新增 {stats['synced_to_vector']} 条")
        return stats

    def forget(self, strategy:ForgetType, threshold: float = 0.1, max_age_days: int = 30) -> int:
        
        forgotten_count=0
        current_time=datetime.now()
        to_remove=[]
        if strategy.value==ForgetType.TIME.value:
            for memory in self.memories:
                memory_time=memory.timestamp
                age_days=(current_time-memory_time).days
                if age_days>max_age_days:
                    to_remove.append(memory.id)

        elif strategy.value==ForgetType.IMPORTANCE.value:
            for memory in self.memories:
                if memory.importance<threshold:
                    to_remove.append(memory.id)

        elif strategy.value==ForgetType.CAPACITY.value:
             if len(self.memories)>self.config.max_capacity:
                sorted_memories=sorted(self.memories,key=lambda x:x.importance)
                num_to_remove=len(self.memories)-self.config.max_capacity
                to_remove=[memory.id for memory in sorted_memories[:num_to_remove]]

        for memory_id in to_remove:
            if self.remove_memory(memory_id):
                forgotten_count+=1
                logger.info(f"已遗忘记忆: {memory_id}")
        return forgotten_count

    def get_all_memories(self)->List[MemoryItem]:
        return self.memories.copy()

    
    def get_entity(self,entity_id:str)->Optional[Entity]:
        return self.entities.get(entity_id)

    def search_entities(self,query:str,limit:int=5,user_id:Optional[str]=None)->list[Entity]:
        """搜索实体"""
        query_lower=query.lower()

        martched_entities=[]
        for entity in self.entities.values():
            score=0
            if query_lower in entity.name.lower():
                score+=10
            if query_lower in entity.description.lower():
                score+=5
            if score>0:
                martched_entities.append((entity,score))
        martched_entities.sort(key=lambda x:x[1],reverse=True)
        return [entity for entity,score in martched_entities[:limit]]

    def _get_entity_id_by_name(self,entity_name:str)->Optional[str]:
        """通过实体名称获取实体ID"""
        return self.entities_name_to_id.get(entity_name)
    def get_related_entities(self,entity_id:str,relation_type:Optional[str]=None,max_hops:int=2,limit:int=10)->list[Entity]:
        """获取相关实体"""
        if entity_id not in self.entities:
            return []

        entity=self.entities[entity_id]
        related_entities=[]
        try:
            if not self.graph_store:
                logger.warning("图存储未初始化，无法获取相关实体")
                return []
            
            related_entities=self.graph_store.get_related_entities(
                entity_name=entity.name,
                rel_type=relation_type,
                max_depth=max_hops,
                limit=limit
            )
            return related_entities
        except Exception as e:
            logger.error(f"获取相关实体失败: {e}")
            return []

    

    