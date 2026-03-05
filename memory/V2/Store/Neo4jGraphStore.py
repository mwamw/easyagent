from argparse import OPTIONAL
try:
    from .GraphStore import GraphStore, Entity, Relation
except ImportError:
    from Store.GraphStore import GraphStore, Entity, Relation
from neo4j import GraphDatabase
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        # 验证连接
        self._driver.verify_connectivity()
        logger.info(f"已连接到 Neo4j: {uri}, 数据库: {database}")

    def close(self):
        """关闭数据库连接"""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ─── 实体操作 ───────────────────────────────────────────

    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: dict) -> bool:
        """添加实体，使用 entity_type 作为节点标签"""
        query = (
            "CREATE (e:`{entity_type}` {entity_id: $entity_id, name: $name, "
            "entity_type: $entity_type, created_at: datetime(), "
            "updated_at: datetime(), frequency: 1})"
        ).replace("{entity_type}", entity_type)
        # 将额外 properties 合并写入
        if properties:
            query = (
                "CREATE (e:`{entity_type}` {entity_id: $entity_id, name: $name, "
                "entity_type: $entity_type, created_at: datetime(), "
                "updated_at: datetime(), frequency: 1}) "
                "SET e += $properties"
            ).replace("{entity_type}", entity_type)
        try:
            with self._driver.session(database=self._database) as session:
                session.run(query, entity_id=entity_id, name=name,
                            entity_type=entity_type, properties=properties or {})
            logger.info(f"已添加实体: {name} ({entity_type})")
            return True
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False

    def add_relation(self, from_entity: str, to_entity: str, relation_type: str, properties: dict) -> bool:
        """添加关系，通过 entity_id 匹配两端节点"""
        # 动态关系类型需要用 APOC 或字符串拼接；这里用安全拼接
        query = (
            "MATCH (a {entity_id: $from_id}), (b {entity_id: $to_id}) "
            "CREATE (a)-[r:`{rel_type}` {relation_type: $rel_type, "
            "created_at: datetime(), updated_at: datetime(), frequency: 1}]->(b) "
            "SET r += $properties"
        ).replace("{rel_type}", relation_type)
        try:
            with self._driver.session(database=self._database) as session:
                session.run(query, from_id=from_entity, to_id=to_entity,
                            rel_type=relation_type, properties=properties or {})
            logger.info(f"已添加关系: {from_entity} -[{relation_type}]-> {to_entity}")
            return True
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return False

    # ─── 删除操作 ───────────────────────────────────────────

    def delete_entity(self, entity_id: str) -> bool:
        """删除实体及其所有关联关系"""
        query = "MATCH (e {entity_id: $entity_id}) DETACH DELETE e"
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, entity_id=entity_id)
                summary = result.consume()
            deleted = summary.counters.nodes_deleted > 0
            if deleted:
                logger.info(f"已删除实体: {entity_id}")
            else:
                logger.warning(f"未找到实体: {entity_id}")
            return deleted
        except Exception as e:
            logger.error(f"删除实体失败: {e}")
            return False

    def delete_relation(self, from_entity: str, to_entity: str, relation_type: str) -> bool:
        """删除指定关系"""
        query = (
            "MATCH (a {entity_id: $from_id})-[r:`{rel_type}`]->(b {entity_id: $to_id}) "
            "DELETE r"
        ).replace("{rel_type}", relation_type)
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, from_id=from_entity, to_id=to_entity)
                summary = result.consume()
            deleted = summary.counters.relationships_deleted > 0
            if deleted:
                logger.info(f"已删除关系: {from_entity} -[{relation_type}]-> {to_entity}")
            else:
                logger.warning(f"未找到关系: {from_entity} -[{relation_type}]-> {to_entity}")
            return deleted
        except Exception as e:
            logger.error(f"删除关系失败: {e}")
            return False

    # ─── 更新操作 ───────────────────────────────────────────

    def update_entity(self, entity: Entity) -> bool:
        """更新实体属性"""
        query = (
            "MATCH (e {entity_id: $entity_id}) "
            "SET e.name = $name, e.entity_type = $entity_type, "
            "e.description = $description, e.frequency = $frequency, "
            "e.updated_at = datetime() "
            "SET e += $properties"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    query,
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    frequency=entity.frequency,
                    properties=entity.properties or {},
                )
                summary = result.consume()
            updated = summary.counters.properties_set > 0
            if updated:
                logger.info(f"已更新实体: {entity.entity_id}")
            else:
                logger.warning(f"未找到要更新的实体: {entity.entity_id}")
            return updated
        except Exception as e:
            logger.error(f"更新实体失败: {e}")
            return False

    # ─── 查询操作 ───────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        query = "MATCH (e {entity_id: $entity_id}) RETURN e"
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, entity_id=entity_id)
                record = result.single()
            if record is None:
                return None
            node = record["e"]
            return self._node_to_entity(node)
        except Exception as e:
            logger.error(f"获取实体失败: {e}")
            return None

    def get_relation(self, from_entity: str, to_entity: str, relation_type: str) -> Optional[Relation]:
        """获取关系"""
        query = (
            "MATCH (a {entity_id: $from_id})-[r:`{rel_type}`]->(b {entity_id: $to_id}) "
            "RETURN r, a.entity_id AS from_id, b.entity_id AS to_id"
        ).replace("{rel_type}", relation_type)
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, from_id=from_entity, to_id=to_entity)
                record = result.single()
            if record is None:
                return None
            rel = record["r"]
            return self._rel_to_relation(rel, record["from_id"], record["to_id"])
        except Exception as e:
            logger.error(f"获取关系失败: {e}")
            return None

    def delete_entity_by_memoryid(self,memory_id:str)->bool:
        """通过记忆ID删除实体"""
        query = (
            "MATCH (e {memory_id: $memory_id}) "
            "DETACH DELETE e"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, memory_id=memory_id)
                summary = result.consume()
            deleted = summary.counters.nodes_deleted > 0
            if deleted:
                logger.info(f"已删除实体: {memory_id}")
            else:
                logger.warning(f"未找到要删除的实体: {memory_id}")
            return deleted
        except Exception as e:
            logger.error(f"删除实体失败: {e}")
            return False
    
    def delete_relation_by_memoryid(self,memory_id:str)->bool:
        """通过记忆ID删除关系"""
        query = (
            "MATCH (a {memory_id: $memory_id})-[r]-(b) "
            "DETACH DELETE r"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, memory_id=memory_id)
                summary = result.consume()
            deleted = summary.counters.relationships_deleted > 0
            if deleted:
                logger.info(f"已删除关系: {memory_id}")
            else:
                logger.warning(f"未找到要删除的关系: {memory_id}")
            return deleted
        except Exception as e:
            logger.error(f"删除关系失败: {e}")
            return False
    # ─── 内部工具方法 ──────────────────────────────────────

    @staticmethod
    def _node_to_entity(node) -> Entity:
        """将 Neo4j Node 转换为 Entity 对象"""
        props = dict(node)
        # 提取已知字段，剩余作为 properties
        entity_id = props.pop("entity_id", "")
        entity_type = props.pop("entity_type", "")
        name = props.pop("name", "")
        description = props.pop("description", "")
        frequency = props.pop("frequency", 1)
        props.pop("created_at", None)
        props.pop("updated_at", None)
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=description,
            properties=props,
        )
        entity.frequency = frequency
        return entity

    @staticmethod
    def _rel_to_relation(rel, from_id: str, to_id: str) -> Relation:
        """将 Neo4j Relationship 转换为 Relation 对象"""
        props = dict(rel)
        relation_type = props.pop("relation_type", rel.type)
        strength = props.pop("strength", 1.0)
        evidence = props.pop("evidence", "")
        frequency = props.pop("frequency", 1)
        props.pop("created_at", None)
        props.pop("updated_at", None)
        relation = Relation(
            from_entity=from_id,
            to_entity=to_id,
            relation_type=relation_type,
            strength=strength,
            evidence=evidence,
            properties=props,
        )
        relation.frequency = frequency
        return relation

    def get_all_entities(self) -> list[Entity]:
        """获取所有实体"""
        query = "MATCH (e) RETURN e"
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                records = result.data()
            entities = [self._node_to_entity(record["e"]) for record in records]
            return entities
        except Exception as e:
            logger.error(f"获取所有实体失败: {e}")
            return []

    def get_all_relations(self) -> list[Relation]:
        """获取所有关系"""
        query = "MATCH (a)-[r]->(b) RETURN r, a.entity_id as from_id, b.entity_id as to_id"
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                # 不使用 .data()，直接迭代 Record 以保留原生 Neo4j Relationship 对象
                relations = []
                for record in result:
                    rel = record["r"]
                    from_id = record["from_id"]
                    to_id = record["to_id"]
                    relations.append(self._rel_to_relation(rel, from_id, to_id))
            return relations
        except Exception as e:
            logger.error(f"获取所有关系失败: {e}")
            return []

    def clear(self) -> bool:
        """清空图数据库"""
        query = "MATCH (n) DETACH DELETE n"
        try:
            with self._driver.session(database=self._database) as session:
                session.run(query)
            return True
        except Exception as e:
            logger.error(f"清空图数据库失败: {e}")
            return False

    def get_related_entities(self,entity_name:str,limit:int,user_id:Optional[str]=None,rel_type:Optional[str]=None,max_depth:int=2)->list[Entity]:
        """获取相关实体"""
        # Neo4j 不允许在 [r*1..N] 中使用参数，必须内联
        depth = int(max_depth)
        # 构建关系模式：有 rel_type 时指定类型，否则匹配所有类型
        if rel_type:
            rel_pattern = f"[*1..{depth}]" if not rel_type else f"[:`{rel_type}`*1..{depth}]"
        else:
            rel_pattern = f"[*1..{depth}]"
        # 构建 WHERE 子句
        conditions = ["e.name = $entity_name"]
        if user_id:
            conditions.append("e.user_id = $user_id")
        where = "WHERE " + " AND ".join(conditions)
        query = (
            f"MATCH (e)-{rel_pattern}-(n) "
            f"{where} "
            "RETURN DISTINCT n "
            "LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, entity_name=entity_name, user_id=user_id, limit=int(limit))
                records = result.data()
            entities = [self._node_to_entity(record["n"]) for record in records]
            return entities
        except Exception as e:
            logger.error(f"获取相关实体失败: {e}")
            return []
    
    def get_entity_relations(self,entity_name:str,user_id:Optional[str]=None,rel_type:Optional[str]=None,max_depth:int=2,limit:int=10)->list[Entity]:
        """获取相关关系"""
        depth = int(max_depth)
        if rel_type:
            rel_pattern = f"[:`{rel_type}`*1..{depth}]"
        else:
            rel_pattern = f"[*1..{depth}]"
        conditions = ["e.name = $entity_name"]
        if user_id:
            conditions.append("e.user_id = $user_id")
        where = "WHERE " + " AND ".join(conditions)
        query = (
            f"MATCH (e)-{rel_pattern}-(n) "
            f"{where} "
            "RETURN DISTINCT e "
            "LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, entity_name=entity_name, user_id=user_id, limit=int(limit))
                records = result.data()
            entities = [self._node_to_entity(record["e"]) for record in records]
            return entities
        except Exception as e:
            logger.error(f"获取相关关系失败: {e}")
            return []

    def get_stats(self)->dict:
        """获取图数据库统计信息"""
        query = "MATCH (n)-[r]->(m) RETURN count(n) as node_count, count(r) as rel_count"
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                record = result.single()
            return {
                "node_count": record["node_count"],
                "rel_count": record["rel_count"],
            }
        except Exception as e:
            logger.error(f"获取图数据库统计信息失败: {e}")
            return {
                "node_count": 0,
                "rel_count": 0,
            }
