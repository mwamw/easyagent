import sys
import os
import uuid
import time
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))

from dotenv import load_dotenv
load_dotenv()

from memory.V2.Store.Neo4jGraphStore import Neo4jGraphStore
from memory.V2.Store.GraphStore import Entity,Relation
if __name__ == "__main__":
    # 示例用法
    store = Neo4jGraphStore(
        uri="bolt://localhost:17687",
        username="neo4j",
        password="password",
        database="neo4j"
    )

    # 创建实体
    entity1 = Entity(
        entity_id="user_123",
        entity_type="user",
        name="张三",
        description="系统管理员",
        properties={"email": "[EMAIL_ADDRESS]", "role": "admin"}
    )
    store.add_entity(entity1.entity_id,name="张三",entity_type="user",properties={"email": "[EMAIL_ADDRESS]", "role": "admin"})

    entity2 = Entity(
        entity_id="project_abc",
        entity_type="project",
        name="项目管理系统",
        description="核心项目",
        properties={"status": "active", "budget": 100000}
    )
    store.add_entity(entity2.entity_id,name="项目管理系统",entity_type="project",properties={"status": "active", "budget": 100000})

    # 创建关系
    relation = Relation(
        from_entity="user_123",
        to_entity="project_abc",
        relation_type="MANAGES",
        strength=0.9,
        evidence="用户在系统中创建了项目",
        properties={"since": "2023-01-01"}
    )
    store.add_relation(from_entity="user_123",to_entity="project_abc",relation_type="MANAGES",properties={"strength": 0.9, "evidence": "用户在系统中创建了项目", "since": "2023-01-01"})

    # 查询实体
    retrieved_entity = store.get_entity("user_123")
    print(f"获取实体: {retrieved_entity}")

    print(retrieved_entity.to_dict())
    # 查询关系
    retrieved_relation = store.get_relation("user_123", "project_abc", "MANAGES")
    print(f"获取关系: {retrieved_relation}")

    #搜索实体
    retrieved_entities = store.get_related_entities("项目管理系统", 10)
    print(f"获取相关实体: {retrieved_entities}")
    for entity in retrieved_entities:
        print(entity.to_dict())


    #搜索实体
    retrieved_relations = store.get_entity_relations("项目管理系统")
    print(f"获取相关实体: {retrieved_relations}")
    for relation in retrieved_relations:
        print(relation.to_dict())
    # 更新实体

    # 删除关系
    # store.delete_relation("user_123", "project_abc", "MANAGES")

    # 关闭连接
    store.close()