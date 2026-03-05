from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from typing import Any
class Entity:
    def __init__(self,entity_id:str,entity_type:str,name:str,description:str,properties:dict[str,Any]):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.name = name
        self.description = description
        self.properties = properties
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1  # 出现频率
    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency,
        }
class Relation:
    def __init__(self,from_entity:str,to_entity:str,relation_type:str,strength:float,evidence:str,properties:dict[str,Any]):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence
        self.properties = properties
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1  # 出现频率
    def to_dict(self):
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency,
        }
class GraphStore(ABC):
    """图数据库接口"""
    @abstractmethod
    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: dict) -> bool:
        """添加实体"""
        pass
    
    @abstractmethod
    def add_relation(self, from_entity: str, to_entity: str, relation_type: str, properties: dict) -> bool:
        """添加关系"""
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        pass

    @abstractmethod
    def delete_relation(self, from_entity: str, to_entity: str, relation_type: str) -> bool:
        """删除关系"""
        pass

    @abstractmethod
    def update_entity(self, entity:Entity) -> bool:
        """更新实体"""
        pass

    @abstractmethod
    def delete_entity_by_memoryid(self,memory_id:str)->bool:
        """通过记忆ID删除实体"""
        pass
    
    @abstractmethod
    def delete_relation_by_memoryid(self,memory_id:str)->bool:
        """通过记忆ID删除关系"""
        pass
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        pass

    @abstractmethod
    def get_relation(self, from_entity: str, to_entity: str, relation_type: str) -> Optional[Relation]:
        """获取关系"""
        pass
    
    @abstractmethod
    def get_all_entities(self) -> list[Entity]:
        """获取所有实体"""
        pass
    
    @abstractmethod
    def get_all_relations(self) -> list[Relation]:
        """获取所有关系"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空图数据库"""
        pass

    @abstractmethod
    def get_related_entities(self,entity_name:str,limit:int,user_id:Optional[str]=None,rel_type:Optional[str]=None,max_depth:int=2)->list[Entity]:
        pass
    
    @abstractmethod
    def get_entity_relations(self,entity_name:str,user_id:str,rel_type:Optional[str]=None,max_depth:int=2,limit:int=10)->list[Entity]:
        """获取相关关系"""
        pass

    @abstractmethod
    def get_stats(self)->dict[str,Any]:
        """获取图数据库统计信息"""
        pass
        