from BaseMemory import MemoryConfig,BaseMemory,MemoryItem,MemoryType,ForgetType
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import os
import torch
import random
import logging
import numpy as np
from Store.VectorStore import VectorStore
from Store.DocumentStore import DocumentStore
from Embedding.BaseEmbeddingModel import BaseEmbeddingModel
from PIL import Image
import io
logger = logging.getLogger(__name__)

class Perception:
    def __init__(self,perception_id:str,data:Any,modality:str,encoding:Optional[List[float]]=None,metadata:Optional[dict[str,Any]]=None):
        self.perception_id = perception_id
        self.data = data
        self.modality = modality  # text, image, audio, video, structured
        self.encoding = encoding or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """计算数据哈希"""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()

class PerceptualMemory(BaseMemory):
    def __init__(self,memory_config:MemoryConfig,document_store:DocumentStore,vector_stores:dict[str,VectorStore],embedding_model:BaseEmbeddingModel,supported_modalities:Optional[List[str]]=None
    ,image_encoder:Optional[str]=None,audio_encoder:Optional[str]=None,video_encoder:Optional[str]=None):
        super().__init__(memory_config)
        self.document_store=document_store
        self.supported_modalities=supported_modalities if supported_modalities else ["text"]
        self.vector_stores=vector_stores
        self.embedding_model=embedding_model
        self.text_embedding_dim=self.embedding_model.dimension

        #多模态编码器
        self.image_encoder=None
        self.image_processor=None
        self.image_embedding_dim=None
        self.audio_encoder=None
        self.audio_processor=None
        self.audio_embedding_dim=None
        if "image" in self.supported_modalities:
            try:
                from transformers import CLIPProcessor,CLIPModel
                image_encoder_name=image_encoder if image_encoder else os.getenv("IMAGE_ENCODER", "openai/clip-vit-base-patch32")
                self.image_processor=CLIPProcessor.from_pretrained(image_encoder_name)
                self.image_encoder=CLIPModel.from_pretrained(image_encoder_name)
                self.image_embedding_dim=self.image_encoder.config.projection_dim if hasattr(self.image_encoder.config, 'projection_dim') else 512
            except Exception as e:
                logger.error(f"加载图像编码器失败: {e}")

        if "audio" in self.supported_modalities:
            try:
                from transformers import ClapProcessor,ClapModel
                audio_encoder_name=audio_encoder if audio_encoder else os.getenv("AUDIO_ENCODER", "laion/clap-htsat-unfused")
                self.audio_processor=ClapProcessor.from_pretrained(audio_encoder_name)
                self.audio_encoder=ClapModel.from_pretrained(audio_encoder_name)
                self.audio_embedding_dim=self.audio_encoder.config.projection_dim if hasattr(self.audio_encoder.config, 'projection_dim') else 512
            except Exception as e:
                logger.error(f"加载音频编码器失败: {e}")
        
        # 多模态向量存储
        if "text" in self.supported_modalities:
            self.text_vector_store=self.vector_stores.get("text")
            if not self.text_vector_store:
                logger.error("需要提供文本向量存储")
        if "image" in self.supported_modalities:
            self.image_vector_store=self.vector_stores.get("image")
            if not self.image_vector_store:
                logger.error("需要提供图像向量存储")
        if "audio" in self.supported_modalities:
            self.audio_vector_store=self.vector_stores.get("audio")
            if not self.audio_vector_store:
                logger.error("需要提供音频向量存储")

        # 缓存
        self.perceptions:dict[str,Perception]={}
        self.perceptual_memories:list[MemoryItem]=[]
        self.modality_index:dict[str,list[str]]={}
        self.id_to_memory:dict[str,MemoryItem]={}


    def add_memory(self,item:MemoryItem)->str:
        """添加单条记忆

        Args:
            item: 记忆对象

        Returns:
            记忆ID，失败返回空字符串
        """
        modality=item.metadata.get("modality","text")
        raw_data=item.metadata.get("raw_data",item.content)
        if modality not in self.supported_modalities:
            logger.warning(f"不支持的模态: {modality}")
            return ""
        
        data_vector=self._encoder_data(raw_data,modality)
        if not data_vector:
            logger.warning(f"{modality}编码器编码失败")
            return ""

        
        perception=Perception(perception_id=f"Perception_{item.id}",data=raw_data,modality=modality,encoding=data_vector,metadata=item.metadata)
        item.metadata["perception_id"]=perception.perception_id
        item.metadata["raw_data"]=raw_data
        item.metadata["modality"]=modality
        ts_int=int(item.timestamp.timestamp())
        # document stroe
        self.document_store.add_memory(
            memory_id=item.id,
            user_id=item.user_id,
            content=item.content,
            memory_type="perceptual",
            timestamp=ts_int,
            importance=item.importance,
            properties={
                "modality":modality,
                "raw_data":raw_data,
                "context":item.metadata.get("context",{}),
                "perception_id":perception.perception_id,
            }
        )

        # vector store
        vector_store=self.vector_stores.get(modality)
        if vector_store:
            vector=perception.encoding
            vector_store.add_vectors(
                vectors=[vector],
                metadata=[{
                    "memory_id":item.id,
                    "user_id":item.user_id,
                    "modality":modality,
                    "timestamp":ts_int,
                    "memory_type":"perceptual",
                    "importance":item.importance,
                    "context":item.metadata.get("context",{}),
                    "perception_id":perception.perception_id,
                }],
                ids=[item.id]
            )
        
        # cache
        self.perceptions[perception.perception_id]=perception
        self.perceptual_memories.append(item)
        self.modality_index.setdefault(modality,[]).append(perception.perception_id)
        self.id_to_memory[item.id]=item
        
        return item.id

    def _encoder_data(self,data:Any,modality:str)->Optional[list[float]]:
        if modality=="text":
            return self._encoder_text(data)
        elif modality=="image":
            return self._encoder_image(data)
        elif modality=="audio":
            return self._encoder_audio(data)
        else:
            logger.warning(f"不支持的模态: {modality}")
            return None
    def _encoder_image(self,data:Union[str,bytes,Image.Image])->Optional[list[float]]:
        """编码图像数据"""
        if self.image_encoder and self.image_processor:
            if isinstance(data,str):
                from PIL import Image
                image=Image.open(data)
            elif isinstance(data,bytes):
                from PIL import Image
                image=Image.open(io.BytesIO(data))
            else:
                image=data
            inputs=self.image_processor(images=image,return_tensors="pt")
            with torch.no_grad():
                feats=self.image_encoder.get_image_features(**inputs)
            return feats[0].detach().cpu().numpy().tolist()
        else:
            logger.warning("图像编码器编码失败")
            return None
    
    def _encoder_audio(self,data:Union[str,bytes,np.ndarray])->Optional[list[float]]:
        """编码音频数据"""
        if self.audio_encoder and self.audio_processor:
            if isinstance(data,str):
                import librosa
                audio_data,sr=librosa.load(data,sr=16000)
            elif isinstance(data,bytes):
                import librosa
                audio_data,sr=librosa.load(io.BytesIO(data),sr=16000)
            else:
                audio_data=data
            inputs=self.audio_processor(audio_data,sampling_rate=16000,return_tensors="pt")
            with torch.no_grad():
                feats=self.audio_encoder.get_audio_features(**inputs)
            return feats[0].detach().cpu().numpy().tolist()
        else:
            logger.warning("音频编码器编码失败")
            return None
    
    def _encoder_text(self,data:str)->Optional[list[float]]:
        """编码文本数据"""
        if self.embedding_model:
            return self.embedding_model.embed([data])[0]
        else:
            logger.warning("文本编码器编码失败")
            return None
    def add_memories_batch(self, items: List[MemoryItem]) -> List[str]:
        """批量添加记忆 (默认实现，子类可覆写以优化)

        Args:
            items: 记忆对象列表

        Returns:
            成功添加的记忆ID列表
        """
        results = []
        for item in items:
            result = self.add_memory(item)
            if result:
                results.append(result)
        return results


    def remove_memory(self,memory_id:str) -> bool:
        #处理cache
        perception_id=f"Perception_{memory_id}"
        removed_cache=False
        if perception_id in self.perceptions:
            removed_perception=self.perceptions.pop(perception_id)
            self.perceptual_memories.remove(self.id_to_memory.pop(memory_id))
            self.modality_index[removed_perception.modality].remove(perception_id)
            removed_cache=True
        #处理document store
        removed_document=False
        try:
            self.document_store.remove_memory(memory_id)
            removed_document=True
        except Exception as e:
            logger.error(f"移除记忆失败: {e}")
        #处理vector store
        removed_vector=False
        try:
            self.vector_stores[removed_perception.modality].remove_vectors([memory_id])
            removed_vector=True
        except Exception as e:
            logger.error(f"移除记忆失败: {e}")
        return removed_cache and removed_document and removed_vector

    
    def update_memory(self,id:str,content:str,importance:Optional[float]=None,metadata:Optional[dict[str,Any]]=None) -> bool:
        if id not in self.id_to_memory:
            return False
        memory=self.id_to_memory[id]
        memory.content=content
        if importance:
            memory.importance=importance
        if metadata:
            memory.metadata.update(metadata)
        
        # 更新document store
        try:
            self.document_store.update_memory(id,content,importance,memory.metadata)
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            return False
        # 更新vector store
        if content or (metadata and "raw_data" in metadata):
            try:
                data_vector=self._encoder_data(memory.metadata["raw_data"],memory.metadata["modality"])
                if data_vector:
                    perception=Perception(
                        perception_id=f"Perception_{id}",
                        data=memory.metadata["raw_data"],
                        modality=memory.metadata["modality"],
                        encoding=data_vector,
                        metadata=memory.metadata
                    )
                    self.perceptions[f"Perception_{id}"]=perception
                    self.vector_stores[memory.metadata["modality"]].remove_vectors([id])
                    self.vector_stores[memory.metadata["modality"]].add_vectors(
                    vectors=[data_vector],
                    metadata=[{
                        "memory_id":id,
                        "user_id":memory.user_id,
                        "modality":memory.metadata["modality"],
                        "timestamp":memory.timestamp,
                        "importance":memory.importance,
                        "context":memory.metadata.get("context",{}),
                        "perception_id":perception.perception_id,
                    }],
                    ids=[id]
                )
            except Exception as e:
                logger.error(f"更新记忆失败: {e}")
                return False
        return True

    def search_memory(self,query:str,limit:int=5,user_id:Optional[str]=None,**kwargs) -> List[MemoryItem]:
        target_modality=kwargs.get("modality",None)
        query_modality=self._detach_query_modality(query)
        session_id=kwargs.get("session_id",None)
        if target_modality and target_modality not in self.supported_modalities:
            logger.warning(f"不支持的模态: {target_modality}")
            return []
        # 向量数据库所搜
        vector_search_result=[]
        if not target_modality:
            try:
                if query_modality=="text":
                    query_text_vector=self._encoder_text(query)
                    query_image_vector=self._encoder_text_clip(query)
                    query_audio_vector=self._encoder_text_clap(query)
                    if not query_text_vector:
                        logger.warning("文本编码失败")
                        return []
                    if not query_image_vector:
                        logger.warning("clip:文本编码图像失败")
                        return []
                    if not query_audio_vector:
                        logger.warning("clap:文本编码音频失败")
                        return []
                    text_vector_store=self.vector_stores["text"]
                    image_vector_store=self.vector_stores["image"]
                    audio_vector_store=self.vector_stores["audio"]
                    where={}
                    where["memory_type"]="perceptual"
                    if user_id:
                        where["user_id"]=user_id
                    where["modality"]="text"
                    text_search_result=text_vector_store.search_similar(
                        query_embedding=query_text_vector,
                        limit=limit,
                        where=where
                    )
                    where["modality"]="image"
                    image_search_result=image_vector_store.search_similar(
                        query_embedding=query_image_vector,
                        limit=limit,
                        where=where
                    )
                    where["modality"]="audio"
                    audio_search_result=audio_vector_store.search_similar(
                        query_embedding=query_audio_vector,
                        limit=limit,
                        where=where
                    )
                    vector_search_result=text_search_result+image_search_result+audio_search_result
                    
                elif query_modality=="image":
                    query_image_vector=self._encoder_image(query)
                    if not query_image_vector:
                        logger.warning("图像编码失败")
                        return []
                    image_vector_store=self.vector_stores["image"]
                    where={}
                    where["memory_type"]="perceptual"
                    if user_id:
                        where["user_id"]=user_id
                    where["modality"]="image"
                    image_search_result=image_vector_store.search_similar(
                        query_embedding=query_image_vector,
                        limit=limit,
                        where=where
                    )
                    vector_search_result=image_search_result
                elif query_modality=="audio":
                    query_audio_vector=self._encoder_audio(query)
                    if not query_audio_vector:
                        logger.warning("音频编码失败")
                        return []
                    audio_vector_store=self.vector_stores["audio"]
                    where={}
                    where["memory_type"]="perceptual"
                    if user_id:
                        where["user_id"]=user_id
                    where["modality"]="audio"
                    audio_search_result=audio_vector_store.search_similar(
                        query_embedding=query_audio_vector,
                        limit=limit,
                        where=where
                    )
                    vector_search_result=audio_search_result
            except Exception as e:
                logger.error(f"搜索记忆失败: {e}")
                return []
        else:
            if target_modality==query_modality:
                try:
                    query_vector=self._encoder_data(query,target_modality)
                    if not query_vector:
                        logger.warning("编码失败")
                        return []
                    vector_store=self.vector_stores[target_modality]
                    where={}
                    where["memory_type"]="perceptual"
                    if user_id:
                        where["user_id"]=user_id
                    where["modality"]=target_modality
                    search_result=vector_store.search_similar(
                        query_embedding=query_vector,
                        limit=limit,
                        where=where
                    )
                    vector_search_result=search_result
                except Exception as e:
                    logger.error(f"搜索记忆失败: {e}")
                    return []
            else:
                try:
                    if query_modality=="text":
                        query_text_vector=self._encoder_text(query)
                        if not query_text_vector:
                            logger.warning("文本编码失败")
                            return []
                        text_vector_store=self.vector_stores["text"]
                        where={}
                        where["memory_type"]="perceptual"
                        if user_id:
                            where["user_id"]=user_id
                        where["modality"]="text"
                        text_search_result=text_vector_store.search_similar(
                            query_embedding=query_text_vector,
                            limit=limit,
                            where=where
                        )
                        vector_search_result=text_search_result
                    else:
                        logger.warning("非文本查询不支持跨模态查询")
                        return []
                except Exception as e:
                    logger.error(f"搜索记忆失败: {e}")
                    return []
        
        # 计算权重并排序
        now_ts=int(datetime.now().timestamp())
        results=[]
        for result in vector_search_result:
            vector_scroe=result.get("similarity",0.0)
            doc=self.document_store.get_memory(result["memory_id"])
            if not doc:
                continue
            age_days = max(0.0, (now_ts - int(doc.timestamp.timestamp())) / 86400.0)
            recency_score=1.0/(1.0+age_days)
            importance=doc.importance if doc.importance else 0.5

            base_relevance = vector_scroe * 0.8 + recency_score * 0.2
            
            # 重要性作为乘法加权因子，范围 [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)
            
            # 最终得分：相似度 * 重要性权重
            combined = base_relevance * importance_weight

            doc.metadata["final_score"]=combined
            doc.metadata["vector_score"]=vector_scroe
            doc.metadata["recency_score"]=recency_score
            results.append((combined,doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in results[:limit]]

    def _encoder_text_clip(self,text:str):
        if self.image_encoder and self.image_processor:
            inputs=self.image_processor(text=text,return_tensors="pt")
            with torch.no_grad():
                feats=self.image_encoder.get_text_features(**inputs)
            return feats[0].detach().cpu().numpy().tolist()
        else:
            logger.warning("文本编码器编码失败")
            return None
    def _encoder_text_clap(self,text:str):
        if self.audio_encoder and self.audio_processor:
            inputs=self.audio_processor(text=text,return_tensors="pt")
            with torch.no_grad():
                feats=self.audio_encoder.get_text_features(**inputs)
            return feats[0].detach().cpu().numpy().tolist()
        else:
            logger.warning("文本编码器编码失败")
            return None
    def _detach_query_modality(self,query:str):
        if query.endswith(".png") or query.endswith(".jpg") or query.endswith(".jpeg") or query.endswith(".webp"):
            return "image"
        elif query.endswith(".mp3") or query.endswith(".wav") or query.endswith(".aac") or query.endswith(".flac"):
            return "audio"
        else:
            return "text"
    def find_memory(self,id:str)->bool:
        return id in self.id_to_memory


    def clear_memory(self):
        try:
            self.document_store.clear_type_memory(MemoryType.PERCEPTUAL)
            logger.info("✅ 成功清空文档存储中的感知记忆")
        except Exception as e:
            logger.error(f"❌ 清空文档存储中的感知记忆失败: {e}")
        for key,value in self.vector_stores.items():
            try:
                value.clear_type_memory(MemoryType.PERCEPTUAL)
                logger.info(f"✅ 成功清空{key}向量存储中的感知记忆")
            except Exception as e:
                logger.error(f"❌ 清空{key}向量存储中的感知记忆失败: {e}")

        self.perceptions.clear()
        self.perceptual_memories.clear()
        self.modality_index.clear()
        self.id_to_memory.clear()




    def get_stats(self) -> dict[str, Any]:
        """获取感知记忆统计信息"""
        # 硬删除模式：所有记忆都是活跃的
        active_memories = self.perceptual_memories
        
        modality_counts = {modality: len(ids) for modality, ids in self.modality_index.items()}
        vs_stats_all = {}
        for mod, store in self.vector_stores.items():
            try:
                vs_stats_all[mod] = store.get_collection_stats()
            except Exception:
                vs_stats_all[mod] = {"store_type": "qdrant"}
        db_stats = self.document_store.get_database_stats()
        
        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 硬删除模式下已遗忘的记忆会被直接删除
            "total_count": len(self.perceptual_memories),  # 总记忆数量
            "perceptions_count": len(self.perceptions),
            "modality_counts": modality_counts,
            "supported_modalities": list(self.supported_modalities),
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "memory_type": "perceptual",
            "vector_stores": vs_stats_all,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }

    def forget(self, strategy:ForgetType, threshold: float = 0.1, max_age_days: int = 30) -> int:
        removed_memories=[]
        forgotten_count=0
        if strategy.value == ForgetType.IMPORTANCE.value:
            for memory in self.perceptual_memories:
                if memory.importance < threshold:
                    removed_memories.append(memory.id)
        
        elif strategy.value == ForgetType.TIME.value:
            now_ts=int(datetime.now().timestamp())
            for memory in self.perceptual_memories:
                age_days = max(0.0, (now_ts - int(memory.timestamp.timestamp())) / 86400.0)
                if age_days > max_age_days:
                    removed_memories.append(memory.id)

        elif strategy.value == ForgetType.CAPACITY.value and len(self.perceptual_memories)>self.config.max_capacity:
            sorted_memories=sorted(self.perceptual_memories,key=lambda x:x.importance)
            removed_count=len(self.perceptual_memories)-self.config.max_capacity
            removed_memories=[memory.id for memory in sorted_memories[:removed_count]]
            
        
        for memory_id in removed_memories:
            if self.remove_memory(memory_id):
                forgotten_count+=1
        if forgotten_count>0:
            logger.info(f"成功遗忘{forgotten_count}条记忆")
        return forgotten_count

    def get_all_memories(self):
        return self.perceptual_memories.copy()

    def get_memory(self,id:str):
        return self.id_to_memory.get(id,None)

    def get_memory_by_user_id(self,user_id:str):
        return [memory for memory in self.perceptual_memories if memory.user_id==user_id]

    def load_from_store(self):
        """从持久化存储加载缓存（程序启动时调用）

        数据流:
          1. 从 document_store (SQLite) 加载所有感知记忆 → 重建 memories, id_to_memory
          2. 从各模态 vector_store 加载向量 → 重建 perceptions, modality_index
          3. 对于 document_store 中有但 vector_store 中缺失的记忆 → 重新编码并写入
        """
        # 清空现有缓存
        self.perceptions.clear()
        self.perceptual_memories.clear()
        self.modality_index.clear()
        self.id_to_memory.clear()

        # 1. 从 document_store 加载所有感知记忆
        try:
            doc_memories = self.document_store.search_memory(
                memory_type=MemoryType.PERCEPTUAL, limit=100000
            )
            logger.info(f"从 document_store 加载到 {len(doc_memories)} 条感知记忆")
        except Exception as e:
            logger.error(f"从 document_store 加载记忆失败: {e}")
            doc_memories = []

        # 2. 从各模态 vector_store 加载向量，建立 memory_id → vector 映射
        vector_map: dict[str, list[float]] = {}  # memory_id → vector
        modality_map: dict[str, str] = {}  # memory_id → modality
        for modality, store in self.vector_stores.items():
            try:
                all_vectors = store.get_all_vectors(with_vector=True)
                for item in all_vectors:
                    mid = item.get("memory_id", "")
                    vector = item.get("vector")
                    if mid and vector is not None:
                        vector_map[mid] = vector if isinstance(vector, list) else list(vector)
                        modality_map[mid] = modality
                logger.info(f"从 {modality} vector_store 加载到 {len(all_vectors)} 条向量")
            except Exception as e:
                logger.error(f"从 {modality} vector_store 加载向量失败: {e}")

        # 3. 重建缓存
        re_encoded_count = 0
        for memory in doc_memories:
            modality = memory.metadata.get("modality", "text")
            raw_data = memory.metadata.get("raw_data", memory.content)
            perception_id = memory.metadata.get("perception_id", f"Perception_{memory.id}")

            # 获取或重新生成向量
            encoding = vector_map.get(memory.id)
            if encoding is None:
                # vector_store 中缺失，需要重新编码并写入
                try:
                    encoding = self._encoder_data(raw_data, modality)
                    if encoding and modality in self.vector_stores:
                        ts_int = int(memory.timestamp.timestamp())
                        self.vector_stores[modality].add_vectors(
                            vectors=[encoding],
                            metadata=[{
                                "memory_id": memory.id,
                                "user_id": memory.user_id,
                                "modality": modality,
                                "timestamp": ts_int,
                                "memory_type": "perceptual",
                                "importance": memory.importance,
                                "context": memory.metadata.get("context", {}),
                                "perception_id": perception_id,
                            }],
                            ids=[memory.id]
                        )
                        re_encoded_count += 1
                except Exception as e:
                    logger.warning(f"重新编码记忆 {memory.id[:8]}... 失败: {e}")
                    encoding = []

            # 重建 Perception
            perception = Perception(
                perception_id=perception_id,
                data=raw_data,
                modality=modality,
                encoding=encoding or [],
                metadata=memory.metadata
            )

            # 写入缓存
            self.perceptions[perception_id] = perception
            self.perceptual_memories.append(memory)
            self.modality_index.setdefault(modality, []).append(perception_id)
            self.id_to_memory[memory.id] = memory

        logger.info(
            f"✅ 缓存加载完成: {len(self.perceptual_memories)} 条记忆, "
            f"{len(self.perceptions)} 个感知, "
            f"重新编码 {re_encoded_count} 条"
        )

    def sync_stores(self) -> dict[str, Any]:
        """检测并修复缓存、document_store、vector_stores 之间的数据不一致

        同步策略:
          - document_store 是数据源头 (持久化)
          - 缓存中有但 document_store 中没有的 → 写入 document_store
          - document_store 中有但缓存中没有的 → 加载到缓存
          - 缓存中有但 vector_store 中没有的 → 重新编码并写入 vector_store
          - vector_store 中有但缓存中没有的 → 视为孤儿数据 (记录但不删除)

        Returns:
            同步结果统计信息
        """
        stats: dict[str, Any] = {
            "cache_count": len(self.perceptual_memories),
            "synced_to_document_store": 0,
            "synced_from_document_store": 0,
            "synced_to_vector_store": 0,
            "orphan_vectors": 0,
            "errors": [],
        }

        # 1. 获取 document_store 中所有感知记忆
        try:
            doc_memories = self.document_store.search_memory(
                memory_type=MemoryType.PERCEPTUAL, limit=100000
            )
            doc_ids = {m.id for m in doc_memories}
            doc_map = {m.id: m for m in doc_memories}
        except Exception as e:
            stats["errors"].append(f"读取 document_store 失败: {e}")
            doc_ids = set()
            doc_map = {}

        # 2. 获取各 vector_store 中所有记忆 ID
        vector_ids_by_modality: dict[str, set[str]] = {}
        for modality, store in self.vector_stores.items():
            try:
                all_vectors = store.get_all_vectors(with_vector=False)
                vector_ids_by_modality[modality] = {
                    item.get("memory_id", "") for item in all_vectors
                }
            except Exception as e:
                stats["errors"].append(f"读取 {modality} vector_store 失败: {e}")
                vector_ids_by_modality[modality] = set()

        all_vector_ids = set()
        for ids in vector_ids_by_modality.values():
            all_vector_ids |= ids

        cache_ids = set(self.id_to_memory.keys())

        # 3. 缓存中有但 document_store 中缺失的 → 写入 document_store
        missing_in_doc = cache_ids - doc_ids
        for memory_id in missing_in_doc:
            memory = self.id_to_memory[memory_id]
            try:
                ts_int = int(memory.timestamp.timestamp())
                self.document_store.add_memory(
                    memory_id=memory.id,
                    user_id=memory.user_id,
                    content=memory.content,
                    memory_type="perceptual",
                    timestamp=ts_int,
                    importance=memory.importance,
                    properties={
                        "modality": memory.metadata.get("modality", "text"),
                        "raw_data": memory.metadata.get("raw_data", memory.content),
                        "context": memory.metadata.get("context", {}),
                        "perception_id": memory.metadata.get("perception_id", ""),
                    }
                )
                stats["synced_to_document_store"] += 1
            except Exception as e:
                stats["errors"].append(f"写入 document_store 失败 {memory_id[:8]}...: {e}")

        # 4. document_store 中有但缓存中缺失的 → 加载到缓存
        missing_in_cache = doc_ids - cache_ids
        for memory_id in missing_in_cache:
            memory = doc_map[memory_id]
            modality = memory.metadata.get("modality", "text")
            raw_data = memory.metadata.get("raw_data", memory.content)
            perception_id = memory.metadata.get("perception_id", f"Perception_{memory_id}")

            # 尝试从 vector_store 获取向量
            encoding: list[float] = []
            if memory_id in all_vector_ids:
                # vector_store 有数据但我们这里没有完整vector，先用空
                pass

            perception = Perception(
                perception_id=perception_id,
                data=raw_data,
                modality=modality,
                encoding=encoding,
                metadata=memory.metadata
            )
            self.perceptions[perception_id] = perception
            self.perceptual_memories.append(memory)
            self.modality_index.setdefault(modality, []).append(perception_id)
            self.id_to_memory[memory_id] = memory
            stats["synced_from_document_store"] += 1

        # 5. 缓存中有但 vector_store 中缺失的 → 重新编码并写入
        for memory_id, memory in self.id_to_memory.items():
            modality = memory.metadata.get("modality", "text")
            modality_vector_ids = vector_ids_by_modality.get(modality, set())

            if memory_id not in modality_vector_ids and modality in self.vector_stores:
                raw_data = memory.metadata.get("raw_data", memory.content)
                try:
                    encoding = self._encoder_data(raw_data, modality)
                    if encoding:
                        ts_int = int(memory.timestamp.timestamp())
                        perception_id = memory.metadata.get("perception_id", f"Perception_{memory_id}")
                        self.vector_stores[modality].add_vectors(
                            vectors=[encoding],
                            metadata=[{
                                "memory_id": memory_id,
                                "user_id": memory.user_id,
                                "modality": modality,
                                "timestamp": ts_int,
                                "memory_type": "perceptual",
                                "importance": memory.importance,
                                "context": memory.metadata.get("context", {}),
                                "perception_id": perception_id,
                            }],
                            ids=[memory_id]
                        )
                        # 更新 perception 的 encoding
                        if perception_id in self.perceptions:
                            self.perceptions[perception_id].encoding = encoding
                        stats["synced_to_vector_store"] += 1
                except Exception as e:
                    stats["errors"].append(f"重新编码失败 {memory_id[:8]}...: {e}")

        # 6. 统计孤儿向量 (vector_store 中有但 cache+doc 都没有的)
        all_known_ids = cache_ids | doc_ids
        orphan_ids = all_vector_ids - all_known_ids
        stats["orphan_vectors"] = len(orphan_ids)

        logger.info(
            f"✅ 同步完成: "
            f"写入doc_store {stats['synced_to_document_store']} 条, "
            f"从doc_store加载 {stats['synced_from_document_store']} 条, "
            f"写入vector_store {stats['synced_to_vector_store']} 条, "
            f"孤儿向量 {stats['orphan_vectors']} 条"
        )
        return stats