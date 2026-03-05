from BaseMemory import BaseMemory, ForgetType,MemoryConfig,MemoryItem
from datetime import datetime,timedelta
import os
import logging
logger=logging.getLogger(__name__)
from typing import Any,Optional
from Embedding.BaseEmbeddingModel import BaseEmbeddingModel
from Store.VectorStore import *
from Store.DocumentStore import *
from typing_extensions import override
class Episode:

    def __init__(self,episode_id:str,user_id:str,session_id:str,timestamp:datetime,content:str,
    context:dict[str,Any],outcome:Optional[str]=None,importance:float=0.5) -> None:
        self.episode_id=episode_id
        self.user_id=user_id
        self.session_id=session_id
        self.timestamp=timestamp
        self.content=content
        self.context=context
        self.outcome=outcome # 事件结果
        self.importance=importance # 事件重要性
        


class EpisodicMemory(BaseMemory):
    def __init__(self,config:MemoryConfig,document_store:DocumentStore,vector_store:VectorStore,embedding_model:BaseEmbeddingModel,storage_backend=None) -> None:
        super().__init__(config)

        self.episodes:list[Episode]=[]
        self.id_to_episode:dict[str,Episode]={}
        self.sessions:dict[str,list[str]]={}
        self.patterns={}

        self.document_store=document_store
        self.vector_store=vector_store
        self.embedding_model=embedding_model
    def add_memory(self,item:MemoryItem)->str:
        #确定id唯一
        if item.id in self.id_to_episode:
            logger.warning(f"记忆id已存在: {item.id}")
            return ""
        session_id=item.metadata.get("session_id", "default_session")
        episode_id=item.id
        user_id=item.user_id
        timestamp=item.timestamp
        content=item.content
        context=item.metadata.get("context", {})
        outcome=item.metadata.get("outcome")
        participants=item.metadata.get("participants", [])
        tags=item.metadata.get("tags", [])
        importance=item.importance

        # 1. 先写 document store（主存储）
        try:
            self.document_store.add_memory(
                memory_id=episode_id,
                user_id=user_id,
                content=content,
                memory_type="episodic",
                timestamp=int(timestamp.timestamp()),
                importance=importance,
                properties={
                    "session_id": session_id,
                    "context": context,
                    "outcome": outcome,
                    "participants": participants,
                    "tags": tags
                }
            )
        except Exception as e:
            logger.error(f"文档存储添加失败，操作取消: {e}")
            return ""  # 主存储失败，直接返回

        # 2. 写 vector store
        vector_stored = False
        try:
            embedding=self.embedding_model.embed([content])
            self.vector_store.add_vectors(
                vectors=embedding,
                metadata=[{
                    "memory_id": episode_id,
                    "user_id": user_id,
                    "memory_type": "episodic",
                    "importance": importance,
                    "session_id": session_id,
                    "content": content
                }],
                ids=[episode_id]
            )
            vector_stored = True
        except Exception as e:
            logger.warning(f"向量存储添加失败（可通过 sync_stores 修复）: {e}")

        # 3. 最后写缓存（持久化成功后）
        episode=Episode(
            episode_id=episode_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            content=content,
            context=context,
            outcome=outcome,
            importance=importance
        )
        self.episodes.append(episode)
        self.id_to_episode[episode_id]=episode
        self.sessions.setdefault(session_id, []).append(episode_id)

        if not vector_stored:
            logger.info(f"记忆 {episode_id[:8]}... 已添加到缓存和文档存储，向量存储待同步")

        return episode_id

            
    @override
    def remove_memory(self, memory_id: str) -> bool:
        
        #处理cache
        removed_cache=False
        # for episode in self.episodes:
        #     if episode.episode_id==memory_id:
        #         remove_episode=episode
        #         self.episodes.remove(episode)
        #         if remove_episode.session_id in self.sessions:
        #             self.sessions[remove_episode.session_id].remove(remove_episode.episode_id)
        #         removed_cache=True
        #         break
        if memory_id in self.id_to_episode:
            episode = self.id_to_episode[memory_id]
            try:
                self.episodes.remove(episode)
            except ValueError:
                logger.warning(f"episode {memory_id} 不在 episodes 列表中")
            
            del self.id_to_episode[memory_id]
    
            try:
                self.sessions.get(episode.session_id, []).remove(episode.episode_id)
            except ValueError:
                logger.warning(f"session {episode.session_id} 不在 sessions 列表中")
            removed_cache = True

        #处理 document store

        removed_document_store=False
        try:
            self.document_store.remove_memory(memory_id)
            removed_document_store=True
        except Exception as e:
            logger.warning(f"文档存储删除失败: {e}")

        #处理 vector store

        removed_vector_store=False
        try:
            self.vector_store.remove_vectors([memory_id])
            removed_vector_store=True
        except Exception as e:
            logger.warning(f"向量存储删除失败: {e}")

        return removed_cache or removed_document_store or removed_vector_store


    @override
    def update_memory(self,id:str,content:str,importance:Optional[float]=None,metadata:Optional[dict[str,Any]]=None) -> bool:
        #处理cache
        updated_cache=False
        # for episode in self.episodes:
        #     if episode.episode_id==id:
        #         if content:
        #             episode.content=content
        #         if importance:
        #             episode.importance=importance
        #         if metadata:
        #             episode.context.update(metadata.get("context", {}))
        #             if "outcome" in metadata:
        #                 episode.outcome=metadata["outcome"]
        #         updated_cache=True
        #         break
        if id in self.id_to_episode:
            episode = self.id_to_episode[id]
            if content:
                episode.content = content
            if importance is not None:
                episode.importance = importance
            if metadata:
                episode.context.update(metadata.get("context", {}))
                if "outcome" in metadata:
                    episode.outcome = metadata["outcome"]
            updated_cache = True
        #处理 document store

        updated_document_store=False
        try:
            self.document_store.update_memory(id,content,importance,metadata)
            updated_document_store=True
        except Exception as e:
            logger.warning(f"文档存储更新失败: {e}")

        #处理 vector store

        updated_vector_store=False
        if content:
            try:
                self.vector_store.remove_vectors([id])
                new_embedding=self.embedding_model.embed([content])
                doc = self.document_store.get_memory(id)
                payload = {
                    "memory_id":id,
                    "user_id": doc.user_id if doc else "",
                    "memory_type": "episodic",
                    "importance": (doc.importance if doc else importance) or 0.5,
                    "session_id": (doc.metadata  if doc else {}).get("session_id"),
                    "content": content
                }
                self.vector_store.add_vectors(
                    vectors=new_embedding,
                    metadata=[payload],
                    ids=[id]
                )
                updated_vector_store=True
            except Exception as e:
                logger.warning(f"向量存储更新失败: {e}")
        return updated_cache or updated_document_store or updated_vector_store

    @override
    def find_memory(self, id: str) -> bool:
        return id in self.id_to_episode

    @override
    def clear_memory(self):
        #清空cache
        self.episodes.clear()
        self.sessions.clear()
        self.id_to_episode.clear()

        #处理document store
        try:
            self.document_store.clear_type_memory(MemoryType.EPISODIC)
        except Exception as e:
            logger.warning(f"清空文档存储失败: {e}")

        #处理vector store
        try:
            self.vector_store.clear_type_memory(MemoryType.EPISODIC)
        except Exception as e:
            logger.warning(f"清空向量存储失败: {e}")

    def get_stats(self) -> dict[str, Any]:
        """获取情景记忆统计信息（合并SQLite与Qdrant）"""
        # 硬删除模式：所有episodes都是活跃的
        active_episodes = self.episodes
        
        db_stats = self.document_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception as e:
            logger.warning(f"获取向量存储统计失败: {e}")
            vs_stats = {"store_type": "qdrant"}
        return {
            "count": len(active_episodes),  # 活跃记忆数量
            "forgotten_count": 0,  # 硬删除模式下已遗忘的记忆会被直接删除
            "total_count": len(self.episodes),  # 总记忆数量
            "sessions_count": len(self.sessions),
            "avg_importance": sum(e.importance for e in active_episodes) / len(active_episodes) if active_episodes else 0.0,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }

    def _calculate_time_span(self) -> int:
        """计算记忆时间跨度（天数）"""
        if not self.episodes:
            return 0
        return int((max(e.timestamp for e in self.episodes) - min(e.timestamp for e in self.episodes)).days)


    def search_memory(self, query: str, limit: int = 5, user_id: Optional[str] = None, **kwargs):
        user_id=user_id
        session_id=kwargs.get("session_id")
        time_range:Optional[tuple[datetime,datetime]]=kwargs.get("time_range")
        importance_threshold:Optional[float]=kwargs.get("importance_threshold")
        
        candidate_ids:Optional[list[str]]=None
        
        if time_range is not None or importance_threshold is not None:
            start_ts=time_range[0].timestamp() if time_range else None
            end_ts=time_range[1].timestamp() if time_range else None
            docs=self.document_store.search_memory(
                user_id=user_id,
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
                start_ts=start_ts,
                end_ts=end_ts,
                importance_threshold=importance_threshold,
                limit=1000
            )
            
            candidate_ids=[doc.id for doc in docs]

        #向量检索
        try:
            query_embedding=self.embedding_model.embed([query])[0]

            where={"memory_type":"episodic"}
            if user_id:
                where["user_id"]=user_id
            hits_memory=self.vector_store.search_similar(
                query_embedding=query_embedding,
                where=where,
                limit=max(limit*5,20)
            )
            
        except Exception as e:
            logger.warning(f"向量搜索失败，回退到关键词搜索: {e}")
            hits_memory=[]

        now_time=int(datetime.now().timestamp())
        result:list[tuple[float,MemoryItem]]=[]
        visited:set[str]=set()
        for hit in hits_memory:
            mem_id=hit.get("memory_id")
            mem_metadata=hit.get(f"metadata",{})
            if not mem_id or mem_id in visited:
                continue

            episode=self.id_to_episode.get(mem_id)
            if episode and episode.context.get("forgotten",False):
                continue

            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            
            if session_id and mem_metadata.get("session_id")!=session_id:
                continue
                
            doc=self.document_store.get_memory(mem_id)
            if not doc:
                continue

            #计算综合分数向量0.6+近因效应0.2+重要性0.2

            similarity_score=hit.get("similarity",0.0)
            age_days = max(0.0, (now_time - int(doc.timestamp.timestamp())) / 86400.0)
            recency_score = 1.0/(1.0+age_days)
            importance_score=doc.importance if doc.importance else 0.5

            base_relevance_score=similarity_score*0.8+recency_score*0.2

            importance_weight=importance_score*0.8+0.4

            final_score=base_relevance_score*importance_weight

            doc.metadata={**doc.metadata,
            "similarity_score":similarity_score,
            "recency_score":recency_score,
            "importance_score":importance_score,
            "base_relevance_score":base_relevance_score,
            "importance_weight":importance_weight,
            "final_score":final_score
            }
            result.append((final_score,doc))

            visited.add(mem_id)

        if not result:
            #向量检索失败采用普通关键字检索
            query=query.lower()
            filtered_memory=self._filter_memory(user_id=user_id,session_id=session_id,time_range=time_range)
            for mem in filtered_memory:
                content=mem.content.lower()
                memory_keywords=set(content.split())
                query_keywords=set(query.split())
                union_len = len(query_keywords.union(memory_keywords))
                keyword_similarity = len(query_keywords.intersection(memory_keywords)) / union_len if union_len > 0 else 0.0
                recency_score = 1.0 / (1.0 + max(0.0, (now_time - int(mem.timestamp.timestamp())) / 86400.0))
                
                base_relevance_score=keyword_similarity*0.8+recency_score*0.2

                simportance_weight=mem.importance*0.8+0.4

                final_score=base_relevance_score*simportance_weight
                    
                mem.metadata={
                **mem.metadata,
                "similarity_score":keyword_similarity,
                "recency_score":recency_score,
                "importance_score":mem.importance,
                "base_relevance_score":base_relevance_score,
                "importance_weight":simportance_weight,
                "final_score":final_score
                }
                result.append((final_score,mem))
                    
        result.sort(key=lambda x:x[0],reverse=True)
        return [mem for _,mem in result[:limit]]
                
            
            
    def _filter_memory(self,user_id:Optional[str]=None,session_id:Optional[str]=None,time_range:Optional[tuple[datetime,datetime]]=None)->list[MemoryItem]:
        where:dict[str,Any]={"memory_type":"episodic"}
        if user_id:
            where["user_id"]=user_id
        if session_id:
            where["session_id"]=session_id
        if time_range:
            where["start_ts"]=time_range[0].timestamp()
            where["end_ts"]=time_range[1].timestamp()

        results:list[MemoryItem]=[]
        for memory in self.episodes:
            if where.get("user_id") and where["user_id"]!=memory.user_id:
                continue
            if where.get("session_id") and where["session_id"]!=memory.session_id:
                continue
            if where.get("start_ts") and where["start_ts"]>memory.timestamp.timestamp():
                continue
            if where.get("end_ts") and where["end_ts"]<memory.timestamp.timestamp():
                continue
            results.append(self._transform_memory(memory))
        return results

    def _transform_memory(self,episode:Episode)->MemoryItem:
        memory_item = MemoryItem(
            id=episode.episode_id,
            content=episode.content,
            type="episodic",
            user_id=episode.user_id,
            timestamp=episode.timestamp,
            importance=episode.importance,
            metadata={
                "session_id":episode.session_id,
                "outcome":episode.outcome,
                "context":episode.context
            }
        )
        return memory_item
    
    def forget(self, strategy:ForgetType, threshold: float = 0.1, max_age_days: int = 30) -> int:
        """情景记忆遗忘机制（硬删除）"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove: list[str] = []  # 收集要删除的记忆ID
        
        # CAPACITY 策略：预先排序一次，避免循环内重复排序
        if strategy.value == ForgetType.CAPACITY.value and len(self.episodes) > self.config.max_capacity:
            sorted_episodes = sorted(self.episodes, key=lambda e: e.importance)
            excess_count = len(self.episodes) - self.config.max_capacity
            to_remove_set = {e.episode_id for e in sorted_episodes[:excess_count]}
            to_remove = list(to_remove_set)
        else:
            for episode in self.episodes:
                should_forget = False
                
                if strategy.value == ForgetType.IMPORTANCE.value:
                    # 基于重要性遗忘
                    if episode.importance < threshold:
                        should_forget = True
                elif strategy.value == ForgetType.TIME.value:
                    # 基于时间遗忘
                    cutoff_time = current_time - timedelta(days=max_age_days)
                    if episode.timestamp < cutoff_time:
                        should_forget = True
                
                if should_forget:
                    to_remove.append(episode.episode_id)
        
        # 执行硬删除
        for episode_id in to_remove:
            if self.remove_memory(episode_id):
                forgotten_count += 1
                logger.info(f"情景记忆硬删除: {episode_id[:8]}... (策略: {strategy})")
        
        return forgotten_count
    def get_all_memories(self)->list[MemoryItem]:
        return [self._transform_memory(episode) for episode in self.episodes]

    def get_session_episodes(self,session_id:str)->list[Episode]:
        return [episode for episode in self.episodes if episode.session_id==session_id]

    def get_timeline(self,user_id:Optional[str]=None,limit:int=50):
        """获取时间线（最近的记忆）"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        
        timeline = []
        for episode in episodes[:limit]:
            timeline.append({
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp.isoformat(),
                "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                "session_id": episode.session_id,
                "importance": episode.importance,
                "outcome": episode.outcome
            })
        
        return timeline   

    def load_from_store(self):
        self.sessions.clear()
        self.episodes.clear()
        self.id_to_episode.clear()
        all_memories=self.document_store.search_memory(memory_type=MemoryType.EPISODIC)
        for memory in all_memories:
            episode_id=memory.id
            user_id=memory.user_id
            session_id=memory.metadata.get("session_id","")
            timestamp=memory.timestamp
            content=memory.content
            context=memory.metadata.get("context",{})
            outcome=memory.metadata.get("outcome","")
            importance=memory.importance
            episode=Episode(
                episode_id=episode_id,
                user_id=user_id,
                session_id=session_id,
                timestamp=timestamp,
                content=content,
                context=context,
                outcome=outcome,
                importance=importance
            )
            self.episodes.append(episode)
            self.sessions.setdefault(session_id, []).append(episode_id)
            self.id_to_episode[episode_id] = episode

    def sync_stores(self) -> dict[str, Any]:
        """检测并修复缓存、文档存储与向量存储之间的数据不一致

        Returns:
            同步结果统计信息
        """
        stats = {
            "cache_count": len(self.episodes),
            "synced_to_vector": 0,
            "removed_orphan_vectors": 0,
            "errors": []
        }

        # 1. 从文档存储获取所有记忆ID
        try:
            doc_memories = self.document_store.search_memory(memory_type=MemoryType.EPISODIC, limit=10000)
            doc_ids = {m.id for m in doc_memories}
        except Exception as e:
            stats["errors"].append(f"无法读取文档存储: {e}")
            return stats

        # 2. 获取向量存储中的所有ID (如果支持)
        vector_ids: set[str] = set()
        try:
            # 尝试获取向量存储中的ID（依赖于具体实现）
            vs_stats = self.vector_store.get_collection_stats()
            if "ids" in vs_stats:
                vector_ids = set(vs_stats["ids"])
        except Exception:
            pass  # 向量存储可能不支持列出所有ID

        # 3. 同步缓存与文档存储
        cache_ids = set(self.id_to_episode.keys())
        
        # 缓存中缺失的记录（从文档存储加载）
        missing_in_cache = doc_ids - cache_ids
        for doc in doc_memories:
            if doc.id in missing_in_cache:
                episode = Episode(
                    episode_id=doc.id,
                    user_id=doc.user_id,
                    session_id=doc.metadata.get("session_id", ""),
                    timestamp=doc.timestamp,
                    content=doc.content,
                    context=doc.metadata.get("context", {}),
                    outcome=doc.metadata.get("outcome"),
                    importance=doc.importance
                )
                self.episodes.append(episode)
                self.id_to_episode[doc.id] = episode
                self.sessions.setdefault(episode.session_id, []).append(doc.id)

        # 4. 将缺失的向量补充到向量存储
        for doc in doc_memories:
            if vector_ids and doc.id in vector_ids:
                continue  # 向量已存在
            
            try:
                embedding = self.embedding_model.embed([doc.content])
                self.vector_store.add_vectors(
                    vectors=embedding,
                    metadata=[{
                        "memory_id": doc.id,
                        "user_id": doc.user_id,
                        "memory_type": "episodic",
                        "importance": doc.importance,
                        "session_id": doc.metadata.get("session_id", ""),
                        "content": doc.content
                    }],
                    ids=[doc.id]
                )
                stats["synced_to_vector"] += 1
            except Exception as e:
                stats["errors"].append(f"同步向量失败 {doc.id[:8]}...: {e}")

        logger.info(f"同步完成: 向量存储新增 {stats['synced_to_vector']} 条")
        return stats
            
    def find_patterns(
        self, 
        user_id: Optional[str] = None, 
        min_frequency: int = 2,
        use_tfidf: bool = True,
        use_semantic_clustering: bool = True,
        n_clusters: int = 5
    ) -> list[dict[str, Any]]:
        """发现用户行为模式（增强版）
        
        使用三种方式发现模式：
        1. jieba 中文分词 + 词频统计
        2. TF-IDF 关键词提取
        3. 向量嵌入语义聚类
        
        Args:
            user_id: 可选，只分析特定用户
            min_frequency: 最小出现频率阈值
            use_tfidf: 是否使用 TF-IDF 提取关键词
            use_semantic_clustering: 是否使用向量聚类发现语义模式
            n_clusters: 聚类数量
            
        Returns:
            模式列表，每个模式包含 type, pattern, frequency, confidence, score
        """
        # 过滤情景
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        
        if not episodes:
            return []
        
        patterns: list[dict[str, Any]] = []
        
        # ========== 1. jieba 分词 + 词频统计 ==========
        try:
            import jieba
            import jieba.analyse
            
            keyword_freq: dict[str, int] = {}
            
            for episode in episodes:
                # 使用 jieba 分词（自动处理中英文）
                words = jieba.lcut(episode.content)
                for word in words:
                    # 过滤：长度 > 1，非纯数字，非纯标点
                    if len(word) > 1 and not word.isdigit() and word.isalnum():
                        keyword_freq[word] = keyword_freq.get(word, 0) + 1
            
            # 筛选频繁关键词
            for keyword, freq in keyword_freq.items():
                if freq >= min_frequency:
                    patterns.append({
                        "type": "keyword",
                        "pattern": keyword,
                        "frequency": freq,
                        "confidence": freq / len(episodes),
                        "score": freq / len(episodes),  # 简单置信度作为分数
                        "method": "jieba_wordfreq"
                    })
                    
        except ImportError:
            logger.warning("jieba 未安装，跳过中文分词。使用 pip install jieba 安装。")
            # 回退到简单分词
            keyword_freq = {}
            for episode in episodes:
                words = episode.content.lower().split()
                for word in words:
                    if len(word) > 2:
                        keyword_freq[word] = keyword_freq.get(word, 0) + 1
            
            for keyword, freq in keyword_freq.items():
                if freq >= min_frequency:
                    patterns.append({
                        "type": "keyword",
                        "pattern": keyword,
                        "frequency": freq,
                        "confidence": freq / len(episodes),
                        "score": freq / len(episodes),
                        "method": "simple_split"
                    })
        
        # ========== 2. TF-IDF 关键词提取 ==========
        if use_tfidf:
            try:
                import jieba.analyse
                
                # 合并所有文档计算 TF-IDF
                all_content = " ".join([e.content for e in episodes])
                
                # 提取 Top-K 关键词及其权重
                tfidf_keywords = jieba.analyse.extract_tags(
                    all_content, 
                    topK=20, 
                    withWeight=True
                )
                
                for keyword, weight in tfidf_keywords:
                    # 统计该关键词在多少个 episode 中出现
                    doc_freq = sum(1 for e in episodes if keyword in e.content)
                    if doc_freq >= min_frequency:
                        patterns.append({
                            "type": "tfidf_keyword",
                            "pattern": keyword,
                            "frequency": doc_freq,
                            "confidence": doc_freq / len(episodes),
                            "score": weight,  # TF-IDF 权重作为分数
                            "tfidf_weight": weight,
                            "method": "tfidf"
                        })
                        
            except ImportError:
                logger.warning("jieba.analyse 不可用，跳过 TF-IDF。")
            except Exception as e:
                logger.warning(f"TF-IDF 分析失败: {e}")
        
        # ========== 3. 上下文模式统计 ==========
        context_freq: dict[str, int] = {}
        for episode in episodes:
            for key, value in episode.context.items():
                if value is not None and str(value).strip():
                    pattern_key = f"{key}:{value}"
                    context_freq[pattern_key] = context_freq.get(pattern_key, 0) + 1
        
        for pattern, freq in context_freq.items():
            if freq >= min_frequency:
                patterns.append({
                    "type": "context",
                    "pattern": pattern,
                    "frequency": freq,
                    "confidence": freq / len(episodes),
                    "score": freq / len(episodes),
                    "method": "context_stats"
                })
        
        # ========== 4. 向量嵌入语义聚类 ==========
        if use_semantic_clustering and len(episodes) >= n_clusters:
            try:
                from sklearn.cluster import KMeans
                import numpy as np
                
                # 获取所有 episode 的向量嵌入
                contents = [e.content for e in episodes]
                embeddings = self.embedding_model.embed(contents)
                
                if hasattr(embeddings, 'tolist'):
                    embeddings = np.array(embeddings)
                else:
                    embeddings = np.array(embeddings)
                
                # K-Means 聚类
                actual_n_clusters = min(n_clusters, len(episodes))
                kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # 分析每个聚类
                for cluster_id in range(actual_n_clusters):
                    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_episodes = [episodes[i] for i in cluster_indices]
                    
                    if len(cluster_episodes) >= min_frequency:
                        # 找到聚类中心最近的 episode 作为代表
                        center = kmeans.cluster_centers_[cluster_id]
                        distances = [np.linalg.norm(embeddings[i] - center) for i in cluster_indices]
                        representative_idx = cluster_indices[np.argmin(distances)]
                        representative_content = episodes[representative_idx].content
                        
                        # 提取聚类的主题关键词
                        try:
                            import jieba.analyse
                            cluster_text = " ".join([e.content for e in cluster_episodes])
                            topic_keywords = jieba.analyse.extract_tags(cluster_text, topK=3)
                            topic_label = ", ".join([str(kw) for kw in topic_keywords]) if topic_keywords else f"Cluster_{cluster_id}"
                        except:
                            topic_label = f"Cluster_{cluster_id}"
                        
                        patterns.append({
                            "type": "semantic_cluster",
                            "pattern": topic_label,
                            "frequency": len(cluster_episodes),
                            "confidence": len(cluster_episodes) / len(episodes),
                            "score": len(cluster_episodes) / len(episodes),
                            "cluster_id": cluster_id,
                            "representative": representative_content[:100] + "..." if len(representative_content) > 100 else representative_content,
                            "episode_ids": [e.episode_id for e in cluster_episodes],
                            "method": "semantic_clustering"
                        })
                        
            except ImportError:
                logger.warning("sklearn 未安装，跳过语义聚类。使用 pip install scikit-learn 安装。")
            except Exception as e:
                logger.warning(f"语义聚类失败: {e}")
        
        # ========== 5. 去重与排序 ==========
        # 按 score 降序排序
        patterns.sort(key=lambda x: x["score"], reverse=True)
        
        # 去除重复的 pattern（保留 score 最高的）
        seen_patterns: set[str] = set()
        unique_patterns: list[dict[str, Any]] = []
        for p in patterns:
            if p["pattern"] not in seen_patterns:
                seen_patterns.add(p["pattern"])
                unique_patterns.append(p)
        
        self.patterns[f"{user_id}_{min_frequency}"]=unique_patterns
        return unique_patterns

    @override
    def add_memories_batch(self, items: list[MemoryItem]) -> list[str]:
        """优化的批量添加记忆
        
        性能优化：
        1. 批量 embed 所有内容（减少 API 调用）
        2. 批量写入向量存储
        3. 逐条写入文档存储（SQLite 事务）
        
        Args:
            items: 记忆对象列表
            
        Returns:
            成功添加的记忆ID列表
        """
        if not items:
            return []
        
        # 过滤已存在的ID
        new_items = [item for item in items if item.id not in self.id_to_episode]
        if not new_items:
            logger.warning("所有记忆ID已存在，跳过批量添加")
            return []
        
        added_ids: list[str] = []
        batch_size = self.config.batch_size
        
        # 分批处理
        for i in range(0, len(new_items), batch_size):
            batch = new_items[i:i + batch_size]
            batch_ids = self._process_batch(batch)
            added_ids.extend(batch_ids)
        
        logger.info(f"批量添加完成: {len(added_ids)}/{len(items)} 条记忆")
        return added_ids

    def _process_batch(self, batch: list[MemoryItem]) -> list[str]:
        """处理单个批次的记忆添加"""
        added_ids: list[str] = []
        
        # 1. 批量 embed
        contents = [item.content for item in batch]
        try:
            embeddings = self.embedding_model.embed(contents)
        except Exception as e:
            logger.error(f"批量 embed 失败: {e}")
            # 回退到逐条添加
            for item in batch:
                result = self.add_memory(item)
                if result:
                    added_ids.append(result)
            return added_ids
        
        # 2. 准备数据
        episodes: list[Episode] = []
        doc_data: list[dict] = []
        vector_data: list[dict] = []
        
        for idx, item in enumerate(batch):
            session_id = item.metadata.get("session_id", "default_session")
            
            episode = Episode(
                episode_id=item.id,
                user_id=item.user_id,
                session_id=session_id,
                timestamp=item.timestamp,
                content=item.content,
                context=item.metadata.get("context", {}),
                outcome=item.metadata.get("outcome"),
                importance=item.importance
            )
            episodes.append(episode)
            
            doc_data.append({
                "memory_id": item.id,
                "user_id": item.user_id,
                "content": item.content,
                "memory_type": "episodic",
                "timestamp": int(item.timestamp.timestamp()),
                "importance": item.importance,
                "properties": {
                    "session_id": session_id,
                    "context": item.metadata.get("context", {}),
                    "outcome": item.metadata.get("outcome"),
                    "participants": item.metadata.get("participants", []),
                    "tags": item.metadata.get("tags", [])
                }
            })
            
            vector_data.append({
                "embedding": embeddings[idx],
                "metadata": {
                    "memory_id": item.id,
                    "user_id": item.user_id,
                    "memory_type": "episodic",
                    "importance": item.importance,
                    "session_id": session_id,
                    "content": item.content
                },
                "id": item.id
            })
        
        # 3. 写入文档存储
        for doc in doc_data:
            try:
                self.document_store.add_memory(**doc)
                added_ids.append(doc["memory_id"])
            except Exception as e:
                logger.error(f"文档存储添加失败 {doc['memory_id'][:8]}...: {e}")
        
        # 4. 批量写入向量存储
        if added_ids:
            try:
                # 只写入成功添加到文档存储的记录
                valid_vectors = [v for v in vector_data if v["id"] in added_ids]
                if valid_vectors:
                    self.vector_store.add_vectors(
                        vectors=[v["embedding"] for v in valid_vectors],
                        metadata=[v["metadata"] for v in valid_vectors],
                        ids=[v["id"] for v in valid_vectors]
                    )
            except Exception as e:
                logger.warning(f"批量向量存储失败（可通过 sync_stores 修复）: {e}")
        
        # 5. 写入缓存
        for idx, episode in enumerate(episodes):
            if episode.episode_id in added_ids:
                self.episodes.append(episode)
                self.id_to_episode[episode.episode_id] = episode
                self.sessions.setdefault(episode.session_id, []).append(episode.episode_id)
        
        return added_ids

    # ==================== 异步方法（优化实现）====================

    async def add_memory_async(self, item: MemoryItem) -> str:
        """异步添加单条记忆
        
        使用 asyncio.to_thread 将同步操作放入线程池执行
        """
        import asyncio
        return await asyncio.to_thread(self.add_memory, item)

    async def add_memories_batch_async(self, items: list[MemoryItem]) -> list[str]:
        """异步批量添加记忆
        
        使用 asyncio.to_thread 将同步批量操作放入线程池执行
        """
        import asyncio
        return await asyncio.to_thread(self.add_memories_batch, items)

    async def search_memory_async(self, query: str, limit: int = 5, user_id: Optional[str] = None, **kwargs) -> list[MemoryItem]:
        """异步搜索记忆
        
        使用 asyncio.to_thread 将同步搜索操作放入线程池执行
        """
        import asyncio
        return await asyncio.to_thread(self.search_memory, query, limit, user_id, **kwargs)

    async def sync_stores_async(self) -> dict[str, Any]:
        """异步同步存储"""
        import asyncio
        return await asyncio.to_thread(self.sync_stores)


if __name__=="__main__":
    pass