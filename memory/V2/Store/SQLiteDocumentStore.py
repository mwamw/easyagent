import sqlite3
from typing import Any,Optional
from .DocumentStore import DocumentStore
from ..BaseMemory import MemoryItem,MemoryType
from datetime import datetime
import json
class SQLiteDocumentStore(DocumentStore):
    def __init__(self,db_path:str):
        self.db_path=db_path
        try:
            self.conn=sqlite3.connect(db_path)
            self.cursor=self.conn.cursor()
        except Exception as e:
            raise Exception(f"Failed to connect to SQLite database: {e}")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                memory_type TEXT,
                timestamp INTEGER,
                importance REAL,
                properties TEXT
            )
        """)
        self.conn.commit()
    def add_memory(self,memory_id:str,user_id:str,content:str,memory_type:str,timestamp:int,importance:float,properties:Optional[dict[str,Any]]=None)->str:
        try:
            self.cursor.execute("""
                INSERT INTO memories (id,user_id,content,memory_type,timestamp,importance,properties)
                VALUES (?,?,?,?,?,?,?)
            """,(memory_id,user_id,content,memory_type,timestamp,importance,json.dumps(properties)))
            self.conn.commit()
        except Exception as e:
            print(f"Failed to add memory: {e}")
        return memory_id

    def remove_memory(self,memory_id:str):
        self.cursor.execute("""
            DELETE FROM memories WHERE id=?
        """,(memory_id,))
        self.conn.commit()
        
    def update_memory(self,memory_id:str,content:str,importance:Optional[float]=None,properties:Optional[dict[str,Any]]=None):
        if content is None and importance is None and properties is None:
            return
        if importance is not None and properties is not None:
            self.cursor.execute("""
                UPDATE memories SET content=?,importance=?,properties=? WHERE id=?
            """,(content,importance,json.dumps(properties),memory_id))
        elif importance is not None:
            self.cursor.execute("""
                UPDATE memories SET content=?,importance=? WHERE id=?
            """,(content,importance,memory_id))
        elif properties is not None:
            self.cursor.execute("""
                UPDATE memories SET content=?,properties=? WHERE id=?
            """,(content,json.dumps(properties),memory_id))
        else:
            self.cursor.execute("""
                UPDATE memories SET content=? WHERE id=?
            """,(content,memory_id))
        self.conn.commit()
    def get_memory(self,memory_id:str)->Optional[MemoryItem]:
        self.cursor.execute("""
            SELECT * FROM memories WHERE id=?
        """,(memory_id,))
        row=self.cursor.fetchone()
        if row is None:
            return None
        return MemoryItem(
            id=row[0],
            user_id=row[1],
            content=row[2],
            type=row[3],
            timestamp=datetime.fromtimestamp(row[4]),
            importance=row[5],
            metadata=json.loads(row[6]) if row[6] and json.loads(row[6]) else {}
        )
    def clear_type_memory(self,memory_type:MemoryType):
        self.cursor.execute("""
            DELETE FROM memories WHERE memory_type=?
        """,(memory_type.value,))
        self.conn.commit()
    def get_database_stats(self)->dict[str,Any]:
        self.cursor.execute("""
            SELECT COUNT(*) FROM memories
        """)
        count=self.cursor.fetchone()[0]

        return {"count":count}
    def search_memory(self,user_id:Optional[str]=None,memory_type:Optional[MemoryType]=None,session_id:Optional[str]=None,start_ts:Optional[float]=None,end_ts:Optional[float]=None,importance_threshold:Optional[float]=None,limit:int=1000)->list[MemoryItem]:
        # 动态构建 WHERE 子句
        conditions = []
        params = []
        
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if memory_type is not None:
            # MemoryType 是枚举，需要转成字符串
            conditions.append("memory_type = ?")
            params.append(memory_type.value if hasattr(memory_type, 'value') else str(memory_type))
        
        if session_id is not None:
            # session_id 存储在 properties JSON 中，需要用 JSON 查询
            conditions.append("json_extract(properties, '$.session_id') = ?")
            params.append(session_id)
        
        if start_ts is not None:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        
        if end_ts is not None:
            conditions.append("timestamp <= ?")
            params.append(end_ts)
        
        if importance_threshold is not None:
            conditions.append("importance >= ?")
            params.append(importance_threshold)
        
        # 构建完整 SQL
        sql = "SELECT * FROM memories"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(sql, tuple(params))
        rows = self.cursor.fetchall()
        
        return [
            MemoryItem(
                id=row[0],
                user_id=row[1],
                content=row[2],
                type=row[3],
                timestamp=datetime.fromtimestamp(row[4]),
                importance=row[5],
                metadata=json.loads(row[6]) if row[6] and json.loads(row[6]) else {}
            )
            for row in rows
        ]