"""
感知记忆 (PerceptualMemory) 完整测试程序
覆盖:
  - 初始化与配置
  - 添加记忆 (文本/图像/批量/不支持模态)
  - 获取记忆 (get_memory / get_all / find / get_by_user_id)
  - 更新记忆 (内容/重要性/元数据)
  - 删除记忆 (remove_memory)
  - 搜索记忆 (文本搜索 / 跨模态搜索)
  - 遗忘策略 (重要性/时间/容量)
  - 编码器 (文本/图像/CLIP文本编码/模态检测)
  - 统计信息 (get_stats)
  - 清空记忆 (clear_memory)
  - Perception 类哈希计算
"""
import sys
import os
import uuid
import traceback
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))

from memory.V2.PerceptualMemory import PerceptualMemory, Perception
from memory.V2.Store.SQLiteDocumentStore import SQLiteDocumentStore
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
from memory.V2.BaseMemory import MemoryConfig, MemoryItem, ForgetType, MemoryType
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ==================== 颜色工具 ====================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def print_pass(msg):
    print(f"  {Colors.GREEN}✅ PASS{Colors.END} - {msg}")

def print_fail(msg, detail=""):
    print(f"  {Colors.RED}❌ FAIL{Colors.END} - {msg}")
    if detail:
        print(f"         {Colors.RED}{detail}{Colors.END}")

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}")


# ==================== 辅助函数 ====================
def make_text_memory(content, importance=0.5, user_id="test_user", memory_id=None):
    """快速创建文本记忆"""
    return MemoryItem(
        id=memory_id or str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        type="perceptual",
        timestamp=datetime.now(),
        importance=importance,
        metadata={"modality": "text"}
    )

def make_image_memory(content, image_path, importance=0.5, user_id="test_user", memory_id=None):
    """快速创建图像记忆"""
    return MemoryItem(
        id=memory_id or str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        type="perceptual",
        timestamp=datetime.now(),
        importance=importance,
        metadata={"modality": "image", "raw_data": image_path}
    )


# ==================== 测试类 ====================
class TestPerceptualMemory:
    def __init__(self):
        print_section("初始化测试环境")
        self.passed = 0
        self.failed = 0
        self.errors = []

        # 使用内存模式的 Qdrant + 临时 SQLite，确保每次测试干净
        self.memory_config = MemoryConfig(max_capacity=5)
        self.document_store = SQLiteDocumentStore(db_path="/home/wxd/LLM/EasyAgent/test/db/test_comprehensive.db")
        self.vector_stores = {
            "text": QdrantVectorStore(way="memory", collection_name="test_text_comp", vector_size=384),
            "image": QdrantVectorStore(way="memory", collection_name="test_image_comp", vector_size=512),
        }
        self.embedding_model = HuggingfaceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.pm = PerceptualMemory(
            self.memory_config,
            self.document_store,
            self.vector_stores,
            self.embedding_model,
            supported_modalities=["text", "image"],
        )
        print_pass("测试环境初始化完成")

    def _reset(self):
        """重置记忆到干净状态"""
        self.pm.clear_memory()

    def _assert(self, condition, msg, detail=""):
        if condition:
            print_pass(msg)
            self.passed += 1
        else:
            print_fail(msg, detail)
            self.failed += 1
            self.errors.append(msg)

    # ==================== 1. Perception 类测试 ====================
    def test_perception_class(self):
        print_section("1. Perception 类测试")

        # 1.1 字符串数据哈希
        p1 = Perception(perception_id="p1", data="hello", modality="text")
        p2 = Perception(perception_id="p2", data="hello", modality="text")
        p3 = Perception(perception_id="p3", data="world", modality="text")
        self._assert(p1.data_hash == p2.data_hash, "相同字符串数据哈希一致")
        self._assert(p1.data_hash != p3.data_hash, "不同字符串数据哈希不同")

        # 1.2 bytes 数据哈希
        p4 = Perception(perception_id="p4", data=b"binary_data", modality="image")
        self._assert(len(p4.data_hash) == 32, "bytes数据哈希长度正确(MD5=32)")

        # 1.3 其他类型数据哈希
        p5 = Perception(perception_id="p5", data=12345, modality="structured")
        self._assert(len(p5.data_hash) == 32, "其他类型数据(int)哈希长度正确")

        # 1.4 默认属性
        self._assert(p1.encoding == [], "默认encoding为空列表")
        self._assert(p1.metadata == {}, "默认metadata为空字典")
        self._assert(isinstance(p1.timestamp, datetime), "timestamp类型正确")

    # ==================== 2. 添加记忆测试 ====================
    def test_add_text_memory(self):
        print_section("2. 添加文本记忆测试")
        self._reset()

        m1 = make_text_memory("今天天气真好，阳光明媚。", importance=0.8)
        result_id = self.pm.add_memory(m1)

        self._assert(result_id == m1.id, "add_memory 返回正确的记忆ID")
        self._assert(len(self.pm.get_all_memories()) == 1, "记忆数量为1")
        self._assert(self.pm.find_memory(m1.id), "find_memory 能找到已添加的记忆")

        # 验证缓存
        perception_id = f"Perception_{m1.id}"
        self._assert(perception_id in self.pm.perceptions, "perception缓存正确")
        self._assert("text" in self.pm.modality_index, "模态索引包含text")
        self._assert(perception_id in self.pm.modality_index["text"], "模态索引中包含该perception_id")

        # 验证 document store
        doc = self.document_store.get_memory(m1.id)
        self._assert(doc is not None, "document_store 中能查到记忆")
        self._assert(doc.content == m1.content, "document_store 内容一致")

    def test_add_image_memory(self):
        print_section("3. 添加图像记忆测试")
        self._reset()

        m = make_image_memory("这是一张小狗图片", "./images/test/dog.jpg", importance=0.6)
        result_id = self.pm.add_memory(m)

        self._assert(result_id == m.id, "图像记忆添加成功返回正确ID")
        self._assert(len(self.pm.get_all_memories()) == 1, "记忆数量为1")
        self._assert("image" in self.pm.modality_index, "模态索引包含image")

        perception = self.pm.perceptions.get(f"Perception_{m.id}")
        self._assert(perception is not None, "perception缓存中有该图像记忆")
        self._assert(len(perception.encoding) == 512, f"CLIP图像编码维度=512 (实际={len(perception.encoding)})")

    def test_add_unsupported_modality(self):
        print_section("4. 添加不支持的模态测试")
        self._reset()

        m = MemoryItem(
            id=str(uuid.uuid4()), user_id="test_user",
            content="一段音频", type="perceptual",
            timestamp=datetime.now(), importance=0.5,
            metadata={"modality": "audio"}
        )
        result_id = self.pm.add_memory(m)
        self._assert(result_id == "", "不支持的模态(audio)返回空字符串")
        self._assert(len(self.pm.get_all_memories()) == 0, "不支持的模态不会添加到记忆中")

    def test_add_memories_batch(self):
        print_section("5. 批量添加记忆测试")
        self._reset()

        items = [
            make_text_memory("批量文本1", importance=0.3),
            make_text_memory("批量文本2", importance=0.5),
            make_text_memory("批量文本3", importance=0.7),
        ]
        result_ids = self.pm.add_memories_batch(items)

        self._assert(len(result_ids) == 3, f"批量添加3条记忆，返回{len(result_ids)}个ID")
        self._assert(len(self.pm.get_all_memories()) == 3, "记忆总数为3")

    # ==================== 3. 获取记忆测试 ====================
    def test_get_memory(self):
        print_section("6. 获取记忆测试")
        self._reset()

        m = make_text_memory("用于获取测试的记忆", importance=0.6)
        self.pm.add_memory(m)

        # get_memory
        fetched = self.pm.get_memory(m.id)
        self._assert(fetched is not None, "get_memory 正常返回")
        self._assert(fetched.content == m.content, "get_memory 内容一致")
        self._assert(fetched.importance == 0.6, "get_memory 重要性一致")

        # get_memory 不存在的 ID
        not_found = self.pm.get_memory("nonexistent_id")
        self._assert(not_found is None, "get_memory 不存在的ID返回None")

    def test_get_all_memories(self):
        print_section("7. 获取全部记忆测试")
        self._reset()

        m1 = make_text_memory("记忆A")
        m2 = make_text_memory("记忆B")
        self.pm.add_memory(m1)
        self.pm.add_memory(m2)

        all_mems = self.pm.get_all_memories()
        self._assert(len(all_mems) == 2, "get_all_memories 返回正确数量")

        # 确保返回的是副本
        all_mems.append(make_text_memory("不应该影响原始列表"))
        self._assert(len(self.pm.get_all_memories()) == 2, "get_all_memories 返回的是副本，修改不影响原列表")

    def test_find_memory(self):
        print_section("8. 查找记忆测试")
        self._reset()

        m = make_text_memory("查找测试")
        self.pm.add_memory(m)

        self._assert(self.pm.find_memory(m.id) == True, "find_memory 对已有记忆返回True")
        self._assert(self.pm.find_memory("fake_id") == False, "find_memory 对不存在记忆返回False")

    def test_get_memory_by_user_id(self):
        print_section("9. 按用户ID获取记忆测试")
        self._reset()

        m1 = make_text_memory("用户A的记忆1", user_id="user_a")
        m2 = make_text_memory("用户A的记忆2", user_id="user_a")
        m3 = make_text_memory("用户B的记忆", user_id="user_b")
        self.pm.add_memory(m1)
        self.pm.add_memory(m2)
        self.pm.add_memory(m3)

        user_a_mems = self.pm.get_memory_by_user_id("user_a")
        self._assert(len(user_a_mems) == 2, "user_a 有2条记忆")

        user_b_mems = self.pm.get_memory_by_user_id("user_b")
        self._assert(len(user_b_mems) == 1, "user_b 有1条记忆")

        user_c_mems = self.pm.get_memory_by_user_id("user_c")
        self._assert(len(user_c_mems) == 0, "user_c 有0条记忆")

    # ==================== 4. 更新记忆测试 ====================
    def test_update_memory(self):
        print_section("10. 更新记忆测试")
        self._reset()

        m = make_text_memory("原始内容", importance=0.5)
        self.pm.add_memory(m)

        # 更新内容和重要性
        success = self.pm.update_memory(m.id, content="更新后的内容", importance=0.9)
        self._assert(success == True, "update_memory 返回True")

        updated = self.pm.get_memory(m.id)
        self._assert(updated.content == "更新后的内容", "内容已更新")
        self._assert(updated.importance == 0.9, "重要性已更新")

        # 更新不存在的记忆
        fail_result = self.pm.update_memory("nonexistent", content="无效更新")
        self._assert(fail_result == False, "更新不存在的记忆返回False")

    # ==================== 5. 删除记忆测试 ====================
    def test_remove_text_memory(self):
        print_section("11. 删除文本记忆测试")
        self._reset()

        m1 = make_text_memory("待删除的记忆", importance=0.5)
        m2 = make_text_memory("保留的记忆", importance=0.8)
        self.pm.add_memory(m1)
        self.pm.add_memory(m2)
        self._assert(len(self.pm.get_all_memories()) == 2, "删除前有2条记忆")

        result = self.pm.remove_memory(m1.id)
        # 注意: remove_memory 目前有 bug (pop 后再访问), 此处记录实际行为
        remaining = self.pm.get_all_memories()
        if result:
            self._assert(len(remaining) == 1, f"删除成功后剩余1条记忆 (实际={len(remaining)})")
            self._assert(remaining[0].id == m2.id, "剩余的记忆是m2")
            self._assert(not self.pm.find_memory(m1.id), "被删除的记忆不可被find_memory找到")
            self._assert(f"Perception_{m1.id}" not in self.pm.perceptions, "被删除的perception已从缓存移除")
        else:
            print_fail("remove_memory 返回False (存在已知bug: pop后再次访问已删除的key)")
            self.failed += 1
            self.errors.append("remove_memory 返回False - 已知bug")

    def test_remove_image_memory(self):
        print_section("12. 删除图像记忆测试")
        self._reset()

        m_img = make_image_memory("将要被删除的图片", "./images/test/dog.jpg", importance=0.5)
        m_text = make_text_memory("保留的文本记忆", importance=0.6)
        self.pm.add_memory(m_img)
        self.pm.add_memory(m_text)
        self._assert(len(self.pm.get_all_memories()) == 2, "删除前有2条记忆(1图+1文)")

        result = self.pm.remove_memory(m_img.id)
        remaining = self.pm.get_all_memories()
        if result:
            self._assert(len(remaining) == 1, f"删除图像记忆后剩余1条 (实际={len(remaining)})")
            self._assert(remaining[0].metadata.get("modality") == "text", "剩余的记忆是文本模态")
            # 验证 modality_index 更新
            image_perceptions = self.pm.modality_index.get("image", [])
            self._assert(f"Perception_{m_img.id}" not in image_perceptions, "image模态索引已移除该记忆")
        else:
            print_fail("remove_memory(图像) 返回False (存在已知bug)")
            self.failed += 1
            self.errors.append("remove_memory(图像) 返回False - 已知bug")

    def test_remove_nonexistent_memory(self):
        print_section("13. 删除不存在的记忆测试")
        self._reset()

        self.pm.add_memory(make_text_memory("正常记忆"))
        result = self.pm.remove_memory("nonexistent_id_12345")
        self._assert(result == False, "删除不存在的记忆返回False")
        self._assert(len(self.pm.get_all_memories()) == 1, "记忆数量不受影响")

    def test_remove_and_verify_stores(self):
        print_section("14. 删除记忆后验证各存储一致性测试")
        self._reset()

        m = make_text_memory("将被删除并验证存储", importance=0.5)
        self.pm.add_memory(m)

        # 删除前验证 document_store 能查到
        doc_before = self.document_store.get_memory(m.id)
        self._assert(doc_before is not None, "删除前: document_store能查到该记忆")

        result = self.pm.remove_memory(m.id)
        if result:
            # 验证 document_store
            doc_after = self.document_store.get_memory(m.id)
            self._assert(doc_after is None, "删除后: document_store中该记忆已移除")

            # 验证缓存一致性
            self._assert(m.id not in self.pm.id_to_memory, "删除后: id_to_memory中已移除")
            self._assert(f"Perception_{m.id}" not in self.pm.perceptions, "删除后: perceptions中已移除")
            self._assert(len(self.pm.get_all_memories()) == 0, "删除后: 记忆总数为0")
        else:
            print_fail("remove_memory 返回False，无法验证存储一致性 (存在已知bug)")
            self.failed += 1
            self.errors.append("remove_and_verify_stores: remove_memory返回False - 已知bug")

    # ==================== 6. 搜索记忆测试 ====================
    def test_search_text_memory(self):
        print_section("15. 文本搜索记忆测试")
        self._reset()

        self.pm.add_memory(make_text_memory("今天天气很好，阳光明媚，适合散步。", importance=0.8))
        self.pm.add_memory(make_text_memory("我喜欢吃苹果和香蕉这些水果。", importance=0.5))
        self.pm.add_memory(make_text_memory("北京是中国的首都，历史悠久。", importance=0.7))

        # 指定 modality=text 搜索
        results = self.pm.search_memory("天气怎么样", limit=3, modality="text")
        self._assert(len(results) > 0, f"文本搜索返回 {len(results)} 条结果")
        # 天气相关的记忆应该排在前面
        self._assert("天气" in results[0].content or "阳光" in results[0].content,
                      "搜索'天气怎么样'应该最先返回天气相关记忆",
                      f"实际第一条: {results[0].content[:30]}")

    def test_search_with_limit(self):
        print_section("16. 搜索限制数量测试")
        self._reset()

        for i in range(5):
            self.pm.add_memory(make_text_memory(f"第{i+1}条测试记忆，内容是关于数字{i+1}的random文本。"))

        results = self.pm.search_memory("测试记忆", limit=2, modality="text")
        self._assert(len(results) <= 2, f"limit=2 时最多返回2条 (实际返回{len(results)}条)")

    def test_search_unsupported_modality(self):
        print_section("17. 搜索不支持的目标模态测试")
        self._reset()

        results = self.pm.search_memory("测试", modality="video")
        self._assert(results == [], "搜索不支持的模态返回空列表")

    # ==================== 7. 编码器测试 ====================
    def test_text_encoder(self):
        print_section("18. 文本编码器测试")

        encoding = self.pm._encoder_text("你好世界")
        self._assert(encoding is not None, "文本编码不为None")
        self._assert(len(encoding) == 384, f"文本编码维度=384 (实际={len(encoding)})")

    def test_image_encoder(self):
        print_section("19. 图像编码器测试")

        encoding = self.pm._encoder_image("./images/test/dog.jpg")
        self._assert(encoding is not None, "图像编码不为None")
        self._assert(len(encoding) == 512, f"图像编码维度=512 (实际={len(encoding)})")

    def test_text_clip_encoder(self):
        print_section("20. CLIP文本编码器测试")

        clip_encoding = self.pm._encoder_text_clip("一只可爱的小狗")
        self._assert(clip_encoding is not None, "CLIP文本编码不为None")
        self._assert(len(clip_encoding) == 512, f"CLIP文本编码维度=512 (实际={len(clip_encoding)})")

    def test_cross_modal_similarity(self):
        print_section("21. 跨模态相似度测试 (CLIP)")

        dog_img_enc = np.array(self.pm._encoder_image("./images/test/dog.jpg")).reshape(1, -1)
        tiger_img_enc = np.array(self.pm._encoder_image("./images/test/tiger.jpg")).reshape(1, -1)
        dog_text_enc = np.array(self.pm._encoder_text_clip("a photo of a dog")).reshape(1, -1)
        tiger_text_enc = np.array(self.pm._encoder_text_clip("a photo of a tiger")).reshape(1, -1)

        sim_dog_dog = cosine_similarity(dog_img_enc, dog_text_enc)[0][0]
        sim_dog_tiger = cosine_similarity(dog_img_enc, tiger_text_enc)[0][0]
        sim_tiger_tiger = cosine_similarity(tiger_img_enc, tiger_text_enc)[0][0]
        sim_tiger_dog = cosine_similarity(tiger_img_enc, dog_text_enc)[0][0]

        print(f"    dog图片 vs 'a dog'文本:   {sim_dog_dog:.4f}")
        print(f"    dog图片 vs 'a tiger'文本: {sim_dog_tiger:.4f}")
        print(f"    tiger图片 vs 'a tiger'文本: {sim_tiger_tiger:.4f}")
        print(f"    tiger图片 vs 'a dog'文本:   {sim_tiger_dog:.4f}")

        self._assert(sim_dog_dog > sim_dog_tiger,
                      f"dog图片与'dog'文本的相似度({sim_dog_dog:.4f}) > 与'tiger'文本({sim_dog_tiger:.4f})")
        self._assert(sim_tiger_tiger > sim_tiger_dog,
                      f"tiger图片与'tiger'文本的相似度({sim_tiger_tiger:.4f}) > 与'dog'文本({sim_tiger_dog:.4f})")

    def test_encoder_data_dispatch(self):
        print_section("22. _encoder_data 模态分发测试")

        text_enc = self.pm._encoder_data("测试文本", "text")
        self._assert(text_enc is not None and len(text_enc) == 384, "_encoder_data text模态分发正确")

        img_enc = self.pm._encoder_data("./images/test/dog.jpg", "image")
        self._assert(img_enc is not None and len(img_enc) == 512, "_encoder_data image模态分发正确")

        unsupported = self.pm._encoder_data("test", "video")
        self._assert(unsupported is None, "_encoder_data 不支持的模态返回None")

    def test_detach_query_modality(self):
        print_section("23. _detach_query_modality 模态检测测试")

        self._assert(self.pm._detach_query_modality("photo.jpg") == "image", "jpg 检测为 image")
        self._assert(self.pm._detach_query_modality("photo.png") == "image", "png 检测为 image")
        self._assert(self.pm._detach_query_modality("photo.jpeg") == "image", "jpeg 检测为 image")
        self._assert(self.pm._detach_query_modality("photo.webp") == "image", "webp 检测为 image")
        self._assert(self.pm._detach_query_modality("audio.mp3") == "audio", "mp3 检测为 audio")
        self._assert(self.pm._detach_query_modality("audio.wav") == "audio", "wav 检测为 audio")
        self._assert(self.pm._detach_query_modality("audio.aac") == "audio", "aac 检测为 audio")
        self._assert(self.pm._detach_query_modality("audio.flac") == "audio", "flac 检测为 audio")
        self._assert(self.pm._detach_query_modality("今天天气好") == "text", "普通文本检测为 text")
        self._assert(self.pm._detach_query_modality("hello world") == "text", "英文文本检测为 text")

    # ==================== 8. 遗忘策略测试 ====================
    def test_forget_by_importance(self):
        print_section("24. 重要性遗忘策略测试")
        self._reset()

        self.pm.add_memory(make_text_memory("高重要性记忆", importance=0.8))
        self.pm.add_memory(make_text_memory("中重要性记忆", importance=0.5))
        self.pm.add_memory(make_text_memory("低重要性记忆1", importance=0.2))
        self.pm.add_memory(make_text_memory("低重要性记忆2", importance=0.1))

        forgotten_count = self.pm.forget(ForgetType.IMPORTANCE, threshold=0.3)
        self._assert(forgotten_count == 2, f"重要性<0.3的记忆被遗忘 (遗忘{forgotten_count}条，期望2条)")

        remaining = self.pm.get_all_memories()
        self._assert(len(remaining) == 2, f"剩余记忆数量为2 (实际={len(remaining)})")
        for m in remaining:
            self._assert(m.importance >= 0.3,
                          f"剩余记忆重要性≥0.3 (记忆'{m.content[:10]}' importance={m.importance})")

    def test_forget_by_time(self):
        print_section("25. 时间遗忘策略测试")
        self._reset()

        # 添加一条 "旧" 记忆 (手动设置时间戳)
        old_memory = MemoryItem(
            id=str(uuid.uuid4()), user_id="test_user",
            content="这是一条很旧的记忆",
            type="perceptual",
            timestamp=datetime.now() - timedelta(days=60),
            importance=0.8,
            metadata={"modality": "text"}
        )
        new_memory = make_text_memory("这是一条新记忆", importance=0.5)

        self.pm.add_memory(old_memory)
        self.pm.add_memory(new_memory)

        forgotten_count = self.pm.forget(ForgetType.TIME, max_age_days=30)
        self._assert(forgotten_count == 1, f"超过30天的记忆被遗忘 (遗忘{forgotten_count}条，期望1条)")

        remaining = self.pm.get_all_memories()
        self._assert(len(remaining) == 1, f"剩余1条记忆")
        self._assert(remaining[0].content == "这是一条新记忆", "剩余的是新记忆")

    def test_forget_by_capacity(self):
        print_section("26. 容量遗忘策略测试")
        self._reset()

        # max_capacity=5, 添加7条记忆
        for i in range(7):
            self.pm.add_memory(make_text_memory(f"容量测试记忆{i}", importance=0.1 * (i + 1)))

        self._assert(len(self.pm.get_all_memories()) == 7, "添加7条记忆成功")

        forgotten_count = self.pm.forget(ForgetType.CAPACITY)
        self._assert(forgotten_count == 2, f"超出容量应遗忘2条 (实际遗忘{forgotten_count}条)")
        self._assert(len(self.pm.get_all_memories()) == 5, f"容量遗忘后剩余5条记忆")

        # 验证被遗忘的是重要性最低的
        remaining = self.pm.get_all_memories()
        remaining_importances = [m.importance for m in remaining]
        self._assert(min(remaining_importances) >= 0.3,
                      f"保留的记忆重要性最低值≥0.3 (实际最低={min(remaining_importances)})")

    def test_forget_no_match(self):
        print_section("27. 遗忘策略无匹配测试")
        self._reset()

        self.pm.add_memory(make_text_memory("高重要性", importance=0.9))
        forgotten = self.pm.forget(ForgetType.IMPORTANCE, threshold=0.1)
        self._assert(forgotten == 0, "无匹配时遗忘数量为0")
        self._assert(len(self.pm.get_all_memories()) == 1, "记忆数量不变")

    # ==================== 9. 统计信息测试 ====================
    def test_get_stats(self):
        print_section("28. 统计信息测试")
        self._reset()

        self.pm.add_memory(make_text_memory("统计测试1", importance=0.4))
        self.pm.add_memory(make_text_memory("统计测试2", importance=0.6))
        self.pm.add_memory(make_image_memory("统计测试图片", "./images/test/dog.jpg", importance=0.8))

        stats = self.pm.get_stats()

        # 基本计数
        self._assert(stats["count"] == 3, f"统计: count=3 (实际={stats['count']})")
        self._assert(stats["total_count"] == 3, f"统计: total_count=3 (实际={stats['total_count']})")
        self._assert(stats["perceptions_count"] == 3, f"统计: perceptions_count=3")
        self._assert(stats["forgotten_count"] == 0, f"统计: forgotten_count=0 (硬删除模式)")
        self._assert(stats["memory_type"] == "perceptual", "统计: memory_type=perceptual")

        # 模态计数
        self._assert("text" in stats["modality_counts"], "统计: modality_counts包含text")
        self._assert(stats["modality_counts"]["text"] == 2, f"统计: text模态数量=2")
        self._assert(stats["modality_counts"]["image"] == 1, f"统计: image模态数量=1")

        # 支持的模态
        self._assert("text" in stats["supported_modalities"], "统计: supported_modalities包含text")
        self._assert("image" in stats["supported_modalities"], "统计: supported_modalities包含image")

        # 平均重要性
        expected_avg = (0.4 + 0.6 + 0.8) / 3
        self._assert(abs(stats["avg_importance"] - expected_avg) < 0.01,
                      f"统计: avg_importance≈{expected_avg:.2f} (实际={stats['avg_importance']:.4f})")

        # vector_stores 子字典
        self._assert("vector_stores" in stats, "统计: 包含vector_stores字段")
        self._assert("text" in stats["vector_stores"], "统计: vector_stores包含text")
        self._assert("image" in stats["vector_stores"], "统计: vector_stores包含image")
        self._assert(isinstance(stats["vector_stores"]["text"], dict), "统计: vector_stores['text']是字典")
        self._assert(isinstance(stats["vector_stores"]["image"], dict), "统计: vector_stores['image']是字典")

        # document_store 子字典
        self._assert("document_store" in stats, "统计: 包含document_store字段")
        self._assert(isinstance(stats["document_store"], dict), "统计: document_store是字典")

    def test_get_stats_empty(self):
        print_section("29. 空状态统计信息测试")
        self._reset()

        stats = self.pm.get_stats()
        self._assert(stats["count"] == 0, "空状态: count=0")
        self._assert(stats["total_count"] == 0, "空状态: total_count=0")
        self._assert(stats["perceptions_count"] == 0, "空状态: perceptions_count=0")
        self._assert(stats["forgotten_count"] == 0, "空状态: forgotten_count=0")
        self._assert(stats["avg_importance"] == 0.0, "空状态: avg_importance=0.0")
        self._assert(stats["modality_counts"] == {}, "空状态: modality_counts为空字典")
        self._assert(stats["memory_type"] == "perceptual", "空状态: memory_type=perceptual")
        self._assert("vector_stores" in stats, "空状态: 包含vector_stores字段")
        self._assert("document_store" in stats, "空状态: 包含document_store字段")

    def test_get_stats_after_operations(self):
        print_section("30. 增删后统计信息一致性测试")
        self._reset()

        # 添加3条记忆
        m1 = make_text_memory("统计操作测试1", importance=0.3)
        m2 = make_text_memory("统计操作测试2", importance=0.6)
        m3 = make_image_memory("统计操作图片", "./images/test/tiger.jpg", importance=0.9)
        self.pm.add_memory(m1)
        self.pm.add_memory(m2)
        self.pm.add_memory(m3)

        stats1 = self.pm.get_stats()
        self._assert(stats1["count"] == 3, "增删一致性: 添加后count=3")
        expected_avg1 = (0.3 + 0.6 + 0.9) / 3
        self._assert(abs(stats1["avg_importance"] - expected_avg1) < 0.01,
                      f"增删一致性: 添加后avg_importance≈{expected_avg1:.2f}")

        # 更新一条记忆的重要性
        self.pm.update_memory(m1.id, content="已更新", importance=0.8)
        stats2 = self.pm.get_stats()
        self._assert(stats2["count"] == 3, "增删一致性: 更新后count不变=3")
        expected_avg2 = (0.8 + 0.6 + 0.9) / 3
        self._assert(abs(stats2["avg_importance"] - expected_avg2) < 0.01,
                      f"增删一致性: 更新后avg_importance≈{expected_avg2:.2f} (实际={stats2['avg_importance']:.4f})")

    # ==================== 10. 清空记忆测试 ====================
    def test_clear_memory(self):
        print_section("31. 清空记忆测试")
        self._reset()

        self.pm.add_memory(make_text_memory("清空测试1"))
        self.pm.add_memory(make_text_memory("清空测试2"))
        self.pm.add_memory(make_image_memory("清空测试图片", "./images/test/tiger.jpg"))

        self._assert(len(self.pm.get_all_memories()) == 3, "清空前有3条记忆")

        self.pm.clear_memory()

        self._assert(len(self.pm.get_all_memories()) == 0, "清空后记忆数量为0")
        self._assert(len(self.pm.perceptions) == 0, "清空后perceptions为空")
        self._assert(len(self.pm.modality_index) == 0, "清空后modality_index为空")
        self._assert(len(self.pm.id_to_memory) == 0, "清空后id_to_memory为空")

    # ==================== 11. 混合文本+图像工作流测试 ====================
    def test_mixed_modality_workflow(self):
        print_section("32. 混合模态完整工作流测试")
        self._reset()

        # 添加
        t1 = make_text_memory("齐晶晶是一个漂亮的女孩，她今年23岁了。", importance=0.7)
        t2 = make_text_memory("姚浩杰是齐晶晶的儿子，今年18岁了。", importance=0.3)
        i1 = make_image_memory("可爱的小狗", "./images/test/dog.jpg", importance=0.6)
        i2 = make_image_memory("凶猛的老虎", "./images/test/tiger.jpg", importance=0.4)

        self.pm.add_memory(t1)
        self.pm.add_memory(t2)
        self.pm.add_memory(i1)
        self.pm.add_memory(i2)

        self._assert(len(self.pm.get_all_memories()) == 4, "混合模态: 成功添加4条记忆")
        self._assert(self.pm.modality_index.get("text", []) != [], "混合模态: text索引非空")
        self._assert(self.pm.modality_index.get("image", []) != [], "混合模态: image索引非空")

        # 统计
        stats = self.pm.get_stats()
        self._assert(stats["modality_counts"]["text"] == 2, "混合模态: text数量=2")
        self._assert(stats["modality_counts"]["image"] == 2, "混合模态: image数量=2")

        # 搜索文本
        text_results = self.pm.search_memory("齐晶晶是谁", modality="text", limit=2)
        self._assert(len(text_results) > 0, f"混合模态文本搜索: 找到{len(text_results)}条结果")

        # 遗忘
        forgotten = self.pm.forget(ForgetType.IMPORTANCE, threshold=0.4)
        self._assert(forgotten >= 1, f"混合模态遗忘: 遗忘了{forgotten}条低重要性记忆")

        remaining = self.pm.get_all_memories()
        for m in remaining:
            self._assert(m.importance >= 0.4,
                          f"混合模态遗忘后: 剩余记忆重要性≥0.4 ('{m.content[:10]}' = {m.importance})")

    # ==================== 12. __str__ / __repr__ 测试 ====================
    def test_str_repr(self):
        print_section("33. __str__ 和 __repr__ 测试")
        self._reset()

        self.pm.add_memory(make_text_memory("str测试"))
        s = str(self.pm)
        self._assert("PerceptualMemory" in s, f"__str__ 包含类名: '{s}'")
        self._assert("count=1" in s, f"__str__ 包含count: '{s}'")

        r = repr(self.pm)
        self._assert(r == s, "__repr__ 与 __str__ 一致")

    # ==================== 13. 多次操作一致性测试 ====================
    def test_add_remove_consistency(self):
        print_section("34. 添加-查询-一致性测试")
        self._reset()

        # 添加5条记忆
        ids = []
        for i in range(5):
            m = make_text_memory(f"一致性测试记忆{i}", importance=0.5)
            mid = self.pm.add_memory(m)
            ids.append(mid)

        self._assert(len(self.pm.get_all_memories()) == 5, "添加5条后总数为5")
        self._assert(len(self.pm.id_to_memory) == 5, "id_to_memory大小为5")
        self._assert(len(self.pm.perceptions) == 5, "perceptions大小为5")

        # 外部缓存与内部一致
        for mid in ids:
            self._assert(self.pm.find_memory(mid), f"记忆 {mid[:8]}... 可被找到")

    # ==================== 14. 边界情况测试 ====================
    def test_empty_state_operations(self):
        print_section("35. 空状态操作测试")
        self._reset()

        # 空状态搜索
        results = self.pm.search_memory("任意查询", modality="text")
        self._assert(results == [], "空状态搜索返回空列表")

        # 空状态获取
        self._assert(self.pm.get_memory("any_id") is None, "空状态get_memory返回None")
        self._assert(self.pm.find_memory("any_id") == False, "空状态find_memory返回False")
        self._assert(self.pm.get_memory_by_user_id("any_user") == [], "空状态get_by_user_id返回空列表")
        self._assert(self.pm.get_all_memories() == [], "空状态get_all返回空列表")

        # 空状态遗忘
        self._assert(self.pm.forget(ForgetType.IMPORTANCE) == 0, "空状态遗忘返回0")

    # ==================== 运行所有测试 ====================
    def run_all(self):
        tests = [
            self.test_perception_class,
            self.test_add_text_memory,
            self.test_add_image_memory,
            self.test_add_unsupported_modality,
            self.test_add_memories_batch,
            self.test_get_memory,
            self.test_get_all_memories,
            self.test_find_memory,
            self.test_get_memory_by_user_id,
            self.test_update_memory,
            self.test_remove_text_memory,
            self.test_remove_image_memory,
            self.test_remove_nonexistent_memory,
            self.test_remove_and_verify_stores,
            self.test_search_text_memory,
            self.test_search_with_limit,
            self.test_search_unsupported_modality,
            self.test_text_encoder,
            self.test_image_encoder,
            self.test_text_clip_encoder,
            self.test_cross_modal_similarity,
            self.test_encoder_data_dispatch,
            self.test_detach_query_modality,
            self.test_forget_by_importance,
            self.test_forget_by_time,
            self.test_forget_by_capacity,
            self.test_forget_no_match,
            self.test_get_stats,
            self.test_get_stats_empty,
            self.test_get_stats_after_operations,
            self.test_clear_memory,
            self.test_mixed_modality_workflow,
            self.test_str_repr,
            self.test_add_remove_consistency,
            self.test_empty_state_operations,
        ]

        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                test_name = test_func.__name__
                print_fail(f"{test_name} 抛出异常", str(e))
                traceback.print_exc()
                self.failed += 1
                self.errors.append(f"{test_name}: {e}")

        # 最终统计
        print_section("测试结果汇总")
        total = self.passed + self.failed
        print(f"  总计: {total} 项")
        print(f"  {Colors.GREEN}通过: {self.passed}{Colors.END}")
        print(f"  {Colors.RED}失败: {self.failed}{Colors.END}")
        if self.errors:
            print(f"\n  {Colors.RED}失败项:{Colors.END}")
            for i, err in enumerate(self.errors, 1):
                print(f"    {i}. {err}")
        print()
        return self.failed == 0


if __name__ == "__main__":
    tester = TestPerceptualMemory()
    success = tester.run_all()
    sys.exit(0 if success else 1)