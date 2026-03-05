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
from core.llm import EasyLLM
from memory.V2.SemanticMemory import SemanticMemory
from memory.V2.Extractor.Extractor import Extractor
from memory.V2.Store.Neo4jGraphStore import Neo4jGraphStore
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
from memory.V2.BaseMemory import MemoryConfig, MemoryItem, MemoryType


def print_section(title: str, num: int):
    print(f"\n{'='*60}")
    print(f"  {num}. {title}")
    print(f"{'='*60}")


def print_result(label: str, success: bool, detail: str = ""):
    icon = "✅" if success else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)


def main():
    # ==================== 初始化 ====================
    print("正在初始化组件...")
    llm = EasyLLM()
    extractor = Extractor(llm, False)
    graph_store = Neo4jGraphStore(uri="bolt://localhost:17687", username="neo4j", password="password")
    vector_store = QdrantVectorStore(way="memory", collection_name="semantic_memory")
    embedding_model = HuggingfaceEmbeddingModel()
    semantic_memory = SemanticMemory(
        memory_config=MemoryConfig(),
        vector_store=vector_store,
        graph_store=graph_store,
        extractor=extractor,
        embedding_model=embedding_model
    )

    # 清空数据库，避免旧数据干扰
    graph_store.clear()
    print("初始化完成，数据库已清空\n")

    passed = 0
    failed = 0
    total = 0
    memory1 = MemoryItem(
            id=str(uuid.uuid4()),
        user_id="test_user",
        type="semantic",
        content="魏星迪今天去了北京，吃了烤鸭",
        timestamp=datetime.now(),
        importance=0.8,
        metadata={}
        )
    memory2 = MemoryItem(
        id=str(uuid.uuid4()),
        user_id="test_user",
        type="semantic",
        content="张三在上海参加了人工智能大会",
        timestamp=datetime.now(),
        importance=0.6,
        metadata={}
    )

    memory3 = MemoryItem(
        id=str(uuid.uuid4()),
        user_id="test_user",
        type="semantic",
        content="李华在深圳的腾讯公司担任高级工程师",
        timestamp=datetime.now(),
        importance=0.7,
        metadata={}
    )
    memory4 = MemoryItem(
        id=str(uuid.uuid4()),
        user_id="user_b",
        type="semantic",
        content="王五昨天在广州品尝了广式早茶",
        timestamp=datetime.now(),
        importance=0.5,
        metadata={}
    )
    # ==================== 1. 测试 add_memory ====================
    def test_add_memory(total,passed,failed):
        print_section("add_memory — 添加多条记忆", 1)
        total += 1
        # 添加记忆
        result1 = semantic_memory.add_memory(memory1)
        print_result("添加记忆1", bool(result1), f"id={result1}")


        result2 = semantic_memory.add_memory(memory2)
        print_result("添加记忆2", bool(result2), f"id={result2}")

        result3 = semantic_memory.add_memory(memory3)
        print_result("添加记忆3", bool(result3), f"id={result3}")


        result4 = semantic_memory.add_memory(memory4)
        print_result("添加记忆4 (user_b)", bool(result4), f"id={result4}")

        all_added = all([result1, result2, result3, result4])
        if all_added:
            passed += 1
        else:
            failed += 1
        print_result("add_memory 整体", all_added, f"共添加 {sum(bool(r) for r in [result1,result2,result3,result4])}/4 条")


        # ==================== 2. 测试重复添加 ====================
        print_section("add_memory — 重复添加检测", 2)
        total += 1

        dup_result = semantic_memory.add_memory(memory1)
        dup_ok = (dup_result == "")
        print_result("重复添加应返回空字符串", dup_ok, f"返回='{dup_result}'")
        if dup_ok:
            passed += 1
        else:
            failed += 1

        # ==================== 3. 测试 find_memory ====================
        print_section("find_memory — 查找记忆", 3)
        total += 1

        found1 = semantic_memory.find_memory(memory1.id)
        found_fake = semantic_memory.find_memory("non-existent-id-12345")
        find_ok = found1 and not found_fake
        print_result("查找已存在的记忆", found1)
        print_result("查找不存在的记忆", not found_fake, f"返回 {found_fake} (期望 False)")
        if find_ok:
            passed += 1
        else:
            failed += 1
    test_add_memory(total,passed,failed)
    # ==================== 4. 测试 get_stats ====================
    def test_get_stats(total,passed,failed):
        print_section("get_stats — 统计信息", 4)
        total += 1

        stats = semantic_memory.get_stats()
        print(f"  记忆数: {stats['count']}")
        print(f"  实体数: {stats['entity_count']}")
        print(f"  关系数: {stats['relation_count']}")
        print(f"  类型: {stats['memory_type']}")
        print(f"  最大容量: {stats['max_capacity']}")
        print(f"  用户列表: {stats['user_ids']}")
        print(f"  平均重要性: {stats['avg_importance']:.4f}")

        stats_ok = (
            stats["count"] == 4
            and stats["entity_count"] > 0
            and stats["relation_count"] >= 0
            and stats["memory_type"] == "semantic"
            and len(stats["user_ids"]) == 2  # test_user + user_b
        )
        print_result("get_stats 数据正确", stats_ok)
        if stats_ok:
            passed += 1
        else:
            failed += 1
    # test_get_stats(total,passed,failed)

    # ==================== 5. 测试 entities_name_to_id ====================
    def test_entities_name_to_id(total,passed,failed):
        print_section("entities_name_to_id — 名称索引", 5)
        total += 1

        name_cache_count = len(semantic_memory.entities_name_to_id)
        entity_count = len(semantic_memory.entities)
        name_ok = name_cache_count == entity_count and name_cache_count > 0
        print(f"  entities 数: {entity_count}")
        print(f"  name_to_id 数: {name_cache_count}")
        # 展示前几个映射
        for name, eid in list(semantic_memory.entities_name_to_id.items())[:5]:
            print(f"    '{name}' → {eid[:16]}...")
        print_result("name_to_id 与 entities 数量一致", name_ok)
        if name_ok:
            passed += 1
        else:
            failed += 1
    # test_entities_name_to_id(total,passed,failed)

    # ==================== 6. 测试 search_memory ====================
    def test_search_memory(total,passed,failed):    
        print_section("search_memory — 混合搜索", 6)
        total += 1

        search_results = semantic_memory.search_memory("北京烤鸭", limit=5, user_id="test_user")
        print(f"  搜索 '北京烤鸭' (user=test_user), 返回 {len(search_results)} 条:")
        for i, mem in enumerate(search_results):
            combined = mem.metadata.get("combined_score", 0)
            vec = mem.metadata.get("vector_score", 0)
            graph = mem.metadata.get("graph_score", 0)
            print(f"    结果{i+1}: combined={combined:.4f} (vec={vec:.4f}, graph={graph:.4f})")
            print(f"           内容: {mem.content[:40]}...")

        search_ok = len(search_results) > 0
        # 第一个结果应该是北京烤鸭那条
        if search_results:
            top = search_results[0]
            search_ok = search_ok and ("北京" in top.content or "烤鸭" in top.content)
        print_result("search_memory 返回相关结果", search_ok)
        if search_ok:
            passed += 1
        else:
            failed += 1
    test_search_memory(total,passed,failed)

    # ==================== 7. 测试 search_memory 跨用户隔离 ====================
    def test_search_memory_isolation(total,passed,failed):
        print_section("search_memory — 用户隔离", 7)
        total += 1

        search_b = semantic_memory.search_memory("广州早茶", limit=5, user_id="user_b")
        print(f"  搜索 '广州早茶' (user=user_b), 返回 {len(search_b)} 条")
        for i, mem in enumerate(search_b):
            print(f"    结果{i+1}: content={mem.content[:40]}... user={mem.user_id}")

        # user_b 的搜索不应该返回 test_user 的记忆
        isolation_ok = True
        for mem in search_b:
            if mem.user_id != "user_b":
                isolation_ok = False
                break
        # 注意：图谱搜索可能跨用户，这里只做宽松检查
        print_result("用户隔离 (向量搜索层面)", isolation_ok or len(search_b) == 0,
                    f"返回 {len(search_b)} 条, 均属于 user_b" if isolation_ok else "存在跨用户结果")
        if isolation_ok or len(search_b) == 0:
            passed += 1
        else:
            failed += 1
    # test_search_memory_isolation(total,passed,failed)

    # ==================== 8. 测试 update_memory ====================
    def test_update_memory(total,passed,failed):
        print_section("update_memory — 更新记忆", 8)
        total += 1

        updated = semantic_memory.update_memory(
            id=memory1.id,
            content="魏星迪昨天去了南京，品尝了鸭血粉丝汤和盐水鸭",
            importance=0.95
        )
        print_result("更新记忆1", updated)

        updated_memory = semantic_memory.id_to_memory.get(memory1.id)
        content_ok = updated_memory is not None and "南京" in updated_memory.content
        importance_ok = updated_memory is not None and updated_memory.importance == 0.95
        update_ok = updated and content_ok and importance_ok
        if updated_memory:
            print(f"  更新后内容: {updated_memory.content}")
            print(f"  更新后重要性: {updated_memory.importance}")
            print(f"  更新后实体数: {len(updated_memory.metadata.get('entities', []))}")
        print_result("内容已正确更新", content_ok)
        print_result("重要性已正确更新", importance_ok)
        if update_ok:
            passed += 1
        else:
            failed += 1

    # test_update_memory(total,passed,failed)
    # ==================== 9. 测试更新不存在的记忆 ====================
    def test_update_memory_nonexistent(total,passed,failed):
        print_section("update_memory — 更新不存在的记忆", 9)
        total += 1

        bad_update = semantic_memory.update_memory(
            id="non-existent-id-67890",
            content="这条记忆不存在"
        )
        bad_update_ok = not bad_update
        print_result("更新不存在的记忆应返回 False", bad_update_ok, f"返回={bad_update}")
        if bad_update_ok:
            passed += 1
        else:
            failed += 1
    
    # test_update_memory_nonexistent(total,passed,failed)

    # ==================== 10. 测试 _get_graph_context ====================
    def test_get_graph_context(total,passed,failed):
        print_section("_get_graph_context — 图谱上下文", 10)
        total += 1

        target_memory = semantic_memory.id_to_memory.get(memory1.id)
        if target_memory:
            context = semantic_memory._get_graph_context(target_memory)
            ctx_entities = context.get("entities", [])
            ctx_relations = context.get("relations", [])
            print(f"  关联实体: {[e['name'] for e in ctx_entities]}")
            rel_labels = ["{}-{}-{}".format(r['from'][:8], r['type'], r['to'][:8]) for r in ctx_relations]
            print(f"  关联关系: {rel_labels}")
            ctx_ok = len(ctx_entities) > 0
        else:
            ctx_ok = False
            print("  ⚠️ 记忆不存在")
        print_result("图谱上下文包含实体", ctx_ok)
        if ctx_ok:
            passed += 1
        else:
            failed += 1
    
    # test_get_graph_context(total,passed,failed)

    # ==================== 11. 测试 remove_memory ====================
    def test_remove_memory(total,passed,failed):
        print_section("remove_memory — 删除记忆", 11)
        total += 1

        stats_before = semantic_memory.get_stats()
        count_before = stats_before["count"]

        removed = semantic_memory.remove_memory(memory2.id)
        still_found = semantic_memory.find_memory(memory2.id)

        stats_after = semantic_memory.get_stats()
        count_after = stats_after["count"]

        remove_ok = removed and not still_found and count_after == count_before - 1
        print_result("删除记忆2", removed)
        print_result("删除后查找应为 False", not still_found, f"find={still_found}")
        print_result("记忆数减1", count_after == count_before - 1,
                    f"{count_before} → {count_after}")
        if remove_ok:
            passed += 1
        else:
            failed += 1
    
    # test_remove_memory(total,passed,failed)

    # ==================== 12. 测试删除不存在的记忆 ====================
    def test_remove_memory_nonexistent(total,passed,failed):
        print_section("remove_memory — 删除不存在的记忆", 12)
        total += 1

        bad_remove = semantic_memory.remove_memory("non-existent-id-abcde")
        bad_remove_ok = not bad_remove
        print_result("删除不存在的记忆应返回 False", bad_remove_ok, f"返回={bad_remove}")
        if bad_remove_ok:
            passed += 1
        else:
            failed += 1
    
    # test_remove_memory_nonexistent(total,passed,failed)

    # ==================== 13. 测试 load_from_store ====================
    def test_load_from_store(total,passed,failed):
        print_section("load_from_store — 从数据库重新加载缓存", 13)
        total += 1

        # 记录当前状态
        pre_load_count = semantic_memory.get_stats()["count"]
        pre_load_entities = len(semantic_memory.entities)
        pre_load_relations = len(semantic_memory.relations)
        print(f"  加载前: {pre_load_count} 条记忆, {pre_load_entities} 实体, {pre_load_relations} 关系")

        # 执行 load_from_store（从向量库和图库重建缓存）
        semantic_memory.load_from_store()

        post_load_count = semantic_memory.get_stats()["count"]
        post_load_entities = len(semantic_memory.entities)
        post_load_relations = len(semantic_memory.relations)
        post_name_to_id = len(semantic_memory.entities_name_to_id)
        print(f"  加载后: {post_load_count} 条记忆, {post_load_entities} 实体, {post_load_relations} 关系")
        print(f"  name_to_id 数: {post_name_to_id}")

        load_ok = (
            post_load_count == pre_load_count  # 记忆数应一致
            and post_load_entities > 0
            and post_name_to_id == post_load_entities  # name_to_id 应与 entities 同步
        )
        print_result("load_from_store 数据一致", load_ok)

        # 验证加载后搜索仍可用
        search_after_load = semantic_memory.search_memory("南京", limit=3, user_id="test_user")
        search_after_ok = len(search_after_load) > 0
        print_result("加载后搜索仍正常", search_after_ok, f"返回 {len(search_after_load)} 条")

        if load_ok and search_after_ok:
            passed += 1
        else:
            failed += 1
    
    # test_load_from_store(total,passed,failed)

    # ==================== 14. 测试 sync_stores ====================
    def test_sync_stores(total,passed,failed):
        print_section("sync_stores — 同步检测", 14)
        total += 1

        sync_result = semantic_memory.sync_stores()
        print(f"  cache_count: {sync_result['cache_count']}")
        print(f"  synced_to_vector: {sync_result['synced_to_vector']}")
        print(f"  synced_from_graph: {sync_result['synced_from_graph']}")
        print(f"  errors: {sync_result['errors']}")

        # 数据已同步，不应有新增同步
        sync_ok = len(sync_result["errors"]) == 0
        print_result("sync_stores 无错误", sync_ok)
        if sync_ok:
            passed += 1
        else:
            failed += 1

        # ==================== 15. 测试 clear_memory ====================
        print_section("clear_memory — 清空所有记忆", 15)
        total += 1

        semantic_memory.clear_memory()
        final_stats = semantic_memory.get_stats()
        clear_ok = (
            final_stats["count"] == 0
            and final_stats["entity_count"] == 0
            and final_stats["relation_count"] == 0
            and len(semantic_memory.entities_name_to_id) == 0
            and len(semantic_memory.memory_embeddings) == 0
        )
        print(f"  记忆数: {final_stats['count']}")
        print(f"  实体数: {final_stats['entity_count']}")
        print(f"  关系数: {final_stats['relation_count']}")
        print(f"  name_to_id 数: {len(semantic_memory.entities_name_to_id)}")
        print(f"  embedding 数: {len(semantic_memory.memory_embeddings)}")
        print_result("所有数据已清空", clear_ok)
        if clear_ok:
            passed += 1
        else:
            failed += 1

    # test_sync_stores(total,passed,failed)

    # ==================== 汇总 ====================
    print(f"\n{'='*60}")
    print(f"  测试汇总")
    print(f"{'='*60}")
    print(f"  总计: {total} 项")
    print(f"  ✅ 通过: {passed} 项")
    print(f"  ❌ 失败: {failed} 项")
    if failed == 0:
        print(f"\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️ 有 {failed} 项测试未通过")


if __name__ == "__main__":
    main()