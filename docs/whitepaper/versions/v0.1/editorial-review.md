# 编辑与审校记录

正文：`easyagent-whitepaper.zh-CN.md`。本附件面向作者，不作为正文中的实验结果。

## 1. 写作提纲与段落职责

1. 摘要与问题定义：为什么领域化会进入运行机制。
2. 两条路线：应用框架留下的协作成本，产品二次开发留下的机制替换成本。
3. 核心思想：Harness 一级抽象、机制与实现分离、知识封装、默认入口与替换出口。
4. 具体方法：Memory / Context / MetaMessage / Session / Storage，及真实公共 API 示例。
5. 三组对比：应用与编排框架、完整 Harness 产品、模块化 Harness SDK。
6. 场景与边界：将优势落到变化任务，明确当前实现范围和选型条件。
7. 结论：把构建 Harness 变成局部可掌握的开发工作。

正文通过隐藏的 `role` 注释标记 opening、challenge、method、evidence、advantage、limitation 和 conclusion 段落组；正式阅读版不显示编辑标签。

## 2. 主张—证据映射

| Claim | Evidence | Status |
| --- | --- | --- |
| EasyAgent 的扩展入口直接对应关键 Harness 机制 | `BaseAgentExecutor`、`BaseMemory`、`BaseContextManager`、`BaseSessionStore` 与公开指南 | supported：限于核对的接口；不推广为所有子系统均完全解耦 |
| Memory 契约不强制一种分类、CRUD 或后端 | `memory/base.py` 的可选 prompt/tools/hooks，及独立 Storage | supported |
| Context 接受准备好的信息，不自行承担 Memory/RAG 检索 | Context 基类、请求准备路径与上下文指南 | supported |
| 会话恢复与产品实例装配分开 | SessionPersistence 与会话恢复指南 | supported：不意味着自动恢复外部任务 |
| 使用模块契约能够降低理解无关实现的需要 | 责任边界与装配示例提供结构解释 | needs evidence：实际学习时间与用户成功率未测量，正文保持价值主张措辞 |
| 模块替换减少维护与升级成本 | 局部接口给出合理机制解释 | needs evidence：未进行纵向维护研究，不报告节省比例 |
| EasyAgent 的方法在当前生态中具有可辨识的设计取向 | 三组对比与具体模块语义 | supported：设计取向；不声称首创或独占 |
| EasyAgent 全面优于其他 Harness SDK | 无充分证据 | removed：不进入正文 |
| S4Code 展示产品 Core 与框架的分离 | 当前工作空间产品结构、模块重构记录 | supported：单一具体案例，非跨领域效果实验 |

## 3. 首轮审校：事实与范围

- 比较覆盖 17 个项目/项目家族分组、19 个官方来源；LangChain 与 LangGraph 按生态关系归为一行。
- 将 Deep Agents、Pydantic AI Harness、OpenHands SDK、Cline SDK 作为直接比较对象，而不是只对比编排框架。
- 明确 OpenAI Agents SDK 的 Sandbox Agent 能力，不沿用“没有 Harness”这一过度简化。
- Microsoft 项目按 Agent Framework 与 AutoGen / Semantic Kernel 的继承关系描述。
- Gemini CLI 的旧 architecture URL 无法作为有效来源，改用官方仓库，删除未经核对的内部包结构断言。
- 不引用反编译来源，不将第三方评估数字转移到 EasyAgent。
- 不运行实验，不改框架或 S4Code 代码；本轮仅创建文档、插图和构建工具。

## 4. 第二轮审校：论证与阅读

已完成。补入第 3.5 节的四个结构优势，将抽象理念落为机制可选、变化局部、基础设施共享与产品边界清楚；把第 8.1 节重复的效果免责声明改写为具体修改落点。压缩了最接近竞品一节中的重复限制说明，保留范围边界和强基线。

通过反向提纲检查，正文各段组均服务于“领域变化为何需要模块契约”这条主线。正文的隐藏角色注释与本附件提纲保留，避免读者看到编辑标签。为独立转发增加参考资料索引。

## 5. 第三轮审校：交付与排版

已完成首两轮导出校正：第一版强制每节分页导致短尾页和大段空白，改为连续分页；修正移动端样式误作用于打印的问题，使封面恢复三栏。SVG 的多子路径箭头改为独立路径，使每条使用关系都有箭头。

自动检查覆盖全部本地引用与 Markdown 锚点、Python 示例语法、SVG 可访问描述、HTML 唯一锚点与三个内嵌图片。发现 Pandoc 默认图形转换丢失 HTML 图片 alt 后，关闭 implicit figures；将 PDF 目标锚点改为短 ASCII ID，消除 Poppler 对超长中文目标名称的警告。

人工检查了封面、目录、架构图、记忆图和比较表的渲染图；PDF 全部页面进行文本与空页检查。导出文件的验证结果由 `validate_whitepaper.py` 可重复生成。以上是文档质量检查，不是 Agent 效果或性能实验。

最终阅读版为 16 页，参考资料独立成页，包含 3 张内嵌 SVG 和 37 个目录/章节锚点。全部本地引用、示例语法、图形 XML、HTML 和 PDF 文本检查通过；仓库 diff 空白检查通过。读者可先查看 PDF，再通过 Markdown 修改正文。

## 6. 五维自检

1. **贡献是否具体？** 是：以模块契约与职责分离说明设计特色；不是抽象地宣称“模块化”。对首创性的主张已排除。
2. **表达是否清楚？** 使用统一的 Harness 定义，区分运行 API、构建契约和源码修改；通过真实接口与场景解释。
3. **实验是否支持效果？** 本版没有实验，因用户明确选择宣传白皮书。所有量化与优越性结论均未纳入。
4. **比较是否完整？** 覆盖三类主流路线，并包括最接近的模块化方案；不将概览级核对伪装成全面兼容性测试。
5. **方法是否夸大可靠性？** 已说明取消、副作用、外部任务恢复与 OS 沙箱边界；模块化不自动提供这些保证。
