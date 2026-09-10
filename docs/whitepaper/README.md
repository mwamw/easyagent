# EasyAgent 技术白皮书

中文技术白皮书 v0.3（框架机制与扩展实践版），2026-09-07。用途是产品定位与架构传播，不包含对照实验或量化效果结论。

- [Markdown 正文](easyagent-whitepaper.zh-CN.md)：后续编辑的唯一正文来源。
- [HTML 阅读版](easyagent-whitepaper.zh-CN.html)：离线图片内嵌，可用浏览器阅读。
- [PDF 转发版](easyagent-whitepaper.zh-CN.pdf)：27 页 A4 排版，适合发送和打印。
- [来源与对比口径](comparison-sources.md)：26 个官方来源与本地实现依据。
- [编辑审校记录](editorial-review.md)：主张—证据映射、段落职责与修改记录。
- `assets/`：五张可编辑 SVG，使用 Noto Sans CJK SC 字体优先渲染。
- [v0.1 历史 PDF](versions/v0.1/easyagent-whitepaper.zh-CN.pdf)：保留最初版本，便于比较修改。
- [v0.2 历史 PDF](versions/v0.2/easyagent-whitepaper.zh-CN.pdf)：保留第三、四章重写前的版本。

## v0.3 新增内容

第三章补上框架内部图和安装/运行机制图，解释默认实现、职责契约与自定义实现的关系。第四章改为开发者实践：工具、提示词、记忆、压缩器、会话存储，以及最终产品装配，逐项说明用户写什么、框架接管什么、哪些约定仍需用户遵守。

现有十张比较/状态表保持不变，另加一张模块分工表。七段代码可通过 `validate_examples.py` 运行离线接口检查；使用假模型响应和内存数据，不访问模型 API。原产品分层图移至 S4Code 案例。

## v0.2 新增内容

新增六张表，正文共十张表：十二维定位总览，记忆/上下文/持久化接口对照，产品运行与构建接口对照，两组最接近的 Harness SDK 对照，以及六种领域定制需求的修改入口。保留 17 个项目或项目家族的覆盖范围，不把资料缺口标成不支持。

推荐先读第 5.1 节，再读第 5.3、6.3、7.2、7.3 和 8.5 节。HTML 在窄屏可横向滚动宽表，PDF 使用重复表头与 EasyAgent 列强调。

## 重建阅读版

需要已安装的 Pandoc；PDF 还需要 Chrome/Chromium。脚本不联网安装依赖，不调用模型。

```bash
cd EasyAgent/docs/whitepaper
python build_whitepaper.py
python build_whitepaper.py --pdf
python validate_whitepaper.py
python validate_examples.py
```

脚本会重新生成本目录同名 HTML/PDF，修改前请以 Markdown 为准；若要保留旧版导出物，请先另存版本文件。正文中的本地技术指南链接需要同时保留仓库文档；外部引用需要联网访问。无需依赖外部网站即可显示正文及全部插图。

本版采用三组定性比较：应用/编排框架、完整 Harness 产品、模块化 Harness SDK。没有使用“唯一”“全面领先”或没有来源的功能打分。未来若更新框架发行版或竞品资料，请同步修改核对日期、来源说明和实现状态。
