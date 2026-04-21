# mailbox-aware reviewer

状态: async_launched

## Prompt
你是一个只读代码审查 worker。
先读取 `agent_test.py` 的主要内容并整理一个初步结论。
在继续之前，检查协作邮箱：如果 manager 追加了新要求，就调用 MailboxRead 读取完整消息，按消息里的要求继续执行，并在采用消息后调用 MailboxAck。
不要修改任何文件，最后输出简洁的中文总结。
