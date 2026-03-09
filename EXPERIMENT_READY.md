# 实验准备完成 - Tool Usage Tracking

## ✅ 新功能已添加

### **工具调用统计（Tool Usage Summary）**

两个 Agent 的输出中都新增了 `tool_usage_summary` 字段，显示每个工具的调用次数。

#### **BaseAgent 输出示例**
```json
{
  "answer": "答案内容...",
  "trajectory": [...],
  "total_cost": 0.05,
  "loops": 5,
  "tool_usage_summary": {
    "keyword_search": 2,
    "semantic_search": 1,
    "read_chunk": 3
  },
  "total_retrieved_tokens": 5000,
  "chunks_read_count": 3,
  ...
}
```

#### **MemoryAgent 输出示例**
```json
{
  "answer": "答案内容...",
  "trajectory": [...],
  "total_cost": 0.08,
  "loops": 10,
  "tool_usage_summary": {
    "memory_update": 9,
    "final_answer": 1
  },
  "total_retrieved_tokens": 8000,
  "chunks_read_count": 9,
  "memory_state": {
    "final_memory": "...",
    "history_size": 9
  },
  ...
}
```

## 📊 样例数量更新

所有文档中的测试样例数量已从 **10** 更新为 **100**。

### **运行命令（已更新）**

#### Base Agent
```bash
python scripts/batch_runner.py \
    --config configs/example.yaml \
    --questions data/musique/questions.json \
    --output results/base \
    --agent-type base \
    --limit 100 --workers 5
```

#### Memory Agent
```bash
python scripts/batch_runner.py \
    --config configs/memory_example.yaml \
    --questions data/musique/questions.json \
    --output results/memory \
    --agent-type memory \
    --limit 100 --workers 5
```

## 🔬 实验分析价值

有了 `tool_usage_summary`，你可以：

### 1. **工具使用模式分析**
```python
# 分析 predictions.jsonl
import json

tool_stats = {}
with open('results/base/predictions.jsonl') as f:
    for line in f:
        result = json.loads(line)
        usage = result['tool_usage_summary']
        for tool, count in usage.items():
            tool_stats[tool] = tool_stats.get(tool, 0) + count

print("Base Agent 工具使用统计：")
for tool, total in sorted(tool_stats.items()):
    print(f"  {tool}: {total} 次")
```

### 2. **效率对比**
- Base Agent: 平均使用多少次工具？
- Memory Agent: 平均处理多少个 chunks？
- 哪个更高效？

### 3. **策略分析**
- Base Agent 偏好哪种搜索？(keyword vs semantic)
- 平均阅读多少个 chunks 才能回答？
- Memory Agent 的记忆更新频率？

### 4. **成本分析**
```python
# 工具使用 vs 成本关系
for line in predictions:
    tools_used = sum(result['tool_usage_summary'].values())
    cost = result['total_cost']
    print(f"工具调用: {tools_used}, 成本: ${cost:.4f}")
```

## 🧪 快速验证

```bash
# 运行测试验证功能
cd d:\arag_mem
$env:PYTHONPATH="src"
python tests/test_tool_usage.py
```

## 📝 修改文件列表

### 核心代码修改
- ✅ `src/arag/agent/base.py` - 添加 `_calculate_tool_usage()` 和输出字段
- ✅ `src/arag/agent/memory_agent.py` - 添加 `_calculate_tool_usage()` 和输出字段

### 文档更新
- ✅ `MEMORY_INTEGRATION.md` - 更新样例数和输出格式
- ✅ `src/arag/core/memory/README.md` - 更新样例数
- ✅ `README.md` - 更新样例数

### 测试文件
- ✅ `tests/test_tool_usage.py` - 新增工具统计测试

## 🚀 现在可以开始实验了！

所有准备工作已完成：
1. ✅ 工具调用统计已添加
2. ✅ 样例数量已设置为 100
3. ✅ 测试已通过
4. ✅ 文档已更新

**立即开始实验：**
```bash
# 设置 API Key
export ARAG_API_KEY="your-key"

# 运行对比实验（100个样例）
python scripts/batch_runner.py --config configs/example.yaml \
    --questions data/musique/questions.json \
    --output results/base --agent-type base --limit 100 --workers 5

python scripts/batch_runner.py --config configs/memory_example.yaml \
    --questions data/musique/questions.json \
    --output results/memory --agent-type memory --limit 100 --workers 5

# 评估
python scripts/eval.py --predictions results/base/predictions.jsonl --workers 5
python scripts/eval.py --predictions results/memory/predictions.jsonl --workers 5
```

**祝实验顺利！** 🎉
