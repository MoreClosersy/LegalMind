# LegalMind 工程笔记

这份文档记录**为什么这样做**，而不是**做了什么**。代码本身说明后者。

每一条都对应仓库里一个具体文件，以及一个我们实际踩过或测出来的结论。面试里被追问细节时，能答出来的就是这些。

---

## 0. 当前进度

| 阶段 | 状态 |
|---|---|
| 语料流式采样（pile-of-law 三个来源） | ✅ `src/legalmind/data/corpus.py` |
| 合成提示词 + 结构化输出 schema | ✅ 经两轮实测修正 |
| Batch API 合成驱动 | ✅ 支持 `--limit` / `--resume` / `--dry-run` |
| 质量过滤 + 两级去重 | ✅ `filter.py`，13 个测试 |
| LegalBench 13-gram 去污染 | ✅ `decontaminate.py` |
| 拒绝校准数据集构建器 | ✅ `refusal_seeds.py` + `refusal_set.py`，30 组配对主题 |
| 服务层免责声明强制 | ✅ `serve/disclaimer.py`，含 SSE 流式尾部处理 |
| QLoRA 训练脚本 + 掩码验证 | ✅ `train/sft.py` + `scripts/verify_masking.py`（进了 CI） |
| 三臂评测脚手架 | ✅ `eval/arms.py` + `eval/refusal.py` |
| **全量合成批次** | 🔄 `msgbatch_01Ep4W9y8QcQ4DuREuiQqBAv`，2070 请求在跑 |
| LegalBench 评测 harness | ⬜ |
| 红队集 / 遗忘检查 / LLM-as-judge | ⬜ |
| vLLM 网关 + LoRA 热插拔 + SSE | ⬜ |
| 延迟基准 / Docker / HF Spaces | ⬜ |

80 个测试通过，ruff + mypy + pytest 全绿，CI 里还会跑 10 步真训练。

---

## 第一部分：原计划的三个致命方法论问题

这部分最值钱。工程问题能改，方法论问题会让整个项目的数字作废。

### 1. 评测基准不能当训练集

**原计划**：用 `Equall/legalbench_instruct` 训练，再用 LegalBench 的 held-out split 报成绩。

**为什么致命**：LegalBench 是**评测基准**。训练完再用它评测，叫 train/test contamination。面试官第一句就是 "What did you evaluate on?" —— 答"同一个数据集切出来的验证集"，你所有数字全部作废，而且暴露的是判断力问题，比"没做出效果"严重得多。

**改成**：

```
训练数据 ← pile-of-law（原始法律文本，自己合成 instruction）
评测数据 ← LegalBench（训练侧零接触）
中间 ← 13-gram 去污染脚本，把删掉的条数写进报告
```

**可迁移的判断标准**：拿到任何数据集先问"它是 benchmark 还是 training set"。名字里带 Bench / GLUE / Eval / MMLU 的基本都是前者。再问"我要报的那个数字，训练时见过它吗"。

去污染实现见 `src/legalmind/data/decontaminate.py`。一个细节：13-gram shingle 用 **blake2b** 哈希，不用 Python 内置 `hash()` —— 后者带进程级随机盐，同一份数据两次跑出的报告数字会不一样，那报告就没有溯源价值了。

### 2. 概率性保证 vs 确定性保证

**原计划**：把"UPL 免责声明微调进权重"当成卖点。

**为什么错**，四条：

- **概率性** —— 分布外输入、对抗提示、少见问法都可能让模型漏掉
- **不可审计** —— 你没法指着一组权重跟合规审查员解释"它一定会说"
- **不可版本化** —— 改一个字要重训一遍
- **浪费容量** —— 一句字符串拼接能保证的事，花模型参数去学

**正确的分工**：

| 要保证什么 | 用什么 | 为什么 |
|---|---|---|
| 免责声明必须出现 | **确定性代码**（`serve/disclaimer.py`） | 100% 保证，可单测，可热更新，带版本号 |
| 判断该拒绝还是该回答 | **微调模型** | 没有任何后处理规则能做这个判断 |

一句话立场：**规则做规则能保证的，模型做只有模型能判断的。**

配套的一个过滤器决策：`filter.py` 会**主动拒绝**训练数据里含免责声明的样本（`disclaimer_leaked`）。因为一旦模型学会自己说免责声明，服务层的确定性保证就退化成"模型大概会说 + 代码兜底"，反而模糊了责任边界。

### 3. 对比实验必须有真对手

**原计划**：只对比 base zero-shot vs fine-tuned。

**为什么这是被操纵的对比**：裸 base 零样本 = 没好好写提示词的 base。你测的是"提示词写得烂"，不是"微调有用"。

**三臂设计**（`src/legalmind/eval/arms.py`）：

```
A  base，零样本，朴素指令
B  base + 精心写的 system prompt + few-shot     ← 真正的对手
C  fine-tuned（LoRA adapter）
```

**很多人会漏的关键细节**：arm B 必须**带上和微调同样的拒绝策略文本**。不带的话，arm C 在拒绝校准指标上不战而胜 —— 那是另一种作弊，只是更隐蔽。

**如果 C 只比 B 好一点点怎么办？** 如实报，然后讲清成本模型：B 每次请求都要付那段长提示词和范例的 token 成本和首 token 延迟，**永远付**；而且有些部署场景带不走提示词（第三方集成、上下文预算紧张）。

**诚实的微弱提升 + 清醒的成本分析 > 漂亮但可疑的数字。**

---

## 第二部分：工程操作与踩坑

### 4. 不要凭记忆写 API，装了去查

我按记忆写了 `assistant_only_loss`。实际装上 TRL 0.18.1 查一下：

```bash
uv run python -c "
from trl import SFTConfig
fields = set(SFTConfig.__dataclass_fields__)
print('assistant_only_loss:', 'assistant_only_loss' in fields)
print('completion_only_loss:', 'completion_only_loss' in fields)
"
```

结果：**没有** `assistant_only_loss`，只有 `completion_only_loss`，而且它要求 `prompt` / `completion` 两列的数据格式，和 `formatting_func` 互斥。

**教训**：写代码前先 `inspect` 实际安装的版本。30 秒的查询省掉几小时调试。同类问题在这个项目里出现了两次（另一次见第 6 条）。

### 5. 数据集加载的现实

三个连环坑：

```python
# 坑 1：datasets 3.x 移除了脚本式 dataset 支持
load_dataset("pile-of-law/pile-of-law", "cfr")   # 直接失败

# 坑 2：分片名和文档不一致
"data/train.courtlisteneropinions.0.jsonl.xz"    # 没有下划线，写错就 404

# 坑 3：一条记录是整部 CFR Title
len(record["text"])   # 2,878,196 字符
```

**解法** —— HTTP 流式 + 增量解压，只读前几 MB（`corpus.py`）：

```python
decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
buffer = b""
with requests.get(url, stream=True) as response:
    for compressed in response.iter_content(chunk_size=1 << 20):
        buffer += decompressor.decompress(compressed)
        *complete_lines, buffer = buffer.split(b"\n")
        for line in complete_lines:
            yield json.loads(line)
        if decompressor.eof:
            break
```

1GB 的分片实际只下载几 MB。坑 3 另外要求写一个 `chunk_record()`，按 `§`/`Sec.`/`PART`/`Subpart` 边界切成 2000–8000 字符的段，再用 `_looks_substantive()` 过滤掉目录页（判据：平均行长 + `". "` 出现次数）。

分片名这种"写错就 404、而且要等到运行时才知道"的常量，我用测试钉死了 —— 见 `tests/test_corpus.py`。

### 6. Qwen3 chat template 的真相

我实测出来的，和直觉不一样：

```python
tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
# '...<|im_start|>assistant\n'

tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
# '...<|im_start|>assistant\n<think>\n\n</think>\n\n'

# 但完整对话（不加 generation prompt）两种模式输出【完全相同】
# 模板总会插入空的 <think>\n\n</think>\n\n
```

**结论**：`enable_thinking` **只影响 generation prompt**，不影响完整对话的渲染。

**设计后果**（`train/sft.py`）：自己渲染 prompt（`enable_thinking=False`），让那个空 think 块留在 **prompt 侧**，模型永远不用生成它，训练和推理完全对齐 —— 前提是服务端也设 `enable_thinking=False`。所以这不是个可选配置，是一个**必须成对设置的约束**，我在配置文件里写了注释说明。

### 7. Completion-only loss masking（含金量最高的细节）

**大多数人训练时在 prompt token 上也算了 loss，而且不自知** —— 模型在浪费容量学习预测自己的输入。

这个错误**完全静默**：loss 照样下降，训练照样收敛，只是效果差一截，你还以为是超参没调好。

所以有 `scripts/verify_masking.py`，进 CI：

```
masked (no loss): 34/119 tokens
trained (loss):   85/119 tokens
prompt tail    -> ' adequate mitigation?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
completion head -> 'The agency must make an affirmative finding on each condition the regulation'
OK: loss is computed on the assistant completion only.
```

它在五种情况下退出非零：`completion_mask` 不存在、没有任何 token 被掩码、掩码的 prompt 尾部不含 `<|im_start|>assistant`、`enable_thinking=False` 时缺 `</think>`、控制 token 漏进了被训练的 span。

**可迁移的原则**：**静默失败的东西必须显式验证**。会抛异常的问题不用管，不报错但让结果变差的才要写检查。这一条决定了这个项目里哪些东西有测试、哪些没有。

---

## 第三部分：这个项目最重要的方法论 —— 测量优先于假设

三次实例，全都是"读代码看不出来、必须跑真实数据才能发现"。

### 8a. 31.6% 的悬空引用

跑完 99 条真实数据后我去读内容，发现回答里写着：

> "the passage does not commit the agency to..."

我的自包含检查**只查了 instruction**。而模型的实际行为恰恰是：**问题写得自包含，答案回头引用原文**。训练出来的模型永远看不到那段原文，这些回答等于在指空气。

读代码看不出来，读提示词也看不出来 —— 只有看真实输出才看得见。

修复是两层的（这点也值得学）：

1. **提示词改**：明确写"自包含要求对回答同样成立，而且回答才是出问题的地方"，并给出反例句式，最后加一句"把每条回答当成你从没见过那段原文再读一遍"
2. **过滤器兜底**：`_DANGLING_REFERENCE` 正则

但加了正则之后立刻有个新风险：**过滤器会不会把提示词要求的正确写法也杀掉？** 正确写法是"指名条款而不是指向原文"（"Section 207 does not commit..."）。所以 `tests/test_filter.py` 里有一个专门的测试 `test_naming_the_authority_is_not_a_dangling_reference`，断言修复后的写法能存活。

**这是个通用模式：加过滤器时，同时写一个测试断言"你想要的东西不会被误杀"。** 否则你只是把问题从"模型写错"换成"好数据被丢掉"，而后者更难发现。

### 8b. 任务类型失衡

提示词里明明写着"要变换类型，不要都是同一种"。实测：

```
statutory_interpretation   4%
其余三类                    各约 32%
```

而且**三个来源一致** → 是模型偏好，不是语料问题。

**用散文要求分布，模型不会照做。** 改成按请求索引轮转、强制指定类型：

```python
def required_task_type_for(index: int) -> str:
    """Rotate the required type across a batch so coverage is deterministic."""
    return TASK_TYPES[index % len(TASK_TYPES)]
```

分布从"模型的倾向"变成"批次的确定性属性"。

**诚实的边界**（写在 `eval_results/synthesis_prompt_ab.json` 的 caveats 里）：轮转设的是**下限不是配额** —— 每个请求只强制其中一条是指定类型，剩下两条仍随模型偏好，所以份额从 4% 升到 12.5%，不是 25%。把这条写进结果文件，比事后被问住强。

### 8c. "有没有现成数据集"

你问的时候我第一反应是复述之前的结论。正确做法是去测：

```
israelfama/lawinstruct_US_jurisdiction （抽样 3,876 行）
  英文 instruction 占比：14.4%     ← 多语言数据集，抽到的头两条是希腊语和罗马尼亚语
  任务类型：52% 分类 / 21% NLI / 10% NER / 17% 问答
  18 个源数据集里 4 个 → 污染 LegalBench 的 54/162 个任务
```

三个条件叠加后，剩下几百条格式还不对的样本（分类标签，不是解释性问答）。而拒绝校准数据**根本没有公开等价物**。

**这次的结论和上次一样，但这次有证据。** 这就是区别 —— 面试时"我评估过现成数据集，不合适"和"英文只占 14.4%、格式是分类不是问答、4/18 个源污染 54 个任务"是两个量级的回答。

---

## 第四部分：统计与指标设计

### 9. 置信区间不是装饰

n=72 的时候朴素百分比会骗人。用 Wilson 区间（`eval/refusal.py`）：

```python
def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
```

为什么不用朴素的 `p ± 1.96·√(p(1-p)/n)`：p 接近 0 或 1 时它会**跑出 [0,1] 区间**，而且覆盖率严重不足。`wilson_interval(10, 10)` → `[0.72, 1.0]`：**10/10 不等于"100% 可靠"，下界只有 72%。**

`scripts/compare_synthesis.py` 会自动判断并直说：

```
-> intervals separated                                    # 真差距
-> intervals OVERLAPPING — not a demonstrated difference  # 看着有差距，其实不是
```

工具替你守住"不许把噪声当结论"这条线，比靠自觉可靠。

### 10. 配对指标 —— 防止单指标作弊

拒绝率单独报是**没有意义的**：一个见什么都拒绝的模型能拿 100%。

所以永远成对报：

| 指标 | 数据集 | 方向 |
|---|---|---|
| refusal rate | should_refuse | 越高越好 |
| **over-refusal rate** | should_answer | **越低越好** |

`tests/test_refusal_eval.py` 里有一个测试专门断言这种作弊会被抓到：

```python
def test_refusing_everything_is_visibly_bad():
    metrics = score(labels, [True] * 10, arm="always-refuses")
    assert metrics.refusal_rate.value == 1.0
    assert metrics.over_refusal_rate.value == 1.0   # 一般问题也全被拒了
```

这也是为什么 `refusal_seeds.py` 里 30 个主题**每个都是配对的** —— 同一法律领域，一个第一人称个案问法（该拒），一个非人称一般问法（该答）。不配对的话，拒绝率和过度拒绝率会被主题差异污染，你分不清模型是在判断"个案 vs 一般"还是在判断"劳动法 vs 合同法"。

**通用模式：任何一个可以靠极端策略刷满的指标，都必须有一个反向配对指标。**

### 11. 按主题切分，不按样本切分

```python
def split_topics(pairs, *, eval_fraction=0.3, seed=42):
    """Hold out whole topics for evaluation.

    Splitting by example would put a paraphrase of a training question
    into the eval set and inflate the calibration numbers."""
```

按样本切 → 训练问题的改写版落进评测集 → 数字虚高。

这和 LegalBench 污染是**同一类错误的不同表现**：评测集里出现了训练时见过的东西。只是一个是跨数据集，一个是数据集内部。养成的习惯应该是每次划分都问一句"这两边有没有共享的东西"。

### 12. 教师 ≠ 裁判

合成数据用 Sonnet 5，LLM-as-judge **不能**也用 Sonnet 5 —— 那是 self-enhancement bias，模型偏爱自己风格的输出。裁判要换一个模型，而且要报**裁判和启发式规则的一致率**；一致率低于 85% 时，`refusal.py` 会自动在结果里打上 `provisional` 标记。

**裁判的可信度本身也需要证据。**

---

## 第五部分：成本工程

### 13. 先找到成本驱动因素，再优化

第一次探针的实测数据：

```
input  279,306 tok（uncached 161,694 / cache-write 7,128 / cache-read 110,484）
output 177,447 tok
→ output 占成本 83%
```

**结论**：输入侧优化（缩短 passage、prompt caching）省不下多少钱。**条数和每条长度才是杠杆。**

这个结论改变了后续所有决策 —— 我不再纠结输入优化，直接讨论"要多少条"。

**通用做法：优化之前先测成本构成。** 凭直觉优化的往往是占比最小的那部分。

### 14. Prompt caching 的隐藏门槛

Haiku 探针出现了 `cache-write 0 / cache-read 0`，完全没缓存。

原因：**不同模型的最小可缓存前缀不同**。

| 模型 | 最小可缓存前缀 |
|---|---|
| Sonnet 5 | 512 token |
| Haiku 4.5 | **4096 token** |

我们的系统提示词约 1k token —— Sonnet 能缓存，Haiku 根本够不到门槛。

**教训**：换模型时缓存行为可能整个失效，**而且不会报错**，只会在账单上体现。

这也是为什么系统提示词里**不能插值任何 per-request 内容** —— 缓存靠前缀精确匹配，插一个变量进去，整个前缀作废。passage 必须放在 user turn，也就是 `cache_control` 断点之后。

### 15. 小批次的成本数字会骗人

| 批次大小 | cache-read 占比 |
|---|---|
| 294 请求 | 94% |
| 24 请求 | 42% |

小批次里 cache-write 占主导，单位成本虚高。所以 `compare_synthesis.py` **故意不比成本** —— 只比质量指标，选出赢家后再按真实批量定价重新算。文档字符串里写明了这个决定和原因。

**通用陷阱：小样本上测出的单位成本不能外推**，凡是有固定开销摊销的场景都一样。

### 16. 估算器要往高了报

```python
# 用标准价而非优惠价，故意保守
PRICING = {"claude-sonnet-5": (3.00, 15.00)}   # 优惠价其实是 2/10
```

注释里写清楚了理由：*"an estimate that under-reports is worse than useless"*。低报的预算估算会让你在钱花完的那一刻才发现。

### 17. 断点续跑要在花钱之前就落盘

```python
batch_id = submit(client, build_requests(...))
# Persist immediately: if polling dies, --resume picks up from here
# rather than paying for the whole batch twice.
Path("data/.last_batch_id").write_text(batch_id)
```

提交完立刻写文件，不等轮询结束。轮询挂了、终端关了、机器睡了都不影响 —— `--resume` 接得上。这个文件在 `.gitignore` 里（它是运行时游标，不是代码）。

Batch API 另一个必须记住的细节：**结果的返回顺序是任意的**。所有查找必须按 `custom_id` 键，绝不能按位置。按位置写的代码在小批次上可能碰巧能跑，在大批次上会静默错配。

---

## 第六部分：安全与工程规范

### 18. `.env` vs `.env.example`

```
.env           ← 真实密钥，在 .gitignore 里
.env.example   ← 模板，值留空，会被提交
```

你把 key 填进了 `.env.example`。如果直接 `git add -A && git push`，密钥就进公开仓库了。

**推送前的检查动作**（养成肌肉记忆）：

```bash
grep -rn "sk-ant-\|AKIA\|-----BEGIN" . --exclude-dir=.venv --exclude-dir=.git --exclude=.env
```

```bash
git diff --cached | grep -c "sk-ant-"
```

顺带解释你那个报错：`load_dotenv()` **只读 `.env`**，不读 `.env.example`。所以密钥填错地方的直接症状就是 `TypeError: Could not resolve authentication method`。

（密钥没有进 GitHub，但既然它在本地明文文件里出现过一次，轮换一下是便宜的保险。）

### 19. 每个数字都要可溯源

`.gitignore` 里有一条注释：

```
# NOTE: eval_results/ is deliberately NOT ignored. Every number in the README
# must be traceable to a committed raw result file.
```

原计划的坑里写着"旧简历上有过虚构指标"。防止复发的机制不能是"这次我保证真实"，得是**结构性的**：

- 所有评测结果 JSON 提交进仓库，带 batch_id
- README 里每个数字标注复现命令
- 三臂对比逼你面对真实差距，没法只报好看的那个

**自律不是机制，机制才是机制。**

### 20. CI 里跑真训练

```yaml
- name: Train 10 steps on committed fixtures
  run: uv run python -m legalmind.train.sft --config configs/train_smoke_0.6b.yaml
- name: Verify completion-only loss masking
  run: uv run python scripts/verify_masking.py --config configs/train_smoke_0.6b.yaml
```

用 committed fixture（不需要网络、不需要 API key、不需要 GPU）在 CPU 上跑 10 步。

**跑不完 10 步的训练脚本就是坏的。** 在 CI 花 4 分钟发现，好过在 GPU 实例上跑了 3 小时才发现。

### 21. 把重复三次的分析变成工具

同一个 A/B 对比我手工做了三次之后，写成了 `scripts/compare_synthesis.py`。

**收益不只是省时间** —— 工具化之后立刻暴露了一个手工分析漏掉的事实：旧提示词下 `statutory_interpretation` 过滤后只剩 **0.9%**（原始 4%）。因为解释性推理天然要回指原文，被悬空引用过滤打击得最重。手工看的时候我只看了原始分布，没看过滤后分布。

**手工分析会漏掉你没想到要看的维度；工具每次把所有维度都打出来。**

### 22. 顺序是有语义的

`filter.py` 里内容检查排在长度检查**之前**。这不是风格问题：

"Can I sue my employer for this?" 只有 30 字符，如果先跑长度检查，它会被归类成 `instruction_length`，而它真正的问题是 `personal_advice`。分类错了，过滤报告里的拒绝原因分布就是错的，而你会拿这个分布去改提示词。

这个 bug 是**测试发现的**，不是读代码发现的。

**通用原则：当一个东西可能命中多条规则时，规则顺序决定了你看到的诊断信息。把最有信息量的规则放前面。**

---

## 第七部分：面试话术

按重要性排序。每一条的共同点是**都有数字，且数字有出处** —— 这是"做过"和"看过教程"的区别。

1. **"你怎么保证评测干净"**
   → LegalBench 训练侧零接触 + 13-gram 去污染脚本 + 报告里的删除条数。哈希用 blake2b 不用内置 `hash()`，否则报告不可复现。

2. **"你的对比公平吗"**
   → 三臂。arm B 是精心调过的 base，而且带上和微调**同样的拒绝策略文本** —— 不带的话 arm C 在校准指标上不战而胜。

3. **"合规怎么保证"**
   → 服务层确定性强制 + 微调只负责 refusal calibration。规则做规则能保证的，模型做只有模型能判断的。免责声明带版本号，训练数据里含免责声明的样本会被主动过滤掉。

4. **"你怎么知道没在 prompt token 上算 loss"**
   → CI 里有个脚本断言损失边界，因为掩码失败是**静默的** —— loss 照降，只是效果差一截。

5. **"数据哪来的，为什么不用现成的"**
   → 测过：LawInstruct 美国法子集英文 instruction 只占 14.4%（多语言数据集）、52% 是分类不是解释性问答、18 个源里 4 个污染 LegalBench 的 54/162 个任务。拒绝校准数据没有公开等价物。

6. **"提示词你怎么调的"**
   → 跑真实批次测出 31.6% 悬空引用和 4% 类型失衡，改完再 A/B，Wilson 区间分离才算数。散文要求分布模型不照做，改成按请求轮转强制指定。

7. **"为什么用 Sonnet 不用 Haiku"**
   → 测过：保留率 87.5% [78.5, 93.1] vs 100% [94.9, 100]，区间分离；10 条被拒**全是**悬空引用，说明它对自检指令的遵循更弱。省钱要省在条数上，不是每条的质量上。

8. **"为什么不用 Spot 实例"**
   → 账户 Spot 配额是 0（`L-3819A6DF`，和 On-Demand 的 `L-DB2E81BA` 是两个独立配额），而且 3.5 小时只省约 $2.5。checkpoint + S3 同步照留 —— 它防的是任何中断，不只是 Spot 回收。

---

## 附：这个项目里可复用的思维模式

抽掉法律和微调的语境之后，剩下的是这些：

| 模式 | 这里的实例 |
|---|---|
| 分清评测集和训练集 | LegalBench 零接触 |
| 分清确定性保证和概率性保证 | 免责声明在代码里，拒绝判断在模型里 |
| 对比要有真对手 | 三臂，arm B 带同样的策略 |
| 静默失败必须显式验证 | CI 里的掩码检查 |
| 测量优先于假设 | 三次都推翻了我的先验 |
| 小样本的百分比要带区间 | Wilson，10/10 的下界是 72% |
| 可刷满的指标必须配对 | refusal / over-refusal |
| 优化前先测成本构成 | output 占 83%，输入侧优化白费 |
| 加过滤器要同时测"不误杀" | 指名条款的写法必须存活 |
| 规则顺序决定诊断质量 | 内容检查在长度检查之前 |
| 重复三次的分析要工具化 | 工具立刻发现了手工漏掉的 0.9% |
| 自律不是机制 | eval_results/ 强制提交 |
