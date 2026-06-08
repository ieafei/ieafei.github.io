# eagle speculator学习
## 描述
位置：vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
核心函数：propose()
外部调用：engine_core.model_executor.sample_tokens()->engine_core.speculator.propose()
流程：
Step N:
  speculator.propose()  →  生成 draft_tokens[0..K-1]，存入 req_states.draft_tokens

Step N+1:
  execute_model()
    └──prepare_inputs()
        └── combine_sampled_and_draft_tokens()
            ├── input_ids = [..., last_sampled, draft_0, draft_1, ..., draft_K-1]
            └── logits_indices = [每个 req 的 logits 位置]
    └── target_model.forward(input_ids)   ← 一次 forward 同时 verify 所有 draft tokens

  sample_tokens()
    └──  self.sample()->rejection.sampler(logits, speculator.logits)  ← 逐个验证，接受/拒绝
          └── speculator.propose()  →  生成下一轮 draft_tokens 

# eagle spec decode在model runner中的执行逻辑
上一轮已经生成的 100 个 token：tok[0..99]
上一轮 sample 出的 last_sampled：L（= tok[99]）
上一轮 propose 的 3 个 draft：d1, d2, d3
## 1.execute_model
input_ids:  [L,   d1,  d2,  d3]    （query_len = 4）
positions:  [99, 100, 101, 102]
KV 读:      slot 0~98（已有）
KV 写:      slot 99~102
output hidden: [h0, h1, h2, h3]    shape [4, hidden_size]
注意：L 在这一轮才正式参与计算并写入 KV——上一轮 sample 它的时候只是从 hidden 走了 lm_head 得到的 token，并没有让它作为 input 走过 forward。
## 2.rejection sample
位置 99 (h0): 验证 d1 → ✅ 接受
位置 100 (h1): 验证 d2 → ✅ 接受
位置 101 (h2): 验证 d3 → ❌ 拒绝，从 h2 重采样得到 t_bonus
位置 102 (h3): 作废

num_sampled  = 3      (d1, d2, t_bonus)
num_rejected = 1      (d3)
last_sampled = t_bonus     ← 写入 req_states.last_sampled_tokens
all_token_ids[100..102] = [d1, d2, t_bonus]    ← 提交到状态
num_computed_tokens 更新到 103
## 3.speculator.propose
  3.1 传参
    last_hidden_states = [h0, h1, h2, h3]   # 全传，shape [4, H]
    num_sampled        = 3
    num_rejected       = 1                  # 用这个让 kernel "逻辑剔除" h3
    last_sampled       = t_bonus
  3.2 kernel内左移input ids
    target input_ids:  [L,  d1, d2, d3]
                       ↓   ↓   ↓
    draft input_ids:   [d1, d2, ?, ?]      ← 左移 1 位
                                ↑
                      写入 t_bonus (last_sampled)
    draft input_ids:   [d1, d2, t_bonus, d3(遗留，无用)]
    hidden 不动:       [h0, h1, h2, h3]
    有效 query_len = 4 - num_rejected = 3
    last_token_index 指向位置 2（即 (t_bonus, h2)）
  3.3 draft model prefill整体forward
    input_ids:  [d1,    d2,    t_bonus,    d3残留]
    hidden:     [h0,    h1,    h2,         h3]
                ↓ embed(input_ids) ⊕ hidden  （EAGLE 的二元组拼接）
                ↓
    embed_in:   [E(d1)⊕h0,  E(d2)⊕h1,  E(t_bonus)⊕h2,  E(d3残)⊕h3]
                ↓ EAGLE 单层 transformer（含 self-attention + FFN）
                ↓ attention 还要把这 4 个位置的 K/V 写到 draft KV cache 的 slot 99~102
                ↓
    out:        [hd0,  hd1,  hd2,  hd3垃圾]      ← 4 个 hidden，物理 shape [4, H]
    只取最后一个有效位置的 hidden
    sample_hidden_states = last_hidden_states[last_token_indices]（只取hd2，不要hd3）
    用这一个 hidden 走 lm_head + sample，产生 d1_new
    self.draft_tokens[:num_reqs, 0] = self.sample_draft
    把 hd2 保存下来，给 multi-step decode 第 2 步用
    self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
  3.4 mutiple_step->_generate_draft，通过draft model计算出来的hd，推理出下一个token，不再
      是 target给的。这就是 EAGLE 自回归的关键——target 给的语义先验只在 step 0 用过一次，从 
      step 1 开始全靠 draft 自己
    

# 问题
## 为什么draft model要以token，hidden_state二元组作为输入去计算draft token？
1.为了避免重复计算，直接使用target model计算好的hidden states
2.draft model网络结构少，没有target model层数多，自然计算出来的隐藏层特征信息也不如target 计算好的hidden states
## 为什么draft model的输入是token[i+1], hidden_states[i]（错位拼接）？
因为draft model需要根据token[i+1]的embeding + target model计算的hidden_state[i]拼接一起后，自己计算出来hidden_state[i+1], 进而采样得到token[i+2]
'''
(token[i+1], hidden[i])  → FC → 1 层 transformer → draft_hidden[i+1] → lm_head → token[i+2]
        ↑           ↑
   实际的下一个    上一步的"压缩信息"
'''
## target model在execute model时计算的一次forward中存储的kv cache应该要在这些draft token确定accept后，才会真正写入存储吧？
核心结论：KV cache 在 forward 时就直接全部写入了，不会等 accept 结果，target model 的 forward 一旦执行完，所有 query 位置（包括后来会被拒的位置）的 K/V 都已经写入物理内存了。没有任何"等 accept 后才提交"的逻辑。
设计安全由下述3个原因保证：
1.accepted token 的 KV 是有效的
2.rejected token 的"垃圾 KV"会被下一轮直接覆盖
  input_ids:  [t_bonus, d1_new, d2_new, d3_new]
  positions:  [102, 103, 104, 105]
  slot 写入:   slot 102 (覆盖!), slot 103, 104, 105
3.在被覆盖之前，垃圾 KV 也读不到，num_computed_tokens 在 postprocess_sampled 之后被推进到 102（即"已提交的最后一个 token 是位置 101 的 d2"）。
下一轮 attention 的 seq_len 计算时，slot 102 不在合法可读范围内
attention 通过 seq_lens 控制可见范围，越界的 KV 永远不会被读到

## target model与draft model共用一个kv cache存储地址吗？
否，draft model 拥有自己独立的 KV cache 存储空间（eagle方法中）
┌────────────────── GPU 显存 ──────────────────┐
│                                                │
│  ┌─────────── target KV cache ──────────┐     │
│  │                                       │     │
│  │  [num_blocks, 32 layers, 2, ...]      │     │
│  │  ↑ target.forward 写入                │     │
│  │  slot 0..max_model_len 共用同一块      │     │
│  └───────────────────────────────────────┘     │
│                                                │
│  ┌─────────── draft KV cache ───────────┐     │
│  │                                       │     │
│  │  [num_blocks, 1 layer, 2, ...]        │     │
│  │  ↑ draft.forward 写入（独立！）         │     │
│  │  跟 target 完全不同的物理内存           │     │
│  │                                       │     │
│  │  prefill 写: slot 99~102              │     │
│  │  step=1 写:  slot 102（覆盖 d3残）     │     │
│  │  step=2 写:  slot 103                  │     │
│  └───────────────────────────────────────┘     │
│                                                │
│  共享 block_tables（逻辑映射表）                │
│  draft / target 各自的 attention layer 通过      │
│  自己的 kv_cache 张量指针访问对应物理内存        │
└────────────────────────────────────────────────┘

## seq_lens变量是什么含义？
seq_lens[i] = 第 i 个 req 在本次 attention forward 时，可以从 KV cache 中读取的有效 K/V 长度（包括本次新写入的 query 部分）。
seq_lens[i] = num_computed_tokens[i] + query_len[i]
            = "已经在 KV cache 里的旧 token 数" + "本次新写入的 token 数"

## slot mapping是什么？
slot_mapping[i] = 本次 forward 中第 i 个 token 在 KV cache 中应该写入（或读取）的物理 slot 编号。




