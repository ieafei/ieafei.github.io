# scheduler.schedule()内部
来源：sched/interface.py抽象函数接口schedule->子类scheduler实现schedule

任务：安排在此调度步骤中需要处理的请求，生成request_id:num_tokens结果，即每个请求对应的需要生成的token数目
    ->new reuqest id: len(prompt)
    ->auto regressively reuqest id: 1
    ->spec decoding, chunked prefill, prefix caching: (1, len(prompt))

每个请求reuqest_id仅包含两个属性：num_computed_tokens（已计算的 token 数量）和 num_tokens_with_spec（包含spec token 在内的总 token 数量）
其中，num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
在每一个调度步中，调度器都会尝试为各请求分配 token，以确保每个请求的 num_computed_tokens 能够追赶上其num_tokens_with_spec

# 流程：
## A.for each running request:
    1.判断当前request是否已经达到tokens上限？
    '''
        request.num_computed_tokens + 2 - request.num_output_placeholders
                    >= request.num_prompt_tokens + request.max_tokens
            continue
    '''
        同步调度：上一步 forward 完成 → token 已经追加到 output_token_ids → scheduler 看到真实数量
        异步调度：num_output_placeholders：上一步 forward 还没完成 → 用 num_output_placeholders 表示"还在飞行中、即将产生但尚未到达"的 token 数
        为什么是num_computed_tokens + 2 - request.num_output_placeholders（spec token全部拒绝，最差情况下token长度）：
        等于(num_computed_tokens + 1) - (num_output_placeholders - 1)
        num_computed_tokens + 1：前一步 forward 完成时，至少会产生 1 个真实 sampled token（即使所有 draft token 都被拒绝，主模型自己还是会采样出 1 个 token）
        (num_output_placeholders - 1)：num_output_placeholders 中最多有 num_output_placeholders - 1 个是 draft token（剩 1 个是必中的 sampled token）
    
    2.计算num_new_tokens
    '''
    num_new_tokens = (
                    request.num_tokens_with_spec
                    + request.num_output_placeholders
                    - request.num_computed_tokens
                )
    '''
        约束1: 不能超过最长prefill_token_threshold
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

        约束2：不能超过当前调度步骤中，所有请求加一起的最多被调度生成的token总数上限，给当前request分配后，更新token_budget
            num_new_tokens = min(num_new_tokens, token_budget)
            token_budget -= num_new_tokens
        约束3：确保加上num_computed_tokens不能超过max_model_len(特别是spec decode一次生成多个num_new_tokens，超过长度就要截断)
            spec decoding 一次性追加 N 个草稿 token（N 通常是 3~5），在接近 max_model_len 边界时，这 N 个 token 很容易一下子"冲出"边界
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)

    if num_new_tokens == 0：
        continue

    3.给每个num_new_token != 0 的running req分配kv cache blocks











