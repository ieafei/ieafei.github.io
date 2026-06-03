# 记录llm.generate()总体流程

1.call llm.generate(prompts, sampling_params)
2.call llm._run_completion（继承自OfflineInferenceMixin功能模块类，不实例化，只提供函数接口，同时需要子类的额外属性）
3.call OfflineInferenceMixin._add_completion_requests，代码块如下，通过_preprocess_cmpl_one对每个prompt进行格式化并且tokenlization，然后通过_render_and_add_requests调用self._add_request向llm engine中add request
'''
return self._render_and_add_requests(
            prompts=(
                self._preprocess_cmpl_one(
                    prompt,
                    tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for prompt in maybe_tqdm(
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )
'''
4.执行self._run_engine，核心是self.llm_engine.step()
'''
    llm_engine.step()
        engine_core.get_output()
            engine_core.step_fn()
            engine_core.post_step()
        output_processor.process_outputs()
        output_processor.update_scheduler_stats()
'''
engine_core.step_fn()执行了重要流程(Schedule, execute, and make output)
其中model_executor中实例化worker
(vllm/v1/executor中的multiproc_executor，ray_executor_v2，uniproc_executor)
(vllm/v1/worker中的cpu、gpu、tpu、xpu worker)
 worker实例化model runner
(vllm/v1/woker/xxx_model_runner，vllm仓库正在完成v2 model runner的迁移，目录vllm/v1/worker/gpu/model_runner.py)
'''
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    model_output = future.result()
    engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
'''
包装成 EngineCoreOutputs（包含每个请求的新 token、是否结束、scheduler 统计等）返回。
post_step(): 更新scheduler的draft token ids

output_processor.process_outputs()承担：
    Detokenize the token ids into text
    perform stop checks.
        1.超出长度限制
        2.采样到 EOS token
        3.采样到 stop_token_ids 中的 token
        4.输出中出现 stop strings
        stop_token_ids	        ✅ 会出现在输出的 token_ids 中，在 GPU 采样阶段就能检测到（Sampler 直接判断）
        stop（stop strings）	❌ 不会出现在输出文本中（被截断掉了），必须在 CPU detokenize 之后才能检测到（需要拼出文本再做字符串匹配） 
        所以stop strings 的处理是在 output_processor.process_outputs() 中完成的
    Create and handle RequestOutput objects.

5.llm._run_engine执行完毕后，request id对llm_engine.step返回的request out进行排序，整体返回
'''
    while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)  
    return sorted(outputs, key=lambda x: int(x.request_id))
'''


