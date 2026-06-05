# 从llm的应用角度理解spec decode
总体流程：draft+verify+reject

## draft model计算draft tokens
![alt text](image-1.png)

## target model在1次forward中验证所有draft token
![alt text](image-4.png)

## reject采样计算accepted token
![alt text](image-5.png)

p(x)和q(x)代表在第x个位置时的probs向量，
p(x, dt)和q(x, dt)target model和draft model在整个p(x)和q(x)中draft token的概率
switch:
   case 1: p(x, dt) > q(x, dt), accept
   case 2：p(x, dt) < q(x, dt), p = random(0, 1)计算
        case 2.1: p < p(x, dt) / q(x, dt)，accept
        case 2.2: reject sample, 计算残差分布后归一化重新采样，得到correction_token
        residual = torch.clamp(p_dist - q_dist, min=0)
        correction = torch.multinomial(residual, num_samples=1)
具体举例：
    ![alt text](image-6.png)

结果：
    ![alt text](image-7.png)

注意，开启spec decode之后，结果会与仅有target model forward不一致，这是因为draft model与target model即使在相同的input ids下，推测出来的下一个token也会不一致，并且在不一致的情况下仍然会被accept！（case 1和2均会导致）

# 提高spec decode accept的方法：
1.同模型家族：例如用 Llama-7B 给 Llama-70B 做草稿，通常远好于混用不同架构，因为它们在相似数据和目标上做的训练，所以相对通用一些。
2.任务可预测性：代码补全、翻译、结构化输出的接受率通常高于创作类写作或开放式聊天。
3.更低温度：采样更确定时，草稿模型与目标模型更可能一致。4.合适的 K 值：太小（K=2）加速不足；太大（K=10）后段 token 常被拒绝。K=4–6 通常是甜点区间。