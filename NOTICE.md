阅读工作区文件  verl 作为训练引擎 我们的代码只负责把协议接到它的扩展点 先后完成sft与rl阶段的grpo训练。
预训练权重在/home/JJ_Group/lih2511/test/nanoGPT/model/language_model/checkpoints/best.safetensors， 
sft与rl数据在/home/JJ_Group/lih2511/test/nanoGPT/model/language_model/data/post_train。
这要求我们设计良好的harness框架 相关实现未完成 在/home/JJ_Group/lih2511/test/nanoGPT/agent  
由于训练数据来自codex 故harness的设计(assistant parser等)应支持codex风格(json输出等) 其余运行时设计、
resume /retry逻辑 智能体的环境(这一部分目前完全没有) verifier 都需要你来设计。相关开源的评测bench也可参考
除此之外 为了防止agent表现受限于预训练模型的能力 我们准备了deepseek的API（网址需自寻）完成训练任务后可以用这个验证
harness/环境是否capable并逐步改进(调取商用模型以进一步修改harness框架的不足之处) 我们认为调取商用模型足够强大 可以将
不良结果倒逼到harness本身的不良上 从而进一步修改harness框架的不足之处

DEEPSEEK_API_KEY: 请在air-node-02/03 通过source ~/.bashrc调用
计算节点: ssh air-node-02/03 多卡训练
训练配置在/home/JJ_Group/lih2511/test/nanoGPT/model/language_model/config/~
实验结果 loss曲线 实验指标请记录并绘图到/home/JJ_Group/lih2511/test/nanoGPT/assets 实验结果分组汇总至/home/JJ_Group/lih2511/test/nanoGPT/logs

有可信改进后请及时将代码上传https://github.com/HuanLi0311/nanoGPT  邮箱huanhuanli104@gmail.com

最后 若你认为相关方法有潜力 或已可形成顶会的论文初稿 请联网下载ICLR的tex template至工作区 并撰写
