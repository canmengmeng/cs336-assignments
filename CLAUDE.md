我正在做CS336 Stanford课程的作业，所有任务如下：
Assignments
Assignment 1: Basics
Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model.
Train a minimal language model.
Assignment 2: Systems
Profile and benchmark the model and layers from Assignment 1 using advanced tools, optimize Attention with your own Triton implementation of FlashAttention2.
Build a memory-efficient, distributed version of the Assignment 1 model training code.
Assignment 3: Scaling
Understand the function of each component of the Transformer.
Query a training API to fit a scaling law to project model scaling.
Assignment 4: Data
Convert raw Common Crawl dumps into usable pretraining data.
Perform filtering and deduplication to improve model performance.
Assignment 5: Alignment and Reasoning RL
Apply supervised finetuning and reinforcement learning to train LMs to reason when solving math problems.
Optional Part 2: implement and apply safety alignment methods such as DPO.

我希望完成任务的仓库框架为：
- src/assignment-1-basics/
- src/assignment-2-systems/
...

python环境为当前目录的uv python3.14，后续统一用uv add 管理库。