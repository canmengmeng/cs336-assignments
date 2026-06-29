# CS336 Assignment 1 实现进度

## 阶段 1: 基础数学函数 ✅
- [x] Step 1:  run_silu                    SiLU 激活
- [x] Step 2:  run_softmax                 Softmax
- [x] Step 3:  run_rmsnorm                 RMSNorm 归一化
- [x] Step 4:  run_linear                  线性变换 (einsum)
- [x] Step 5:  run_embedding               嵌入查找
- [x] 补:      run_swiglu                  SwiGLU FFN (einsum)

## 阶段 2: 损失函数与数据工具 ⏳
- [x] Step 6:  run_cross_entropy           交叉熵损失
- [x] Step 7:  run_get_batch               批数据采样

## 阶段 3: 注意力机制
- [x] Step 8:  run_scaled_dot_product_attention   缩放点积注意力
- [x] Step 9:  run_rope                          旋转位置编码
- [x] Step 10: run_multihead_self_attention       多头自注意力
- [x] Step 11: run_multihead_self_attention_with_rope  带 RoPE 多头注意力

## 阶段 4: Transformer 模型
- [x] Step 12: run_transformer_block      单个 Transformer 块
- [x] Step 13: run_transformer_lm         完整 Transformer LM

## 阶段 5: 训练基础设施
- [x] Step 14: run_gradient_clipping      梯度裁剪
- [x] Step 15: get_adamw_cls              AdamW 优化器类
- [x] Step 16: run_get_lr_cosine_schedule 余弦学习率调度

## 阶段 6: 检查点管理
- [x] Step 17: run_save_checkpoint        保存检查点
- [x] Step 18: run_load_checkpoint        加载检查点

## 阶段 7: 分词器
- [x] Step 19: get_tokenizer              BPE 分词器
- [x] Step 20: run_train_bpe              BPE 训练算法
