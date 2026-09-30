# N72R19R1 reference audit

FINAL GOAL: `Selective Identity Memory Update`

审计时间：2026-09-30。所有仓库均为仓库外 shallow clone，位置为
`/data3/liuyeqiang/research_references/N72R19R1/`；本项目没有复制第三方实现。

## 1. CKP — Continual Knowledge Purification for Noisy Lifelong ReID

- Repository: [zhoujiahuan1991/MM2024-CKP](https://github.com/zhoujiahuan1991/MM2024-CKP)
- Commit: `183218eb1624027a1991586c931b242dd08d3a35`
- Paper/year: ACM MM 2024, *Mitigate Catastrophic Remembering via Continual Knowledge Purification for Noisy Lifelong Person Re-Identification*.
- Inspected files: `reid/trainer_noisy.py:100-233,257-312`, `lreid_dataset/datasets/get_data_loaders_noisy.py:120-132`, `continual_train_noisy.py:133-169`.
- Exact mechanism: 用 DBSCAN/Jaccard feature distance 形成 pseudo clusters；在每个 refined label 内计算 one-hot pseudo-label 与均值的平方距离，将 outlier 置为最大距离，再以阈值产生 clean/keep flag；保留 old-model distance，并将被保留样本用于 anti-forgetting KL。`get_data_purify` 只把被选数据加入新的训练集。
- Useful: 新 observation/label 在进入长期知识前先被打分、过滤和净化；pseudo distance、cluster、clean selection、clean/noisy ratio、PR-AUC 是 R1 的观察选择审计启发。
- Not appropriate: 它依赖图像级 ReID 分类/聚类和 noisy label，不能直接当作本项目 512-D identity observation selector；R1 不复制 DBSCAN、分类头或其大模型。
- Borrowed: conceptual only; no code borrowed.

## 2. LSTKC — Long Short-Term Knowledge Consolidation

- Repository: [zhoujiahuan1991/AAAI2024-LSTKC](https://github.com/zhoujiahuan1991/AAAI2024-LSTKC)
- Commit: `c63a3af3fa437e2a744a72c21738e8e68e171687`
- Paper/year: AAAI 2024, *LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification*.
- Inspected files: `continual_train.py:115-146,296-315`, `reid/trainer.py:27-160`.
- Exact mechanism: 每个增量任务前保存 `model_old`；在新旧模型 feature affinity matrix 上计算 absolute difference 的平均值，以 `alpha = 1 - Difference` 自适应融合新旧参数；训练时对新旧 affinity 做带 TP/FP/FN/TN 规则的 KL consolidation。
- Useful: 已验证旧知识必须作为稳定 reference；plasticity 可以由 representation affinity change 调节，而不是让新模型完全替换旧模型。
- Not appropriate: 参数级融合和分类器扩展不是本阶段的 identity state update；R1 保留冻结 GRU teacher，不重建或融合 GRU 参数。
- Borrowed: conceptual only; no code borrowed.

## 3. DKP — Distribution-aware Knowledge Prototyping

- Repository: [zhoujiahuan1991/CVPR2024-DKP](https://github.com/zhoujiahuan1991/CVPR2024-DKP)
- Commit: `1860ce983b007656d290b18f0fe5e24bababb331`
- Paper/year: CVPR 2024, *Distribution-aware Knowledge Prototyping for Non-exemplar Lifelong Person Re-identification*.
- Inspected files: `continual_train.py:123-145`, `reid/trainer.py:30-103`, `reid/models/resnet_uncertainty.py:63-145`.
- Exact mechanism: 从旧数据提取每个 identity 的 `mean_features` 与 `mean_vars`；以 Gaussian noise 从 prototype mean/variance 采样历史 representation，再比较当前 feature 对这些 samples 的 affinity，并以 KL loss 约束新模型保持旧分布关系。
- Useful: identity 不是必然的单一静态中心；prototype variance/分布可以作为 future uncertainty 或 observation evidence 的后续诊断。
- Not appropriate: R1 不引入 multi-prototype 或重新训练 uncertain encoder；第一版 selector 只用显式小维度 evidence。
- Borrowed: conceptual only; no code borrowed.

## 4. PAEMA — Prompt-Guided Adaptive Knowledge Consolidation

- Repository: [zhoujiahuan1991/IJCV2024-PAEMA](https://github.com/zhoujiahuan1991/IJCV2024-PAEMA)
- Commit: `cff86fa7c2415e9e8fc5784a24d0e4b952042a95`
- Paper/year: IJCV 2024, *Exemplar-Free Lifelong Person Re-identification via Prompt-Guided Adaptive Knowledge Consolidation*.
- Inspected files: `utils/prompt_pool.py:10-81`, `lreid/operation/train_p_s.py:49-80`, `lreid/models/metagraph_fd.py:55-114`.
- Exact mechanism: `PromptPool` 维护 key 与 prompt 参数；feature 进入 meta-graph 后通过 prototype/meta vertices 和 cross-graph correlation 选择/转移 knowledge；增量训练保存 old graph vertex，并用 stability loss 约束新 graph。
- Useful: 历史知识可以被选择性调用；稳定 reference 与选择性 retrieval 可以分离。
- Not appropriate: prompt pool、meta-graph 和 Transformer 规模过大，且不是 frozen 512-D GRU 的 observation selection；R1 不采用这些结构。
- Borrowed: conceptual only; no code borrowed.

## 5. C2R — Continual Compatible Representation

- Repository: [PKU-ICST-MIPL/C2R_CVPR2024](https://github.com/PKU-ICST-MIPL/C2R_CVPR2024)
- Commit: `5aa372bf71eb6d200821ada3ba0e4f5b726be64a`
- Paper/year: CVPR 2024, *Learning Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identification*.
- Inspected files: `pkd/models/trans_net.py:24-103`, `pkd/losses/kd_loss.py:29-62`.
- Exact mechanism: RBT block 有多条 transformation paths、learnable prototypes、compatibility attention 和 2-way adaptive weights；输出保留 residual `output + x_inter`；另有 teacher score 的 temperature-scaled KD。
- Useful: 新能力应以兼容旧 representation space 的小 residual 模块增加，而不是重建 identity space。
- Not appropriate: R1 不训练 compatibility transform、prototype bank 或 classifier；只保留“冻结 teacher + 小 selector”的兼容原则。
- Borrowed: conceptual only; no code borrowed.

## 6. DKC — Differentiated Knowledge Consolidation

- Repository: [pku-icst-mipl/dkc-cvpr2025](https://github.com/pku-icst-mipl/dkc-cvpr2025)
- Commit: `897b5b42d683cd1911b0f3f64795cbd7fb38adb8`
- Paper/year: CVPR 2025, *DKC: Differentiated Knowledge Consolidation for Cloth-Hybrid Lifelong Person Re-identification*.
- Inspected files: `reid/trainer.py:43-169,175-245`, `reid/models/recon_net.py:24-76`, `continual_train_base.py:365-497`.
- Exact mechanism: 旧模型在后续 phase 中 frozen/eval；当前 feature 与旧/历史 prototype distribution 做 affinity/KL consolidation；`ReconNet` 通过 residual transformation path 将新 feature 映射回旧知识关系，并对 instance/feature-group relation 分别约束。
- Useful: 可区分需要保留的 old relation 与允许改变的 new relation；支持 R1 把 update selection 与 frozen representation 分开。
- Not appropriate: 它仍是图像 encoder/continual classifier 训练，不是 frame-level identity memory selection；R1 不使用其 reconstruction network。
- Borrowed: conceptual only; no code borrowed.

## 7. CSDP — Continual Self-Paced Dual-Knowledge Purification

- Repository: [zhoujiahuan1991/TPAMI-CSDP](https://github.com/zhoujiahuan1991/TPAMI-CSDP)
- Commit: `d1e3b4a073da703e9f0c3b602a6cfc6e8f67e51c`
- Paper/year: TPAMI 2026, *Mitigate Catastrophic Remembering via Continual Self-Paced Dual-Knowledge Purification for Noisy Lifelong Person Re-Identification*.
- Inspected files: `reid/trainer_noisy.py:63-72,293-355`, `lreid_dataset/datasets/get_data_loaders_noisy.py:120-132`, `continual_train_noisy.py:149-169`.
- Exact mechanism: `select_self_pace` 先按 confidence 阈值得到 max keep，再按 epoch/stride 从 `base_rate` 渐进确定保留比例；warm-up 之后才用 purified subset；model prediction 与 label 的融合更新 refined label，并重复 cluster/purify。
- Useful: 先 clean/mild，再 progressive hard corruption 的 curriculum；keep ratio、precision、recall 应单独记录。
- Not appropriate: 它的 self-paced schedule 针对 noisy labels，不能直接替代 future utility；R1 先完成最小 baseline，再只在 train-dev 上决定是否增加 curriculum。
- Borrowed: conceptual only; no code borrowed.

## 8. LCC-ReID — Lifelong Clothes-Changing ReID

- Repository: [joyner-7/LCC_ReID](https://github.com/joyner-7/LCC_ReID)
- Commit: `131478c5b854d9ff54826fd60c7b2333b30d841a`
- Paper/year: repository describes lifelong clothes-changing person ReID; inspected implementation snapshot is 2026.
- Inspected files: `continual_train.py:37-63,150-213`, `reid/trainer.py:13-38,119-174,289-311`, `reid/loss/loss_uncertrainty.py:102-152`.
- Exact mechanism: 每个 identity 维护 global prototype 与 count；新 prototype 与 old prototype 先按样本量决定 base/query，再做 bias calibration 和 count-weighted fusion；old model 的 repeated-identity `f_id` 用 frozen KD；prototype pool 参与 hard positive/negative ranking。
- Useful: human-confirmed first observation 应作为 immutable anchor/reference 保存；dynamic state 与 anchor 不应混为一个向量。
- Not appropriate: 其 prototype fusion、clothing disentanglement 和 classifier growth 需要图像训练，R1 不把 anchor 当作直接替代 learned state。
- Borrowed: conceptual only; no code borrowed.

## 9. MOTIP — Multiple Object Tracking as ID Prediction

- Repository: [MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP)
- Commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- Paper/year: CVPR 2025, *Multiple Object Tracking as ID Prediction*.
- Inspected files: `models/runtime_tracker.py:91-223,328-390`, `models/motip/id_decoder.py:96-228`, `models/motip/motip.py:30-49`.
- Exact mechanism: runtime tracker 将 trajectory features/boxes/times/ID labels 与 current unknown detections 一起送入 trajectory model 和 ID decoder；decoder 用 cross-attention、causal time mask 和 same-frame self-attention；最终通过 Hungarian/object-max/id-max 等竞争分配协议输出 ID。
- Useful: identity decision 必须看到 candidate competition，不应只看单个 candidate 与 memory 的 self-similarity。
- Not appropriate: R1 不实现 MOTIP、assignment、Hungarian 或 candidate stream；只将 competition evidence 压缩为 selector 的显式特征。
- Borrowed: conceptual only; no code borrowed.

## 10. SAS-VPReID — Scale-Adaptive Video Person ReID

- Repository: [YangQiWei3/SAS-VPReID](https://github.com/YangQiWei3/SAS-VPReID)
- Commit: `e24d8d06235bf68bf1866e07177c1c556b3b50cb`
- Paper/year: WACV 2026 VReID-XFD Workshop, *SAS-VPReID*.
- Inspected files: `model/make_model_clipreid.py:90-151,532-570`, `model/make_model_clipvideoreid_reidadapter_pbp.py`.
- Exact mechanism: CLIP-based visual backbone 后接 memory projection 与 image-specific prompt decoder；跨 frame message token/attention 进行 temporal memory diffusion；README 明确提出 multi-proxy memory、multi-granularity temporal modeling 和 Mamba/shape prior。
- Useful: temporal scales 和 multi-proxy 是单 state 失败后的后续方向。
- Not appropriate: R1 明确禁止 CLIP、更大 encoder、Transformer/Mamba、multi-proxy first version；当前问题是 observation selection，不是 backbone capacity。
- Borrowed: conceptual only; no code borrowed.

## Synthesis for R1

十个项目共同支持三条可迁移原则：新观察必须先评估；旧的有效表示必须保留为稳定 reference；identity decision 要使用竞争关系。它们没有授权 R1 引入大模型、prompt、multi-proxy 或重新训练 encoder。R1 的最小实现因此限定为 frozen N72R18 GRU、immutable human anchor、10–20 维 competition-aware evidence、small selector 和 hard selective update。
