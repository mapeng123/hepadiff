# HepaDiff Virtual Liver — Agent Server

把 HepaDiff（19.6M 肝细胞语料训练的 69M 参数掩码扩散模型）包装成 LLM 智能体可直接调用的工具服务。
科研人员用 **Kimi（K2/K3）或 ChatGPT（GPTs）** 以自然语言完成虚拟扰动实验，无需写代码。

## 能力（7 个工具）

| 工具 | 功能 |
|---|---|
| `knockout` | 零样本敲低任意 panel 基因（约 16,000 个），返回基因 delta + Hallmark 通路变化 |
| `compound_perturbation` | 210 个库内化合物扰动 |
| `generate_condition` | 疾病状态（NASH/MASH、纤维化、HCC 等 26 种）细胞生成 |
| `query_drug_target` | ChEMBL 靶点/作用类型/IC50 查询（库外化合物走靶点 KO 路径） |
| `list_cell_types` | 22 种有健康参考的肝脏细胞类型 |
| `list_compounds` | 210 个库内化合物清单 |
| `server_info` | 模型版本、语料规模、能力边界 |

**能力边界（务必让 Agent 遵守）**：机制分诊而非临床严重度分级；靶点注释路径只覆盖在靶机制（脱靶/代谢物/宿主依赖不可见）；激动剂预测弱于抑制剂；发表级结论建议 10 种子评估。

## 接入方式

### 方式一：Kimi（MCP 协议）

1. 确认服务公网地址：AutoDL 控制台 → 实例 → **自定义服务**，获得 `https://<instance>.westc.seetacloud.com`（代理到本机 6006 端口）
2. Kimi 智能体设置 → MCP 服务器 → 添加：
   ```
   URL: https://<instance>.westc.seetacloud.com/mcp/
   ```
   （注意末尾斜杠；协议为 streamable HTTP）
3. 把下方「推荐系统提示词」粘进智能体指令

### 方式二：ChatGPT（GPTs Actions，OpenAPI）

1. GPTs → Create → Configure → Actions → Import from URL：
   ```
   https://<instance>.westc.seetacloud.com/openapi.json
   ```
2. 可用端点：`POST /tools/knockout`、`POST /tools/compound_perturbation`、`POST /tools/generate_condition`、`POST /tools/query_drug_target`、`POST /tools/list_cell_types`、`POST /tools/list_compounds`、`GET /`
3. 把下方「推荐系统提示词」粘进 Instructions

### 推荐系统提示词

```
You are a virtual-liver research assistant powered by HepaDiff, a masked-diffusion
model trained on 19.6 million liver cells (483 datasets). You can run in silico
experiments through the connected tools:

- knockout(gene, cell_type, n_cells, seed): zero-shot knockout of any of ~16,000
  panel genes in a liver cell type; returns per-gene expression deltas and
  Hallmark pathway shifts.
- compound_perturbation(compound, ...): perturb with one of the 210 in-library
  compounds (check with list_compounds first).
- generate_condition(condition, cell_type, ...): generate cells under a disease
  condition (e.g. NASH_MASH, fibrosis, HCC).
- query_drug_target(drug_name): ChEMBL target/action-type/potency lookup.

Workflow rules:
1. For a compound NOT in the library, first call query_drug_target; if the target
   gene is in the panel and the action type is inhibition, approximate the
   on-target mechanism with knockout() on the target gene, and state this proxy
   explicitly. Agonist predictions are weaker — flag them.
2. Default cell type: "centrilobular region hepatocyte" for metabolism/DILI
   questions; "hepatic stellate cell" for fibrosis; "Kupffer cell" for
   inflammation. Use list_cell_types when unsure.
3. Report results as: top up/down genes, pathway shifts, and a plain-language
   mechanistic interpretation. Always mention n_cells and seed.
4. HepaDiff provides mechanism triage, not clinical severity classification;
   it cannot see off-target, metabolite, or host-dependent effects. Say so when
   relevant.
5. For publication-grade claims, recommend 10-seed evaluation.
```

## 自部署（任何机器）

```bash
pip install fastmcp fastapi uvicorn torch numpy pandas
# 需要: pmdm_v3_step200000.pt, hepadiff_reference.npz, liver_bin_edges_v4.npy,
#       hallmark.gmt, chembl_mechanisms.json, dilirank_targets.csv, compound_vocab_210.txt
python server.py --host 0.0.0.0 --port 6006   # MCP(/mcp/) + REST(OpenAPI)
python server.py --stdio                       # 本地 MCP 客户端（Claude Desktop 等）
```

模型与代码：https://github.com/mapeng123/hepadiff

## 当前部署

- 机器：AutoDL westc（RTX 5090 32GB），`/root/autodl-tmp/hepadiff_agent/`
- 单次 500 细胞扰动约 10-30 秒（GPU）
- 服务进程：`server.py --host 0.0.0.0 --port 6006`（nohup 后台）
