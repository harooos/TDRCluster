# TDRCluster

**LLM驱动的动态意图聚类系统**

一个面向无标签用户查询的动态意图聚类系统。当前版本采用单一 BFS 队列状态机，将“物理切分”和“语义收敛”拆分给不同引擎：`UMAP + HDBSCAN` 负责物理提纯，LLM 负责维护 running taxonomy。

## 🌟 核心特性

- **🧠 双引擎架构**: `UMAP + HDBSCAN` 负责物理切分，LLM 负责语义裁决
- **🔄 BFS 状态机**: 采用统一队列循环，持续处理 root task 与后续 resplit task
- **🎯 Taxonomy维护**: 支持 `create / merge / split` 三种核心语义动作，`merge` 可按需更新已有类别的名称、定义和边界
- **🏷️ 结构化类别**: 类别包含 `name / definition / boundary_notes / positive_examples`
- **🧹 残差池**: 显式记录噪声簇与无法继续稳定切分的残差簇
- **📊 多数据集**: 支持banking77等标准数据集，也支持自定义CSV数据
- **🔧 多provider**: 支持各种LLM服务提供商

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 至少4GB内存  
- 网络连接（用于LLM API调用）

### 安装

```bash
git clone https://github.com/your-repo/TDRCluster
cd TDRCluster
pip install -r requirements.txt
```

### 配置

复制 `config/config.example.yaml` 为本地 `config/config.yaml` 后再编辑：

```yaml
runtime:
  dataset: "banking77"
  taxonomy_context: null  # 可选：补充领域背景或命名偏好

# LLM配置
llm:
  base_url: "your-llm-api-base-url" 
  api_key: "your-llm-api-key"
  model_name: "your-model-name"

# Embedding配置  
embedding:
  base_url: "your-embedding-api-base-url"
  api_key: "your-embedding-api-key"
  model_name: "your-embedding-model"
```

`taxonomy_context` 是可选的。如果不填，reviewer 只依据已有 taxonomy 和当前 micro-cluster 样本做判断。

### 运行

```bash
python main.py
```

当前版本通过 `config/config.yaml` 控制运行数据集、采样规模与物理切分参数。

## 📁 项目结构

```
TDRCluster/
├── main.py                    # 主程序入口
├── config/
│   ├── config.yaml           # 配置文件
│   └── config_loader.py      # 配置加载器
├── core/                     # 核心模块
│   ├── state.py              # 数据结构定义  
│   ├── tools.py              # 工具函数
│   ├── prompts.py            # LLM提示模板
│   └── graph.py              # LangGraph工作流
├── services/                 # 服务层
│   ├── llm_service.py        # LLM服务
│   ├── embedding_service.py  # 向量化服务
│   ├── clustering_service.py # 聚类服务
│   └── dataset_manager.py    # 数据集管理
├── data/                     # 数据存储
│   ├── raw_data/             # 原始CSV数据
│   └── processed_data/       # 处理后的数据和embedding
└── output/                   # 聚类结果输出
```

## ⚙️ 工作流程

1. **数据准备**: 加载 CSV，并为 query 生成或读取 embedding
2. **Root Task 入队**: 将全量 query 作为未分化大簇放入 BFS 队列
3. **物理切分**: 使用 `UMAP + HDBSCAN` 生成 micro-clusters 与 residual clusters
4. **LLM 审查**: 结合 running taxonomy 对每个 micro-cluster 做 `create / merge / split`
5. **动态收敛**: 更新 taxonomy，或把仍不纯的 cluster 重新入队继续切分
6. **结果输出**: 导出分类结果与 residual pool 摘要

## 🔧 自定义数据集

只需要包含 `query` 列的CSV文件：

```csv
query
"I need to reset my account password"
"How can I change the shipping address for my order?"
"The app crashes when I try to upload a file"
```

将文件放入 `data/raw_data/` 目录，系统会自动处理。

## 📊 输出结果

- **详细结果**: `output/dataset_clustering_timestamp.csv`
- **摘要报告**: `output/dataset_summary_timestamp.json`

摘要报告会同时包含：

- 最终 taxonomy 的结构化类别信息
- residual cluster 的数量、样本与残差原因

## 🤝 贡献

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支  
5. 创建 Pull Request

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情。

---

**让AI理解你的用户意图** 🚀
