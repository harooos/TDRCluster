# TDRCluster

**LLM驱动的动态意图聚类系统**

一个基于大语言模型的智能聚类系统，专门用于无标签用户查询的自动化意图分类。采用"自顶向下"的细分策略，结合LLM的语义理解能力，生成具有明确业务含义的高质量聚类结果。

## 🌟 核心特性

- **🧠 LLM驱动**: 使用大语言模型进行语义理解和聚类决策
- **🔄 迭代细分**: 采用队列式BFS算法，自顶向下逐层细分
- **🎯 智能决策**: 支持创建、分配、细分三种决策模式  
- **🏷️ 自动描述**: 为每个类别生成包含典型样本的描述
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

编辑 `config/config.yaml`：

```yaml
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

### 运行

```bash
# 使用banking77数据集
python main.py

# 使用自定义数据集
python main.py --dataset your_dataset

# 快速测试（采样）
python main.py --sample 100
```

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

1. **数据准备**: 自动检测CSV格式，生成文本向量
2. **初始聚类**: 使用K-Means进行初始聚类
3. **LLM审查**: 分析cluster质量，决策下一步操作
4. **动态调整**: 执行创建/分配/细分操作
5. **迭代优化**: 循环执行直到达到最佳聚类效果

## 🔧 自定义数据集

只需要包含 `query` 列的CSV文件：

```csv
query
"我的银行卡余额不够了"
"怎么查看交易记录" 
"账户被锁定了怎么办"
```

将文件放入 `data/raw_data/` 目录，系统会自动处理。

## 📊 输出结果

- **详细结果**: `output/dataset_clustering_timestamp.csv`
- **摘要报告**: `output/dataset_summary_timestamp.json`

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