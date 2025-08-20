#!/usr/bin/env python3
"""
TDRCluster主启动文件 - 简化版本
"""
import json
from pathlib import Path
from datetime import datetime

# 配置管理和核心组件  
from config.config_loader import CONFIG
from core.graph import TDRClusterGraph
from services.dataset_manager import DatasetManager
from services.llm_service import LLMService


def load_params() -> dict:
    """加载运行参数"""
    runtime_config = CONFIG.get('runtime', {})
    clustering_config = CONFIG.get('clustering', {})
    
    return {
        'dataset': runtime_config.get('dataset', 'banking77'),
        'initial_k': clustering_config.get('initial_k', 10),
        'sample_size': runtime_config.get('sample_size', None),
        'output_dir': 'output'  # 硬编码输出路径
    }




def save_results(final_state: dict, output_dir: str, dataset_name: str) -> tuple[Path, Path]:
    """保存聚类结果"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果到CSV
    print(f"💾 保存结果到: {output_dir}/")
    
    import pandas as pd
    
    results_data = []
    categories = final_state.get('categories', {})
    
    for cat_id, category in categories.items():
        for query in category.queries:
            results_data.append({
                'query_id': query.id,
                'query_content': query.content,
                'category_id': category.id,
                'category_description': category.description,
                'dataset': dataset_name,
                'timestamp': timestamp
            })
    
    csv_path = None
    if results_data:
        results_df = pd.DataFrame(results_data)
        csv_path = output_path / f"{dataset_name}_clustering_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"   ✓ 详细结果: {csv_path}")
    
    # 保存摘要到JSON
    summary_data = {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'total_queries': sum(len(cat.queries) for cat in categories.values()),
        'total_categories': len(categories),
        'categories': {}
    }
    
    for cat_id, category in categories.items():
        summary_data['categories'][cat_id] = {
            'description': category.description,
            'query_count': len(category.queries),
            'samples': category.samples[:5]
        }
    
    json_path = output_path / f"{dataset_name}_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 摘要报告: {json_path}")
    
    return csv_path, json_path


def print_results_summary(final_state: dict, dataset_name: str):
    """打印结果摘要"""
    categories = final_state.get('categories', {})
    tasks_remaining = final_state.get('tasks', [])
    
    print("\n" + "="*70)
    print(f"📊 TDR聚类结果摘要 - {dataset_name}")
    print("="*70)
    
    print(f"最终类别数: {len(categories)}")
    print(f"剩余任务数: {len(tasks_remaining)}")
    
    total_queries = sum(len(cat.queries) for cat in categories.values())
    print(f"总分类查询数: {total_queries}")
    
    if categories:
        print(f"\n📁 类别详情:")
        for i, (cat_id, category) in enumerate(categories.items(), 1):
            query_count = len(category.queries)
            print(f"\n{i}. {cat_id} ({query_count} 个查询)")
            print(f"   描述: {category.description}")
            
            if category.samples:
                print("   样本:")
                for j, sample in enumerate(category.samples[:3]):
                    print(f"     {j+1}. {sample[:60]}...")
    
    print("="*70)


def main():
    """主函数"""
    print("🚀 TDRCluster - LLM驱动的动态意图聚类系统\n")
    
    # 初始化服务
    print("⚙️ 初始化LLM服务...")
    llm_service = LLMService(CONFIG.get('llm', {}))
    
    print("✅ LLM服务初始化完成\n")
    
    # 加载运行参数
    params = load_params()
    
    # 显示运行参数
    print("⚙️ 运行参数:")
    print(f"   数据集: {params['dataset']}")
    print(f"   初始K值: {params['initial_k']}")
    if params['sample_size']:
        print(f"   采样大小: {params['sample_size']}")
    print(f"   输出目录: {params['output_dir']}")
    print()
    
    # 初始化数据集管理器
    print("🔍 初始化数据集管理器...")
    dataset_manager = DatasetManager()
    
    # 加载数据集
    try:
        queries = dataset_manager.load_dataset_as_queries(
            params['dataset'], 
            params['sample_size']
        )
        
        if not queries:
            print("❌ 没有加载到任何查询数据")
            return
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 初始化TDR系统
    print("🔧 初始化TDR聚类系统...")
    try:
        tdr_graph = TDRClusterGraph(llm_service)
        print("   ✓ TDR系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 运行聚类流程
    print(f"\n⚡ 开始TDR聚类流程...")
    print("   这可能需要几分钟时间，请耐心等待...")
    print("="*70)
    
    try:
        final_state = tdr_graph.run(
            initial_queries=queries,
            initial_k=params['initial_k'],
            dataset_name=params['dataset']
        )
        print("="*70)
        print("✅ TDR聚类流程完成!")
        
    except Exception as e:
        print(f"❌ 聚类流程失败: {e}")
        return
    
    # 显示结果摘要
    print_results_summary(final_state, params['dataset'])
    
    # 保存结果
    try:
        csv_path, json_path = save_results(
            final_state, 
            params['output_dir'], 
            params['dataset']
        )
        
        print(f"\n💾 结果已保存:")
        if csv_path:
            print(f"   详细数据: {csv_path}")
        print(f"   摘要报告: {json_path}")
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
    
    print(f"\n🎉 TDRCluster聚类任务完成!")


if __name__ == "__main__":
    main()