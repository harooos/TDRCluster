import os
import shutil
import json
from pathlib import Path
from core.state import CategoryNode
from core.tools import visualize_category_tree

# 创建测试输出目录
test_output_dir = "test_output"
Path(test_output_dir).mkdir(exist_ok=True)

# 创建测试树结构
test_tree = {
    "root": CategoryNode(id="root", description="根节点", children=[
        CategoryNode(id="child1", description="子节点1"),
        CategoryNode(id="child2", description="子节点2", children=[
            CategoryNode(id="grandchild1", description="孙子节点1")
        ])
    ]),
    "child1": CategoryNode(id="child1", description="子节点1"),
    "child2": CategoryNode(id="child2", description="子节点2", children=[
        CategoryNode(id="grandchild1", description="孙子节点1")
    ]),
    "grandchild1": CategoryNode(id="grandchild1", description="孙子节点1")
}

try:
    # 调用测试函数
    visualize_category_tree(test_tree, test_output_dir)
    
    # 验证输出文件是否存在
    json_path = Path(test_output_dir) / "category_tree.json"
    assert json_path.exists(), "JSON文件未生成"
    
    # 验证JSON内容
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        assert "root" in content, "根节点不存在"
        assert content["root"]["description"] == "根节点", "根节点描述不正确"
    
    # 检查图片文件是否存在(不强制要求)
    img_path = Path(test_output_dir) / "category_tree.png"
    if img_path.exists():
        print("Graphviz已安装，图片文件已生成")
    else:
        print("Graphviz未安装，跳过图片生成检查")
    
    print("所有测试通过!")
except Exception as e:
    print(f"测试过程中出现错误: {str(e)}")
# finally:
#     # 清理测试输出目录
#     # if os.path.exists(test_output_dir):
    #     shutil.rmtree(test_output_dir)