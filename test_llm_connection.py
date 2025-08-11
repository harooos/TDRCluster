import unittest
from openai import OpenAI
from config.settings import settings
import json

class TestLLMConnection(unittest.TestCase):
    """测试大模型API连接和基本功能"""
    
    def setUp(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
    
    def test_api_connection(self):
        """测试API基础连接"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "请回复'测试成功'"}],
                max_tokens=10,
                timeout=10
            )
            # 处理可能的字符串响应
            if isinstance(response, str):
                response = json.loads(response)
            content = response.choices[0].message.content
            self.assertIn("测试成功", content)
            print("API连接测试成功")
        except Exception as e:
            self.fail(f"API连接失败: {str(e)}")
    
    def test_complex_response(self):
        """测试复杂请求处理能力"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system", 
                    "content": "你是一个测试助手，请将用户输入重复一遍"
                }, {
                    "role": "user", 
                    "content": "这是一条测试消息123"
                }],
                temperature=0,
                timeout=15
            )
            # 处理可能的字符串响应
            if isinstance(response, str):
                response = json.loads(response)
            content = response.choices[0].message.content.strip()
            self.assertEqual("这是一条测试消息123", content)
            print("复杂请求测试成功")
        except Exception as e:
            self.fail(f"复杂请求处理失败: {str(e)}")

if __name__ == "__main__":
    unittest.main()