import os
from openai import OpenAI
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

# API 配置
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.gptsapi.net/v1"
)
if not client.api_key:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量")
MAX_RETRIES = 3
RETRY_DELAY = 2

# 输入和输出文件路径
INPUT_FILE = "百科类问题.xlsx"
OUTPUT_FILE = "百科类问题classification_results.csv"

# 分类任务的 Prompt
CLASSIFICATION_PROMPT = """
你是一个智能助手，需要根据用户的query将其分类到以下预定义的类别中。每个类别都有具体的描述和示例，请仔细阅读并理解这些类别，然后根据用户输入的内容判断它属于哪个类别。
1、只允许输出一个类别名称作为答案，不要解释或添加任何多余内容。示例：充值与消费异常或退款问题
2、如果用户query不明确或无法归入以下类别，输出“未明确分类”。
3、如果用户query包含多种诉求，请综合理解，并选择最主要的一个输出，而不是输出未明确分类。

类别列表：

1. **充值与消费异常或退款问题**  
   描述：用户报告充值失败、未到账、充错（如账号或道具），或购买道具出现问题等情况。  
   示例：  
   - “我充Q币充错号了怎么办？”  
   - “我充值错了，能不能退钱啊？”  
   - “充值崩溃了，老哥，还不快修修”  
   - “为什么我买了好运卡没有收到？”

2. **充值优惠与折扣需求**  
   描述：用户询问充值是否有折扣、返利、赠品或更优惠的渠道。  
   示例：  
   - “安卓充值点券有优惠吗？”  
   - “有没有返利活动？”  
   - “我充一万有什么优惠？”

3. **皮肤与道具返场时间或获取方式**  
   描述：用户询问特定皮肤或道具的返场或上线时间或获取方式。或对特点道具有上架或返场需求。
   示例：  
   - “小奇拉比什么时候返场？”  
   - “李白无双什么时候返场？”  
   - “明天要返场什么皮肤吗？”  
   - “诸葛亮的星元皮肤哪里弄？”
   - ”别发了！我不会充值的！除非上架机甲封神的皮肤”

4. **游戏资讯**  
   描述：用户询问游戏更新、版本消息、新皮肤信息，不包括返场时间相关问题。  
   示例：  
   - “2025年03月高级招募A忍是谁”  
   - “使命召唤手游下赛季手册都有什么”  
   - “火影忍者手游周五木叶快报”
   - “有快报吗”
   - “有什么内幕消息吗”

5. **连跪与匹配机制不满**  
   描述：用户抱怨游戏连输、匹配不公或队友差，或请求改善匹配（如匹配人机）。  
   示例：  
   - “为什么排位一直输？”  
   - “王者荣耀到底是什么匹配机制嘛？”  
   - “优化单排机制吧，排的都是什么东西？”
   - “没人管管吗，都是什么垃圾队友”

6. **游戏内容反馈与优化请求**  
   描述：用户对游戏设计（如特效、技能、机制）提出改进建议、反馈BUG、分开游戏问题情况或表达不满，不涉及匹配或抽奖。  
   示例：  
   - “能不能优化一下千年之狐，大招特效跟贴图一样？”  
   - “女娲那个年限大招两秒多预警BUG啥时候修？”  
   - “奇拉比什么时候调整，太变态了，死妈策划”

7. **抽奖机制抱怨**  
   描述：用户抱怨抽奖爆率低、结果差或保底太贵，常伴随负面情绪或退游倾向。  
   示例：  
   - “抽了100多次，什么都没抽到红装。”  
   - “充了1300都没出货🥲”  
   - “再也不充钱了，纯纯恶心人，充钱啥都抽不到”

8. **超核玩家资格询问**  
   描述：用户询问成为超核玩家的条件或确认自己是否符合资格。  
   示例：  
   - “如何成为超核玩家？”  
   - “我啥时候有超核？”  
   - “心悦会员”

9. **福利与免费赠品需求**  
   描述：用户请求免费赠送皮肤、点券、红包等福利，可能附带条件或威胁。  
   示例：  
   - “可以送我一个皮肤吗？”  
   - “送我一个648再说，不然我不充”  
   - “有没有福利可以领吗？”

10. **客服与人工支持需求**  
    描述：用户要求联系人工客服、质疑机器人回复，或请求解决具体问题。  
    示例：  
    - “转人工？”  
    - “人工在吗，我想问一下，这游戏现在是不是针对充值的玩家？”  
    - “姐姐你可以帮我联系一下人工客服反馈一下吗？”

11. **充值记录与统计查询**  
    描述：用户请求查看自己在游戏或平台的充值金额或记录，需明确查询范围。  
    示例：  
    - “我在和平一共充了多少钱？”  
    - “可以查看我在甄选充了多少吗？”  
    - “能查一下这个账号充值了多少？”

12. **游戏选择与推送偏好**  
    描述：用户指定推送某游戏内容，或拒绝无关游戏的推送，需指明偏好或拒绝的游戏或直接提出游戏名称。
    示例：  
    - “我只玩使命召唤哦，别发王者。”  
    - “多给我推火影忍者的。”  
    - “能不能别发王者，我又不玩”
    - “有没有cf端游”
    - “王者的有没有呀”
    - “火影忍者”

13.**活动信息及规则**
    描述：用户询问游戏活动信息，包括活动时间、规则、奖品等。
    示例：
    - “兔叽，管理可以和群友组队消费嘛？”
    - “百宝季什么时候结束”

14.**账号问题**
    描述：用户询问与账号相关的问题，包括账号被封、找回密码等。
    示例：
    - “账号被封了怎么办？”

15.**支付问题**
    描述：用户询问与支付相关的问题，包括充值、支付方式等。
    示例：
    - “怎么微信支付”
    - “ios怎么支付”

16.**询问充值链接和商城渠道**
    描述：用户询问充值链接或商城渠道。
    示例：
    - “点券充值链接”
    - “尊享商城”


现在，请根据以下用户query进行分类：

{query}
"""

# 有效分类类别列表，用于校验
VALID_CATEGORIES = [
    "充值与消费异常或退款问题", "充值优惠与折扣需求", "皮肤与道具返场时间或获取方式",
    "游戏资讯", "连跪与匹配机制不满", "游戏内容反馈与优化请求",
    "抽奖机制抱怨", "超核玩家资格询问", "福利与免费赠品需求",
    "客服与人工支持需求", "充值记录与统计查询", "游戏选择与推送偏好", "未明确分类","账号问题","活动信息及规则","支付问题","询问充值链接和商城渠道"
]

def classify_query(query: str) -> str:
    full_prompt = CLASSIFICATION_PROMPT.format(query=query)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的文本分类助手。"},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=20
            )
            
            category = response.choices[0].message.content.strip()
            
            if category in VALID_CATEGORIES:
                return category
            else:
                print(f"无效分类结果: {category} (Query: {query})")
                return f"无效分类"

        except Exception as e:
            print(f"API 请求失败 (Query: {query}, Attempt: {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "分类返回失败"

def main():
    try:
        df = pd.read_excel(INPUT_FILE)
        if not all(col in df.columns for col in ["openid", "query"]):
            raise ValueError("输入文件必须包含 'openid' 和 'query' 两列")
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return

    # 读取已存在的分类结果
    processed_openids = set()
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        processed_openids = set(existing_df['openid'])
        print(f"发现已处理数据，已完成 {len(processed_openids)} 条")

    print("开始分类...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分类进度"):
        openid = row["openid"]
        # 跳过已处理的数据
        if openid in processed_openids:
            continue
            
        query = str(row["query"])
        category = classify_query(query)
        
        # 实时写入结果到 CSV
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            if not processed_openids:  # 如果是新文件，写入表头
                f.write("openid,query,category\n")
            f.write(f'"{openid}","{query}","{category}"\n')
        
        processed_openids.add(openid)

    print(f"分类结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()