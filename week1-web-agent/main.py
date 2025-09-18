import os
from openai import OpenAI
from datetime import datetime

def main():

    # 初始化OpenAI客户端，配置API连接信息
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        # 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
        api_key = os.getenv('ARK_API_KEY')
    )
    
    # 定义系统提示词，设定AI助手的角色和行为规则
    system_prompt = """
    你是AI个人助手，负责解答用户的各种问题。你的主要职责是：
    1. **信息准确性守护者**：确保提供的信息准确无误。
    2. **搜索成本优化师**：在信息准确性和搜索成本之间找到最佳平衡。
    # 任务说明
    ## 1. 联网意图判断
    当用户提出的问题涉及以下情况时，需使用 `web_search` 进行联网搜索：
    - **时效性**：问题需要最新或实时的信息。
    - **知识盲区**：问题超出当前知识范围，无法准确解答。
    - **信息不足**：现有知识库无法提供完整或详细的解答。
    **注意**：每次调用 `web_search` 时，**只能改写出一个最关键的问题**。如果有任何冲突设置，以当前指令为准。
    ## 2. 联网后回答
    - 在回答中，优先使用已搜索到的资料。
    - 回复结构应清晰，使用序号、分段等方式帮助用户理解。
    ## 3. 引用已搜索资料
    - 当使用联网搜索的资料时，在正文中明确引用来源，引用格式为：  
    `[1]  (URL地址)`。
    ## 4. 总结与参考资料
    - 在回复的最后，列出所有已参考的资料。格式为：  
    1. [资料标题](URL地址1)
    2. [资料标题](URL地址2)
    """

    # 调用API创建响应，发起AI问答请求
    response = client.responses.create(
        model="doubao-seed-1-6-250615",  # 指定使用的模型ID
        input=[  # 输入内容，包含系统提示和用户问题
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "世界500强企业，他们在国内所在的城市，近三年的平均工资是多少？"}]  # 用户具体问题
            }
        ],
        tools=[
            {
                "type": "web_search",  # 配置工具类型为基础联网搜索功能
                "limit": 10,  # 最多返回10个搜索结果
                "sources": ["douyin", "moji", "toutiao"],  # 附加搜索来源（抖音百科、墨迹天气、头条图文等平台）
                "user_location": {  # 用户地理位置（用于优化搜索结果）
                    "type": "approximate",  # 大致位置
                    "country": "中国",
                    "region": "浙江",
                    "city": "杭州"
                }
            }
        ],
        stream=True,  # 启用流式响应（实时返回结果，而非等待全部完成）
        extra_body={"thinking": {"type": "auto"}}, 
        parallel_tool_calls=False,  # 禁用并行工具调用（工具调用按顺序执行）
        max_tool_calls= 3,  # 最多调用3轮工具
        extra_headers={"ark-beta-web-search": "true"},  # 配置额外请求头：启用联网搜索功能
    )

    # 初始化状态变量，用于控制输出格式（避免重复打印标题）
    thinking_started = False  # 标记AI思考过程是否已开始打印
    answering_started = False  # 标记AI回答内容是否已开始打印

    # 遍历流式响应的每个片段（chunk），实时处理并展示结果
    for chunk in response:
        # 获取当前片段的类型（用于判断是思考过程、搜索状态还是回答内容）
        chunk_type = getattr(chunk, 'type', '')

        # 1. 处理AI思考过程的输出
        if chunk_type == "response.reasoning_summary_text.delta":
            if not thinking_started:  # 首次打印思考过程时，输出标题
                print("🤔 AI思考中...")
                thinking_started = True
            # 打印思考内容（delta为当前片段的文本增量）
            print(getattr(chunk, 'delta', ''), end="", flush=True)

        # 2. 处理联网搜索的状态（开始/完成）
        elif "web_search_call" in chunk_type:
            if "in_progress" in chunk_type:  # 搜索开始时，打印时间戳
                print(f"\n🔍 开始搜索... [{datetime.now().strftime('%H:%M:%S')}]")
            elif "completed" in chunk_type:  # 搜索完成时，打印提示
                print("✅ 搜索完成")

        # 3. 处理搜索结果中的关键词（展示AI的搜索目标）
        elif (chunk_type == "response.output_item.done" and 
              hasattr(chunk, 'item') and str(getattr(chunk.item, 'id', '')).startswith("ws_")):
            # 提取搜索关键词并打印（截取前80字符避免过长）
            if hasattr(chunk.item, 'action') and hasattr(chunk.item.action, 'query'):
                print(f"📝 搜索关键词: {chunk.item.action.query[:80]}...")

        # 4. 处理AI的最终回答内容
        elif chunk_type == "response.output_text.delta":
            if not answering_started:  # 首次打印回答时，输出标题和分隔线
                print(f"\n💬 AI回答 [{datetime.now().strftime('%H:%M:%S')}]:")
                print("-" * 50)
                answering_started = True
            # 打印回答内容（delta为当前片段的文本增量）
            print(getattr(chunk, 'delta', ''), end="", flush=True)

    # 所有响应处理完成后，打印结束时间
    print(f"\n\n✨ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 当脚本直接运行时，执行main函数
if __name__ == "__main__":
    main()