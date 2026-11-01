import asyncio
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import time

# 从.env文件中加载环境变量
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="deepseek-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

async def demo_async_abatch():
    """演示 abatch 异步批量调用"""
    print("=== 演示: abatch 的异步批量效果 ===")
    start_time = time.perf_counter()
    print("程序开始...")

    # 准备一批提问
    prompts = [
        "解释什么是人工智能",
        "解释什么是机器学习",
        "解释什么是深度学习"
    ]

    print(">>> 发起 abatch 异步批量请求...")
    # abatch：一次性并发多个请求
    responses = await model.abatch(prompts)

    print(">>> 获取批量结果：")
    for idx, res in enumerate(responses):
        print(f"【问题{idx+1}】{prompts[idx]}")
        print(f"回答：{res.content}\n")

    end_time = time.perf_counter()
    print(f"=== 总运行耗时: {end_time - start_time:.2f}s ===")


async def main():
    """主函数"""
    await demo_async_abatch()


if __name__ == "__main__":
    asyncio.run(main())
