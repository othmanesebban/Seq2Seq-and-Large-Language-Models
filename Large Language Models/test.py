import openai
import asyncio

openai.api_key = "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"

async def main():
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test simple"}]
    )
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    asyncio.run(main())
