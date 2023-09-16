import chainlit as cl
from langchain import PromptTemplate, OpenAI, LLMChain  
import os
from credentials import OPENAI_API_KEY
from PIL import Image
import io

image_path = "/images"
# os.mkdir(image_path)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

template = """Question: {question}

Answer: Let's think step by step."""

@cl.on_chat_start
async def main():
    image = None
    while image is None:
        image = await cl.AskFileMessage(
            content="Please upload an image to begin",
            accept=['image/png'],
            max_size_mb=20,
            timeout=180
        ).send()
    image = io.BytesIO(image[0].content)
    image = Image.save(f"{image_path}/image.png")
    # image = image.save(f"{image_path}/image.png")
    prompt = PromptTemplate(template=template, input_variables = ["question"])
    llm_chain = LLMChain(prompt = prompt,llm=OpenAI(temperature=0,streaming=True),verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message : str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()


