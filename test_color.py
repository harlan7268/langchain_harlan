import os
from typing import List

from langchain.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


model = Tongyi(
    model_name='qwen-max',
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_kwargs={"temperature":0.01},
    )

def get_docs(role: str="student") -> List[Document]:
    docs = []
    if role == "student":
        docs = [
            Document(page_content="李立喜欢红色但不喜欢黑色"),
            Document(page_content="李华喜欢绿色但更喜欢白色")
        ]
    elif role == "teacher":
        docs = [
            Document(page_content="倪老师喜欢蓝色和紫色"),
            Document(page_content="张老师喜欢棕色")
        ]
    return docs

prompt_1 = PromptTemplate.from_template("每个人喜欢的颜色是什么:\n\n{context}")

answer_1_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_1,
    output_parser=StrOutputParser()
    )
answer_1 = answer_1_chain.invoke({"context": get_docs("teacher")})
print("********************")
print(answer_1)
print("********************")
answer_1_improve_chain = {"context": get_docs} | answer_1_chain
answer_1_improve = answer_1_improve_chain.invoke("teacher")
print("********************")
print(answer_1_improve)
print("********************")


prompt_2 = PromptTemplate.from_template("谁最喜欢{color}:\n\n{context}")

answer_2_chain = {
    "context": RunnableLambda(lambda x: x['role']) | answer_1_improve_chain,
    "color": RunnableLambda(lambda x: x['color'])    
} | prompt_2 | model

answer_2 = answer_2_chain.invoke({"role": "teacher","color": "蓝色"})
print("********************")
print(answer_2)
print("********************")