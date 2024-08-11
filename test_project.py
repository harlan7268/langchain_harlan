import os
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain.output_parsers import PydanticOutputParser

from  langchain_core.pydantic_v1 import BaseModel, Field

class LogModel(BaseModel):
    user_input: str = Field(description="用户输入内容")
    llm_output: str = Field(description="大模型输出结果")

prompt = PromptTemplate.from_template("按照如下格式回答用户问题\n {_format}\n{query}")
model = Tongyi(
    model_name='qwen-max',
    model_kwargs={"temperature":0.1},
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    )
parser = PydanticOutputParser(pydantic_object=LogModel)

while True:
    user_input = input("User: ")
    if user_input == "quit":
        break
    res = model.invoke(prompt.format(
        _format=parser.get_format_instructions(),
        query=user_input
    ))
    log_model_obj = parser.parse(res)
    print(f"AI: {log_model_obj.llm_output}")
    _time = datetime.now().strftime("%H:%M:%S")
    print(f"{_time}: INFO: USER: {log_model_obj.user_input}, AI: {log_model_obj.llm_output}")

