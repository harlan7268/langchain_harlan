import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms.tongyi import Tongyi
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
model = Tongyi(
    model_name='qwen-max',
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_kwargs={'temperature': 0.01},
    streaming=True)
print(model.model_kwargs)
parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

for s in chain.stream({"query": '给我讲100字的笑话'}):
    print(s)


output = model.invoke(prompt.format(query="给我讲100字的笑话"))
print(output)
print(type(output))
print(parser.parse(output))

model_2 = model = Tongyi(
    model_name='qwen-max',
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_kwargs={'temperature': 0.01},
    )

class Joke(BaseModel):
    content: str = Field(description="笑话内容")
    reason: str = Field(description="好笑的原因")

parser_2 = PydanticOutputParser(pydantic_object=Joke)

prompt_2 = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser_2.get_format_instructions()},
)

prompt_2.format(query="给我讲个笑话")

model_2.invoke(prompt_2.format(query="给我讲个笑话"))

joke_obj = parser_2.parse(model_2.invoke(prompt_2.format(query="给我讲个笑话")))

print(joke_obj)
print(joke_obj.content)
print(joke_obj.reason)
# import os
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_community.llms.tongyi import Tongyi

# # 设置 API 密钥
# # os.environ['DASHSCOPE_API_KEY'] = 'your_dashscope_api_key_here'

# # 初始化 Tongyi 模型
# model = Tongyi(
#     model_name='qwen-max',
#     dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
#     # model_kwargs={'temperature': 0.01},
#     streaming=True
# )

# # 定义查询链
# from langchain.chains.llm import LLMChain
# from langchain_core.prompts import PromptTemplate

# template = """给我讲100字的笑话"""
# prompt = PromptTemplate.from_template(template)

# chain = LLMChain(
#     llm=model,
#     prompt=prompt,
#     output_parser=JsonOutputParser()
# )

# # 运行查询链并调试输出信息
# try:
#     for s in chain.stream({"query": '给我讲100字的笑话'}):
#         print(s)
# except TypeError as e:
#     print(f"TypeError: {e}")
#     import traceback
#     traceback.print_exc()
#     # 输出调试信息
#     print("Debug Information:")
#     print("Model Configuration:")
#     print(model.model_kwargs)



# # 自定义合并字典的函数，处理类型不匹配的问题
# def merge_dicts_safe(d1, d2):
#     for k, v in d2.items():
#         if k in d1:
#             if isinstance(d1[k], int) and isinstance(v, int):
#                 d1[k] += v
#             elif isinstance(d1[k], list) and isinstance(v, list):
#                 d1[k].extend(v)
#             else:
#                 raise TypeError(f"Unsupported type for key {k}: {type(d1[k])} vs {type(v)}")
#         else:
#             d1[k] = v
#     return d1

# # 运行查询链并调试输出信息
# try:
#     for s in chain.stream({"query": '给我讲100字的笑话'}):
#         print(s)
# except TypeError as e:
#     print(f"TypeError: {e}")
#     import traceback
#     traceback.print_exc()
#     # 输出调试信息
#     print("Debug Information:")
#     print("Model Configuration:")
#     print(model.model_kwargs)