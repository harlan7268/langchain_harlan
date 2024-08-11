import os
from langchain.prompts import PromptTemplate
import langchain_community
from langchain_community.llms.tongyi import Tongyi

# for key, value in os.environ.items():
#     print(f"{key}: {value}")
# print(os.environ['DASHSCOPE_API_KEY'])
# print(os.getenv('DASHSCOPE_API_KEY','No API Key Found'))
# print(os.getenv('NUMBER_OF_PROCESSORS', 'No API Key Found'))

print(dir(langchain_community))
pr = PromptTemplate.from_template("讲关于{topic}的故事")
a = pr.format(topic='足球')
print(a)
print(pr)
print(type(pr))

pr_2 = (PromptTemplate.from_template("讲关于{topic}的故事") + ",确保要好笑" + "\n\n使用{language}输出")
b = pr_2.format(topic='足球', language='中文')
print(b)
print(pr_2)
print(type(pr_2)) 

pr_3 = PromptTemplate.from_template("讲关于{topic}的故事")
model = Tongyi(model_name='qwen-max',dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))
t = model.invoke(pr_3.format(topic='足球'))
print(t)



