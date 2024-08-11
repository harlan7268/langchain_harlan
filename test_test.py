from langchain_community.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
responses = [
    "您好",
    "我能做什么",
    "end"
]

fake_list_llm = FakeListLLM(responses=responses)
prompt = PromptTemplate.from_template("")
print(fake_list_llm.invoke(prompt.format()))
print(fake_list_llm.invoke(prompt.format()))
print(fake_list_llm.invoke(prompt.format()))
print(fake_list_llm.invoke(prompt.format()))
