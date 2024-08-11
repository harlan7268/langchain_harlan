from langchain_experimental.llm_bash.base import LLMBashChain
from langchain_community.llms.tongyi import Tongyi
from langchain.prompts import PromptTemplate
from langchain.chains.api.base import APIChain
from langchain_core.runnables import RunnableLambda


model = Tongyi(
    model_name="qwen-max",
    model_kwargs={"temperature": 0.001}
)

bash_chain = LLMBashChain.from_llm(model)
query = "查询操作系统所有的磁盘使用情况"
a = bash_chain.invoke(query)
print(a)



prompt = PromptTemplate.from_template("""{context}\n\n根据如上的巡检结果，判断是否需要告警，如果需要请总结并返回告警内容，否则返回空字符串""")

alarm_chain = {
    "context": lambda x: bash_chain.invoke(query)["answer"]
} | prompt | model

b = alarm_chain.invoke({})
print(b)

HTTPBIN_DOCS = """
# API 使用文档

## 概述

此API用于向指定的URL发送告警信息。当系统检测到特定告警条件时，可以调用此API告之管理员或者记录系统的状态。

## API 信息

- **URL**: http://httpbin.org/get
- **业务说明**: 发送告警信息到服务器，用于系统告警通知或者日志记录。
- **请求方式**: `GET`

## 请求参数

| 参数名 | 类型 | 描述                        | 是否必须 | 实例值     |
| --------| ----------| --------------------| -------| ------------|
| alarm | string | 具体的告警信息描述  | 是  | "alarm information"  |

"""

api_chian = APIChain.from_llm_and_api_docs(
    llm=model,
    api_docs=HTTPBIN_DOCS,
    limit_to_domains=["http://httpbin.org/get"],
    verbose=True
    )

alarm_api_chain = alarm_chain | api_chian
c = alarm_api_chain.invoke({})
print(c)





