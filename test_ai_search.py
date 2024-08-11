import os
from typing import List
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from serpapi import Client
from langchain_community.llms.tongyi import Tongyi

class SearchResultItem(BaseModel):
    title: str
    link: str
    snippet: str

class SearchResults(BaseModel):
    results: List[SearchResultItem] 


prompt = PromptTemplate.from_template("""根据搜索引擎结果，回答用户问题
                                      search_results: {search_results}
                                      query: {query}
                                      """)

def get_Search_results(query: str) -> SearchResults:
    params = {
        "engine": "google",
        "q" : query
    }

    client = Client(api_key=os.environ["SERPAPI_KEY"])
    results = client.search(params)
    organic_results = results["organic_results"]
    search_results = SearchResults(
        results=[
            SearchResultItem(
                title=organic_result["title"],
                link=organic_result["link"],
                snippet=organic_result["snippet"]
            )
            for organic_result in organic_results
        ]
    )
    return search_results

model = Tongyi(
    model_name='qwen-max',
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_kwargs={"temperature":0.01},
    )

search_results = get_Search_results("上春山")
chain = {
    "search_results": lambda x: get_Search_results(x),
    "query": lambda x: x,
} | prompt | model

query = "上春山"

r = chain.invoke(query)
print(r)





