# few-shot Prompt template,使用少量示例提示模版
import os
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings



examples = [
    {
        "question":"你好",
        "answer":"""'你好'这不是学科问题，回答：'你好，请说出具体的学科问题'"""
    },
    {
        "question":"介绍下清朝历史",
        "answer":"""'介绍下清朝的历史'是历史问题，回答：'请联系李老师'"""
    },
    {
        "question":"东京是哪个国家的",
        "answer":"""'东京是哪个国家的'是地理问题，回答：'请联系倪老师'"""
    }
]

example_prompt = PromptTemplate(
    input_variables=["question","answer"],template="Question: {question}\n{answer}"
)

# print(example_prompt.format(**examples[0]))


all_example_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请参考下面的示例，回答问题：\n<example>",
    suffix="</example>\n\nQuestion: {input}\nAI:",
    input_variables=["input"],
) 

question = "元朝最出名的历史人物是谁"

# print(all_example_prompt.format(input=question))

model = Tongyi(model_name='qwen-max',dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))
t = model.invoke(all_example_prompt.format(input=question))
# print(t)


embeddings = DashScopeEmbeddings(model="text-embedding-v1")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Chroma,
    k=2,
)

selected_examples = example_selector.select_examples({"question":question})
# print(f"Examples most similar to the input: {question}")

# for example in selected_examples:
#     print("\n")
#     for k, v in example.items():
#         print(f"{k}: {v}")

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="请参考下面的示例，回答问题：\n<example>",
    suffix="</example>\n\nQuestion: {input}\nAI:",
    input_variables=["input"],
)

print(few_shot_prompt.format(input=question))
