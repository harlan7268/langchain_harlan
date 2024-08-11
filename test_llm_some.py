from typing import Any, List
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate


class Superman(BaseLLM):

    def _generate(self, prompts: List[str], *args, **kwargs) -> LLMResult:
        results = []
        for prompt in prompts:
            generation_obj = Generation(text=f"copy {prompt}")
            results.append(generation_obj)
        return LLMResult(generations=[results])
    @property
    def _llm_type(self) -> str:
        return "Superman"
    
model = Superman()
print(model)
r = model.invoke("hi")
print(r)

prompt = PromptTemplate.from_template('say hi')
chain = prompt | model
t = chain.invoke({})
print(t)