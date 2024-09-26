from langchain_community.llms import Ollama

def GenerateResponse(retrieved_docs, query, model_name, promptTemplate):

    if len(retrieved_docs) >= 1:
        combined_context = " ".join([doc.page_content for doc in retrieved_docs])
        prompt = promptTemplate.replace("[context_str]", combined_context).replace("[query_str]", query)
        llm = Ollama(model = model_name)
        response = llm.invoke(prompt)

        return response