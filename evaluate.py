from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from datasets import Dataset
from Youtube_Chatbot import main_chain, retriever, embedding

# Separate, higher-token-limit LLM just for judging
judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=4000)

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding)

answer_relevancy_metric = AnswerRelevancy(strictness=1)

test_questions = [
    "What does the speaker believe about the future of robotics?"
]

results = []
for q in test_questions:
    retrieved_docs = retriever.invoke(q)
    context = [doc.page_content for doc in retrieved_docs]
    answer = main_chain.invoke(q)
    
    results.append({
        "question": q,
        "answer": answer,
        "contexts": context,
    })
    print(f"Done: {q}")

eval_dataset = Dataset.from_list(results)

scores = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy_metric],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)
print(scores)