from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#models 
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, max_tokens=1000)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#structuring output
parser = StrOutputParser()

#spliiting text accordingly meaning
text_splitter  = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    min_chunk_size=300
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions strictly based on the given YouTube video transcript.

Instructions:
- Only use information present in the transcript below to answer the question.
- Each chunk of transcript starts with a timestamp in brackets, like [4:07] (minutes:seconds).
- When citing a timestamp, copy the exact value shown in brackets — do not calculate or estimate it yourself.
- If the transcript does not contain enough information to answer, respond exactly with: "I don't know based on the provided transcript."
- Do not use any outside knowledge or make assumptions beyond what is stated.
- Cite the approximate timestamp where the answer occurs.
- Keep your answer clear and concise.

Transcript:
{context}"""),
    ("human", "{question}\n\nAnswer:")
])


def format_docs(retrieved_doc):
    return '\n\n'.join(doc.page_content for doc in retrieved_doc)


def build_chain_for_video(video_id: str):
    # fetch transcript for this video
    api = YouTubeTranscriptApi()

    try:
        fetched_transcript = api.fetch(video_id, languages=['en'])
        transcript = ' '.join(snippet.text for snippet in fetched_transcript)
    except TranscriptsDisabled:
        raise ValueError("No transcript available for this video")

    #chunks
    chunks = text_splitter.create_documents([transcript])

    # separate Chroma collection per video (so videos don't mix in search results)

    vector_db = Chroma(
        persist_directory='chroma_db',
        embedding_function=embedding,
        collection_name='transcripts123'
    )

    if vector_db._collection.count() == 0:
        vector_db.add_documents(chunks)

    retriever = vector_db.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 4, 'lambda_mult': 0.65}
    )

    parllel_chain = RunnableParallel(
        {
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        }
    )

    chain = parllel_chain | prompt | llm | parser
    return chain