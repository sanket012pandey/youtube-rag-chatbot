from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnablePassthrough ,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


#model set 
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, max_tokens=1000)

embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")

#parser
parser = StrOutputParser()

#Indexing
#1. doc ingestion
video_id = "7ARBJQn6QkM" #only id needed
api = YouTubeTranscriptApi()

try:
    
    fetched_transcript = api.fetch(video_id, languages=['en'])
    
    transcript = ' '.join(snippet.text for snippet in fetched_transcript)
    

except TranscriptsDisabled:
   print('no available transcript for the video')

#chunking 
splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap = 200
)

chunks = splitter.create_documents([transcript])

#store in db & creating external kb for agent
vector_db = Chroma(
    persist_directory='chroma_db',
    embedding_function= embedding,
    collection_name='transcripts123'
)

#to avoid duplicacy and adding same again and again 
if vector_db._collection.count() == 0:
    vector_db.add_documents(chunks) #external kb prepared

#retriever
retriever = vector_db.as_retriever(
    search_type = 'mmr' , 
    search_kwargs ={'k':4 , 'lambda_mult':0.65}
    )



#prompt template 
prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions strictly based on the given YouTube video transcript.

Instructions:
- Only use information present in the transcript below to answer the question.
- If the transcript does not contain enough information to answer, respond exactly with: "I don't know based on the provided transcript."
- Do not use any outside knowledge or make assumptions beyond what is stated.
- Keep your answer clear and concise.

Transcript:
{context}

Question: {question}

Answer:""",
    input_variables=['context', 'question']
)


#formatting context 
def format_docs(retrieved_doc):
    context_text = '\n\n'.join(docs.page_content for docs in retrieved_doc)
    return context_text

parllel_chain = RunnableParallel(
    {
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
    }
)

main_chain = parllel_chain | prompt | llm | parser

results = main_chain.invoke("sumarize the video in 5 points")
print(results)


if __name__ == "__main__":
    results = main_chain.invoke("sumarize the video in 5 points")
    print(results)
    
    results2 = main_chain.invoke("What does the speaker believe about the future of robotics?")
    print(results2)