from fastapi import FastAPI , HTTPException
from schema import InputSchema , OutputSchema
from rag_pipeline import build_chain_for_video

app = FastAPI(title='YouTube RAG Chatbot API')



@app.post('/ask' , response_model=OutputSchema)
def ask_question(request:InputSchema):
    try:
        chain = build_chain_for_video(request.video_id)
        answer = chain.invoke(request.question)
        return OutputSchema(
            video_id=request.video_id,
            question=request.question,
            answer=answer
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
