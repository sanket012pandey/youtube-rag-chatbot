from fastapi import FastAPI , HTTPException
from schema import InputSchema , OutputSchema
from rag_pipeline import build_chain_for_video
from cache_layer import get_cache_result, save_new_cache

app = FastAPI(title='YouTube RAG Chatbot API')



@app.post('/ask' , response_model=OutputSchema)
def ask_question(request:InputSchema):
    try:
        cached_response = get_cache_result(request)
        if cached_response:
            return cached_response

        
        response_data = save_new_cache(request)
        return response_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
