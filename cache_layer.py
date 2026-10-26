import redis
from schema import InputSchema , OutputSchema
import json
import hashlib
from rag_pipeline import build_chain_for_video
redis_client = redis.Redis(host='127.0.0.1' , port = 6379 , db = 0)

def gen_cache_key(input:InputSchema):
    raw = json.dumps(input.model_dump() , sort_keys=True)
    return f'cache_key:{hashlib.sha256(raw.encode()).hexdigest()}'

def get_cache_result(input:InputSchema):
    cache_key = gen_cache_key(input)
    cached_result = redis_client.get(cache_key)
    if cached_result:
        print('successfully returned result from cache')
        return OutputSchema(**json.loads(cached_result))
    return None

def save_new_cache(input: InputSchema):
    cache_key = gen_cache_key(input)
    video_identifier = input.video_id
    chain = build_chain_for_video(video_identifier)
    answer = chain.invoke(input.question)
    
    response_data = OutputSchema(
        video_id=input.video_id,
        question=input.question,
        answer=answer
    )
    
    redis_client.setex(cache_key, 600, response_data.model_dump_json())
    return response_data
