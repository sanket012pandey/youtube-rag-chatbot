from pydantic import BaseModel , Field

#input schema
class InputSchema(BaseModel):
    video_id:str = Field(... , description='u will get a video id , eg-> 7ARBJQn6QkM')
    question:str = Field(... , description='user input')


class OutputSchema(BaseModel):
    video_id:str
    question:str
    answer:str = Field(... , description='output genrated by the model')
    

      

