#so now we are creating a graph
#nd the first thing you create a stae
import os
#1 typed dict

from typing import TypedDict

class State(TypedDict):
    topic : str 
    summary : str
    score : str


#2 pydantic model

#it is good at datavalidation and type checking

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str
    score : int
    summary  : str = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
         raise ValueError("score must be positive")

        #python dataclasees
        # 
        # #standard python dataclass but it is used very rarely
        # 
from dataclasses import dataclass, field
@dataclass

class State:
    topic : str = ""
    summary : str = ""
    messages : list = field(default_factory=list)

from langgraph.graph import MessagesState
class State(MessagesState):
    user_name : str
    language: str
#mesages field is already included with add_mesages reducer
#just ass your extra fields




        

