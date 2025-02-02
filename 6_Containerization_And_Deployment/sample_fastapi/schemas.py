from pydantic import BaseModel
from typing import Literal


class studentRequestModel(BaseModel):

    class_name : Literal["ML_in_Prod_1","ML_in_Prod_2", "Big_Data"]

    stu_name: str = "Mg ba"
    stu_id : int = 123

