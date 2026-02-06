from pydantic import BaseModel, ConfigDict, Field
from typing import List


class CustomerRecord(BaseModel):
    # Beklenmeyen alan gelirse 422 ile reddet.
    model_config = ConfigDict(extra="forbid")

    Gender: int
    Senior_Citizen: int
    Partner: int
    Dependents: int
    Tenure_Months: int
    Phone_Service: int
    Paperless_Billing: int
    Monthly_Charges: float
    Total_Charges: float
    CLTV: float

    Multiple_Lines_No: int
    Multiple_Lines_No_phone_service: int
    Multiple_Lines_Yes: int

    Internet_Service_DSL: int
    Internet_Service_Fiber_optic: int
    Internet_Service_No: int

    Online_Security_No: int
    Online_Security_No_internet_service: int
    Online_Security_Yes: int

    Online_Backup_No: int
    Online_Backup_No_internet_service: int
    Online_Backup_Yes: int

    Device_Protection_No: int
    Device_Protection_No_internet_service: int
    Device_Protection_Yes: int

    Tech_Support_No: int
    Tech_Support_No_internet_service: int
    Tech_Support_Yes: int

    Streaming_TV_No: int
    Streaming_TV_No_internet_service: int
    Streaming_TV_Yes: int

    Streaming_Movies_No: int
    Streaming_Movies_No_internet_service: int
    Streaming_Movies_Yes: int

    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int

    Payment_Method_Bank_transfer_automatic: int
    Payment_Method_Credit_card_automatic: int
    Payment_Method_Electronic_check: int
    Payment_Method_Mailed_check: int


class PredictionRequest(BaseModel):
    # Request içinde ek alan olmasın; records boş gelmesin.
    model_config = ConfigDict(extra="forbid")

    records: List[CustomerRecord] = Field(min_length=1)
