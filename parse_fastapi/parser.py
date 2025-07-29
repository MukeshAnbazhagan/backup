# import json
# import uuid
# from fastapi import APIRouter, Depends, HTTPException, status
# from typing import Dict, Any, Optional
# import pandas as pd
# import logging
# from sqlalchemy.orm import Session
# from sqlalchemy import text
# from pydantic import BaseModel, EmailStr, StringConstraints, Field
# from typing import Optional, Dict, Any, List
# from typing_extensions import Annotated
# from datetime import datetime
# from database import get_db
# from helper import (
#     filter_empty_sections,
#     parse_advice_custom,
#     parse_diagnosis_custom,
#     parse_otc_custom,
#     parse_red_flags_custom,
#     parse_dietary_advice,
#     parse_lab_test_custom,
#     parse_precaution_custom,
#     clean_and_renumber
# )


# class PredictRequest(BaseModel):
#     answers: Dict[str, str] = {}

# class JsonProtocolPredictor:
#     def __init__(self, protocol_name: str, conn):
#         self.protocol_name = protocol_name
#         self.conn = conn

#         full_uuid = str(uuid.uuid4())
#         unique_id = full_uuid[:8] + full_uuid[-8:]
#         self.qna_session_id = unique_id
#         self.logger = logging.getLogger(__name__)
#         self.questions = self._load_json_from_db()
#         self.question_map = {q["q_id"]: q for q in self.questions}
#         self.display_question_counter = 1
#         self.display_to_actual_qid_map = {}

#     def _load_json_from_db(self) -> list:
#         # Use SQLAlchemy's text() for raw SQL queries
#         result = self.conn.execute(
#             text("SELECT decision_tree FROM protocol WHERE LOWER(protocol_name) = :protocol_name"),
#             {"protocol_name": self.protocol_name.lower()}
#         ).fetchone()

#         if not result:
#             raise Exception(f"Protocol '{self.protocol_name}' not found in the database")
        
#         try:
#             return result[0]

#         except json.JSONDecodeError as e:
#             raise Exception(f"Invalid JSON format in database: {e}")


#     def _predict(self, current_qid: int, answers: dict) -> dict:
#         while current_qid in self.question_map:
#             node = self.question_map[current_qid]
#             answer_key = f"Q{self.display_question_counter}"
#             self.display_to_actual_qid_map[answer_key] = current_qid

#             if answer_key not in answers:
#                 return node

#             selected_answer = answers[answer_key].strip().lower()
#             for option in node.get("options", []):
#                 if option["opt_value"].strip().lower() == selected_answer:
#                     if "diagnosis" in node:  
#                         return {"diagnosis": node}
#                     current_qid = option.get("next_q_id")
#                     self.display_question_counter += 1
#                     break
#             else:
#                 raise Exception(f"No matching option for question Q{current_qid}")
#         raise Exception("Reached invalid or undefined question flow.")
    
#     def _find_actual_qid_from_display(self, display_number: int, answers: dict) -> int:
#         current_qid = self.questions[0]["q_id"]  # Start from first question

#         # Rebuild path based on answers and map display Qn to actual q_id
#         self.display_question_counter = 1
#         self.display_to_actual_qid_map = {}

#         for i in range(1, display_number + 1):
#             answer_key = f"Q{i}"
#             node = self.question_map.get(current_qid)
#             self.display_to_actual_qid_map[answer_key] = current_qid

#             if answer_key not in answers:
#                 break

#             selected_answer = answers[answer_key].strip().lower()
#             matched = False
#             for option in node.get("options", []):
#                 if option["opt_value"].strip().lower() == selected_answer:
#                     current_qid = option.get("next_q_id")
#                     matched = True
#                     self.display_question_counter += 1
#                     break

#             if not matched:
#                 raise Exception(f"No matching option for question {answer_key}")

#         return current_qid


#     def predict(self, question_request: dict) -> dict:
#         try:
#             answers = question_request.get("answers", {})

#             if not answers:
#                 self.display_question_counter = 1
#                 # Get the first question from the questions list
#                 if not self.questions:
#                     raise Exception("No questions found in the protocol")
#                 current_qid = self.questions[0]["q_id"]
#                 print(current_qid)

#             else:
#                 # Find the highest displayed question number answered
#                 answered_numbers = [int(k[1:]) for k in answers.keys()]
#                 last_displayed_q = max(answered_numbers)
#                 self.display_question_counter = last_displayed_q
                
#                 # Find the actual current_qid from the last answered question
#                 # This requires tracking the mapping between display numbers and actual q_ids
#                 # You might need to store this mapping in the session or reconstruct it
#                 current_qid = self._find_actual_qid_from_display(last_displayed_q, answers)
#                 print(last_displayed_q)
                
#             result_node = self._predict(current_qid, answers)
#             if result_node.get("q_tag") == "DIAGNOSIS":
#                 node = result_node
#                 diag = parse_diagnosis_custom(node.get("diagnosis", {})) or []
#                 otc = parse_otc_custom(node.get("otc_medication", [])) or []
#                 adv = parse_advice_custom(node.get("advice", [])) or []
#                 prec = parse_precaution_custom(node.get("precautions", [])) or []
#                 rf = parse_red_flags_custom(node.get("red_flags", [])) or []
#                 lab = parse_lab_test_custom(node.get("lab_tests", [])) or []

#                 # Clean and renumber data
#                 diag = clean_and_renumber(diag, ["no", "title", "description"])
#                 otc = clean_and_renumber(otc, ["no", "title", "medicine_name", "dosage_duration", "type", "intake_type", "intake_schedules"])
#                 adv = clean_and_renumber(adv, ["no", "title"])
#                 prec = clean_and_renumber(prec, ["no", "title"])
#                 rf = clean_and_renumber(rf, ["no", "title"])
#                 lab = clean_and_renumber(lab, ["no", "title"])

#                 result = {
#                     "Diagnosis": diag,
#                     "Over the counter medication": otc,
#                     "Advice": adv,
#                     "Precautions": prec,
#                     "Red_flags": rf,
#                     "Lab tests": lab
#                 }
                

#                 result_1 = {
#                     "is_final" : True,
#                     "Diagnosis": diag,
#                     "Over the counter medication": otc,
#                     "Advice": adv,
#                     "Precautions": prec,
#                     "Red_flags": rf,
#                     "Lab tests": lab
#                 }

#                 return filter_empty_sections(result_1) | {"is_final": True}

#             return {
#                 "is_final": False,
#                 "qid": f"Q{self.display_question_counter}",
#                 "q_tag": result_node.get("q_tag", ""),
#                 "question": result_node.get("question", ""),
#                 "options": [opt.get("opt_value", "") for opt in result_node.get("options", [])]
#             }
#         except Exception as e:
#             self.logger.error(f"Error processing prediction request: {str(e)}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Error processing request: {str(e)}"
#             )
        

# def create_protocol_router_llm(protocol_name: str) -> APIRouter:
#     router = APIRouter(prefix=f"/updated-template/{protocol_name.lower()}", tags=[protocol_name])

#     @router.post("/predict")
#     async def predict_endpoint(
#         question_request: PredictRequest,
#         conn = Depends(get_db)  # inject DB connection
#     ):
#         try:
#             predictor = JsonProtocolPredictor(protocol_name, conn)
#             return predictor.predict(question_request.dict())
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=str(e))

#     return router


#  # type: ignore



import json
import uuid
import os
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional
import pandas as pd
import logging
from pydantic import BaseModel, EmailStr, StringConstraints, Field
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated
from datetime import datetime
from helper import (
    filter_empty_sections,
    parse_advice_custom,
    parse_diagnosis_custom,
    parse_otc_custom,
    parse_red_flags_custom,
    parse_dietary_advice,
    parse_lab_test_custom,
    parse_precaution_custom,
    clean_and_renumber
)


class PredictRequest(BaseModel):
    answers: Dict[str, str] = {}

class JsonProtocolPredictor:
    def __init__(self, protocol_name: str, json_directory: str = "./protocols"):
        self.protocol_name = protocol_name
        self.json_directory = json_directory

        full_uuid = str(uuid.uuid4())
        unique_id = full_uuid[:8] + full_uuid[-8:]
        self.qna_session_id = unique_id
        self.logger = logging.getLogger(__name__)
        self.questions = self._load_json_from_file()
        self.question_map = {q["q_id"]: q for q in self.questions}
        self.display_question_counter = 1
        self.display_to_actual_qid_map = {}

    def _load_json_from_file(self) -> list:
        # Construct the file path based on protocol name
        filename = f"{self.protocol_name.lower()}.json"
        filepath = os.path.join(self.json_directory, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise Exception(f"Protocol file '{filename}' not found in directory '{self.json_directory}'")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # If the JSON structure has the data nested under a key (like 'decision_tree'), 
                # you might need to access it. Adjust this based on your JSON structure:
                # return data['decision_tree']  # If nested
                return data  # If the JSON file contains the questions array directly
                
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format in file '{filename}': {e}")
        except Exception as e:
            raise Exception(f"Error reading file '{filename}': {e}")

    def _predict(self, current_qid: int, answers: dict) -> dict:
        while current_qid in self.question_map:
            node = self.question_map[current_qid]
            answer_key = f"Q{self.display_question_counter}"
            self.display_to_actual_qid_map[answer_key] = current_qid

            if answer_key not in answers:
                return node

            selected_answer = answers[answer_key].strip().lower()
            for option in node.get("options", []):
                if option["opt_value"].strip().lower() == selected_answer:
                    if "diagnosis" in node:  
                        return {"diagnosis": node}
                    current_qid = option.get("next_q_id")
                    self.display_question_counter += 1
                    break
            else:
                raise Exception(f"No matching option for question Q{current_qid}")
        raise Exception("Reached invalid or undefined question flow.")
    
    def _find_actual_qid_from_display(self, display_number: int, answers: dict) -> int:
        current_qid = self.questions[0]["q_id"]  # Start from first question

        # Rebuild path based on answers and map display Qn to actual q_id
        self.display_question_counter = 1
        self.display_to_actual_qid_map = {}

        for i in range(1, display_number + 1):
            answer_key = f"Q{i}"
            node = self.question_map.get(current_qid)
            self.display_to_actual_qid_map[answer_key] = current_qid

            if answer_key not in answers:
                break

            selected_answer = answers[answer_key].strip().lower()
            matched = False
            for option in node.get("options", []):
                if option["opt_value"].strip().lower() == selected_answer:
                    current_qid = option.get("next_q_id")
                    matched = True
                    self.display_question_counter += 1
                    break

            if not matched:
                raise Exception(f"No matching option for question {answer_key}")

        return current_qid

    def predict(self, question_request: dict) -> dict:
        try:
            answers = question_request.get("answers", {})

            if not answers:
                self.display_question_counter = 1
                # Get the first question from the questions list
                if not self.questions:
                    raise Exception("No questions found in the protocol")
                current_qid = self.questions[0]["q_id"]
                print(current_qid)

            else:
                # Find the highest displayed question number answered
                answered_numbers = [int(k[1:]) for k in answers.keys()]
                last_displayed_q = max(answered_numbers)
                self.display_question_counter = last_displayed_q
                
                # Find the actual current_qid from the last answered question
                # This requires tracking the mapping between display numbers and actual q_ids
                # You might need to store this mapping in the session or reconstruct it
                current_qid = self._find_actual_qid_from_display(last_displayed_q, answers)
                print(last_displayed_q)
                
            result_node = self._predict(current_qid, answers)
            if result_node.get("q_tag") == "DIAGNOSIS":
                node = result_node
                diag = parse_diagnosis_custom(node.get("diagnosis", {})) or []
                otc = parse_otc_custom(node.get("otc_medication", [])) or []
                adv = parse_advice_custom(node.get("advice", [])) or []
                prec = parse_precaution_custom(node.get("precautions", [])) or []
                rf = parse_red_flags_custom(node.get("red_flags", [])) or []
                lab = parse_lab_test_custom(node.get("lab_tests", [])) or []

                # Clean and renumber data
                diag = clean_and_renumber(diag, ["no", "title", "description"])
                otc = clean_and_renumber(otc, ["no", "title", "medicine_name", "dosage_duration", "type", "intake_type", "intake_schedules"])
                adv = clean_and_renumber(adv, ["no", "title"])
                prec = clean_and_renumber(prec, ["no", "title"])
                rf = clean_and_renumber(rf, ["no", "title"])
                lab = clean_and_renumber(lab, ["no", "title"])

                result = {
                    "Diagnosis": diag,
                    "Over the counter medication": otc,
                    "Advice": adv,
                    "Precautions": prec,
                    "Red_flags": rf,
                    "Lab tests": lab
                }
                
                result_1 = {
                    "is_final" : True,
                    "Diagnosis": diag,
                    "Over the counter medication": otc,
                    "Advice": adv,
                    "Precautions": prec,
                    "Red_flags": rf,
                    "Lab tests": lab
                }

                return filter_empty_sections(result_1) | {"is_final": True}

            return {
                "is_final": False,
                "qid": f"Q{self.display_question_counter}",
                "q_tag": result_node.get("q_tag", ""),
                "question": result_node.get("question", ""),
                "options": [opt.get("opt_value", "") for opt in result_node.get("options", [])]
            }
        except Exception as e:
            self.logger.error(f"Error processing prediction request: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing request: {str(e)}"
            )
        

def create_protocol_router_llm(protocol_name: str, json_directory: str = "./protocols") -> APIRouter:
    router = APIRouter(prefix=f"/updated-template/{protocol_name.lower()}", tags=[protocol_name])

    @router.post("/predict")
    async def predict_endpoint(question_request: PredictRequest):
        try:
            predictor = JsonProtocolPredictor(protocol_name, json_directory)
            return predictor.predict(question_request.dict())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return router


 # type: ignore