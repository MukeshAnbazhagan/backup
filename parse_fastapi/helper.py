from contextlib import contextmanager
from sqlalchemy import create_engine, text
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
from jose import JWTError, jwt
from sqlalchemy import create_engine, text
import pandas as pd
import json
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
import os
from pathlib import Path
import psycopg2

Base=declarative_base()

security = HTTPBearer()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database


def get_question_mapping(df):
    try:
        mapping = {}
        columns = df.columns.tolist()
        
        answer_columns = [col for col in columns if col.startswith('Answer')]
        max_questions = len(answer_columns)
        
        logger.debug(f"Found {max_questions} answer columns")
        
        # For each question number
        for i in range(1, max_questions + 1):
            answer_col = f'Answer{i}'
            subtitle_col = f'Subtitle{i}'
            
            # Skip if required columns don't exist
            if answer_col not in columns or subtitle_col not in columns:
                continue
                
            question_col = f'Question{i}'
            if question_col not in columns:
                for col in columns:
                    if col not in [answer_col, subtitle_col] and not any(c in col for c in ['Answer', 'Subtitle']):
                        if df[col].notna().any():  # Check if column has any non-NA values
                            first_value = df[col].iloc[0]
                            if pd.notna(first_value) and str(first_value).strip():
                                question_col = col
                                break
            
            if question_col in columns:
                mapping[i] = {
                    "question": question_col,
                    "subtitle": subtitle_col,
                    "answer": answer_col
                }
        
        if not mapping:
            raise ValueError("No valid question mappings found in dataset")
            
        return mapping
        
    except Exception as e:
        logger.error(f"Error creating question mapping: {str(e)}")
        raise
    



def parse_diagnosis_custom(diagnosis_data: dict) -> list:
    """
    Parse diagnosis data from the JSON format.
    """
    if not diagnosis_data:
        return []

    try:
        return [
            {
                "no": str(diagnosis_data.get("no",diagnosis_data.get("diagnosis_no", ""))).strip(),
                "title": diagnosis_data.get("title",diagnosis_data.get("diagnosis_title", "")).strip(),
                "description": diagnosis_data.get("description",diagnosis_data.get("diagnosis_description", "")).strip()
            }
        ]
    except Exception as e:
        print(f"Error parsing diagnosis: {e}")
        return []
    
'''
def parse_advice_custom(advice_data: list) -> list:
    """
    Parse advice data from the JSON format.
    """
    if not advice_data:
        return []

    try:
        return [
            {
                "no": str(item.get("no",item.get("advice_no", ""))).strip(),
                "title": item.get("title",item.get("advice_title", "")).strip(),
                "description": item.get("description",item.get("advice_description", "")).strip()
            }
            for item in advice_data
        ]
    except Exception as e:
        print(f"Error parsing advice: {e}")
        return []

def parse_red_flags_custom(red_flags_data: list) -> list:
    """
    Parse red flags data from the JSON format.
    """
    if not red_flags_data:
        return []

    try:
        return [
            {
                "no": str(item.get("no",item.get("red_flag_no", ""))).strip(),
                "description": item.get("description",item.get("red_flag_description", "")).strip()
            }
            for item in red_flags_data
        ]
    except Exception as e:
        print(f"Error parsing red flags: {e}")
        return []

def parse_lab_test_custom(lab_test_data: list) -> list:
    """
    Parse lab test data from the JSON format.
    """
    if not lab_test_data:
        return []

    try:
        return [
            {
                "no": str(item.get("no",item.get("lab_test_no", ""))).strip(),
                "title": item.get("title",item.get("lab_test_title", "")).strip()
            }
            for item in lab_test_data
        ]
    except Exception as e:
        print(f"Error parsing lab tests: {e}")
        return []

    
def parse_precaution_custom(precaution_data: list) -> list:
    """
    Parse precaution data from the JSON format.
    """
    if not precaution_data:
        return []

    try:
        return [
            {
                "no": str(item.get("no",item.get("precaution_no", ""))).strip(),
                "title": item.get("title",item.get("precaution_title", "")).strip(),
                "description": item.get("description",item.get("precaution_description", "")).strip()
            }
            for item in precaution_data
        ]
    except Exception as e:
        print(f"Error parsing precautions: {e}")
        return []
'''

def parse_otc_custom(otc_data: list) -> list:
    """
    Parse OTC medication data from the JSON format.
    """
    if not otc_data:
        return []

    try:
        parsed_list = []
        for item in otc_data:
            parsed_list.append({
                "no": str(item.get("no",item.get("otc_no", ""))).strip(),
                "title": item.get("title",item.get("otc_title", "")).strip(),
                "medicine_name": item.get("medicine_name", (item.get("medicine",item.get("otc_medicine_name", "")))).strip(),
                "dosage_duration": item.get("dosage_duration", item.get("otc_dosage_duration", "")).strip(),
                "type": item.get("type", item.get("otc_type", "")).strip(),
                "intake_type": item.get("intake_type", item.get("otc_intake_type", "")).strip(),
                "intake_schedules": item.get("intake_schedules", item.get("otc_intake_schedules", "")).strip()
            })

        return parsed_list

    except Exception as e:
        print(f"Error parsing OTC medications: {e}")
        return []


def parse_dietary_advice(text):
    if not text or text == '-':
        return None
    if pd.isna(text):
        return None
    advices = []
    parts = text.split(';')
    current_advice = {}
    for part in parts:
        if 'dietry_advice.no' in part:
            if current_advice:
                advices.append(current_advice)
            current_advice = {}
            current_advice['no'] = part.split(':')[1].strip()
        elif ':' in part:
            key = part.split('.')[1].split(':')[0].strip()
            value = part.split(':')[1].strip()
            current_advice[key] = value
    if current_advice:
        advices.append(current_advice)
    return advices


def parse_advice_custom(advice_data: list) -> list:
    """
    Parse advice data from a list of strings to list of dicts with 'no' and 'title'.
    """
    if not advice_data:
        return []

    try:
        return [
            {
                "no": str(i + 1),
                "title": advice.strip()
            }
            for i, advice in enumerate(advice_data)
        ]
    except Exception as e:
        print(f"Error parsing advice: {e}")
        return []

def parse_red_flags_custom(red_flags_data: list) -> list:
    """
    Parse red flags data from a list of strings to list of dicts with 'no' and 'title'.
    """
    if not red_flags_data:
        return []

    try:
        return [
            {
                "no": str(i + 1),
                "title": flag.strip()
            }
            for i, flag in enumerate(red_flags_data)
        ]
    except Exception as e:
        print(f"Error parsing red flags: {e}")
        return []

def parse_lab_test_custom(lab_test_data: list) -> list:
    """
    Parse lab test data from a list of strings to list of dicts with 'no' and 'title'.
    """
    if not lab_test_data:
        return []

    try:
        return [
            {
                "no": str(i + 1),
                "title": test.strip()
            }
            for i, test in enumerate(lab_test_data)
        ]
    except Exception as e:
        print(f"Error parsing lab tests: {e}")
        return []

def parse_precaution_custom(precaution_data: list) -> list:
    """
    Parse precaution data from a list of strings to list of dicts with 'no' and 'title'.
    """
    if not precaution_data:
        return []

    try:
        return [
            {
                "no": str(i + 1),
                "title": precaution.strip()
            }
            for i, precaution in enumerate(precaution_data)
        ]
    except Exception as e:
        print(f"Error parsing precautions: {e}")
        return []
      

def store_qna_2(conn, user_id, qna_session_id, curr_q_num, question, answer):
    query = text("""
        INSERT INTO qna_session_2(qna_session_id, user_id, q_id, question, answer)
        VALUES (:qna_session_id, :user_id, :q_id, :question, :answer)
    """)
    conn.execute(query, {
        'qna_session_id': qna_session_id,
        'user_id': user_id,
        'q_id': curr_q_num,
        'question': question,
        'answer': answer
    })
    conn.commit()


def clean_and_renumber(data: list, keys: list) -> list:
    """
    Clean and renumber a list of dictionaries.
    """
    cleaned = []
    for idx, item in enumerate(data):
        cleaned_item = {k: item.get(k, "").strip() for k in keys}
        cleaned_item["no"] = str(idx + 1)
        cleaned.append(cleaned_item)
    return cleaned


def filter_empty_sections(result: dict) -> dict:
    """
    Filter out empty sections from the result.
    """
    filtered = {}
    for key, value in result.items():
        if isinstance(value, list) and len(value) > 0:
            filtered[key] = value
        elif value:
            filtered[key] = value
    return filtered