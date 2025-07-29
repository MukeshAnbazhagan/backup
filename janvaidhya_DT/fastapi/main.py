# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional, Generator
# import psycopg2
# import psycopg2.extras
# import json
# import itertools
# import time
# from contextlib import contextmanager

# app = FastAPI(title="Simple Decision Tree API", version="1.0.0")

# # Database Configuration
# DB_CONFIG = {
#     "host": "localhost",
#     "port": 5432,
#     "database": "postgres",
#     "user": "postgres",
#     "password": "Muki@1302"
# }

# @contextmanager
# def get_db_connection():
#     conn = None
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         yield conn
#     except Exception as e:
#         if conn:
#             conn.rollback()
#         raise e
#     finally:
#         if conn:
#             conn.close()

# class SeparateCategoryDecisionTreeGenerator:
#     def __init__(self):
#         self.age_groups = ["0-2", "3-12", "13-18", "19-40", "41-65", "66+"]
#         self.genders = ["Male", "Female"]
        
#     def load_questions_json(self, questions_data: List[Dict]) -> Dict:
#         return {"questions": questions_data}

#     def organize_questions_by_demographics(self, questions_data: List[Dict]) -> Dict:
#         questions_by_category = {}
        
#         for question in questions_data:
#             age_group = question.get("age_group")
#             gender = question.get("gender", "Both")
            
#             if not age_group:
#                 continue
                
#             if age_group not in questions_by_category:
#                 questions_by_category[age_group] = {}
                
#             if gender not in questions_by_category[age_group]:
#                 questions_by_category[age_group][gender] = []
                
#             questions_by_category[age_group][gender].append(question)
        
#         combined_questions = {}
        
#         for age_group in self.age_groups:
#             if age_group not in questions_by_category:
#                 continue
                
#             combined_questions[age_group] = {}
            
#             both_questions = questions_by_category[age_group].get("Both", [])
#             male_questions = questions_by_category[age_group].get("Male", [])
#             female_questions = questions_by_category[age_group].get("Female", [])
            
#             if age_group in ["0-2", "3-12"]:
#                 if both_questions:
#                     combined_questions[age_group]["Male"] = both_questions.copy()
#                     combined_questions[age_group]["Female"] = both_questions.copy()
#             else:
#                 if both_questions or male_questions:
#                     combined_questions[age_group]["Male"] = both_questions + male_questions
#                 if both_questions or female_questions:
#                     combined_questions[age_group]["Female"] = both_questions + female_questions
        
#         return combined_questions

#     def create_separate_category_trees(self, combined_questions: Dict) -> Dict[str, Dict]:
#         category_trees = {}
        
#         for age_group in self.age_groups:
#             if age_group not in combined_questions:
#                 continue
                
#             for gender in self.genders:
#                 if gender not in combined_questions[age_group]:
#                     continue
                    
#                 category_key = f"{age_group}_{gender}"
#                 questions_list = combined_questions[age_group][gender]
                
#                 if not questions_list:
#                     continue
                
#                 decision_tree = self._create_complete_category_tree(age_group, gender, questions_list)
                
#                 if decision_tree:
#                     category_trees[category_key] = {
#                         "age_category": age_group,
#                         "gender": gender,
#                         "total_nodes": len(decision_tree),
#                         "decision_tree": decision_tree
#                     }
        
#         return category_trees

#     def _create_complete_category_tree(self, target_age: str, target_gender: str, questions_list: List[Dict]) -> List[Dict]:
#         decision_tree = []
#         current_q_id = 1
#         current_id = 1
        
#         # Gender Question
#         gender_question = {
#             "id": current_id,
#             "q_id": current_q_id,
#             "q_tag": "GENERAL",
#             "question": "What is your gender?",
#             "options": []
#         }
        
#         for i, gender in enumerate(["Male", "Female"]):
#             option = {
#                 "option_id": i + 1,
#                 "opt_value": gender,
#                 "next_q_id": 2 if gender == target_gender else None,
#                 "nextQuestion": 2 if gender == target_gender else None,
#                 "childQuestion": 2 if gender == target_gender else None,
#                 "childDiagnosis": None
#             }
#             gender_question["options"].append(option)
        
#         decision_tree.append(gender_question)
#         current_q_id += 1
#         current_id += 1
        
#         # Age Question
#         age_question = {
#             "id": current_id,
#             "q_id": current_q_id,
#             "q_tag": "GENERAL", 
#             "question": "What is your age group?",
#             "options": []
#         }
        
#         for i, age_group in enumerate(self.age_groups):
#             option = {
#                 "option_id": i + 1,
#                 "opt_value": age_group,
#                 "next_q_id": 3 if age_group == target_age else None,
#                 "nextQuestion": 3 if age_group == target_age else None,
#                 "childQuestion": 3 if age_group == target_age else None,
#                 "childDiagnosis": None
#             }
#             age_question["options"].append(option)
        
#         decision_tree.append(age_question)
#         current_q_id += 1
#         current_id += 1
        
#         # Actual questions
#         if questions_list:
#             questions_tree = self._create_questions_tree(questions_list, current_q_id, current_id)
#             if questions_tree:
#                 decision_tree.extend(questions_tree)
        
#         return decision_tree

#     def _create_questions_tree(self, questions_list: List[Dict], start_q_id: int, start_id: int) -> List[Dict]:
#         questions_tree = []
#         current_q_id = start_q_id
#         current_id = start_id
        
#         all_paths = self._generate_all_answer_combinations(questions_list)
        
#         if not all_paths:
#             return questions_tree
        
#         self._create_branching_questions_recursive(
#             questions_list, all_paths, 0, current_q_id, current_id, questions_tree
#         )
        
#         return questions_tree

#     def _create_branching_questions_recursive(self, questions_list: List[Dict], all_paths: List[tuple], 
#                                             question_index: int, current_q_id: int, current_id: int, 
#                                             questions_tree: List[Dict]) -> int:
        
#         if question_index >= len(questions_list):
#             return None
        
#         current_question = questions_list[question_index]
#         question_id = current_q_id
        
#         question_node = {
#             "id": current_id,
#             "q_id": question_id,
#             "q_tag": current_question.get("q_tag", "SYMPTOMS"),
#             "question": current_question["question"],
#             "options": []
#         }
        
#         paths_by_answer = {}
#         for path in all_paths:
#             if len(path) > question_index:
#                 answer = path[question_index]
#                 if answer not in paths_by_answer:
#                     paths_by_answer[answer] = []
#                 paths_by_answer[answer].append(path)
        
#         next_available_q_id = current_q_id + 1
        
#         for option in current_question.get("options", []):
#             option_value = option["opt_value"]
            
#             if option_value in paths_by_answer:
#                 filtered_paths = paths_by_answer[option_value]
                
#                 if question_index == len(questions_list) - 1:
#                     next_q_id = None
#                 else:
#                     next_q_id = next_available_q_id
#                     next_available_q_id = self._create_branching_questions_recursive(
#                         questions_list,
#                         filtered_paths,
#                         question_index + 1,
#                         next_available_q_id,
#                         current_id + 1 + len([q for q in questions_tree if q["q_id"] >= next_available_q_id]),
#                         questions_tree
#                     )
#                     if next_available_q_id is None:
#                         next_available_q_id = current_q_id + len(questions_tree) + 2
                
#                 question_node["options"].append({
#                     "option_id": option["option_id"],
#                     "opt_value": option_value,
#                     "next_q_id": next_q_id,
#                     "nextQuestion": next_q_id,
#                     "childQuestion": next_q_id,
#                     "childDiagnosis": None
#                 })
        
#         questions_tree.append(question_node)
#         return next_available_q_id

#     def _generate_all_answer_combinations(self, questions_list: List[Dict]) -> List[tuple]:
#         if not questions_list:
#             return []
        
#         option_lists = []
#         for question in questions_list:
#             options = [opt["opt_value"] for opt in question.get("options", [])]
#             if not options:
#                 return []
#             option_lists.append(options)
        
#         if option_lists:
#             return list(itertools.product(*option_lists))
#         return []

#     def generate_separate_category_trees(self, questions_data: List[Dict]) -> Dict[str, Dict]:
#         json_data = self.load_questions_json(questions_data)
#         if not json_data:
#             return {}
        
#         questions_data = json_data.get("questions", [])
#         if not questions_data:
#             return {}
        
#         combined_questions = self.organize_questions_by_demographics(questions_data)
#         category_trees = self.create_separate_category_trees(combined_questions)
        
#         return category_trees

# # Database Operations
# def create_decision_tree_protocol(protocol_id: int = 1, created_by: int = 1) -> int:
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             INSERT INTO decision_tree (protocol_id, is_approved, is_active, created_by, approved_by)
#             VALUES (%s, %s, %s, %s, %s)
#             RETURNING id
#         """, (protocol_id, False, True, created_by, None))
        
#         decision_tree_id = cursor.fetchone()[0]
#         conn.commit()
        
#         return decision_tree_id

# def store_category_tree(decision_tree_id: int, age_category: str, gender: str, dt_path_json: Dict):
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             INSERT INTO decision_tree_categories (decision_tree_id, age_category, gender, dt_path)
#             VALUES (%s, %s, %s, %s)
#             RETURNING id
#         """, (decision_tree_id, age_category, gender, json.dumps(dt_path_json)))
        
#         category_id = cursor.fetchone()[0]
#         conn.commit()
        
#         return category_id

# def get_decision_tree_from_db(decision_tree_id: int, age_category: str, gender: str):
#     with get_db_connection() as conn:
#         cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
#         cursor.execute("""
#             SELECT dtc.*, dt.protocol_id, dt.is_approved, dt.is_active, dt.created_by, dt.approved_by, dt.created_time
#             FROM decision_tree_categories dtc
#             JOIN decision_tree dt ON dtc.decision_tree_id = dt.id
#             WHERE dtc.decision_tree_id = %s AND dtc.age_category = %s AND dtc.gender = %s
#         """, [decision_tree_id, age_category, gender])
        
#         result = cursor.fetchone()
#         if not result:
#             return None
        
#         result_dict = dict(result)
#         if result_dict['dt_path']:
#             try:
#                 result_dict['dt_path'] = json.loads(result_dict['dt_path']) if isinstance(result_dict['dt_path'], str) else result_dict['dt_path']
#             except:
#                 result_dict['dt_path'] = {"error": "Failed to parse JSON"}
        
#         return result_dict

# def stream_decision_tree_from_db(decision_tree_id: int, age_category: str, gender: str, chunk_size: int = 1000) -> Generator[str, None, None]:
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             SELECT dt_path
#             FROM decision_tree_categories 
#             WHERE decision_tree_id = %s AND age_category = %s AND gender = %s
#         """, [decision_tree_id, age_category, gender])
        
#         result = cursor.fetchone()
#         if not result or not result[0]:
#             yield f'{{"error": "No decision tree found for decision_tree_id={decision_tree_id}, age_category={age_category}, gender={gender}"}}'
#             return
        
#         try:
#             tree_data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            
#             if isinstance(tree_data, dict) and 'decision_tree' in tree_data:
#                 decision_tree = tree_data['decision_tree']
#                 total_nodes = tree_data.get('total_nodes', len(decision_tree))
#             elif isinstance(tree_data, list):
#                 decision_tree = tree_data
#                 total_nodes = len(decision_tree)
#             else:
#                 yield '{"error": "Unexpected JSON structure"}'
#                 return
            
#             response = {
#                 "decision_tree_id": decision_tree_id,
#                 "age_category": age_category,
#                 "gender": gender,
#                 "total_nodes": total_nodes,
#                 "decision_tree": decision_tree
#             }
            
#             json_str = json.dumps(response)
#             for i in range(0, len(json_str), chunk_size):
#                 chunk = json_str[i:i + chunk_size]
#                 yield chunk
#                 time.sleep(0.001)
            
#         except Exception as e:
#             yield f'{{"error": "Streaming failed: {str(e)}"}}'

# # API Models
# class QuestionOption(BaseModel):
#     option_id: int
#     opt_value: str

# class Question(BaseModel):
#     q_tag: str
#     question: str
#     options: List[QuestionOption]
#     age_group: Optional[str] = None
#     gender: Optional[str] = None

# class QuestionInput(BaseModel):
#     questions: List[Question]

# # API Endpoints - Only 3 Essential APIs
# @app.post("/generate-decision-tree")
# async def generate_decision_tree(questions_input: QuestionInput):
#     """Generate decision trees for all categories"""
#     try:
#         questions_data = [q.dict() for q in questions_input.questions]
        
#         generator = SeparateCategoryDecisionTreeGenerator()
#         category_trees = generator.generate_separate_category_trees(questions_data)
        
#         if not category_trees:
#             raise HTTPException(status_code=400, detail="Failed to generate category trees")
        
#         decision_tree_id = create_decision_tree_protocol()
        
#         stored_categories = []
#         for category_key, tree_data in category_trees.items():
#             age_category = tree_data["age_category"]
#             gender = tree_data["gender"]
            
#             category_id = store_category_tree(decision_tree_id, age_category, gender, tree_data)
            
#             stored_categories.append({
#                 "category_id": category_id,
#                 "age_category": age_category,
#                 "gender": gender,
#                 "total_nodes": tree_data["total_nodes"]
#             })
        
#         return {
#             "success": True,
#             "decision_tree_id": decision_tree_id,
#             "categories_stored": len(stored_categories),
#             "categories": stored_categories
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/decision-tree/{decision_tree_id}/{age_category}/{gender}")
# async def get_decision_tree(decision_tree_id: int, age_category: str, gender: str):
#     """Get decision tree directly"""
#     try:
#         result = get_decision_tree_from_db(decision_tree_id, age_category, gender)
        
#         if not result:
#             raise HTTPException(status_code=404, detail=f"No decision tree found for decision_tree_id={decision_tree_id}, age_category={age_category}, gender={gender}")
        
#         return result
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/decision-tree/{decision_tree_id}/{age_category}/{gender}/stream")
# async def stream_decision_tree(decision_tree_id: int, age_category: str, gender: str, chunk_size: int = 1000):
#     """Stream decision tree for large data"""
#     try:
#         return StreamingResponse(
#             stream_decision_tree_from_db(decision_tree_id, age_category, gender, chunk_size),
#             media_type="application/json",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive"
#             }
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


#----------------------------------------working above ----------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Generator
import psycopg2
import json
import itertools
import time
from contextlib import contextmanager

app = FastAPI(title="Simple Decision Tree API", version="1.0.0")

# API Models
class QuestionOption(BaseModel):
    option_id: int
    opt_value: str

class Question(BaseModel):
    q_id: int  # ✅ Added this field
    q_tag: str
    question: str
    options: List[QuestionOption]
    age_group: Optional[str] = None
    gender: Optional[str] = None

class QuestionInput(BaseModel):
    questions: List[Question]

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "Muki@1302"
}

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

class SeparateCategoryDecisionTreeGenerator:
    def __init__(self):
        self.age_groups = ["0-2", "3-12", "13-18", "19-40", "41-65", "66+"]
        self.genders = ["Male", "Female"]
        
    def load_questions_json(self, questions_data: List[Dict]) -> Dict:
        return {"questions": questions_data}

    def organize_questions_by_demographics(self, questions_data: List[Dict]) -> Dict:
        questions_by_category = {}
        
        for question in questions_data:
            age_group = question.get("age_group")
            gender = question.get("gender", "Both")
            
            if not age_group:
                continue
                
            if age_group not in questions_by_category:
                questions_by_category[age_group] = {}
                
            if gender not in questions_by_category[age_group]:
                questions_by_category[age_group][gender] = []
                
            questions_by_category[age_group][gender].append(question)
        
        combined_questions = {}
        
        for age_group in self.age_groups:
            if age_group not in questions_by_category:
                continue
                
            combined_questions[age_group] = {}
            
            both_questions = questions_by_category[age_group].get("Both", [])
            male_questions = questions_by_category[age_group].get("Male", [])
            female_questions = questions_by_category[age_group].get("Female", [])
            
            if age_group in ["0-2", "3-12"]:
                if both_questions:
                    combined_questions[age_group]["Male"] = both_questions.copy()
                    combined_questions[age_group]["Female"] = both_questions.copy()
            else:
                if both_questions or male_questions:
                    combined_questions[age_group]["Male"] = both_questions + male_questions
                if both_questions or female_questions:
                    combined_questions[age_group]["Female"] = both_questions + female_questions
        
        return combined_questions

    def create_separate_category_trees(self, combined_questions: Dict) -> Dict[str, List]:
        category_trees = {}
        
        for age_group in self.age_groups:
            if age_group not in combined_questions:
                continue
                
            for gender in self.genders:
                if gender not in combined_questions[age_group]:
                    continue
                    
                category_key = f"{age_group}_{gender}"
                questions_list = combined_questions[age_group][gender]
                
                if not questions_list:
                    continue
                
                decision_tree = self._create_complete_category_tree(age_group, gender, questions_list)
                
                if decision_tree:
                    category_trees[category_key] = decision_tree
        
        return category_trees

    def _create_complete_category_tree(self, target_age: str, target_gender: str, questions_list: List[Dict]) -> List[Dict]:
        decision_tree = []
        current_logical_q_id = 1  # This is for q_id (logical question identifier)
        current_unique_id = 1     # This is for id (unique path identifier)
        
        # Gender Question
        gender_question = {
            "id": current_unique_id,
            "q_id": current_logical_q_id,  # Same for all gender questions
            "q_tag": "GENERAL",
            "question": "What is your gender?",
            "options": []
        }
        
        # Age question will have id = 2
        next_age_id = current_unique_id + 1
        
        for i, gender in enumerate(["Male", "Female"]):
            option = {
                "option_id": i + 1,
                "opt_value": gender,
                "next_q_id": next_age_id if gender == target_gender else None,  # Points to id, not q_id
                "nextQuestion": next_age_id if gender == target_gender else None,
                "childQuestion": next_age_id if gender == target_gender else None,
                "childDiagnosis": None
            }
            gender_question["options"].append(option)
        
        decision_tree.append(gender_question)
        current_logical_q_id += 1  # Next logical question
        current_unique_id += 1     # Next unique id
        
        # Age Question
        age_question = {
            "id": current_unique_id,
            "q_id": current_logical_q_id,  # Same for all age questions
            "q_tag": "GENERAL", 
            "question": "What is your age group?",
            "options": []
        }
        
        # First actual question will have id = 3
        next_first_question_id = current_unique_id + 1
        
        for i, age_group in enumerate(self.age_groups):
            option = {
                "option_id": i + 1,
                "opt_value": age_group,
                "next_q_id": next_first_question_id if age_group == target_age else None,  # Points to id, not q_id
                "nextQuestion": next_first_question_id if age_group == target_age else None,
                "childQuestion": next_first_question_id if age_group == target_age else None,
                "childDiagnosis": None
            }
            age_question["options"].append(option)
        
        decision_tree.append(age_question)
        current_logical_q_id += 1  # Next logical question  
        current_unique_id += 1     # Next unique id
        
        # Actual questions
        if questions_list:
            questions_tree = self._create_questions_tree(questions_list, current_logical_q_id, current_unique_id)
            if questions_tree:
                decision_tree.extend(questions_tree)
        
        return decision_tree

    def _create_questions_tree(self, questions_list: List[Dict], start_logical_q_id: int, start_unique_id: int) -> List[Dict]:
        questions_tree = []
        current_logical_q_id = start_logical_q_id
        current_unique_id = start_unique_id
        
        all_paths = self._generate_all_answer_combinations(questions_list)
        
        if not all_paths:
            return questions_tree
        
        # Use mutable tracker for unique_id
        id_tracker = {"current_unique_id": current_unique_id}
        
        self._create_branching_questions_recursive(
            questions_list, all_paths, 0, current_logical_q_id, id_tracker, questions_tree
        )
        
        return questions_tree

    def _create_branching_questions_recursive(self, questions_list: List[Dict], all_paths: List[tuple], 
                                            question_index: int, current_logical_q_id: int, id_tracker: Dict, 
                                            questions_tree: List[Dict]) -> int:
        
        if question_index >= len(questions_list):
            return None
        
        current_question = questions_list[question_index]
        
        question_node = {
            "id": id_tracker["current_unique_id"],           # Unique for each path (incremental)
            "q_id": current_question.get("q_id"),            # Use original q_id from JSON input
            "q_tag": current_question.get("q_tag", "SYMPTOMS"),
            "question": current_question["question"],
            "options": []
        }
        
        # Increment unique ID for next node
        current_node_id = id_tracker["current_unique_id"]
        id_tracker["current_unique_id"] += 1
        
        paths_by_answer = {}
        for path in all_paths:
            if len(path) > question_index:
                answer = path[question_index]
                if answer not in paths_by_answer:
                    paths_by_answer[answer] = []
                paths_by_answer[answer].append(path)
        
        for option in current_question.get("options", []):
            option_value = option["opt_value"]
            
            if option_value in paths_by_answer:
                filtered_paths = paths_by_answer[option_value]
                
                if question_index == len(questions_list) - 1:
                    # Last question - no next question
                    next_id = None
                else:
                    # Recursively create next question and get its unique id
                    next_id = id_tracker["current_unique_id"]  # This will be the id of next node
                    self._create_branching_questions_recursive(
                        questions_list,
                        filtered_paths,
                        question_index + 1,
                        current_logical_q_id,
                        id_tracker,
                        questions_tree
                    )
                
                question_node["options"].append({
                    "option_id": option["option_id"],
                    "opt_value": option_value,
                    "next_q_id": next_id,      # Points to unique id, not logical q_id
                    "nextQuestion": next_id,
                    "childQuestion": next_id,
                    "childDiagnosis": None
                })
        
        questions_tree.append(question_node)
        return current_node_id

    def _generate_all_answer_combinations(self, questions_list: List[Dict]) -> List[tuple]:
        if not questions_list:
            return []
        
        option_lists = []
        for question in questions_list:
            options = [opt["opt_value"] for opt in question.get("options", [])]
            if not options:
                return []
            option_lists.append(options)
        
        if option_lists:
            return list(itertools.product(*option_lists))
        return []

    def generate_separate_category_trees(self, questions_data: List[Dict]) -> Dict[str, List]:
        json_data = self.load_questions_json(questions_data)
        if not json_data:
            return {}
        
        questions_data = json_data.get("questions", [])
        if not questions_data:
            return {}
        
        combined_questions = self.organize_questions_by_demographics(questions_data)
        category_trees = self.create_separate_category_trees(combined_questions)
        
        return category_trees

# Database Operations
def create_decision_tree_protocol(protocol_id: int = 1, created_by: int = 1) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO decision_tree (protocol_id, is_approved, is_active, created_by, approved_by)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (protocol_id, False, True, created_by, None))
        
        decision_tree_id = cursor.fetchone()[0]
        conn.commit()
        
        return decision_tree_id

def store_category_tree(decision_tree_id: int, age_category: str, gender: str, dt_path_json: List):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO decision_tree_categories (decision_tree_id, age_category, gender, dt_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (decision_tree_id, age_category, gender, json.dumps(dt_path_json)))
        
        category_id = cursor.fetchone()[0]
        conn.commit()
        
        return category_id

def get_decision_tree_from_db(decision_tree_id: int, age_category: str, gender: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dt_path
            FROM decision_tree_categories
            WHERE decision_tree_id = %s AND age_category = %s AND gender = %s
        """, [decision_tree_id, age_category, gender])
        
        result = cursor.fetchone()
        if not result or not result[0]:
            return None
        
        try:
            decision_tree = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return decision_tree
        except:
            return [{"error": "Failed to parse JSON"}]

def stream_decision_tree_from_db(decision_tree_id: int, age_category: str, gender: str, chunk_size: int = 1000) -> Generator[str, None, None]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dt_path
            FROM decision_tree_categories 
            WHERE decision_tree_id = %s AND age_category = %s AND gender = %s
        """, [decision_tree_id, age_category, gender])
        
        result = cursor.fetchone()
        if not result or not result[0]:
            yield f'[{{"error": "No decision tree found for decision_tree_id={decision_tree_id}, age_category={age_category}, gender={gender}"}}]'
            return
        
        try:
            decision_tree = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            
            json_str = json.dumps(decision_tree)
            for i in range(0, len(json_str), chunk_size):
                chunk = json_str[i:i + chunk_size]
                yield chunk
                time.sleep(0.001)
            
        except Exception as e:
            yield f'[{{"error": "Streaming failed: {str(e)}"}}]'


@app.post("/generate-decision-tree")
async def generate_decision_tree(questions_input: QuestionInput):
    """Generate decision trees for all categories"""
    try:
        questions_data = [q.dict() for q in questions_input.questions]
        
        generator = SeparateCategoryDecisionTreeGenerator()
        category_trees = generator.generate_separate_category_trees(questions_data)
        
        if not category_trees:
            raise HTTPException(status_code=400, detail="Failed to generate category trees")
        
        decision_tree_id = create_decision_tree_protocol()
        
        stored_categories = []
        for category_key, decision_tree in category_trees.items():
            age_category, gender = category_key.split('_', 1)
            
            category_id = store_category_tree(decision_tree_id, age_category, gender, decision_tree)
            
            stored_categories.append({
                "category_id": category_id,
                "age_category": age_category,
                "gender": gender,
                "total_nodes": len(decision_tree)
            })
        
        return {
            "success": True,
            "decision_tree_id": decision_tree_id,
            "categories_stored": len(stored_categories),
            "categories": stored_categories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decision-tree/{decision_tree_id}/{age_category}/{gender}")
async def get_decision_tree(decision_tree_id: int, age_category: str, gender: str):
    """Get decision tree directly"""
    try:
        result = get_decision_tree_from_db(decision_tree_id, age_category, gender)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"No decision tree found for decision_tree_id={decision_tree_id}, age_category={age_category}, gender={gender}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decision-tree/{decision_tree_id}/{age_category}/{gender}/stream")
async def stream_decision_tree(decision_tree_id: int, age_category: str, gender: str, chunk_size: int = 1000):
    """Stream decision tree for large data"""
    try:
        return StreamingResponse(
            stream_decision_tree_from_db(decision_tree_id, age_category, gender, chunk_size),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)