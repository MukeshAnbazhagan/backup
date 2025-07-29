from typing import Any
import json
import streamlit as st
import os
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage,HumanMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
import json
from textwrap import dedent
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
import os
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
import streamlit.components.v1 as components
from graphviz import Source
import copy

# Translation functionality
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Medical Decision Tree",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Simple custom styling */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Info boxes styling */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Expander styling */
    streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Language selector styling */
    .language-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Language configuration
SUPPORTED_LANGUAGES = {
    "English": {"code": "eng_Latn", "flag": "🇺🇸"},
    "Hindi": {"code": "hin_Deva", "flag": "🇮🇳"},
    "Tamil": {"code": "tam_Taml", "flag": "🇮🇳"},
    "Telugu": {"code": "tel_Telu", "flag": "🇮🇳"},
    "Malayalam": {"code": "mal_Mlym", "flag": "🇮🇳"},
    "Kannada": {"code": "kan_Knda", "flag": "🇮🇳"}
}

# Translation class
class IndicTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.processors = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, src_lang, tgt_lang):
        """Load translation model for specific language pair"""
        model_key = f"{src_lang}_{tgt_lang}"
        
        if model_key not in self.models:
            if src_lang == "eng_Latn":
                model_name = "ai4bharat/indictrans2-en-indic-1B"
            else:
                model_name = "ai4bharat/indictrans2-indic-en-1B"
            
            with st.spinner(f"Loading translation model..."):
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                ).to(self.device)
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                self.processors[model_key] = IndicProcessor(inference=True)
    
    def translate_text(self, text, src_lang, tgt_lang):
        """Translate text from source to target language"""
        if src_lang == tgt_lang:
            return text
        
        model_key = f"{src_lang}_{tgt_lang}"
        self.load_model(src_lang, tgt_lang)
        
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        processor = self.processors[model_key]
        
        # Preprocess
        batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        
        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Postprocess
        translations = processor.postprocess_batch(generated_tokens, lang=tgt_lang)
        return translations[0]

# Initialize translator
if TRANSLATION_AVAILABLE:
    if 'translator' not in st.session_state:
        st.session_state.translator = IndicTranslator()

def translate_json_content(json_data, target_lang_code):
    """Translate JSON content to target language"""
    if not TRANSLATION_AVAILABLE or target_lang_code == "eng_Latn":
        return json_data
    
    translated_data = copy.deepcopy(json_data)
    translator = st.session_state.translator
    
    # Translate questions
    for question in translated_data["questions"]:
        question["question"] = translator.translate_text(
            question["question"], "eng_Latn", target_lang_code
        )
        for option in question["options"]:
            option["opt_value"] = translator.translate_text(
                option["opt_value"], "eng_Latn", target_lang_code
            )
    
    # Translate diagnoses
    for diagnosis in translated_data["diagnosis"]:
        diagnosis["diagnosis"]["diagnosis_title"] = translator.translate_text(
            diagnosis["diagnosis"]["diagnosis_title"], "eng_Latn", target_lang_code
        )
        diagnosis["diagnosis"]["diagnosis_description"] = translator.translate_text(
            diagnosis["diagnosis"]["diagnosis_description"], "eng_Latn", target_lang_code
        )
        
        # Translate lab tests
        for lab_test in diagnosis["lab_tests"]:
            lab_test["lab_test_title"] = translator.translate_text(
                lab_test["lab_test_title"], "eng_Latn", target_lang_code
            )
        
        # Translate medications
        for med in diagnosis["otc_medication"]:
            med["otc_title"] = translator.translate_text(
                med["otc_title"], "eng_Latn", target_lang_code
            )
            med["medicine_name"] = translator.translate_text(
                med["medicine_name"], "eng_Latn", target_lang_code
            )
        
        # Translate advice
        for advice in diagnosis["advice"]:
            advice["advice_title"] = translator.translate_text(
                advice["advice_title"], "eng_Latn", target_lang_code
            )
            advice["advice_description"] = translator.translate_text(
                advice["advice_description"], "eng_Latn", target_lang_code
            )
        
        # Translate red flags
        for flag in diagnosis["red_flags"]:
            flag["red_flag_description"] = translator.translate_text(
                flag["red_flag_description"], "eng_Latn", target_lang_code
            )
        
        # Translate precautions
        for precaution in diagnosis["precautions"]:
            precaution["precaution_title"] = translator.translate_text(
                precaution["precaution_title"], "eng_Latn", target_lang_code
            )
            precaution["precaution_description"] = translator.translate_text(
                precaution["precaution_description"], "eng_Latn", target_lang_code
            )
    
    return translated_data

# [Previous model classes remain the same]
# --- Reasoning Structure ---
class ReasoningSteps(BaseModel):
    reasoning: Optional[str] = Field(default='',description="Detailed reasoning for this step")
    critique: Optional[str] = Field(default='',description="Validation of reasoning step against the task")
    conclusion: Optional[str] = Field(default='',description="Final conclusion derived from this reasoning step")

# --- Option Structure ---
class Option(BaseModel):
    option_id: int = Field(description="Unique identifier for the option (starting from 1)")
    opt_value: str = Field(description="Text value of the option that the patient can select")

# --- Question Node Structure ---
class QuestionSyntax(BaseModel):
    
    q_id: int = Field(description="Unique question ID")
    q_tag: Optional[str] = Field(default="", description="Tag indicating which blueprint category this question belongs to")
    question: Optional[str] = Field(description="The diagnostic question to ask")
    options: Optional[List[Option]] = Field(description="List of possible options")

# --- Diagnosis Node Structure ---
class DiagnosisNode(BaseModel):
    q_id: int
    q_tag: str = "DIAGNOSIS"
    diagnosis: Dict[str, Union[int, str]] = Field(description="Diagnosis summary including title and description")
    lab_tests: List[str]
    otc_medication: List[str]
    advice: List[str]
    red_flags: List[str]
    precautions: List[str]

# --- Record During Generation ---
class QueryRecord(BaseModel):
    questions: List[QuestionSyntax]
    diagnosis: List[DiagnosisNode]
    reasoning: Optional[List[ReasoningSteps]]= None

# --- Final Output Record ---
class FinalRecord(BaseModel):
    questions: List[QuestionSyntax]
    diagnosis: List[DiagnosisNode]

# --- LangGraph State ---
class QnAState(BaseModel):
    protocol: str
    chat_history: List[Any] = Field(default_factory=list)
    output: Dict[str, Any] = Field(default_factory=dict)

# --- Updated Record structures ---
class UpdateRecord(BaseModel):
    questions: List[QuestionSyntax]

class UpdateDiagnosisRecord(BaseModel):
    diagnosis: List[DiagnosisNode]

# Sample JSON data for lang-pref tab
SAMPLE_LANG_PREF_JSON = {
    "questions": [
        {
            "q_id": 1,
            "question": "What are your main symptoms?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "Dry cough",
                    "next_q_id": 2
                },
                {
                    "option_id": 2,
                    "opt_value": "Productive cough (with mucus)",
                    "next_q_id": 3
                },
                {
                    "option_id": 3,
                    "opt_value": "Wheezing",
                    "next_q_id": 4
                }
            ]
        },
        {
            "q_id": 2,
            "question": "How long have you had the cough?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "Less than 3 weeks",
                    "next_q_id": 5
                },
                {
                    "option_id": 2,
                    "opt_value": "3-8 weeks",
                    "next_q_id": 16
                },
                {
                    "option_id": 3,
                    "opt_value": "More than 8 weeks",
                    "next_q_id": 17
                }
            ]
        },
        {
            "q_id": 3,
            "question": "How would you describe the quality of your cough?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "Barking",
                    "next_q_id": 16
                },
                {
                    "option_id": 2,
                    "opt_value": "Tickling",
                    "next_q_id": 5
                },
                {
                    "option_id": 3,
                    "opt_value": "Deep and chesty",
                    "next_q_id": 17
                }
            ]
        },
        {
            "q_id": 4,
            "question": "On a scale of 1-5, how severe is your cough (1=Mild, 5=Severe)?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "1 - Mild",
                    "next_q_id": 16
                },
                {
                    "option_id": 2,
                    "opt_value": "2 - Moderate",
                    "next_q_id": 5
                },
                {
                    "option_id": 3,
                    "opt_value": "3 - Severe",
                    "next_q_id": 17
                },
                {
                    "option_id": 4,
                    "opt_value": "4 - Very Severe",
                    "next_q_id": 17
                }
            ]
        },
        {
            "q_id": 5,
            "question": "Do you have a fever?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "No fever",
                    "next_q_id": 16
                },
                {
                    "option_id": 2,
                    "opt_value": "Mild fever (below 100.4°F)",
                    "next_q_id": 16
                },
                {
                    "option_id": 3,
                    "opt_value": "High fever (100.4°F or higher)",
                    "next_q_id": 17
                }
            ]
        }
    ],
    "diagnosis": [
        {
            "q_id": 16,
            "q_tag": "DIAGNOSIS",
            "diagnosis": {
                "diagnosis_no": 1,
                "diagnosis_title": "Viral Upper Respiratory Infection",
                "diagnosis_description": "A common viral infection affecting the upper respiratory tract, typically causing mild cough, runny nose, and low-grade fever."
            },
            "lab_tests": [
                {
                    "lab_test_no": 1,
                    "lab_test_title": "Complete Blood Count"
                },
                {
                    "lab_test_no": 2,
                    "lab_test_title": "Rapid viral panel (if severe)"
                }
            ],
            "otc_medication": [
                {
                    "otc_no": 1,
                    "otc_title": "Cough suppressant",
                    "medicine_name": "Dextromethorphan",
                    "dosage_duration": "15-30mg every 4-6 hours",
                    "type": "Oral",
                    "intake_type": "With or without food",
                    "intake_schedules": "As needed for cough"
                }
            ],
            "advice": [
                {
                    "advice_no": 1,
                    "advice_title": "Rest and Hydration",
                    "advice_description": "Get plenty of rest and drink lots of fluids to help your body fight the infection."
                },
                {
                    "advice_no": 2,
                    "advice_title": "Symptom Monitoring",
                    "advice_description": "Monitor symptoms and seek medical attention if they worsen or persist beyond 7-10 days."
                }
            ],
            "red_flags": [
                {
                    "red_flag_no": 1,
                    "red_flag_description": "High fever above 101.3°F (38.5°C)"
                },
                {
                    "red_flag_no": 2,
                    "red_flag_description": "Difficulty breathing or shortness of breath"
                }
            ],
            "precautions": [
                {
                    "precaution_no": 1,
                    "precaution_title": "Isolation",
                    "precaution_description": "Stay home to avoid spreading the infection to others."
                },
                {
                    "precaution_no": 2,
                    "precaution_title": "Hand Hygiene",
                    "precaution_description": "Wash hands frequently and cover coughs and sneezes."
                }
            ]
        },
        {
            "q_id": 17,
            "q_tag": "DIAGNOSIS",
            "diagnosis": {
                "diagnosis_no": 2,
                "diagnosis_title": "Asthma Exacerbation",
                "diagnosis_description": "A worsening of asthma symptoms including cough, wheezing, shortness of breath, and chest tightness due to airway inflammation and constriction."
            },
            "lab_tests": [
                {
                    "lab_test_no": 1,
                    "lab_test_title": "Peak flow measurement"
                },
                {
                    "lab_test_no": 2,
                    "lab_test_title": "Chest X-ray (if severe)"
                }
            ],
            "otc_medication": [
                {
                    "otc_no": 1,
                    "otc_title": "Bronchodilator",
                    "medicine_name": "Albuterol inhaler",
                    "dosage_duration": "2 puffs every 4-6 hours",
                    "type": "Inhaled",
                    "intake_type": "Via metered-dose inhaler",
                    "intake_schedules": "As needed for symptoms"
                }
            ],
            "advice": [
                {
                    "advice_no": 1,
                    "advice_title": "Trigger Avoidance",
                    "advice_description": "Identify and avoid known asthma triggers such as allergens, smoke, or cold air."
                },
                {
                    "advice_no": 2,
                    "advice_title": "Medication Adherence",
                    "advice_description": "Take controller medications as prescribed even when feeling well."
                }
            ],
            "red_flags": [
                {
                    "red_flag_no": 1,
                    "red_flag_description": "Severe difficulty breathing or speaking in short phrases"
                },
                {
                    "red_flag_no": 2,
                    "red_flag_description": "Blue lips or fingernails (cyanosis)"
                }
            ],
            "precautions": [
                {
                    "precaution_no": 1,
                    "precaution_title": "Emergency Plan",
                    "precaution_description": "Have an asthma action plan and know when to seek emergency care."
                },
                {
                    "precaution_no": 2,
                    "precaution_title": "Regular Monitoring",
                    "precaution_description": "Monitor peak flow readings and track symptom patterns."
                }
            ]
        }
    ]
}

# [All function definitions remain the same - keeping them exactly as before]
def run_intermediate_chain(state: QnAState):
    # Define base parser and OutputFixingParser with LLM for auto-correction
    base_parser = PydanticOutputParser(pydantic_object=QueryRecord)
    parser_int = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=ChatOllama(model="gemma3:27b", base_url="http://localhost:11434",temperature=0,max_retries=2)
    )
    # format_instructions_int = base_parser.get_format_instructions()

    # Prompt template for intermediate reasoning and generation
    intermediate_prompt_template = """You are generating medical Q&A JSON.
        STEP 1: Understand the structure
          - Three main sections: questions, diagnosis, reasoning
          - All are ARRAYS of objects

        STEP 2: Question format
        {{
          "q_id": integer,
          "q_tag": one of [DEMOGRAPHICS, RED_FLAGS, LOCATION, CHARACTER, ASSOCIATED_SYMPTOMS, RISK_FACTORS],
          "question": "Yes/no questions? or according to medical interview",
          "options": [{{"option_id": 1, "opt_value": "yes"}}, {{"option_id": 2, "opt_value": "no"}}]
        }}

        STEP 3: Diagnosis format  
        {{
          "q_id": 101,
          "q_tag": "DIAGNOSIS",
          "diagnosis": {{"title": "Name", "description": "Details"}},
          "lab_tests": array,
          "otc_medication": array,
          "advice": array,
          "red_flags": array,
          "precautions": array
        }}

      STEP 4: Generate for protocol: {protocol}
      Requirements:
      - 15-25 questions total 
      - START with demographics first, then progress through symptoms
      - 8-12 comprehensive diagnoses
      - Cover spectrum from mild to severe conditions
      - Skip any questions already in: {chat_history}

        OUTPUT ONLY THIS JSON:
        {{
          "questions": [...],
          "diagnosis": [...],  
          "reasoning": [...]
        }}
"""

    # Create prompt
    intermediate_prompt = PromptTemplate(
        template=intermediate_prompt_template,
        input_variables=["chat_history", "protocol"]
    )

    # LLM (used inside the chain and also in parser)
    llm = ChatOllama(model="gemma3:27b", base_url="http://localhost:11434", stream=True, temperature=0, top_p=0.95)

    # Final intermediate chain
    intermediate_chain = intermediate_prompt | llm | parser_int

    # Prepare input for the chain
    input_data = {
        "chat_history": state.chat_history,
        "protocol": state.protocol
    }

    # Invoke the chain
    result_int = intermediate_chain.invoke(input_data)

    # Update chat history
    chat_history_int = state.chat_history.copy()
    for question in result_int.questions:
        chat_history_int.append({
        "question": AIMessage(content=question.question),
        "Options": HumanMessage(content="\n".join([f"{opt.option_id}: {opt.opt_value}" for opt in question.options]))
    })

    # Final record and state update
    final_record_int = FinalRecord(**result_int.model_dump())
    print(f"intermediate:{final_record_int}")
    # print(f'intermediate_response:{result_int}')

    state.output["qna_set"] = final_record_int
    state.chat_history = chat_history_int

    return state

def modify_questions_bulk_with_llm(qna_set_questions, user_query):
    from langchain.prompts import PromptTemplate
    

    prompt = PromptTemplate(
        template=""" 
        You are a QnA set editor.

        ## Input:
        - `qna_set`: {qna_set}
        - `user_query`: {user_query}

        ## Objective:
        You will receive:
        - A QnA set (`qna_set`), which is a list of question objects with attributes like `q_id`, `question`, `options`, etc.
        - A user request (`user_query`), which specifies a targeted change to the QnA set (e.g., updating a question, modifying an ID, changing options).
        - If the user query is to update a new question, generate a new question and update that `q_id`.
        - If the user query is to replace a question with another, replace it accordingly.

        ## Task:
        - Carefully parse the `user_query`.
        - Apply **only** the change(s) explicitly requested in the `user_query`.
        - Do **not** make any other changes to the QnA set.
        - Return the modified QnA set as a JSON object in this format:

        ```json
        {{
        "questions": [
            {{
            "q_id": 1,
            "q_tag": "...",
            "question": "...",
            "options": [
                {{"option_id": 1, "opt_value": "..."}},
                ...
            ]
            }},
            ...
        ]
        }}        
        """,
                input_variables=['qna_set', 'user_query']
            )

    llm = ChatOllama(model="mistral-small3.1:latest", base_url="http://localhost:11434", stream=True, temperature=0)
    parser=PydanticOutputParser(pydantic_object=UpdateRecord)

    chain = prompt | llm | parser

    response = chain.invoke({
        'user_query': user_query,
        'qna_set': qna_set_questions
        })

    return response

def modify_diagnosis_bulk_with_llm(qna_set_diagnosis, user_query):
    from langchain.prompts import PromptTemplate
    

    prompt = PromptTemplate(
        template=""" 
        You are a medical diagnosis editor.

        ## Input:
        - `diagnosis_set`: {diagnosis_set}
        - `user_query`: {user_query}

        ## Objective:
        You will receive:
        - A diagnosis set (`diagnosis_set`), which is a list of diagnosis objects with attributes like `q_id`, `diagnosis`, `Description', `lab_tests`, `otc_medication`, `advice`, `red_flags`, and `precautions`.
        - A user request (`user_query`), which specifies a targeted change to the diagnosis set (e.g., updating a diagnosis title, modify Descriptions, changing precautions, modifying otc_medication, modifying lab_tests

        ## Task:
        - Carefully parse the `user_query`.
        - Apply **only** the change(s) explicitly requested in the `user_query`.
        - Change the `Description', `lab_tests`, `otc_medication`, `advice`, `red_flags`, and `precautions` according to diagnosis title changes.
        - Do **not** make any other changes to the diagnosis set.
        - Return the modified diagnosis set as a JSON object in this format:

        ```json
        {{
        "diagnosis": [
            {{
            "q_id": 101,
            "q_tag": "DIAGNOSIS",
            "diagnosis": {{"title": "Disease Name", "description": "Disease description"}},
            "lab_tests": ["Test 1", "Test 2"],
            "otc_medication": ["Medication 1", "Medication 2"],
            "advice": ["Advice 1", "Advice 2"],
            "red_flags": ["Red flag 1", "Red flag 2"],
            "precautions": ["Precaution 1", "Precaution 2"]
            }},
            ...
        ]
        }}
        """,
        input_variables=['diagnosis_set', 'user_query']
    )

    llm = ChatOllama(model="mistral-small3.1:latest", base_url="http://localhost:11434", stream=True, temperature=0)
    parser = PydanticOutputParser(pydantic_object=UpdateDiagnosisRecord)

    chain = prompt | llm | parser

    response = chain.invoke({
        'user_query': user_query,
        'diagnosis_set': qna_set_diagnosis
    })

    return response


# --- Build the LangGraph State Graph ---
builder = StateGraph(QnAState)
builder.add_node("run_intermediate_chain",run_intermediate_chain)
builder.add_edge(START, "run_intermediate_chain")
builder.add_edge("run_intermediate_chain", END)
memory=MemorySaver()
config = {"configurable": {"thread_id": "1"}}
graph = builder.compile(checkpointer=memory)


def generate_medical_dag(protocol, question_set, diagnosis_list):
    """
    Generate a clinically valid Directed Acyclic Graph (DAG) for medical decision-making.
    
    Args:
        protocol (str): The medical condition being addressed (e.g., "cough", "headache")
        question_set (list): A pre-defined set of clinical questions with their options
        diagnosis_list (list): A pre-defined set of possible diagnoses
        
    Returns:
        dict: The response containing the DAG structure
    """
    from langchain.prompts import PromptTemplate
    
    
    # Create the prompt template for DAG generation
    # Create the prompt template for DAG generation
    prompt = PromptTemplate(template="""
    ## Task
    Create a clinically valid Directed Acyclic Graph (DAG) for medical decision-making by organizing ONLY the questions and diagnoses I provide.
    
    ## Medical Condition
    {protocol}
    
    ## Available Questions (Use ONLY these exact questions)
    {question_set}
    
    ## Available Diagnoses (Use ONLY these exact diagnoses)
    {diagnosis_list}
    
    ## Requirements
    1. Your DAG MUST use ONLY the provided questions and diagnoses - do not invent new ones
    2. Each node should be labeled with the exact question text or diagnosis name
    3. Organize questions in a medically sound sequence
    4. Each path must end with a diagnosis from the provided list
    5. Create a valid DAG with no loops
    6. Maximum 8 questions per path

    ### Clinical Design Principles

    1).Start Broad, Narrow Gradually: Begin with general questions before specific ones.
    2).Front-Load Red Flags: Ask about critical symptoms early.
    3).Respect Clinical Precedence: Follow standard medical interview structure (HPI → ROS → etc.).
    4).Minimize Question Count: Each question should meaningfully narrow the differential.
    5).Logical Grouping: Related questions should flow together.
    6).Respect Bayesian Reasoning: Prior probabilities should influence question sequencing.
    7).Actionable Endpoints: Each diagnosis should lead to clear next steps.

    ### Validation Checklist

    - Are all paths clinically valid?
    - Do red flag symptoms trigger appropriate urgent pathways?
    - Are age-specific considerations incorporated?
    - Do all paths lead to appropriate diagnoses?
    - Are the diagnostic confidence levels appropriate?
    - Are follow-up recommendations appropriate for each diagnosis?
    - Would this DAG pass peer review by specialists in this field?
    - Have you optimized for both sensitivity and specificity?
    - Have you used ONLY questions from the provided {question_set}?
    - Have you used ONLY diagnoses from the provided {diagnosis_list}?
    - Do all questions match their exact text from {question_set}?

    ### Final Instructions

    - You may NOT create new questions - use only those in {question_set}
    - You may NOT create new diagnoses - use only those in {diagnosis_list}
    - You must maintain exact text matches with the provided components
    - Return ONLY the mermaid diagram code without any other text, explanations, or thinking tags
    - The diagram should start with mermaid and end with  - nothing else
    
    ## Output Format
    Return ONLY the mermaid flowchart code with no additional commentary:
    
    ```mermaid
    graph TD
    [Your graph here]
    ```""",
    input_variables=['protocol', 'question_set', 'diagnosis_list'])
    
    # Format question_set and diagnosis_list for better prompt understanding
    formatted_questions = json.dumps(question_set, indent=2)
    formatted_diagnoses = json.dumps(diagnosis_list, indent=2)
                        
    
    # Initialize the LLM
    llm = ChatOllama(
        model="qwen3:32b",
        base_url="http://localhost:11434",
        stream=True,
        temperature=0
    )
    
    # Create and invoke the chain
    chain = prompt | llm
    response = chain.invoke({
        'protocol': protocol,
        'question_set': formatted_questions,
        'diagnosis_list': formatted_diagnoses
    })
    
    return response

def extract_mermaid_code(input_text):
    """
    Extract only the Mermaid code from the input text.
    
    Args:
        input_text (str): The input text containing Mermaid code blocks.
        
    Returns:
        str: Extracted Mermaid code.
    """
    # Regular expression to match Mermaid code blocks
    mermaid_regex = r'```mermaid\s*([\s\S]*?)```'
    
    # Find all Mermaid code blocks
    matches = re.findall(mermaid_regex, input_text)
    
    if not matches:
        return "No Mermaid code found in the input."
    
    # Extract only the Mermaid code without the backticks
    extracted_code = "\n\n".join(match.strip() for match in matches)
    
    return extracted_code

def convert_mermaid_to_graphviz(mermaid_code, protocol="Medical Condition"):
    """
    Convert Mermaid diagram to Graphviz DOT language using LLM
    
    Args:
        mermaid_code (str): The mermaid diagram code
        protocol (str): The medical protocol being addressed
        
    Returns:
        str: Graphviz DOT code
    """
    # Prompt for converting mermaid to graphviz
    mermaid_prompt = PromptTemplate(
        template="""You are a medical knowledge graph engineer.
## Task
You will be given a **Mermaid diagram** representing a clinical decision-making flow (typically a Directed Acyclic Graph - DAG) for the condition: {protocol}. Your task is to:
### 1. Validate the Mermaid Diagram
- Check for Mermaid **syntax errors**.
- Check for **clinical structure errors**, including:
  - Redundant or duplicated nodes
  - Cycles (should be acyclic)
  - Terminal diagnosis nodes that have further branches (they shouldn't)
  - Missing clinical reasoning steps
  - Delayed red flag assessment
  - Missing or unclear yes/no logic
### 2. Modify the Mermaid Code (if necessary)
- Ensure the structure is **clinically valid** and **logically sound** for {protocol}.
- Add missing edge labels ("Yes", "No") where appropriate.
- Maintain a **max depth of 8** questions.
- Terminal nodes must represent **diagnoses only** and should not branch.
### 3. Convert to Graphviz DOT Language
Generate a **Graphviz DOT** version of the corrected DAG with:
- `rankdir=HR` for top-to-bottom flow.
- Styled nodes:
  - **Decision Nodes**: `shape=box`, `fillcolor="#E3F2FD"`
  - **Yes/No Decision Points**: `shape=diamond`, `fillcolor="#FFF9C4"`
  - **Diagnosis Nodes**: `shape=ellipse`, `fillcolor="#FFCDD2"`
- Edges labeled with "Yes"/"No" when branching.
- Each node must have a **clear and concise label**.
---
## OUTPUT FORMAT
### Step 1: Mermaid Validation Report
- Is the Mermaid diagram valid? (Yes/No)
- If not, list all detected structural or clinical issues.
### Step 2: Corrected Mermaid Diagram
```mermaid
<INSERT_CORRECTED_MERMAID_HERE>
```
### Step 3: Graphviz DOT Code (Styled)
```dot
digraph Clinical_DAG {{
  rankdir=HR;
  node [style=filled];
  // Example node types:
  A [label="Do you have a fever?", shape=box, fillcolor="#E3F2FD"];
  B [label="Is the fever high?", shape=diamond, fillcolor="#FFF9C4"];
  C [label="Otitis Media", shape=ellipse, fillcolor="#FFCDD2"];
  // Example edges:
  A -> B;
  B -> C [label="Yes"];
}}
```
## Input Mermaid Code:
{mermaid}""",
        input_variables=['mermaid', 'protocol']
    )
    
    # LLM for conversion
    llm_mer = ChatOllama(
        model="mistral-small3.1:latest",
        base_url="http://localhost:11434",
        stream=True,
        temperature=0
    )
    
    # Create and invoke chain
    mermaid_chain = mermaid_prompt | llm_mer
    graphviz_res = mermaid_chain.invoke({
        'mermaid': mermaid_code,
        'protocol': protocol
    })
    
    # Extract DOT code
    def extract_dot_code(response):
        match = re.search(r"```dot\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    dot_res = extract_dot_code(graphviz_res.content)
    return dot_res

# --- Enhanced Streamlit App ---
st.title("🏥 Medical Decision Tree")

# Streamlit Sidebar - Only Doctor Validation tab now
with st.sidebar:
    st.header("🩺 Doctor Validation")
    st.divider()
    
    # Create a styled container for the quick guide
    with st.container():
        st.markdown("### Quick Guide")
        st.markdown("""
        - ✅ Review generated questions
        - ✏️ Modify questions as needed
        - 🔬 Update diagnoses
        - 🌳 Generate decision tree
        - 🌍 Multilingual support
        """)
        
    st.divider()
    
    # Add some helpful stats
    if 'qna_set' in st.session_state:
        st.metric("Total Questions", len(st.session_state.qna_set.questions))
        st.metric("Total Diagnoses", len(st.session_state.qna_set.diagnosis))

# Doctor Validation Tab
st.header("Doctor Validation - Generated Questions & Diagnosis")

# Run graph only if not done before
if 'qna_set_resp' not in st.session_state:
    with st.spinner("🔄 Generating medical QnA set..."):
        protocol = "Fever"
        initial_state = QnAState(chat_history=[], protocol=protocol)
        qna_set_resp = graph.invoke(initial_state, config)
        st.session_state.qna_set_resp = qna_set_resp

qna_set_resp = st.session_state.qna_set_resp

# Initialize if not already
if "qna_set" not in st.session_state:
    st.session_state.qna_set = qna_set_resp["output"]["qna_set"]

qna_set = st.session_state.qna_set  # Always use latest state

# Create tabs for different functions - Added lang-pref tab
qa_tab1, qa_tab2, qa_tab3, qa_tab4, qa_tab5 = st.tabs(["📋 View Data", "✏️ Modify Questions", "🔬 Modify Diagnosis", "🌳 Decision Tree", "🌍 Lang-Pref"])

with qa_tab1:
    # Display Questions
    st.subheader("Questions and Options")
    
    for question in st.session_state.qna_set.questions:
        # Create a container for each question
        with st.container():
            col1, col2 = st.columns([1, 9])
            
            with col1:
                # Display question ID badge
                st.markdown(f"""
                <div style="background-color: #3498db; color: white; 
                padding: 0.5rem; border-radius: 50%; text-align: center; 
                font-weight: bold; width: 50px; height: 50px; 
                display: flex; align-items: center; justify-content: center;">
                Q{question.q_id}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Display question text
                st.markdown(f"**{question.question}**")
                
                if len(question.options) > 0:
                # Display options in columns for better layout
                 option_cols = st.columns(len(question.options))
                

                for i, opt in enumerate(question.options):
                    with option_cols[i]:
                        st.info(f"**{opt.option_id}:** {opt.opt_value}")
            
            st.markdown("---")  # Add separator between questions

    # Display Diagnoses
    st.divider()
    st.subheader("Diagnoses")
    
    for idx, diagnosis in enumerate(st.session_state.qna_set.diagnosis, 1):
        # Create an expander for each diagnosis
        with st.expander(f"🏥 {diagnosis.diagnosis['title']}", expanded=False):
            # Create two columns for layout
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Display diagnosis badge
                st.markdown(f"""
                <div style="background-color: #e74c3c; color: white; 
                padding: 1rem; border-radius: 50%; text-align: center; 
                font-weight: bold; width: 60px; height: 60px; 
                display: flex; align-items: center; justify-content: center;
                margin: auto;">
                D{idx}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write(f"**Description:** {diagnosis.diagnosis['description']}")
            
            st.divider()
            
            # Display details in a more organized way
            if diagnosis.red_flags:
                st.error("🚨 **Red Flags**")
                for flag in diagnosis.red_flags:
                    st.markdown(f"- {flag}")
                st.markdown("")
            
            if diagnosis.advice:
                st.success("📋 **Advice**")
                for advice in diagnosis.advice:
                    st.markdown(f"- {advice}")
                st.markdown("")
            
            if diagnosis.precautions:
                st.warning("⚠️ **Precautions**")
                for precaution in diagnosis.precautions:
                    st.markdown(f"- {precaution}")
                st.markdown("")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if diagnosis.otc_medication:
                    st.info("💊 **OTC Medications**")
                    for med in diagnosis.otc_medication:
                        st.markdown(f"- {med}")
            
            with col2:
                if diagnosis.lab_tests:
                    st.info("🧪 **Lab Tests**")
                    for test in diagnosis.lab_tests:
                        st.markdown(f"- {test}")

with qa_tab2:
    # Question Modification UI
    st.subheader("Modify Questions via Instruction")
    q_instruction = st.text_area("Enter your instruction for modifying questions", height=100)

    if st.button("Apply Question Modification", type="primary"):
        if q_instruction:
            with st.spinner("Modifying questions..."):
                new_qna_set = modify_questions_bulk_with_llm(
                    st.session_state.qna_set.questions,
                    q_instruction
                )
                
                updated_questions = []
                for q in new_qna_set.questions:
                    # Create a new QuestionSyntax object from each returned question
                    updated_question = QuestionSyntax(
                        q_id=q.q_id,
                        q_tag=q.q_tag,
                        question=q.question,
                        options=[Option(option_id=opt.option_id, opt_value=opt.opt_value) for opt in q.options]
                    )
                    updated_questions.append(updated_question)

                # Update state
                st.session_state.qna_set.questions = updated_questions

                st.success("✅ Questions modified successfully!")
                st.rerun()
        else:
            st.error("❌ Please provide a valid instruction.")

with qa_tab3:
    # Diagnosis Modification UI
    st.subheader("Modify Diagnosis via Instruction")
    d_instruction = st.text_area("Enter your instruction for modifying diagnoses", height=100)

    if st.button("Apply Diagnosis Modification", type="primary"):
        if d_instruction:
            with st.spinner("Modifying diagnoses..."):
                updated_diagnosis = modify_diagnosis_bulk_with_llm(
                    st.session_state.qna_set.diagnosis,
                    d_instruction
                )
                
                diagnosis_objects = []
                for d in updated_diagnosis.diagnosis:
                    # Create a new DiagnosisNode object from each returned diagnosis
                    diagnosis_node = DiagnosisNode(
                        q_id=d.q_id,
                        q_tag=d.q_tag,
                        diagnosis=d.diagnosis,
                        lab_tests=d.lab_tests,
                        otc_medication=d.otc_medication,
                        advice=d.advice,
                        red_flags=d.red_flags,
                        precautions=d.precautions
                    )
                    diagnosis_objects.append(diagnosis_node)

                # Update state
                st.session_state.qna_set.diagnosis = diagnosis_objects

                st.success("✅ Diagnoses modified successfully!")
                st.rerun()
        else:
            st.error("❌ Please provide a valid instruction.")

with qa_tab4:
    # Save button and DAG generation
    st.header("Generate Decision Tree")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌳 Generate Decision Tree", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating medical decision tree..."):
                # Convert qna_set to the format expected by generate_medical_dag
                questions_json = []
                for q in st.session_state.qna_set.questions:
                    question_dict = {
                        "q_id": q.q_id,
                        "q_tag": q.q_tag,
                        "question": q.question,
                        "options": [{"option_id": opt.option_id, "opt_value": opt.opt_value} for opt in q.options]
                    }
                    questions_json.append(question_dict)
                
                diagnoses_json = []
                for d in st.session_state.qna_set.diagnosis:
                    diagnosis_dict = {
                        "q_id": d.q_id,
                        "q_tag": d.q_tag,
                        "diagnosis": d.diagnosis,
                        "lab_tests": d.lab_tests,
                        "otc_medication": d.otc_medication,
                        "advice": d.advice,
                        "red_flags": d.red_flags,
                        "precautions": d.precautions
                    }
                    diagnoses_json.append(diagnosis_dict)
                
                # Store protocol for consistent use
                current_protocol = st.session_state.qna_set_resp["protocol"]
                
                # Generate the DAG
                dag_response = generate_medical_dag(
                    protocol=current_protocol,
                    question_set=questions_json, 
                    diagnosis_list=diagnoses_json
                )
                
                # Extract the mermaid diagram
                mermaid_code = extract_mermaid_code(dag_response.content)
                
                # Convert mermaid to graphviz - passing the protocol
                dot_code = convert_mermaid_to_graphviz(mermaid_code, current_protocol)
                
                # Store in session state
                st.session_state.mermaid_code = mermaid_code
                st.session_state.dot_code = dot_code
                st.session_state.protocol = current_protocol
                
                # Display the diagram
                st.success(f"✅ Decision tree for {current_protocol} generated successfully!")
            
            # Display the Graphviz diagram
            st.markdown(f'<h3 style="color: #2c3e50; margin-top: 2rem;">🗺️ {st.session_state.protocol} Decision Tree Diagram</h3>', unsafe_allow_html=True)
            
            # Render the Graphviz graph
            if 'dot_code' in st.session_state and st.session_state.dot_code:
                try:
                    graphviz_source = Source(st.session_state.dot_code)
                    st.graphviz_chart(st.session_state.dot_code)
                except Exception as e:
                    st.error(f"Error rendering Graphviz diagram: {str(e)}")
                    st.text("Falling back to mermaid diagram...")
                    
                    # Fallback to mermaid if graphviz fails
                    html_code = f"""
                    <div class="mermaid">
                    graph TD
                    {st.session_state.mermaid_code}
                    </div>

                    <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true }});
                    </script>
                    """
                    
                    components.html(html_code, height=600, scrolling=True)
            else:
                st.error("No Graphviz DOT code was generated. Displaying mermaid diagram instead.")
                
                # Fallback to mermaid
                html_code = f"""
                <div class="mermaid">
                graph TD
                {st.session_state.mermaid_code}
                </div>

                <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
                </script>
                """
                
                components.html(html_code, height=600, scrolling=True)
            
            # Show raw codes in collapsible sections
            with st.expander("Show Raw Mermaid Code"):
                st.code(st.session_state.mermaid_code, language="mermaid")
                
            with st.expander("Show Raw Graphviz DOT Code"):
                st.code(st.session_state.dot_code, language="dot")

# NEW TAB: Lang-Pref with Translation
with qa_tab5:
    st.header("🌍 Multilingual Sample JSON")
    
    # Language selector
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Select Language:")
    
    with col2:
        # Language selection dropdown
        language_options = [f"{info['flag']} {lang}" for lang, info in SUPPORTED_LANGUAGES.items()]
        selected_language = st.selectbox(
            "Choose language:",
            language_options,
            index=0,
            label_visibility="collapsed"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Extract selected language code
    selected_lang_name = selected_language.split(" ")[1]
    selected_lang_code = SUPPORTED_LANGUAGES[selected_lang_name]["code"]
    
    # Initialize translated JSON in session state
    if 'translated_json' not in st.session_state:
        st.session_state.translated_json = {}
    
    # Check if we need to translate
    if selected_lang_code != "eng_Latn":
        if selected_lang_code not in st.session_state.translated_json:
            if TRANSLATION_AVAILABLE:
                with st.spinner(f"Translating to {selected_lang_name}..."):
                    translated_json = translate_json_content(SAMPLE_LANG_PREF_JSON, selected_lang_code)
                    st.session_state.translated_json[selected_lang_code] = translated_json
            else:
                st.error("Translation models not available. Please install required packages.")
                translated_json = SAMPLE_LANG_PREF_JSON
        else:
            translated_json = st.session_state.translated_json[selected_lang_code]
    else:
        translated_json = SAMPLE_LANG_PREF_JSON
    
    # Display sample statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample Questions", len(translated_json["questions"]))
    with col2:
        st.metric("Sample Diagnoses", len(translated_json["diagnosis"]))
    with col3:
        st.metric("Current Language", selected_lang_name)
    
    st.divider()
    
    # Create sub-tabs for better organization
    lang_tab1, lang_tab2, lang_tab3 = st.tabs(["📋 Questions", "🏥 Diagnoses", "📄 Full JSON"])
    
    with lang_tab1:
        st.subheader(f"Sample Questions in {selected_lang_name}")
        
        for question in translated_json["questions"]:
            # Create a container for each question
            with st.container():
                col1, col2 = st.columns([1, 9])
                
                with col1:
                    # Display question ID badge
                    st.markdown(f"""
                    <div style="background-color: #9b59b6; color: white; 
                    padding: 0.5rem; border-radius: 50%; text-align: center; 
                    font-weight: bold; width: 50px; height: 50px; 
                    display: flex; align-items: center; justify-content: center;">
                    Q{question["q_id"]}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Display question text
                    st.markdown(f"**{question['question']}**")
                    
                    # Display options with next_q_id
                    option_cols = st.columns(len(question["options"]))
                    for i, opt in enumerate(question["options"]):
                        with option_cols[i]:
                            st.info(f"**{opt['option_id']}:** {opt['opt_value']}\n\n➡️ Next: Q{opt['next_q_id']}")
                
                st.markdown("---")
    
    with lang_tab2:
        st.subheader(f"Sample Diagnoses in {selected_lang_name}")
        
        for diagnosis in translated_json["diagnosis"]:
            # Create an expander for each diagnosis
            with st.expander(f"🏥 {diagnosis['diagnosis']['diagnosis_title']}", expanded=False):
                # Create two columns for layout
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Display diagnosis badge
                    st.markdown(f"""
                    <div style="background-color: #e67e22; color: white; 
                    padding: 1rem; border-radius: 50%; text-align: center; 
                    font-weight: bold; width: 60px; height: 60px; 
                    display: flex; align-items: center; justify-content: center;
                    margin: auto;">
                    D{diagnosis['diagnosis']['diagnosis_no']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"**Q_ID:** {diagnosis['q_id']}")
                    st.write(f"**Description:** {diagnosis['diagnosis']['diagnosis_description']}")
                
                st.divider()
                
                # Display red flags
                if diagnosis.get("red_flags"):
                    st.error("🚨 **Red Flags**")
                    for flag in diagnosis["red_flags"]:
                        st.markdown(f"- {flag['red_flag_description']}")
                    st.markdown("")
                
                # Display advice
                if diagnosis.get("advice"):
                    st.success("📋 **Advice**")
                    for advice in diagnosis["advice"]:
                        st.markdown(f"- **{advice['advice_title']}:** {advice['advice_description']}")
                    st.markdown("")
                
                # Display precautions
                if diagnosis.get("precautions"):
                    st.warning("⚠️ **Precautions**")
                    for precaution in diagnosis["precautions"]:
                        st.markdown(f"- **{precaution['precaution_title']}:** {precaution['precaution_description']}")
                    st.markdown("")
                
                # Display medications and lab tests in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if diagnosis.get("otc_medication"):
                        st.info("💊 **OTC Medications**")
                        for med in diagnosis["otc_medication"]:
                            st.markdown(f"- **{med['medicine_name']}:** {med['dosage_duration']}")
                
                with col2:
                    if diagnosis.get("lab_tests"):
                        st.info("🧪 **Lab Tests**")
                        for test in diagnosis["lab_tests"]:
                            st.markdown(f"- {test['lab_test_title']}")
    
    with lang_tab3:
        st.subheader(f"Complete JSON in {selected_lang_name}")
        
        # Display JSON with syntax highlighting
        st.json(translated_json)
        
        # Download button for the JSON
        json_string = json.dumps(translated_json, indent=2, ensure_ascii=False)
        st.download_button(
            label=f"📥 Download {selected_lang_name} JSON",
            data=json_string,
            file_name=f"sample_medical_{selected_lang_name.lower()}.json",
            mime="application/json",
            type="secondary"
        )
        
        # Translation status
        if TRANSLATION_AVAILABLE:
            st.success("✅ Real-time translation enabled")
        else:
            st.warning("⚠️ Translation models not available. Install required packages for multilingual support.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem;">
        🏥 Medical Decision Tree Generator with Multilingual Support - Built with Streamlit & LangGraph
    </div>
    """, 
    unsafe_allow_html=True
)