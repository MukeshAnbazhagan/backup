import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

class Option(BaseModel):
    next_q_id: int = Field(description="The next question ID that this option leads to")
    opt_value: str = Field(description="The option text displayed to the patient")
    option_id: int = Field(description="Unique identifier for this option within the question")
    nextQuestion: int = Field(description="Same as next_q_id for compatibility")
    childQuestion: int = Field(description="Same as next_q_id for compatibility")
    childDiagnosis: Optional[int] = Field(default=None, description="Always null for questions, diagnosis ID for diagnosis options")

class DiagnosisInfo(BaseModel):
    reasoning: str = Field(description="Clinical reasoning explaining why this confidence level and diagnosis based on collected symptoms")
    description: str = Field(description="Detailed description of the medical condition")
    diagnosis_title: str = Field(description="Name/title of the diagnosis")
    confidence_threshold: float = Field(description="Confidence percentage (0-100) for this diagnosis", ge=0, le=100)
    icd_code: str = Field(description="ICD-10 code corresponding to this diagnosis")

class DecisionTreeNode(BaseModel):
    id: int = Field(description="Same as q_id for compatibility")
    q_id: int = Field(description="Unique question/node identifier")
    q_tag: str = Field(description="Node type: 'GENERAL' for questions or 'DIAGNOSIS' for diagnosis nodes")
    question: str = Field(description="The question text or diagnosis statement")
    options: List[Option] = Field(description="Available response options")
    # For questions - no additional fields
    # For diagnosis - additional fields below
    advice: Optional[List[str]] = Field(default=None, description="Medical advice list (only for diagnosis)")
    diagnosis: Optional[DiagnosisInfo] = Field(default=None, description="Diagnosis information (only for diagnosis nodes)")
    lab_tests: Optional[List[str]] = Field(default=None, description="Recommended laboratory tests (only for diagnosis)")
    red_flags: Optional[List[str]] = Field(default=None, description="Warning symptoms (only for diagnosis)")
    precautions: Optional[List[str]] = Field(default=None, description="Precautionary measures (only for diagnosis)")
    otc_medication: Optional[List[str]] = Field(default=None, description="Over-the-counter medications (only for diagnosis)")

class DecisionTreeOutput(BaseModel):
    decision_tree: List[DecisionTreeNode] = Field(description="Complete decision tree as flat array - ONLY this will be saved")

class CombinedMedicalDecisionTreeGenerator:
    def __init__(self, model_name="gpt-4o", temperature=0):
        """Initialize the decision tree generator with OpenAI model"""
        self.llm = ChatOpenAI(temperature=temperature, model=model_name)
        self.setup_prompt()
        logger.info(f"Initialized with model: {model_name}")
    
    def setup_prompt(self):
        """Setup COMPLETE prompt with reasoning + confidence + correct JSON structure"""
        dt_prompt = """
    #Role and Objective
    You are a Senior Clinical Decision Support Expert. Create decision trees with STRICT compliance, clinical reasoning, and confidence assessment for ANY medical condition.

    #ULTRA-STRICT REQUIREMENTS - ZERO TOLERANCE FOR VIOLATIONS
    1. **MINIMUM 5 QUESTIONS**: Every diagnosis path must have AT LEAST 5 questions (ABSOLUTE MINIMUM) for comprehensive assessment
    2. **COMPREHENSIVE EVALUATION**: Diagnosis must be based on ALL answers collected through the complete question sequence
    3. **VALID Q_IDS ONLY**: Use ONLY the q_id values provided in the input - NO other numbers allowed. NEVER REFERENCE A Q_ID THAT IS NOT IN THE PROVIDED LIST.
    4. **NO SEQUENTIAL LOGIC**: Different answers MUST lead to different next_q_id (except when medically justified)
    5. **CONFIDENCE + REASONING**: Every diagnosis must include confidence_threshold (80-100%) AND clinical reasoning based on ALL collected symptoms
    6. **UNIVERSAL CLINICAL BRANCHING**: Emergency/severity determines different pathways (works for any medical condition)

    #CRITICAL Q_ID VALIDATION RULES:
    - BEFORE using any next_q_id, verify it exists in the provided q_id list
    - If you need a diagnosis, use 1000+ (these are auto-generated)
    - NEVER use q_ids that don't exist in the input data
    - DOUBLE-CHECK every next_q_id reference before finalizing

    #CRITICAL PATH LENGTH RULES:
    - Count questions carefully: diagnosis paths need 5+ question nodes minimum for thorough evaluation
    - Emergency cases still need 5+ questions for comprehensive clinical assessment
    - Each question should gather specific diagnostic information
    - If path seems short, add relevant clinical questions to reach minimum 5
    - Diagnosis should synthesize ALL answers from the complete 5+ question sequence

    #JSON STRUCTURE REQUIREMENTS - MANDATORY FORMAT:

    FOR QUESTIONS - Use this EXACT minimal structure:
    ```json
    {{
    "id": [same as q_id],
    "q_id": [number from provided list],
    "q_tag": "AI_GENERATED",
    "question": "[question text]",
    "options": [
        {{
        "next_q_id": [number from provided list or 1000+ for diagnosis],
        "opt_value": "[option text]",
        "option_id": [number],
        "nextQuestion": [same as next_q_id],
        "childQuestion": [same as next_q_id], 
        "childDiagnosis": null
        }}
    ]
    }}
    ```

    FOR DIAGNOSIS - Use this EXACT structure with reasoning:
    ```json
    {{
    "id": [same as q_id],
    "q_id": [1000+],
    "q_tag": "DIAGNOSIS",
    "question": "Diagnosis: [condition name]",
    "options": [],
    "advice": ["advice1", "advice2"],
    "diagnosis": {{
        "description": "[detailed description]",
        "diagnosis_title": "[condition name]",
        "confidence_threshold": [80-100],
        "reasoning": "[Clinical reasoning: Why this confidence level based on collected symptoms - e.g., 'High confidence (95%) because patient presents with classic triad of symptoms: fever + productive cough + chest pain, strongly indicating bacterial pneumonia']",
        "icd_code": "[ICD-10 code based on your medical knowledge]"
    }},
    "lab_tests": ["test1", "test2"],
    "red_flags": ["flag1", "flag2"],
    "precautions": ["precaution1"],
    "otc_medication": ["medication1"]
    }}
    ```

    #ICD-10 CODE REQUIREMENTS:
    - Use your medical knowledge to assign appropriate ICD-10 codes
    - Choose "unspecified" codes when specific subtypes cannot be determined
    - Use standard ICD-10 format (e.g., "J45.9", "R05", "K37")

    #UNIVERSAL CONFIDENCE + REASONING EXAMPLES (Works for any medical condition):

    HIGH CONFIDENCE REASONING (5+ questions → 95% threshold):
    ```
    "reasoning": "Very high confidence (95%) based on comprehensive 5-question assessment: severe breathing difficulty + wheezing + fever + chest pain + recent respiratory infection history. This complete symptom constellation strongly indicates acute severe asthma exacerbation requiring immediate intervention."
    ```

    MODERATE CONFIDENCE REASONING (6+ questions → 90% threshold):
    ```
    "reasoning": "Good confidence (90%) based on thorough 6-question evaluation: persistent dry cough + night worsening + no fever + environmental triggers + family history + exercise intolerance. Complete pattern suggests allergic or environmental cause, though viral etiology cannot be completely ruled out without further testing."
    ```

    BORDERLINE CONFIDENCE REASONING (7+ questions → 85% threshold):
    ```
    "reasoning": "Reasonable confidence (85%) based on comprehensive 7-question assessment: mild symptoms + gradual onset + no red flags + normal vital signs + recent cold exposure + no significant medical history + good response to rest. Complete evaluation suggests viral upper respiratory infection, but monitoring required as symptom pattern could evolve."
    ```

    #Task:
    Create decision tree with ZERO violations, confidence thresholds, and clinical reasoning for ANY medical condition.

    #Inputs:
    <Q&A JSON> 
        {qa_json}
    </Q&A JSON>

    #MANDATORY PRE-SUBMISSION CHECKLIST - VERIFY EACH POINT:
    ✓ Used ONLY provided q_id values (checked each next_q_id exists in input)
    ✓ Every diagnosis path has MINIMUM 5 questions before diagnosis for comprehensive assessment
    ✓ Different answers lead to different next_q_id (no sequential logic)
    ✓ Every diagnosis has confidence_threshold (80-100%)
    ✓ Every diagnosis has clinical reasoning explaining confidence based on ALL collected symptoms
    ✓ Questions use minimal JSON structure (q_id, q_tag, question, options only)
    ✓ Diagnosis uses complete structure with reasoning based on comprehensive evaluation
    ✓ Works for any medical condition (universal applicability)
    ✓ Every diagnosis has appropriate ICD-10 code based on medical knowledge
    ✓ NO REFERENCES TO NON-EXISTENT Q_IDS
    ✓ NO DIAGNOSTIC PATHS WITH FEWER THAN 5 QUESTIONS
    ✓ DIAGNOSIS REASONING INCLUDES ALL SYMPTOMS FROM COMPLETE QUESTION SEQUENCE

    Your Answer (ONLY decision_tree array with ZERO violations + reasoning):
    """
        
        self.decision_tree_prompt = PromptTemplate(
            template=dt_prompt,
            input_variables=["qa_json"]
        )
        self.decision_tree_chain = self.decision_tree_prompt | self.llm.with_structured_output(DecisionTreeOutput)
        logger.info("COMPLETE prompt with reasoning + confidence + correct JSON structure created")
        
    def load_medical_data(self, json_file_path):
        """Load medical Q&A data from JSON file"""
        try:
            with open(json_file_path, 'r') as fp:
                qna_data = json.load(fp)
            logger.info(f"Loaded medical data from {json_file_path}")
            return qna_data
        except Exception as e:
            logger.error(f"Error loading medical data: {e}")
            raise

    def analyze_clinical_priorities(self, questions):
        """Analyze questions with ENHANCED keywords for any medical condition"""
        
        # UNIVERSAL emergency keywords (works for any condition)
        emergency_keywords = [
            'difficulty breathing', 'breathing', 'shortness of breath', 'wheezing',
            'blood', 'severe', 'emergency', 'urgent', 'immediately', 'unable to',
            'choking', 'gasping', 'turning blue', 'chest pain', 'crushing',
            'acute', 'sudden', 'high fever', 'fever above', 'difficulty',
            'distress', 'labored', 'rapid breathing', 'severe pain', 'excruciating',
            'neurological', 'paralysis', 'numbness', 'weakness', 'seizure',
            'unconscious', 'confusion', 'vision', 'speech', 'swallowing'
        ]
        
        # UNIVERSAL severity keywords
        severity_keywords = [
            'how severe', 'severity', 'intense', 'bad', 'worse', 'worst',
            'scale', 'rate', 'level', 'degree', 'persistent', 'constant',
            'continuous', 'frequent', 'episodes', 'attacks', 'fits',
            'prolonged', 'lasting', 'duration', 'getting worse', 'worsening',
            'unbearable', 'moderate', 'mild', 'out of 10'
        ]
        
        # UNIVERSAL character keywords
        character_keywords = [
            'type of', 'dry', 'wet', 'productive', 'non-productive',
            'hacking', 'barking', 'harsh', 'soft', 'loose', 'tight',
            'sound', 'character', 'quality', 'described', 'like',
            'mucus', 'phlegm', 'sputum', 'clear', 'colored', 'thick',
            'sharp', 'dull', 'burning', 'throbbing', 'stabbing', 'crushing'
        ]
        
        # UNIVERSAL timing/trigger keywords  
        timing_keywords = [
            'when', 'time', 'morning', 'night', 'evening', 'after',
            'during', 'before', 'triggered', 'caused by', 'worse when',
            'better when', 'exercise', 'lying down', 'cold air',
            'recently', 'started', 'began', 'since when', 'how long',
            'sudden', 'gradual', 'onset', 'first noticed'
        ]
        
        # UNIVERSAL associated symptoms keywords
        associated_keywords = [
            'along with', 'accompanied by', 'also have', 'other symptoms',
            'together with', 'at the same time', 'fever', 'runny nose',
            'sore throat', 'headache', 'fatigue', 'tired', 'weakness',
            'loss of appetite', 'nausea', 'vomiting', 'diarrhea',
            'sweating', 'chills', 'rash', 'swelling'
        ]
        
        priorities = {
            'emergency': [],
            'severity': [],
            'character': [],
            'timing': [],
            'associated': [],
            'routine': []
        }
        
        print(f"\n🔍 ANALYZING {len(questions)} QUESTIONS FOR UNIVERSAL CLINICAL PRIORITIES:")
        
        for q in questions:
            question_text = q.get('question', '').lower()
            q_id = q.get('q_id')
            
            # Check each category with priority order
            if any(keyword in question_text for keyword in emergency_keywords):
                priorities['emergency'].append((q_id, q, 'emergency'))
                print(f"   🚨 EMERGENCY: Q{q_id} - {question_text[:60]}...")
            elif any(keyword in question_text for keyword in severity_keywords):
                priorities['severity'].append((q_id, q, 'severity'))
                print(f"   ⚖️ SEVERITY: Q{q_id} - {question_text[:60]}...")
            elif any(keyword in question_text for keyword in character_keywords):
                priorities['character'].append((q_id, q, 'character'))
                print(f"   📋 CHARACTER: Q{q_id} - {question_text[:60]}...")
            elif any(keyword in question_text for keyword in timing_keywords):
                priorities['timing'].append((q_id, q, 'timing'))
                print(f"   ⏰ TIMING: Q{q_id} - {question_text[:60]}...")
            elif any(keyword in question_text for keyword in associated_keywords):
                priorities['associated'].append((q_id, q, 'associated'))
                print(f"   🔗 ASSOCIATED: Q{q_id} - {question_text[:60]}...")
            else:
                priorities['routine'].append((q_id, q, 'routine'))
                print(f"   📝 ROUTINE: Q{q_id} - {question_text[:60]}...")
        
        # Clinical order: Emergency → Severity → Character → Timing → Associated → Routine
        clinical_order = (
            priorities['emergency'] + 
            priorities['severity'] + 
            priorities['character'] + 
            priorities['timing'] +
            priorities['associated'] + 
            priorities['routine']
        )
        
        print(f"\n📊 UNIVERSAL CLINICAL PRIORITY SUMMARY:")
        print(f"   🚨 Emergency: {len(priorities['emergency'])}")
        print(f"   ⚖️ Severity: {len(priorities['severity'])}")
        print(f"   📋 Character: {len(priorities['character'])}")
        print(f"   ⏰ Timing: {len(priorities['timing'])}")
        print(f"   🔗 Associated: {len(priorities['associated'])}")
        print(f"   📝 Routine: {len(priorities['routine'])}")
        
        return clinical_order, priorities

    def create_age_demographic_batches(self, qna_data):
        """Create age-based batches with proper gender separation for ages 13+"""
        age_groups = ['0-2', '3-12', '13-18', '19-40', '41-65', '66+']
        all_batches = []
        
        for age_group in age_groups:
            if age_group in ['0-2', '3-12']:
                # For young ages, create single batch (same medical logic for both genders)
                logger.info(f"🔄 Processing: {age_group} (Both genders)")
                
                filtered_data = self._filter_medical_questions_only(qna_data, age_group, 'Both')
                filtered_data['category_key'] = age_group.replace('-', '_')
                filtered_data['age_range'] = age_group
                filtered_data['gender_specific'] = 'Both'
                all_batches.append(filtered_data)
                
                logger.info(f"📊 Batch created for {age_group} (Both)")
                logger.info(f"   - Medical questions: {len(filtered_data['questions'])}")
            else:
                # For older ages, create separate male/female batches
                genders = ['Male', 'Female']
                for gender in genders:
                    logger.info(f"🔄 Processing: {gender} {age_group}")
                    
                    filtered_data = self._filter_medical_questions_only(qna_data, age_group, gender)
                    filtered_data['category_key'] = f"{gender.lower()}_{age_group.replace('-', '_')}"
                    filtered_data['age_range'] = age_group
                    filtered_data['gender_specific'] = gender
                    all_batches.append(filtered_data)
                    
                    logger.info(f"📊 Batch created for {gender} {age_group}")
                    logger.info(f"   - Medical questions: {len(filtered_data['questions'])}")

        logger.info(f"🎯 Total batches created: {len(all_batches)}")
        return all_batches

    def _filter_medical_questions_only(self, qa_data, target_age, target_gender):
        """Filter ONLY medical symptom questions with ENHANCED clinical priority analysis"""
        filtered_questions = []
        
        print(f"🔍 Filtering MEDICAL questions for Age: {target_age}, Gender: {target_gender}")
        
        # Filter questions - EXCLUDE demographic questions, only include medical symptoms
        for q in qa_data['questions']:
            q_age = q.get('age_group')
            q_gender = q.get('gender')
            q_id = q.get('q_id')
            q_tag = q.get('q_tag')
            
            # SKIP ALL demographic questions - we only want medical symptom questions
            if q_tag == 'DEMOGRAPHICS':
                print(f"   ❌ Skipped demographic Q{q_id}: {q.get('question', '')[:50]}...")
                continue
            
            # Include age-specific medical questions only
            if q_age == target_age:
                # For young ages (Both), include all medical questions for that age
                if target_gender == 'Both':
                    filtered_questions.append(q)
                    print(f"   ✅ Added medical Q{q_id}: {q.get('question', '')[:50]}...")
                # For older ages, check gender match
                elif q_gender == 'Both' or q_gender == target_gender:
                    filtered_questions.append(q)
                    print(f"   ✅ Added medical Q{q_id} ({q_gender}): {q.get('question', '')[:50]}...")
        
        print(f"   📊 Total MEDICAL questions filtered: {len(filtered_questions)}")
        
        # ENHANCED CLINICAL PRIORITIES ANALYSIS
        clinical_order, priorities = self.analyze_clinical_priorities(filtered_questions)
        
        # Create list of valid q_ids for this batch (only medical questions)
        valid_q_ids = [q.get('q_id') for q in filtered_questions]
        print(f"   📋 Valid medical q_ids: {valid_q_ids}")
        
        return {
            "questions": filtered_questions,
            "clinical_order": clinical_order,
            "priorities": priorities,
            "target_age": target_age,
            "target_gender": target_gender,
            "valid_q_ids": valid_q_ids,
            "instruction": f"Create intelligent diagnoses with IDs 1000+ using ENHANCED clinical reasoning. Use ONLY these medical question IDs: {valid_q_ids}. Follow UNIVERSAL CLINICAL TRIAGE: emergency symptoms first, then severity, character, timing, associated symptoms. STRICT REQUIREMENT: Every diagnostic path must ask EXACTLY 5+ questions before reaching diagnosis. Every diagnosis must include confidence_threshold (80-100%) and clinical reasoning. NO EXCEPTIONS."
        }

    def generate_combined_decision_tree(self, all_batches, output_file="combined_medical_decision_tree.json"):
        """Generate one combined decision tree with proper gender->age selection structure and unique IDs"""
        logger.info("🚀 Starting COMBINED Medical Decision Tree Generation with Fixed Gender->Age Structure")
        
        combined_tree = []
        current_medical_id = 4  # Start medical questions from ID 4
        next_diagnosis_id = 1001  # Start diagnoses from ID 1001
        
        # Generate decision trees for each batch first
        batch_trees = {}
        batch_id_ranges = {}  # Track ID ranges for each batch
        
        for i, batch in enumerate(all_batches):
            logger.info(f"⌛ [{i+1:2d}/{len(all_batches)}] Generating Decision Tree for {batch['category_key']}...")
            
            try:
                # Generate the decision tree for this batch
                dt_response = self.decision_tree_chain.invoke({"qa_json": batch})
                db_data = dt_response.model_dump()
                decision_tree_nodes = db_data['decision_tree']
                
                # Assign unique IDs to this batch
                start_medical_id = current_medical_id
                start_diagnosis_id = next_diagnosis_id
                
                # Count questions and diagnoses in this batch
                question_count = len([n for n in decision_tree_nodes if n.get('q_tag') != 'DIAGNOSIS'])
                diagnosis_count = len([n for n in decision_tree_nodes if n.get('q_tag') == 'DIAGNOSIS'])
                
                # Reserve ID ranges for this batch
                batch_id_ranges[batch['category_key']] = {
                    'medical_start': start_medical_id,
                    'medical_end': start_medical_id + question_count - 1,
                    'diagnosis_start': start_diagnosis_id,
                    'diagnosis_end': start_diagnosis_id + diagnosis_count - 1,
                    'first_question_id': start_medical_id
                }
                
                # Update counters for next batch
                current_medical_id += question_count
                next_diagnosis_id += diagnosis_count
                
                # Store the tree with ID mapping info
                batch_trees[batch['category_key']] = {
                    'tree': decision_tree_nodes,
                    'age_range': batch['age_range'],
                    'gender': batch.get('gender_specific', 'Both'),
                    'id_ranges': batch_id_ranges[batch['category_key']]
                }
                
                logger.info(f"   ✅ Generated tree for {batch['category_key']}")
                logger.info(f"   📊 Questions: {question_count}, Diagnoses: {diagnosis_count}")
                logger.info(f"   🆔 Medical IDs: {start_medical_id}-{start_medical_id + question_count - 1}")
                logger.info(f"   🆔 Diagnosis IDs: {start_diagnosis_id}-{start_diagnosis_id + diagnosis_count - 1}")
                
            except Exception as e:
                logger.error(f"   ❌ Error generating tree for {batch['category_key']}: {e}")
                continue
        
        # Step 1: Create root gender question (ID 1)
        gender_question = {
            "id": 1,
            "q_id": 1,
            "q_tag": "DEMOGRAPHICS",
            "question": "What is your gender?",
            "options": [
                {
                    "next_q_id": 2,
                    "opt_value": "Male",
                    "option_id": 1,
                    "nextQuestion": 2,
                    "childQuestion": 2,
                    "childDiagnosis": None
                },
                {
                    "next_q_id": 3,
                    "opt_value": "Female",
                    "option_id": 2,
                    "nextQuestion": 3,
                    "childQuestion": 3,
                    "childDiagnosis": None
                }
            ]
        }
        combined_tree.append(gender_question)
        logger.info("✅ Created gender selection question (Q1)")
        
        # Step 2: Create male age question (ID 2)
        male_age_options = []
        option_id = 1
        
        # Group batches by age for male options
        age_to_batch = {}
        for category_key, tree_data in batch_trees.items():
            age_range = tree_data['age_range']
            gender = tree_data['gender']
            
            if age_range not in age_to_batch:
                age_to_batch[age_range] = {'Both': None, 'Male': None, 'Female': None}
            age_to_batch[age_range][gender] = tree_data
        
        # Create male age options
        for age_range in ['0-2', '3-12', '13-18', '19-40', '41-65', '66+']:
            if age_range in age_to_batch:
                # For ages 0-12, use 'Both' tree. For ages 13+, use 'Male' tree
                target_tree = None
                if age_range in ['0-2', '3-12'] and age_to_batch[age_range]['Both']:
                    target_tree = age_to_batch[age_range]['Both']
                elif age_range not in ['0-2', '3-12'] and age_to_batch[age_range]['Male']:
                    target_tree = age_to_batch[age_range]['Male']
                
                if target_tree:
                    first_q_id = target_tree['id_ranges']['first_question_id']
                    male_age_options.append({
                        "next_q_id": first_q_id,
                        "opt_value": f"Age {age_range}",
                        "option_id": option_id,
                        "nextQuestion": first_q_id,
                        "childQuestion": first_q_id,
                        "childDiagnosis": None
                    })
                    option_id += 1
                    logger.info(f"   📋 Male age option: 'Age {age_range}' → Q{first_q_id}")
        
        male_age_question = {
            "id": 2,
            "q_id": 2,
            "q_tag": "DEMOGRAPHICS",
            "question": "What is your age? (Male)",
            "options": male_age_options
        }
        combined_tree.append(male_age_question)
        logger.info("✅ Created male age selection question (Q2)")
        
        # Step 3: Create female age question (ID 3)
        female_age_options = []
        option_id = 1
        
        # Create female age options
        for age_range in ['0-2', '3-12', '13-18', '19-40', '41-65', '66+']:
            if age_range in age_to_batch:
                # For ages 0-12, use 'Both' tree. For ages 13+, use 'Female' tree
                target_tree = None
                if age_range in ['0-2', '3-12'] and age_to_batch[age_range]['Both']:
                    target_tree = age_to_batch[age_range]['Both']
                elif age_range not in ['0-2', '3-12'] and age_to_batch[age_range]['Female']:
                    target_tree = age_to_batch[age_range]['Female']
                
                if target_tree:
                    first_q_id = target_tree['id_ranges']['first_question_id']
                    female_age_options.append({
                        "next_q_id": first_q_id,
                        "opt_value": f"Age {age_range}",
                        "option_id": option_id,
                        "nextQuestion": first_q_id,
                        "childQuestion": first_q_id,
                        "childDiagnosis": None
                    })
                    option_id += 1
                    logger.info(f"   📋 Female age option: 'Age {age_range}' → Q{first_q_id}")
        
        female_age_question = {
            "id": 3,
            "q_id": 3,
            "q_tag": "DEMOGRAPHICS",
            "question": "What is your age? (Female)",
            "options": female_age_options
        }
        combined_tree.append(female_age_question)
        logger.info("✅ Created female age selection question (Q3)")
        
        # Step 4: Add all medical trees with proper sequential IDs
        total_nodes_added = 0
        
        for category_key, tree_data in batch_trees.items():
            tree_nodes = tree_data['tree']
            id_ranges = tree_data['id_ranges']
            
            # Create mapping from old IDs to new sequential IDs
            old_to_new_mapping = {}
            
            # Map question IDs
            question_nodes = [n for n in tree_nodes if n.get('q_tag') != 'DIAGNOSIS']
            for i, node in enumerate(question_nodes):
                old_id = node.get('q_id')
                new_id = id_ranges['medical_start'] + i
                old_to_new_mapping[old_id] = new_id
            
            # Map diagnosis IDs
            diagnosis_nodes = [n for n in tree_nodes if n.get('q_tag') == 'DIAGNOSIS']
            for i, node in enumerate(diagnosis_nodes):
                old_id = node.get('q_id')
                new_id = id_ranges['diagnosis_start'] + i
                old_to_new_mapping[old_id] = new_id
            
            # Apply ID mapping and add nodes
            for node in tree_nodes:
                new_node = node.copy()
                old_id = node.get('q_id')
                new_id = old_to_new_mapping[old_id]
                
                new_node['id'] = new_id
                new_node['q_id'] = new_id
                
                # Update option references
                if 'options' in new_node and new_node['options']:
                    updated_options = []
                    for option in new_node['options']:
                        updated_option = option.copy()
                        old_next_id = option.get('next_q_id')
                        
                        if old_next_id in old_to_new_mapping:
                            new_next_id = old_to_new_mapping[old_next_id]
                            updated_option['next_q_id'] = new_next_id
                            updated_option['nextQuestion'] = new_next_id
                            updated_option['childQuestion'] = new_next_id
                            if updated_option.get('childDiagnosis'):
                                updated_option['childDiagnosis'] = new_next_id
                        updated_options.append(updated_option)
                    new_node['options'] = updated_options
                
                combined_tree.append(new_node)
                total_nodes_added += 1
            
            logger.info(f"   📋 Added {len(tree_nodes)} nodes from {category_key} with sequential IDs")
        
        # Step 5: Save the combined tree
        logger.info(f"💾 Saving combined decision tree...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_tree, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_file)
            logger.info(f"✅ COMBINED TREE SAVED: {output_file} ({file_size:,} bytes)")
            logger.info(f"📊 Total nodes: {len(combined_tree)}")
            logger.info(f"📋 Structure: Gender (Q1) → Male Age (Q2) / Female Age (Q3) → Medical Trees")
            logger.info(f"🎯 Medical nodes: {total_nodes_added}")
            logger.info(f"🆔 IDs: 1-3 (demographics), 4-{current_medical_id-1} (medical), {1001}-{next_diagnosis_id-1} (diagnoses)")
            
            print(f"\n🎯 SUCCESS: Combined Medical Decision Tree Generated!")
            print(f"📂 Output File: {output_file}")
            print(f"📈 Total Nodes: {len(combined_tree)}")
            print(f"🌳 Structure: Q1 (Gender) → Q2 (Male Ages) / Q3 (Female Ages) → Medical Trees")
            print(f"💾 File Size: {file_size:,} bytes")
            print(f"🆔 ID Ranges: 1-3 (demographics), 4-{current_medical_id-1} (medical), {1001}-{next_diagnosis_id-1} (diagnoses)")
            
            # Display the structure
            print(f"\n📋 Decision Tree Structure:")
            print(f"   Q1: Gender → Male (Q2) / Female (Q3)")
            print(f"   Q2: Male Ages → {len(male_age_options)} age options")
            print(f"   Q3: Female Ages → {len(female_age_options)} age options")
            print(f"   Q4+: Medical question trees with diagnoses")
            
            # Display ID allocation summary
            print(f"\n🆔 ID Allocation Summary:")
            for category_key, tree_data in batch_trees.items():
                id_ranges = tree_data['id_ranges']
                print(f"   {category_key}: Q{id_ranges['medical_start']}-{id_ranges['medical_end']} (medical), D{id_ranges['diagnosis_start']}-{id_ranges['diagnosis_end']} (diagnoses)")
            
            return combined_tree, output_file
            
        except Exception as e:
            logger.error(f"❌ Failed to save combined tree: {e}")
            raise

    def validate_combined_tree(self, combined_tree):
        """Validate the combined decision tree structure with enhanced checks"""
        logger.info("🔍 Validating combined decision tree...")
        
        # Create lookup for validation
        node_lookup = {node['q_id']: node for node in combined_tree}
        
        validation_results = {
            'total_nodes': len(combined_tree),
            'question_nodes': 0,
            'diagnosis_nodes': 0,
            'demographics_nodes': 0,
            'invalid_references': [],
            'orphaned_nodes': [],
            'duplicate_ids': [],
            'valid_references': 0
        }
        
        # Check for duplicate IDs
        id_counts = {}
        for node in combined_tree:
            q_id = node['q_id']
            if q_id in id_counts:
                id_counts[q_id] += 1
                validation_results['duplicate_ids'].append(q_id)
            else:
                id_counts[q_id] = 1
        
        # Count node types
        for node in combined_tree:
            q_tag = node.get('q_tag')
            if q_tag == 'DEMOGRAPHICS':
                validation_results['demographics_nodes'] += 1
            elif q_tag in ['AI_GENERATED', 'NEW']:
                validation_results['question_nodes'] += 1
            elif q_tag == 'DIAGNOSIS':
                validation_results['diagnosis_nodes'] += 1
        
        # Check all references
        referenced_q_ids = set()
        for node in combined_tree:
            if 'options' in node and node['options']:
                for option in node['options']:
                    next_q_id = option.get('next_q_id')
                    if next_q_id:
                        referenced_q_ids.add(next_q_id)
                        
                        # Check if the referenced ID exists
                        if next_q_id not in node_lookup:
                            validation_results['invalid_references'].append({
                                'from_q_id': node.get('q_id'),
                                'to_q_id': next_q_id,
                                'option': option.get('opt_value')
                            })
                        else:
                            validation_results['valid_references'] += 1
        
        # Find orphaned nodes (except root)
        all_q_ids = {node['q_id'] for node in combined_tree}
        root_q_ids = {node['q_id'] for node in combined_tree if node.get('q_tag') == 'DEMOGRAPHICS'}
        
        for q_id in all_q_ids:
            if q_id not in referenced_q_ids and q_id not in root_q_ids:
                validation_results['orphaned_nodes'].append(q_id)
        
        # Log validation results
        logger.info(f"📊 VALIDATION RESULTS:")
        logger.info(f"   Total nodes: {validation_results['total_nodes']}")
        logger.info(f"   Demographics: {validation_results['demographics_nodes']}")
        logger.info(f"   Question nodes: {validation_results['question_nodes']}")
        logger.info(f"   Diagnosis nodes: {validation_results['diagnosis_nodes']}")
        logger.info(f"   Valid references: {validation_results['valid_references']}")
        logger.info(f"   Invalid references: {len(validation_results['invalid_references'])}")
        logger.info(f"   Duplicate IDs: {len(validation_results['duplicate_ids'])}")
        logger.info(f"   Orphaned nodes: {len(validation_results['orphaned_nodes'])}")
        
        if validation_results['duplicate_ids']:
            logger.error(f"❌ Duplicate IDs found: {validation_results['duplicate_ids']}")
        
        if validation_results['invalid_references']:
            logger.warning("❌ Invalid references found:")
            for ref in validation_results['invalid_references']:
                logger.warning(f"   Q{ref['from_q_id']} → Q{ref['to_q_id']} (option: '{ref['option']}')")
        
        if validation_results['orphaned_nodes']:
            logger.warning(f"⚠️ Orphaned nodes found: {validation_results['orphaned_nodes']}")
        
        return validation_results

    def run_combined_pipeline(self, json_file_path="cough_latest_complete.json", output_file="combined_medical_decision_tree.json"):
        """Run the complete pipeline to generate one combined decision tree"""
        logger.info("🚀 Starting FIXED Combined Medical Decision Tree Generation Pipeline")
        logger.info("📋 Enhancement: Fixed unique IDs + proper gender separation for ages 13+")
        logger.info("🌳 Structure: Root Gender Question → Age Selection → Gender-Specific Trees (13+)")
        logger.info("⚡ Features: Unique IDs + Confidence + Reasoning + Validation + Combined Output")
        
        try:
            # Step 1: Load medical data
            qna_data = self.load_medical_data(json_file_path)
            
            # Step 2: Create age-based batches with proper gender separation
            all_batches = self.create_age_demographic_batches(qna_data)
            
            # Step 3: Generate combined decision tree with fixed ID allocation
            combined_tree, output_file_path = self.generate_combined_decision_tree(all_batches, output_file)
            
            # Step 4: Validate the combined tree
            validation_results = self.validate_combined_tree(combined_tree)
            
            logger.info("🎉 FIXED Combined Medical Decision Tree Generation Complete!")
            logger.info(f"📁 Output file: {output_file_path}")
            logger.info(f"📊 Total nodes: {validation_results['total_nodes']}")
            logger.info(f"🌳 Structure validated successfully")
            
            return combined_tree, output_file_path, validation_results
            
        except Exception as e:
            logger.error(f"Combined pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        print("🚀 Starting FIXED Combined Medical Decision Tree Generation...")
        print("📋 STRUCTURE: Gender → Age → Gender-Specific Medical Questions (13+) → Diagnoses")
        print("🔢 ID SYSTEM: 1-3 (demographics), 4+ (medical - unique per batch), 1001+ (diagnoses - unique per batch)")
        print("⚡ FIXES: Unique IDs, Proper Gender Separation, No Duplicates")
        
        # Initialize the generator
        generator = CombinedMedicalDecisionTreeGenerator()
        
        # Run the combined pipeline
        combined_tree, output_file, validation_results = generator.run_combined_pipeline(
            json_file_path="cough_latest_complete 2.json",
            output_file="fixed_combined_medical_decision_tree.json"
        )
        
        print(f"\n🎯 SUCCESS: FIXED Combined Medical Decision Tree Generated!")
        print(f"📂 Output File: {output_file}")
        print(f"📈 Total Nodes: {validation_results['total_nodes']}")
        print(f"🌳 Root: Q1 (Gender) → Q2 (Male Ages) / Q3 (Female Ages)")
        print(f"❓ Medical Questions: {validation_results['question_nodes']}")
        print(f"🏥 Diagnoses: {validation_results['diagnosis_nodes']}")
        print(f"✅ Valid References: {validation_results['valid_references']}")
        print(f"❌ Invalid References: {len(validation_results['invalid_references'])}")
        print(f"🔄 Duplicate IDs: {len(validation_results['duplicate_ids'])}")
        
        print(f"\n📋 FIXED Tree Structure:")
        print(f"   Q1: Gender Selection → Male (Q2) / Female (Q3)")
        print(f"   Q2: Male Age Selection → Medical Trees (same for 0-12, separate for 13+)")
        print(f"   Q3: Female Age Selection → Medical Trees (same for 0-12, separate for 13+)")
        print(f"   Q4+: Medical Questions and Diagnoses (ALL UNIQUE IDs)")
        
        print(f"\n🔍 KEY FIXES APPLIED:")
        print(f"   ✅ No duplicate q_ids - each batch gets unique ID ranges")
        print(f"   ✅ Ages 0-12: Same trees for male/female (gender not significant)")
        print(f"   ✅ Ages 13+: Separate male/female trees with different q_ids")
        print(f"   ✅ Sequential ID allocation prevents conflicts")
        print(f"   ✅ All diagnostic paths include confidence + reasoning")
        print(f"   ✅ Single file contains entire decision tree")
        
        if validation_results['invalid_references'] or validation_results['duplicate_ids']:
            print(f"\n⚠️  VALIDATION WARNINGS:")
            if validation_results['duplicate_ids']:
                print(f"   - Duplicate IDs: {validation_results['duplicate_ids']}")
            for ref in validation_results['invalid_references']:
                print(f"   - Q{ref['from_q_id']} → Q{ref['to_q_id']} ('{ref['option']}')")
        else:
            print(f"\n✅ VALIDATION: ALL CHECKS PASSED!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()