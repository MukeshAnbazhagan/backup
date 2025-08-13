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

class MedicalDecisionTreeGenerator:
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
    "q_id": [number from provided list],
    "q_tag": "NEW",
    "question": "[question text]",
    "options": [
        {{
        "next_q_id": [number from provided list or 1000+ for diagnosis],
        "opt_value": "[option text]",
        "option_id": [number]
        }}
    ]
    }}
    ```

    FOR DIAGNOSIS - Use this EXACT structure with reasoning:
    ```json
    {{
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

    #UNIVERSAL CLINICAL LOGIC (Works for any medical protocol):

    HIGH-RISK SYMPTOMS (Any condition - MUST still have 5+ questions + comprehensive reasoning):
    - Emergency indicators → Urgent-focused pathway BUT still ask 5+ questions for thorough assessment
    - Severe pain/symptoms → Priority pathway BUT still collect comprehensive evidence through 5+ questions
    - Red flag symptoms → Specialized pathway BUT still build complete clinical picture with 5+ questions

    CONFIDENCE THRESHOLDS WITH REASONING (Universal):
    - 95-100%: Classic symptom constellation, definitive pattern (requires 5+ comprehensive questions)
    - 90-94%: Strong evidence, minor uncertainties (requires 6+ thorough questions)
    - 85-89%: Good evidence, some differential considerations (requires 7+ detailed questions)
    - 80-84%: Reasonable evidence, monitoring recommended (requires 8+ comprehensive questions)

    #UNIVERSAL CLINICAL PATHWAY EXAMPLES:

    EMERGENCY PATHWAY (Any urgent condition - MINIMUM 5 questions → 95% confidence):
    ```
    Q1: [Most urgent symptom]? → Emergency finding
    Q2: [Severity assessment]? → High severity  
    Q3: [Associated urgent symptoms]? → Related findings
    Q4: [Risk factors/timing]? → Pattern confirmation
    Q5: [Additional clinical signs]? → Complete assessment
    → DIAGNOSIS: [Urgent condition]
    confidence_threshold: 95
    reasoning: "Very high confidence (95%) due to comprehensive 5-question emergency assessment: [list all 5 symptoms/findings]. This complete clinical picture strongly indicates [condition] requiring immediate intervention."
    ```

    STANDARD PATHWAY (Moderate symptoms - any condition - MINIMUM 5 questions):
    ```
    Q1: [Primary symptom]? → Present
    Q2: [Character/quality]? → Specific type
    Q3: [Associated symptoms]? → Related findings
    Q4: [Duration/triggers]? → Pattern assessment
    Q5: [History/context]? → Background factors
    → DIAGNOSIS: [Standard condition]
    confidence_threshold: 90
    reasoning: "Good confidence (90%) based on comprehensive 5-question evaluation: [list all 5 findings]. This complete symptom pattern strongly suggests [condition], with all major diagnostic criteria met."
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

    #REASONING REQUIREMENTS:
    - Explain WHY this confidence level based on ALL symptoms collected through 5+ questions
    - Reference specific symptom combinations from the complete question sequence
    - Mention any uncertainties or differential diagnoses considered
    - Justify the confidence percentage with comprehensive clinical evidence from all answers
    - Synthesize the complete clinical picture from all collected information

    #ZERO TOLERANCE VIOLATIONS - WILL CAUSE IMMEDIATE FAILURE:
    - Using q_id not in provided input = INVALID
    - Any diagnosis after <5 questions = INVALID
    - Sequential logic without justification = INVALID
    - Missing confidence_threshold = INVALID
    - Missing clinical reasoning based on comprehensive assessment = INVALID
    - Wrong JSON structure = INVALID
    - Missing ICD-10 code = INVALID
    - Referencing non-existent q_ids = INVALID
    - Diagnosis not based on complete symptom picture = INVALID

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

    def create_demographic_batches(self, qna_data):
        """Create optimized batches: single files for young ages, separate for older ages"""
        age_groups = ['0-2', '3-12', '13-18', '19-40', '41-65', '66+']
        all_batches = []

        for age_group in age_groups:
            if age_group in ['0-2', '3-12']:
                # For young ages, create single batch (same medical logic for both genders)
                logger.info(f"🔄 Processing: {age_group} (Both genders)")
                
                filtered_data = self._filter_medical_questions_only(qna_data, age_group, 'Both')
                filtered_data['category_key'] = age_group.replace('-', '_')
                filtered_data['filename'] = f"{age_group.replace('-', '_')}_both.json"
                all_batches.append(filtered_data)
                
                logger.info(f"📊 Batch created for {age_group} (Both)")
                logger.info(f"   - Medical questions: {len(filtered_data['questions'])}")
                logger.info(f"   - AI will generate diagnoses with confidence + reasoning")
            else:
                # For older ages, create separate male/female batches
                genders = ['Male', 'Female']
                for gender in genders:
                    logger.info(f"🔄 Processing: {gender} {age_group}")
                    
                    filtered_data = self._filter_medical_questions_only(qna_data, age_group, gender)
                    filtered_data['category_key'] = f"{gender.lower()}_{age_group.replace('-', '_')}"
                    filtered_data['filename'] = f"{gender.lower()}_{age_group.replace('-', '_')}.json"
                    all_batches.append(filtered_data)
                    
                    logger.info(f"📊 Batch created for {gender} {age_group}")
                    logger.info(f"   - Medical questions: {len(filtered_data['questions'])}")
                    logger.info(f"   - AI will generate diagnoses with confidence + reasoning")

        logger.info(f"🎯 Total batches created: {len(all_batches)}")
        logger.info(f"🧠 AI will create diagnoses with clinical reasoning")
        logger.info(f"📋 NO demographic questions included")
        logger.info(f"⚡ STRICT 3-4 question minimum enforced")
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
            "instruction": f"Create intelligent diagnoses with IDs 1000+ using ENHANCED clinical reasoning. Use ONLY these medical question IDs: {valid_q_ids}. Follow UNIVERSAL CLINICAL TRIAGE: emergency symptoms first, then severity, character, timing, associated symptoms. STRICT REQUIREMENT: Every diagnostic path must ask EXACTLY 3-4+ questions before reaching diagnosis. Every diagnosis must include confidence_threshold (80-100%) and clinical reasoning. NO EXCEPTIONS."
        }

    def trace_all_paths(self, decision_tree):
        """Trace all possible paths from root to diagnosis"""
        
        # Create lookup for quick access
        node_lookup = {node['q_id']: node for node in decision_tree}
        
        # Find root nodes (nodes that aren't referenced by any other node)
        referenced_q_ids = set()
        for node in decision_tree:
            if node.get('q_tag') == 'NEW':
                for opt in node.get('options', []):
                    referenced_q_ids.add(opt.get('next_q_id'))
        
        root_nodes = [node for node in decision_tree 
                      if node.get('q_tag') == 'NEW' and node.get('q_id') not in referenced_q_ids]
        
        all_paths = []
        
        def traverse_path(current_q_id, current_path):
            if current_q_id not in node_lookup:
                # Dead end - invalid reference
                return
                
            current_node = node_lookup[current_q_id]
            new_path = current_path + [current_node]
            
            if current_node.get('q_tag') == 'DIAGNOSIS':
                # End of path
                all_paths.append(new_path)
                return
            
            # Continue traversing
            for opt in current_node.get('options', []):
                traverse_path(opt.get('next_q_id'), new_path)
        
        # Start traversal from each root
        for root in root_nodes:
            traverse_path(root.get('q_id'), [])
        
        return all_paths

    def validate_decision_tree_comprehensive(self, decision_tree, valid_q_ids):
        """Comprehensive validation including reasoning field"""
        
        validation_results = {
            'sequential_violations': [],
            'invalid_q_id_references': [],
            'short_path_violations': [],
            'missing_confidence': [],
            'missing_reasoning': [],
            'missing_icd_code': [],
            'proper_branching': 0,
            'total_issues': 0
        }
        
        print(f"\n🔍 COMPREHENSIVE VALIDATION WITH REASONING:")
        print("=" * 70)
        
        # Get all node q_ids in the tree
        tree_q_ids = {node.get('q_id') for node in decision_tree}
        diagnosis_q_ids = {node.get('q_id') for node in decision_tree if node.get('q_tag') == 'DIAGNOSIS'}
        
        print(f"\n📊 TREE ANALYSIS:")
        print(f"   Valid input q_ids: {sorted(valid_q_ids)}")
        print(f"   Tree q_ids: {sorted(tree_q_ids)}")
        print(f"   Diagnosis q_ids: {sorted(diagnosis_q_ids)}")
        
        # VALIDATION 1: Check for invalid q_id references
        print(f"\n🔍 VALIDATION 1: Invalid Q_ID References")
        invalid_refs_found = False
        for node in decision_tree:
            if node.get('q_tag') == 'NEW' and node.get('options'):
                q_id = node.get('q_id')
                question = node.get('question', '')[:50]
                
                for opt in node.get('options', []):
                    next_q_id = opt.get('next_q_id')
                    
                    # Check if next_q_id exists in tree or valid input q_ids
                    if next_q_id not in tree_q_ids:
                        validation_results['invalid_q_id_references'].append({
                            'q_id': q_id,
                            'question': question,
                            'invalid_next_q_id': next_q_id,
                            'option': opt.get('opt_value')
                        })
                        print(f"   ❌ Q{q_id}: '{opt.get('opt_value')}' → Q{next_q_id} (DOESN'T EXIST)")
                        invalid_refs_found = True
        
        if not invalid_refs_found:
            print("   ✅ All Q_ID references are valid")
        
        # VALIDATION 2: Check sequential logic
        print(f"\n🔍 VALIDATION 2: Sequential Logic")
        for node in decision_tree:
            if node.get('q_tag') == 'NEW' and node.get('options'):
                q_id = node.get('q_id')
                question = node.get('question', '')[:50]
                options = node.get('options', [])
                
                if len(options) >= 2:
                    next_q_ids = [opt.get('next_q_id') for opt in options]
                    unique_next_ids = set(next_q_ids)
                    
                    if len(unique_next_ids) == 1:
                        validation_results['sequential_violations'].append({
                            'q_id': q_id,
                            'question': question,
                            'next_q_id': next_q_ids[0],
                            'options': [opt.get('opt_value') for opt in options]
                        })
                        print(f"   ❌ Q{q_id}: {question}...")
                        print(f"      All answers → Q{next_q_ids[0]} (SEQUENTIAL LOGIC)")
                    else:
                        validation_results['proper_branching'] += 1
                        print(f"   ✅ Q{q_id}: {question}...")
                        for opt in options:
                            print(f"      '{opt.get('opt_value')}' → Q{opt.get('next_q_id')}")
        
        # VALIDATION 3: Check path lengths
        print(f"\n🔍 VALIDATION 3: Path Length Analysis")
        all_paths = self.trace_all_paths(decision_tree)
        for i, path in enumerate(all_paths):
            path_length = len([p for p in path if p['q_tag'] == 'NEW'])
            diagnosis_node = path[-1] if path and path[-1]['q_tag'] == 'DIAGNOSIS' else None
            
            if path_length < 5:  # Changed from 3 to 5
                validation_results['short_path_violations'].append({
                    'path_number': i+1,
                    'path_length': path_length,
                    'path_q_ids': [p['q_id'] for p in path],
                    'diagnosis': diagnosis_node.get('question') if diagnosis_node else 'Unknown'
                })
                print(f"   ❌ Path {i+1}: {path_length} questions → {diagnosis_node.get('question', 'Unknown') if diagnosis_node else 'No diagnosis'}")
                print(f"      Q_IDs: {' → '.join(str(p['q_id']) for p in path)}")
                print(f"      ERROR: Need minimum 5 questions for comprehensive assessment")
            else:
                print(f"   ✅ Path {i+1}: {path_length} questions → {diagnosis_node.get('question', 'Unknown') if diagnosis_node else 'No diagnosis'}")

        # Also update the logging message in generate_decision_trees method:
        logger.info(f"   ⚡ ENFORCING UNIVERSAL clinical triage with confidence + reasoning + 5 question minimum")
        
        # VALIDATION 4: Check confidence and reasoning
        print(f"\n🔍 VALIDATION 4: Confidence + Reasoning")
        for node in decision_tree:
            if node.get('q_tag') == 'DIAGNOSIS':
                q_id = node.get('q_id')
                diagnosis_info = node.get('diagnosis', {})
                diagnosis_title = diagnosis_info.get('diagnosis_title', 'Unknown')
                
                # Check confidence
                if not diagnosis_info.get('confidence_threshold'):
                    validation_results['missing_confidence'].append({
                        'q_id': q_id,
                        'diagnosis': diagnosis_title
                    })
                    print(f"   ❌ Q{q_id}: Missing confidence threshold")
                else:
                    confidence = diagnosis_info.get('confidence_threshold')
                    print(f"   ✅ Q{q_id}: {confidence}% confidence")
                
                # Check reasoning
                if not diagnosis_info.get('reasoning'):
                    validation_results['missing_reasoning'].append({
                        'q_id': q_id,
                        'diagnosis': diagnosis_title
                    })
                    print(f"   ❌ Q{q_id}: Missing clinical reasoning")
                else:
                    reasoning = diagnosis_info.get('reasoning', '')[:100]
                    print(f"   ✅ Q{q_id}: Reasoning provided - {reasoning}...")

                if not diagnosis_info.get('icd_code'):
                    validation_results['missing_icd_code'].append({  # Direct append
                        'q_id': q_id,
                        'diagnosis': diagnosis_title
                    })
                    print(f"   ❌ Q{q_id}: Missing ICD-10 code")
                else:
                    icd_code = diagnosis_info.get('icd_code')
                    print(f"   ✅ Q{q_id}: ICD code provided - {icd_code}")
        
        # Calculate total issues
        validation_results['total_issues'] = (
            len(validation_results['sequential_violations']) +
            len(validation_results['invalid_q_id_references']) +
            len(validation_results['short_path_violations']) +
            len(validation_results['missing_confidence']) +
            len(validation_results['missing_reasoning']) +
            len(validation_results.get('missing_icd_code', []))  # ADD THIS LINE
        )
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   ✅ Proper branching: {validation_results['proper_branching']} questions")
        print(f"   ❌ Sequential violations: {len(validation_results['sequential_violations'])}")
        print(f"   ❌ Invalid Q_ID references: {len(validation_results['invalid_q_id_references'])}")
        print(f"   ❌ Short path violations: {len(validation_results['short_path_violations'])}")
        print(f"   ❌ Missing confidence: {len(validation_results['missing_confidence'])}")
        print(f"   ❌ Missing reasoning: {len(validation_results['missing_reasoning'])}")
        print(f"   🎯 Total issues: {validation_results['total_issues']}")
        print(f"   ❌ Missing ICD codes: {len(validation_results.get('missing_icd_code', []))}")
        
        return validation_results

    def generate_decision_trees(self, all_batches, output_folder="enhanced_medical_decision_tree"):
        """Generate decision trees with comprehensive validation"""
        # Create output folder at the very beginning
        current_dir = os.getcwd()
        output_path = os.path.join(current_dir, output_folder)
        os.makedirs(output_path, exist_ok=True)
        
        print(f"📁 Created output folder: {output_path}")
        logger.info(f"📁 Created output folder: {output_path}")
        
        categorized_trees = {}
        saved_files = {}
        total_nodes = 0
        
        for i, batch in enumerate(all_batches, 1):
            target_age = batch['target_age']
            target_gender = batch['target_gender']
            category_key = batch['category_key']
            filename = batch['filename']
            
            logger.info(f"⌛ [{i:2d}/{len(all_batches)}] Generating ENHANCED Decision Tree for {category_key}...")
            logger.info(f"   📊 Input Data: {len(batch['questions'])} medical questions")
            logger.info(f"   🚨 Emergency priorities: {len(batch['priorities']['emergency'])}")
            logger.info(f"   ⚖️ Severity priorities: {len(batch['priorities']['severity'])}") 
            logger.info(f"   🔗 Associated symptoms: {len(batch['priorities']['associated'])}")
            logger.info(f"   ⚡ ENFORCING UNIVERSAL clinical triage with confidence + reasoning")
            # logger.info(f"   🎯 Missing ICD codes: {len(validation_results.get('missing_icd_code', []))}")
            
            try:
                # Generate the decision tree with ENHANCED requirements
                dt_response = self.decision_tree_chain.invoke({"qa_json": batch})
                db_data = dt_response.model_dump()
                
                # COMPREHENSIVE VALIDATION
                decision_tree_nodes = db_data['decision_tree']
                validation_results = self.validate_decision_tree_comprehensive(
                    decision_tree_nodes, 
                    batch['valid_q_ids']
                )
                
                logger.info(f"   ✅ ENHANCED Decision Tree Completed for {category_key}")
                logger.info(f"   📈 Generated {len(decision_tree_nodes)} nodes")
                logger.info(f"   🔀 Proper branching: {validation_results['proper_branching']} questions")
                logger.info(f"   ❌ Total issues: {validation_results['total_issues']}")
                logger.info(f"   🎯 Sequential violations: {len(validation_results['sequential_violations'])}")
                logger.info(f"   🎯 Invalid Q_ID refs: {len(validation_results['invalid_q_id_references'])}")
                logger.info(f"   🎯 Short paths: {len(validation_results['short_path_violations'])}")
                logger.info(f"   🎯 Missing confidence: {len(validation_results['missing_confidence'])}")
                logger.info(f"   🎯 Missing reasoning: {len(validation_results['missing_reasoning'])}")
                
                # Store category data
                diagnosis_count = len([node for node in decision_tree_nodes if node.get('q_tag') == 'DIAGNOSIS'])
                question_count = len([node for node in decision_tree_nodes if node.get('q_tag') == 'NEW'])
                
                category_data = {
                    'age_group': target_age,
                    'gender': target_gender,
                    'decision_tree': decision_tree_nodes,
                    'validation_results': validation_results,
                    'metadata': {
                        'total_nodes': len(decision_tree_nodes),
                        'question_nodes': question_count,
                        'diagnosis_nodes': diagnosis_count,
                        'medical_questions_used': len(batch['questions']),
                        'generation_method': 'AI_ENHANCED_CONFIDENCE_REASONING',
                        'valid_q_ids': batch['valid_q_ids']
                    }
                }
                
                categorized_trees[category_key] = category_data
                total_nodes += len(decision_tree_nodes)
                
                # IMMEDIATELY save this category to file (ONLY decision_tree array)
                json_filepath = os.path.join(output_path, filename)
                
                print(f"\n💾 SAVING ENHANCED DECISION_TREE: {filename}")
                logger.info(f"💾 Saving ENHANCED decision_tree array: {filename}")
                
                # Save ONLY the decision_tree array as requested (no metadata, no wrapper)
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(decision_tree_nodes, f, indent=2, ensure_ascii=False)
                
                # Verify file was created and show file size
                if os.path.exists(json_filepath):
                    file_size = os.path.getsize(json_filepath)
                    print(f"✅ SAVED: {filename} ({file_size:,} bytes)")
                    print(f"📂 Location: {json_filepath}")
                    print(f"📋 Content: ENHANCED decision_tree array ({len(decision_tree_nodes)} nodes)")
                    print(f"🎯 Validation: {validation_results['total_issues']} issues found")
                    print(f"🔍 You can now open and verify this enhanced JSON file!")
                    logger.info(f"✅ File saved: {filename} ({file_size:,} bytes)")
                    logger.info(f"📋 Saved {question_count} questions + {diagnosis_count} diagnoses")
                else:
                    print(f"❌ FAILED to save: {filename}")
                    logger.error(f"❌ Failed to save: {filename}")
                
                # Track saved files
                saved_files[category_key] = {
                    'json_file': json_filepath,
                    'filename': filename,
                    'nodes_count': len(decision_tree_nodes),
                    'question_nodes': question_count,
                    'diagnosis_nodes': diagnosis_count,
                    'file_size': file_size if os.path.exists(json_filepath) else 0,
                    'validation_issues': validation_results['total_issues']
                }
                
                print(f"📊 Progress: {i}/{len(all_batches)} categories completed")
                print("-" * 70)
                
            except Exception as e:
                logger.error(f"   ❌ Error generating decision tree for {category_key}: {e}")
                print(f"❌ Error generating {category_key}: {e}")
                continue
        
        logger.info(f"🎯 Total decision tree nodes generated: {total_nodes}")
        logger.info(f"📂 Categories created: {len(categorized_trees)}")
        
        return categorized_trees, output_path, saved_files

    def run_complete_pipeline(self, json_file_path="cough_latest_complete.json", output_folder="enhanced_medical_decision_tree"):
        """Run the complete ENHANCED pipeline with all latest implementations"""
        logger.info("🚀 Starting ENHANCED Medical Decision Tree Generation Pipeline")
        logger.info("📋 Enhancement 1: Confidence threshold + clinical reasoning for all diagnoses")
        logger.info("📄 Enhancement 2: Comprehensive validation (Q_ID refs, path lengths, reasoning)")
        logger.info("⚡ Enhancement 3: Universal clinical logic (works for any medical condition)")
        logger.info("🎯 Enhancement 4: Strict 3-4 question minimum with clinical branching")
        logger.info("💾 Enhancement 5: Enhanced JSON structure with proper validation")
        logger.info("🧠 Enhancement 6: AI clinical reasoning for every diagnosis")
        logger.info("🔧 Enhancement 7: Zero tolerance for sequential logic violations")
        
        try:
            # Step 1: Load medical data
            qna_data = self.load_medical_data(json_file_path)
            
            # Step 2: Create optimized demographic batches (medical questions only)
            all_batches = self.create_demographic_batches(qna_data)
            
            # Step 3: Generate ENHANCED decision trees (saves each category immediately)
            categorized_trees, output_path, saved_files = self.generate_decision_trees(all_batches, output_folder)
            
            # Calculate totals for summary
            total_nodes = sum(data['metadata']['total_nodes'] for data in categorized_trees.values())
            total_questions = sum(data['metadata']['question_nodes'] for data in categorized_trees.values())
            total_diagnoses = sum(data['metadata']['diagnosis_nodes'] for data in categorized_trees.values())
            total_issues = sum(data['validation_results']['total_issues'] for data in categorized_trees.values())
            
            logger.info("🎉 ENHANCED Medical Decision Tree Generation Complete!")
            logger.info(f"📊 Total categories: {len(categorized_trees)}")
            logger.info(f"📈 Total nodes: {total_nodes}")
            logger.info(f"❓ Question nodes: {total_questions}")
            logger.info(f"🏥 Diagnosis nodes: {total_diagnoses}")
            logger.info(f"🎯 Total validation issues: {total_issues}")
            logger.info(f"📁 Output folder: {output_path}")
            logger.info("✨ All files contain ENHANCED decision_tree arrays with confidence + reasoning")
            logger.info("⚡ All diagnostic paths require 3-4+ questions minimum")
            logger.info("🔧 All question mappings validated comprehensively")
            logger.info("📋 Universal clinical logic for any medical condition")
            
            return categorized_trees, output_path, saved_files
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        print("🚀 Starting ENHANCED Medical Decision Tree Generation...")
        print("📋 ENHANCEMENT 1: Confidence threshold + clinical reasoning")
        print("📄 ENHANCEMENT 2: Comprehensive validation system")
        print("⚡ ENHANCEMENT 3: Universal clinical logic (any medical condition)")
        print("🎯 ENHANCEMENT 4: Strict 3-4 question minimum enforcement")
        print("💾 ENHANCEMENT 5: Enhanced JSON structure validation")
        print("🧠 ENHANCEMENT 6: AI clinical reasoning for every diagnosis")
        print("🔧 ENHANCEMENT 7: Zero tolerance sequential logic detection")
        
        # Initialize the ENHANCED generator
        generator = MedicalDecisionTreeGenerator()
        
        # Run the complete ENHANCED pipeline
        categorized_results, output_folder, saved_files = generator.run_complete_pipeline(
            json_file_path="cough_latest_complete 2.json",
            output_folder="enhanced_medical_decision_tree"
        )
        
        # Calculate totals
        total_nodes = sum(data['metadata']['total_nodes'] for data in categorized_results.values())
        total_questions = sum(data['metadata']['question_nodes'] for data in categorized_results.values())
        total_diagnoses = sum(data['metadata']['diagnosis_nodes'] for data in categorized_results.values())
        total_issues = sum(data['validation_results']['total_issues'] for data in categorized_results.values())
        
        print(f"\n🎯 SUCCESS: ENHANCED Medical Decision Tree Generated!")
        print(f"📂 Output Folder: {output_folder}")
        print(f"📈 Total Categories: {len(categorized_results)}")
        print(f"📊 Total Nodes: {total_nodes}")
        print(f"❓ Question Nodes: {total_questions}")
        print(f"🏥 Diagnosis Nodes: {total_diagnoses}")
        print(f"🎯 Total Validation Issues: {total_issues}")
        print(f"📋 ENHANCED features: Confidence + Reasoning + Validation")
        print(f"📄 CLEAN JSON: ONLY decision_tree arrays saved")
        print(f"⚡ UNIVERSAL: Works for any medical condition")
        print(f"💾 IMMEDIATE save: Each category file saved after generation")
        print(f"🎨 INTELLIGENT: AI creates diagnoses with clinical reasoning")
        print(f"🔧 VALIDATED: All mappings and paths comprehensively checked")
        
        print(f"\n📁 Enhanced File Structure:")
        print(f"   enhanced_medical_decision_tree/")
        for category_key, file_info in saved_files.items():
            filename = file_info['filename']
            q_nodes = file_info['question_nodes']
            d_nodes = file_info['diagnosis_nodes']
            issues = file_info['validation_issues']
            print(f"   ├── {filename} ✅ ({q_nodes}Q + {d_nodes}D, {issues} issues)")
        
        print(f"\n📋 Enhanced Validation Results:")
        for category_key, file_info in saved_files.items():
            age_group = categorized_results[category_key]['age_group'] 
            gender = categorized_results[category_key]['gender']
            filename = file_info['filename']
            issues = file_info['validation_issues']
            
            validation_data = categorized_results[category_key]['validation_results']
            sequential = len(validation_data['sequential_violations'])
            invalid_refs = len(validation_data['invalid_q_id_references'])
            short_paths = len(validation_data['short_path_violations'])
            missing_conf = len(validation_data['missing_confidence'])
            missing_reason = len(validation_data['missing_reasoning'])
            
            gender_display = f"({gender})" if gender != 'Both' else "(Both genders)"
            print(f"   • {age_group} {gender_display}: {filename}")
            if issues == 0:
                print(f"     └─ ✅ PERFECT: 0 validation issues")
            else:
                print(f"     └─ ⚠️  Issues: {sequential} sequential, {invalid_refs} invalid refs, {short_paths} short paths, {missing_conf} missing confidence, {missing_reason} missing reasoning")
        
        print(f"\n🔍 ENHANCED VERIFICATION TIPS:")
        print(f"   ✅ Open any file - clean decision_tree array with confidence + reasoning")
        print(f"   ✅ Check diagnosis nodes for confidence_threshold (80-100%)")
        print(f"   ✅ Check diagnosis nodes for clinical reasoning")
        print(f"   ✅ Verify each diagnostic path has 3-4+ questions minimum")
        print(f"   ✅ Check that all next_q_id values point to existing questions")
        print(f"   ✅ Universal logic works for any medical condition")
        print(f"   ✅ Files are located at: {output_folder}")
        
        # Display sample enhanced structure from first category
        if categorized_results:
            first_category = list(categorized_results.keys())[0]
            sample_tree = categorized_results[first_category]['decision_tree'][:2]
            print(f"\n📋 Sample ENHANCED Decision Tree Structure ({first_category}):")
            print("   Structure: Clean decision_tree array with confidence + reasoning")
            print(json.dumps(sample_tree, indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()