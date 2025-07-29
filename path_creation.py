import json
import os
import itertools
from typing import Dict, List, Any, Tuple

class DAGDecisionTreeGenerator:
    def __init__(self):
        self.decision_tree = []
        self.current_q_id = 1
        self.age_groups = ["0-2", "3-12", "13-18", "19-40", "41-65", "66+"]
        self.diagnosis_counter = 100000001
        
    def load_questions_json(self, file_path: str) -> Dict:
        """Load questions from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"✅ Successfully loaded questions from: {file_path}")
            return data
        except FileNotFoundError:
            print(f"❌ Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in {file_path}")
            return None

    def organize_questions_by_demographics(self, questions_data: List[Dict]) -> Dict:
        """Organize and combine questions by demographics"""
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
        
        # COMBINE questions properly
        combined_questions = {}
        
        print("\n🔗 COMBINING Questions by Demographics:")
        print("=" * 50)
        
        for age_group in self.age_groups:
            if age_group not in questions_by_category:
                continue
                
            combined_questions[age_group] = {}
            
            both_questions = questions_by_category[age_group].get("Both", [])
            male_questions = questions_by_category[age_group].get("Male", [])
            female_questions = questions_by_category[age_group].get("Female", [])
            
            print(f"\n🎯 Age Group: {age_group}")
            print(f"   📝 Both: {len(both_questions)}, Male: {len(male_questions)}, Female: {len(female_questions)}")
            
            # For children: use same "Both" questions for both genders
            if age_group in ["0-2", "3-12"]:
                if both_questions:
                    combined_questions[age_group]["Male"] = both_questions.copy()
                    combined_questions[age_group]["Female"] = both_questions.copy()
            else:
                # For adults: combine Both + Gender-specific
                if both_questions or male_questions:
                    combined_questions[age_group]["Male"] = both_questions + male_questions
                if both_questions or female_questions:
                    combined_questions[age_group]["Female"] = both_questions + female_questions
        
        return combined_questions

    def create_dag_decision_tree(self, combined_questions: Dict):
        """Create DAG decision tree with unique paths"""
        
        # Step 1: Create Gender Question (ID: 1)
        gender_question = {
            "q_id": 1,
            "q_tag": "NEW",
            "question": "What is your gender?",
            "options": [
                {"option_id": 1, "opt_value": "Female", "next_q_id": 2},
                {"option_id": 2, "opt_value": "Male", "next_q_id": 3}
            ]
        }
        self.decision_tree.append(gender_question)
        self.current_q_id = 4  # Reserve IDs 2,3 for age questions
        
        # Step 2: Create Age Questions
        female_age_options = []
        male_age_options = []
        
        for i, age_group in enumerate(self.age_groups):
            if age_group in combined_questions:
                # Female age option
                if "Female" in combined_questions[age_group]:
                    start_q_id = self.current_q_id
                    self._create_dag_structure(
                        combined_questions[age_group]["Female"], 
                        age_group, 
                        "Female"
                    )
                    female_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id
                    })
                
                # Male age option  
                if "Male" in combined_questions[age_group]:
                    start_q_id = self.current_q_id
                    self._create_dag_structure(
                        combined_questions[age_group]["Male"], 
                        age_group, 
                        "Male"
                    )
                    male_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id
                    })
        
        # Female age question (ID: 2)
        female_age_question = {
            "q_id": 2,
            "q_tag": "NEW",
            "question": "What is your age group?",
            "options": female_age_options
        }
        self.decision_tree.append(female_age_question)
        
        # Male age question (ID: 3)
        male_age_question = {
            "q_id": 3,
            "q_tag": "NEW",
            "question": "What is your age group?",
            "options": male_age_options
        }
        self.decision_tree.append(male_age_question)

    def _create_dag_structure(self, questions_list: List[Dict], age_group: str, gender: str):
        """Create DAG structure where each path gets unique question IDs"""
        
        if not questions_list:
            return
            
        print(f"\n🌳 Creating DAG for {age_group} {gender}: {len(questions_list)} questions")
        
        # Calculate total possible paths
        total_paths = 1
        for question in questions_list:
            options_count = len(question.get("options", []))
            if options_count > 0:
                total_paths *= options_count
        
        print(f"   📊 Total unique paths: {total_paths:,}")
        
        # Generate all possible answer combinations
        all_paths = self._generate_all_answer_combinations(questions_list)
        
        if not all_paths:
            print(f"   ⚠️ No valid paths for {age_group} {gender}")
            return
            
        print(f"   ✅ Generated {len(all_paths):,} answer combinations")
        
        # Create tree structure recursively - each path gets unique question IDs
        root_q_id = self.current_q_id
        self._build_tree_recursively(questions_list, 0, [], all_paths, age_group, gender)

    def _build_tree_recursively(self, questions_list: List[Dict], question_index: int, 
                               current_path: List[str], all_paths: List[Tuple], 
                               age_group: str, gender: str):
        """Recursively build the tree structure"""
        
        if question_index >= len(questions_list):
            # We've reached a leaf node - create diagnosis
            self._create_diagnosis_for_path(current_path, all_paths, age_group, gender)
            return
        
        current_question = questions_list[question_index]
        question_options = current_question.get("options", [])
        
        if not question_options:
            return
        
        # Group paths by their answer at this level
        paths_by_answer = {}
        for path in all_paths:
            if question_index < len(path):
                answer = path[question_index]
                if answer not in paths_by_answer:
                    paths_by_answer[answer] = []
                paths_by_answer[answer].append(path)
        
        # Create question node for this level
        question_node = {
            "q_id": self.current_q_id,
            "q_tag": current_question.get("q_tag", "NEW"),
            "question": current_question["question"],
            "options": []
        }
        
        current_question_id = self.current_q_id
        self.current_q_id += 1
        
        # Create options and recursive calls
        for option in question_options:
            option_value = option["opt_value"]
            
            if option_value in paths_by_answer:
                # Determine next question ID
                if question_index == len(questions_list) - 1:
                    # Last question - point to diagnosis
                    next_q_id = self.diagnosis_counter
                    # Create diagnosis for this specific path
                    paths_for_this_option = paths_by_answer[option_value]
                    for path in paths_for_this_option:
                        self._create_diagnosis_for_single_path(path, age_group, gender)
                else:
                    # Point to next question level
                    next_q_id = self.current_q_id
                    # Recursively build the next level
                    new_path = current_path + [option_value]
                    self._build_tree_recursively(
                        questions_list, 
                        question_index + 1, 
                        new_path,
                        paths_by_answer[option_value], 
                        age_group, 
                        gender
                    )
                
                question_node["options"].append({
                    "option_id": option["option_id"],
                    "opt_value": option_value,
                    "next_q_id": next_q_id
                })
        
        self.decision_tree.append(question_node)

    def _generate_all_answer_combinations(self, questions_list: List[Dict]) -> List[Tuple]:
        """Generate all possible answer combinations"""
        if not questions_list:
            return []
        
        option_lists = []
        for question in questions_list:
            options = [opt["opt_value"] for opt in question.get("options", [])]
            if not options:
                return []  # Can't create paths if any question has no options
            option_lists.append(options)
        
        if option_lists:
            return list(itertools.product(*option_lists))
        return []

    def _create_diagnosis_for_single_path(self, path: Tuple, age_group: str, gender: str):
        """Create a diagnosis for a single path"""
        
        diagnosis_node = {
            "q_id": self.diagnosis_counter,
            "q_tag": "DIAGNOSIS",
            "diagnosis": {
                "diagnosis_title": f"Health Assessment for {age_group} {gender} - Path {self.diagnosis_counter - 100000000}",
                "description": f"Specific diagnosis for {age_group} {gender} with path: {' → '.join(path)}"
            },
            "red_flags": self._generate_red_flags_for_demographic(age_group, gender),
            "otc_medication": self._generate_otc_for_demographic(age_group, gender),
            "advice": self._generate_advice_for_demographic(age_group, gender),
            "precautions": self._generate_precautions_for_demographic(age_group, gender),
            "lab_tests": self._generate_lab_tests_for_demographic(age_group, gender),
            "diet": self._generate_diet_for_demographic(age_group, gender)
        }
        
        self.decision_tree.append(diagnosis_node)
        self.diagnosis_counter += 1
        
        print(f"   🎯 Created diagnosis {self.diagnosis_counter - 100000001} for path: {' → '.join(path)}")

    def _create_diagnosis_for_path(self, path: List[str], all_paths: List[Tuple], 
                                 age_group: str, gender: str):
        """Create diagnosis for a specific path - used for leaf nodes"""
        # This is called when we reach the end of the question chain
        # In the recursive structure, diagnoses are created in _create_diagnosis_for_single_path
        pass

    def _generate_red_flags_for_demographic(self, age_group: str, gender: str) -> List[str]:
        """Generate red flags specific to demographic"""
        red_flags = ["Sudden worsening of symptoms", "High fever over 101.3°F"]
        
        if age_group in ["0-2", "3-12"]:
            red_flags.extend(["Signs of dehydration", "Lethargy or unusual behavior"])
        elif age_group in ["66+"]:
            red_flags.extend(["Confusion", "Falls", "Medication interactions"])
        
        if gender == "Female" and age_group in ["13-18", "19-40", "41-65"]:
            red_flags.append("Severe pelvic pain or abnormal bleeding")
        
        return red_flags

    def _generate_otc_for_demographic(self, age_group: str, gender: str) -> List[Dict]:
        """Generate OTC medications for demographic"""
        otc = []
        
        if age_group in ["0-2", "3-12"]:
            otc.append({
                "otc_no": "1",
                "otc_title": "Pediatric Care",
                "otc_medicine_name": "Age-appropriate medications only",
                "otc_dosage_duration": "As prescribed",
                "otc_type": "Consult pediatrician",
                "otc_intake_type": "Professional guidance required",
                "otc_intake_schedules": "As directed by healthcare provider"
            })
        else:
            otc.append({
                "otc_no": "1",
                "otc_title": "General Pain Relief",
                "otc_medicine_name": "Acetaminophen or Ibuprofen",
                "otc_dosage_duration": "3-7 days",
                "otc_type": "Tablet/Capsule",
                "otc_intake_type": "Follow package instructions",
                "otc_intake_schedules": "Every 4-6 hours as needed"
            })
        
        return otc

    def _generate_advice_for_demographic(self, age_group: str, gender: str) -> List[str]:
        """Generate advice for demographic"""
        advice = ["Rest and stay hydrated", "Monitor symptoms closely"]
        
        if age_group in ["0-2", "3-12"]:
            advice.append("Maintain normal feeding schedule and ensure adequate sleep")
        elif age_group in ["66+"]:
            advice.append("Regular medication review and fall prevention")
        
        if gender == "Female" and age_group in ["19-40", "41-65"]:
            advice.append("Consider hormonal factors that may affect symptoms")
        
        return advice

    def _generate_precautions_for_demographic(self, age_group: str, gender: str) -> List[str]:
        """Generate precautions for demographic"""
        precautions = ["Practice good hygiene", "Avoid known triggers"]
        
        if age_group in ["0-2", "3-12"]:
            precautions.append("Close supervision and immediate medical attention for concerning symptoms")
        elif age_group in ["66+"]:
            precautions.append("Monitor for drug interactions and cognitive changes")
        
        return precautions

    def _generate_lab_tests_for_demographic(self, age_group: str, gender: str) -> List[str]:
        """Generate lab tests for demographic"""
        lab_tests = ["Complete Blood Count (CBC)"]
        
        if age_group in ["19-40", "41-65", "66+"]:
            lab_tests.append("Basic Metabolic Panel")
        
        if age_group in ["66+"]:
            lab_tests.append("Comprehensive health screening")
        
        return lab_tests

    def _generate_diet_for_demographic(self, age_group: str, gender: str) -> List[Dict]:
        """Generate diet recommendations for demographic"""
        diet = [
            {
                "diet_no": 1,
                "category": "General Nutrition",
                "title": "Balanced Diet",
                "description": f"Age-appropriate balanced nutrition for {age_group} {gender}",
                "duration": "Ongoing"
            }
        ]
        
        if age_group in ["0-2", "3-12"]:
            diet.append({
                "diet_no": 2,
                "category": "Pediatric Nutrition",
                "title": "Growth-supporting Foods",
                "description": "Foods that support healthy growth and development",
                "duration": "Throughout childhood"
            })
        
        return diet

    def generate_dag_tree(self, questions_json_path: str) -> bool:
        """Main function to generate DAG decision tree"""
        print("🚀 Starting DAG Decision Tree Generation")
        print("   🌳 Each unique path gets its own question IDs")
        print("   🎯 Every path combination leads to a unique diagnosis")
        print("   ✅ DAG Structure: Same questions, different IDs per path")
        print("=" * 60)
        
        # Load data
        json_data = self.load_questions_json(questions_json_path)
        if not json_data:
            return False
        
        questions_data = json_data.get("questions", [])
        if not questions_data:
            print("❌ No questions found in JSON file")
            return False
        
        print(f"📥 Loaded {len(questions_data)} questions")
        
        # Organize questions
        combined_questions = self.organize_questions_by_demographics(questions_data)
        
        # Create DAG decision tree
        self.create_dag_decision_tree(combined_questions)
        
        print("=" * 60)
        print("✅ DAG Decision Tree Generation Finished!")
        print(f"🎯 Total diagnoses created: {self.diagnosis_counter - 100000001}")
        return True

    def save_decision_tree(self, output_path: str):
        """Save the decision tree"""
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(self.decision_tree, file, indent=2, ensure_ascii=False)
            
            self._print_final_summary(output_path)
            return True
            
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            return False

    def _print_final_summary(self, output_path: str):
        """Print final summary"""
        question_nodes = [n for n in self.decision_tree if n.get("q_tag") != "DIAGNOSIS"]
        diagnosis_nodes = [n for n in self.decision_tree if n.get("q_tag") == "DIAGNOSIS"]
        
        print(f"\n💾 DAG decision tree saved to: {output_path}")
        print(f"\n📈 FINAL SUMMARY:")
        print(f"   • Total Nodes: {len(self.decision_tree):,}")
        print(f"   • Question Nodes: {len(question_nodes):,}")
        print(f"   • Diagnosis Nodes: {len(diagnosis_nodes):,}")
        print(f"   ✅ Structure: DAG with unique paths and question IDs")
        print(f"   ✅ Each unique answer combination gets its own diagnosis")

def main():
    """Generate DAG decision tree"""
    print("🌳 DAG Decision Tree Generator")
    print("   🎯 Each path gets unique question IDs (same content, different IDs)")
    print("   📝 Every answer combination → unique diagnosis")
    print("   ✅ Creates DAG structure like your JSON example")
    print("=" * 70)
    
    generator = DAGDecisionTreeGenerator()
    
    # Update these paths
    input_json_path = "back_pain_questions.json"
    output_json_path = "./dag_decision_tree.json"
    
    success = generator.generate_dag_tree(input_json_path)
    
    if success:
        generator.save_decision_tree(output_json_path)
        print(f"\n🎉 SUCCESS! DAG decision tree generated!")
        print(f"📁 Input: {input_json_path}")
        print(f"📁 Output: {output_json_path}")
        print(f"🌟 DAG structure: Each path → unique question IDs → unique diagnosis!")
    else:
        print(f"\n❌ FAILED! Check your input file and try again.")

if __name__ == "__main__":
    main()