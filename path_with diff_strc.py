import json
import os
import itertools
from typing import Dict, List, Any, Tuple

class SimpleDecisionTreeGenerator:
    def __init__(self):
        self.decision_tree = []
        self.current_q_id = 1
        self.current_id = 1  # New sequential ID starting from 1
        self.q_id_to_id_map = {}  # Map q_id to id for childQuestion
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

    def create_simple_decision_tree(self, combined_questions: Dict):
        """Create simple linear decision tree"""
        
        # Step 1: Create Gender Question (ID: 1)
        gender_question = {
            "id": self.current_id,
            "q_id": 1,
            "q_tag": "GENERAL",
            "question": "What is your gender?",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "Male",
                    "next_q_id": 2,
                    "nextQuestion": 2,
                    "childQuestion": None,  # Will be updated after all questions created
                    "childDiagnosis": None
                },
                {
                    "option_id": 2,
                    "opt_value": "Female", 
                    "next_q_id": 3,
                    "nextQuestion": 3,
                    "childQuestion": None,  # Will be updated after all questions created
                    "childDiagnosis": None
                }
            ]
        }
        self.decision_tree.append(gender_question)
        self.q_id_to_id_map[1] = self.current_id
        self.current_id += 1
        
        # Step 2: Create placeholder age questions (will be updated with options later)
        # Female age question (ID: 2)
        female_age_question = {
            "id": self.current_id,
            "q_id": 2,
            "q_tag": "NEW",
            "question": "What is your age group?",
            "options": []  # Will be populated later
        }
        self.decision_tree.append(female_age_question)
        self.q_id_to_id_map[2] = self.current_id
        female_age_id = self.current_id
        self.current_id += 1
        
        # Male age question (ID: 3)
        male_age_question = {
            "id": self.current_id,
            "q_id": 3,
            "q_tag": "NEW", 
            "question": "What is your age group?",
            "options": []  # Will be populated later
        }
        self.decision_tree.append(male_age_question)
        self.q_id_to_id_map[3] = self.current_id
        male_age_id = self.current_id
        self.current_id += 1
        
        # Step 3: Set starting q_id for symptom questions
        self.current_q_id = 4
        
        # Step 4: Create symptom chains and populate age question options
        female_age_options = []
        male_age_options = []
        
        for i, age_group in enumerate(self.age_groups):
            if age_group in combined_questions:
                # Female age option
                if "Female" in combined_questions[age_group]:
                    start_q_id = self.current_q_id
                    self._create_linear_question_chain(
                        combined_questions[age_group]["Female"], 
                        age_group, 
                        "Female"
                    )
                    female_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id,
                        "nextQuestion": start_q_id,
                        "childQuestion": None,  # Will be updated after all questions created
                        "childDiagnosis": None
                    })
                
                # Male age option  
                if "Male" in combined_questions[age_group]:
                    start_q_id = self.current_q_id
                    self._create_linear_question_chain(
                        combined_questions[age_group]["Male"], 
                        age_group, 
                        "Male"
                    )
                    male_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id,
                        "nextQuestion": start_q_id,
                        "childQuestion": None,  # Will be updated after all questions created
                        "childDiagnosis": None
                    })
        
        # Step 5: Update age questions with their options
        # Find and update female age question
        for node in self.decision_tree:
            if node["id"] == female_age_id:
                node["options"] = female_age_options
                break
        
        # Find and update male age question  
        for node in self.decision_tree:
            if node["id"] == male_age_id:
                node["options"] = male_age_options
                break
        
        # Step 6: Update all childQuestion references
        self._update_child_question_references()

    def _create_linear_question_chain(self, questions_list: List[Dict], age_group: str, gender: str):
        """Create a simple linear chain of questions ending in ONE diagnosis"""
        
        if not questions_list:
            return
            
        print(f"\n🔗 Creating linear chain for {age_group} {gender}: {len(questions_list)} questions")
        
        # Calculate total possible paths
        total_paths = 1
        for question in questions_list:
            options_count = len(question.get("options", []))
            if options_count > 0:
                total_paths *= options_count
        
        print(f"   📊 Total paths: {total_paths:,}")
        
        # Generate all possible answer combinations
        all_paths = self._generate_all_answer_combinations(questions_list)
        
        if not all_paths:
            print(f"   ⚠️ No valid paths for {age_group} {gender}")
            return
            
        print(f"   ✅ Generated {len(all_paths):,} answer combinations")
        
        # Create linear question chain
        for i, question in enumerate(questions_list):
            current_q_id = self.current_q_id
            self.current_q_id += 1
            
            # Create question node
            question_node = {
                "id": self.current_id,
                "q_id": current_q_id,
                "q_tag": question.get("q_tag", "GENERAL"),
                "question": question["question"],
                "options": []
            }
            self.q_id_to_id_map[current_q_id] = self.current_id
            self.current_id += 1
            
            # Add options
            for option in question.get("options", []):
                if i == len(questions_list) - 1:
                    # Last question - all options point to SAME diagnosis
                    next_q_id = self.diagnosis_counter
                    option_data = {
                        "option_id": option["option_id"],
                        "opt_value": option["opt_value"],
                        "next_q_id": next_q_id,
                        "nextQuestion": None,
                        "childQuestion": None,
                        "childDiagnosis": next_q_id
                    }
                else:
                    # Point to next question in chain
                    next_q_id = self.current_q_id
                    option_data = {
                        "option_id": option["option_id"],
                        "opt_value": option["opt_value"],
                        "next_q_id": next_q_id,
                        "nextQuestion": next_q_id,
                        "childQuestion": None,  # Will be updated after all questions created
                        "childDiagnosis": None
                    }
                
                question_node["options"].append(option_data)
            
            self.decision_tree.append(question_node)
        
        # Create ONE diagnosis for ALL paths in this demographic
        self._create_single_diagnosis(all_paths, questions_list, age_group, gender)

    def _update_child_question_references(self):
        """Update all childQuestion references to point to the correct id values"""
        print("\n🔄 Updating childQuestion references...")
        
        for node in self.decision_tree:
            if "options" in node:
                for option in node["options"]:
                    if option.get("nextQuestion") and option["childQuestion"] is None:
                        next_q_id = option["nextQuestion"]
                        if next_q_id in self.q_id_to_id_map:
                            option["childQuestion"] = self.q_id_to_id_map[next_q_id]
                            print(f"   ✅ Updated q_id {next_q_id} -> id {option['childQuestion']}")
        
        print("✅ All childQuestion references updated!")

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

    def _create_single_diagnosis(self, all_paths: List[Tuple], questions_list: List[Dict], 
                                age_group: str, gender: str):
        """Create ONE diagnosis that represents all possible paths for this demographic"""
        
        diagnosis_node = {
            "id": self.current_id,
            "q_id": self.diagnosis_counter,
            "q_tag": "DIAGNOSIS",
            "diagnosis": {
                "diagnosis_title": f"Health Assessment for {age_group} {gender}",
                "description": f"Comprehensive health assessment for {age_group} {gender} based on {len(questions_list)} questions and {len(all_paths):,} possible answer combinations."
            },
            "red_flags": self._generate_red_flags_for_demographic(age_group, gender),
            "otc_medication": self._generate_otc_for_demographic(age_group, gender),
            "advice": self._generate_advice_for_demographic(age_group, gender),
            "precautions": self._generate_precautions_for_demographic(age_group, gender),
            "lab_tests": self._generate_lab_tests_for_demographic(age_group, gender),
            "diet": self._generate_diet_for_demographic(age_group, gender)
        }
        
        self.decision_tree.append(diagnosis_node)
        self.q_id_to_id_map[self.diagnosis_counter] = self.current_id
        self.current_id += 1
        self.diagnosis_counter += 1
        
        print(f"   🎯 Created diagnosis {self.diagnosis_counter - 100000001} for {age_group} {gender} ({len(all_paths):,} paths)")

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

    def generate_simple_tree(self, questions_json_path: str) -> bool:
        """Main function to generate simple decision tree"""
        print("🚀 Starting SIMPLE Decision Tree Generation")
        print("   📝 Linear question chains for each demographic")
        print("   🎯 Each demographic gets ONE diagnosis for ALL paths")
        print("   ✅ Simple structure: Gender → Age → Questions → Single Diagnosis")
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
        
        # Create simple decision tree
        self.create_simple_decision_tree(combined_questions)
        
        print("=" * 60)
        print("✅ SIMPLE Decision Tree Generation Finished!")
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
        
        print(f"\n💾 Simple decision tree saved to: {output_path}")
        print(f"\n📈 FINAL SUMMARY:")
        print(f"   • Total Nodes: {len(self.decision_tree):,}")
        print(f"   • Question Nodes: {len(question_nodes):,}")
        print(f"   • Diagnosis Nodes: {len(diagnosis_nodes):,}")
        print(f"   ✅ Structure: Linear chains with single diagnosis per demographic")
        print(f"   ✅ Each demographic has ONE diagnosis covering all possible paths")

def main():
    """Generate simple linear decision tree"""
    print("🎯 SIMPLE Linear Decision Tree Generator")
    print("   📝 Each demographic gets linear question chain")
    print("   🎯 ALL answer combinations → ONE diagnosis per demographic")
    print("   ✅ Example: 3 questions with 2 options each = 8 paths → 1 diagnosis")
    print("=" * 70)
    
    generator = SimpleDecisionTreeGenerator()
    
    # Update these paths
    input_json_path = "back_pain_questions.json"
    output_json_path = "./simple_decision_tree.json"
    
    success = generator.generate_simple_tree(input_json_path)
    
    if success:
        generator.save_decision_tree(output_json_path)
        print(f"\n🎉 SUCCESS! Simple decision tree generated!")
        print(f"📁 Input: {input_json_path}")
        print(f"📁 Output: {output_json_path}")
        print(f"🌟 Linear structure: All paths per demographic → Single diagnosis!")
    else:
        print(f"\n❌ FAILED! Check your input file and try again.")

if __name__ == "__main__":
    main()