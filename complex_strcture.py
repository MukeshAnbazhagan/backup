import json
import os
import itertools
from typing import Dict, List, Any, Tuple

class TrueDAGDecisionTreeGenerator:
    def __init__(self):
        self.decision_tree = []
        self.current_q_id = 1
        self.current_id = 1  # Added for id field
        self.age_groups = ["0-2", "3-12", "13-18", "19-40", "41-65", "66+"]
        self.question_id_map = {}  # To track created questions
        
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

    def create_true_dag_tree(self, combined_questions: Dict):
        """Create true DAG where each branch gets unique question IDs"""
        
        # Step 1: Create Gender Question (ID: 1)
        gender_question = {
            "id": self.current_id,
            "q_id": 1,
            "q_tag": "GENERAL",
            "question": "What is your gender?",
            "age_group": "All",
            "gender": "All",
            "options": [
                {
                    "option_id": 1,
                    "opt_value": "Male",
                    "next_q_id": 2,
                    "nextQuestion": 2,
                    "childQuestion": 2,
                    "childDiagnosis": None
                },
                {
                    "option_id": 2,
                    "opt_value": "Female",
                    "next_q_id": 0,  # Will be updated later
                    "nextQuestion": 0,
                    "childQuestion": 0,
                    "childDiagnosis": None
                }
            ]
        }
        self.decision_tree.append(gender_question)
        self.current_q_id = 2
        self.current_id += 1
        
        # Step 2: Create Male age question and all its branches
        male_age_question_id = self.current_q_id
        male_age_options = []
        
        # First create all male branches
        for i, age_group in enumerate(self.age_groups):
            if age_group in combined_questions and "Male" in combined_questions[age_group]:
                start_q_id = self._create_complete_branch_structure(
                    combined_questions[age_group]["Male"], 
                    age_group, 
                    "Male"
                )
                if start_q_id:
                    male_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id,
                        "nextQuestion": start_q_id,
                        "childQuestion": start_q_id,
                        "childDiagnosis": None
                    })
        
        # Create male age question
        male_age_question = {
            "id": self.current_id,
            "q_id": male_age_question_id,
            "q_tag": "NEW",
            "question": "What is your age group?",
            "age_group": "All",
            "gender": "Male",
            "options": male_age_options
        }
        self.decision_tree.append(male_age_question)
        self.current_id += 1
        
        # Step 3: Now create female age question ID and update gender question
        female_age_question_id = self.current_q_id
        self.current_q_id += 1
        
        # Update gender question with correct female age question ID
        gender_question["options"][1]["next_q_id"] = female_age_question_id
        gender_question["options"][1]["nextQuestion"] = female_age_question_id
        gender_question["options"][1]["childQuestion"] = female_age_question_id
        
        # Step 4: Create all female branches
        female_age_options = []
        for i, age_group in enumerate(self.age_groups):
            if age_group in combined_questions and "Female" in combined_questions[age_group]:
                start_q_id = self._create_complete_branch_structure(
                    combined_questions[age_group]["Female"], 
                    age_group, 
                    "Female"
                )
                if start_q_id:
                    female_age_options.append({
                        "option_id": i + 1,
                        "opt_value": age_group,
                        "next_q_id": start_q_id,
                        "nextQuestion": start_q_id,
                        "childQuestion": start_q_id,
                        "childDiagnosis": None
                    })
        
        # Create female age question
        female_age_question = {
            "id": self.current_id,
            "q_id": female_age_question_id,
            "q_tag": "NEW",
            "question": "What is your age group?",
            "age_group": "All",
            "gender": "Female",
            "options": female_age_options
        }
        self.decision_tree.append(female_age_question)
        self.current_id += 1

    def _create_complete_branch_structure(self, questions_list: List[Dict], age_group: str, gender: str):
        """Create complete branch structure with all questions"""
        if not questions_list:
            return None
            
        print(f"\n🌳 Creating branch for {age_group} {gender}: {len(questions_list)} questions")
        
        # Generate all possible paths through the questions
        all_paths = self._generate_all_answer_combinations(questions_list)
        
        if not all_paths:
            return None
            
        # Create the initial branching question
        first_question_id = self._create_branching_questions(
            questions_list, 
            all_paths, 
            0,  # Start at first question
            age_group, 
            gender
        )
        
        return first_question_id

    def _create_branching_questions(self, questions_list: List[Dict], all_paths: List[Tuple], 
                                   question_index: int, age_group: str, gender: str) -> int:
        """Recursively create branching questions"""
        
        if question_index >= len(questions_list):
            # We've reached the end - return None for final node
            return None
        
        # Create current question node
        current_question = questions_list[question_index]
        question_id = self.current_q_id
        self.current_q_id += 1
        
        question_node = {
            "id": self.current_id,
            "q_id": question_id,
            "q_tag": current_question.get("q_tag", "NEW"),
            "question": current_question["question"],
            "age_group": age_group,
            "gender": gender,
            "options": []
        }
        self.current_id += 1
        
        # Group paths by their answer at this question index
        paths_by_answer = {}
        for path in all_paths:
            if len(path) > question_index:
                answer = path[question_index]
                if answer not in paths_by_answer:
                    paths_by_answer[answer] = []
                paths_by_answer[answer].append(path)
        
        # Create options for each possible answer
        for option in current_question.get("options", []):
            option_value = option["opt_value"]
            
            if option_value in paths_by_answer:
                # Filter paths that have this answer
                filtered_paths = paths_by_answer[option_value]
                
                if question_index == len(questions_list) - 1:
                    # Last question - point to null (end of path)
                    next_q_id = None
                else:
                    # More questions - recurse
                    next_q_id = self._create_branching_questions(
                        questions_list,
                        filtered_paths,
                        question_index + 1,
                        age_group,
                        gender
                    )
                
                question_node["options"].append({
                    "option_id": option["option_id"],
                    "opt_value": option_value,
                    "next_q_id": next_q_id,
                    "nextQuestion": next_q_id,
                    "childQuestion": next_q_id,
                    "childDiagnosis": None
                })
        
        # Add question to decision tree
        self.decision_tree.append(question_node)
        return question_id

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

    def generate_true_dag_tree(self, questions_json_path: str) -> bool:
        """Main function to generate true DAG decision tree"""
        print("🚀 Starting TRUE DAG Decision Tree Generation")
        print("   🌳 Each option branch gets completely unique question IDs")
        print("   🎯 Same question content, different IDs per branch")
        print("   ✅ Creates branching structure with null endpoints")
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
        
        # Create true DAG decision tree
        self.create_true_dag_tree(combined_questions)
        
        print("=" * 60)
        print("✅ TRUE DAG Decision Tree Generation Finished!")
        print(f"🎯 Total nodes created: {len(self.decision_tree)}")
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
        question_nodes = [n for n in self.decision_tree]
        
        print(f"\n💾 TRUE DAG decision tree saved to: {output_path}")
        print(f"\n📈 FINAL SUMMARY:")
        print(f"   • Total Nodes: {len(self.decision_tree):,}")
        print(f"   • Question Nodes: {len(question_nodes):,}")
        print(f"   ✅ Structure: TRUE DAG - each branch gets unique question IDs")
        print(f"   ✅ Final nodes point to null instead of diagnosis")
        print(f"   ✅ Age and gender added to all nodes")

def main():
    """Generate true DAG decision tree"""
    print("🌳 TRUE DAG Decision Tree Generator")
    print("   🎯 Each branch gets unique question IDs")
    print("   📝 Final nodes point to null (no diagnosis)")
    print("   ✅ Age and gender fields added to all nodes")
    print("=" * 70)
    
    generator = TrueDAGDecisionTreeGenerator()
    
    # Update these paths
    input_json_path = "back_pain_questions.json"
    output_json_path = "./true_dag_decision_tree.json"
    
    success = generator.generate_true_dag_tree(input_json_path)
    
    if success:
        generator.save_decision_tree(output_json_path)
        print(f"\n🎉 SUCCESS! TRUE DAG decision tree generated!")
        print(f"📁 Input: {input_json_path}")
        print(f"📁 Output: {output_json_path}")
        print(f"🌟 Structure: Final nodes → null, with age/gender fields!")
    else:
        print(f"\n❌ FAILED! Check your input file and try again.")

if __name__ == "__main__":
    main()