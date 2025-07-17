import os
import json
import logging
from dotenv import load_dotenv
import re
import time

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_Project_Svc_Account-94e8590.json"

import os
import json
import logging
import re
import time
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt

# --- Configuration & Environment Setup ---
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "GCP_Project_Svc_Account-94e8590.json")

PROJECT_ID = "REPLACE_YOUR_PROJECT_ID_HERE"
LOCATION = "us-east4"
MODEL_NAME = "gemini-2.5-pro"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Vertex AI Initialization ---
vertexai.init(project=PROJECT_ID, location=LOCATION)
logger.info(f"Initialized Vertex AI for project: {PROJECT_ID}, location: {LOCATION}, model: {MODEL_NAME}")

# Validate credentials by attempting a dummy generation
try:
    _ = GenerativeModel(MODEL_NAME).generate_content("hello", tools=[], stream=False)
    logger.info("Verified Vertex AI model credentials successfully.")
except Exception as e:
    logger.error(f"Failed to verify Vertex AI setup: {e}")
    raise SystemExit("Vertex AI initialization failed. Check credentials or project setup.")

# --- Difficulty Levels ---
DIFFICULTY_LEVELS = [
    "child_simple_physics",
    "beginner_ai_basics",
    "intermediate_ml_concepts",
    "advanced_deep_learning",
    "expert_transformer_architectures",
    "researcher_cutting_edge_ai",
    "phd_level_theoretical_ai",
    "ultimate_deepmind_challenge"
]

# --- Vertex AI Service ---
class VertexAIService:
    def __init__(self, model_name=MODEL_NAME):
        self.model = GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
        logger.info(f"VertexAIService initialized with model: {model_name}")

    
    def get_true_false_question(self, difficulty: str) -> dict | None:
        prompt = self._generate_prompt(difficulty)
        retries = 3

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Generating question for '{difficulty}' (attempt {attempt})")
                response = self.model.generate_content(prompt, generation_config=self.generation_config)

                if not response or not response.candidates:
                    logger.warning("No response candidates. Retrying...")
                    continue

                raw_text = response.candidates[0].content.text.strip()

                # Try to extract JSON from code block first
                match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_text, re.DOTALL)
                json_str = match.group(1) if match else raw_text

                # Try parsing JSON
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError: {e} | Raw text: {raw_text[:120]}")
                    continue

                if not isinstance(data, dict):
                    logger.warning(f"Expected JSON object but got: {type(data).__name__}")
                    continue

                required_keys = {"statement", "is_true", "explanation"}
                if not required_keys.issubset(data):
                    logger.warning(f"Incomplete JSON keys. Got: {list(data.keys())}")
                    continue

                # Final structure check
                if not isinstance(data["statement"], str) or not isinstance(data["is_true"], bool) or not isinstance(data["explanation"], str):
                    logger.warning("Invalid data types in JSON. Retrying...")
                    continue

                return data

            except Exception as e:
                logger.exception(f"Unexpected error during question generation: {e}")

            time.sleep(1)

        logger.error("Max retries reached. Failed to generate a valid question.")
        return None
    
    def _generate_prompt(self, difficulty: str) -> str:
        prompt_map = {
            "child_simple_physics": "The statement should be about basic physical ideas for kids using robots. Avoid AI jargon.",
            "beginner_ai_basics": "Cover core AI concepts like neural networks or data, using beginner-friendly language.",
            "intermediate_ml_concepts": "Focus on ML topics like supervised learning, clustering, model evaluation.",
            "advanced_deep_learning": "Include deep learning terms like CNNs, RNNs, backpropagation, or GANs.",
            "expert_transformer_architectures": "Dive into attention mechanisms, encoders/decoders, and fine-tuning in Transformers.",
            "researcher_cutting_edge_ai": "Explore cutting-edge concepts such as foundation models, explainability, or causality.",
            "phd_level_theoretical_ai": "Go deep into theory—ethics, complexity, probabilistic modeling.",
            "ultimate_deepmind_challenge": "Challenge experts with obscure, theoretical, or experimental frontier topics."
        }
        context = prompt_map.get(difficulty, "")

        return f"""
Generate a unique and challenging True/False statement about AI.
It should be concise. Provide a detailed one-paragraph explanation for why it's true or false.
Respond strictly in JSON:

{{
  "statement": "Example statement",
  "is_true": true,
  "explanation": "Explanation here."
}}

Difficulty context: {context}
Ensure no text outside the JSON object.
"""




# --- QuizManager Class (The Game Logic) ---
class QuizManager:
    def __init__(self, difficulty_levels: list):
        self.difficulty_levels = difficulty_levels
        self.console = Console()
        self.reset_state() # Call reset_state after console is initialized
        logger.info("QuizManager initialized.")

    def reset_state(self):
        self.score = 0
        # --- FIX: Ensure current_difficulty_index is valid even if "beginner_ai_basics" is not present ---
        if "beginner_ai_basics" in self.difficulty_levels:
            self.current_difficulty_index = self.difficulty_levels.index("beginner_ai_basics")
        else:
            self.current_difficulty_index = 0 # Default to the very first level
            logger.warning("Default difficulty 'beginner_ai_basics' not found in DIFFICULTY_LEVELS, starting at the first level.")

        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.total_questions_answered = 0
        self.question_history = []
        logger.info("QuizManager state reset.")

    def record_answer(self, question: dict, user_answer: bool, is_correct: bool):
        self.total_questions_answered += 1
        if is_correct:
            self.score += 1
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0 # Reset incorrect streak
            self.console.print(Text("\n🎉 Correct! 🎉", style="bold green"))
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0 # Reset correct streak
            self.console.print(Text("\n😔 Incorrect.", style="bold red"))
        
        # --- FIX: Use .get() for explanation to be defensive ---
        explanation_text = question.get('explanation', 'No explanation provided.')
        self.console.print(Panel(Text(f"Explanation: {explanation_text}", style="gold3"), title="Explanation", border_style="gold3"))
        
        self.question_history.append({
            "statement": question.get("statement", "N/A"),
            "correct_answer": question.get("is_true", None),
            "user_answer": user_answer,
            "is_correct": is_correct,
            "explanation": explanation_text, # Use the potentially defensive-retrieved explanation
            "difficulty_at_time": self.difficulty_levels[self.current_difficulty_index]
        })
        
        self._adjust_difficulty()
        logger.info(f"Answer recorded. Score: {self.score}, Current Difficulty Index: {self.current_difficulty_index}, streak_correct: {self.consecutive_correct}, streak_incorrect: {self.consecutive_incorrect}")

    def _adjust_difficulty(self):
        old_difficulty_index = self.current_difficulty_index
        
        # Logic for difficulty adjustment based on consecutive answers
        if self.consecutive_correct >= 3:
            self.current_difficulty_index += 2
            # No reset for consecutive_correct here, as it's reset by record_answer when incorrect occurs.
            # If the next answer is correct, streak continues. If incorrect, it resets there.
            logger.info(f"Difficulty jumped up by 2 levels (3+ consecutive correct). Current streak: {self.consecutive_correct}")
        elif self.consecutive_correct >= 1:
            self.current_difficulty_index += 1
            logger.info(f"Difficulty increased by 1 level (1+ consecutive correct). Current streak: {self.consecutive_correct}")
        
        # Only adjust downwards if incorrect streak progresses significantly
        elif self.consecutive_incorrect >= 2:
            self.current_difficulty_index -= 2
            # No reset for consecutive_incorrect here, as it's reset by record_answer when correct occurs.
            logger.info(f"Difficulty dropped by 2 levels (2+ consecutive incorrect). Current streak: {self.consecutive_incorrect}")
        elif self.consecutive_incorrect == 1:
            self.current_difficulty_index -= 1
            logger.info(f"Difficulty decreased by 1 level (1 consecutive incorrect). Current streak: {self.consecutive_incorrect}")

        # Ensure difficulty stays within bounds
        self.current_difficulty_index = max(0, min(self.current_difficulty_index, len(self.difficulty_levels) - 1))
        
        if old_difficulty_index != self.current_difficulty_index:
            self.console.print(Text(f"\nDifficulty adjusted to: {self.difficulty_levels[self.current_difficulty_index].replace('_', ' ').title()}", style="italic blue"))

    def get_current_difficulty(self) -> str:
        return self.difficulty_levels[self.current_difficulty_index]

    def save_state(self, filename: str = "quiz_state.json"):
        state = {
            "score": self.score,
            "current_difficulty_index": self.current_difficulty_index,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_incorrect": self.consecutive_incorrect,
            "total_questions_answered": self.total_questions_answered,
            "question_history": self.question_history
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Quiz state saved to {filename}")
        except IOError as e:
            logger.error(f"Error saving quiz state: {e}")

    def load_state(self, filename: str = "quiz_state.json") -> bool:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.score = state.get("score", 0)
                # Ensure the loaded difficulty index is valid
                default_difficulty_index = self.difficulty_levels.index("beginner_ai_basics") if "beginner_ai_basics" in self.difficulty_levels else 0
                self.current_difficulty_index = state.get("current_difficulty_index", default_difficulty_index)
                self.consecutive_correct = state.get("consecutive_correct", 0)
                self.consecutive_incorrect = state.get("consecutive_incorrect", 0)
                self.total_questions_answered = state.get("total_questions_answered", 0)
                self.question_history = state.get("question_history", [])
                
                self.current_difficulty_index = max(0, min(self.current_difficulty_index, len(self.difficulty_levels) - 1))

                logger.info(f"Quiz state loaded from {filename}. Current difficulty: {self.get_current_difficulty()}")
                self.console.print(Text(f"\nResumed game with score: {self.score}, current difficulty: {self.get_current_difficulty().replace('_', ' ').title()}", style="green"))
                return True
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load quiz state from {filename}. Starting new game. Error: {e}")
                self.reset_state() # Reset state if loading fails
                return False
        logger.info("No saved quiz state found. Starting new game.")
        self.reset_state() # Ensure initial state is set for a new game
        return False

# --- CLIQuizApp Class (The User Interface) ---
class CLIQuizApp:
    def __init__(self):
        self.console = Console()
        self.ai_service = VertexAIService()
        self.quiz_manager = QuizManager(DIFFICULTY_LEVELS)
        self.quiz_manager.load_state() # Load state after quiz_manager is initialized
        logger.info("CLIQuizApp initialized.")

    def display_welcome(self):
        welcome_text = Text("""
🧠  Google AI Adaptive True/False Quiz  🧠

Welcome to the ultimate AI knowledge challenge!
Answer True or False questions generated by Google's Gemini AI.
🔥 Your difficulty adapts based on your performance.
❓ Learn from detailed explanations for every answer.

Type 'T' for True, 'F' for False, 'Q' to quit, 'S' for summary.
Press Enter after your choice.
""", justify="center", style="bold cyan")
        self.console.print(Panel(welcome_text, title="AI Quiz Master", border_style="cyan"))
        time.sleep(1) # Pause for reading

    def get_user_answer(self) -> str | None:
        while True:
            response = Prompt.ask(
                Text("Your answer", style="bold magenta"), 
                choices=["t", "f", "q", "s"], 
                case_sensitive=False
            ).strip().lower()
            
            if response in ['t', 'f', 'q', 's']:
                return response
            else:
                self.console.print(Text("Invalid input. Please type 'T', 'F', 'Q', or 'S'.", style="red"))

        

    def run_quiz(self):
        self.display_welcome()
        while True:
            current_difficulty = self.quiz_manager.get_current_difficulty()
            self.console.print(Text(f"\n--- Current Difficulty: {current_difficulty.replace('_', ' ').title()} ---", style="bold underline blue"))

            question_data = self.ai_service.get_true_false_question(current_difficulty)

            if not question_data:
                self.console.print(Text("Could not generate a question. Please try again later or check logs for AI service errors.", style="bold red"))
                time.sleep(2)
                continue

            try:
                question_statement = question_data["statement"]
                is_true_actual = question_data["is_true"]
                explanation = question_data["explanation"]
            except KeyError as e:
                # This block serves as a final defensive check, though VertexAIService aims to prevent this.
                logger.error(f"Malformed question data received from AI, missing key: {e}. Raw data: {question_data}")
                self.console.print(Text(f"Received incomplete question data from AI. Retrying...", style="bold red"))
                time.sleep(1)
                continue # Skip this question

            self.console.print(Panel(
                Text(question_statement, style="bold white"),
                title="True or False?",
                border_style="yellow"
            ))

            user_input = self.get_user_answer()

            if user_input == 'q':
                self.quiz_manager.save_state()
                self.display_summary() # Display summary before quitting
                self.console.print(Text("\nThanks for playing! Your progress has been saved.", style="bold green"))
                break
            elif user_input == 's':
                self.display_summary()
                continue # Go back to the main loop to ask another question

            user_answer_bool = (user_input == 't')
            is_correct = (user_answer_bool == is_true_actual)

            self.quiz_manager.record_answer(question_data, user_answer_bool, is_correct)
            
            # Add a small pause for the user to process the explanation
            time.sleep(2) 
            self.console.print("\n" + "="*80 + "\n") # Separator

    def display_summary(self):
        self.console.print(Panel(
            Text("Quiz Summary", style="bold underline blue"),
            title_align="center",
            border_style="blue"
        ))
        
        total = self.quiz_manager.total_questions_answered
        correct = self.quiz_manager.score
        
        self.console.print(Text(f"Total Questions Answered: {total}", style="bold green"))
        self.console.print(Text(f"Correct Answers: {correct}", style="bold green"))
        if total > 0:
            accuracy = (correct / total) * 100
            self.console.print(Text(f"Accuracy: {accuracy:.2f}%", style="bold green"), highlight=True)
        else:
            self.console.print(Text("Accuracy: N/A (No questions answered)", style="bold green"))

        if not self.quiz_manager.question_history:
            self.console.print(Text("\nNo questions in history yet.", style="italic"))
            return

        self.console.print(Panel(
            Text("Question History", style="bold underline yellow"),
            title_align="center",
            border_style="yellow"
        ))

        for i, q_record in enumerate(self.quiz_manager.question_history):
            status = "✅ Correct" if q_record.get('is_correct') else "❌ Incorrect"
            user_ans_str = "True" if q_record.get('user_answer') else "False"
            correct_ans_str = "True" if q_record.get('correct_answer') else "False"
            
            # Use .get() for additional robustness in history display
            statement_at_record = q_record.get('statement', 'N/A')
            explanation_at_record = q_record.get('explanation', 'No explanation recorded.')
            difficulty_at_record = q_record.get('difficulty_at_time', 'Unknown Difficulty').replace('_', ' ').title()

            self.console.print(
                Panel(
                    Text(f"Statement: {statement_at_record}\n", style="white") +
                    Text(f"Your Answer: {user_ans_str} ({status})\n", style="magenta") +
                    Text(f"Correct Answer: {correct_ans_str}\n", style="green") +
                    Text(f"Explanation: {explanation_at_record}\n", style="gold3") +
                    Text(f"Difficulty: {difficulty_at_record}", style="italic blue"),
                    title=f"Question {i+1}",
                    border_style="gray"
                )
            )
            self.console.print("-" * 60) # Separator for questions
        
        # Option to clear history or continue for next time
        clear_history = Prompt.ask(
            Text("\nDo you want to clear your quiz history for the next session (y/n)?", style="bold yellow"),
            choices=["y", "n"],
            case_sensitive=False
        ).strip().lower()
        if clear_history == 'y':
            self.quiz_manager.reset_state()
            self.console.print(Text("Quiz history cleared. Starting fresh next time.", style="green"))
            self.quiz_manager.save_state() # Save empty state to disk
        else:
            self.quiz_manager.save_state() # Save current state even if not cleared
            self.console.print(Text("Quiz history retained for next session.", style="green"))


# --- Main Execution ---
if __name__ == "__main__":
    app = CLIQuizApp()
    try:
        app.run_quiz()
    except KeyboardInterrupt:
        # --- FIX: Display summary on KeyboardInterrupt ---
        app.quiz_manager.save_state()
        app.display_summary() # Display summary on exit via Ctrl+C
        app.console.print(Text("\n\nQuiz interrupted. Your progress has been saved.", style="bold yellow"))
    except Exception as e:
        logger.exception("An unhandled error occurred during quiz execution:")
        app.console.print(Text(f"\nAn unexpected error occurred: {e}. Please check the logs.", style="bold red"))
        app.quiz_manager.save_state() # Attempt to save state even on error
