#!/usr/bin/env python3
"""
UCB Quiz Master - An adaptive learning quiz application using Upper Confidence Bound algorithm.
"""
import sys
from quiz_manager import QuizManager
import config


def display_question(question: dict, question_num: int, subtopic: str):
    """Display a question to the user."""
    print("\n" + "="*60)
    print(f"Question {question_num}")
    print(f"Subtopic: {subtopic}")
    print("="*60)
    print(f"\n{question['question']}\n")
    
    for option, text in sorted(question['options'].items()):
        print(f"  {option}. {text}")
    print()


def get_user_answer() -> str:
    """Get and validate user input."""
    while True:
        answer = input("Your answer (A/B/C/D) or 'q' to quit, 's' for stats: ").strip().upper()
        
        if answer == 'Q':
            return 'Q'
        elif answer == 'S':
            return 'S'
        elif answer in ['A', 'B', 'C', 'D']:
            return answer
        else:
            print("❌ Invalid input. Please enter A, B, C, D, 'q' to quit, or 's' for stats.")


def display_result(result: dict):
    """Display the result of the user's answer."""
    print()
    if result['is_correct']:
        print("✅ Correct! Well done!")
    else:
        print(f"❌ Incorrect. The correct answer is: {result['correct_answer']}")
        print(f"\n💡 Explanation:")
        print(f"   {result['explanation']}")
    print()


def main():
    """Main application entry point."""
    print("\n" + "🎓"*30)
    print("      WELCOME TO UCB QUIZ MASTER")
    print("   Adaptive Learning with Upper Confidence Bound")
    print("🎓"*30 + "\n")
    
    # Check if API key is configured
    if not config.OPENAI_API_KEY:
        print("❌ Error: OpenAI API key not configured!")
        print("\nPlease follow these steps:")
        print("1. Create a .env file in the project directory")
        print("2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here")
        print("3. Get an API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Get topic from user
    print("This application will help you learn by identifying your weak areas")
    print("and focusing on them using an intelligent algorithm.\n")
    
    topic = input("📚 Enter the topic you want to quiz on: ").strip()
    
    if not topic:
        print("❌ Topic cannot be empty!")
        sys.exit(1)
    
    # Optional: customize number of subtopics
    num_subtopics_input = input("\n🔢 Number of subtopics to generate (default 5, press Enter to skip): ").strip()
    num_subtopics = 5
    if num_subtopics_input:
        try:
            num_subtopics = int(num_subtopics_input)
            if num_subtopics < 2:
                print("⚠️  Using minimum of 2 subtopics")
                num_subtopics = 2
            elif num_subtopics > 10:
                print("⚠️  Using maximum of 10 subtopics")
                num_subtopics = 10
        except ValueError:
            print("⚠️  Invalid number, using default of 5")
    
    # Initialize quiz manager
    quiz = QuizManager(topic)
    
    try:
        quiz.initialize(num_subtopics)
    except Exception as e:
        print(f"\n❌ Error initializing quiz: {e}")
        print("Please check your OpenAI API key and internet connection.")
        sys.exit(1)
    
    # Main quiz loop
    print("="*60)
    print("🚀 LET'S BEGIN!")
    print("="*60)
    print("\nThe algorithm will adapt to your performance and focus on your weak areas.")
    print("Type 'q' at any time to quit, or 's' to see your progress.\n")
    
    input("Press Enter to start...")
    
    question_count = 0
    
    try:
        while True:
            # Get next question
            question = quiz.get_next_question()
            
            if not question:
                print("\n⚠️  No more questions available!")
                break
            
            question_count += 1
            
            # Display question
            display_question(question, question_count, quiz.current_subtopic)
            
            # Get user answer
            answer = get_user_answer()
            
            if answer == 'Q':
                print("\n👋 Thanks for learning with UCB Quiz Master!")
                break
            elif answer == 'S':
                print(quiz.get_progress_summary())
                question_count -= 1  # Don't count this as a question
                input("\nPress Enter to continue...")
                continue
            
            # Submit answer and get result
            result = quiz.submit_answer(answer)
            
            # Display result
            display_result(result)
            
            # Every 10 questions, show a mini summary
            if question_count % 10 == 0:
                print("\n" + "-"*60)
                print(f"📈 Progress Check - {question_count} questions answered")
                print("-"*60)
                weakest = quiz.get_weakest_subtopic()
                print(f"🎯 Current focus area: {weakest}")
                print("\nType 's' on the next question to see detailed statistics.")
                print("-"*60)
            
            input("Press Enter for next question...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Quiz interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"\nTotal questions answered: {quiz.get_questions_answered()}")
    print(quiz.get_progress_summary())
    
    weakest = quiz.get_weakest_subtopic()
    print(f"🎯 Your weakest area: {weakest}")
    print("\n💪 Keep practicing to improve!\n")
    
    print("="*60)
    print("Thanks for using UCB Quiz Master!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

