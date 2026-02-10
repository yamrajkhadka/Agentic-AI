"""
HerAI - Main Application
Romantic AI Assistant powered by Llama 3.3 70B
"""

import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import utilities
from utils.llm_config import get_llm_instance

# Import agents
from agents.mood_detector import MoodDetector
from agents.memory_agent import MemoryAgent
from agents.romantic_agent import RomanticAgent
from agents.surprise_agent import SurpriseAgent
from agents.safety_agent import SafetyAgent


class HerAI:
    """Main HerAI Application"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize HerAI
        
        Args:
            api_key: Groq API key for Llama 3.3 70B (optional, reads from env)
        """
        print("💕 Initializing HerAI...")
        
        # Get LLM instance
        self.llm = get_llm_instance(api_key)
        self.use_llm = self.llm is not None
        
        if self.use_llm:
            print("✅ Llama 3.3 70B connected successfully!")
        else:
            print("⚠️  Running in fallback mode (no LLM). Set GROQ_API_KEY for best experience.")
        
        # Initialize agents
        print("🤖 Loading agents...")
        self.mood_detector = MoodDetector(llm=self.llm)
        self.memory_agent = MemoryAgent()
        self.romantic_agent = RomanticAgent(llm=self.llm, personality="Yamraj")
        self.surprise_agent = SurpriseAgent(llm=self.llm)
        self.safety_agent = SafetyAgent(strictness="medium")
        
        print("✅ HerAI ready!\n")
    
    def process_message(self, message: str) -> Dict:
        """
        Process a message from girlfriend
        
        Args:
            message: Her message
            
        Returns:
            Dict with response and metadata
        """
        print(f"\n💬 Processing: '{message}'\n")
        
        # Step 1: Detect mood
        print("1️⃣  Detecting mood...")
        mood_result = self.mood_detector.detect(message, use_llm=self.use_llm)
        mood = mood_result['mood']
        mood_emoji = mood_result['emoji']
        print(f"   Mood: {mood} {mood_emoji}")
        
        # Step 2: Check if it's a task request
        task_type = self._detect_task_type(message)
        
        if task_type:
            print(f"2️⃣  Task detected: {task_type}")
            response = self._handle_task(message, task_type, mood)
        else:
            # Step 2: Retrieve memories if needed
            memories = []
            if mood in ['sad', 'stressed', 'angry', 'romantic']:
                print("2️⃣  Retrieving memories...")
                memories = self.memory_agent.retrieve_memories(message, k=2)
                print(f"   Found {len(memories)} relevant memories")
            
            # Step 3: Generate romantic response
            print("3️⃣  Generating response...")
            response = self.romantic_agent.generate_message(
                mood=mood,
                context=message,
                memories=memories
            )
        
        # Step 4: Safety check
        print("4️⃣  Safety check...")
        safety_result = self.safety_agent.validate_and_fix(response)
        final_response = safety_result['fixed_text']
        
        print(f"   Safety score: {safety_result['fixed_score']}/100")
        print(f"\n✅ Response ready!\n")
        
        return {
            'response': final_response,
            'mood': mood,
            'mood_emoji': mood_emoji,
            'safe': safety_result['fixed_safe'],
            'safety_score': safety_result['fixed_score'],
            'task_type': task_type
        }
    
    def _detect_task_type(self, message: str) -> str:
        """Detect if message is asking for a specific task"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['write a poem', 'poem for', 'make a poem']):
            return 'poem'
        elif any(word in message_lower for word in ['write a joke', 'tell me a joke', 'joke about yamraj', 'make fun of yourself']):
            return 'joke'
        elif any(word in message_lower for word in ['write a story', 'tell me a story']):
            return 'story'
        elif any(word in message_lower for word in ['write a letter', 'love letter']):
            return 'letter'
        elif any(word in message_lower for word in ['plan a date', 'date idea', 'what should we do']):
            return 'date_plan'
        elif any(word in message_lower for word in ['good morning', 'morning']):
            return 'good_morning'
        elif any(word in message_lower for word in ['good night', 'night']):
            return 'good_night'
        elif any(word in message_lower for word in ['sorry', 'apologize', 'my bad']):
            return 'apology'
        
        return None
    
    def _handle_task(self, message: str, task_type: str, mood: str) -> str:
        """Handle specific task requests"""
        
        if task_type == 'poem':
            theme = 'love'  # Default theme
            if 'miss' in message.lower():
                theme = 'missing'
            elif 'thank' in message.lower() or 'appreciate' in message.lower():
                theme = 'appreciation'
            return self.romantic_agent.generate_poem(theme)
        
        elif task_type == 'joke':
            return self.romantic_agent.generate_joke_about_yamraj(message)
        
        elif task_type == 'date_plan':
            date_plan = self.surprise_agent.plan_virtual_date(message)
            response = f"{date_plan['title']}\n\n{date_plan['description']}\n\n"
            response += "Here's how we can do it:\n"
            for i, step in enumerate(date_plan['steps'], 1):
                response += f"{i}. {step}\n"
            if date_plan.get('suggestions'):
                response += f"\n💡 Tip: {date_plan['suggestions'][0]}"
            return response
        
        elif task_type == 'good_morning':
            return self.romantic_agent.generate_good_morning()
        
        elif task_type == 'good_night':
            return self.romantic_agent.generate_good_night()
        
        elif task_type == 'apology':
            context = message.replace('sorry', '').replace('apologize', '').strip()
            return self.romantic_agent.generate_apology(context)
        
        else:
            # General task handling with LLM
            return self.romantic_agent.handle_task(message, task_type)
    
    def chat(self):
        """Interactive chat mode"""
        print("="*60)
        print("💕 HerAI - Your Romantic AI Assistant")
        print("="*60)
        print("\nYamraj is here to talk! Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get input
                user_input = input("Her: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nYamraj: I'll miss you! Talk to you soon 💕")
                    break
                
                # Process message
                result = self.process_message(user_input)
                
                # Display response
                print(f"\nYamraj {result['mood_emoji']}: {result['response']}\n")
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\nYamraj: Goodbye, love! 💕")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    # Get API key from environment or user input
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("\n⚠️  No GROQ_API_KEY found in environment")
        print("For best experience, set your Groq API key:")
        print("   export GROQ_API_KEY='your-key-here'")
        print("\nOr enter it now (press Enter to skip):")
        api_key = input("API Key: ").strip()
    
    # Initialize HerAI
    app = HerAI(api_key=api_key if api_key else None)
    
    # Test mode or chat mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode
        print("\n" + "="*60)
        print("🧪 TEST MODE")
        print("="*60 + "\n")
        
        test_messages = [
            "I love you so much! ❤️",
            "I miss you... 😢",
            "Write a poem for me about love",
            "Tell me a joke about yourself Yamraj",
            "I'm so stressed with work",
            "Good morning!",
            "Plan a virtual date for us"
        ]
        
        for msg in test_messages:
            print(f"\n{'='*60}")
            result = app.process_message(msg)
            print(f"\n💬 Her: {msg}")
            print(f"\n💕 Yamraj {result['mood_emoji']}: {result['response']}")
            print(f"\n📊 Mood: {result['mood']} | Safety: {result['safety_score']}/100")
    else:
        # Interactive chat mode
        app.chat()


if __name__ == "__main__":
    main()