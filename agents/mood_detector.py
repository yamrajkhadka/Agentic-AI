"""
Mood Detection Agent with Llama 3.3 70B
Analyzes emotional state from user messages
"""

from typing import Dict, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MoodDetector:
    """Detects user's emotional state from their message"""
    
    MOODS = {
        'happy': ['happy', 'great', 'wonderful', 'amazing', 'excited', 'love', 'yay', '😊', '😄', '❤️'],
        'sad': ['sad', 'miss', 'lonely', 'down', 'unhappy', 'cry', '😢', '😭'],
        'stressed': ['stress', 'tired', 'exhausted', 'overwhelm', 'busy', 'anxious', '😰', '😫'],
        'romantic': ['love', 'kiss', 'hug', 'cuddle', 'romance', 'date', '💕', '💖', '😘'],
        'playful': ['haha', 'lol', 'fun', 'play', 'tease', 'silly', '😜', '😝', '🤪'],
        'angry': ['angry', 'mad', 'upset', 'annoyed', 'frustrated', '😠', '😡'],
        'neutral': []
    }
    
    def __init__(self, llm=None):
        """
        Initialize mood detector
        
        Args:
            llm: Optional language model for advanced mood detection (Llama 3.3 70B)
        """
        self.llm = llm
        
        if llm:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Yamraj, an emotionally intelligent boyfriend who deeply understands his girlfriend's feelings.

Analyze her message and detect her emotional state. Respond with ONLY ONE WORD from these options:
happy, sad, stressed, romantic, playful, angry, neutral

Consider:
- Word choice and tone
- Emojis and punctuation  
- Context and subtext
- What she really means, not just what she says
- Her emotional needs behind the words

Be perceptive and caring in your analysis."""),
                ("user", "{message}")
            ])
            self.chain = self.prompt | llm | StrOutputParser()
    
    def detect_mood_simple(self, message: str) -> str:
        """
        Simple keyword-based mood detection (fast, no LLM needed)
        
        Args:
            message: User's message
            
        Returns:
            Detected mood as string
        """
        message_lower = message.lower()
        
        # Count matches for each mood
        mood_scores = {}
        for mood, keywords in self.MOODS.items():
            if mood == 'neutral':
                continue
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                mood_scores[mood] = score
        
        # Return mood with highest score, or neutral if no matches
        if mood_scores:
            return max(mood_scores.items(), key=lambda x: x[1])[0]
        return 'neutral'
    
    def detect_mood_llm(self, message: str) -> str:
        """
        LLM-based mood detection using Llama 3.3 70B (more nuanced, requires LLM)
        
        Args:
            message: User's message
            
        Returns:
            Detected mood as string
        """
        if not self.llm:
            return self.detect_mood_simple(message)
        
        try:
            mood = self.chain.invoke({"message": message}).strip().lower()
            # Validate response
            if mood in self.MOODS:
                return mood
            return 'neutral'
        except Exception as e:
            print(f"⚠️  LLM mood detection failed, using simple detection: {e}")
            return self.detect_mood_simple(message)
    
    def detect(self, message: str, use_llm: bool = True) -> Dict[str, str]:
        """
        Main detection method - defaults to LLM if available
        
        Args:
            message: User's message
            use_llm: Whether to use LLM-based detection (default: True)
            
        Returns:
            Dict with mood and emoji
        """
        if use_llm and self.llm:
            mood = self.detect_mood_llm(message)
        else:
            mood = self.detect_mood_simple(message)
        
        # Get appropriate emoji
        emoji = self._get_mood_emoji(mood)
        
        return {
            'mood': mood,
            'emoji': emoji,
            'message': message
        }
    
    def _get_mood_emoji(self, mood: str) -> str:
        """Get emoji for a mood"""
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'stressed': '😰',
            'romantic': '💕',
            'playful': '😜',
            'angry': '😠',
            'neutral': '😐'
        }
        return emoji_map.get(mood, '😐')
    
    def get_mood_description(self, mood: str) -> str:
        """Get a description of the detected mood"""
        descriptions = {
            'happy': "You seem happy and cheerful!",
            'sad': "You seem a bit down. I'm here for you.",
            'stressed': "You seem stressed. Let me help you relax.",
            'romantic': "You're in a romantic mood! 💖",
            'playful': "You're being playful and fun!",
            'angry': "You seem upset. Want to talk about it?",
            'neutral': "You're in a calm mood."
        }
        return descriptions.get(mood, "I'm here for you!")


# Example usage
if __name__ == "__main__":
    # Test without LLM
    detector = MoodDetector()
    
    test_messages = [
        "I love you so much! ❤️",
        "I miss you... feeling lonely",
        "Haha you're so silly! 😜",
        "I'm so stressed with work",
        "Just a normal day"
    ]
    
    print("🧠 Mood Detection Demo (Simple Mode)\n")
    for msg in test_messages:
        result = detector.detect(msg, use_llm=False)
        print(f"Message: {msg}")
        print(f"Mood: {result['mood']} {result['emoji']}")
        print(f"Description: {detector.get_mood_description(result['mood'])}")
        print()