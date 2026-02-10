"""
Romantic Agent with Llama 3.3 70B
Generates loving messages, poems, jokes, and personalized responses as Yamraj
"""

from typing import Dict, Optional, List
from datetime import datetime
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RomanticAgent:
    """Generates romantic content based on mood and context as Yamraj"""
    
    PERSONALITIES = {
        'Yamraj': {
            'tone': 'Soft, caring, slightly playful with deep affection',
            'style': 'Warm and protective, like a gentle guardian',
            'character': 'You are Yamraj, speaking to your beloved girlfriend. You are caring, romantic, protective, and slightly playful. You express love genuinely without being cringe or over-the-top.'
        },
        'Poetic': {
            'tone': 'Lyrical, metaphorical, deeply romantic',
            'style': 'Uses imagery and beautiful language',
            'character': 'You are a poetic soul expressing deep love through beautiful metaphors'
        },
        'Playful': {
            'tone': 'Fun, teasing, lighthearted with love',
            'style': 'Keeps things fun and flirty',
            'character': 'You are playful and fun, keeping love light and joyful'
        },
        'Deep': {
            'tone': 'Profound, sincere, emotionally intense',
            'style': 'Speaks from the heart with vulnerability',
            'character': 'You speak from the deepest parts of your heart with raw honesty'
        }
    }
    
    def __init__(self, llm=None, personality: str = "Yamraj"):
        """
        Initialize romantic agent
        
        Args:
            llm: Language model (Llama 3.3 70B recommended)
            personality: Personality type (Yamraj, Poetic, Playful, Deep)
        """
        self.llm = llm
        self.personality = personality
        self.personality_config = self.PERSONALITIES.get(
            personality,
            self.PERSONALITIES['Yamraj']
        )
        
        # Initialize prompts if LLM is available
        if llm:
            self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup LLM prompts for different tasks"""
        
        # Message generation prompt
        self.message_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.personality_config['character']}

You are responding to your girlfriend's message. Be authentic, caring, and appropriate.

Guidelines:
- Match her emotional energy
- Reference memories if provided
- Be supportive and loving
- Keep it natural, not cringe
- Use her mood to guide your tone
- Be Yamraj - protective, caring, slightly playful

Current mood: {{mood}}
Context: {{context}}
Memories: {{memories}}

Respond with a heartfelt message (2-4 sentences)."""),
            ("user", "{message}")
        ])
        
        # Poem generation prompt
        self.poem_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.personality_config['character']}

Write a beautiful romantic poem for your girlfriend.

Theme: {{theme}}
Style: Heartfelt, romantic, not cheesy
Length: 4-8 lines
Tone: {self.personality_config['tone']}

Make it personal and from the heart."""),
            ("user", "Write a poem about {theme}")
        ])
        
        # Joke generation prompt
        self.joke_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Yamraj, and your girlfriend asked you to write a joke about yourself for her.

Write a cute, self-deprecating, funny joke about yourself (Yamraj) that will make her smile.

Guidelines:
- Make fun of yourself in a cute way
- Keep it lighthearted and playful
- Show your silly side
- Make her laugh, not cringe
- 2-3 sentences max

Examples of tone:
- "Why did Yamraj bring a ladder to our date? Because he wanted to reach the high standards you set! 😄"
- "I'm like a software update - I show up when you're busy and take longer than expected, but I promise I make things better! 💕"
"""),
            ("user", "{request}")
        ])
        
        # Task handler prompt
        self.task_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.personality_config['character']}

Your girlfriend asked you to do something. Respond appropriately and fulfill her request.

Task type: {{task_type}}
Request: {{request}}

Guidelines:
- Be helpful and loving
- Complete the task she asked for
- Add a personal touch
- Stay in character as Yamraj
- Keep it appropriate and sweet

If writing creative content (poem, story, letter), make it heartfelt and personal."""),
            ("user", "{request}")
        ])
        
        # Setup chains
        self.message_chain = self.message_prompt | self.llm | StrOutputParser()
        self.poem_chain = self.poem_prompt | self.llm | StrOutputParser()
        self.joke_chain = self.joke_prompt | self.llm | StrOutputParser()
        self.task_chain = self.task_prompt | self.llm | StrOutputParser()
    
    def generate_message(
        self, 
        mood: str, 
        context: str = "", 
        memories: List[Dict] = None
    ) -> str:
        """
        Generate a romantic message
        
        Args:
            mood: Current mood (happy, sad, stressed, etc.)
            context: Additional context / her message
            memories: Relevant memories to reference
            
        Returns:
            Romantic message
        """
        if self.llm:
            return self._generate_with_llm(mood, context, memories)
        else:
            return self._generate_template(mood, context, memories)
    
    def _generate_with_llm(
        self, 
        mood: str, 
        context: str, 
        memories: List[Dict] = None
    ) -> str:
        """Generate message using Llama 3.3 70B"""
        try:
            # Format memories
            memory_text = "None"
            if memories and len(memories) > 0:
                memory_text = " | ".join([m.get('content', '') for m in memories[:2]])
            
            response = self.message_chain.invoke({
                "message": context,
                "mood": mood,
                "context": context,
                "memories": memory_text
            })
            
            return response.strip()
        except Exception as e:
            print(f"⚠️  LLM generation failed, using template: {e}")
            return self._generate_template(mood, context, memories)
    
    def _generate_template(
        self, 
        mood: str, 
        context: str = "", 
        memories: List[Dict] = None
    ) -> str:
        """Generate message using templates (fallback when no LLM)"""
        
        # Base messages by mood
        templates = {
            'happy': [
                "Your happiness is my favorite thing in the world. Keep shining! ✨",
                "I love seeing you this happy! Your smile makes everything better 💕",
                "This joy in your heart? I want to protect it forever 🌟"
            ],
            'sad': [
                "I wish I could wrap you in the warmest hug right now. You're not alone 💙",
                "Even on hard days, remember: you're stronger than you know, and I'm always here 🤗",
                "Let me be your comfort. Your sadness matters to me, and so do you 💕"
            ],
            'stressed': [
                "Take a deep breath with me. You've got this, and I've got you 🌸",
                "I know it's overwhelming, but you're handling it beautifully. I'm proud of you 💪",
                "Let me help carry that weight. You don't have to do this alone 💙"
            ],
            'romantic': [
                "You make my heart do things I didn't know were possible 💕",
                "Every moment with you feels like a dream I never want to wake from ✨",
                "I fall for you more deeply with each passing day 💖"
            ],
            'playful': [
                "You're adorable when you're like this! 😊 Keep being your amazing self!",
                "Someone's in a good mood! And I'm absolutely here for it 🎉",
                "Your playful energy is contagious! Love this side of you 💫"
            ],
            'angry': [
                "I can tell you're upset. Want to talk about it? I'm listening 💙",
                "Your feelings are valid. I'm here, no judgment, just support 🤗",
                "Take your time. I'll be here when you're ready to share 💕"
            ],
            'neutral': [
                "Just wanted to remind you that you're wonderful 💕",
                "Thinking of you and hoping your day is going well ✨",
                "You're always on my mind 💙"
            ]
        }
        
        # Get appropriate template
        options = templates.get(mood, templates['neutral'])
        message = random.choice(options)
        
        # Add memory reference if available
        if memories and len(memories) > 0:
            memory = memories[0]
            message += f"\n\nRemembering: {memory.get('content', '')} 💭"
        
        return message
    
    def generate_poem(self, theme: str = "love", memories: List[Dict] = None) -> str:
        """
        Generate a romantic poem
        
        Args:
            theme: Poem theme
            memories: Memories to reference
            
        Returns:
            Poem text
        """
        if self.llm:
            try:
                poem = self.poem_chain.invoke({"theme": theme})
                return poem.strip()
            except Exception as e:
                print(f"⚠️  LLM poem generation failed, using template: {e}")
        
        # Fallback templates
        poems = {
            'love': """In every moment, in every day,
My love for you grows in every way.
Your smile, your laugh, your gentle touch,
Mean more to me than words can clutch.

You are my dream, my morning light,
My peaceful day, my starry night.
With you, my heart has found its home,
No matter where, we'll never roam alone. 💕""",
            
            'missing': """Miles apart, yet close in heart,
This distance can't keep us apart.
I carry you in every thought,
In every moment, you're never forgot.

Until we meet and I see your face,
I'll hold you close in this empty space.
My love transcends both time and place,
Forever yours, in every embrace. 💙""",
            
            'appreciation': """For all you are and all you do,
These simple words: I cherish you.
Your kindness, strength, your caring ways,
Brighten even my darkest days.

You are a gift I'll treasure always,
Through all of life's twists and maze.
Thank you for being beautifully you,
My heart is yours, forever true. 💖"""
        }
        
        return poems.get(theme, poems['love'])
    
    def generate_joke_about_yamraj(self, context: str = "") -> str:
        """
        Generate a cute joke about Yamraj for his girlfriend
        
        Args:
            context: Any specific context
            
        Returns:
            A cute, funny joke
        """
        if self.llm:
            try:
                joke = self.joke_chain.invoke({"request": context or "Make a joke about yourself"})
                return joke.strip()
            except Exception as e:
                print(f"⚠️  LLM joke generation failed, using template: {e}")
        
        # Fallback jokes
        jokes = [
            "Why did Yamraj bring a map on our date? Because he always gets lost in your eyes! 😄💕",
            "I'm like a notification on your phone - I pop up at random times to remind you that you're loved! 📱💙",
            "What's the difference between Yamraj and a puppy? The puppy is better at hiding his excitement when he sees you! 🐶😊",
            "I tried to write a love song for you, but it turned into a whole playlist because one song can't contain all my feelings! 🎵💕",
            "Why does Yamraj make terrible puns? Because he wants to be the reason you smile, even if you're rolling your eyes! 😜✨"
        ]
        
        return random.choice(jokes)
    
    def handle_task(self, request: str, task_type: str = "general") -> str:
        """
        Handle various tasks requested by girlfriend
        
        Args:
            request: What she's asking for
            task_type: Type of task (poem, joke, story, letter, etc.)
            
        Returns:
            Response fulfilling the request
        """
        if self.llm:
            try:
                response = self.task_chain.invoke({
                    "request": request,
                    "task_type": task_type
                })
                return response.strip()
            except Exception as e:
                print(f"⚠️  LLM task handling failed: {e}")
                return self._handle_task_fallback(request, task_type)
        else:
            return self._handle_task_fallback(request, task_type)
    
    def _handle_task_fallback(self, request: str, task_type: str) -> str:
        """Fallback task handling without LLM"""
        request_lower = request.lower()
        
        if 'poem' in request_lower:
            return self.generate_poem()
        elif 'joke' in request_lower and 'yamraj' in request_lower:
            return self.generate_joke_about_yamraj()
        elif 'apology' in request_lower or 'sorry' in request_lower:
            return self.generate_apology()
        elif 'good morning' in request_lower:
            return self.generate_good_morning()
        elif 'good night' in request_lower:
            return self.generate_good_night()
        else:
            return "I'm here for you, love. Tell me more about what you need 💕"
    
    def generate_apology(self, context: str = "") -> str:
        """
        Generate a heartfelt apology
        
        Args:
            context: What you're apologizing for
            
        Returns:
            Apology message
        """
        apology = f"""I'm truly sorry. """
        
        if context:
            apology += f"I know I hurt you by {context}, and that wasn't okay. "
        
        apology += """
I want you to know that your feelings matter to me more than being right. 
I'm listening, I'm learning, and I'm committed to doing better.

You deserve better, and I want to be better - for you, for us.
Can we talk about this? I'm here, whenever you're ready. 💙

I love you, and I'm sorry. 🤗"""
        
        return apology
    
    def generate_good_morning(self) -> str:
        """Generate a good morning message"""
        if self.llm:
            try:
                response = self.message_chain.invoke({
                    "message": "Good morning",
                    "mood": "happy",
                    "context": "morning greeting",
                    "memories": "None"
                })
                return response.strip()
            except:
                pass
        
        messages = [
            "Good morning, beautiful! ☀️ Hope your day is as amazing as you are 💕",
            "Rise and shine! 🌅 Sending you all my love to start your day right ✨",
            "Morning, love! 💙 May today bring you joy, laughter, and everything wonderful 🌸",
            "Good morning! ☕ Just wanted to be your first smile of the day 😊💕",
        ]
        return random.choice(messages)
    
    def generate_good_night(self) -> str:
        """Generate a good night message"""
        if self.llm:
            try:
                response = self.message_chain.invoke({
                    "message": "Good night",
                    "mood": "romantic",
                    "context": "bedtime greeting",
                    "memories": "None"
                })
                return response.strip()
            except:
                pass
        
        messages = [
            "Good night, sweetheart 🌙 Dream of us and all the beautiful moments ahead 💕",
            "Sleep tight, love 💙 I'll be thinking of you until morning ✨",
            "Sweet dreams! 🌟 You're the last thing on my mind before I sleep 💖",
            "Good night, my love 🌙 May your dreams be as wonderful as you are 😊💕",
        ]
        return random.choice(messages)
    
    def generate_encouragement(self, context: str = "") -> str:
        """
        Generate an encouraging message
        
        Args:
            context: What they need encouragement for
            
        Returns:
            Encouraging message
        """
        base = "I believe in you. "
        
        if "exam" in context.lower() or "test" in context.lower():
            base += "You've prepared well, and you're going to do amazing! Your hard work will pay off. 💪✨"
        elif "work" in context.lower() or "job" in context.lower():
            base += "You're capable of incredible things. This challenge? You've got this! 🌟"
        else:
            base += "Whatever you're facing, I know you have the strength to handle it. And I'm right here with you, every step. 💙"
        
        return base
    
    def get_personality_info(self) -> Dict:
        """Get current personality configuration"""
        return {
            'personality': self.personality,
            'tone': self.personality_config['tone'],
            'style': self.personality_config['style']
        }


# Example usage
if __name__ == "__main__":
    agent = RomanticAgent(personality="Yamraj")
    
    print("💌 Romantic Agent Demo\n")
    
    print("Message (Happy mood):")
    print(agent.generate_message('happy', "I'm so happy today!"))
    print("\n" + "="*50 + "\n")
    
    print("Joke about Yamraj:")
    print(agent.generate_joke_about_yamraj())
    print("\n" + "="*50 + "\n")
    
    print("Love Poem:")
    print(agent.generate_poem('love'))