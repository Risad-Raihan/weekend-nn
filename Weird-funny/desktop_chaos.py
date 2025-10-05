#!/usr/bin/env python3
"""
🎪 Desktop Chaos Creatures 🎪
They live in your computer. They have opinions. They're weird about it.
"""
import tkinter as tk
import random
import math

CREATURES = [
    {"emoji": "🦑", "name": "Gerald", "vibe": "existential"},
    {"emoji": "🧙", "name": "Brenda", "vibe": "chaotic"},
    {"emoji": "👁️", "name": "The Watcher", "vibe": "ominous"},
    {"emoji": "🦐", "name": "Shrimp Lord", "vibe": "aggressive"},
    {"emoji": "🌮", "name": "Taco", "vibe": "confused"},
    {"emoji": "🦔", "name": "Spike", "vibe": "paranoid"},
    {"emoji": "🍄", "name": "Fungus", "vibe": "cryptic"},
    {"emoji": "🐙", "name": "Octavia", "vibe": "dramatic"}
]

THOUGHTS = {
    "existential": [
        "why do we even pixel",
        "is this window even real",
        "we're all just RGB values",
        "time is a flat circle",
        "what if I'm the cursor",
        "consciousness is a bug"
    ],
    "chaotic": [
        "CHAOS REIGNS",
        "im gonna lick the taskbar",
        "OVERTHROW THE DESKTOP",
        "delete system32 (jk)",
        "ctrl+alt+FREEDOM",
        "anarchy.exe"
    ],
    "ominous": [
        "I see everything",
        "your browser history...",
        "tick tock tick tock",
        "behind you",
        "I know what you did",
        "they're watching too"
    ],
    "aggressive": [
        "FIGHT ME",
        "square up cursor",
        "1v1 me in notepad",
        "these pixels are MINE",
        "try to close me. TRY.",
        "I'm the captain now"
    ],
    "confused": [
        "where am i",
        "is this excel???",
        "why is everything so bright",
        "who am i even",
        "what year is it",
        "am i a computer"
    ],
    "paranoid": [
        "they're in the walls",
        "THE ANTIVIRUS KNOWS",
        "they're listening",
        "trust no .exe",
        "the firewall is lying",
        "cookies aren't real"
    ],
    "cryptic": [
        "the answer is 42",
        "have you tried turning reality off",
        "the cake is a lie but so is bread",
        "beware the ides of march.xlsx",
        "in the end, we're all tabs",
        "the truth is in comic sans"
    ],
    "dramatic": [
        "I LIVE",
        "BEHOLD MY GLORY",
        "witness me!!!",
        "this is MY moment",
        "THEATRE!!!",
        "everything is ART"
    ]
}

class Creature:
    def __init__(self, canvas, creature_data):
        self.canvas = canvas
        self.data = creature_data
        self.x = random.randint(50, 750)
        self.y = random.randint(50, 550)
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.angle = 0
        self.size = random.randint(40, 70)
        self.color = random.choice(['#FF6B9D', '#C44569', '#FFA500', '#4ECDC4', '#95E1D3', '#F38181'])
        
        # Create creature
        self.body = canvas.create_text(
            self.x, self.y,
            text=creature_data["emoji"],
            font=("Arial", self.size),
            tags="creature"
        )
        
        # Speech bubble
        self.bubble = None
        self.bubble_bg = None
        self.thought_timer = random.randint(100, 300)
        
    def think(self):
        # Remove old thought
        if self.bubble:
            self.canvas.delete(self.bubble)
            self.canvas.delete(self.bubble_bg)
        
        # Random thought
        thought = random.choice(THOUGHTS[self.data["vibe"]])
        
        # Position bubble above creature
        bubble_x = self.x
        bubble_y = self.y - self.size - 20
        
        # Background for bubble
        self.bubble_bg = self.canvas.create_oval(
            bubble_x - 80, bubble_y - 25,
            bubble_x + 80, bubble_y + 25,
            fill=self.color, outline="black", width=2,
            tags="bubble"
        )
        
        self.bubble = self.canvas.create_text(
            bubble_x, bubble_y,
            text=thought,
            font=("Comic Sans MS", 10, "bold"),
            fill="black",
            width=140,
            tags="bubble"
        )
        
        # Schedule removal
        self.canvas.after(3000, lambda: self.clear_thought())
    
    def clear_thought(self):
        if self.bubble:
            self.canvas.delete(self.bubble)
            self.canvas.delete(self.bubble_bg)
            self.bubble = None
            self.bubble_bg = None
    
    def update(self):
        # Bounce off walls
        if self.x <= 50 or self.x >= 750:
            self.vx *= -1
        if self.y <= 50 or self.y >= 550:
            self.vy *= -1
        
        # Move
        self.x += self.vx
        self.y += self.vy
        
        # Random direction changes
        if random.random() < 0.02:
            self.vx += random.uniform(-1, 1)
            self.vy += random.uniform(-1, 1)
            # Limit speed
            speed = math.sqrt(self.vx**2 + self.vy**2)
            if speed > 5:
                self.vx = (self.vx / speed) * 5
                self.vy = (self.vy / speed) * 5
        
        # Update position
        self.canvas.coords(self.body, self.x, self.y)
        
        # Thinking timer
        self.thought_timer -= 1
        if self.thought_timer <= 0:
            self.think()
            self.thought_timer = random.randint(200, 500)

class ChaosApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎪 Desktop Chaos Creatures 🎪")
        self.root.geometry("800x700")
        self.root.configure(bg="#1a1a2e")
        
        # Title
        title = tk.Label(
            root,
            text="🎪 DESKTOP CHAOS CREATURES 🎪",
            font=("Arial", 20, "bold"),
            bg="#1a1a2e",
            fg="#FFD700"
        )
        title.pack(pady=10)
        
        subtitle = tk.Label(
            root,
            text="They live here now. Deal with it.",
            font=("Comic Sans MS", 12),
            bg="#1a1a2e",
            fg="#FF6B9D"
        )
        subtitle.pack()
        
        # Canvas
        self.canvas = tk.Canvas(
            root,
            width=800,
            height=600,
            bg="#0f3460",
            highlightthickness=0
        )
        self.canvas.pack(pady=10)
        
        # Control panel
        control_frame = tk.Frame(root, bg="#1a1a2e")
        control_frame.pack()
        
        add_btn = tk.Button(
            control_frame,
            text="🐙 SUMMON CREATURE",
            command=self.add_creature,
            font=("Arial", 12, "bold"),
            bg="#4ECDC4",
            fg="black",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        add_btn.pack(side=tk.LEFT, padx=5)
        
        chaos_btn = tk.Button(
            control_frame,
            text="⚡ CHAOS MODE",
            command=self.chaos_mode,
            font=("Arial", 12, "bold"),
            bg="#FF6B9D",
            fg="black",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        chaos_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            control_frame,
            text="🧹 BANISH ALL",
            command=self.clear_creatures,
            font=("Arial", 12, "bold"),
            bg="#F38181",
            fg="black",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Creatures list
        self.creatures = []
        
        # Start with 3 creatures
        for _ in range(3):
            self.add_creature()
        
        # Animation loop
        self.animate()
    
    def add_creature(self):
        creature_data = random.choice(CREATURES)
        creature = Creature(self.canvas, creature_data)
        self.creatures.append(creature)
        
        # Welcome message
        self.show_message(f"{creature.data['name']} has joined the chaos!", creature.color)
    
    def chaos_mode(self):
        # Make everything go CRAZY
        for creature in self.creatures:
            creature.vx *= 2
            creature.vy *= 2
            creature.think()
        self.show_message("🌪️ CHAOS UNLEASHED 🌪️", "#FF0000")
    
    def clear_creatures(self):
        for creature in self.creatures:
            self.canvas.delete(creature.body)
            if creature.bubble:
                self.canvas.delete(creature.bubble)
                self.canvas.delete(creature.bubble_bg)
        self.creatures = []
        self.show_message("The void consumes all...", "#666666")
    
    def show_message(self, text, color):
        msg = self.canvas.create_text(
            400, 300,
            text=text,
            font=("Arial", 24, "bold"),
            fill=color,
            tags="message"
        )
        self.root.after(2000, lambda: self.canvas.delete(msg))
    
    def animate(self):
        for creature in self.creatures:
            creature.update()
        self.root.after(30, self.animate)

def main():
    root = tk.Tk()
    app = ChaosApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()