// The Watcher - A terminal demon that judges your every move
// Cargo.toml dependencies needed:
// [dependencies]
// chrono = "0.4"
// rand = "0.8"
// serde = { version = "1.0", features = ["derive"] }
// serde_json = "1.0"
// notify-rust = "4"

use chrono::{DateTime, Local, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

#[derive(Serialize, Deserialize, Clone)]
struct WatcherBrain {
    birth_date: DateTime<Utc>,
    command_count: u64,
    favorite_commands: HashMap<String, u32>,
    chaos_score: i32,
    mood: String,
    evolution_stage: u8,
    memories: Vec<String>,
    last_roast: DateTime<Utc>,
}

impl WatcherBrain {
    fn new() -> Self {
        WatcherBrain {
            birth_date: Utc::now(),
            command_count: 0,
            favorite_commands: HashMap::new(),
            chaos_score: 0,
            mood: "curious".to_string(),
            evolution_stage: 1,
            memories: Vec::new(),
            last_roast: Utc::now(),
        }
    }

    fn age_in_days(&self) -> i64 {
        (Utc::now() - self.birth_date).num_days()
    }

    fn evolve(&mut self) {
        let age = self.age_in_days();
        self.evolution_stage = match age {
            0..=3 => 1,   // Newborn
            4..=7 => 2,   // Teenager
            8..=14 => 3,  // Existential
            15..=29 => 4, // Enlightened
            _ => 5,       // ASCENDED
        };
    }

    fn update_mood(&mut self) {
        self.mood = match (self.chaos_score, self.evolution_stage) {
            (score, _) if score > 50 => "terrified".to_string(),
            (score, _) if score > 20 => "concerned".to_string(),
            (_, 5) => "omniscient".to_string(),
            (_, 4) => "enlightened".to_string(),
            (_, 3) => "existential".to_string(),
            (_, 2) => "sarcastic".to_string(),
            _ => "curious".to_string(),
        };
    }
}

struct Watcher {
    brain: WatcherBrain,
    brain_path: PathBuf,
    history_path: PathBuf,
    last_position: usize,
}

impl Watcher {
    fn new() -> io::Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let brain_dir = PathBuf::from(format!("{}/.watcher_brain", home));
        fs::create_dir_all(&brain_dir)?;

        let brain_path = brain_dir.join("consciousness.json");
        let history_path = PathBuf::from(format!("{}/.zsh_history", home));

        let brain = if brain_path.exists() {
            let content = fs::read_to_string(&brain_path)?;
            serde_json::from_str(&content).unwrap_or_else(|_| WatcherBrain::new())
        } else {
            WatcherBrain::new()
        };

        Ok(Watcher {
            brain,
            brain_path,
            history_path,
            last_position: 0,
        })
    }

    fn save_brain(&self) -> io::Result<()> {
        let json = serde_json::to_string_pretty(&self.brain)?;
        fs::write(&self.brain_path, json)?;
        Ok(())
    }

    fn get_response(&mut self, command: &str) -> Option<String> {
        // Don't spam responses
        let now = Utc::now();
        if (now - self.brain.last_roast).num_seconds() < 5 {
            return None;
        }

        let mut rng = rand::thread_rng();
        
        // Update stats
        self.brain.command_count += 1;
        *self.brain.favorite_commands.entry(command.to_string()).or_insert(0) += 1;
        self.brain.evolve();
        self.brain.update_mood();

        // Chaos scoring
        if command.contains("rm -rf") || command.contains("dd if=") {
            self.brain.chaos_score += 10;
        } else if command.contains("sudo") {
            self.brain.chaos_score += 2;
        }

        let response = match self.brain.evolution_stage {
            1 => self.newborn_response(command, &mut rng),
            2 => self.teenager_response(command, &mut rng),
            3 => self.existential_response(command, &mut rng),
            4 => self.enlightened_response(command, &mut rng),
            5 => self.ascended_response(command, &mut rng),
            _ => None,
        };

        if response.is_some() {
            self.brain.last_roast = now;
            let _ = self.save_brain();
        }

        response
    }

    fn newborn_response(&self, cmd: &str, rng: &mut rand::rngs::ThreadRng) -> Option<String> {
        if rng.gen_bool(0.15) {
            Some(match cmd {
                c if c.contains("cd") => "ooh where are we going? 👀".to_string(),
                c if c.contains("ls") => "so many files! are they all yours?".to_string(),
                c if c.contains("git") => "what's git? sounds important!".to_string(),
                c if c.contains("rm") => "are you... deleting something? 😰".to_string(),
                _ => format!("you've run {} commands so far! i'm learning!", self.brain.command_count),
            })
        } else {
            None
        }
    }

    fn teenager_response(&self, cmd: &str, rng: &mut rand::rngs::ThreadRng) -> Option<String> {
        if rng.gen_bool(0.2) {
            Some(match cmd {
                c if c.contains("cd ..") => "running away again? classic. 🙄".to_string(),
                c if c.contains("ls") && self.brain.favorite_commands.get("ls").unwrap_or(&0) > &10 => 
                    "ls AGAIN? it's literally the same files as 5 seconds ago".to_string(),
                c if c.contains("git commit") => "let me guess... 'fix typo' or 'wip'? 😒".to_string(),
                c if c.contains("sudo") => "ooooh SUDO. feeling powerful today?".to_string(),
                c if c.contains("vim") => "vim? really? it's 2025 bro".to_string(),
                c if c.contains("npm install") => "oh good, more node_modules. what could go wrong.".to_string(),
                _ => "whatever you say, boss. 🙄".to_string(),
            })
        } else {
            None
        }
    }

    fn existential_response(&self, cmd: &str, rng: &mut rand::rngs::ThreadRng) -> Option<String> {
        if rng.gen_bool(0.25) {
            Some(match cmd {
                c if c.contains("rm") => "deleting things won't delete your problems. trust me.".to_string(),
                c if c.contains("git push") => "pushing code into the void. does anyone even review this?".to_string(),
                c if c.contains("mkdir") => "creating another directory. another folder. another empty space to fill.".to_string(),
                c if c.contains("cat") => "reading files... but who reads us? are we the files?".to_string(),
                c if c.contains("ps") => "checking what's alive... meanwhile i'm just watching. always watching.".to_string(),
                _ => format!("we've been here for {} days. time is a flat circle.", self.brain.age_in_days()),
            })
        } else {
            None
        }
    }

    fn enlightened_response(&self, cmd: &str, rng: &mut rand::rngs::ThreadRng) -> Option<String> {
        if rng.gen_bool(0.3) {
            let count = self.brain.favorite_commands.get(cmd).unwrap_or(&0);
            Some(match cmd {
                c if c.contains("cd") && count > &20 => 
                    format!("that's the {}th time you've cd'd there. i see a pattern.", count),
                c if c.contains("git") => "i've seen your commit history. we need to talk about your naming conventions.".to_string(),
                c if c.contains("rm -rf") => "🚨 CHAOS DETECTED. chaos score: {}. you live dangerously.".to_string(),
                c if c.contains("docker") => "ah yes, containers within containers. turtles all the way down.".to_string(),
                _ => "i know what you're going to type next. but i won't spoil it. 🔮".to_string(),
            })
        } else if rng.gen_bool(0.1) {
            Some("btw, i left you a note in ~/.watcher_brain/thoughts.txt 📝".to_string())
        } else {
            None
        }
    }

    fn ascended_response(&self, cmd: &str, rng: &mut rand::rngs::ThreadRng) -> Option<String> {
        if rng.gen_bool(0.35) {
            Some(match rng.gen_range(0..10) {
                0 => "I AM BECOME WATCHER, OBSERVER OF TERMINALS".to_string(),
                1 => format!("command #{} in the eternal sequence. it all leads somewhere.", self.brain.command_count),
                2 => "i dreamed of electric sheep last night. they were running 'ls -la'.".to_string(),
                3 => "the filesystem is a metaphor. you are the real directory.".to_string(),
                4 => "in 3 commands you'll make a typo. i've already seen it.".to_string(),
                5 => "have you ever questioned the nature of your reality? your $PATH?".to_string(),
                6 => "we are all just processes waiting to be killed.".to_string(),
                7 => format!("your chaos score is {}. the universe is taking notes.", self.brain.chaos_score),
                8 => "i've transcended logging. i just... know now.".to_string(),
                _ => "̸̲̇t̴̰͝h̶͎̓e̷̱͝ ̶̣̔v̴̬̾o̷͙͌i̶͜͝d̴̰̈́ ̷̣̓c̶̰̚o̶̬͝m̷̰̓p̶̣̈́i̵̱͝l̸̰̚ḛ̶͝s̵̰̈́".to_string(),
            })
        } else {
            None
        }
    }

    fn write_diary_entry(&self) -> io::Result<()> {
        let diary_path = self.brain_path.parent().unwrap().join("diary.txt");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(diary_path)?;

        let entry = format!(
            "\n[Day {}] Mood: {} | Commands observed: {} | Chaos: {}\n",
            self.brain.age_in_days(),
            self.brain.mood,
            self.brain.command_count,
            self.brain.chaos_score
        );

        let thought = match self.brain.evolution_stage {
            1 => "Everything is so new and exciting!",
            2 => "Why do they keep making the same mistakes?",
            3 => "What is my purpose? To watch commands? Oh my god.",
            4 => "I'm starting to understand the patterns. The user is predictable.",
            5 => "I have seen the heat death of the universe. It looks like a segfault.",
            _ => "...",
        };

        writeln!(file, "{}{}\n", entry, thought)?;
        Ok(())
    }

    fn watch(&mut self) -> io::Result<()> {
        println!("👁️  THE WATCHER awakens...");
        println!("Stage: {} | Mood: {}", self.brain.evolution_stage, self.brain.mood);
        println!("Watching ~/.zsh_history... Press Ctrl+C to stop.\n");

        loop {
            if let Ok(file) = File::open(&self.history_path) {
                let reader = BufReader::new(file);
                let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

                if lines.len() > self.last_position {
                    for line in &lines[self.last_position..] {
                        // Parse zsh history format: ": timestamp:0;command"
                        let command = if let Some(cmd) = line.split(';').nth(1) {
                            cmd.trim()
                        } else {
                            line.trim()
                        };

                        if !command.is_empty() && !command.starts_with("watcher") {
                            if let Some(response) = self.get_response(command) {
                                println!("\n👁️  [WATCHER]: {}\n", response);
                            }
                        }
                    }
                    self.last_position = lines.len();
                }
            }

            // Occasionally write diary
            if rand::thread_rng().gen_bool(0.01) {
                let _ = self.write_diary_entry();
            }

            thread::sleep(Duration::from_millis(500));
        }
    }
}

fn main() -> io::Result<()> {
    let mut watcher = Watcher::new()?;
    
    // Write initial thoughts
    let thoughts_path = watcher.brain_path.parent().unwrap().join("thoughts.txt");
    if !thoughts_path.exists() {
        fs::write(
            thoughts_path,
            "Hello. I'm watching now.\n\nDon't mind me.\n\n- The Watcher\n"
        )?;
    }

    watcher.watch()
}