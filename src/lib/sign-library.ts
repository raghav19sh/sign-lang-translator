export type SignEntry = {
  id: string;
  phrase: string;
  gloss: string[];
  category: "Conversation" | "Everyday" | "Help";
  note: string;
  demoUrl?: string;
};

// These entries are a starter gloss-practice vocabulary, not an ASL translation engine.
// Add only demonstrations that your team has permission to use and has reviewed with ASL users.
export const signLibrary: SignEntry[] = [
  {
    id: "hello",
    phrase: "Hello",
    gloss: ["HELLO"],
    category: "Conversation",
    note: "Greeting. Usage and regional variants can differ.",
  },
  {
    id: "thank-you",
    phrase: "Thank you",
    gloss: ["THANK-YOU"],
    category: "Conversation",
    note: "A common expression of thanks.",
  },
  {
    id: "please",
    phrase: "Please",
    gloss: ["PLEASE"],
    category: "Everyday",
    note: "Polite request marker; context matters.",
  },
  {
    id: "help",
    phrase: "Help",
    gloss: ["HELP"],
    category: "Help",
    note: "Useful vocabulary for assistance. Do not use this prototype for emergencies.",
  },
  {
    id: "yes",
    phrase: "Yes",
    gloss: ["YES"],
    category: "Everyday",
    note: "Affirmative response.",
  },
  {
    id: "no",
    phrase: "No",
    gloss: ["NO"],
    category: "Everyday",
    note: "Negative response.",
  },
  {
    id: "good-morning",
    phrase: "Good morning",
    gloss: ["GOOD", "MORNING"],
    category: "Conversation",
    note: "Phrase-level practice prompt. Validate gloss and ordering before publication.",
  },
  {
    id: "how-are-you",
    phrase: "How are you?",
    gloss: ["YOU", "HOW"],
    category: "Conversation",
    note: "Simplified gloss cue only; not a complete grammatical translation.",
  },
  {
    id: "i-need-help",
    phrase: "I need help",
    gloss: ["I", "NEED", "HELP"],
    category: "Help",
    note: "Practice sequence. Have an ASL reviewer validate before relying on it.",
  },
  {
    id: "my-name",
    phrase: "My name is…",
    gloss: ["MY", "NAME"],
    category: "Conversation",
    note: "Use fingerspelling for a name when appropriate; context and grammar matter.",
  },
];

export function findSignByPhrase(phrase: string): SignEntry | undefined {
  const normalized = phrase.trim().toLocaleLowerCase().replace(/[?.!,]/g, "").replace(/\s+/g, " ");
  return signLibrary.find((entry) => {
    const candidate = entry.phrase.toLocaleLowerCase().replace(/[?.!,…]/g, "").replace(/\s+/g, " ");
    return candidate === normalized;
  });
}
