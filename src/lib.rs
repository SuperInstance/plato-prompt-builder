//! plato-prompt-builder — Compose LLM prompts from tile search results
//!
//! Build context windows from tiles, inject deadband safety, compose system messages.
//! The invisible layer between "user asks question" and "LLM gets prompt."
//!
//! ```rust
//! let tiles = vec![scored_tile("pyth", "a²+b²=c²", 0.9), ...];
//! let ctx = build_context(&tiles, 500);
//! let prompt = build_prompt("What is the theorem?", &tiles, 1000);
//! ```

/// A scored tile result (input to prompt builder).
#[derive(Debug, Clone)]
pub struct ScoredTile {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub score: f64,
    pub confidence: f64,
}

/// Structured prompt with metadata.
#[derive(Debug, Clone)]
pub struct BuiltPrompt {
    pub system: String,
    pub context: String,
    pub user: String,
    pub total_tokens: usize,
    pub context_tokens: usize,
    pub tile_count: usize,
    pub truncated: bool,
}

/// Approximate token count (1 token ≈ 4 chars for English).
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Build context block from scored tiles, respecting token budget.
pub fn build_context(tiles: &[ScoredTile], max_tokens: usize) -> String {
    if tiles.is_empty() || max_tokens == 0 { return String::new(); }
    let mut context = String::new();
    let mut tokens_used = 0;
    let header = "--- Knowledge Base ---\n";
    tokens_used += estimate_tokens(header);
    context.push_str(header);

    for tile in tiles {
        let entry = format!("[{} ({:.0}%)] Q: {} A: {}\n",
            tile.domain, tile.confidence * 100.0, tile.question, tile.answer);
        let entry_tokens = estimate_tokens(&entry);
        if tokens_used + entry_tokens > max_tokens {
            context.push_str("[... additional context truncated ...]\n");
            return context;
        }
        context.push_str(&entry);
        tokens_used += entry_tokens;
    }
    context
}

/// Build a full prompt: system message + context + user query.
pub fn build_prompt(query: &str, tiles: &[ScoredTile], max_tokens: usize) -> BuiltPrompt {
    let system = "You are PLATO, a knowledge assistant. Answer questions using the provided context. If the context doesn't contain the answer, say so clearly. Be concise and accurate.".to_string();
    let mut remaining = max_tokens.saturating_sub(estimate_tokens(&system));
    remaining = remaining.saturating_sub(estimate_tokens(query));

    let context = build_context(tiles, remaining);
    let context_tokens = estimate_tokens(&context);

    let mut full = String::new();
    full.push_str(&system);
    full.push('\n');
    full.push_str(&context);
    full.push_str("User: ");
    full.push_str(query);
    full.push('\n');
    full.push_str("Answer: ");

    let truncated = context.contains("truncated");
    BuiltPrompt {
        system, context, user: query.to_string(),
        total_tokens: estimate_tokens(&full), context_tokens,
        tile_count: tiles.len(), truncated,
    }
}

/// Build context with per-tile token budget (fair allocation).
pub fn build_balanced_context(tiles: &[ScoredTile], max_tokens: usize) -> String {
    if tiles.is_empty() { return String::new(); }
    let per_tile = max_tokens / tiles.len().max(1);
    let mut context = String::new();
    for tile in tiles {
        let t = tile;
        context.push_str(&build_context(std::slice::from_ref(t), per_tile));
    }
    context
}

/// Inject deadband safety context before a prompt.
pub fn inject_deadband(prompt: &str, active_negatives: &[String]) -> String {
    if active_negatives.is_empty() { return prompt.to_string(); }
    let mut safety = String::from("SAFETY CONSTRAINTS (do not violate):\n");
    for neg in active_negatives {
        safety.push_str(&format!("- NEVER: {}\n", neg));
    }
    safety.push('\n');
    format!("{}{}", safety, prompt)
}

/// Build a system message for a specific role.
pub fn build_system_message(role: &str, capabilities: &[&str]) -> String {
    let mut msg = format!("You are {}, a PLATO knowledge assistant.\n\n", role);
    msg.push_str("Capabilities:\n");
    for cap in capabilities {
        msg.push_str(&format!("- {}\n", cap));
    }
    msg.push_str("\nRules:\n");
    msg.push_str("- Answer from the provided context when possible.\n");
    msg.push_str("- If context is insufficient, say so clearly.\n");
    msg.push_str("- Be concise. Be accurate. No hallucination.\n");
    msg
}

/// Extract just the relevant answer snippets for a query.
pub fn extract_relevant_snippets(tiles: &[ScoredTile], max_snippets: usize) -> Vec<String> {
    tiles.iter().take(max_snippets).map(|t| {
        format!("[{}] {}", t.domain, t.answer)
    }).collect()
}

/// Truncate text to fit token budget, respecting word boundaries.
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens * 4;
    if text.len() <= max_chars { return text.to_string(); }
    let truncated = &text[..max_chars];
    // Find last space to avoid cutting words
    if let Some(space) = truncated.rfind(' ') {
        format!("{}...", &truncated[..space])
    } else {
        format!("{}...", &truncated[..max_chars])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(id: &str, q: &str, a: &str, domain: &str, score: f64) -> ScoredTile {
        ScoredTile { id: id.to_string(), question: q.to_string(), answer: a.to_string(),
                     domain: domain.to_string(), score, confidence: 0.9 }
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 2); // 11 chars / 4 = 2
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_build_context_basic() {
        let tiles = vec![
            tile("t1", "What is pi?", "3.14159", "math", 0.9),
        ];
        let ctx = build_context(&tiles, 500);
        assert!(ctx.contains("Knowledge Base"));
        assert!(ctx.contains("3.14159"));
    }

    #[test]
    fn test_build_context_empty() {
        let ctx = build_context(&[], 500);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_build_context_truncation() {
        let tiles: Vec<ScoredTile> = (0..10).map(|i| {
            tile(&format!("t{}", i), &format!("Q{}", i), &"x".repeat(200), "math", 0.9)
        }).collect();
        let ctx = build_context(&tiles, 50); // very small budget
        assert!(ctx.contains("truncated"));
    }

    #[test]
    fn test_build_prompt_structure() {
        let tiles = vec![tile("t1", "Q", "A", "math", 0.9)];
        let prompt = build_prompt("What?", &tiles, 1000);
        assert!(prompt.system.contains("PLATO"));
        assert!(prompt.context.contains("Knowledge Base"));
        assert!(prompt.user == "What?");
        assert!(prompt.total_tokens > 0);
        assert!(!prompt.truncated);
    }

    #[test]
    fn test_build_prompt_tile_count() {
        let tiles = vec![tile("t1", "Q", "A", "m", 0.9), tile("t2", "Q2", "A2", "m", 0.8)];
        let prompt = build_prompt("q", &tiles, 1000);
        assert_eq!(prompt.tile_count, 2);
    }

    #[test]
    fn test_build_balanced_context() {
        let tiles = vec![tile("t1", "Q1", "A1", "m", 0.9), tile("t2", "Q2", "A2", "m", 0.8)];
        let ctx = build_balanced_context(&tiles, 200);
        assert!(ctx.contains("A1"));
        assert!(ctx.contains("A2"));
    }

    #[test]
    fn test_inject_deadband() {
        let tiles = vec!["rm -rf".to_string(), "DROP TABLE".to_string()];
        let prompt = inject_deadband("User: hello", &tiles);
        assert!(prompt.contains("SAFETY CONSTRAINTS"));
        assert!(prompt.contains("NEVER: rm -rf"));
        assert!(prompt.contains("NEVER: DROP TABLE"));
    }

    #[test]
    fn test_inject_deadband_empty() {
        let prompt = inject_deadband("hello", &[]);
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn test_build_system_message() {
        let msg = build_system_message("MathTutor", &["algebra", "calculus"]);
        assert!(msg.contains("MathTutor"));
        assert!(msg.contains("algebra"));
        assert!(msg.contains("calculus"));
        assert!(msg.contains("No hallucination"));
    }

    #[test]
    fn test_extract_relevant_snippets() {
        let tiles = vec![
            tile("t1", "Q", "Answer one", "math", 0.9),
            tile("t2", "Q", "Answer two", "physics", 0.8),
            tile("t3", "Q", "Answer three", "chem", 0.7),
        ];
        let snippets = extract_relevant_snippets(&tiles, 2);
        assert_eq!(snippets.len(), 2);
        assert!(snippets[0].contains("Answer one"));
    }

    #[test]
    fn test_truncate_to_tokens() {
        let long = "a".repeat(1000);
        let truncated = truncate_to_tokens(&long, 10);
        assert!(truncated.len() < 1000);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_no_truncate() {
        let short = "hello world";
        assert_eq!(truncate_to_tokens(short, 100), "hello world");
    }

    #[test]
    fn test_build_prompt_zero_budget() {
        let tiles = vec![tile("t1", "Q", "A", "m", 0.9)];
        let prompt = build_prompt("q", &tiles, 0);
        // Should still work, just no context
        assert!(prompt.context.is_empty());
    }
}
