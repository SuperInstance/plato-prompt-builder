/// A scored tile for prompt building — minimal struct.
pub struct ScoredTile {
    pub id: usize,
    pub content: String,
    pub score: f64,
}

/// Build a full LLM prompt from tile search results.
/// Format: system context header, then each tile content, then the query.
/// Respects max_tokens (approximate: 4 chars per token).
pub fn build_prompt(query: &str, tiles: &[ScoredTile], max_tokens: usize) -> String {
    let header = "=== Context ===\n";
    let footer = format!("\n=== Query ===\n{}", query);
    let overhead = header.len() + footer.len();
    let max_chars = max_tokens * 4;

    let context_budget = if max_chars > overhead {
        max_chars - overhead
    } else {
        0
    };

    let context = build_context(tiles, context_budget / 4);

    format!("{}{}{}", header, context, footer)
}

/// Build context block from tiles, truncating to fit max_tokens (4 chars/token).
/// Returns concatenated tile contents separated by newlines, truncated.
pub fn build_context(tiles: &[ScoredTile], max_tokens: usize) -> String {
    if tiles.is_empty() {
        return String::new();
    }

    let max_chars = max_tokens * 4;
    let mut result = String::new();

    for (i, tile) in tiles.iter().enumerate() {
        let sep = if i == 0 { "" } else { "\n" };
        let candidate = format!("{}{}", sep, tile.content);

        if result.len() + candidate.len() <= max_chars {
            result.push_str(&candidate);
        } else {
            // Truncate to fit
            let remaining = max_chars.saturating_sub(result.len() + sep.len());
            if remaining > 0 {
                result.push_str(sep);
                let truncated: String = tile.content.chars().take(remaining).collect();
                result.push_str(&truncated);
            }
            break;
        }
    }

    result
}

/// Score and sort tiles by relevance to query (simple word overlap score).
/// Returns tiles sorted descending by score.
pub fn rank_for_context(mut tiles: Vec<ScoredTile>, query: &str) -> Vec<ScoredTile> {
    let query_words: Vec<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if query_words.is_empty() {
        return tiles;
    }

    for tile in tiles.iter_mut() {
        let content_lower = tile.content.to_lowercase();
        let content_words: Vec<&str> = content_lower.split_whitespace().collect();
        let overlap = query_words
            .iter()
            .filter(|qw| content_words.contains(&qw.as_str()))
            .count();
        tile.score = overlap as f64;
    }

    tiles.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    tiles
}

/// Prepend deadband safety context to a prompt if check failed.
/// If violations is non-empty, prepend "[SAFETY CONTEXT]: {violations joined}" to prompt.
pub fn inject_deadband(prompt: &str, violations: &[String]) -> String {
    if violations.is_empty() {
        return prompt.to_string();
    }

    let safety_header = format!("[SAFETY CONTEXT]: {}", violations.join(", "));
    format!("{}\n{}", safety_header, prompt)
}

/// Build a system message from role and capabilities list.
/// Format: "You are {role}. Capabilities: {capabilities joined by ', '}."
pub fn build_system_message(role: &str, capabilities: &[&str]) -> String {
    let caps = capabilities.join(", ");
    format!("You are {}. Capabilities: {}.", role, caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(id: usize, content: &str, score: f64) -> ScoredTile {
        ScoredTile {
            id,
            content: content.to_string(),
            score,
        }
    }

    // 1. build_prompt with empty tiles returns valid string with query
    #[test]
    fn test_build_prompt_empty_tiles_contains_query() {
        let result = build_prompt("what is rust?", &[], 1000);
        assert!(result.contains("what is rust?"));
        assert!(result.contains("=== Query ==="));
        assert!(result.contains("=== Context ==="));
    }

    // 2. build_prompt respects max_tokens limit
    #[test]
    fn test_build_prompt_respects_max_tokens() {
        let tiles = vec![
            make_tile(1, &"a".repeat(200), 1.0),
            make_tile(2, &"b".repeat(200), 0.5),
        ];
        let max_tokens = 50; // 200 chars total budget
        let result = build_prompt("query", &tiles, max_tokens);
        // result should not wildly exceed the token budget
        assert!(result.len() <= max_tokens * 4 + 60);
    }

    // 3. build_context truncates to fit max_tokens
    #[test]
    fn test_build_context_truncates() {
        let tiles = vec![
            make_tile(1, "hello world this is a long string of content", 1.0),
        ];
        let result = build_context(&tiles, 3); // 3 tokens = 12 chars
        assert!(result.len() <= 12);
        assert!(!result.is_empty());
    }

    // 4. rank_for_context sorts by word overlap descending
    #[test]
    fn test_rank_for_context_sorts_descending() {
        let tiles = vec![
            make_tile(1, "cat sat on mat", 0.0),
            make_tile(2, "the rust programming language is fast and safe", 0.0),
            make_tile(3, "rust is safe", 0.0),
        ];
        let ranked = rank_for_context(tiles, "rust safe");
        // tile 1 has 0 overlaps — should be last
        assert_eq!(ranked.last().unwrap().id, 1);
        // top tiles should have higher scores than last
        assert!(ranked[0].score >= ranked[ranked.len() - 1].score);
    }

    // 5. rank_for_context with empty query returns tiles unchanged order
    #[test]
    fn test_rank_for_context_empty_query_unchanged() {
        let tiles = vec![
            make_tile(10, "apple", 0.0),
            make_tile(20, "banana", 0.0),
            make_tile(30, "cherry", 0.0),
        ];
        let ranked = rank_for_context(tiles, "");
        assert_eq!(ranked[0].id, 10);
        assert_eq!(ranked[1].id, 20);
        assert_eq!(ranked[2].id, 30);
    }

    // 6. inject_deadband with no violations returns prompt unchanged
    #[test]
    fn test_inject_deadband_no_violations() {
        let prompt = "This is the prompt.";
        let result = inject_deadband(prompt, &[]);
        assert_eq!(result, prompt);
    }

    // 7. inject_deadband with violations prepends safety context
    #[test]
    fn test_inject_deadband_with_violations() {
        let prompt = "Do the thing.";
        let violations = vec!["limit exceeded".to_string(), "rate too high".to_string()];
        let result = inject_deadband(prompt, &violations);
        assert!(result.starts_with("[SAFETY CONTEXT]: limit exceeded, rate too high"));
        assert!(result.contains("Do the thing."));
    }

    // 8. build_system_message formats correctly
    #[test]
    fn test_build_system_message_format() {
        let msg = build_system_message("Oracle", &["search", "summarize", "plan"]);
        assert_eq!(msg, "You are Oracle. Capabilities: search, summarize, plan.");
    }

    // 9. build_system_message with empty capabilities
    #[test]
    fn test_build_system_message_empty_capabilities() {
        let msg = build_system_message("Forgemaster", &[]);
        assert_eq!(msg, "You are Forgemaster. Capabilities: .");
    }

    // 10. build_context with empty tiles returns empty string
    #[test]
    fn test_build_context_empty_tiles() {
        let result = build_context(&[], 1000);
        assert_eq!(result, "");
    }

    // 11. build_prompt with multiple tiles includes top tiles
    #[test]
    fn test_build_prompt_includes_top_tiles() {
        let tiles = vec![
            make_tile(1, "first tile content", 1.0),
            make_tile(2, "second tile content", 0.5),
        ];
        let result = build_prompt("my query", &tiles, 1000);
        assert!(result.contains("first tile content"));
        assert!(result.contains("second tile content"));
        assert!(result.contains("my query"));
    }

    // 12. rank_for_context uses case-insensitive word overlap
    #[test]
    fn test_rank_for_context_case_insensitive() {
        let tiles = vec![
            make_tile(1, "Rust is Great", 0.0),
            make_tile(2, "python is nice", 0.0),
        ];
        let ranked = rank_for_context(tiles, "rust great");
        // tile 1 should rank first due to case-insensitive match
        assert_eq!(ranked[0].id, 1);
        assert!(ranked[0].score > ranked[1].score);
    }

    // 13. build_context joins multiple tiles with newlines
    #[test]
    fn test_build_context_multiple_tiles_newline_separated() {
        let tiles = vec![
            make_tile(1, "first", 1.0),
            make_tile(2, "second", 0.5),
        ];
        let result = build_context(&tiles, 1000);
        assert_eq!(result, "first\nsecond");
    }

    // 14. inject_deadband single violation formats correctly
    #[test]
    fn test_inject_deadband_single_violation() {
        let prompt = "Execute plan.";
        let violations = vec!["threshold breached".to_string()];
        let result = inject_deadband(prompt, &violations);
        assert!(result.starts_with("[SAFETY CONTEXT]: threshold breached\n"));
        assert!(result.ends_with("Execute plan."));
    }
}
