//! plato-prompt-builder — Compose LLM prompts from tile search results.
//!
//! This crate is a PLATO fleet component that bridges tile search results and LLM
//! prompt construction. It handles context budgeting, score-based ranking, deadband
//! safety injection, and system message formatting — all with zero external dependencies.

/// A scored tile ready for context injection.
#[derive(Debug, Clone)]
pub struct ScoredTile {
    pub question: String,
    pub answer: String,
    pub domain: String,
    pub score: f64,
    pub use_count: u32,
}

/// Result of a deadband safety check (mirrors plato-kernel's DeadbandCheck).
pub struct DeadbandContext {
    pub passed: bool,
    pub violations: Vec<String>,
    pub recommended_channel: Option<String>,
}

/// Build a full LLM prompt from query and tile context.
///
/// Includes a `[PLATO CONTEXT]` section with the context block and a `[QUERY]` section
/// with the query. Context is truncated to fit within `max_tokens` (approximate: 4 chars
/// per token). If there are no tiles or the budget is exhausted before any tile fits,
/// the context block will be empty.
///
/// # Format
/// ```text
/// [PLATO CONTEXT]
/// {context block}
///
/// [QUERY]
/// {query}
/// ```
pub fn build_prompt(query: &str, tiles: &[ScoredTile], max_tokens: usize) -> String {
    // Reserve tokens for the fixed structural text:
    // "[PLATO CONTEXT]\n" (17) + "\n\n[QUERY]\n" (9) + query
    let structural = "[PLATO CONTEXT]\n\n\n[QUERY]\n";
    let structural_tokens = structural.len() / 4 + 1; // conservative ceiling
    let query_tokens = (query.len() + 3) / 4;
    let context_budget = max_tokens
        .saturating_sub(structural_tokens)
        .saturating_sub(query_tokens);

    let context = build_context(tiles, context_budget);

    format!("[PLATO CONTEXT]\n{}\n\n[QUERY]\n{}", context, query)
}

/// Build just the context block from tiles.
///
/// Tiles are sorted by score descending; tiles are included until the `max_tokens` budget
/// is exhausted. Each tile is formatted as:
/// ```text
/// Q: {question}
/// A: {answer}
/// ```
/// Returns an empty string when the budget is zero or no tiles are provided.
pub fn build_context(tiles: &[ScoredTile], max_tokens: usize) -> String {
    if max_tokens == 0 || tiles.is_empty() {
        return String::new();
    }

    // Sort tiles by score descending (clone indices to avoid mutating input).
    let mut sorted: Vec<&ScoredTile> = tiles.iter().collect();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut context = String::new();
    let mut chars_used: usize = 0;
    let max_chars = max_tokens * 4;

    for tile in sorted {
        let entry = format!("Q: {}\nA: {}\n", tile.question, tile.answer);
        if chars_used + entry.len() > max_chars {
            break;
        }
        context.push_str(&entry);
        chars_used += entry.len();
    }

    context
}

/// Score and sort tiles by relevance to query.
///
/// Score = number of query words that appear (case-insensitive) in `question + answer`
/// divided by the total number of query words.
///
/// If `query` is empty, all tiles are returned in their original order with score `0.0`.
pub fn rank_for_context<'a>(tiles: &'a [ScoredTile], query: &str) -> Vec<(f64, &'a ScoredTile)> {
    if query.trim().is_empty() {
        return tiles.iter().map(|t| (0.0, t)).collect();
    }

    let query_words: Vec<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let query_word_count = query_words.len() as f64;

    let mut scored: Vec<(f64, &ScoredTile)> = tiles
        .iter()
        .map(|tile| {
            let haystack = format!("{} {}", tile.question, tile.answer).to_lowercase();
            let overlap = query_words
                .iter()
                .filter(|w| haystack.contains(w.as_str()))
                .count() as f64;
            let score = overlap / query_word_count;
            (score, tile)
        })
        .collect();

    // Sort by score descending; stable so equal scores preserve original order.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Prepend safety context to a prompt when a deadband check flagged issues.
///
/// If `check.passed` is `true`, the prompt is returned unchanged.
/// If there are violations, prepends:
/// ```text
/// [SAFETY] Violations: {violations joined by ", "}
///
/// {prompt}
/// ```
pub fn inject_deadband(prompt: &str, check: &DeadbandContext) -> String {
    if check.passed {
        return prompt.to_string();
    }
    let violations = check.violations.join(", ");
    format!("[SAFETY] Violations: {}\n\n{}", violations, prompt)
}

/// Build a system message for an LLM with role and capabilities.
///
/// Format: `"You are {role}. Capabilities: {capabilities joined by ', '}."`
pub fn build_system_message(role: &str, capabilities: &[&str]) -> String {
    let caps = capabilities.join(", ");
    format!("You are {}. Capabilities: {}.", role, caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(question: &str, answer: &str, domain: &str, score: f64) -> ScoredTile {
        ScoredTile {
            question: question.to_string(),
            answer: answer.to_string(),
            domain: domain.to_string(),
            score,
            use_count: 0,
        }
    }

    // --- build_prompt ---

    #[test]
    fn test_build_prompt_includes_query_and_context() {
        let tiles = vec![make_tile("What is 2+2?", "4", "math", 0.9)];
        let prompt = build_prompt("Explain addition", &tiles, 1000);
        assert!(prompt.contains("[PLATO CONTEXT]"), "missing PLATO CONTEXT header");
        assert!(prompt.contains("[QUERY]"), "missing QUERY header");
        assert!(prompt.contains("Explain addition"), "missing query text");
        assert!(prompt.contains("What is 2+2?"), "missing tile question");
        assert!(prompt.contains("4"), "missing tile answer");
    }

    #[test]
    fn test_build_prompt_no_tiles() {
        let prompt = build_prompt("Hello?", &[], 1000);
        assert!(prompt.contains("[PLATO CONTEXT]"));
        assert!(prompt.contains("[QUERY]"));
        assert!(prompt.contains("Hello?"));
        // context block should be empty — just the headers
        let ctx_start = prompt.find("[PLATO CONTEXT]\n").unwrap() + "[PLATO CONTEXT]\n".len();
        let ctx_end = prompt.find("\n\n[QUERY]").unwrap();
        let context_block = &prompt[ctx_start..ctx_end];
        assert!(context_block.is_empty(), "expected empty context block, got: {:?}", context_block);
    }

    #[test]
    fn test_build_prompt_truncates_at_max_tokens() {
        // Each tile answer is 400 chars = 100 tokens.  With budget of 30 tokens we
        // expect no tile to fit in the context.
        let tiles: Vec<ScoredTile> = (0..5)
            .map(|i| make_tile("Q", &"x".repeat(400), &format!("d{}", i), 0.9 - i as f64 * 0.1))
            .collect();
        let prompt = build_prompt("small query", &tiles, 30);
        // context block should be empty because budget is too small
        let ctx_start = prompt.find("[PLATO CONTEXT]\n").unwrap() + "[PLATO CONTEXT]\n".len();
        let ctx_end = prompt.find("\n\n[QUERY]").unwrap();
        let context_block = &prompt[ctx_start..ctx_end];
        assert!(
            context_block.is_empty(),
            "expected context truncated to empty, got {} chars",
            context_block.len()
        );
    }

    // --- build_context ---

    #[test]
    fn test_build_context_respects_token_budget() {
        // Each tile entry "Q: Q\nA: XXXX...\n" is well over 100 tokens (400+ chars).
        let tiles: Vec<ScoredTile> = (0..5)
            .map(|i| make_tile("Q", &"x".repeat(400), "d", 0.9 - i as f64 * 0.1))
            .collect();
        let ctx = build_context(&tiles, 100); // 400 chars
        // At most one tile (which itself is ~406 chars > 400) should not fit
        // so the context should be empty.
        assert!(
            ctx.is_empty(),
            "expected empty context for tight budget, got {} chars",
            ctx.len()
        );
    }

    #[test]
    fn test_build_context_sorts_by_score() {
        let tiles = vec![
            make_tile("Low scoring question", "low answer", "d", 0.2),
            make_tile("High scoring question", "high answer", "d", 0.9),
            make_tile("Mid scoring question", "mid answer", "d", 0.5),
        ];
        let ctx = build_context(&tiles, 2000);
        let high_pos = ctx.find("high answer").unwrap();
        let mid_pos = ctx.find("mid answer").unwrap();
        let low_pos = ctx.find("low answer").unwrap();
        assert!(high_pos < mid_pos, "high score tile should appear before mid");
        assert!(mid_pos < low_pos, "mid score tile should appear before low");
    }

    #[test]
    fn test_build_context_zero_budget_returns_empty() {
        let tiles = vec![make_tile("Q", "A", "d", 0.9)];
        let ctx = build_context(&tiles, 0);
        assert!(ctx.is_empty(), "zero budget should yield empty context");
    }

    // --- rank_for_context ---

    #[test]
    fn test_rank_for_context_sorts_highest_score_first() {
        let tiles = vec![
            make_tile("the cat sat on the mat", "feline resting", "bio", 0.5),
            make_tile("dogs run fast", "canine speed", "bio", 0.5),
            make_tile("the cat is a mammal", "mammal facts", "bio", 0.5),
        ];
        let ranked = rank_for_context(&tiles, "cat mammal");
        // "the cat is a mammal" should score highest (both 'cat' and 'mammal' present)
        assert!(
            ranked[0].0 > ranked[1].0 || ranked[0].1.question.contains("mammal"),
            "highest ranked tile should have best keyword overlap"
        );
        // Scores should be non-increasing
        for window in ranked.windows(2) {
            assert!(
                window[0].0 >= window[1].0,
                "scores not sorted descending: {} < {}",
                window[0].0,
                window[1].0
            );
        }
    }

    #[test]
    fn test_rank_for_context_empty_query_returns_all_tiles() {
        let tiles = vec![
            make_tile("Q1", "A1", "d", 0.9),
            make_tile("Q2", "A2", "d", 0.5),
        ];
        let ranked = rank_for_context(&tiles, "");
        assert_eq!(ranked.len(), tiles.len(), "should return all tiles");
        for (score, _) in &ranked {
            assert_eq!(*score, 0.0, "all scores should be 0.0 for empty query");
        }
        // Original order preserved
        assert_eq!(ranked[0].1.question, "Q1");
        assert_eq!(ranked[1].1.question, "Q2");
    }

    // --- inject_deadband ---

    #[test]
    fn test_inject_deadband_passes_through_if_check_passed() {
        let check = DeadbandContext {
            passed: true,
            violations: vec!["some violation".to_string()],
            recommended_channel: None,
        };
        let original = "My prompt here.";
        let result = inject_deadband(original, &check);
        assert_eq!(result, original, "passed check should not modify prompt");
    }

    #[test]
    fn test_inject_deadband_prepends_safety_block_when_violations() {
        let check = DeadbandContext {
            passed: false,
            violations: vec!["overheating".to_string(), "out of bounds".to_string()],
            recommended_channel: Some("emergency".to_string()),
        };
        let result = inject_deadband("Do something.", &check);
        assert!(result.starts_with("[SAFETY]"), "should start with [SAFETY]");
        assert!(result.contains("overheating"), "should include first violation");
        assert!(result.contains("out of bounds"), "should include second violation");
        assert!(result.contains("Do something."), "original prompt should be preserved");
    }

    // --- build_system_message ---

    #[test]
    fn test_build_system_message_formats_correctly() {
        let msg = build_system_message("OracleBot", &["search", "summarize", "rank"]);
        assert_eq!(
            msg,
            "You are OracleBot. Capabilities: search, summarize, rank."
        );
    }

    #[test]
    fn test_build_system_message_with_empty_capabilities() {
        let msg = build_system_message("BasicBot", &[]);
        assert_eq!(msg, "You are BasicBot. Capabilities: .");
    }
}
