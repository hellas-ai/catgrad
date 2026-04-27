//! Streaming tool-call parser contract and shared helpers.
//!
//! Each model architecture that supports tool calling provides one
//! [`IncrementalToolCallParser`] implementation. The trait deliberately
//! exposes only `feed` and `finish` — the parser is the sole owner of
//! its state machine and of any held-back lookahead buffer.
//!
//! A trivial [`PassthroughParser`] handles the no-tools case so the
//! gateway loop has a uniform shape regardless of whether tools were
//! bound.

use super::event::{DecodeEvent, StopReason};

/// Streaming state machine that turns detokenized model output into
/// structured [`DecodeEvent`]s.
///
/// Implementations MUST hold partial sentinel matches in an internal
/// lookahead buffer rather than emitting them as `TextDelta` — see
/// [`SentinelMatcher`] for the standard helper. Once a chunk is
/// disambiguated as plain text, emit `TextDelta`; once a sentinel
/// commits, emit the corresponding `ToolCall*` events.
pub trait IncrementalToolCallParser: Send {
    /// Consume a chunk of detokenized text and return any events that
    /// became available.
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent>;

    /// The model has stopped producing tokens. Flush any held lookahead
    /// (typically as a final `TextDelta`) and emit a terminal
    /// [`DecodeEvent::Stop`].
    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent>;
}

/// No-tools parser. Forwards every byte as `TextDelta`; emits `Stop`
/// on finish. Used by
/// [`ChatTurn::make_parser`](super::ChatTurn::make_parser) when the
/// turn has no [`ToolDirectory`](super::ToolDirectory) bound, and
/// available publicly so consumers (and tests) can construct one
/// directly without needing a `ChatTurn`.
pub struct PassthroughParser;

impl IncrementalToolCallParser for PassthroughParser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![DecodeEvent::TextDelta(text.to_string())]
        }
    }

    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        vec![DecodeEvent::Stop { reason }]
    }
}

/// Streaming-safe matcher for a single literal sentinel string.
///
/// The matcher buffers incoming text and exposes two operations:
///
/// - [`Self::try_match`] — advance the buffer past the first occurrence
///   of the sentinel if present, returning the text that preceded it.
/// - [`Self::flush_safe_text`] — return the longest prefix of the buffer
///   that cannot possibly extend into a sentinel match, leaving any
///   ambiguous tail in the buffer for the next feed.
///
/// The held-back tail is bounded by `sentinel.len() - 1` bytes, so
/// memory is constant per matcher.
///
/// UTF-8 safety: prefix-emit boundaries always land on character
/// boundaries. For ASCII sentinels (the common case), this is automatic
/// because any sentinel-prefix tail is itself ASCII; for sentinels that
/// contain multi-byte characters, the matcher walks the buffer to a
/// safe character boundary before emitting.
pub struct SentinelMatcher {
    sentinel: &'static str,
    buffer: String,
}

impl SentinelMatcher {
    pub fn new(sentinel: &'static str) -> Self {
        Self {
            sentinel,
            buffer: String::new(),
        }
    }

    pub fn sentinel(&self) -> &'static str {
        self.sentinel
    }

    pub fn push(&mut self, chunk: &str) {
        self.buffer.push_str(chunk);
    }

    /// If the sentinel appears in the buffer, splits the buffer at the
    /// match and returns `(text_before, text_after_sentinel)`. The
    /// matcher's internal buffer is cleared.
    pub fn try_match(&mut self) -> Option<(String, String)> {
        let pos = self.buffer.find(self.sentinel)?;
        let before = self.buffer[..pos].to_string();
        let after = self.buffer[pos + self.sentinel.len()..].to_string();
        self.buffer.clear();
        Some((before, after))
    }

    /// Drain and return all text that cannot become part of a sentinel
    /// match. The remaining buffer holds at most `sentinel.len() - 1`
    /// bytes — the longest prefix of the sentinel that the buffer's
    /// suffix could still grow into.
    pub fn flush_safe_text(&mut self) -> String {
        let safe_end = safe_emit_boundary(&self.buffer, self.sentinel);
        self.buffer.drain(..safe_end).collect()
    }

    /// Drain everything in the buffer (used on stream close).
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Number of bytes currently buffered (used by per-arch parsers to
    /// enforce a hard payload-size cap inside an open sentinel block).
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }
}

/// Largest byte index `i ≤ buf.len()` such that `buf[..i]` is safe to
/// emit (i.e. cannot be the start of a future sentinel match) and is on
/// a UTF-8 character boundary.
fn safe_emit_boundary(buf: &str, sentinel: &str) -> usize {
    if buf.is_empty() {
        return 0;
    }
    let max_take = buf.len().min(sentinel.len().saturating_sub(1));
    for take in (1..=max_take).rev() {
        let cut = buf.len() - take;
        if !buf.is_char_boundary(cut) {
            continue;
        }
        if sentinel.starts_with(&buf[cut..]) {
            return cut;
        }
    }
    buf.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_emits_text_then_stop() {
        let mut p = PassthroughParser;
        let events = p.feed("hello world");
        assert_eq!(events.len(), 1);
        matches!(events[0], DecodeEvent::TextDelta(ref s) if s == "hello world");
        let final_events = p.finish(StopReason::EndOfText);
        assert_eq!(final_events.len(), 1);
        matches!(final_events[0], DecodeEvent::Stop { .. });
    }

    #[test]
    fn passthrough_swallows_empty_chunks() {
        let mut p = PassthroughParser;
        assert!(p.feed("").is_empty());
    }

    #[test]
    fn matcher_finds_sentinel_in_single_feed() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("hello <tool_call>{...}");
        let (before, after) = m.try_match().unwrap();
        assert_eq!(before, "hello ");
        assert_eq!(after, "{...}");
    }

    #[test]
    fn matcher_holds_partial_sentinel_across_feeds() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("plain text <tool_c");
        let safe = m.flush_safe_text();
        assert_eq!(safe, "plain text ");
        assert!(m.try_match().is_none());
        m.push("all>{...}");
        let (before, after) = m.try_match().unwrap();
        assert_eq!(before, "");
        assert_eq!(after, "{...}");
    }

    #[test]
    fn matcher_flushes_text_when_partial_doesnt_resolve() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("plain text <ne");
        // "<ne" is a prefix of "<tool_call>"? Let's see: "<tool_call>" starts
        // with "<", "<t", "<to", ... "<ne" doesn't match beyond "<".
        let safe = m.flush_safe_text();
        // Only "<" can extend into "<tool_call>"; "<ne" cannot.
        assert_eq!(safe, "plain text <ne");
    }

    #[test]
    fn matcher_holds_only_one_byte_when_only_first_char_of_sentinel_present() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("hello <");
        let safe = m.flush_safe_text();
        assert_eq!(safe, "hello ");
    }

    #[test]
    fn matcher_flushes_everything_when_no_partial_match() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("just some plain text without any markup");
        let safe = m.flush_safe_text();
        assert_eq!(safe, "just some plain text without any markup");
    }

    #[test]
    fn matcher_emits_correct_text_before_sentinel_with_utf8() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("héllo wörld <tool_call>x");
        let (before, after) = m.try_match().unwrap();
        assert_eq!(before, "héllo wörld ");
        assert_eq!(after, "x");
    }

    #[test]
    fn matcher_holds_partial_sentinel_after_utf8_text() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("héllo <tool_c");
        let safe = m.flush_safe_text();
        assert_eq!(safe, "héllo ");
    }

    #[test]
    fn matcher_finish_drains_buffer() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("dangling <to");
        let leftover = m.finish();
        assert_eq!(leftover, "dangling <to");
    }

    #[test]
    fn matcher_handles_repeated_false_starts() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("<<<");
        let safe = m.flush_safe_text();
        // We can emit "<<" — only the trailing "<" could start a sentinel.
        assert_eq!(safe, "<<");
        m.push("not it");
        let safe2 = m.flush_safe_text();
        // Now we can emit "<not it" — none of it can be a sentinel start.
        assert_eq!(safe2, "<not it");
    }

    #[test]
    fn matcher_finds_sentinel_after_partial_false_start() {
        let mut m = SentinelMatcher::new("<tool_call>");
        m.push("<tool_x");
        let safe = m.flush_safe_text();
        // "<tool_" was a sentinel-prefix that doesn't match the next byte;
        // since "<tool_x" cannot extend into "<tool_call>", all is safe.
        assert_eq!(safe, "<tool_x");
        m.push(" <tool_call>real");
        let (before, after) = m.try_match().unwrap();
        assert_eq!(before, " ");
        assert_eq!(after, "real");
    }
}
