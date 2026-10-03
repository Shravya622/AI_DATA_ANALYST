import { useState, useRef, useEffect, KeyboardEvent, FormEvent } from 'react';
import ResponseRenderer, { QueryResponseData } from './ResponseRenderer';

type MessageContent = string | QueryResponseData;

interface Message {
  role: 'user' | 'assistant';
  content: MessageContent;
  isError?: boolean;
}

interface Props {
  sessionId: string;
  pendingQuery: string | null;
  consumePendingQuery: () => string | null;
}

function isQueryResponseData(content: MessageContent): content is QueryResponseData {
  return typeof content === 'object' && content !== null && 'answer' in content;
}

const EXAMPLE_PROMPTS = [
  'Which region generated the highest revenue?',
  'Show monthly revenue trends as a chart.',
  'Detect anomalies in the dataset.',
  'Give me a full profile of the dataset.',
];

function ChatPanel({ sessionId, pendingQuery, consumePendingQuery }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (pendingQuery && !loading) {
      const q = consumePendingQuery();
      if (q) {
        setTimeout(() => submitQueryText(q), 0);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingQuery]);

  async function submitQueryText(query: string) {
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/sessions/' + sessionId + '/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error('Server error (HTTP ' + response.status + '): ' + text);
      }
      const data: QueryResponseData = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error: ' + message, isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQueryText(input);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    submitQueryText(input);
  }

  return (
    <div className="chat-panel">
      <div className="chat-panel__messages" role="log" aria-live="polite" aria-label="Chat history">

        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <div className="chat-empty__icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
            </div>
            <h2 className="chat-empty__title">Ask questions about your data</h2>
            <p className="chat-empty__subtitle">
              Upload a CSV and ask anything in plain English.
            </p>
            <div className="chat-empty__examples" aria-label="Example prompts">
              {EXAMPLE_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  className="chat-empty__example"
                  onClick={() => { if (!loading) submitQueryText(prompt); }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={'chat-message chat-message--' + msg.role}>
            <span className="chat-message__label">
              {msg.role === 'user' ? 'You' : 'AI Analyst'}
            </span>

            {msg.role === 'user' ? (
              <div className="chat-bubble chat-bubble--user">
                {msg.content as string}
              </div>
            ) : isQueryResponseData(msg.content) ? (
              <div className="chat-bubble chat-bubble--rich">
                <ResponseRenderer response={msg.content} />
              </div>
            ) : (
              <div className={'chat-bubble ' + (msg.isError ? 'chat-bubble--error' : 'chat-bubble--assistant')}>
                {msg.content as string}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-message chat-message--assistant">
            <span className="chat-message__label">AI Analyst</span>
            <div className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
              <div className="thinking-dots"><span /><span /><span /></div>
              Analyzing your question...
            </div>
          </div>
        )}

        <div ref={bottomRef} aria-hidden="true" />
      </div>

      <div className="chat-input-area">
        <form className="chat-input-form" onSubmit={handleSubmit} noValidate>
          <textarea
            className="chat-input-textarea"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your data... (Enter to send, Shift+Enter for new line)"
            disabled={loading}
            rows={1}
            aria-label="Query input"
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={loading || !input.trim()}
            aria-label="Send query"
            title="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatPanel;
