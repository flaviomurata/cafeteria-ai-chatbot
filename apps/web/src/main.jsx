import { useLayoutEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const suggestions = [
  'Quais unidades estão abertas agora?',
  'Como funciona o reembolso?',
  'Quais são os benefícios dos parceiros?',
]

function ArrowUpIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
      <path d="M10 15V5m0 0 4 4m-4-4L6 9" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function SparkIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
      <path d="m10 2 .84 4.16L15 7l-4.16.84L10 12l-.84-4.16L5 7l4.16-.84L10 2Zm5 10 .47 2.53L18 15l-2.53.47L15 18l-.47-2.53L12 15l2.53-.47L15 12Z" fill="currentColor" />
    </svg>
  )
}

function resizeComposer(textarea) {
  if (!textarea) return

  const maxHeight = 134
  textarea.style.height = 'auto'
  textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
}

function Chat() {
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState([])
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  useLayoutEffect(() => {
    resizeComposer(inputRef.current)
  }, [message])

  function chooseSuggestion(suggestion) {
    setMessage(suggestion)
    setError('')
    inputRef.current?.focus()
  }

  async function sendMessage(event) {
    event.preventDefault()
    const text = message.trim()
    if (!text || isSending) return

    setMessage('')
    setError('')
    setConversation((items) => [...items, { role: 'user', text }])
    setIsSending(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Não foi possível responder agora.')

      setConversation((items) => [
        ...items,
        { role: 'assistant', text: body.response, sources: body.sources },
      ])
    } catch {
      setError('Não foi possível responder agora. Tente novamente.')
    } finally {
      setIsSending(false)
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <div className="app-shell">
      <main className="workspace" id="assistant">
        <header className="hero">
          <p className="kicker">Assistente Aurora</p>
          <h1>Olá, como posso ajudar?</h1>
          <p>Encontre respostas sobre unidades, parceiros e políticas do Café Aurora.</p>
        </header>

        <section className={`conversation${conversation.length ? ' has-messages' : ''}`} aria-label="Conversa" role="log" aria-live="polite" aria-relevant="additions text">
          {conversation.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon"><SparkIcon /></div>
              <p className="welcome-title">Por onde começamos?</p>
              <p className="welcome-copy">Escolha uma sugestão ou escreva a sua pergunta.</p>
              <div className="suggestions" aria-label="Sugestões de perguntas">
                {suggestions.map((suggestion) => (
                  <button className="suggestion" type="button" key={suggestion} onClick={() => chooseSuggestion(suggestion)}>
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {conversation.map((item, index) => (
            <article className={`message ${item.role}`} key={`${item.role}-${index}`}>
              <p>{item.text}</p>
              {item.sources?.length > 0 && (
                <ul className="sources" aria-label="Fontes consultadas">
                  {item.sources.map((source) => (
                    <li key={`${source.document_name}-${source.location}`}>
                      {source.document_name} <span>{source.location}</span>
                    </li>
                  ))}
                </ul>
              )}
            </article>
          ))}
          <p className="status" role="status">
            {isSending && <span className="status-text">Consultando o conhecimento do Café Aurora...</span>}
          </p>
        </section>

        <form className="composer-wrap" onSubmit={sendMessage}>
          <label className="sr-only" htmlFor="message">Sua pergunta</label>
          <div className="composer">
            <textarea
              ref={inputRef}
              id="message"
              name="message"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Pergunte qualquer coisa"
              rows="1"
              aria-invalid={Boolean(error)}
              aria-describedby={error ? 'message-error' : 'composer-hint'}
            />
            <button className="send-button" type="submit" disabled={isSending || !message.trim()} aria-label={isSending ? 'Enviando pergunta' : 'Enviar pergunta'}>
              <ArrowUpIcon />
            </button>
          </div>
          <p className="composer-hint" id="composer-hint">Enter para enviar. Shift + Enter para uma nova linha.</p>
          {error && <p className="error" id="message-error" role="alert">{error}</p>}
        </form>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<Chat />)
