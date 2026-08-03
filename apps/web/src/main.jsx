import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

function Chat() {
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState([])
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')

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
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <main className="chat-shell">
      <header>
        <p className="eyebrow">Café Aurora</p>
        <h1>Como posso ajudar?</h1>
        <p className="intro">Pergunte sobre as informações dos parceiros e das unidades.</p>
      </header>

      <section className="conversation" aria-live="polite">
        {conversation.length === 0 && <p className="empty">Sua conversa começa aqui.</p>}
        {conversation.map((item, index) => (
          <article className={`message ${item.role}`} key={`${item.role}-${index}`}>
            <p>{item.text}</p>
            {item.sources?.length > 0 && (
              <ul className="sources" aria-label="Fontes">
                {item.sources.map((source) => <li key={`${source.document_name}-${source.location}`}>{source.document_name} - {source.location}</li>)}
              </ul>
            )}
          </article>
        ))}
        {isSending && <p className="status">Consultando o conhecimento do Café Aurora...</p>}
      </section>

      <form onSubmit={sendMessage}>
        <label htmlFor="message">Sua pergunta</label>
        <div className="composer">
          <textarea id="message" value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ex.: Qual é a política de reembolso?" rows="3" />
          <button type="submit" disabled={isSending}>{isSending ? 'Enviando' : 'Enviar'}</button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<Chat />)
