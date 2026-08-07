import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import Chat, {
  Bubble,
  Typing,
  TypingBubble,
  VisuallyHidden,
  useMessages,
} from '@chatui/core'
import '@chatui/core/dist/index.css'
import './document-sources.css'

const quickReplies = [
  { name: 'Quais produtos contêm leite?' },
  { name: 'O que fazer quando um cliente recebe um pedido incorreto?' },
  { name: 'Qual unidade funciona até mais tarde?' },
]

const initialMessages = [
  {
    type: 'greeting',
    content: {
      text: 'Olá, sou a assistente do Café Aurora. Posso ajudar com unidades, parceiros e políticas.',
    },
    user: {
      avatar: 'https://gw.alicdn.com/imgextra/i2/O1CN01fPEB9P1ylYWgaDuVR_!!6000000006619-0-tps-132-132.jpg',
      avatarAlt: 'Aurora',
    },
    hasTime: false,
  },
]

const portugueseLocales = {
  Composer: {
    send: 'Enviar',
  },
}

function useReducedMotion() {
  const [reduceMotion, setReduceMotion] = useState(() => (
    typeof window !== 'undefined'
      && window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches
  ))

  useEffect(() => {
    const mediaQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!mediaQuery) return undefined

    const updatePreference = () => setReduceMotion(mediaQuery.matches)
    updatePreference()
    mediaQuery.addEventListener?.('change', updatePreference)

    return () => mediaQuery.removeEventListener?.('change', updatePreference)
  }, [])

  return reduceMotion
}

function displayDocumentName(documentName) {
  return documentName.replace(/\s*-\s*Caf[eé] Aurora\s*$/i, '')
}

function Sources({ sources }) {
  const documentNames = [...new Set(
    sources?.map(({ document_name: documentName }) => displayDocumentName(documentName)) ?? [],
  )]

  if (!documentNames.length) return null

  return (
    <aside className="DocumentSources" aria-label="Fontes consultadas">
      <span className="DocumentSources-label">Fontes:</span>{' '}
      {documentNames.map((documentName, index) => (
        <span className="DocumentSources-name" key={documentName}>
          {index > 0 && ' · '}
          {documentName}
        </span>
      ))}
    </aside>
  )
}

export function ChatPage() {
  const { messages, appendMsg, updateMsg } = useMessages(initialMessages)
  const [isSending, setIsSending] = useState(false)
  const composerRef = useRef(null)
  const sendingRef = useRef(false)
  const reduceMotion = useReducedMotion()

  function focusComposer() {
    const focus = () => document.querySelector('.Composer-input')?.focus()
    if (window.requestAnimationFrame) {
      window.requestAnimationFrame(focus)
    } else {
      window.setTimeout(focus, 0)
    }
  }

  function chooseSuggestion(item) {
    composerRef.current?.setText(item.name)
    focusComposer()
  }

  function canSend() {
    return !sendingRef.current
  }

  async function requestAnswer(text, pendingMessageId) {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Não foi possível responder agora.')

      updateMsg(pendingMessageId, {
        type: 'answer',
        content: { text: body.response, sources: body.sources },
        hasTime: false,
      })
    } catch {
      updateMsg(pendingMessageId, {
        type: 'system',
        content: {
          text: 'Não foi possível responder agora.',
          action: {
            text: 'Tentar novamente',
            once: true,
            onClick: () => retryMessage(pendingMessageId, text),
          },
        },
        hasTime: false,
      })
      composerRef.current?.setText(text)
      focusComposer()
    } finally {
      sendingRef.current = false
      setIsSending(false)
    }
  }

  async function retryMessage(messageId, text) {
    if (sendingRef.current) return

    sendingRef.current = true
    setIsSending(true)
    composerRef.current?.setText('')
    updateMsg(messageId, {
      type: 'pending',
      content: {},
      hasTime: false,
    })

    await requestAnswer(text, messageId)
  }

  async function sendMessage(type, value) {
    const text = value.trim()
    if (type !== 'text' || !text || sendingRef.current) return

    sendingRef.current = true
    setIsSending(true)

    appendMsg({
      type: 'user',
      content: { text },
      position: 'right',
      hasTime: false,
    })

    const pendingMessageId = appendMsg({
      type: 'pending',
      content: {},
      hasTime: false,
    })

    await requestAnswer(text, pendingMessageId)
  }

  function renderMessageContent(message) {
    const content = message.content ?? {}

    switch (message.type) {
      case 'greeting':
      case 'user':
        return <Bubble content={content.text} />
      case 'pending':
        return <Typing />
      case 'answer':
        return (
          <div className="MessageResponse">
            {reduceMotion ? (
              <Bubble content={content.text} />
            ) : (
              <TypingBubble
                content={content.text}
                aria-hidden="true"
              />
            )}
            {!reduceMotion && <VisuallyHidden>{content.text}</VisuallyHidden>}
            <Sources sources={content.sources} />
          </div>
        )
      default:
        return null
    }
  }

  return (
    <Chat
      locale="en-US"
      locales={portugueseLocales}
      colorScheme="auto"
      navbar={{ title: 'Aurora Responde' }}
      messages={messages}
      renderMessageContent={renderMessageContent}
      quickReplies={quickReplies}
      quickRepliesVisible={messages.length === initialMessages.length && !isSending}
      onQuickReplyClick={chooseSuggestion}
      composerRef={composerRef}
      placeholder="Pergunte qualquer coisa"
      onBeforeSend={canSend}
      onSend={sendMessage}
    />
  )
}

const rootElement = document.getElementById('root')
if (rootElement) {
  createRoot(rootElement).render(<ChatPage />)
}
