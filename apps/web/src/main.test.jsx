import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatPage } from './main.jsx'

function renderChat() {
  window.requestAnimationFrame = vi.fn()
  return render(<ChatPage />)
}

async function chooseQuickReply(user, name) {
  await user.click(screen.getByText(name).closest('button'))
}

beforeEach(() => {
  globalThis.fetch = vi.fn()
})

describe('ChatPage', () => {
  it('renders the default ChatUI interface immediately', () => {
    renderChat()

    expect(document.querySelector('.ChatApp')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Aurora Responde' })).toBeTruthy()
    expect(document.querySelector('.Avatar img')).toBeTruthy()
    expect(screen.getByText('Olá, sou a assistente do Café Aurora. Posso ajudar com unidades, parceiros e políticas.')).toBeTruthy()
  })

  it('prefills the ChatUI composer from a quick reply', async () => {
    const user = userEvent.setup()
    renderChat()

    await chooseQuickReply(user, 'O que fazer quando um cliente recebe um pedido incorreto?')

    expect(screen.getByPlaceholderText('Pergunte qualquer coisa').value).toBe('O que fazer quando um cliente recebe um pedido incorreto?')
  })

  it('uses ChatUI\'s conditional send button while typing', async () => {
    const user = userEvent.setup()
    renderChat()

    expect(screen.queryByRole('button', { name: 'Enviar' })).toBeNull()

    await user.type(screen.getByPlaceholderText('Pergunte qualquer coisa'), 'Olá')

    expect(screen.getByRole('button', { name: 'Enviar' }).classList.contains('Composer-sendBtn')).toBe(true)
  })

  it('shows typing, then an accessible completed response with sources', async () => {
    window.matchMedia.mockImplementation((query) => ({
      matches: query === '(prefers-reduced-motion: reduce)',
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }))
    let resolveRequest
    globalThis.fetch.mockReturnValue(new Promise((resolve) => {
      resolveRequest = resolve
    }))
    const user = userEvent.setup()
    renderChat()

    await chooseQuickReply(user, 'Quais produtos contêm leite?')
    await user.click(screen.getByRole('button', { name: 'Enviar' }))

    expect(document.querySelector('.Typing')).toBeTruthy()
    expect(document.querySelector('.Typing-text')).toBeNull()

    resolveRequest({
      ok: true,
      json: async () => ({
        response: 'A unidade Centro atende aos sábados.',
        sources: [
          { document_name: 'Manual de Operações - Café Aurora', location: 'página 12' },
          { document_name: 'Manual de Operações - Café Aurora', location: 'página 13' },
        ],
      }),
    })

    await waitFor(() => {
      expect(screen.getByText('A unidade Centro atende aos sábados.')).toBeTruthy()
    })
    const sources = screen.getByLabelText('Fontes consultadas')
    expect(sources.classList.contains('DocumentSources')).toBe(true)
    expect(sources.parentElement.classList.contains('MessageResponse')).toBe(true)
    expect(screen.getByText('Manual de Operações')).toBeTruthy()
    expect(sources.querySelectorAll('.DocumentSources-name')).toHaveLength(1)
    expect(sources.textContent).not.toMatch(/Café Aurora|página 12/)
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ message: 'Quais produtos contêm leite?' }),
    }))
  })

  it('uses TypingBubble when motion is allowed', async () => {
    window.matchMedia.mockImplementation((query) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }))
    globalThis.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ response: 'Resposta animada.', sources: [] }),
    })
    const user = userEvent.setup()
    renderChat()

    await chooseQuickReply(user, 'Quais produtos contêm leite?')
    await user.click(screen.getByRole('button', { name: 'Enviar' }))

    await waitFor(() => {
      expect(document.querySelector('.Bubble[aria-hidden="true"]')).toBeTruthy()
    })
  })

  it('shows a system error and retries without duplicating the question', async () => {
    globalThis.fetch
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ response: 'Benefícios disponíveis.', sources: [] }),
      })
    const user = userEvent.setup()
    renderChat()

    await chooseQuickReply(user, 'Qual unidade funciona até mais tarde?')
    await user.click(screen.getByRole('button', { name: 'Enviar' }))

    await waitFor(() => {
      expect(screen.getByText('Não foi possível responder agora.')).toBeTruthy()
    })
    expect(document.querySelector('.SystemMessage')).toBeTruthy()
    expect(screen.getByPlaceholderText('Pergunte qualquer coisa').value).toBe('Qual unidade funciona até mais tarde?')

    await user.click(screen.getByRole('button', { name: 'Tentar novamente' }))

    await waitFor(() => {
      expect(screen.getByText('Benefícios disponíveis.')).toBeTruthy()
    })
    expect(globalThis.fetch).toHaveBeenCalledTimes(2)
    expect(document.querySelectorAll('.Message.right')).toHaveLength(1)
    expect(screen.getByPlaceholderText('Pergunte qualquer coisa').value).toBe('')
  })
})
