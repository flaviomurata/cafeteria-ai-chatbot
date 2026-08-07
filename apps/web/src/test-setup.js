import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

class ResizeObserver {
  observe() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserver

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })),
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
