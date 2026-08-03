"""Partner-knowledge response contract shared across application boundaries."""

DEFAULT_RELEVANCE_THRESHOLD = 0.45

APPROVED_PARTNER_DOCUMENT_NAMES = frozenset(
    {
        "Catálogo de Produtos e Ingredientes - Café Aurora",
        "Controle de Estoque",
        "Configuração das Unidades",
        "Manual de Operações das Unidades - Café Aurora",
        "Guia de Atendimento ao Cliente",
        "Política de Despesas e Reembolsos",
    }
)

SCOPE_REFUSAL = (
    "Só posso responder a perguntas apoiadas pelo conhecimento dos "
    "Parceiros do Café Aurora."
)

GROUNDING_SERVICE_UNAVAILABLE = (
    "O serviço de respostas fundamentadas está temporariamente indisponível."
)
