import {
  generateText,
  jsonSchema,
  stepCountIs,
  streamText,
  tool,
} from "ai";

type SearchUserDocumentsInput = {
  query: string;
  limit?: number;
  document_ids?: string[];
  rerank?: boolean;
  use_lexical_search?: boolean;
  max_chars?: number;
  max_tokens?: number;
};

type PromptCitation = {
  citation_id: string;
  title?: string;
  section_title?: string;
  section_path?: string;
  chunk_index?: number;
  source_type?: string;
  result_type?: string;
};

type PromptSourcePreview = {
  citation_id: string;
  title: string;
  locator_label: string | null;
  source_type: string | null;
  result_type: string | null;
  truncated: boolean;
};

type PromptSourceLocator = {
  chunk_index: number | null;
  section_path: string | null;
  page_number: number | null;
  page_index: number | null;
  slide_number: number | null;
  sheet_name: string | null;
  row_range: string | null;
  line_start: number | null;
  line_end: number | null;
  bbox: [number, number, number, number] | null;
  figure_id: string | null;
  figure_caption: string | null;
  figure_thumbnail_url: string | null;
};

type SearchUserDocumentsResult = {
  ok: true;
  query: string;
  context_text: string;
  snippets: Array<{
    citation_id: string;
    rank: number;
    text: string;
    score: number;
    source: PromptCitation;
    locator: PromptSourceLocator;
    token_estimate: number;
    char_count: number;
    retrieval_metadata?: {
      quality?: {
        verdict?: string;
        details?: string;
        char_count?: number;
        page_count?: number;
        meaningful_ratio?: number;
        mojibake_ratio?: number;
        text_to_page_ratio?: number;
      };
      rerank?: {
        provider?: string;
        model?: string;
        provider_score?: number;
        search_score?: number;
        original_rank?: number;
        rerank_rank?: number;
        rank_delta?: number;
      };
    };
    truncated: boolean;
  }>;
  citations: PromptCitation[];
  source_previews: PromptSourcePreview[];
  citation_summary: string;
  dropped_count: number;
  max_snippets: number;
  max_chars: number | null;
  max_tokens: number | null;
  token_estimate: number;
  char_count: number;
  truncated: boolean;
};

const searchUserDocumentsInputSchema = jsonSchema<SearchUserDocumentsInput>({
  type: "object",
  additionalProperties: false,
  properties: {
    query: { type: "string", minLength: 1, pattern: "\\S" },
    limit: { type: "integer", minimum: 1, maximum: 20, default: 5 },
    document_ids: {
      type: "array",
      items: { type: "string", minLength: 1, pattern: "\\S" },
    },
    rerank: { type: "boolean", default: false },
    use_lexical_search: {
      type: "boolean",
      default: true,
      description:
        "Controls configured lexical/exact-match expansion only; query-plan defaults remain provider capability-aware.",
    },
    max_chars: { type: "integer", minimum: 256, maximum: 12000, default: 3000 },
    max_tokens: { type: "integer", minimum: 64, maximum: 4000 },
  },
  required: ["query"],
});

function assertSearchUserDocumentsResult(
  value: unknown,
): asserts value is SearchUserDocumentsResult {
  if (!value || typeof value !== "object") {
    throw new Error("search_user_documents returned a non-object payload");
  }

  const payload = value as Partial<SearchUserDocumentsResult>;
  if (
    payload.ok !== true ||
    typeof payload.query !== "string" ||
    typeof payload.context_text !== "string" ||
    !Array.isArray(payload.snippets) ||
    !Array.isArray(payload.citations) ||
    !Array.isArray(payload.source_previews) ||
    typeof payload.citation_summary !== "string" ||
    !isInteger(payload.dropped_count) ||
    !isInteger(payload.max_snippets) ||
    !isIntegerOrNull(payload.max_chars) ||
    !isIntegerOrNull(payload.max_tokens) ||
    !isInteger(payload.token_estimate) ||
    !isInteger(payload.char_count) ||
    typeof payload.truncated !== "boolean" ||
    !payload.snippets.every(isSearchSnippet) ||
    !payload.citations.every(isPromptCitation) ||
    !payload.source_previews.every(isPromptSourcePreview)
  ) {
    throw new Error("search_user_documents returned an invalid payload");
  }
}

function isIntegerOrNull(value: unknown): value is number | null {
  return isInteger(value) || value === null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value);
}

function isOptionalString(value: unknown): value is string | undefined {
  return value === undefined || typeof value === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const allowed = new Set(keys);
  return Object.keys(value).every((key) => allowed.has(key));
}

function isPromptCitation(value: unknown): value is PromptCitation {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "citation_id",
      "title",
      "section_title",
      "section_path",
      "chunk_index",
      "source_type",
      "result_type",
    ])
  ) {
    return false;
  }
  return (
    typeof value.citation_id === "string" &&
    isOptionalString(value.title) &&
    isOptionalString(value.section_title) &&
    isOptionalString(value.section_path) &&
    (value.chunk_index === undefined || isInteger(value.chunk_index)) &&
    isOptionalString(value.source_type) &&
    isOptionalString(value.result_type)
  );
}

function isPromptSourceLocator(value: unknown): value is PromptSourceLocator {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "chunk_index",
      "section_path",
      "page_number",
      "page_index",
      "slide_number",
      "sheet_name",
      "row_range",
      "line_start",
      "line_end",
      "bbox",
      "figure_id",
      "figure_caption",
      "figure_thumbnail_url",
    ])
  ) {
    return false;
  }
  return (
    (isInteger(value.chunk_index) || value.chunk_index === null) &&
    (typeof value.section_path === "string" || value.section_path === null) &&
    (isInteger(value.page_number) || value.page_number === null) &&
    (isInteger(value.page_index) || value.page_index === null) &&
    (isInteger(value.slide_number) || value.slide_number === null) &&
    (typeof value.sheet_name === "string" || value.sheet_name === null) &&
    (typeof value.row_range === "string" || value.row_range === null) &&
    (isInteger(value.line_start) || value.line_start === null) &&
    (isInteger(value.line_end) || value.line_end === null) &&
    (Array.isArray(value.bbox)
      ? value.bbox.length === 4 && value.bbox.every(isFiniteNumber)
      : value.bbox === null) &&
    (typeof value.figure_id === "string" || value.figure_id === null) &&
    (typeof value.figure_caption === "string" || value.figure_caption === null) &&
    (typeof value.figure_thumbnail_url === "string" ||
      value.figure_thumbnail_url === null)
  );
}

function isSearchSnippet(
  value: unknown,
): value is SearchUserDocumentsResult["snippets"][number] {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "citation_id",
      "rank",
      "text",
      "score",
      "source",
      "locator",
      "token_estimate",
      "char_count",
      "retrieval_metadata",
      "truncated",
    ])
  ) {
    return false;
  }
  return (
    typeof value.citation_id === "string" &&
    isInteger(value.rank) &&
    typeof value.text === "string" &&
    isFiniteNumber(value.score) &&
    isPromptCitation(value.source) &&
    isPromptSourceLocator(value.locator) &&
    isInteger(value.token_estimate) &&
    isInteger(value.char_count) &&
    (value.retrieval_metadata === undefined ||
      isSnippetRetrievalMetadata(value.retrieval_metadata)) &&
    typeof value.truncated === "boolean"
  );
}

function isSnippetRetrievalMetadata(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if (!hasExactKeys(value, ["quality", "rerank"])) {
    return false;
  }
  if (
    value.quality !== undefined &&
    !isSnippetQualityMetadata(value.quality)
  ) {
    return false;
  }
  if (value.rerank === undefined) {
    return true;
  }
  return isSnippetRerankMetadata(value.rerank);
}

function isSnippetQualityMetadata(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "verdict",
      "details",
      "char_count",
      "page_count",
      "meaningful_ratio",
      "mojibake_ratio",
      "text_to_page_ratio",
    ])
  ) {
    return false;
  }
  return (
    isOptionalString(value.verdict) &&
    isOptionalString(value.details) &&
    (value.char_count === undefined || isInteger(value.char_count)) &&
    (value.page_count === undefined || isInteger(value.page_count)) &&
    (value.meaningful_ratio === undefined ||
      isFiniteNumber(value.meaningful_ratio)) &&
    (value.mojibake_ratio === undefined || isFiniteNumber(value.mojibake_ratio)) &&
    (value.text_to_page_ratio === undefined ||
      isFiniteNumber(value.text_to_page_ratio))
  );
}

function isSnippetRerankMetadata(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "provider",
      "model",
      "provider_score",
      "search_score",
      "original_rank",
      "rerank_rank",
      "rank_delta",
    ])
  ) {
    return false;
  }
  return (
    (value.provider === undefined || typeof value.provider === "string") &&
    (value.model === undefined || typeof value.model === "string") &&
    (value.provider_score === undefined || isFiniteNumber(value.provider_score)) &&
    (value.search_score === undefined || isFiniteNumber(value.search_score)) &&
    (value.original_rank === undefined || isInteger(value.original_rank)) &&
    (value.rerank_rank === undefined || isInteger(value.rerank_rank)) &&
    (value.rank_delta === undefined || isInteger(value.rank_delta))
  );
}

function isPromptSourcePreview(
  value: unknown,
): value is SearchUserDocumentsResult["source_previews"][number] {
  if (!isRecord(value)) {
    return false;
  }
  if (
    !hasExactKeys(value, [
      "citation_id",
      "title",
      "locator_label",
      "source_type",
      "result_type",
      "truncated",
    ])
  ) {
    return false;
  }
  return (
    typeof value.citation_id === "string" &&
    typeof value.title === "string" &&
    (typeof value.locator_label === "string" || value.locator_label === null) &&
    (typeof value.source_type === "string" || value.source_type === null) &&
    (typeof value.result_type === "string" || value.result_type === null) &&
    typeof value.truncated === "boolean"
  );
}

const searchUserDocuments = tool({
  description:
    "Search user documents through an application endpoint backed by rag-core.",
  inputSchema: searchUserDocumentsInputSchema,
  execute: async (input, { abortSignal }): Promise<SearchUserDocumentsResult> => {
    const response = await fetch("https://your-app.example.com/api/search-user-documents", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${process.env.SEARCH_API_TOKEN ?? ""}`,
      },
      signal: abortSignal,
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      throw new Error(`search-user-documents failed with ${response.status}`);
    }

    const payload: unknown = await response.json();
    assertSearchUserDocumentsResult(payload);
    return payload;
  },
  toModelOutput: ({ output }) => ({
    type: "text",
    value: formatSearchResultForModel(output),
  }),
});

function summarizeSearchResult(value: SearchUserDocumentsResult): Record<string, unknown> {
  return {
    ok: value.ok,
    snippet_count: value.snippets.length,
    citation_count: value.citations.length,
    dropped_count: value.dropped_count,
    truncated: value.truncated,
  };
}

function formatSearchResultForModel(value: SearchUserDocumentsResult): string {
  const lines = [`Query: ${value.query}`, "", "Context:", value.context_text];
  if (value.citation_summary.trim()) {
    lines.push("", "Citations:", value.citation_summary);
  }
  if (value.truncated) {
    lines.push(
      "",
      `Note: search output was truncated to ${value.max_snippets} snippets.`,
    );
  }
  return lines.join("\n");
}

function summarizeToolOutput(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }
  if (value.output && isRecord(value.output)) {
    const output = value.output;
    try {
      assertSearchUserDocumentsResult(output);
      return { ...value, output: summarizeSearchResult(output) };
    } catch {
      return { ...value, output: "[invalid search_user_documents payload]" };
    }
  }
  return value;
}

function summarizeUnknownError(value: unknown): string {
  if (value instanceof Error) {
    return value.message;
  }
  return String(value);
}

export async function runSingleStepToolCall(prompt: string): Promise<void> {
  const result = await generateText({
    model: "anthropic/claude-sonnet-4.6",
    prompt,
    tools: {
      search_user_documents: searchUserDocuments,
    },
    stopWhen: stepCountIs(3),
  });

  console.log(result.text);
  console.log(result.toolResults.map(summarizeToolOutput));
}

export async function runStreamingToolCall(prompt: string): Promise<void> {
  const result = streamText({
    model: "anthropic/claude-sonnet-4.6",
    prompt,
    tools: {
      search_user_documents: searchUserDocuments,
    },
    stopWhen: stepCountIs(3),
    onStepFinish(step) {
      console.log(
        "step",
        step.stepNumber,
        "tool-results",
        step.toolResults.map(summarizeToolOutput),
      );
    },
  });

  for await (const part of result.fullStream) {
    if (part.type === "text-delta") {
      process.stdout.write(part.text);
    }
    if (part.type === "tool-result" && part.toolName === "search_user_documents") {
      const output = part.output;
      assertSearchUserDocumentsResult(output);
      console.log("\n[tool-result]", summarizeSearchResult(output));
    }
    if (part.type === "tool-error" && part.toolName === "search_user_documents") {
      console.error("\n[tool-error]", {
        toolName: part.toolName,
        input: part.input,
        error: summarizeUnknownError(part.error),
      });
    }
    if (part.type === "error") {
      console.error("\n[stream-error]", summarizeUnknownError(part.error));
    }
  }
}
