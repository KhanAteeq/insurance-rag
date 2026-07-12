/**
 * Shared TypeScript models mirroring the FastAPI response schemas
 * (app/api/models.py). Keeping them in one place gives the frontend a single,
 * type-safe contract with the backend.
 */

/** A single cited source chunk returned with an answer. */
export interface Source {
  id: string;
  document_name: string;
  page_number: number;
  chunk_number: number;
  section: string;
  score: number;
}

/** Per-stage latency breakdown (milliseconds). */
export interface Timings {
  retrieval_ms: number;
  llm_ms: number;
  total_ms: number;
}

/** Body sent to POST /api/ask (and /api/ask/stream). */
export interface AskRequest {
  question: string;
  top_k?: number | null;
  document_name?: string | null;
}

/** Body returned by POST /api/ask. */
export interface AskResponse {
  question: string;
  answer: string;
  sources: Source[];
  model: string;
  timings: Timings;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  grounded: boolean;
}

/** Ollama dependency health. */
export interface OllamaHealth {
  reachable: boolean;
  model: string;
  model_available: boolean;
  models: string[];
}

/** Overall service health from GET /api/health. */
export interface HealthResponse {
  status: string;
  app_name: string;
  app_version: string;
  total_chunks: number;
  embedding_model: string;
  llm: OllamaHealth;
}

/** One indexed document + its chunk count. */
export interface DocumentInfo {
  document_name: string;
  chunk_count: number;
}

/** GET /api/documents response. */
export interface DocumentsResponse {
  total_documents: number;
  total_chunks: number;
  documents: DocumentInfo[];
}

/** Result of ingesting a single PDF. */
export interface IngestResponse {
  document_name: string;
  file_hash: string;
  pages_extracted: number;
  chunks_created: number;
  chunks_stored: number;
  skipped: boolean;
  message: string;
}

/** The role of a message in the chat transcript. */
export type ChatRole = 'user' | 'assistant';

/** A single message rendered in the chat window. */
export interface ChatMessage {
  role: ChatRole;
  content: string;
  sources?: Source[];
  timings?: Timings;
  grounded?: boolean;
  pending?: boolean;
  error?: boolean;
}

/** Streaming event shapes emitted by POST /api/ask/stream (NDJSON). */
export type StreamEvent =
  | { type: 'sources'; sources: Source[]; grounded: boolean }
  | { type: 'token'; token: string }
  | { type: 'done'; timings: Timings; grounded: boolean; model: string }
  | { type: 'error'; message: string };
