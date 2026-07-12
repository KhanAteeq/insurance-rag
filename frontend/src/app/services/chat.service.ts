import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  AskRequest,
  AskResponse,
  DocumentsResponse,
  HealthResponse,
  IngestResponse,
  StreamEvent,
} from '../models/chat.models';

/**
 * ChatService — the single gateway between the Angular UI and the FastAPI
 * backend. It exposes typed methods for every endpoint the UI needs.
 *
 * The streaming method uses the Fetch API directly (rather than HttpClient) so
 * it can read the NDJSON body incrementally and emit tokens as they arrive.
 */
@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl;

  /** GET /api/health */
  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/api/health`);
  }

  /** GET /api/documents */
  documents(): Observable<DocumentsResponse> {
    return this.http.get<DocumentsResponse>(`${this.baseUrl}/api/documents`);
  }

  /** POST /api/ask (blocking) */
  ask(request: AskRequest): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.baseUrl}/api/ask`, request);
  }

  /** POST /api/ingest (multipart upload) */
  ingest(file: File, force = false): Observable<IngestResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<IngestResponse>(
      `${this.baseUrl}/api/ingest?force=${force}`,
      form,
    );
  }

  /**
   * POST /api/ask/stream — stream tokens as they are generated.
   *
   * Returns an async generator of parsed StreamEvent objects. The caller can
   * `for await (const event of chatService.askStream(...))` to render a live
   * "typing" answer.
   *
   * @param request the question payload
   * @param signal  optional AbortSignal to cancel an in-flight request
   */
  async *askStream(
    request: AskRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent, void, unknown> {
    const response = await fetch(`${this.baseUrl}/api/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok || !response.body) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const body = await response.json();
        detail = body?.detail ?? detail;
      } catch {
        /* non-JSON error body — keep the default detail */
      }
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      // Read the NDJSON stream chunk-by-chunk, emitting one event per line.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) {
            yield JSON.parse(line) as StreamEvent;
          }
        }
      }

      // Flush any trailing partial line.
      const tail = buffer.trim();
      if (tail) {
        yield JSON.parse(tail) as StreamEvent;
      }
    } finally {
      reader.releaseLock();
    }
  }
}
