import { CommonModule } from '@angular/common';
import {
  AfterViewChecked,
  Component,
  ElementRef,
  OnInit,
  ViewChild,
  inject,
} from '@angular/core';
import { TextFieldModule } from '@angular/cdk/text-field';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';

import {
  ChatMessage,
  HealthResponse,
  Source,
} from '../../models/chat.models';
import { ChatService } from '../../services/chat.service';

/**
 * ChatComponent — the main chat UI for the Insurance RAG assistant.
 *
 * Features:
 *   * Sends questions to the backend and renders answers.
 *   * Optional streaming mode (live token-by-token typing).
 *   * Shows cited sources and per-stage timings for each answer.
 *   * PDF upload to ingest new documents.
 *   * Live backend/Ollama health indicator.
 */
@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    TextFieldModule,
    MatToolbarModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule,
    MatProgressBarModule,
    MatSlideToggleModule,
    MatTooltipModule,
    MatSnackBarModule,
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent implements OnInit, AfterViewChecked {
  private readonly chat = inject(ChatService);
  private readonly snackBar = inject(MatSnackBar);

  @ViewChild('scrollAnchor') private scrollAnchor?: ElementRef<HTMLDivElement>;
  @ViewChild('fileInput') private fileInput?: ElementRef<HTMLInputElement>;

  messages: ChatMessage[] = [];
  question = '';
  loading = false;
  streaming = true;
  health: HealthResponse | null = null;

  private abortController: AbortController | null = null;
  private shouldScroll = false;

  ngOnInit(): void {
    this.refreshHealth();
    this.messages.push({
      role: 'assistant',
      content:
        'Hello! I am your insurance claims assistant. Ask me anything about ' +
        'the ingested claim documents and I will answer using only their ' +
        'contents, with citations.',
    });
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  /** Fetch backend + Ollama health for the status indicator. */
  refreshHealth(): void {
    this.chat.health().subscribe({
      next: (h) => (this.health = h),
      error: () => (this.health = null),
    });
  }

  /** True when the backend + model are fully ready. */
  get healthy(): boolean {
    return !!this.health && this.health.status === 'ok';
  }

  /** Submit the current question. */
  send(): void {
    const text = this.question.trim();
    if (!text || this.loading) {
      return;
    }

    this.messages.push({ role: 'user', content: text });
    this.question = '';
    this.shouldScroll = true;

    if (this.streaming) {
      void this.sendStreaming(text);
    } else {
      this.sendBlocking(text);
    }
  }

  /** Blocking request — waits for the full answer. */
  private sendBlocking(text: string): void {
    this.loading = true;
    const placeholder: ChatMessage = {
      role: 'assistant',
      content: '',
      pending: true,
    };
    this.messages.push(placeholder);
    this.shouldScroll = true;

    this.chat.ask({ question: text }).subscribe({
      next: (res) => {
        placeholder.content = res.answer;
        placeholder.sources = res.sources;
        placeholder.timings = res.timings;
        placeholder.grounded = res.grounded;
        placeholder.pending = false;
        this.shouldScroll = true;
      },
      error: (err) => {
        placeholder.content = this.errorText(err);
        placeholder.pending = false;
        placeholder.error = true;
        this.loading = false;
      },
      complete: () => (this.loading = false),
    });
  }

  /** Streaming request — renders tokens as they arrive. */
  private async sendStreaming(text: string): Promise<void> {
    this.loading = true;
    this.abortController = new AbortController();
    const placeholder: ChatMessage = {
      role: 'assistant',
      content: '',
      pending: true,
    };
    this.messages.push(placeholder);
    this.shouldScroll = true;

    try {
      for await (const event of this.chat.askStream(
        { question: text },
        this.abortController.signal,
      )) {
        switch (event.type) {
          case 'sources':
            placeholder.sources = event.sources;
            placeholder.grounded = event.grounded;
            break;
          case 'token':
            placeholder.content += event.token;
            placeholder.pending = false;
            this.shouldScroll = true;
            break;
          case 'done':
            placeholder.timings = event.timings;
            placeholder.grounded = event.grounded;
            placeholder.pending = false;
            break;
          case 'error':
            placeholder.content =
              placeholder.content || `Error: ${event.message}`;
            placeholder.error = true;
            placeholder.pending = false;
            break;
        }
      }
    } catch (err) {
      placeholder.content = this.errorText(err);
      placeholder.error = true;
      placeholder.pending = false;
    } finally {
      this.loading = false;
      this.abortController = null;
      this.shouldScroll = true;
    }
  }

  /** Cancel an in-flight streaming request. */
  stop(): void {
    this.abortController?.abort();
    this.loading = false;
  }

  /** Trigger the hidden file input. */
  chooseFile(): void {
    this.fileInput?.nativeElement.click();
  }

  /** Handle a selected PDF and ingest it. */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.notify('Only PDF files can be ingested.');
      input.value = '';
      return;
    }

    this.loading = true;
    this.chat.ingest(file).subscribe({
      next: (res) => {
        this.notify(res.message);
        this.refreshHealth();
      },
      error: (err) => this.notify(this.errorText(err)),
      complete: () => {
        this.loading = false;
        input.value = '';
      },
    });
  }

  /** Send on Enter (Shift+Enter inserts a newline). */
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  /** Build a readable citation label for a source. */
  sourceLabel(s: Source): string {
    return `${s.document_name} · p.${s.page_number} · ${Math.round(
      s.score * 100,
    )}%`;
  }

  trackByIndex(index: number): number {
    return index;
  }

  private notify(message: string): void {
    this.snackBar.open(message, 'Dismiss', { duration: 5000 });
  }

  private errorText(err: unknown): string {
    if (err instanceof Error) {
      return err.message;
    }
    const e = err as { error?: { detail?: string }; message?: string };
    return e?.error?.detail ?? e?.message ?? 'An unexpected error occurred.';
  }

  private scrollToBottom(): void {
    this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: 'smooth' });
  }
}
