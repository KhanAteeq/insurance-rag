import { Component } from '@angular/core';
import { ChatComponent } from './components/chat/chat.component';

/**
 * Root application component. It simply hosts the chat UI; all interaction
 * lives in ChatComponent.
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatComponent],
  template: '<app-chat></app-chat>',
})
export class AppComponent {}
