import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideHttpClient, withFetch } from '@angular/common/http';

/**
 * Root application providers (standalone bootstrap — no NgModule).
 *
 *  * provideHttpClient(withFetch()) — enables HttpClient and the Fetch backend,
 *    which is required for reading streaming (NDJSON) responses.
 *  * provideAnimations()            — needed by Angular Material components.
 */
export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideHttpClient(withFetch()),
    provideAnimations(),
  ],
};
