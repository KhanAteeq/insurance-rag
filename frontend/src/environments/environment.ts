/**
 * Environment configuration.
 *
 * `apiBaseUrl` points at the FastAPI backend. During local development the
 * backend runs on port 8000 while `ng serve` runs on 4200; the backend's CORS
 * config already allows the 4200 origin.
 */
export const environment = {
  production: false,
  apiBaseUrl: 'https://insurance-rag-api-hachc3b9eta7c9ab.westcentralus-01.azurewebsites.net',
};
