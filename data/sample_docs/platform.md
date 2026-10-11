# Meridian Platform: deployment and operations

This is a fictional product manual used for DocuLens-AI's example benchmark.

Meridian stores workspace documents in a regional object store. At rest, documents are
encrypted using AES-256. Transport connections use TLS 1.3. Customers select their
workspace region during provisioning. The available regions are Mumbai and Frankfurt.

The standard workspace retains audit logs for 90 days. The enterprise workspace retains
audit logs for 365 days. An administrator can export logs as JSON Lines from the Audit page.

Backups run every 24 hours. Restores can target any completed daily snapshot from the
previous 30 days. A restore requires approval from a workspace administrator.

Meridian supports PDF, Markdown, and UTF-8 text documents. A single uploaded document
must not exceed 20 MB. Image-only PDFs require an OCR preprocessing step before ingestion.

The ingestion worker retries transient failures twice. Permanent parsing failures go
to the dead-letter queue, where an administrator can inspect the failure reason.
