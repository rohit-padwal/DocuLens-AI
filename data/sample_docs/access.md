# Meridian Platform: access and support

Meridian uses three roles. A viewer can search and read documents. An editor can also
upload documents and update document labels. A workspace administrator can additionally
manage users, approve restores, export audit logs, and configure retention policies.

API requests use a workspace API key in the X-API-Key header. A revoked key stops working
immediately. API keys can be rotated from the workspace Security page.

Standard support responds within two business days. Enterprise support responds within
four business hours. Both plans provide email support; only enterprise includes phone support.

The public search API accepts up to 60 requests per minute per workspace. When the limit
is exceeded, the service returns HTTP 429 with a Retry-After header measured in seconds.
