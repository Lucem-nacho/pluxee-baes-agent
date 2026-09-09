"""Simula el job de ingesta cada 24h del Twitter/X de Pluxee.

En esta versión, "refrescar" significa releer data/social/tweets_mock.json
(el mock no cambia solo; se edita a mano para simular nuevos posts) y
re-indexar los tweets normativos nuevos en la colección "manual_faq_social".
En producción esto correría como cron job contra la API real de X.

Pendiente de implementar.
"""

if __name__ == "__main__":
    raise NotImplementedError
