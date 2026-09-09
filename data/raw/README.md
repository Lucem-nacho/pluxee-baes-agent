# Fuentes normativas oficiales (PDF)

## `REX-DN-00556-2026_APRUEBA-MODIFICACION-MANUAL-BAES-2026.pdf`

Contiene **dos fuentes lógicas distintas en un mismo archivo** (verificado
extrayendo el texto de cada página con `pypdf`):

- **Páginas PDF 1-3 y 55**: el texto propio de la Resolución Exenta DN-00556/2026
  (vistos, considerandos, resuelvo, artículos finales, firma).
- **Páginas PDF 4-54**: el "Manual de Orientaciones Técnicas BAES 2026" anexado
  íntegro (51 páginas, numeración interna propia "Página: X de 51").

`rag/loaders/pdf_loader.py` procesa este único archivo con dos funciones
(`cargar_resolucion`, `cargar_manual_baes`) que devuelven fragmentos con
`fuente` y `fecha` distintas según el rango de páginas.

## `2026_MANUAL GUÍA USUARIO.pdf`

Pese al nombre, **no es el manual normativo**: es una guía de uso de la app
Pluxee (activar tarjeta, pagar con clave dinámica/QR, cambiar PIN, agregar
correo). Contenido procedimental sobre acciones de cuenta, fuera del foco de
"restricciones normativas de compra" del caso, y adyacente a temas que deben
ESCALAR (no ejecutar acciones sobre la cuenta).

**Decisión (confirmada con el usuario): no se indexa en esta versión.**
`rag/loaders/pdf_loader.py` no la carga a propósito. Queda en esta carpeta
solo como respaldo del archivo original.
