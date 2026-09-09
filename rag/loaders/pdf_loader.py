"""Carga de fuentes normativas oficiales en PDF (interna, estática).

Hallazgo relevante (documentado también en docs/arquitectura.md): el único
PDF descargable de Junaeb —
`data/raw/REX-DN-00556-2026_APRUEBA-MODIFICACION-MANUAL-BAES-2026.pdf` —
contiene en realidad DOS fuentes lógicas distintas en un mismo archivo:

- Páginas PDF 1-3 y 55: el texto propio de la Resolución Exenta (vistos,
  considerandos, resuelvo, artículos finales, firma).
- Páginas PDF 4-54: el "Manual de Orientaciones Técnicas BAES 2026" anexado
  íntegro, con numeración interna propia de 1 a 51 ("Página: X de 51"),
  verificado extrayendo el texto de cada página.

Por eso este loader expone dos funciones que leen el MISMO archivo pero
devuelven fragmentos con `fuente` distinta, en vez de una función por archivo.

`data/raw/2026_MANUAL GUÍA USUARIO.pdf` existe en el repo pero NO se carga
aquí a propósito: es una guía de uso de la app Pluxee (activar tarjeta, pagar,
cambiar PIN, agregar correo) — contenido procedimental sobre acciones de
cuenta, fuera del foco normativo de restricciones de compra, y adyacente a
temas que deben ESCALAR. Decisión confirmada con el usuario.
"""

from pathlib import Path

import pypdf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESOLUCION_MANUAL_PDF = RAW_DIR / "REX-DN-00556-2026_APRUEBA-MODIFICACION-MANUAL-BAES-2026.pdf"

# Índices 0-based dentro del PDF (confirmado inspeccionando el texto extraído
# de cada página, buscando el marcador "RESUELVO" y la numeración interna
# "Página: X de 51" del manual anexado).
RESOLUCION_PAGE_INDICES = [0, 1, 2, 54]   # PDF 1-3 (vistos/considerandos/resuelvo) + PDF 55 (artículos finales/firma)
MANUAL_PAGE_INDEX_RANGE = range(3, 54)    # PDF 4-54 -> numeración interna del manual 1-51

FUENTE_RESOLUCION = "Resolución Exenta DN-00556/2026"
FUENTE_MANUAL = "Manual de Orientaciones Técnicas BAES 2026"

# Fechas extraídas del propio documento (portada del anexo: "Santiago, enero
# 2026"; fecha de firma según metadata del PDF). Aproximaciones documentadas
# aquí porque el PDF no trae una fecha de resolución explícita en el texto.
FECHA_MANUAL = "2026-01"
FECHA_RESOLUCION = "2026-02-20"


def _extraer_paginas(reader: pypdf.PdfReader, indices) -> list[dict]:
    fragmentos = []
    for i in indices:
        texto = (reader.pages[i].extract_text() or "").strip()
        if texto:
            fragmentos.append({"texto": texto, "pagina_pdf": i + 1})
    return fragmentos


def cargar_resolucion(path: str | Path = RESOLUCION_MANUAL_PDF) -> list[dict]:
    """Extrae solo el texto propio de la Resolución Exenta DN-00556/2026
    (vistos, considerandos, resuelvo, artículos finales) — sin el manual anexado."""
    reader = pypdf.PdfReader(str(path))
    fragmentos = _extraer_paginas(reader, RESOLUCION_PAGE_INDICES)
    for f in fragmentos:
        f["fuente"] = FUENTE_RESOLUCION
        f["fecha"] = FECHA_RESOLUCION
    return fragmentos


def cargar_manual_baes(path: str | Path = RESOLUCION_MANUAL_PDF) -> list[dict]:
    """Extrae el Manual de Orientaciones Técnicas BAES 2026 (51 páginas),
    anexado dentro del mismo PDF de la resolución en las páginas PDF 4-54."""
    reader = pypdf.PdfReader(str(path))
    fragmentos = _extraer_paginas(reader, MANUAL_PAGE_INDEX_RANGE)
    for pagina_manual, f in enumerate(fragmentos, start=1):
        f["fuente"] = FUENTE_MANUAL
        f["fecha"] = FECHA_MANUAL
        f["pagina_manual"] = pagina_manual  # numeración interna 1-51, para citar "sección/página X del manual"
    return fragmentos
