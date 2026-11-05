# Transcriptor

Herramienta local para transcribir videos MP4 y audios WAV con Whisper de
Lemonade y generar resúmenes con Gemma. Requiere Python 3.12, `uv`, FFmpeg con
`ffprobe` y el servidor local de Lemonade en `http://localhost:13305`.

## Preparar el entorno

```powershell
uv sync
```

## Procesar un archivo

```powershell
uv run transcriptor "E:\videos\reunion.mp4"
```

Por defecto se usan `Whisper-Large-v3-Turbo` y `gemma4-it-e4b-FLM`. Los
resultados se guardan en una carpeta `<nombre>_transcripcion` junto al archivo:

- `<nombre>.txt`: transcripción con marcas de tiempo.
- `<nombre>.srt`: subtítulos con marcas de tiempo.
- `<nombre>.json`: segmentos y metadatos estructurados.
- `<nombre>_resumen.md`: resumen detallado de Gemma.

Los archivos largos se normalizan a WAV mono de 16 kHz y se procesan
secuencialmente en bloques de cinco minutos, con cinco segundos de solapamiento.
El estado local permite reanudar bloques ya transcritos si el proceso se
interrumpe. La transcripción completa se conserva antes de solicitar el resumen.

### Opciones

```powershell
uv run transcriptor --help
uv run transcriptor "reunion.wav" --summary-model gemma4-it-12b-FLM
uv run transcriptor "reunion.mp4" --language auto --transcribe-only
uv run transcriptor "reunion.mp4" --output-dir "E:\Transcripciones"
```

El resumen se divide en partes de hasta 22 000 caracteres y, cuando hay varias,
se consolidan en una síntesis final. La API solo recibe audio de Lemonade en la
red local; Gemma recibe texto para resumir, nunca el archivo de audio o video.

## Unir y resumir varias transcripciones

Para tratar varias reuniones o partes consecutivas como un solo tema, pasa sus
archivos TXT **en orden cronológico** al comando independiente:

```powershell
uv run resumir-transcripciones `
  "E:\transcripciones\sesion-1.txt" `
  "E:\transcripciones\sesion-2.txt"
```

Genera un solo resumen Markdown junto al primer TXT, con sufijo
`_resumen_conjunto.md`. Para elegir otra ubicación o el modelo Gemma:

```powershell
uv run resumir-transcripciones "primera.txt" "segunda.txt" `
  --output "E:\Transcripciones\aplicaciones_resumen.md" `
  --model gemma4-it-12b-FLM
```

Los textos se envían a Gemma por fragmentos, en el orden indicado. El comando
no modifica ni concatena físicamente los TXT originales.

## Pruebas

```powershell
uv run python -m unittest discover -s tests
```