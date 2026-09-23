# Estudio: NVIDIA Nemotron 3 Diarization en scribe

Fecha: 2026-09-23 · Estado: **estudio, sin implementar ni medir todavía**

## Qué es

[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), publicado por NVIDIA el 23-09-2026:

- Modelo de diarización end-to-end (sucesor de Streaming Sortformer), ~100M parámetros.
- Hasta **8 hablantes**, canales en orden de aparición, resolución de 10 ms.
- Un solo checkpoint sirve para **streaming** (latencia de 0,32 s a 10 s) y **offline** (buffer de 30,4 s).
- Licencia **OpenMDW 1.1** (permite uso comercial). **Ojo:** el repo `-preview` usa una licencia *solo para evaluación*; hay que usar el de disponibilidad general.
- Solo diariza: **no transcribe**. El cambio mejora la atribución de hablantes; el WER se queda igual.

## Por qué es viable ya

FluidAudio, nuestra única dependencia de inferencia, lo integró el mismo día
([#883](https://github.com/FluidInference/FluidAudio/pull/883), release **v0.17.1**):

- `Nemotron3Diarizer` con API offline (`processComplete`) y streaming
  (`appendAudio` / `processBufferedAudio` / `finishStream`), idéntica frame a frame.
- CoreML en ANE, pesos en `FluidInference/nemotron-3-diarization-coreml`, construidos a partir del checkpoint GA (no del preview).
- Presets: `fast` / `low` (1,04 s de latencia), `fast32` (2,88 s, el recomendado),
  `fast128` (10,56 s), `offline` (30,4 s, solo en GPU) y variantes W8A8 de 95 MB.
- La doc de FluidAudio explica cómo juntarlo con ASR en streaming: `consumeWordTimings()` + el
  hablante con más actividad en cada palabra.

scribe usa ahora FluidAudio **0.13.4** con pyannote community-1 (`OfflineDiarizerManager`).

## Números publicados (sin medir aún en scribe)

| Fuente | Dataset / protocolo | Resultado |
|---|---|---|
| FluidAudio (#883) | AMI **MHM**, 16 reuniones, collar 0 | fast128 **9,36%** · offline 9,47% · fast32 9,53% · low 9,75% DER |
| FluidAudio Benchmarks.md | AMI **SDM**, 16 reuniones, pyannote community-1 offline | 10,6% DER (12/16 con el nº de hablantes correcto) |
| FluidAudio Benchmarks.md | VoxConverse, pyannote, stepRatio 0,1 (nuestra config) | 13,89% DER |
| NVIDIA | DIHARD III | 12,73% (30,4 s) · 13,55% (0,32 s) |
| NVIDIA | CALLHOME-part2 | 9,10% (antes 10,32%) |
| NVIDIA | AISHELL-4, baja latencia | 9,8% frente a 27,2% de Streaming Sortformer v2.1 |

Las filas de AMI **no son comparables** entre sí: MHM y SDM usan micrófonos distintos, y el collar y
el tratamiento del solapamiento también cambian. La propia NVIDIA reconoce que pierde contra pyannote
community-1 **en AMI** y gana en el resto de benchmarks. Lo previsible es una mejora en conversación
con solapamiento, llamadas telefónicas y audio variado, y un empate en reuniones limpias.
**Hay que medirlo antes de cambiar el diarizador por defecto.**

## Encaje con los objetivos

### 1. Mejorar la diarización en ficheros (lo que scribe hace hoy)

Probablemente sí, pero con estas limitaciones:

- **Máximo 8 hablantes.** Earnings-21 tiene ficheros con 8 (One Gas) y hay reuniones con más.
  pyannote no tiene ese límite. Hay que mantener pyannote como alternativa para `--speakers > 8`.
- **`--speakers` no se aplica.** El modelo no acepta un número de hablantes objetivo. Como mucho,
  se podría reagrupar a posteriori (fusionar canales). Hoy el README lo presenta como la mejor
  forma de mejorar la precisión.
- **Tamaño:** 190 MB (o 95 MB en W8A8) frente a ~50 MB de pyannote.
- **Velocidad:** debería ser igual o más rápido. pyannote con `stepRatio=0.1` baja scribe de
  ~130x a ~30x en tiempo real. Nemotron no tiene la fase de embeddings y clustering.

### 2. Tiempo real

Ahora mismo scribe solo procesa ficheros. Para una transcripción en vivo haría falta:

| Pieza | Opción en FluidAudio | Nota |
|---|---|---|
| Captura | `AVAudioEngine` (mic) / ScreenCaptureKit (audio del sistema) | Hay que escribirla en scribe |
| ASR en streaming (EN) | Parakeet Unified 0.6B | 2,21% WER en LibriSpeech en streaming (TDT v3: ~2,5%), ~1 s de chunk, con puntuación |
| ASR en streaming (ES/FR/DE/IT/PT) | Nemotron 3.5 Streaming, tier `latin` | FLEURS ES 4,80% frente a 4,5% de TDT v3 en batch |
| Diarización en streaming | Nemotron 3 `fast` (1,04 s) o `fast32` (2,88 s) | `fast` deja hueco en el ANE para el ASR |

Es viable técnicamente. Todo el trabajo del modelo lo hace FluidAudio; en scribe hay que escribir la
captura, la unión de palabras con hablantes y la salida incremental.

### 3. Mejorar el WER

Este modelo no cambia el WER. Con lo que ya ofrece FluidAudio hay una opción aparte:
**Parakeet Unified en batch** para inglés (1,68% frente a ~2,5% de TDT v3 en LibriSpeech, y con
puntuación). TDT v3 se quedaría para el resto de idiomas.

## Riesgos

1. **Salto de FluidAudio de 0.13.4 a ≥0.17.1.** Desde FluidAudio v0.15.6 (#801/#802) cambió la semántica de
   `clusteringThreshold`: ahora es una distancia euclídea y `nuevo = sqrt(2 − 2·antiguo)`.
   El `0.7` que usamos en `Transcribe.swift` equivale a **~0.775** en la API nueva.
   Si no se ajusta, **la diarización con pyannote empeora aunque no toquemos nada más**.
2. **La evaluación actual no mide la diarización.** `experiments/eval` solo calcula WER (quita
   las etiquetas de hablante). Sin DER, o al menos WDER/cpWER, no podemos decir que "mejoran los
   scores de diarización".
3. Límite de 8 hablantes y `--speakers` sin efecto (ver arriba).
4. Los pesos gated en HF solo aplicaban antes de la GA. Hay que confirmar que la descarga funciona
   sin `HF_TOKEN`.

## Plan propuesto

1. **Subir FluidAudio a 0.17.1** y remapear `clusteringThreshold` a 0.775. Comprobar que el WER
   de Earnings-21, TED-LIUM y MLS no cambia.
2. **Añadir DER y WDER a `experiments/eval`:** AMI SDM y VoxConverse (con RTTM, vía
   `fluidaudiocli diarization-benchmark` / `nemotron3-benchmark`) y Earnings-21 (que trae etiquetas
   de hablante) para WDER de extremo a extremo. Guardar la línea base de pyannote.
3. **Añadir `--diarizer pyannote|nemotron3`** (con `--diarizer-preset`) en `transcribe` y medir
   los presets `offline`, `fast128` y `fast32` frente a pyannote.
4. **Si Nemotron gana:** usarlo por defecto cuando no se pase `--speakers` o sea ≤ 8, y recurrir a
   pyannote en los demás casos.
5. **Tiempo real:** nuevo subcomando `scribe live` con mic → Parakeet Unified o Nemotron 3.5
   `latin` + Nemotron 3 `fast`, y salida incremental en txt o JSONL.

Nota: los pasos 1 a 3 necesitan un Mac con Apple Silicon (CoreML y ANE). No se pueden ejecutar en
CI Linux.

## Fuentes

- Blog de NVIDIA: https://huggingface.co/blog/nvidia/nemotron-diarization
- Model card: https://huggingface.co/nvidia/Nemotron-3-Diarization
- Preview (solo evaluación): https://huggingface.co/nvidia/Nemotron-3-Diarization-preview
- MarkTechPost: https://www.marktechpost.com/2026/09/23/nvidia-releases-nemotron-3-diarization/
- Unite.AI: https://www.unite.ai/nvidia-releases-nemotron-3-diarization-open-weight-speaker-model/
- FluidAudio: `Documentation/Diarization/Nemotron3.md`, `Documentation/Benchmarks.md`, commit `3ca01bb` (#883)
