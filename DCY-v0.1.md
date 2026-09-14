# DCY — Dynamic Context Injection: especificación experimental v0.1

## 1. Tesis, alcance y definiciones

**Definición.** DCY es un controlador externo que, para una tarea y un estado persistente, selecciona evidencia de una base versionada, la representa bajo un presupuesto de tokens y la entrega a un LLM estándar. Su unidad de trabajo es una inferencia acotada, no una ventana virtual del transformer. ORMT es el serializador con pérdida controlada y procedencia explícita de esa evidencia.

Sea `D_v` una instantánea de conocimiento, `g_t` un goal, `s_t` el estado, `m` el modelo/tokenizador, `W_m` la ventana física total y `o_t` la reserva de salida. Entonces:

```text
e_t = Retrieve(D_v, g_t, s_t)
c_t = ORMT(e_t, g_t, s_t, m, b_t)
b_t = W_m - tokens(system + task + protocol + state) - o_t
tokens_m(c_t) <= b_t
(s_{t+1}, g_{t+1}) = Commit(Validate(Parse(LLM(c_t))), s_t)
```

`b_t` puede ser negativo: en ese caso el protocolo debe rechazar la llamada o pedir una ventana mayor. La **ventana física** incluye instrucciones, goal, evidencia, estado, formato de respuesta y salida prevista. **Contexto direccionable** es el corpus externo que DCY puede consultar en la instantánea; medirlo en tokens de `m` es posible, pero no implica que esos tokens entren al modelo. **Working set** es la evidencia efectivamente presentada en una llamada. “Virtual tokens” debe ser una etiqueta contable (`tokens_m(D_v)`), no una unidad de capacidad cognitiva. El cociente entre tamaño del corpus y ventana sólo expresa escala de almacenamiento.

El problema exacto es el costo y la pérdida de foco de presentar material irrelevante y de hacer que el LLM opere el sistema de búsqueda. DCY intenta sustituir esas operaciones por recuperación y estado deterministas **sin reducir la tasa de resolución**. No puede sustituir información ausente ni razonamiento que requiera muchos hechos simultáneos. V1 es `DB → texto acotado → LLM`; V2, que requeriría entrenamiento y/o arquitectura modificada, sería memoria latente externa. No deben mezclarse sus resultados.

## 2. Amenazas a la hipótesis e invariantes

El fallo dominante puede estar en la entrada, no en el transformer: goal mal formulado, dependencia omitida por el parser, símbolo mal resuelto, término léxico distinto del usado en código, estado anterior erróneo, fuente obsoleta, ranking que entierra una excepción o compresión que elimina una condición crítica. En 512 tokens también pueden agotarse instrucciones y reserva de salida antes de incluir evidencia suficiente. Una tarea de invariantes globales, refactorización transversal o correlación de muchas trazas puede exigir un conjunto simultáneo mayor que la ventana. Más inferencias pueden costar más tiempo y tokens que una lectura larga. Datos y comentarios del repositorio pueden intentar inyectar instrucciones; son evidencia no confiable. Hay que medir asimismo cobertura del indexador por lenguaje, coste de reconstrucción y fuga de información entre snapshots del benchmark.

Los doce invariantes son buenas **políticas de V1**, con matices:

| Regla | Decisión v0.1 |
|---|---|
| I1–I3: sin archivos, SQL ni chunks manuales | El LLM recibe entidades y puede emitir `NEED <id> <reason>` o `INSUFFICIENT`; el runtime elige spans. Prohibir toda solicitud de evidencia ocultaría errores de retrieval. |
| I4, I7 | Estado externo versionado; no asumir que una compresión anterior es un hecho. |
| I5 | Un goal activo; referencias acotadas a padre y dependencias. |
| I6 | Presupuesto por llamada **incluida la salida**; abortar antes de excederlo. |
| I8 | Recuperación byte exacta desde snapshot/hash; “source exacto” no significa que siempre quepa en el prompt. |
| I9 | DB e IR independientes de modelo; cambiar modelo exige recontar tokens, recalibrar ranking/renderer y repetir evaluación. |
| I10 | Objetivo empírico: longitud mediana y p95 del prompt no crecen con el corpus; número de llamadas y fallos sí podrían crecer. |
| I11 | Parsing, pruebas, agregaciones y verificaciones ejecutadas fuera del LLM. |
| I12 | Cada elemento debe tener razón y procedencia registradas; “relevancia” es estimada, no verdad conocida en producción. |

## 3. Antecedentes y posición honesta

| Trabajo/proyecto | Coincidencia real | Diferencia que DCY debería probar |
|---|---|---|
| [RAG original](https://arxiv.org/abs/2005.11401) y retrieval estándar | Memoria no paramétrica + recuperación antes de generar. | DB tipada de código, goals y presupuestos medidos. **DCY V1 sigue siendo una forma de RAG**, aunque su control esté fuera del LLM. |
| [GraphRAG](https://arxiv.org/abs/2404.16130), [implementación Microsoft](https://microsoft.github.io/graphrag/) | Grafo, búsqueda local/global y síntesis estructurada. | Evitar extracción y resumen por LLM en V1; privilegiar hechos de parser y fuente verificable. La agregación global de GraphRAG puede ganar en preguntas globales. |
| [MemGPT](https://arxiv.org/abs/2310.08560) / [Letta](https://www.letta.com/blog/memgpt-and-letta/) | Memoria por niveles, estado externo, ilusión de contexto ampliado. | Política de recuperación y escritura principalmente determinista; no pedir al LLM que administre memoria. El símil de memoria virtual ya existe. |
| [RETRO](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf), [LongMem](https://papers.nips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf) | Memoria externa recuperada. | Usan componentes de modelo/entrenamiento distintos; son referentes de V2, no baselines equivalentes de V1. |
| [Aider Repo Map](https://aider.chat/2023/10/22/repomap.html) | Tree-sitter, símbolos y mapa compacto de repo. | Hipótesis incremental de goal, ranking, procedencia y presupuesto estricto; un repo map bien configurado es baseline obligatorio. |
| [GraphCoder](https://arxiv.org/abs/2406.07003), [RepoCoder](https://arxiv.org/abs/2303.12570), [Repoformer](https://arxiv.org/abs/2403.10059) | Grafos/recuperación iterativa/selectiva para código. | DCY debe demostrar valor más allá de esas combinaciones, en tareas de reparación y control externo; GraphCoder estudia principalmente completion. Repoformer ya muestra que recuperar siempre puede perjudicar. |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | Navegación y herramientas para tareas reales de repositorio. | Comparar quién decide búsqueda y cuántos tokens/calls consume; la interfaz de agente puede ser mejor cuando el goal es ambiguo. |
| [MCP](https://modelcontextprotocol.io/specification/2025-11-25/basic) | Intercambio de herramientas/recursos. | Transporte e I/O; por sí solo no define selección, compresión ni presupuesto. |
| [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) | Reducen costos de atención/I/O o mejoran manejo de KV. | No virtualizan conocimiento semántico; DCY tampoco comprime KV ni acelera kernels. |

Ya existen **todas las piezas principales**. La combinación de grafo de código, recuperación, memoria externa y prompt compacto tampoco basta como reivindicación novedosa. La contribución defendible, si aparece, sería una política reproducible de **scheduling por goal + selección con procedencia + renderizado adaptable al tokenizador**, acompañada de una frontera cuantificada de calidad/costo al bajar a 512–2000 tokens. El protocolo y las trazas de fallos podrían ser un resultado útil aun si la mejora de calidad es pequeña. No sería honesto afirmar “millones de tokens entendidos”, equivalencia con atención nativa, complejidad total independiente de `N`, novedad de la metáfora de memoria virtual ni superioridad general frente a GraphRAG o agentes.

## 4. Arquitectura y responsabilidades

```text
snapshot Git + archivos ──> indexador ──> SQLite: entidades, relaciones, FTS, spans
                                               │
tarea ──> Goal Engine ──> Retrieve/Rank ──> Context IR ──> ORMT ──> LLM
              ▲                 │                │                 │
              └──── state/event log <── validator/parser <── respuesta estructurada
                                │
                         test/patch runner
```

El indexador calcula SHA-256 de bytes por archivo, descarta archivos idénticos, reparsea sólo los cambiados y actualiza en una transacción entidades, aristas y FTS. El diff Git ayuda a enumerar candidatos, pero **no es fuente de verdad**: comparar hashes detecta cambios fuera de Git; deletes y renames requieren reconciliación. Tree-sitter permite parsing incremental, pero en V0 basta reparsear el archivo cambiado, que simplifica invalidez de IDs y relaciones. El análisis de llamadas está limitado por despacho dinámico, macros y reflection; registrar aristas `resolved`, `possible` y `unresolved`. [Tree-sitter](https://tree-sitter.github.io/tree-sitter/index.html) y [SQLite FTS5](https://www.sqlite.org/fts5.html) son suficientes para la primera prueba.

El Goal Engine no crea un DAG arbitrario con otro LLM: usa plantillas por tipo de tarea (`localize → inspect → decide → patch → test`) y permite transiciones explícitas al detectar hechos o fallos. Retrieval resuelve candidatos; ORMT sólo los serializa, no decide la verdad. El validador comprueba IDs, hashes, patch y resultados de pruebas; cualquier hecho inferido queda como hipótesis hasta disponer de evidencia. Un adaptador MCP normaliza resultados en entidades con procedencia y política de tamaño; no inyecta respuestas crudas. La implementación actual incluye un servidor MCP `stdio` fino sobre el CLI: el host le pide una vista acotada y sólo el host decide qué se envía a la API del modelo. No intercepta llamadas arbitrarias a esa API. No se requiere `mlpack`, ANN ni embeddings hasta que FTS+grafo fallen de manera medida.

### SQLite v0.1

```sql
PRAGMA foreign_keys = ON;
CREATE TABLE snapshots(
  id INTEGER PRIMARY KEY, git_commit TEXT, created_at TEXT NOT NULL,
  indexer_version TEXT NOT NULL, config_hash TEXT NOT NULL
);
CREATE TABLE files(
  id INTEGER PRIMARY KEY, snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  path TEXT NOT NULL, sha256 TEXT NOT NULL, language TEXT, size_bytes INTEGER NOT NULL,
  content BLOB NOT NULL, UNIQUE(snapshot_id,path)
);
CREATE TABLE entities(
  id INTEGER PRIMARY KEY, snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  file_id INTEGER NOT NULL REFERENCES files(id), kind TEXT NOT NULL,
  qualified_name TEXT NOT NULL, signature TEXT,
  start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL,
  parser_confidence REAL NOT NULL DEFAULT 1.0,
  CHECK(start_byte >= 0 AND end_byte > start_byte)
);
CREATE TABLE relations(
  id INTEGER PRIMARY KEY,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
  src_id INTEGER NOT NULL REFERENCES entities(id),
  dst_id INTEGER REFERENCES entities(id),
  kind TEXT NOT NULL, confidence REAL NOT NULL,
  evidence_file_id INTEGER REFERENCES files(id),
  evidence_start_byte INTEGER, evidence_end_byte INTEGER
);
CREATE TABLE goals(
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, parent_id INTEGER REFERENCES goals(id),
  kind TEXT NOT NULL, statement TEXT NOT NULL, status TEXT NOT NULL,
  acceptance TEXT NOT NULL, priority INTEGER NOT NULL, budget_tokens INTEGER NOT NULL,
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id)
);
CREATE TABLE test_results(
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, goal_id INTEGER REFERENCES goals(id),
  command TEXT NOT NULL, exit_code INTEGER, summary TEXT NOT NULL,
  output_sha256 TEXT, snapshot_id INTEGER NOT NULL REFERENCES snapshots(id)
);
CREATE TABLE facts(
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, goal_id INTEGER REFERENCES goals(id),
  claim TEXT NOT NULL, status TEXT NOT NULL,
  source_entity_id INTEGER REFERENCES entities(id),
  source_sha256 TEXT, test_result_id INTEGER REFERENCES test_results(id),
  UNIQUE(run_id,goal_id,claim,source_entity_id)
);
CREATE TABLE events(
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, goal_id INTEGER REFERENCES goals(id),
  kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE entity_fts USING fts5(
  qualified_name, signature, path
);
CREATE INDEX ix_entities_name ON entities(snapshot_id,qualified_name);
CREATE INDEX ix_rel_src ON relations(snapshot_id,src_id,kind);
CREATE INDEX ix_rel_dst ON relations(snapshot_id,dst_id,kind);
CREATE INDEX ix_goals_run ON goals(run_id,status,priority);
```

`entity_fts.rowid = entities.id`; el indexador mantiene FTS en la misma transacción y una prueba de integridad lo verifica. FTS5 `bm25()` devuelve mejor coincidencia con **valor menor**, detalle que el ranking debe respetar. Los bytes originales permanecen en `files.content`; `SRC` exige `snapshot_id`, `sha256` y rango válido, y devuelve sólo el span pedido, con límite. Para escalar, puede separarse blob comprimido de metadatos sin cambiar IDs lógicos. En producción agregar tablas tipadas para tests, esquemas, diffs y outputs normalizados; incluirlas antes del experimento sólo agranda la superficie de fallo. Un identificador de entidad para logs debe incluir snapshot y `id`, nunca un alias corto global como `S84` sin tabla de resolución.

## 5. Goal Engine, Context IR y ORMT

Un goal es `{id, kind, statement, acceptance, dependencies, budget, status}`. Estados: `queued → active → resolved|failed|blocked`. Cada transición escribe un evento; sólo los eventos validados alteran hechos confirmados. El goal padre termina cuando sus criterios de aceptación y dependencias pasan. Los subgoals no son verdad: son una agenda revisable. Al fallar la recuperación, el controlador ensancha la consulta una vez y después emite `INSUFFICIENT` o eleva el presupuesto; no crea una cascada ilimitada de prompts.

Context IR es una estructura tipada previa al texto:

```json
{
  "ir_version": "0.1", "snapshot": 7, "goal": "G4",
  "items": [
    {"entity": 84, "kind": "function", "fact": "inactive_or_expired_returns_403",
     "source": {"file": 12, "sha256": "...", "bytes": [1200, 1560]},
     "status": "extracted", "confidence": 0.99, "reason": "caller_of_login"}
  ]
}
```

`status` distingue `source_exact`, `extracted`, `inferred` y `hypothesis`. Un extractor determinista puede afirmar una condición sintáctica; no puede elevar automáticamente a verdad semántica una llamada dinámica. ORMT ofrece `exact` (span), `standard` (nombres y relaciones), `compact` (abreviaturas con leyenda) y `omit`. La opción extrema `S84 !ACTIVE|expired→403` omite operador, precedencia, nullability y localización: válida para orientación, peligrosa para parche. Toda representación lleva ID y acceso al source. No afirmar “compresión sin pérdida” de texto semántico; la garantía es **recuperabilidad del original**.

Selección y codificación se optimizan por separado: `Retrieve` maximiza recall candidato; `ORMT` decide qué entra. El costo en tokens se calcula con el tokenizador real, incluidos separadores y leyenda. IDs cortos sólo ahorran si su diccionario cuesta menos que repetir nombres. El renderer deja margen de seguridad y produce un manifiesto `{item, reason, source, token_cost, dropped_reason}` para auditoría. Model-agnostic significa IR estable y renderers/calibración por modelo, no output idéntico en todos.

### Recuperación determinista y asignación de tokens

1. Anclar términos de la tarea a rutas, símbolos, endpoints y mensajes de error mediante normalización y FTS; conservar top `K0`, más coincidencias exactas de path/symbol.
2. Expandir un máximo de `h=2` saltos por relaciones relevantes al tipo de goal (caller/callee, reads/writes, tests), con fanout topado; incluir aristas no resueltas como señal de incertidumbre.
3. Puntuar con características fijadas antes del test: coincidencia exacta, BM25 normalizado dentro de la consulta, distancia de grafo, tipo de arista, cercanía al diff, test fallido, evidencia confirmada y penalización por incertidumbre. Orden estable por `(score desc, path, start_byte, id)`. No entrenar pesos con tareas de prueba.
4. Deduplicar hechos y spans; formar paquetes que mantienen condición y consecuencia juntas. Seleccionar por ganancia marginal/costo en tokens, con reserva para source exacto de la hipótesis principal.
5. Si falta un antecedente indispensable, degradar hechos de menor prioridad; si no cabe el paquete causal mínimo, declarar `budget_insufficient`. Solicitar source bajo demanda únicamente mediante ID y razón, con máximo de rondas.

Pseudocódigo:

```text
seeds = exact_matches(goal) ∪ top_fts(goal, K0)
candidates = bounded_graph_expand(seeds, depth=2, fanout=F)
ranked = stable_sort(score(c, goal, state), candidates)
bundles = provenance_preserving_dedup(ranked)
for bundle in bundles:
    choose best of {exact, standard, compact, omit}
    if marginal_utility / exact_token_cost fits remaining budget: emit
if missing_required_premise: widen_once_or_abstain()
```

Una función de utilidad posible es `relevance × evidence_quality × novelty × goal_coverage`, menos riesgo de pérdida; no se conoce perfectamente y debe calibrarse en validación. Usar knapsack exacto no merece la complejidad inicial. Reparto inicial **para una ventana total de 2K**, no universal: instrucciones/tarea/protocolo 300, estado 120, evidencia 1100, salida 400, margen 128 (=2048). Para 512: 95 + 35 + 210 + 140 + 32 = 512; es una condición adversaria y puede no permitir reparación. El output de un patch extenso no cabe: aplicar ediciones estructuradas pequeñas o declarar fallo, contabilizando todas las llamadas.

### API preliminar

El contrato interno es JSON Lines versionado; no exponer SQL al modelo:

```json
{"op":"start","v":"0.1","snapshot":7,"task":"Login returns 403","model":"M","window":2048}
{"op":"next","run":"R31","goal":"G4","max_input":1500,"reserve_output":400}
{"op":"context","run":"R31","goal":"G4","items":[84,44],"prompt_tokens":1432,"manifest":"M31"}
{"op":"observation","run":"R31","goal":"G4","kind":"NEED_SOURCE","entity":44,"reason":"check expiry comparator"}
{"op":"source","snapshot":7,"entity":44,"max_bytes":800}
{"op":"commit","run":"R31","goal":"G4","kind":"hypothesis","claim":"expiry comparator inverted","evidence":[44]}
```

`next` arma el prompt y verifica `input + reserved_output <= window`; `source` verifica snapshot/hash; `commit` valida esquema y procedencia. El LLM sólo puede responder `ANSWER`, `HYPOTHESIS`, `EDIT`, `NEED_SOURCE` o `INSUFFICIENT`; la lógica de control es del runtime. El adaptador del modelo registra tokens reportados **y** recalculados; si difieren, falla cerrado. No se introducen herramientas de búsqueda libres en el prompt de DCY.

## 6. Dos ejecuciones de ejemplo

**Caso local: 403 en login.** La tarea menciona “login 403”; FTS encuentra endpoint `Auth.login`, la expansión de llamadas incluye `Subscription.validate`, y el test fallido enlaza al endpoint. Goal `G1/localize` presenta: `S12 Auth.login CALL>S84; T91 login_active_subscription expects 200; S84 status/expiry→403` con procedencias. El modelo emite `NEED_SOURCE S84: check expiry comparator`. El runtime entrega exactamente el span `S84` en la siguiente llamada y verifica su SHA. Supongamos que el código contiene `if (expiry > now) return 403;`: el modelo propone hipótesis “comparador invertido”; el validador la mantiene como hipótesis hasta que un test rojo y un parche mínimo (`>` por `<`) produzcan test verde. `G2/patch` recibe source exacto y condición de aceptación, emite `EDIT` con rango byte y hash previo; el patch runner aplica y ejecuta la prueba. `G3/validate` recibe resultado resumido, no el log entero; registra `ROOTCAUSE` con hash, test y diff. Si `expiry` es nullable o la semántica de zona horaria no cabe, el sistema declara incertidumbre o sube a 2K: el resumen `expired→403` no justifica el cambio por sí solo.

**Caso global que tensiona DCY.** “Cambiar todas las rutas que interpretan `subscription.status` como activo, incluyendo workers y SQL.” El graph scan lista 64 usos, 11 ambiguos y 3 lenguajes. Un prompt de 512 tokens no puede presentar 64 contextos ni verificar interacciones simultáneas. El runtime puede ejecutar una agregación determinista por patrón y partir la reparación por módulo, pero debe comprobar al final invariantes globales con tests y diff completo. Si los 11 usos ambiguos requieren correlación humana o LLM conjunta, `budget_insufficient` o calidad inferior es el resultado correcto. Este caso evita confundir direccionabilidad con capacidad de síntesis global.

## 7. Caché y prefetch

Caché de tres niveles: **hot** = bundles y source del goal activo; **warm** = top candidatos de dependencias inmediatas; **cold** = SQLite/FTS y blobs. Clave de caché: `(snapshot, goal kind/text hash, state facts hash, ranker version, renderer version, model/tokenizer, budget)`. Invalidar ante cambio de fuente, estado que altere ranking o versión de algoritmo. Prefetch sólo de top candidatos de `patch/test` cuando `root_cause` supera un umbral calibrado; no enviar tokens anticipadamente al modelo. Medir hit rate, trabajo desperdiciado y latencia p95. Prefetch puede empeorar caché y CPU; desactivarlo en ablation.

## 8. Complejidad y costo real

Para un transformer denso de `L` capas y ancho `d`, un prefill de `p` tokens cuesta esquemáticamente `O[L(p d² + p² d)]`, con constantes, arquitectura y kernels relevantes. Decodificar `q` tokens con KV cache cuesta aproximadamente `O[L(q d² + d(pq + q(q−1)/2))]`, más lectura/escritura KV y overhead; sin KV cache, recalcular el prefijo sería mucho peor. KV por secuencia escala aproximadamente con `L × (p+q) × n_kv_heads × head_dim`, formato y batch. En ventanas de 512–2K el término de proyecciones/MLP, ancho de banda, launch overhead, batch y latencia de servidor pueden dominar el `p²`. FlashAttention reduce I/O sin eliminar trabajo aritmético cuadrático de atención exacta; PagedAttention mejora gestión de KV, no la relevancia del contexto. [FlashAttention](https://arxiv.org/abs/2205.14135), [vLLM](https://arxiv.org/abs/2309.06180).

Con `a` llamadas DCY y longitud `p_i,q_i`, la comparación correcta es:

```text
T_DCY = T_index_amortized + Σ_i(T_retrieve_i + T_render_i + T_prefill(p_i)
         + T_decode(p_i,q_i) + T_network_i + T_validation_i) + T_tests
T_baseline = T_baseline_index_amortized + Σ_j(T_prefill(P_j)
             + T_decode(P_j,Q_j) + T_network_j) + T_tests
```

La fórmula `A(N/G)² + R(N)` sólo serviría bajo supuestos artificiales: particiones iguales y correctas, un único prefill por goal, sin tokens generados, sin instrucciones repetidas, sin cruces, sin costos fijos y con atención como cuello de botella. `R(N)` no es un escalar único: indexación inicial `Ω(bytes del corpus)`, actualización proporcional a archivos cambiados más dependencias invalidadas, consulta FTS según postings y grafo según nodos/aristas visitados, agregaciones globales potencialmente `Ω(N)`. Si `A` crece con `G`, la ventaja se erosiona. Prefijos compartidos/KV cache pueden abaratar baselines y DCY; medir configuración idéntica y tiempo real, no extrapolar FLOPs.

## 9. Benchmark reproducible y ablations

La unidad es tarea en snapshot inmóvil, con gold de spans necesarios y test/criterio de éxito. Estratos: localización de hecho, reparación local de 1–2 archivos, multi-hop 3–5 archivos, cambio global, fallo del parser/reflectivo y perturbaciones de nombres (mismo significado, poca coincidencia léxica). 100k, 1M y 5M+ tokens son **tamaño de corpus bajo un tokenizador fijado**, no todos los repositorios/tareas de SWE-bench tienen esos tamaños. Construir dos colecciones: issues reales con tests ocultos y repos sintéticos controlados con distractores, para aislar escalabilidad. Los casos sintéticos no bastan para afirmar utilidad real. Los issues reales deben fijar commit anterior al arreglo, excluir el patch y pruebas ocultas del índice, y auditar contaminación. [SWE-bench](https://github.com/swe-bench/SWE-bench/blob/main/README.md) es punto de partida, no benchmark único; OpenAI ha cuestionado recientemente la capacidad de SWE-bench Verified para diferenciar sistemas frontera, por defectos de pruebas y contaminación. [Evaluación crítica](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/).

Seis condiciones: A raw/truncado bajo la misma ventana (más un **raw largo** como techo externo, no comparación de presupuesto); B chunks FTS/BM25 y opcional embeddings declarados; C graph RAG de código sin resumen LLM adicional; D repo map estilo Aider; E agente con búsqueda de archivos y herramientas; F DCY. Mismo checkpoint, cuantización, tokenizer, plantilla, temperatura, hardware, harness de edición, tests y límites de llamadas/tiempo. Ventanas 512, 1024, 2048, 4096, 8192 incluyen salida. Algunos modelos no funcionan a 512 por mínimo de plantilla o capacidad de instrucción; reportar inviabilidad, no excluirla. Separar comparaciones **igual presupuesto por llamada**, **igual presupuesto total de tokens**, **igual tiempo** e **igual llamadas**; de otro modo F gana por cómputo adicional o E pierde por herramientas artificialmente capadas. Para A, raw completo sólo es posible si cabe; no llamar “raw” a un truncado como si fuera contexto completo. Para GraphRAG, excluir resúmenes hechos por un LLM extra o contabilizar sus tokens/costo de indexación.

Registrar por tarea y llamada: éxito por tests y revisión de patch, input/output tokens reales, suma de tokens procesados, latencia total, TTFT por llamada y primer TTFT, inferencia, retrieval, render, indexación cold/incremental amortizada por `k` tareas, CPU/RAM/VRAM pico, llamadas, bytes de I/O, caché, fallos por etapa y goal completion. Reportar p50/p95 y éxito con intervalo de confianza pareado por tarea; publicar listas de tareas, seeds, pesos, versiones, hashes, hardware, comandos y trazas sin gold oculto. Context precision/recall requiere etiquetar spans **suficientes**, no sólo spans del patch; medir `Recall@B = fracción de hechos gold necesarios presentes o recuperados antes de decidir`, `EvidencePrecision@B = fracción de tokens de evidencia que soportan un hecho gold`, adjudicada por humano con acuerdo entre anotadores. `SourceAccuracy = solicitudes SRC que devuelven bytes/hash correctos / solicitudes válidas` mide integridad mecánica, no pertinencia.

Las métricas propuestas necesitan cirugía:

| Propuesta | Problema | Métrica operativa |
|---|---|---|
| VCR = virtual/physical | Inflable añadiendo basura; no mide acceso útil. | `AddressableScale = tokens(corpus snapshot)/W`, siempre junto a `Recall@B`, éxito y pendiente de costo al crecer corpus. |
| ID = relevante/inyectado | “Información” no tiene unidad natural; relevancia dependiente de tarea. | `EvidencePrecision@B` por tokens o spans, con anotación y gold suficiente. |
| RD = razonamiento útil/tokens LLM | “Útil” es contrafactual y chain-of-thought no observable de forma fiable. | `Success per 1M total LLM tokens`, `successful tasks/hour`, `calls per success`; tokens de tool/search/output por categoría sólo como diagnóstico. |

Ablations mínimas: FTS solo; +grafo; +goals; +ORMT compacto; +estado; +prefetch; source exacto habilitado/deshabilitado; oracle retrieval; ranking con spans correctos pero orden aleatorio; distintos `B`; repos con aristas incompletas. **Oracle retrieval** mide el techo del renderer/modelo; si falla, mejorar ranking no ayudará. Prefetch y embeddings quedan fuera del primer resultado científico salvo evidencia de necesidad.

## 10. Riesgo científico, criterios y publicación

Hipótesis falsables:

1. **H1 calidad/costo:** en tareas locales/multi-hop a 2K, DCY supera al mejor baseline de igual modelo y presupuesto total en al menos **10 puntos porcentuales de éxito** o mantiene éxito dentro de **−3 puntos** con **≥30% menos tokens LLM y sin aumento >20% de latencia mediana**. Son umbrales de decisión propuestos, no estimaciones.
2. **H2 escala:** de 100k a 1M tokens, con tareas emparejadas en dificultad, `p95(prompt_tokens)` se mantiene bajo la ventana y éxito cae menos de **5 puntos**; `p95(retrieval+render)` crece menos de 2× después de índice caliente. Medir aparte 5M+.
3. **H3 componentes:** a 2K, quitar grafo o ORMT reduce éxito o aumenta tokens bajo comparación pareada; si no, esos componentes no se justifican. Goal prefetch sólo se conserva si reduce latencia p95 neta sin perjudicar éxito.

El mínimo para invertir más es H1 en un conjunto real y uno controlado, con intervalo pareado que excluya una mejora trivial y trazas auditables; no basta alto VCR. Un resultado negativo fuerte es que oracle retrieval a 512/1K fracasa por falta de espacio causal o que, a 2K, un repo map/FTS afinado iguala DCY en calidad y lo mejora en latencia/costo. El resultado también puede ser **régimen limitado**: DCY útil para bugs locales, inútil para síntesis global. Publicar límites y matriz de errores; no seleccionar sólo tareas ganadas.

## 11. Implementación y orden de trabajo

Roadmap corregido: benchmark/harness y gold **antes** de optimizar el runtime. Crear un prototipo C++20 con CMake, SQLite/FTS5 y Tree-sitter para **un lenguaje y un tipo de tarea**; la complejidad de gramáticas múltiples impediría interpretar fallos. Usar JSONL para trazas y Python sólo para ejecutar/analizar benchmark si simplifica. Primero snapshot + hashes + source exacto + símbolos + aristas básicas; después ranking FTS/grafo; después dos renderings ORMT y medidor de tokens del modelo; después goals/estado. Ejecutar baselines en paralelo conceptual bajo el mismo harness. Embeddings/ANN, `mlpack`, MCP, prefetch y V2 son hitos posteriores condicionados por ablations. El índice incremental se prueba con un cambio real por archivo antes de escalar. No hace falta crear una plataforma de agentes.

```text
dcy/
  CMakeLists.txt
  include/dcy/{index,db,retrieve,ir,render,goal,protocol}.hpp
  src/{index,db,retrieve,ir,render,goal,protocol,main}.cpp
  sql/schema.sql
  grammars/README.md
  tests/{index_invalidation,source_integrity,budget}.cpp
  bench/{tasks.jsonl,run.py,score.py,baselines.py,README.md}
  docs/{protocol-v0.1,measurement}.md
```

La estructura es propuesta, no archivos ya implementados. El ejecutable mínimo ofrece `dcy index <repo>`, `dcy inspect <task> --budget 2048`, `dcy source <snapshot:id>` y `dcy bench <manifest>`. Un adaptador de inferencia puede invocar una API local desde el harness; el núcleo C++ no depende del proveedor. Pruebas necesarias: invalidación de archivo borrado/renombrado, recuperación byte exacta y rechazo cuando `input+output>W`; el benchmark cubre comportamiento, no tests que repiten la implementación.

### Entregables finales solicitados

**A. DCY v0.1 mínima (propuesta experimental original).** Snapshot inmutable, esquema anterior, un parser, FTS5+grafo hasta dos saltos, un goal activo, dos renderings, medidor de tokens, `NEED_SOURCE`, hechos/eventos versionados, límite de 2K y trazas completas. Excluir embeddings, prefetch y modelo modificado. La implementación actual incorporó un adaptador MCP `stdio` acotado por petición posterior; esto no implica que snapshot inmutable, tokenizador exacto ni el resto de esta propuesta ya estén implementados.

**B. Experimento de siete días.** Día 1: fijar modelo/hardware, 20–30 tareas y gold, corpus 100k/1M; día 2: harness y baselines raw/FTS/repo map; día 3: índice/snapshot; día 4: retrieval/IR/source; día 5: ORMT/goal y ejecución; día 6: ablations y auditoría de fallos; día 7: repetir seeds, intervalos y decisión. GraphRAG y agente pueden requerir más tiempo para implementación justa: si no están listos, declarar el primer experimento **piloto**, sin proclamar victoria sobre ellos. Un tamaño de 20–30 tareas no da potencia para cambios pequeños; las 7 jornadas establecen viabilidad, no publicación.

**C. MVP mínimo.** CLI C++20/SQLite/Tree-sitter + harness Python; una gramática, tres comandos principales, 2K tokens, reparaciones locales. Éxito significa producir trazas que expliquen recuperación, prompt, costo y resultado, incluso si la hipótesis falla.

**D. Tres hipótesis falsables.** H1 calidad/costo, H2 escala y H3 valor marginal de componentes, con umbrales y controles arriba.

**E. Lectura previa al código serio.** Prioridad: [MemGPT](https://arxiv.org/abs/2310.08560), [Aider Repo Map](https://aider.chat/2023/10/22/repomap.html), [GraphCoder](https://arxiv.org/abs/2406.07003), [GraphRAG](https://arxiv.org/abs/2404.16130), [RepoCoder](https://arxiv.org/abs/2303.12570), [Repoformer](https://arxiv.org/abs/2403.10059), [SWE-agent](https://arxiv.org/abs/2405.15793), [RAG](https://arxiv.org/abs/2005.11401), [RETRO](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf), [LongMem](https://papers.nips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf), [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9.pdf), [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention](https://arxiv.org/abs/2309.06180), [SWE-bench](https://arxiv.org/abs/2310.06770); para implementación, [Tree-sitter](https://tree-sitter.github.io/tree-sitter/index.html), [SQLite FTS5](https://www.sqlite.org/fts5.html) y [MCP](https://modelcontextprotocol.io/specification/2025-11-25/basic). Leer los dos primeros y los tres de recuperación de código antes de decidir novedad; leer RETRO/LongMem para separar V2.

## Fuentes primarias

- Lewis et al., “[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401),” 2020.
- Packer et al., “[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560),” 2023; Letta, “[MemGPT Is Now Part of Letta](https://www.letta.com/blog/memgpt-and-letta/),” 2024.
- Edge et al., “[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130),” 2024; Microsoft, [GraphRAG documentation](https://microsoft.github.io/graphrag/).
- Aider, “[Building a better repository map with tree sitter](https://aider.chat/2023/10/22/repomap.html),” 2023.
- GraphCoder authors, “[GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model](https://arxiv.org/abs/2406.07003),” 2024; RepoCoder authors, “[Repository-Level Code Completion Through Iterative Retrieval and Generation](https://arxiv.org/abs/2303.12570),” 2023; Wu et al., “[Repoformer: Selective Retrieval for Repository-Level Code Completion](https://arxiv.org/abs/2403.10059),” 2024.
- Yang et al., “[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793),” 2024; Jimenez et al., “[SWE-bench](https://arxiv.org/abs/2310.06770),” 2023; [SWE-bench official repository](https://github.com/swe-bench/SWE-bench/blob/main/README.md).
- Borgeaud et al., “[Improving Language Models by Retrieving from Trillions of Tokens](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf),” ICML 2022; Wang et al., “[Augmenting Language Models with Long-Term Memory](https://papers.nips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf),” NeurIPS 2023.
- Liu et al., “[Lost in the Middle: How Language Models Use Long Contexts](https://aclanthology.org/2024.tacl-1.9.pdf),” TACL 2024.
- Dao et al., “[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135),” 2022; Kwon et al., “[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180),” 2023.
- Tree-sitter, [official documentation](https://tree-sitter.github.io/tree-sitter/index.html); SQLite, [FTS5 documentation](https://www.sqlite.org/fts5.html); Model Context Protocol, [specification](https://modelcontextprotocol.io/specification/2025-11-25/basic).
- OpenAI, “[Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/),” 2026.
