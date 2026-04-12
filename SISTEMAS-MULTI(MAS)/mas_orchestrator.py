"""
mas_orchestrator.py  ·  Nano-Orchestrator v2 — Memory + Self-Healing Edition
=============================================================================
Arquitectura:
  Requerimiento
    └─► [MemoryLayer]  ──(hit > 0.95)──► Respuesta Directa (0 costo API)
          │ (miss)
          ▼
    [Parallel Generation]  (GPT-4o · Claude · Gemini — async)
          ▼
    [Synthesizer / Scorer]  → solución candidata
          ▼
    [Doc Hunter + Code Auditor]  → reporte de errores
          │ (errors found)
          ▼
    [Self-Healing Loop]  → única ronda de corrección vía Synthesizer
          ▼
    [Memory.store()]  → guarda solución aprobada
          ▼
    Resultado Final
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx                         # pip install httpx
import chromadb                      # pip install chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mas")


# ──────────────────────────────────────────────────────────────────────────────
# Configuración central  (edita aquí, no disperses os.environ por todo el código)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # API keys cargadas dinámicamente desde el entorno (evita subirlas a GitHub)
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")

    # Generadores Activos (Smart Fallbacks)
    use_openai: bool = True
    use_claude: bool = False     # Apagado por defecto para evitar gastos si no hay API key de pago
    use_gemini: bool = True

    # Modelos
    openai_model: str = "gpt-4o"
    claude_model: str = "claude-3-opus-20240229"
    gemini_model: str = "gemini-1.5-pro-latest"
    ollama_model: str = "codellama"          # modelo local para auditoría

    # Umbrales
    memory_similarity_threshold: float = 0.95   # coseno; 1.0 = idéntico
    self_healing_max_rounds: int = 3             # Hasta 3 intentos de corrección autónoma

    # ChromaDB
    chroma_persist_dir: str = "./mas_memory"
    chroma_collection: str = "approved_solutions"

    # Timeouts (segundos)
    cloud_timeout: float = 60.0
    ollama_timeout: float = 90.0


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Tipos de datos internos
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Generation:
    model: str
    content: str
    latency: float
    error: Optional[str] = None


@dataclass
class AuditReport:
    syntax_errors: list[str] = field(default_factory=list)
    missing_methods: list[str] = field(default_factory=list)
    version_alerts: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.syntax_errors or self.missing_methods or self.version_alerts)

    def summary(self) -> str:
        parts: list[str] = []
        if self.syntax_errors:
            parts.append("SYNTAX: " + "; ".join(self.syntax_errors))
        if self.missing_methods:
            parts.append("MISSING_METHODS: " + "; ".join(self.missing_methods))
        if self.version_alerts:
            parts.append("VERSION_ALERTS: " + "; ".join(self.version_alerts))
        return "\n".join(parts) if parts else "OK"


@dataclass
class OrchestratorResult:
    solution: str
    source: str                  # "memory" | "generated" | "healed"
    audit: AuditReport
    generations: list[Generation] = field(default_factory=list)
    memory_id: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# 1. CAPA DE MEMORIA SEMÁNTICA  (ChromaDB local)
# ──────────────────────────────────────────────────────────────────────────────
class MemoryLayer:
    """
    Almacena soluciones aprobadas como vectores semánticos.
    Consulta antes de gastar tokens en la nube.
    """

    def __init__(self) -> None:
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"   # ~80 MB, se descarga una sola vez
        )
        client = chromadb.PersistentClient(path=CFG.chroma_persist_dir)
        self.col = client.get_or_create_collection(
            name=CFG.chroma_collection,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("MemoryLayer: %d soluciones en caché", self.col.count())

    def query(self, requirement: str) -> Optional[tuple[str, float]]:
        """
        Busca la solución más similar al requerimiento.
        Devuelve (solución, similitud) o None si no hay candidato suficientemente cercano.
        
        ChromaDB devuelve distancias coseno en [0, 2]:
            distancia 0  → idéntico  → similitud 1.0
            distancia 2  → opuesto   → similitud -1.0
        Convertimos con:  similitud = 1 - distancia / 2
        """
        if self.col.count() == 0:
            return None

        results = self.col.query(
            query_texts=[requirement],
            n_results=1,
            include=["documents", "distances", "metadatas"],
        )
        if not results["documents"] or not results["documents"][0]:
            return None

        raw_distance = results["distances"][0][0]        # coseno en [0, 2]
        similarity = 1.0 - raw_distance / 2.0

        log.info("MemoryLayer: mejor similitud = %.4f  (umbral = %.2f)",
                 similarity, CFG.memory_similarity_threshold)

        if similarity >= CFG.memory_similarity_threshold:
            solution = results["documents"][0][0]
            return solution, similarity
        return None

    def store(self, requirement: str, solution: str, metadata: dict | None = None) -> str:
        """Guarda solución aprobada. Retorna el ID asignado."""
        doc_id = hashlib.sha256(requirement.encode()).hexdigest()[:16]
        self.col.upsert(
            ids=[doc_id],
            documents=[solution],
            metadatas=[{
                "requirement_snippet": requirement[:200],
                "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **(metadata or {}),
            }],
        )
        log.info("MemoryLayer: solución guardada con id=%s", doc_id)
        return doc_id


# ──────────────────────────────────────────────────────────────────────────────
# 2. DOC HUNTER  (AST + PyPI + versiones locales)
# ──────────────────────────────────────────────────────────────────────────────
# Mapa de firmas conocidas por versión para paquetes críticos.
# Amplía según necesites.
KNOWN_API_SIGNATURES: dict[str, dict[str, set[str]]] = {
    "google-generativeai": {
        # >= 0.5.x  — nueva API
        "0.5": {"GenerativeModel", "configure", "embed_content", "count_tokens"},
        # < 0.4     — API legacy
        "0.3": {"generate_text", "chat", "embed_text"},
    },
    "openai": {
        "1.x": {"OpenAI", "AsyncOpenAI", "chat.completions.create"},
        "0.x": {"Completion.create", "ChatCompletion.create"},
    },
}


def _get_installed_version(package: str) -> Optional[str]:
    """Consulta la versión instalada sin importar el paquete."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _get_pypi_latest(package: str) -> Optional[str]:
    """Consulta PyPI para la versión latest (síncrono, solo en Doc Hunter)."""
    try:
        resp = httpx.get(f"https://pypi.org/pypi/{package}/json", timeout=8)
        resp.raise_for_status()
        return resp.json()["info"]["version"]
    except Exception:
        return None


def _extract_imports(code: str) -> dict[str, str]:
    """
    Devuelve {alias_o_nombre: paquete} de todos los imports del código.
    Ejemplo: import google.generativeai as genai → {"genai": "google-generativeai"}
    """
    aliases: dict[str, str] = {}
    pkg_map = {
        "google.generativeai": "google-generativeai",
        "openai": "openai",
        "anthropic": "anthropic",
        "chromadb": "chromadb",
        "httpx": "httpx",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return aliases

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                canon = pkg_map.get(alias.name, alias.name.split(".")[0])
                aliases[alias.asname or alias.name] = canon
        elif isinstance(node, ast.ImportFrom) and node.module:
            canon = pkg_map.get(node.module, node.module.split(".")[0])
            for alias in node.names:
                aliases[alias.asname or alias.name] = canon
    return aliases


def _extract_method_calls(code: str) -> list[str]:
    """Devuelve lista de 'objeto.método' detectados en el AST."""
    calls: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return calls

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                # obj.method(...)
                if isinstance(func.value, ast.Name):
                    calls.append(f"{func.value.id}.{func.attr}")
                # obj.sub.method(...)
                elif isinstance(func.value, ast.Attribute):
                    calls.append(f"{func.value.attr}.{func.attr}")
    return calls


class DocHunter:
    """
    Analiza el código generado para detectar:
    - Errores de sintaxis Python
    - Métodos que no existen en la versión instalada (google-generativeai, openai, etc.)
    - Desajustes entre versión instalada y versión esperada por el modelo
    """

    def run(self, code: str) -> AuditReport:
        report = AuditReport()
        report.syntax_errors = self._check_syntax(code)
        if report.syntax_errors:
            # Con sintaxis rota no tiene sentido continuar el análisis
            return report

        imports = _extract_imports(code)
        calls = _extract_method_calls(code)

        report.version_alerts = self._check_versions(code, imports)
        report.missing_methods = self._check_methods(calls, imports)
        return report

    # ── privados ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_syntax(code: str) -> list[str]:
        errors: list[str] = []
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"línea {exc.lineno}: {exc.msg}")
        return errors

    @staticmethod
    def _check_versions(code: str, imports: dict[str, str]) -> list[str]:
        alerts: list[str] = []
        packages_used = set(imports.values())

        for pkg in packages_used:
            if pkg not in KNOWN_API_SIGNATURES:
                continue

            installed = _get_installed_version(pkg)
            if not installed:
                alerts.append(f"{pkg}: no instalado localmente")
                continue

            latest = _get_pypi_latest(pkg)
            if latest and latest != installed:
                alerts.append(
                    f"{pkg}: instalado={installed} | PyPI latest={latest} "
                    "(considera actualizar)"
                )

            # Chequeo específico google-generativeai — firma de métodos
            if pkg == "google-generativeai":
                alerts.extend(
                    DocHunter._audit_genai_signature(code, installed)
                )
        return alerts

    @staticmethod
    def _audit_genai_signature(code: str, installed_version: str) -> list[str]:
        """
        Si el código usa métodos de la API legacy (<0.4) pero tiene >=0.5 instalado
        (o viceversa), genera alertas concretas.
        """
        alerts: list[str] = []
        major_minor = ".".join(installed_version.split(".")[:2])

        new_api = KNOWN_API_SIGNATURES["google-generativeai"]["0.5"]
        old_api = KNOWN_API_SIGNATURES["google-generativeai"]["0.3"]

        uses_new = any(sym in code for sym in new_api)
        uses_old = any(sym in code for sym in old_api)

        try:
            v = float(major_minor)
        except ValueError:
            return alerts

        if v >= 0.5 and uses_old:
            alerts.append(
                "google-generativeai: el código usa API legacy (generate_text / "
                f"embed_text) pero instalada es v{installed_version}. "
                "Migra a GenerativeModel.generate_content()."
            )
        if v < 0.5 and uses_new:
            alerts.append(
                "google-generativeai: el código usa GenerativeModel (API >=0.5) "
                f"pero instalada es v{installed_version}. Actualiza con: "
                "pip install -U google-generativeai"
            )
        return alerts

    @staticmethod
    def _check_methods(calls: list[str], imports: dict[str, str]) -> list[str]:
        """
        Intenta importar los módulos presentes e introspeccionar métodos.
        Solo reporta ausencias confirmadas; no genera falsos positivos.
        """
        missing: list[str] = []
        checked: set[str] = set()

        for call in calls:
            parts = call.split(".")
            if len(parts) < 2:
                continue
            obj_name, method = parts[0], parts[-1]
            pkg = imports.get(obj_name)
            if not pkg or (pkg, method) in checked:
                continue
            checked.add((pkg, method))

            # Intento de inspección ligera (sólo si ya instalado)
            try:
                import importlib
                mod = importlib.import_module(pkg.replace("-", "_"))
                obj = getattr(mod, obj_name, None) or mod
                if obj and not hasattr(obj, method):
                    missing.append(
                        f"{obj_name}.{method}() no encontrado en {pkg} instalado"
                    )
            except (ImportError, ModuleNotFoundError):
                pass   # paquete no instalado → no emitimos falso positivo
            except Exception:
                pass
        return missing


# ──────────────────────────────────────────────────────────────────────────────
# 3. CODE AUDITOR LOCAL  (Ollama)
# ──────────────────────────────────────────────────────────────────────────────
class CodeAuditor:
    """
    Envía el código a un modelo local (Ollama) para una revisión semántica
    que va más allá del análisis estático: lógica, seguridad, coherencia.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        Eres un auditor de código Python experto. Tu tarea es REVISAR el código que te
        entreguen y responder ÚNICAMENTE en JSON con el siguiente esquema:
        {
          "approved": true | false,
          "issues": ["descripción del problema 1", ...],
          "suggestions": ["mejora concisa 1", ...]
        }
        Si no hay problemas, devuelve approved:true con listas vacías.
        NO incluyas explicaciones fuera del JSON.
    """)

    async def audit(self, code: str) -> dict:
        payload = {
            "model": CFG.ollama_model,
            "system": self.SYSTEM_PROMPT,
            "prompt": f"```python\n{code}\n```",
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=CFG.ollama_timeout) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate", json=payload
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "{}")
                # Extraer JSON aunque el modelo añada texto extra
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as exc:
            log.warning("CodeAuditor: Ollama no disponible (%s) — se omite", exc)
        return {"approved": True, "issues": [], "suggestions": []}


# ──────────────────────────────────────────────────────────────────────────────
# 4. GENERADORES PARALELOS  (GPT · Claude · Gemini)
# ──────────────────────────────────────────────────────────────────────────────
async def _call_openai(requirement: str) -> Generation:
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=CFG.cloud_timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {CFG.openai_api_key}"},
                json={
                    "model": CFG.openai_model,
                    "messages": [
                        {"role": "system", "content": "Eres un experto en Python. "
                         "Genera código limpio, tipado y documentado."},
                        {"role": "user", "content": requirement},
                    ],
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return Generation("gpt-4o", content, time.monotonic() - t0)
    except Exception as exc:
        return Generation("gpt-4o", "", time.monotonic() - t0, error=str(exc))


async def _call_claude(requirement: str) -> Generation:
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=CFG.cloud_timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CFG.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": CFG.claude_model,
                    "max_tokens": 4096,
                    "system": "Eres un experto en Python. Genera código limpio, "
                              "tipado y documentado.",
                    "messages": [{"role": "user", "content": requirement}],
                },
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            return Generation("claude", content, time.monotonic() - t0)
    except Exception as exc:
        return Generation("claude", "", time.monotonic() - t0, error=str(exc))


async def _call_gemini(requirement: str) -> Generation:
    t0 = time.monotonic()
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{CFG.gemini_model}:generateContent?key={CFG.gemini_api_key}"
        )
        async with httpx.AsyncClient(timeout=CFG.cloud_timeout) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": requirement}]}],
                "generationConfig": {"temperature": 0.3},
            })
            resp.raise_for_status()
            content = (
                resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            )
            return Generation("gemini", content, time.monotonic() - t0)
    except Exception as exc:
        return Generation("gemini", "", time.monotonic() - t0, error=str(exc))


async def parallel_generate(requirement: str) -> list[Generation]:
    """Llama a los modelos habilitados en paralelo; descarta errores silenciosamente."""
    log.info("Generación paralela iniciada…")
    tasks = []
    
    if CFG.use_openai: tasks.append(_call_openai(requirement))
    if CFG.use_claude: tasks.append(_call_claude(requirement))
    if CFG.use_gemini: tasks.append(_call_gemini(requirement))

    if not tasks:
        raise RuntimeError("Todos los generadores están desactivados en Config.")

    results = await asyncio.gather(*tasks)
    
    for g in results:
        if g.error:
            log.warning("  %s → ERROR: %s", g.model, g.error)
        else:
            log.info("  %s → OK  (%.1fs, %d chars)", g.model, g.latency, len(g.content))
    return [g for g in results if not g.error and g.content]


# ──────────────────────────────────────────────────────────────────────────────
# 5. SINTETIZADOR / SCORER  (+ Self-Healing)
# ──────────────────────────────────────────────────────────────────────────────
SYNTHESIZER_SYSTEM = textwrap.dedent("""\
    Eres un arquitecto senior de software. Recibirás varias implementaciones del mismo
    requerimiento. Tu tarea es seleccionar la MEJOR y, si es necesario, fusionar lo
    mejor de cada una. Prioriza: (1) corrección, (2) legibilidad, (3) eficiencia.
    Devuelve ÚNICAMENTE el bloque de código Python final, sin explicaciones previas.
""")

HEALER_SYSTEM = textwrap.dedent("""\
    Eres un ingeniero de software experto en corrección de errores. Se te entregará:
    1. Un requerimiento original.
    2. Un bloque de código con errores conocidos.
    3. Un reporte detallado de esos errores.
    Tu tarea: corregir ÚNICAMENTE los errores reportados sin cambiar la lógica restante.
    Devuelve ÚNICAMENTE el código Python corregido, sin explicaciones.
""")


async def _synthesize_via_gemini(system_prompt: str, user_message: str) -> str:
    """
    Usa Gemini como sintetizador/sanador centralizado.
    Reemplaza a Claude para evitar costos usando la capa gratuita de Google AI Studio.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{CFG.gemini_model}:generateContent?key={CFG.gemini_api_key}"
    )
    async with httpx.AsyncClient(timeout=CFG.cloud_timeout) as client:
        resp = await client.post(url, json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_message}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
        })
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def synthesize(requirement: str, generations: list[Generation]) -> str:
    """Produce la mejor solución a partir de las generaciones paralelas."""
    candidates = "\n\n".join(
        f"### Candidato {i+1} ({g.model})\n```python\n{g.content}\n```"
        for i, g in enumerate(generations)
    )
    user_msg = f"**Requerimiento:**\n{requirement}\n\n**Candidatos:**\n{candidates}"
    log.info("Sintetizando %d candidatos (via Gemini)…", len(generations))
    return await _synthesize_via_gemini(SYNTHESIZER_SYSTEM, user_msg)


async def heal(requirement: str, code: str, audit: AuditReport) -> str:
    """Una única ronda de corrección automática basada en el reporte de auditoría."""
    user_msg = textwrap.dedent(f"""\
        **Requerimiento original:**
        {requirement}

        **Código con errores:**
        ```python
        {code}
        ```

        **Reporte de errores:**
        {audit.summary()}
    """)
    log.info("Self-Healing: enviando código + reporte al sanador (via Gemini)…")
    return await _synthesize_via_gemini(HEALER_SYSTEM, user_msg)


def _extract_code_block(text: str) -> str:
    """Extrae el contenido de un bloque ```python ... ``` o devuelve el texto tal cual."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 6. ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
class NanoOrchestrator:
    def __init__(self) -> None:
        self.memory = MemoryLayer()
        self.doc_hunter = DocHunter()
        self.code_auditor = CodeAuditor()

    async def run(self, requirement: str) -> OrchestratorResult:
        log.info("=" * 60)
        log.info("REQUERIMIENTO: %s", requirement[:120])
        log.info("=" * 60)

        # ── Paso 1: Consultar memoria semántica ──────────────────────────────
        cache_hit = self.memory.query(requirement)
        if cache_hit:
            solution, similarity = cache_hit
            code = _extract_code_block(solution)
            audit = self.doc_hunter.run(code)
            log.info("CACHE HIT  (similitud=%.4f) — 0 tokens gastados ✓", similarity)
            return OrchestratorResult(
                solution=code,
                source="memory",
                audit=audit,
            )

        # ── Paso 2: Generación paralela en la nube ───────────────────────────
        generations = await parallel_generate(requirement)
        if not generations:
            raise RuntimeError("Todos los modelos fallaron. Revisa las API keys.")

        # ── Paso 3: Síntesis / Scoring ───────────────────────────────────────
        raw_solution = await synthesize(requirement, generations)
        code = _extract_code_block(raw_solution)

        # ── Paso 4: Doc Hunter + Auditoría local ─────────────────────────────
        log.info("Auditoría Doc Hunter…")
        audit = self.doc_hunter.run(code)
        log.info("Doc Hunter: %s", audit.summary())

        log.info("Auditoría Ollama (CodeAuditor)…")
        ollama_result = await self.code_auditor.audit(code)
        if not ollama_result.get("approved") and ollama_result.get("issues"):
            # Incorporamos los issues de Ollama al AuditReport
            audit.missing_methods.extend(
                [f"[Ollama] {issue}" for issue in ollama_result["issues"]]
            )

        # ── Paso 5: Self-Healing Loop (Bucle dinámico) ──────────────────────────
        source = "generated"
        heals_done = 0
        
        while audit.has_issues and heals_done < CFG.self_healing_max_rounds:
            log.warning("Ronda %d/%d de Self-Healing: corrigiendo errores...", 
                        heals_done + 1, CFG.self_healing_max_rounds)
            
            healed_raw = await heal(requirement, code, audit)
            code = _extract_code_block(healed_raw)

            # Re-auditar el código corregido
            audit = self.doc_hunter.run(code)
            
            # Auditoría secundaria opcional (Ollama) si la primaria (sintaxis) pasó
            if not audit.syntax_errors:
                ollama_healed = await self.code_auditor.audit(code)
                if not ollama_healed.get("approved") and ollama_healed.get("issues"):
                    audit.missing_methods.extend(
                        [f"[Ollama-post-heal] {i}" for i in ollama_healed["issues"]]
                    )

            source = "healed"
            heals_done += 1
            log.info("Post-healing %d audit: %s", heals_done, audit.summary())
            
        if audit.has_issues:
            log.error("Self-Healing falló tras %d rondas. Errores remanentes: %s", 
                      heals_done, audit.summary())
        elif source == "healed":
            log.info("Self-healing completado exitosamente en %d rondas.", heals_done)

        # ── Paso 6: Guardar en memoria (solo si no hay errores críticos) ─────
        mem_id = None
        if not audit.syntax_errors:           # no guardamos código roto
            mem_id = self.memory.store(
                requirement=requirement,
                solution=code,
                metadata={
                    "source": source,
                    "models_used": ",".join(g.model for g in generations),
                    "had_healing": str(source == "healed"),
                },
            )

        return OrchestratorResult(
            solution=code,
            source=source,
            audit=audit,
            generations=generations,
            memory_id=mem_id,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI  (entry-point mínimo)
# ──────────────────────────────────────────────────────────────────────────────
async def _main() -> None:
    # ─── Ejemplo de uso ───────────────────────────────────────────────────────
    requirement = (
        "Escribe una función Python async que reciba una lista de URLs "
        "y retorne un dict {url: status_code} usando httpx con concurrencia "
        "limitada (semáforo de 5). Incluye manejo de errores y type hints."
    )

    orchestrator = NanoOrchestrator()
    result = await orchestrator.run(requirement)

    print("\n" + "═" * 60)
    print(f"  RESULTADO FINAL  (fuente: {result.source.upper()})")
    print("═" * 60)
    print(result.solution)
    print("\n── Auditoría ────────────────────────────────────────────")
    print(result.audit.summary() or "Sin problemas detectados ✓")
    if result.memory_id:
        print(f"\n── Memoria: guardado con id={result.memory_id}")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(_main())
