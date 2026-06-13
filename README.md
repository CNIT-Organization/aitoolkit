# aitoolkit

Centralized AI clients — **LLM, Embeddings, Speech-to-Text, Text-to-Speech, and
RAG** — targeting self-hosted, OpenAI-compatible services. Reusable across
projects via a single git install.

## Why

Replaces scattered, provider-specific AI code (cloud-LLM wrappers, local Whisper,
ad-hoc LangChain) with one thin, stable, dependency-light package. The core is
**100% LangChain-free**; LangChain is an opt-in extra used only where LangGraph
orchestration needs it.

## Design principles

1. **Depend on stable interfaces, hide volatile implementations.** The public API
   is a small set of clients + plain types; provider SDKs stay internal.
2. **Core is dependency-light and LangChain-free.** Heavy/optional things (qdrant,
   redis, langchain) live behind extras and import lazily.
3. **No project specifics in the package.** Hosts, model ids, voices, collection
   names, and keywords are parameters/config — never hardcoded.
4. **OpenAI-compatible first.** LLM, embeddings and STT use the `openai` SDK; only
   TTS is a small custom `httpx` client.
5. **Async-first**, with sync convenience wrappers where ergonomics demand it.

## Install

```bash
# core only (LLM, embeddings, STT, TTS)
pip install "aitoolkit @ git+https://github.com/CNIT-Organization/aitoolkit.git@v0.2.0"

# with RAG (Qdrant) + caching + LangChain bridge
pip install "aitoolkit[all] @ git+https://github.com/CNIT-Organization/aitoolkit.git@v0.2.0"

# pick exactly what a service needs
pip install "aitoolkit[rag,cache] @ git+...@v0.2.0"
pip install "aitoolkit[rag,langchain] @ git+...@v0.2.0"
```

Extras: `rag` (qdrant-client) · `cache` (redis) · `langchain` (langchain-core +
langchain-openai) · `all`.

## Configuration

All config is environment-driven (`AITOOLKIT_*`, see [.env.example](.env.example)),
but every client also accepts explicit overrides, so no env is strictly required.

## Quick start

```python
import asyncio
from aitoolkit import get_llm_client, get_embeddings_client, get_stt_client, get_tts_client

async def main():
    llm = get_llm_client()
    print(await llm.chat("Say hello in one short sentence."))

    async for tok in llm.stream("Count to five."):
        print(tok, end="", flush=True)

    emb = get_embeddings_client()
    vecs = await emb.aembed_documents(["first document", "second document"])
    print("dim:", emb.dimension)

    stt = get_stt_client()
    result = await stt.transcribe("audio.wav", language="en")
    print(result.text)

    tts = get_tts_client()
    audio = await tts.synthesize("Hello world", voice="your-voice-id")
    open("out.wav", "wb").write(audio)

    # multi-speaker: synthesize each turn with its own voice and stitch to one WAV
    dialogue = await tts.synthesize_dialogue([
        {"voice_id": "voice-a", "text": "Welcome to the overview."},
        {"voice_id": "voice-b", "text": "Let's dive in."},
    ])
    open("dialogue.wav", "wb").write(dialogue)

asyncio.run(main())
```

### Structured output

```python
from pydantic import BaseModel
class Flashcard(BaseModel):
    question: str
    answer: str

card = await get_llm_client().chat_structured(Flashcard, "Make a flashcard about the water cycle.")
```

### RAG (`aitoolkit[rag]`)

```python
from aitoolkit.rag import get_rag_agent
agent = get_rag_agent(collection_name="documents")
await agent.add_documents(["chunk 1", "chunk 2"], file_id="doc-42")
answer = await agent.answer_question("What does the document say about safety?")
```

### LangChain bridge (`aitoolkit[langchain]`)

```python
from aitoolkit.integrations.langchain import to_chat_model, LangChainEmbeddings
chat_model = to_chat_model(temperature=0.3)   # a LangChain BaseChatModel for LangGraph
embeddings = LangChainEmbeddings()            # a LangChain Embeddings
```

## Testing

```bash
pip install -e ".[all]" --group dev
pytest                       # unit tests (mocked)
AITOOLKIT_RUN_LIVE=1 pytest  # also run live smoke tests against your endpoints
```

Live tests auto-skip when the configured endpoints are unreachable.
