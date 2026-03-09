# ReMemR1 Memory Module Bundle

This folder contains the core Python files needed to migrate the Memory module to another project.

## Included files

- `recurrent/impls/memory_revisit.py`
- `recurrent/impls/async_memory.py`
- `recurrent/impls/tf_idf_retriever.py`
- `recurrent/interface.py`
- `recurrent/utils.py`
- `recurrent/async_utils.py`

## Minimal runtime dependencies

- `torch`
- `numpy`
- `transformers`
- `omegaconf`
- `typing_extensions`
- `scikit-learn`
- `aiohttp`
- `httpx`
- `openai`
- `tensordict`

## How to use in training (synchronous mode)

1. Ensure your recurrent loader can import a register object from a Python file.
2. Point config to `recurrent/impls/memory_revisit.py`.
3. Load `REGISTER` from that file.

Example config key used in this repo:

`recurrent.memory.path="recurrent/impls/memory_revisit.py"`

## Async mode note

`recurrent/impls/async_memory.py` imports:

`from recurrent.impls.memory import MemoryConfig, TEMPLATE, TEMPLATE_FINAL_BOXED`

In this repo snapshot, `recurrent/impls/memory.py` does not exist. If you use async mode,
replace that import with:

`from recurrent.impls.memory_revisit import MemoryConfig, TEMPLATE, TEMPLATE_FINAL_BOXED`

or create your own `memory.py` alias.

