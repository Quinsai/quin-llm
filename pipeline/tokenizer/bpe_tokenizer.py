"""A small, dependency-free byte-level BPE tokenizer.

The tokenizer works on UTF-8 bytes, so any input string can be encoded without
an unknown-token fallback.  Token ids are laid out as:

    0..255      raw bytes
    256..259    special tokens (<PAD>, <BOS>, <EOS>, <UNK>)
    260..       merged BPE tokens

Training uses the classic word-frequency BPE algorithm: pre-tokenize text into
words, count adjacent pairs weighted by word frequency, and repeatedly merge the
most frequent pair.  A pair -> words index keeps each merge proportional to the
number of affected words instead of rescanning the whole corpus.
"""

from __future__ import annotations

import heapq
import pickle
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

SPECIAL_TOKENS: tuple[str, ...] = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")

# Split text into word-ish chunks.  Keeping the leading space in the chunk lets
# the model learn " word" as a token, which is useful for English.
SPLIT_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class BPETokenizer:
    version = 2
    CHUNK_CACHE_SIZE = 200_000

    def __init__(
        self,
        vocab_size: int = 8192,
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = list(special_tokens)
        self.special_ids = {tok: 256 + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_ids.items()}

        self.target_vocab_size = vocab_size
        self.merges: list[tuple[int, int, int]] = []
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.merge_result: dict[tuple[int, int], int] = {}
        self.token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size = 256 + len(self.special_tokens)
        # Encoding the same chunk repeatedly is the main cost of preparing a
        # large corpus.  Cache chunk -> token ids for the lifetime of the
        # process; the cache is not part of the serialized tokenizer.
        self._chunk_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    # ------------------------------------------------------------------
    # special token helpers
    # ------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.special_ids["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.special_ids["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_ids["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.special_ids["<UNK>"]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return SPLIT_PATTERN.findall(text)

    @staticmethod
    def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out: list[int] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 2,
        max_words: int = 200_000,
        max_chars: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Learn merge rules from an iterable of text documents."""
        target = vocab_size or self.target_vocab_size
        if target <= self.vocab_size:
            raise ValueError(f"vocab_size must be > {self.vocab_size}")
        # Merge rules change during training, so any cached encodings from a
        # previous run are no longer valid.
        self._chunk_cache.clear()

        chunk_freq: Counter[str] = Counter()
        n_chars = 0
        with tqdm(
            total=max_chars,
            desc="tokenize: read",
            unit="char",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not verbose,
        ) as read_bar:
            for text in texts:
                chunk_freq.update(self._pretokenize(text))
                n_chars += len(text)
                read_bar.update(len(text))
                if max_chars is not None and n_chars >= max_chars:
                    break

        # Convert word strings to byte-id tuples and aggregate duplicates.
        word_freqs: dict[tuple[int, ...], int] = {}
        for word, freq in chunk_freq.items():
            ids = tuple(word.encode("utf-8"))
            if ids:
                word_freqs[ids] = word_freqs.get(ids, 0) + freq
        if not word_freqs:
            raise ValueError("no training text found")
        if len(word_freqs) > max_words:
            word_freqs = dict(sorted(word_freqs.items(), key=lambda kv: -kv[1])[:max_words])

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        heap = [(-count, a, b) for (a, b), count in pair_counts.items()]
        heapq.heapify(heap)

        num_merges = target - self.vocab_size
        for merge_i in tqdm(
            range(num_merges),
            desc="tokenize: merges",
            unit="merge",
            dynamic_ncols=True,
            disable=not verbose,
        ):
            best: tuple[tuple[int, int], int] | None = None
            while heap:
                neg_count, a, b = heapq.heappop(heap)
                pair = (a, b)
                count = pair_counts.get(pair, 0)
                if count <= 0 or count != -neg_count:
                    continue  # stale heap entry
                if count < min_freq:
                    heapq.heappush(heap, (-count, a, b))
                    break
                best = (pair, count)
                break
            if best is None:
                break

            pair, _ = best
            new_id = self.vocab_size
            changed_pairs: set[tuple[int, int]] = set()

            for word in list(pair_to_words.get(pair, ())):
                freq = word_freqs.get(word)
                if not freq:
                    continue
                new_word = self._merge_word(word, pair, new_id)
                if new_word == word:
                    continue

                old_counts = Counter(zip(word, word[1:]))
                for p, c in old_counts.items():
                    pair_counts[p] -= c * freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    words = pair_to_words.get(p)
                    if words is not None:
                        words.discard(word)
                        if not words:
                            pair_to_words.pop(p, None)
                    changed_pairs.add(p)

                new_counts = Counter(zip(new_word, new_word[1:]))
                for p, c in new_counts.items():
                    pair_counts[p] += c * freq
                    pair_to_words.setdefault(p, set()).add(new_word)
                    changed_pairs.add(p)

                del word_freqs[word]
                word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

            self.merges.append((pair[0], pair[1], new_id))
            self.merge_ranks[pair] = len(self.merges) - 1
            self.merge_result[pair] = new_id
            self.token_bytes[new_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            self.vocab_size += 1

            for p in changed_pairs:
                count = pair_counts.get(p, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, p[0], p[1]))

        self.target_vocab_size = target
        if verbose:
            print(
                f"  tokenizer: learned {len(self.merges)} merges from "
                f"{len(word_freqs)} word types / {n_chars / 1e6:.1f}M chars, "
                f"vocab_size={self.vocab_size}"
            )

    # ------------------------------------------------------------------
    # encoding / decoding
    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        ids = list(chunk.encode("utf-8"))
        if not self.merge_ranks or len(ids) < 2:
            return tuple(ids)
        while len(ids) > 1:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids[best_i : best_i + 2] = [self.merge_result[pair]]
        return tuple(ids)

    def _encode_chunk_cached(self, chunk: str) -> tuple[int, ...]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk)
            return cached

        ids = self._encode_chunk(chunk)
        self._chunk_cache[chunk] = ids
        if len(self._chunk_cache) > self.CHUNK_CACHE_SIZE:
            self._chunk_cache.popitem(last=False)
        return ids

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in self._pretokenize(text):
            ids.extend(self._encode_chunk_cached(chunk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out = bytearray()
        for raw in ids:
            i = int(raw)
            if i in self.id_to_special:
                continue
            data = self.token_bytes.get(i)
            if data is not None:
                out.extend(data)
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "special_tokens": self.special_tokens,
            "special_ids": self.special_ids,
            "merges": self.merges,
            "token_bytes": self.token_bytes,
            "vocab_size": self.vocab_size,
            "target_vocab_size": self.target_vocab_size,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        obj = cls(vocab_size=data.get("target_vocab_size", data["vocab_size"]))
        obj.special_tokens = data["special_tokens"]
        obj.special_ids = data["special_ids"]
        obj.id_to_special = {i: tok for tok, i in obj.special_ids.items()}
        obj.merges = data["merges"]
        obj.merge_ranks = {(a, b): i for i, (a, b, _) in enumerate(obj.merges)}
        obj.merge_result = {(a, b): new_id for a, b, new_id in obj.merges}
        obj.token_bytes = data["token_bytes"]
        obj.vocab_size = data["vocab_size"]
        obj.target_vocab_size = data.get("target_vocab_size", obj.vocab_size)
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)


def iter_texts_from_sources(sources: list[dict], max_chars: int | None = None) -> Iterator[str]:
    """Yield text from a list of source configs, respecting a global char budget."""
    total = 0
    weight_sum = sum(float(s.get("weight", 1.0)) for s in sources) or 1.0
    for source in sources:
        weight = float(source.get("weight", 1.0))
        remaining = (
            None
            if max_chars is None
            else max(0, int(max_chars * weight / weight_sum))
        )
        used = 0
        for text in _iter_one_source(source):
            yield text
            used += len(text)
            total += len(text)
            if remaining is not None and used >= remaining:
                break
            if max_chars is not None and total >= max_chars:
                return


def _iter_one_source(source: dict) -> Iterator[str]:
    from pipeline.sources import iter_source_texts

    yield from iter_source_texts(source)
