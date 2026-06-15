from __future__ import annotations

"""Local open-vocabulary semantic scorer backed by CLIP.

This replaces the heuristic placeholder with a learned model that runs fully
offline on CPU or GPU — no cloud API in the loop, which matters for edge and
disconnected missions. It implements the same SemanticVisionScorer protocol as
the local and OpenAI scorers, so it slots into vision_lab, mission evaluation,
and the benchmark suite unchanged.

Heavy dependencies (torch, open_clip) are imported lazily, mirroring the
optional-import pattern in px4_controller_interface. Install with:

    pip install '.[ml]'        # or: pip install torch open-clip-torch

Scoring approach: the mission objective is expanded into positive prompts
(target description, category phrasings) and a fixed set of background
prompts. CLIP ranks the crop against all prompts; the softmax probability mass
on positive prompts becomes the score, mapped onto the project's standard
decision bands. Every result keeps needs_human_review=True — the scorer
prioritizes, the analyst decides.
"""

import numpy as np

from autonomy.types import MissionObjective, SemanticDecision, SemanticVisionResult, TargetDetection


BACKGROUND_PROMPTS = [
    "an aerial photo of empty terrain",
    "an aerial photo of vegetation and trees",
    "an aerial photo of open water with waves",
    "an aerial photo of bare ground with shadows",
    "an aerial photo of rocks and natural debris",
    "an aerial photo of an empty road",
    "a blurry low-quality aerial photo of nothing in particular",
]

CATEGORY_PROMPTS = {
    "person": ["an aerial photo of a person on the ground", "a drone photo of a human figure"],
    "vehicle": ["an aerial photo of a car or truck", "a drone photo of a parked vehicle"],
    "boat": ["an aerial photo of a boat on the water", "a drone photo of a vessel near a shoreline"],
    "aircraft": ["an aerial photo of an aircraft on the ground"],
    "debris": ["an aerial photo of scattered wreckage or debris"],
    "signal": ["an aerial photo of smoke, fire, a flare, or a bright signal marker"],
}


class ClipSemanticVisionScorer:
    """CLIP-based scorer implementing the SemanticVisionScorer protocol.

    ``encode_image`` / ``encode_text`` may be injected for tests; when omitted,
    open_clip and torch are loaded lazily on first use.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        encode_image=None,
        encode_text=None,
    ) -> None:
        self.model_name = f"clip-local:{model_name}"
        self._clip_model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._encode_image = encode_image
        self._encode_text = encode_text
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}

    # -- protocol -----------------------------------------------------------

    def score(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
        crop_bgr: np.ndarray | None,
        detection: TargetDetection,
    ) -> SemanticVisionResult:
        if not detection.detected:
            return SemanticVisionResult(
                score=0.0,
                decision=SemanticDecision.REJECT,
                explanation="No visual candidate was proposed.",
                model_name=self.model_name,
            )
        image = crop_bgr if crop_bgr is not None and crop_bgr.size else frame_bgr
        return self._score_image(objective=objective, image_bgr=image, context="candidate crop")

    def score_full_frame(
        self,
        *,
        objective: MissionObjective,
        frame_bgr: np.ndarray,
    ) -> SemanticVisionResult:
        result = self._score_image(objective=objective, image_bgr=frame_bgr, context="full frame")
        tags = sorted(set(result.tags + ["full_frame_scan"]))
        return SemanticVisionResult(
            score=result.score,
            decision=result.decision,
            explanation=result.explanation,
            model_name=result.model_name,
            tags=tags,
            needs_human_review=True,
        )

    # -- internals ----------------------------------------------------------

    def _score_image(self, *, objective: MissionObjective, image_bgr: np.ndarray, context: str) -> SemanticVisionResult:
        positive_prompts = mission_prompts(objective)
        prompts = positive_prompts + BACKGROUND_PROMPTS
        image_embedding = self._image_embedding(image_bgr)
        text_embeddings = self._text_embeddings(tuple(prompts))
        similarities = text_embeddings @ image_embedding
        probabilities = _softmax(similarities * 100.0)
        positive_mass = float(np.sum(probabilities[: len(positive_prompts)]))
        top_index = int(np.argmax(probabilities))
        top_prompt = prompts[top_index]
        score = round(max(0.0, min(1.0, positive_mass)), 3)
        decision = _decision_for_score(score)
        tags = [f"clip_top:{top_prompt}"]
        if top_index < len(positive_prompts):
            tags.append("clip_positive_top")
        explanation = (
            f"CLIP compared the {context} against {len(positive_prompts)} mission prompts and "
            f"{len(BACKGROUND_PROMPTS)} background prompts. Positive probability mass: {score:.3f}. "
            f"Best match: '{top_prompt}'."
        )
        return SemanticVisionResult(
            score=score,
            decision=decision,
            explanation=explanation,
            model_name=self.model_name,
            tags=tags,
            needs_human_review=True,
        )

    def _image_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot score an empty image.")
        encode = self._encode_image or self._load_backend()[0]
        embedding = np.asarray(encode(image_bgr), dtype=np.float32).reshape(-1)
        return _normalize(embedding)

    def _text_embeddings(self, prompts: tuple[str, ...]) -> np.ndarray:
        cached = self._text_cache.get(prompts)
        if cached is not None:
            return cached
        encode = self._encode_text or self._load_backend()[1]
        embeddings = np.asarray(encode(list(prompts)), dtype=np.float32)
        embeddings = np.stack([_normalize(row) for row in embeddings])
        self._text_cache[prompts] = embeddings
        return embeddings

    def _load_backend(self):
        if self._encode_image is not None and self._encode_text is not None:
            return self._encode_image, self._encode_text
        try:
            import torch
            import open_clip
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "CLIP scorer requires torch, open-clip-torch, and pillow. "
                "Install with: pip install '.[ml]'"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name, pretrained=self._pretrained
        )
        tokenizer = open_clip.get_tokenizer(self._clip_model_name)
        model = model.to(device).eval()

        def encode_image(image_bgr: np.ndarray) -> np.ndarray:
            rgb = image_bgr[:, :, ::-1]
            pil_image = Image.fromarray(rgb.astype(np.uint8))
            tensor = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(tensor)
            return features.squeeze(0).float().cpu().numpy()

        def encode_text(prompts: list[str]) -> np.ndarray:
            tokens = tokenizer(prompts).to(device)
            with torch.no_grad():
                features = model.encode_text(tokens)
            return features.float().cpu().numpy()

        self._encode_image = encode_image
        self._encode_text = encode_text
        return encode_image, encode_text


def mission_prompts(objective: MissionObjective) -> list[str]:
    prompts: list[str] = []
    description = (objective.target_description or "").strip()
    if description:
        prompts.append(f"an aerial photo of {description}")
        prompts.append(f"a drone photo of {description}")
    for category in objective.extracted_categories:
        prompts.extend(CATEGORY_PROMPTS.get(category, []))
    if objective.extracted_colors and description:
        colors = " ".join(objective.extracted_colors)
        prompts.append(f"an aerial photo of a {colors} colored object that is {description}")
    if not prompts:
        prompts.append(f"an aerial photo of {objective.raw_request}")
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique = []
    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    return unique


def _decision_for_score(score: float) -> SemanticDecision:
    if score >= 0.75:
        return SemanticDecision.LIKELY_MATCH
    if score >= 0.55:
        return SemanticDecision.POSSIBLE_MATCH
    if score >= 0.2:
        return SemanticDecision.NEEDS_REVIEW
    return SemanticDecision.REJECT


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector
