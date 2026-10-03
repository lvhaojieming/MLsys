"""Opt-in vLLM 0.8.4 scoring fallback for long prompt-logprob requests.

The quantized model forward remains on NPU. Only probability normalization and
sampler bookkeeping move to CPU, retaining the complete context and vocabulary.
Enable explicitly through a scoring server's sitecustomize with install().
"""
import copy
import os


def install():
    from vllm.model_executor.layers.sampler import Sampler
    if getattr(Sampler, '_moqe_cpu_scoring', False):
        return
    original = Sampler.forward
    threshold = int(os.environ.get('MOQE_SCORE_CPU_THRESHOLD', '2048'))

    def forward(self, logits, sampling_metadata):
        if logits is None or logits.shape[0] <= threshold:
            return original(self, logits, sampling_metadata)
        if self.include_gpu_probs_tensor or sampling_metadata.skip_sampler_cpu_output:
            raise ValueError('CPU scoring fallback requires synchronous, non-speculative sampling')
        if any(group.sampling_params.temperature != 0 for group in sampling_metadata.seq_groups):
            raise ValueError('CPU scoring fallback supports deterministic reference scoring only')
        metadata = copy.copy(sampling_metadata)
        metadata.reuse_sampling_tensors = False
        metadata.categorized_sample_indices = {
            key: value.cpu() for key, value in metadata.categorized_sample_indices.items()
        }
        if metadata.selected_token_indices is not None:
            metadata.selected_token_indices = metadata.selected_token_indices.cpu()
        # Copy before converting dtype so FP32 normalization never occupies HBM.
        cpu_logits = logits.cpu().float()
        return original(self, cpu_logits, metadata)

    Sampler.forward = forward
    Sampler._moqe_cpu_scoring = True
