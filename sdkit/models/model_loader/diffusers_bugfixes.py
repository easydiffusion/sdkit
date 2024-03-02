# waiting for https://github.com/damian0815/compel/pull/49
def apply_compel_pooled_patch():
    import torch

    from compel import Compel, EmbeddingsProvider
    from compel.embeddings_provider import EmbeddingsProviderMulti

    def build_conditioning_tensor(self, text: str) -> torch.Tensor:
        """
        Build a conditioning tensor by parsing the text for Compel syntax, constructing a Conjunction, and then
        building a conditioning tensor from that Conjunction.
        """
        conjunction = self.parse_prompt_string(text)
        conditioning, _ = self.build_conditioning_tensor_for_conjunction(conjunction)

        if self.requires_pooled:
            pooled = self.conditioning_provider.get_pooled_embeddings([text], device=self.device)
            return conditioning, pooled
        else:
            return conditioning

    def multi_get_pooled_embeddings(self, texts, attention_mask=None, device=None):
        pooled = [
            self.embedding_providers[provider_index].get_pooled_embeddings(texts, attention_mask, device=device)
            for provider_index, requires_pooled in enumerate(self.requires_pooled_mask)
            if requires_pooled
        ]

        if len(pooled) == 0:
            return None

        return torch.cat(pooled, dim=-1)

    def get_pooled_embeddings(self, texts, attention_mask=None, device=None):
        device = device or self.text_encoder.device

        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = text_encoder_output.text_embeds

        return pooled

    Compel.build_conditioning_tensor = build_conditioning_tensor
    EmbeddingsProvider.get_pooled_embeddings = get_pooled_embeddings
    EmbeddingsProviderMulti.get_pooled_embeddings = multi_get_pooled_embeddings


apply_compel_pooled_patch()
