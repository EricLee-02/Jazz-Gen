import torch
import torch.nn as nn


class JazzGenerationModel(nn.Module):

    def __init__(self,harmony_encoder,melody_decoder):

        super().__init__()

        self.harmony_encoder = harmony_encoder

        for p in self.harmony_encoder.parameters():
            p.requires_grad=False

        self.melody_decoder = melody_decoder



    def forward(self,harmony,melody_input):

        memory = self.harmony_encoder.encode(
            **harmony
        )

        logits = self.melody_decoder(
            melody_input,
            memory
        )

        return logits



    @torch.no_grad()
    def encode_harmony(self,harmony):

        self.eval()

        memory = self.harmony_encoder.encode(
            **harmony
        )

        return memory



    @torch.no_grad()
    def generate(
        self,
        harmony,
        bos_id,
        eos_id=None,
        max_length=512,
        temperature=0.8,
        top_k=20
    ):


        self.eval()


        # ==========================
        # Harmony Encoding
        # ==========================

        memory=self.harmony_encoder.encode(
            **harmony
        )


        # ==========================
        # start token
        # ==========================

        generated=torch.tensor(
            [[bos_id]],
            device=memory.device
        )


        # ==========================
        # autoregressive generation
        # ==========================

        for step in range(max_length-1):


            logits=self.melody_decoder(
                generated,
                memory
            )


            next_logits=logits[:,-1,:]


            # temperature

            next_logits = next_logits / temperature



            # top-k sampling

            if top_k:

                values,indices=torch.topk(
                    next_logits,
                    top_k
                )


                probs=torch.softmax(
                    values,
                    dim=-1
                )


                sample=torch.multinomial(
                    probs,
                    1
                )


                next_token=indices.gather(
                    1,
                    sample
                )


            else:

                probs=torch.softmax(
                    next_logits,
                    dim=-1
                )

                next_token=torch.multinomial(
                    probs,
                    1
                )


            generated=torch.cat(
                [
                    generated,
                    next_token
                ],
                dim=1
            )


            if eos_id is not None:

                if next_token.item()==eos_id:
                    break



        return generated