import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, pipeline
import torch.nn as nn
from copy import deepcopy
# from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from warnings import warn

class MLPProjection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MLPProjection, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.bs = batch_size

    def forward(self, x):
        # Reshape the input to [batch_size, sequence_length, input_size]
        x = x.view(x.size(0), -1)  # Reshape to [48, 128*4096]

        # Project the input through the MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Reshape the output to the desired shape [48, 128, 64]
        x = x.view(x.size(0), self.bs, -1)

        return x


class VAE(nn.Module):
    def __init__(self, batch_size):
        super(VAE, self).__init__()
        self.bs = batch_size

        self.mlp_enc = MLPProjection(self.bs*4096, 256, self.bs*64, self.bs)
        self.mlp_dec = MLPProjection(self.bs*64, 256, self.bs*4096, self.bs)

        self.encoder = nn.Sequential(
            nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=4096, nhead=1), num_layers=1),
            self.mlp_enc
        )

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

        self.decoder = nn.Sequential(
            self.mlp_dec,
            nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=4096, nhead=1), num_layers=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # x = x.view(x.shape[0], -1)  # Flatten the input for the encoder
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, heads):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=input_dim, nhead=heads, dim_feedforward=hidden_dim) for _ in range(n_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, heads):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=input_dim, nhead=heads, dim_feedforward=hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, src, memory):
        for layer in self.layers:
            src = layer(tgt=src, memory=memory)
        return self.output_layer(src)


class TransformerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads):
        super(TransformerVAE, self).__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, n_layers, heads)
        self.to_latent = nn.Linear(input_dim, 2 * latent_dim)  # Outputs mean and log variance
        self.decoder = TransformerDecoder(input_dim, hidden_dim, n_layers, heads)

    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        mean, log_var = torch.chunk(latent, 2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        decoded = self.decoder(z, z)
        return decoded

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var


class TransformerVAEV2(TransformerVAE):
    """
    Modified bugs.
    """
    def __init__(self, *args, **kwargs):
        super(TransformerVAEV2, self).__init__(*args, **kwargs)

    def sample(self, mean, logvar, num_samples=10):
        samples = [self.reparameterize(mean, logvar) for _ in range(num_samples)]
        return torch.stack(samples)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, x, z, mean, log_var


FEATURE_MAP= {}


class FinalLayerNormV2(torch.nn.Module):
    def __init__(self, final_layernorm, vae):
        super(FinalLayerNormV2, self).__init__()
        self.ori_layernorm = final_layernorm
        self.vae = vae

    def forward(self, x):
        x = self.ori_layernorm(x)
        x, x_ori, _, _, _ = self.vae(x)
        global FEATURE_MAP
        if 'rec' in FEATURE_MAP.keys():
            FEATURE_MAP = {}
        FEATURE_MAP['rec'] = x
        FEATURE_MAP['ori'] = x_ori
        return x

def get_feature_map():
    return FEATURE_MAP

SCORE_LIST = []
def get_score_hook(module, input, output):
    global SCORE_LIST
    SCORE_LIST.append(output)

def get_score_list():
    global SCORE_LIST
    if SCORE_LIST[0].shape[1] == 1:
        warn(f"SCORE_LIST[0] shape is {SCORE_LIST[0].shape}!", RuntimeWarning)
    SCORE_LIST[0] = SCORE_LIST[0].mean(dim=1, keepdim=True)
    res = torch.cat(SCORE_LIST[0:], dim=1)
    SCORE_LIST = []
    return res

vae_conf = {"input_dim" : 4096, "hidden_dim" : 512, "latent_dim" : 4096, "n_layers" : 8, "heads" : 16}
vae = TransformerVAEV2(**vae_conf).bfloat16().cuda()
# vae = TransformerVAEV2(**vae_conf).cuda()


model_name = "meta-llama/Llama-2-7b-chat-hf"
use_auth_token = "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"

tokenizer = LlamaTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

llama2_model = AutoModelForCausalLM.from_pretrained(model_name).bfloat16().cuda()
# llama2_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
ori_llama2 = deepcopy(llama2_model)

ori_llama2.model.embed_tokens = llama2_model.model.embed_tokens
ori_llama2.model.layers = llama2_model.model.layers
ori_llama2.lm_head = llama2_model.lm_head

ori_llama2.lm_head.register_forward_hook(get_score_hook)


llama2_model.model.norm = FinalLayerNormV2(llama2_model.model.norm, vae)

for params in llama2_model.parameters():
    params.requires_grad = False
for params in llama2_model.model.norm.vae.parameters():
    params.requires_grad = True


llama2 = {"vae_model": llama2_model, "ori_model": ori_llama2, "tokenizer": tokenizer, "FEATURE_MAP": get_feature_map, "args":['model', 'norm'], "use_score": False, "SCORE_LIST": get_score_list}
__all__ = [llama2]


if __name__ == "__main__":
    prompt =  "Hi, who are you?"
    model_input = tokenizer([prompt, "whats up?"], padding=True, return_tensors="pt").to("cuda:1")
    llama2_model.eval()
    tg = pipeline(task="text-generation", tokenizer=tokenizer, model=llama2_model, device="cuda:1")
    features = tg(["Hi, who are you?", "whats up?"], output_scores=True)
    # tmp = fe(**model_input)
    tmp = llama2_model.generate(**model_input, max_new_tokens=8, return_dict_in_generate=True, output_scores=True, temperature=0.9)
    response = tokenizer.decode(tmp.sequences, skip_special_tokens=True)
    print(response)




