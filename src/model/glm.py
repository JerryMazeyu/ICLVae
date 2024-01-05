import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModel, AutoTokenizer, AutoModelForCausalLM)
from copy import deepcopy



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



model_name = "THUDM/chatglm-6b"
use_auth_token = "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"
vae_conf = {"input_dim" : 4096, "hidden_dim" : 512, "latent_dim" : 4096, "n_layers" : 8, "heads" : 16}

glm_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_auth_token=use_auth_token).half().cuda()
vae = TransformerVAEV2(**vae_conf).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=use_auth_token)
ori_glm = deepcopy(glm_model)

ori_glm.transformer.word_embeddings = glm_model.transformer.word_embeddings
ori_glm.transformer.layers = glm_model.transformer.layers
ori_glm.lm_head = glm_model.lm_head

glm_model.transformer.final_layernorm = FinalLayerNormV2(glm_model.transformer.final_layernorm, vae)

for params in glm_model.parameters():
    params.requires_grad = False
for params in glm_model.transformer.final_layernorm.vae.parameters():
    params.requires_grad = True

glm = {"vae_model": glm_model, "ori_model": ori_glm, "tokenizer": tokenizer, "FEATURE_MAP": get_feature_map}
__all__ = [glm]