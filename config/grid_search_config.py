import itertools
from tqdm import tqdm
BATCH_SIZE = [64, 128, 256, 512]
LR = [0.8e-3, 1.0e-3, 1.3e-3, 0.5e-4]
EPOCH = [300]
FUSION = ["MF", "LF"]
ATTENTION_HEADS = [1,2,3]
H_DIM = [128]
PE = [True, False]
MF_TYPE = ["default", "audio_refuse", "video_refuse", "self_attention", "multi_attention", "self_cross_attention"]

AVERAGE_OVER = 3 # trials per configurations
CONFIGURATION_LIST = [BATCH_SIZE, LR, EPOCH, FUSION, ATTENTION_HEADS, H_DIM, PE, MF_TYPE]

CONFIGURATIONS = list(itertools.product(*CONFIGURATION_LIST))
# for config in tqdm(CONFIGURATIONS):
#     batch_size, lr, epoch, fusion, attention_heads, h_dim, pe = config
#     print(batch_size, lr, epoch, fusion, attention_heads, h_dim, pe)

