import torch
from omegaconf import DictConfig, open_dict

cp_path = 'xlsr_53_56k.pt'
cp = torch.load(cp_path)
wrong_key = ['eval_wer','eval_wer_config', 'eval_wer_tokenizer', 'eval_wer_post_process', 'autoregressive']
cfg = DictConfig(cp['cfg'])
with open_dict(cfg):
    for k in wrong_key:
        cfg.task.pop(k)
cp['cfg'] = cfg
torch.save(cp, 'xlsr_53_56k_new.pt')