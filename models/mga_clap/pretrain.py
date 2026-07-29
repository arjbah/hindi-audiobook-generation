#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

from torch.cuda.amp import GradScaler
import time
from pprint import PrettyPrinter
import torch
import argparse
import ruamel.yaml as yaml
from ruamel.yaml import YAML
from tqdm import tqdm
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.data import add_shared_args
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader, indicvoices_dataloader
from models.ase_model import ASE
import torch.distributed as dist
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def train(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    model.train()
    autocast = torch.cuda.amp.autocast
    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        with autocast():
            loss = model(audio, text, idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


def main(language):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/pretrain.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_ub8k_esc50_vggsound.json', type=str,
                        help="Blacklist file.")
    parser.add_argument('--local_rank', default=-1, type=int)
    # -s/--batch_size already exists. gender defaults to male so that omitting
    # the flag reproduces this file's previous hardcoded Male filter.
    add_shared_args(parser, include_batch_size=False, gender_default="male")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

    # setup distribution mode
    init_distributed_mode(args)
    device = config["device"]

    # setup seed
    seed = (args.seed if args.seed is not None else config["seed"]) + get_rank()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    config["data_args"]["batch_size"] = args.batch_size
    setup_seed(seed)

    # create pretrain dataloader
    dataloader = pretrain_dataloader(config,
                                     bucket=False,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=is_dist_avail_and_initialized(),
                                     num_tasks=get_world_size(),
                                     global_rank=get_rank(),
                                     split="train",
                                     language=language,
                                     gender=args.gender,
                                     limit=args.limit)
    
    
    """clotho_datamodule = AudioCaptionDataModule("Clotho")
    clotho_test_loader = clotho_datamodule.test_dataloader()
    ac_datamodule = AudioCaptionDataModule("AudioCaps")
    ac_test_loader = ac_datamodule.test_dataloader()"""
    test_loader = pretrain_dataloader(config,
                                     bucket=False,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=is_dist_avail_and_initialized(),
                                     num_tasks=get_world_size(),
                                     global_rank=get_rank(),
                                     split="test",
                                     language=language,
                                     gender=args.gender,
                                     limit=args.limit)
    test_loader_indicvoices = indicvoices_dataloader(config,
                                     bucket=False,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=is_dist_avail_and_initialized(),
                                     num_tasks=get_world_size(),
                                     global_rank=get_rank(),
                                     language=language,
                                     limit=args.limit)
    
    # setup model
    model = ASE(config)
    model.to(device)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(
        language.lower(), root=args.output_dir if args.output_dir else 'outputs')

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')
    
    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    desed_stats = []
    """clotho_stats = []
    ac_stats = []"""
    val_stats = []

    scaler = GradScaler()
    best_r1 = -1
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        
        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch, scaler)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()
        
        # validate on Clotho
        """if is_main_process():
            clotho_metrics = validate_re(model_without_ddp, clotho_test_loader, device)
            main_logger.info(f'Clotho statistics for epoch [{epoch}]:\t t2a-r1: {clotho_metrics["t2a"][0]:.3f}, t2a-r5: {clotho_metrics["t2a"][1]:.3f}, a2t-r1: {clotho_metrics["a2t"][0]:.3f}, a2t-r5: {clotho_metrics["a2t"][1]:.3f}.')
            clotho_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["t2a"][1] + clotho_metrics["a2t"][0] + clotho_metrics["a2t"][1])
            if clotho_stats[-1] >= max(clotho_stats):
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
                torch.save(sav_obj, str(model_output_dir) + "/clotho_best_model.pt")
                
        # validate on AudioCaps
        if is_main_process():
            ac_metrics = validate_re(model_without_ddp, ac_test_loader, device)
            main_logger.info(f'AudioCaps statistics for epoch [{epoch}]:\t t2a-r1: {ac_metrics["t2a"][0]:.3f}, t2a-r5: {ac_metrics["t2a"][1]:.3f}, a2t-r1: {ac_metrics["a2t"][0]:.3f}, a2t-r5: {ac_metrics["a2t"][1]:.3f}.')
            ac_stats.append(ac_metrics["t2a"][0] + ac_metrics["t2a"][1] + ac_metrics["a2t"][0] + ac_metrics["a2t"][1])
            if ac_stats[-1] >= max(ac_stats):
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
                torch.save(sav_obj, str(model_output_dir) + "/ac_best_model.pt")"""
        
        if is_main_process():
            val_metrics = validate_re(model_without_ddp, test_loader, device)
            main_logger.info(
                f"Epoch [{epoch}] | Rasa metrics | "
                f"T2A: R@1={val_metrics['t2a'][0]:.3f}, R@5={val_metrics['t2a'][1]:.3f}, R@10={val_metrics['t2a'][2]:.3f}, "
                f"R@50={val_metrics['t2a'][3]:.3f}, MedR={val_metrics['t2a'][4]:.3f}, MeanR={val_metrics['t2a'][5]:.3f}, mAP={val_metrics['t2a'][6]:.3f} | "
                f"A2T: R@1={val_metrics['a2t'][0]:.3f}, R@5={val_metrics['a2t'][1]:.3f}, R@10={val_metrics['a2t'][2]:.3f}, "
                f"R@50={val_metrics['a2t'][3]:.3f}, MedR={val_metrics['a2t'][4]:.3f}, MeanR={val_metrics['a2t'][5]:.3f}, mAP={val_metrics['a2t'][6]:.3f}"
            )
            indicvoices_val_metrics = validate_re(model_without_ddp, test_loader_indicvoices, device)
            main_logger.info(
                f"Epoch [{epoch}] | IndicVoices metrics | "
                f"T2A: R@1={indicvoices_val_metrics['t2a'][0]:.3f}, R@5={indicvoices_val_metrics['t2a'][1]:.3f}, R@10={indicvoices_val_metrics['t2a'][2]:.3f}, "
                f"R@50={indicvoices_val_metrics['t2a'][3]:.3f}, MedR={indicvoices_val_metrics['t2a'][4]:.3f}, MeanR={indicvoices_val_metrics['t2a'][5]:.3f}, mAP={indicvoices_val_metrics['t2a'][6]:.3f} | "
                f"A2T: R@1={indicvoices_val_metrics['a2t'][0]:.3f}, R@5={indicvoices_val_metrics['a2t'][1]:.3f}, R@10={indicvoices_val_metrics['a2t'][2]:.3f}, "
                f"R@50={indicvoices_val_metrics['a2t'][3]:.3f}, MedR={indicvoices_val_metrics['a2t'][4]:.3f}, MeanR={indicvoices_val_metrics['a2t'][5]:.3f}, mAP={indicvoices_val_metrics['a2t'][6]:.3f}"
            )
            val_stats.append(
                val_metrics["t2a"][0]
            )
            if val_stats[-1] > best_r1:
                best_r1 = val_stats[-1]
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "language": language,
                }
                torch.save(sav_obj, str(model_output_dir) + "/val_rasa_best_model.pt")

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()

    main_logger.info("Done.")


"""@torch.no_grad()
def validate_sed(model, dataloader, sed_classes, raw_sed_classes, sed_gt, sed_dur, device, name):
    model.eval()
    _, word_embeds, attn_mask = model.encode_text(sed_classes)
    _, text_embeds, _ = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    scores_raw_dic, scores_postprocessed_dic = {}, {}
    for _, (audio, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device, non_blocking=True)
        _, frame_embeds = model.encode_audio(audio)
        _, frame_embeds, _ = model.msc(frame_embeds.unsqueeze(1), model.codebook)
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        similarity = frame_embeds @ text_embeds.t()
        scores_raw, scores_postprocessed = post_process_sed(similarity, raw_sed_classes)
        scores_raw_dic[filename[0].split(".wav")[0]] = scores_raw
        scores_postprocessed_dic[filename[0].split(".wav")[0]] = scores_postprocessed
        
    pop_lst = []

    for k in scores_postprocessed_dic.keys():
        if k not in sed_gt.keys():
            pop_lst.append(k)
    for k in pop_lst:       
            scores_postprocessed_dic.pop(k)
     
    psds1_sed_scores_eval = compute_psds_from_scores(
        scores_postprocessed_dic,
        sed_gt,
        sed_dur,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=1,
    )

    psds2_sed_scores_eval = compute_psds_from_scores(
        scores_postprocessed_dic,
        sed_gt,
        sed_dur,
        dtc_threshold=0.1,
        gtc_threshold=0.1,
        cttc_threshold=0.3,
        alpha_ct=0.5,
        alpha_st=1,
    )

    eb_f1 = compute_collar_f1(scores_postprocessed_dic, sed_gt)
    seg_f1 = compute_seg_f1(scores_postprocessed_dic, sed_gt, sed_dur)

    return psds1_sed_scores_eval, psds2_sed_scores_eval, eb_f1["macro_average"], seg_f1["macro_average"]"""

@torch.no_grad()
def validate_re(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        _, frame_embeds = model.encode_audio(audio)
        audio_embeds = model.msc(frame_embeds, model.codebook)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        
        _, word_embeds, attn_mask = model.encode_text(text)
        text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP_a]}

if __name__ == '__main__':
    for language in [
        "Assamese",
        "Bengali",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Malayalam",
        "Marathi",
        "Tamil",
        "Telugu",
    ]:
        main(language)
