import argparse
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.graph2seq_series_rel import Graph2SeqSeriesRel
from models.seq2seq import Seq2Seq
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import load_vocab, S2SDataset, G2SDataset
from utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger

import math
import shutil
import warnings
warnings.filterwarnings('ignore')


def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)

    # initialization ----------------- vocab
    if not os.path.exists(args.vocab_file):
        raise ValueError(f"Vocab file {args.vocab_file} not found!")
    vocab = load_vocab(args.vocab_file)
    vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

    # initialization ----------------- model
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == "s2s":
        model_class = Seq2Seq
        dataset_class = S2SDataset
    elif args.model == "g2s_series_rel":
        model_class = Graph2SeqSeriesRel
        dataset_class = G2SDataset
        assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")

    model = model_class(args, vocab)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    if args.load_from:
        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")
        model.load_state_dict(pretrain_state_dict, strict=False)

    model.to(device)
    model.train()

    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")

    # initialization ----------------- optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    scheduler = NoamLR(
        optimizer,
        model_size=args.decoder_hidden_size,
        warmup_steps=args.warmup_steps
    )

    # initialization ----------------- data
    train_dataset = dataset_class(args, file=args.train_bin)
    valid_dataset = dataset_class(args, file=args.valid_bin)

    total_step = 0
    accum = 0
    losses, accs = [], []

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()

    logging.info("Start training")
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(args.epoch):
        model.zero_grad()

        train_dataset.sort()
        train_dataset.shuffle_in_bucket(bucket_size=1000)
        train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda _batch: _batch[0],
            num_workers=16,
            pin_memory=True
        )

        for batch_idx, batch in enumerate(train_loader):
            if total_step > args.max_steps:
                logging.info("Max steps reached, finish training")
                exit(0)

            batch.to(device)
            with torch.autograd.profiler.profile(enabled=args.do_profile,
                                                 record_shapes=args.record_shapes,
                                                 use_cuda=torch.cuda.is_available()) as prof:

                # Enables autocasting for the forward pass (model + loss)
                with torch.cuda.amp.autocast(enabled=args.enable_amp):
                    loss_all, acc = model(batch)
                    old_loss, aux_loss, kld_loss = loss_all
                    kla_coef = min(math.tanh(2. * total_step / args.max_steps - 3) + 1, 1) * 0.03
                    loss = old_loss + 0.1 * aux_loss + kla_coef * kld_loss

                # Exits the context manager before backward()
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                losses.append(loss.item())
                accs.append(acc.item() * 100)

                accum += 1

                if accum == args.accumulation_count:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                    scheduler.step()

                    g_norm = grad_norm(model)
                    model.zero_grad()
                    accum = 0
                    total_step += 1

            if args.do_profile:
                logging.info(prof
                             .key_averages(group_by_input_shape=args.record_shapes)
                             .table(sort_by="cuda_time_total"))
                sys.stdout.flush()

            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                logging.info(f"Step {total_step}, loss: {np.mean(losses)}, acc: {np.mean(accs)}, "
                             f"p_norm: {param_norm(model)}, g_norm: {g_norm}, "
                             f"lr: {get_lr(optimizer): .6f}, elapsed time: {time.time() - o_start: .0f}")
                sys.stdout.flush()
                losses, accs = [], []

            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                model.eval()
                eval_count = 100
                eval_meters = [0.0, 0.0]

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    num_workers=16,
                    pin_memory=True
                )

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break
                        eval_batch.to(device)

                        eval_loss_all, eval_acc = model(eval_batch)
                        old_loss, aux_loss, kld_loss = eval_loss_all
                        kla_coef = min(math.tanh(2. * total_step / args.max_steps - 3) + 1, 1) * 0.03
                        eval_loss = old_loss + 0.1 * aux_loss + kla_coef * kld_loss
                        eval_meters[0] += eval_loss.item() / eval_count
                        eval_meters[1] += eval_acc * 100 / eval_count

                logging.info(f"Evaluation (with teacher) at step {total_step}, eval loss: {eval_meters[0]}, "
                             f"eval acc: {eval_meters[1]}")
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sys.stdout.flush()

                model.train()

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1

                model.eval()
                eval_count = 100

                valid_dataset.sort()
                valid_dataset.shuffle_in_bucket(bucket_size=1000)
                valid_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.valid_batch_size
                )
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=lambda _batch: _batch[0],
                    num_workers=16,
                    pin_memory=True
                )

                accs_token = []
                accs_seq = []

                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(valid_loader):
                        if eval_idx >= eval_count:
                            break

                        eval_batch.to(device)
                        results = model.predict_step(
                            reaction_batch=eval_batch,
                            batch_size=eval_batch.size,
                            beam_size=args.beam_size,
                            n_best=1,
                            temperature=1.0,
                            min_length=args.predict_min_len,
                            max_length=args.predict_max_len
                        )[0]
                        predictions = [t[0].cpu().numpy() for t in results["predictions"]]

                        for i, prediction in enumerate(predictions):
                            acc_seq, acc_token = 0, 0
                            target_ids = None
                            for j in torch.arange(eval_batch.pres[i], eval_batch.posts[i]):
                                tgt_length = valid_dataset.tgt_lens[j]
                                tgt_token_ids = valid_dataset.tgt_token_ids[j][:tgt_length]
                                acc_seq = max(np.array_equal(tgt_token_ids, prediction[:tgt_length]), acc_seq)
                                while len(prediction) < tgt_length:
                                    prediction = np.append(prediction, vocab["_PAD"])
                                if np.mean(tgt_token_ids == prediction[:tgt_length])>=acc_token:
                                    acc_token = np.mean(tgt_token_ids == prediction[:tgt_length])
                                    target_ids = tgt_token_ids

                            accs_token.append(acc_token)
                            accs_seq.append(acc_seq)

                            if eval_idx % 20 == 0 and i == 0:
                                logging.info(f"Target text: {' '.join([vocab_tokens[idx] for idx in target_ids])}")
                                logging.info(f"Predicted text: {' '.join([vocab_tokens[idx] for idx in prediction])}")
                                logging.info(f"acc_token: {acc_token}, acc_seq: {acc_seq}\n")

                logging.info(f"Evaluation (without teacher) at step {total_step}, "
                             f"eval acc (token): {np.mean(accs_token)}, "
                             f"eval acc (sequence): {np.mean(accs_seq)}")
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sys.stdout.flush()

                model.train()

                logging.info(f"Saving at step {total_step}")
                sys.stdout.flush()

                state = {
                    "args": args,
                    "state_dict": model.state_dict()
                }
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                torch.save(state, os.path.join(args.save_dir, f"model.{total_step}_{n_iter}.pt"))
                if n_iter >= args.keep_last_ckpt-1:
                    old_iter = n_iter - args.keep_last_ckpt
                    old_path = os.path.join(args.save_dir, f"model_{(old_iter+1) * args.save_iter}_{old_iter}.pt")
                    shutil.rmtree(old_path)

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            model.zero_grad()
            accum = 0


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    torch.set_printoptions(profile="full")
    main(args)
