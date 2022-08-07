import argparse
import logging
import numpy as np
import os
import sys
import torch
from models.graph2seq_series_rel import Graph2SeqSeriesRel
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, S2SDataset, G2SDataset, tokenize_smiles
from utils.train_utils import log_tensor, param_count, set_seed, setup_logger

import warnings
warnings.filterwarnings('ignore')


def get_predict_parser():
    parser = argparse.ArgumentParser("predict")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)

    if args.do_predict and os.path.exists(args.result_file):
        logging.info(f"Result file found at {args.result_file}, skipping prediction")
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    elif args.do_predict and not os.path.exists(args.result_file):

        # initialization ----------------- model
        assert os.path.exists(args.load_from), f"{args.load_from} does not exist!"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

        for attr in ["mpn_type", "rel_pos"]:
            try:
                getattr(pretrain_args, attr)
            except AttributeError:
                setattr(pretrain_args, attr, getattr(args, attr))

        assert args.model == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"
        if args.model == "s2s":
            model_class = Seq2Seq
            dataset_class = S2SDataset
        elif args.model == "g2s_series_rel":
            model_class = Graph2SeqSeriesRel
            dataset_class = G2SDataset
            args.compute_graph_distance = True
            assert args.compute_graph_distance
        else:
            raise ValueError(f"Model {args.model} not supported!")

        # initialization ----------------- vocab
        vocab = load_vocab(pretrain_args.vocab_file)
        vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict, strict=False)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        model.to(device)
        model.eval()

        logging.info(model)
        logging.info(f"Number of parameters = {param_count(model)}")
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        # initialization ----------------- data
        test_dataset = dataset_class(pretrain_args, file=args.test_bin)
        test_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.predict_batch_size
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            num_workers=16,
            pin_memory=True
        )

        all_predictions = []
        supp_info = []

        with torch.no_grad():
            with open(args.test_tgt.replace('tgt','src'), 'r') as f:
                lines = f.readlines()
            import json
            ans_path = os.path.join("/".join(args.test_tgt.split('/')[:-1]), "raw_test.json")
            with open(ans_path, 'r') as jfile:
                ans = json.load(jfile)
            for test_idx, test_batch in enumerate(test_loader):
                if test_idx % args.log_iter == 0:
                    logging.info(f"Doing inference on test step {test_idx}")
                    sys.stdout.flush()

                test_batch.to(device)
                if hasattr(model.decoder, 'cvae'):
                    results, sample_z = model.predict_step( ## sample_z.shape=torch.Size([51, 30])  [batch_size, beam_size]
                        reaction_batch=test_batch,
                        batch_size=test_batch.size,
                        beam_size=args.beam_size,
                        n_best=args.n_best,
                        temperature=args.temperature,
                        min_length=args.predict_min_len,
                        max_length=args.predict_max_len
                    )
                else:
                    results = model.predict_step(
                        reaction_batch=test_batch,
                        batch_size=test_batch.size,
                        beam_size=args.beam_size,
                        n_best=args.n_best,
                        temperature=args.temperature,
                        min_length=args.predict_min_len,
                        max_length=args.predict_max_len
                    )

                for i, predictions in enumerate(results["predictions"]):
                    idx = test_batch.data_indice[i]
                    if idx.cpu() not in test_dataset.ptr:
                        continue
                    product = lines[idx].strip()
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = " ".join(predicted_tokens)
                        if smi in ans[product]['reactant']:  #['C O C ( = O ) c 1 c c ( Cl ) c c c 1 Br']
                            if 'cover' in ans[product].keys():
                                ans[product]['cover'] += 1
                            else:
                                ans[product]['cover'] = 1
                            ans[product]['reactant'].remove(smi)
                        smis.append("".join(predicted_tokens))
                    
                    smis = ",".join(smis)
                    all_predictions.append(f"{smis}\n")
                    if hasattr(model.decoder, 'cvae'):
                        supp_info.append(f"1-to-N: {ans[product]['reaction']}; Product id {idx}: {''.join(product.split())}; Discrete Z: {sample_z[i].tolist()}\n")
                    else:
                        supp_info.append(f"1-to-N: {ans[product]['reaction']}; Product id {idx}: {''.join(product.split())}\n")

        save_dir = os.path.join(*args.result_file.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metric = {}
        for k in ans.keys():
            if ans[k]['reaction'] not in metric.keys():
                metric[ans[k]['reaction']] = {}
                metric[ans[k]['reaction']]['num'] = 0
                metric[ans[k]['reaction']]['covers'] = []
            metric[ans[k]['reaction']]['num'] += 1
            metric[ans[k]['reaction']]['covers'].append(1. - len(ans[k]['reactant']) * 1. / ans[k]['reaction'])
        for k in metric.keys():
            metric[k]['coverage'] = np.mean(metric[k]['covers'])
        import json
        json_str = json.dumps(metric)
        with open(args.result_file+'.json', 'w') as json_file:
            json_file.write(json_str)
        logging.info(json_str)
        with open(args.result_file, "w") as of:
            of.writelines(all_predictions)
        with open(args.result_file[:-4]+'_info.txt', "w") as of:
            of.writelines(supp_info)

    if args.do_score:
        invalid = 0

        total = len(test_dataset.ptr) - 1
        accuracies = np.zeros([total, args.n_best], dtype=np.float32)
        with open(args.test_tgt, "r") as f_tgt, open(args.result_file, "r") as f_predict:
            targets = f_tgt.readlines()
            for i, line_predict in enumerate(f_predict):
                line_predict = "".join(line_predict.split())
                smis_predict = line_predict.split(",")
                smis_predict = [canonicalize_smiles(smi, trim=False) for smi in smis_predict]
                if not smis_predict[0]:
                    invalid += 1
                smis_predict = [smi for smi in smis_predict if smi and not smi == "CC"]
                smis_predict = list(dict.fromkeys(smis_predict))  ## Deduplication

                tgt_pre = test_dataset.ptr[i].item()
                tgt_post = test_dataset.ptr[i+1].item()
                line_tgts = [s.strip() for s in targets[tgt_pre:tgt_post]]
                smi_tgts = []
                for line_tgt in line_tgts:
                    smi_tgt = "".join(line_tgt.split())
                    smi_tgt = canonicalize_smiles(smi_tgt, trim=False)
                    if not smi_tgt or smi_tgt == "CC":
                        continue
                    smi_tgts.append(smi_tgt)

                for j, smi in enumerate(smis_predict):
                    if smi in smi_tgts:
                        accuracies[i, j:] = 1.0
                        break


        logging.info(f"Total: {total}, "
                     f"top 1 invalid: {invalid / total * 100: .2f} %")

        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(args.n_best):
            logging.info(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")
        logging.info(f"Top 1,3,5,10 accuracy: {mean_accuracies[0] * 100: .2f},{mean_accuracies[2] * 100: .2f},"
        f"{mean_accuracies[4] * 100: .2f},{mean_accuracies[9] * 100: .2f}")


if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    # set random seed (just in case)
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args, warning_off=True)

    torch.set_printoptions(profile="full")
    main(args)
